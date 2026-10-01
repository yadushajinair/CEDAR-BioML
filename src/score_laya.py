"""Primary LAYA ranker: pairwise binary equivalence scoring with Agent.predict_batch.

For every (source, candidate) pair one state is built and the SAME binary `choice` question is
asked. The score is LAYA's probability of the EQUIVALENT option:

    result["answers"]["equivalence"]["probabilities"]["EQUIVALENT"]          -> LayaScore

laya rounds that field to 4 decimals, which makes many of the 100 candidates tie exactly. The two
option logits behind it are therefore also captured at full precision and written as
LayaLogitMargin = z[EQUIVALENT] - z[NOT_EQUIVALENT]. P(EQUIVALENT) = sigmoid(margin / T), so the
margin orders candidates exactly as the un-rounded probability does; it only breaks rounding
ties. Both are stored and both rankings are produced.

Progress is appended query-block by query-block, so an interrupted job resumes where it stopped.

Usage:
  python src/score_laya.py --pair NCIT-DOID --split valid --rep D --method laya_zeroshot
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path

from build_pair_states import (EQUIVALENT, LAYA_REPO, LAYA_REVISION, NOT_EQUIVALENT, QID,
                               QUESTIONS, PairStateBuilder)
from utils import (ALL_SPLITS, CANDIDATE_COUNT, OUTPUTS, PAIRS, dump_json, fmt_metrics, rank_by_scores,
                   ranking_metrics, read_cands, score_matrix, write_ranking)

COLUMNS = ["QueryIndex", "SrcEntity", "TgtCandidate", "OriginalCandidatePosition", "LayaScore",
           "LayaLogitMargin", "LayaChoice", "StateTruncated"]


def load_agent(model: str, device: str | None = None):
    """Load a LAYA checkpoint: the pinned English Hub checkpoint, or a local fine-tuned dir."""
    import laya

    if os.path.isdir(model):
        return laya.load(model, device=device)
    return laya.load(model, device=device, revision=LAYA_REVISION)


def install_logit_capture(agent) -> None:
    """Attach each question's raw option logits (option order = criteria order) to its answer.

    laya exposes only 4-decimal probabilities; `_decode_answers` is where the raw logit rows are
    turned into them, so we wrap it and copy the same rows out untouched."""
    original = agent._decode_answers

    def decode(logits, act, items, ids, internal, offset, **kwargs):
        answers = original(logits, act, items, ids, internal, offset, **kwargs)
        for j, qid in enumerate(ids):
            if internal[qid].get("option_order") is not None:
                raise RuntimeError("logit capture assumes options are not permuted")
            row = logits[offset + j, : len(items[j]["markers"])]
            answers[qid]["raw_logits"] = [float(v) for v in row]
        return answers

    agent._decode_answers = decode


def choice_temperature(agent, n_options: int = 2) -> float:
    from laya.common import QTYPES, temp_bucket

    qt = QTYPES["choice"]
    return float(agent.temperature_by_options.get(temp_bucket(qt, n_options), agent.temperature[qt]))


class LayaPairScorer:
    def __init__(self, model: str, pair: str, rep: str, max_len: int | None = None,
                 batch_size: int = 16, device: str | None = None, precision: str = "amp"):
        self.agent = load_agent(model, device)
        install_logit_capture(self.agent)
        self.precision = precision
        if precision == "fp32":
            # laya's default on CUDA is bf16 autocast, which quantises the option logits and
            # makes them depend on batch composition; full precision removes that noise.
            import torch

            self.agent.amp_enabled = False
            self.agent.dtype = torch.float32
            self.agent.model.float()
        self.max_len = int(max_len or self.agent.cfg.get("max_len", 512))
        self.batch_size = batch_size
        self.builder = PairStateBuilder(pair, rep, self.agent.tok, self.max_len)
        self.temperature = choice_temperature(self.agent)

    def score_states(self, states: list[str]) -> list[dict]:
        results = self.agent.predict_batch(states, QUESTIONS, batch_size=self.batch_size,
                                           sort_by_length=True, max_len=self.max_len)
        out = []
        for res in results:
            ans = res["answers"][QID]
            raw = ans["raw_logits"]
            out.append({"LayaScore": ans["probabilities"][EQUIVALENT],
                        "LayaLogitMargin": raw[0] - raw[1],
                        "LayaChoice": ans["choice"],
                        "StateTruncated": int(bool(res["usage"]["truncated"])),
                        "raw": res})
        return out

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[dict]:
        return self.score_states([self.builder.state(s, t) for s, t in pairs])


def completed_queries(path: Path) -> int:
    """Number of fully written query blocks in a partial scores file (truncates a torn tail)."""
    if not path.is_file():
        return 0
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    if not rows or rows[0] != COLUMNS:
        return 0
    body = [r for r in rows[1:] if len(r) == len(COLUMNS)]
    done = len(body) // CANDIDATE_COUNT
    if len(body) != done * CANDIDATE_COUNT or len(body) != len(rows) - 1:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(COLUMNS)
            w.writerows(body[: done * CANDIDATE_COUNT])
    return done


def gpu_info() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"gpu": "none (CPU)"}
    return {"gpu": torch.cuda.get_device_name(0),
            "gpu_max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS))
    ap.add_argument("--split", required=True, choices=list(ALL_SPLITS))
    ap.add_argument("--rep", default="D", choices=["A", "B", "C", "D"])
    ap.add_argument("--model", default=LAYA_REPO, help="Hub id or fine-tuned checkpoint dir")
    ap.add_argument("--method", required=True, help="run name, e.g. laya_zeroshot")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="16 measured fastest on an L40S (outputs/bench_batch_size.json)")
    ap.add_argument("--precision", default="amp", choices=["amp", "fp32"],
                    help="amp = laya's default bf16 autocast; fp32 = full precision")
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--queries-per-chunk", type=int, default=32)
    ap.add_argument("--limit-queries", type=int, default=None, help="debug: first N queries only")
    ap.add_argument("--print-raw", type=int, default=0, help="print N raw LAYA results")
    args = ap.parse_args()

    import laya
    import torch

    run_name = f"{args.method}_{args.rep}"
    out_dir = Path(args.out_dir) if args.out_dir else OUTPUTS / args.pair / args.split / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "scores.tsv"

    queries = read_cands(args.pair, args.split)
    if args.limit_queries:
        queries = queries[: args.limit_queries]
    scorer = LayaPairScorer(args.model, args.pair, args.rep, args.max_len, args.batch_size,
                            precision=args.precision)
    print(f"[laya] laya=={laya.__version__} model={args.model} precision={args.precision} "
          f"revision={getattr(scorer.agent, 'revision', None)} device={scorer.agent.device} "
          f"max_len={scorer.max_len} state_room={scorer.builder.state_room} "
          f"per_concept={scorer.builder.per_concept} choice:2 temperature={scorer.temperature:.4f}")

    done = completed_queries(scores_path)
    if done:
        print(f"[laya] resuming: {done}/{len(queries)} queries already scored")
    prior = {}
    meta_path = out_dir / "run_meta.json"
    if done and meta_path.is_file():
        import json
        prior = json.loads(meta_path.read_text())
    runtime = float(prior.get("runtime_s", 0.0)) if done else 0.0
    scored_now = 0
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    mode = "a" if done else "w"
    with open(scores_path, mode, newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        if not done:
            w.writerow(COLUMNS)
        for start in range(done, len(queries), args.queries_per_chunk):
            chunk = queries[start:start + args.queries_per_chunk]
            pairs = [(q["src"], c) for q in chunk for c in q["cands"]]
            t0 = time.time()
            scored = scorer.score_pairs(pairs)
            dt = time.time() - t0
            runtime += dt
            scored_now += len(pairs)
            k = 0
            for qi, q in enumerate(chunk, start=start):
                for ci, cand in enumerate(q["cands"]):
                    s = scored[k]
                    if args.print_raw and k < args.print_raw and start == done:
                        print(f"--- raw LAYA result #{k}  gold={cand == q['gold']}\n"
                              f"{scorer.builder.state(q['src'], cand)}\n=> {s['raw']}")
                    w.writerow([qi, q["src"], cand, ci, "%.4f" % s["LayaScore"],
                                "%.6f" % s["LayaLogitMargin"], s["LayaChoice"], s["StateTruncated"]])
                    k += 1
            fh.flush()
            os.fsync(fh.fileno())
            dump_json(meta_path, {"runtime_s": runtime, "partial": True})
            n_done = min(len(queries), start + len(chunk))
            print(f"[laya] {n_done}/{len(queries)} queries  chunk {len(pairs) / dt:.0f} pairs/s",
                  flush=True)

    # --- rank: primary = full-precision margin; secondary = the API's 4-decimal probability.
    margin = score_matrix(scores_path, queries, "LayaLogitMargin")
    api = score_matrix(scores_path, queries, "LayaScore")
    trunc = score_matrix(scores_path, queries, "StateTruncated")
    n_pairs = sum(len(q["cands"]) for q in queries)
    # consistency: the API probability must be sigmoid(margin / T) rounded to 4 decimals
    worst = max(abs(1.0 / (1.0 + math.exp(-m / scorer.temperature)) - a)
                for mq, aq in zip(margin, api) for m, a in zip(mq, aq))
    tied = sum(1 for aq in api if sum(1 for a in aq if a == max(aq)) > 1)
    rank_margin = [rank_by_scores(q["cands"], m) for q, m in zip(queries, margin)]
    rank_api = [rank_by_scores(q["cands"], a) for q, a in zip(queries, api)]
    write_ranking(out_dir / "ranking.tsv", queries, rank_margin)
    api_dir = out_dir.parent / (out_dir.name + ".api4dp")
    write_ranking(api_dir / "ranking.tsv", queries, rank_api)

    meta = {
        "method": args.method, "checkpoint": args.model, "representation": args.rep,
        "laya_version": laya.__version__, "model_revision": getattr(scorer.agent, "revision", None),
        "max_len": scorer.max_len, "batch_size": args.batch_size, "precision": args.precision,
        "rank_key": "LayaLogitMargin (full-precision ordering of P(EQUIVALENT))",
        "runtime_s": runtime, "queries": len(queries), "pairs_scored": n_pairs,
        "pairs_per_sec": n_pairs / max(runtime, 1e-9),
        "states_truncated": int(sum(sum(t) for t in trunc)),
        "queries_with_tied_top_api_prob": tied,
        "max_abs_diff_api_prob_vs_sigmoid_margin": worst,
        "choice2_temperature": scorer.temperature,
        "cpu_fallback_count": int(getattr(scorer.agent, "cpu_fallback_count", 0)), **gpu_info(),
    }
    if meta["cpu_fallback_count"]:
        print(f"[laya] WARNING: {meta['cpu_fallback_count']} batches hit GPU OOM and were "
              "re-run on CPU; lower --batch-size")
    dump_json(meta_path, meta)
    dump_json(api_dir / "run_meta.json", {
        **meta, "method": args.method + ".api4dp",
        "rank_key": "LayaScore = probabilities['EQUIVALENT'] (4 dp), ties by pool position"})
    print(f"[laya] {args.pair} {args.split}: {len(queries)} queries, {n_pairs} pairs, "
          f"{runtime:.1f}s, {meta['pairs_per_sec']:.0f} pairs/s, truncated states="
          f"{meta['states_truncated']}, max|api - sigmoid(margin/T)|={worst:.2e}, "
          f"queries with a tied top API prob={tied}")
    print(f"[laya] {gpu_info()}")
    if queries[0]["gold"] is not None:
        print(f"   margin (primary) {fmt_metrics(ranking_metrics(queries, rank_margin))}")
        print(f"   API prob 4dp     {fmt_metrics(ranking_metrics(queries, rank_api))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

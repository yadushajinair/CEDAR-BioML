"""SECONDARY ablation: one 100-way LAYA `choice` question per query (NOT the main system).

The state is the source concept; each of the 100 candidates becomes one option
("T07: <label>; <synonyms>"). LAYA's default head budget (192 tokens on the English checkpoint)
would leave ~1 token per option, so -- as the model card instructs for high-cardinality option
sets -- `head_max_len` and `max_len` are raised per query to fit every option untruncated (LAYA
still caps each option at 48 tokens). Options differ per query, so this path cannot share one
question across states the way the pairwise ranker does: it is one forward pass per query.

  python src/score_laya_100way.py --pair NCIT-DOID --split valid --method laya100_zeroshot
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from build_entity_text import entity_text, n_tokens
from build_pair_states import LAYA_REPO
from ontology_loader import load_entities
from score_laya import gpu_info, install_logit_capture, load_agent
from utils import ALL_SPLITS, OUTPUTS, PAIRS, dump_json, fmt_metrics, rank_by_scores, ranking_metrics, read_cands, write_ranking

QID = "match"
INSTRUCTIONS = ("Select the target ontology class that denotes the same biomedical concept as the "
                "source ontology class. Choose the equivalent class, not one that is merely "
                "related, broader or narrower.")
STATE_BUDGET = 240      # tokens for the source concept description
OPTION_TEXT_CAP = 40    # LAYA truncates every option to 48 tokens including its key


def option_text(entity: dict, tok, with_synonyms: bool) -> str:
    text = entity["label"]
    if with_synonyms:
        for syn in entity["synonyms"] + entity["related_synonyms"]:
            if n_tokens(tok, text + "; " + syn) > OPTION_TEXT_CAP:
                break
            text += "; " + syn
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS))
    ap.add_argument("--split", required=True, choices=list(ALL_SPLITS))
    ap.add_argument("--model", default=LAYA_REPO)
    ap.add_argument("--method", required=True)
    ap.add_argument("--rep", default="D", help="representation of the SOURCE concept (state)")
    ap.add_argument("--option-synonyms", action="store_true", help="append synonyms to options")
    ap.add_argument("--limit-queries", type=int, default=None)
    ap.add_argument("--print-raw", type=int, default=1)
    args = ap.parse_args()

    import laya

    info = PAIRS[args.pair]
    src_entities, tgt_entities = load_entities(info["src"]), load_entities(info["tgt"])
    queries = read_cands(args.pair, args.split)
    if args.limit_queries:
        queries = queries[: args.limit_queries]
    agent = load_agent(args.model)
    install_logit_capture(agent)
    tok = agent.tok
    out_dir = OUTPUTS / args.pair / args.split / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    rankings, head_lens, max_lens, collapsed = [], [], [], 0
    runtime = 0.0
    with open(out_dir / "scores.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["QueryIndex", "SrcEntity", "TgtCandidate", "OriginalCandidatePosition",
                    "Laya100Prob", "Laya100Logit"])
        for qi, q in enumerate(queries):
            keys = ["T%02d" % j for j in range(len(q["cands"]))]
            criteria = {k: option_text(tgt_entities[c], tok, args.option_synonyms)
                        for k, c in zip(keys, q["cands"])}
            questions = {QID: {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}}
            # head budget that fits every option whole: [MASK] + "key: text" per option (<= 49)
            opt_tokens = sum(min(48, n_tokens(tok, " %s: %s" % (k, v))) + 1 for k, v in criteria.items())
            head_max_len = opt_tokens + n_tokens(tok, "choice question: " + INSTRUCTIONS) + 32
            max_len = head_max_len + STATE_BUDGET + 16
            state = entity_text(src_entities[q["src"]], info["src"], "SOURCE", args.rep, tok, STATE_BUDGET)
            t0 = time.time()
            res = agent.system_one(state, questions, max_len=max_len, head_max_len=head_max_len)
            runtime += time.time() - t0
            ans = res["answers"][QID]
            logits = ans["raw_logits"]
            if len(logits) != len(keys):
                raise RuntimeError(f"query {qi}: {len(logits)} option logits for {len(keys)} options")
            collapsed += int("options" in res["usage"])
            head_lens.append(head_max_len)
            max_lens.append(res["usage"]["input_tokens"])
            if qi < args.print_raw:
                shown = {k: ans["probabilities"][k] for k in keys[:5]}
                print(f"--- raw 100-way result, query {qi}\nSTATE:\n{state}\nfirst options: "
                      f"{ {k: criteria[k] for k in keys[:5]} }\nchoice={ans['choice']} "
                      f"confidence={ans['confidence']} first probabilities={shown}\n"
                      f"usage={res['usage']}")
            for ci, (cand, k, z) in enumerate(zip(q["cands"], keys, logits)):
                w.writerow([qi, q["src"], cand, ci, "%.4f" % ans["probabilities"][k], "%.6f" % z])
            rankings.append(rank_by_scores(q["cands"], logits))
            if (qi + 1) % 100 == 0:
                print(f"[100way] {qi + 1}/{len(queries)} queries  {(qi + 1) / runtime:.1f} queries/s",
                      flush=True)

    write_ranking(out_dir / "ranking.tsv", queries, rankings)
    n_pairs = sum(len(q["cands"]) for q in queries)
    meta = {"method": args.method, "checkpoint": args.model,
            "representation": f"state={args.rep}; options=label" + ("+synonyms" if args.option_synonyms else ""),
            "laya_version": laya.__version__, "runtime_s": runtime, "queries": len(queries),
            "pairs_scored": n_pairs, "pairs_per_sec": n_pairs / max(runtime, 1e-9),
            "head_max_len_mean": sum(head_lens) / len(head_lens), "head_max_len_max": max(head_lens),
            "input_tokens_mean": sum(max_lens) / len(max_lens), "input_tokens_max": max(max_lens),
            "queries_with_collapsed_options": collapsed,
            "cpu_fallback_count": int(getattr(agent, "cpu_fallback_count", 0)), **gpu_info()}
    dump_json(out_dir / "run_meta.json", meta)
    print(f"[100way] {args.pair} {args.split}: {len(queries)} queries in {runtime:.1f}s; input tokens "
          f"mean {meta['input_tokens_mean']:.0f} max {meta['input_tokens_max']}; collapsed-option "
          f"queries {collapsed}; {gpu_info()}")
    if queries[0]["gold"] is not None:
        print(f"   100-way  {fmt_metrics(ranking_metrics(queries, rankings))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

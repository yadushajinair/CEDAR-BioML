"""Build LAYA fine-tuning data (official JSONL schema) from the BioML TRAIN split only.

Each line is one case in the schema `research/scripts/finetune_single_device.py` expects:

    {"state": "<pair state>", "questions": {"equivalence": {...the ranking question...}},
     "gold": {"equivalence": {"probabilities": {"EQUIVALENT": 1.0, "NOT_EQUIVALENT": 0.0},
                              "label": "EQUIVALENT"}}}

Positives: (source, gold target) of a TRAIN query.
Negatives: `--neg-ratio` non-gold candidates from THAT query's own pool (the ranking setting):
           the lexically hardest half, plus a random half for coverage.

A fixed slice of TRAIN queries (grouped by source entity, so a source never straddles the
split) is reserved as "train-dev" for checkpoint selection. VALID and TEST are never read here.

  python src/build_finetune_data.py --pair NCIT-DOID --rep D --neg-ratio 5
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path

from build_pair_states import EQUIVALENT, NOT_EQUIVALENT, QID, QUESTIONS, PairStateBuilder, load_tokenizer
from lexical_baseline import LexicalScorer
from utils import OUTPUTS, PAIRS, dump_json, read_cands


def dev_query_indices(queries: list[dict], dev_frac: float, seed: int) -> list[int]:
    """Reserve ~dev_frac of TRAIN queries, split by source entity. Depends only on the pool
    file, the fraction and the seed, so every representation / negative ratio shares it."""
    sources = sorted({q["src"] for q in queries})
    random.Random(seed).shuffle(sources)
    dev_sources = set(sources[: max(1, round(len(sources) * dev_frac))])
    return [i for i, q in enumerate(queries) if q["src"] in dev_sources]


def data_dir(pair: str, rep: str, neg_ratio: int) -> Path:
    return OUTPUTS / "finetune_data" / pair / f"rep-{rep}_neg-{neg_ratio}"


def gold_block(positive: bool) -> dict:
    label = EQUIVALENT if positive else NOT_EQUIVALENT
    return {QID: {"probabilities": {EQUIVALENT: 1.0 if positive else 0.0,
                                    NOT_EQUIVALENT: 0.0 if positive else 1.0}, "label": label}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS))
    ap.add_argument("--rep", default="D", choices=["A", "B", "C", "D"])
    ap.add_argument("--neg-ratio", type=int, default=5, help="negatives per positive")
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    queries = read_cands(args.pair, "train")
    dev_idx = set(dev_query_indices(queries, args.dev_frac, args.seed))
    golds_of = collections.defaultdict(set)  # a source may have several TRAIN golds
    for q in queries:
        golds_of[q["src"]].add(q["gold"])

    builder = PairStateBuilder(args.pair, args.rep, load_tokenizer(), args.max_len)
    lex = LexicalScorer(args.pair)
    n_hard = math.ceil(args.neg_ratio / 2)
    out = data_dir(args.pair, args.rep, args.neg_ratio)
    out.mkdir(parents=True, exist_ok=True)
    n_pos = n_neg = 0
    max_state_tokens = 0
    with open(out / "train.jsonl", "w", encoding="utf-8") as fh:
        for qi, q in enumerate(queries):
            if qi in dev_idx:
                continue
            feats = lex.score_query(q)
            # never use another known gold of the same source as a "negative"
            pool = [(f["LexScore"], ci) for ci, (f, c) in enumerate(zip(feats, q["cands"]))
                    if c not in golds_of[q["src"]]]
            pool.sort(key=lambda x: (-x[0], x[1]))
            hard = [ci for _s, ci in pool[:n_hard]]
            rest = [ci for _s, ci in pool[n_hard:]]
            rnd = random.Random(args.seed * 1_000_003 + qi).sample(
                rest, min(len(rest), args.neg_ratio - len(hard)))
            rows = [(q["gold"], True)] + [(q["cands"][ci], False) for ci in hard + rnd]
            for tgt, positive in rows:
                state = builder.state(q["src"], tgt)
                max_state_tokens = max(max_state_tokens, builder.state_tokens(q["src"], tgt))
                fh.write(json.dumps({"state": state, "questions": QUESTIONS,
                                     "gold": gold_block(positive)}, ensure_ascii=False) + "\n")
                n_pos += positive
                n_neg += not positive
    meta = {"pair": args.pair, "rep": args.rep, "neg_ratio": args.neg_ratio, "seed": args.seed,
            "dev_frac": args.dev_frac, "max_len": args.max_len,
            "train_queries_total": len(queries), "train_fit_queries": len(queries) - len(dev_idx),
            "train_dev_queries": len(dev_idx), "train_dev_query_indices": sorted(dev_idx),
            "positives": n_pos, "negatives": n_neg, "hard_negatives_per_query": n_hard,
            "max_state_tokens": max_state_tokens, "state_room": builder.state_room,
            "source": "local.train.cands.tsv only (no valid/test labels)"}
    dump_json(out / "meta.json", meta)
    print(f"[ft-data] {args.pair} rep={args.rep} neg={args.neg_ratio}: {n_pos} positives + "
          f"{n_neg} negatives from {meta['train_fit_queries']} queries "
          f"({meta['train_dev_queries']} queries reserved as train-dev); "
          f"max state tokens {max_state_tokens}/{builder.state_room} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

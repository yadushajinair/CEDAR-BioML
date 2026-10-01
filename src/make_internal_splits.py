"""Materialise the internal splits of the official TRAIN file as pool files of their own.

  train-dev         the TRAIN queries reserved by build_finetune_data.dev_query_indices (split by
                    source entity). Never used for gradient updates; used ONLY to pick the epoch,
                    the negative ratio and (optionally) the ensemble weight.
  train-fit-sample  a fixed random sample of the remaining TRAIN queries, i.e. queries the model
                    WAS fine-tuned on. Scoring it shows how far "seen" performance is from held-out.

Rows are copied verbatim from local.train.cands.tsv, so the official validate_ranking.py and
score_local.py work on them unchanged. The official VALID and TEST files are not read.

  python src/make_internal_splits.py --pair NCIT-DOID
"""
from __future__ import annotations

import argparse
import random

from build_finetune_data import dev_query_indices
from utils import PAIRS, cands_path, read_cands


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS))
    ap.add_argument("--dev-frac", type=float, default=0.1)   # must match build_finetune_data
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit-sample", type=int, default=400)
    args = ap.parse_args()

    queries = read_cands(args.pair, "train")
    dev = dev_query_indices(queries, args.dev_frac, args.seed)
    dev_set = set(dev)
    fit = [i for i in range(len(queries)) if i not in dev_set]
    sample = sorted(random.Random(args.seed).sample(fit, min(args.fit_sample, len(fit))))
    with open(cands_path(args.pair, "train"), encoding="utf-8") as fh:
        header, *lines = fh.read().splitlines()
    assert len(lines) == len(queries)
    for split, idx in (("train-dev", dev), ("train-fit-sample", sample)):
        path = cands_path(args.pair, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join([header] + [lines[i] for i in idx]) + "\n", encoding="utf-8")
        srcs = {queries[i]["src"] for i in idx}
        print(f"[splits] {args.pair} {split}: {len(idx)} queries ({len(srcs)} sources) -> {path}")
    overlap = {queries[i]["src"] for i in dev} & {queries[i]["src"] for i in fit}
    print(f"[splits] sources shared between train-dev and train-fit: {len(overlap)}")
    return 0 if not overlap else 1


if __name__ == "__main__":
    raise SystemExit(main())

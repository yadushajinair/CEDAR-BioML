"""Inspect the released BioML 2026 local-ranking files without assuming anything about them.

Prints, per pair and split: header, number of queries, candidates per query, gold encoding,
repeated sources, duplicate candidates, gold membership/position in the pool, and how much the
splits overlap. Read-only.
"""
from __future__ import annotations

import collections
import csv
import sys

from utils import CANDIDATE_COUNT, PAIRS, SPLITS, cands_path, read_cands


def main() -> int:
    ok = True
    for pair in PAIRS:
        print(f"\n================ {pair} ================")
        srcs, tgts = {}, {}
        for split in SPLITS:
            path = cands_path(pair, split)
            with open(path, newline="", encoding="utf-8") as fh:
                header = next(csv.reader(fh, delimiter="\t"))
            qs = read_cands(pair, split)
            sizes = collections.Counter(len(q["cands"]) for q in qs)
            dup_cands = sum(1 for q in qs if len(set(q["cands"])) != len(q["cands"]))
            src_counts = collections.Counter(q["src"] for q in qs)
            repeated = sum(1 for c in src_counts.values() if c > 1)
            print(f"[{split}] {path.name}")
            print(f"  header            : {header}")
            print(f"  rows (= queries)  : {len(qs)}   (one row per query; candidates are a "
                  f"Python-list literal in TgtCandidates)")
            print(f"  candidates/query  : {dict(sizes)}")
            print(f"  queries w/ duplicate candidates: {dup_cands}")
            print(f"  distinct sources  : {len(src_counts)}   sources appearing >1x: {repeated}")
            if sizes != {CANDIDATE_COUNT: len(qs)} or dup_cands:
                ok = False
                print("  !! not exactly 100 distinct candidates for every query")
            if qs and qs[0]["gold"] is not None:
                in_pool = sum(1 for q in qs if q["gold"] in q["cands"])
                pos = [q["cands"].index(q["gold"]) for q in qs if q["gold"] in q["cands"]]
                deciles = collections.Counter(p // 10 for p in pos)
                print(f"  gold column       : TgtEntity (one full IRI per query); in pool for "
                      f"{in_pool}/{len(qs)} queries")
                print(f"  gold pool position: mean={sum(pos) / max(1, len(pos)):.1f} "
                      f"first-slot={sum(1 for p in pos if p == 0)} "
                      f"per-decile={[deciles.get(d, 0) for d in range(10)]}")
                if in_pool != len(qs):
                    ok = False
            else:
                print("  gold column       : absent (test split is gold-stripped)")
            srcs[split] = set(src_counts)
            tgts[split] = {c for q in qs for c in q["cands"]}
            print(f"  distinct candidate targets: {len(tgts[split])}")
            print(f"  example src       : {qs[0]['src']}")
            print(f"  example candidate : {qs[0]['cands'][0]}")
        print("  source overlap    : train&valid=%d  train&test=%d  valid&test=%d" % (
            len(srcs["train"] & srcs["valid"]), len(srcs["train"] & srcs["test"]),
            len(srcs["valid"] & srcs["test"])))
    print("\nALL STRUCTURAL CHECKS PASSED" if ok else "\nSTRUCTURAL PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

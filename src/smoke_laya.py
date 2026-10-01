"""Tiny LAYA smoke test: print the RAW API output for a handful of labelled pairs.

Takes the first N queries of a labelled split and, for each, scores the gold target (positive)
and its lexically closest non-gold candidate (hard negative). Small enough for a CPU.

  python src/smoke_laya.py --pair NCIT-DOID --split train --n-queries 5 --device cpu
"""
from __future__ import annotations

import argparse
import json
import math

from build_pair_states import EQUIVALENT, QID, QUESTIONS
from lexical_baseline import LexicalScorer
from score_laya import LayaPairScorer
from utils import PAIRS, read_cands


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="NCIT-DOID", choices=sorted(PAIRS))
    ap.add_argument("--split", default="train", choices=["train", "valid"])
    ap.add_argument("--rep", default="D")
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--n-queries", type=int, default=5)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import laya

    queries = read_cands(args.pair, args.split)[: args.n_queries]
    lex = LexicalScorer(args.pair)
    scorer = LayaPairScorer(args.model, args.pair, args.rep, batch_size=4, device=args.device)
    agent = scorer.agent
    print(f"laya=={laya.__version__}  model={args.model}  revision={agent.revision}  "
          f"device={agent.device}  cfg max_len={agent.cfg.get('max_len')} "
          f"head_max_len={agent.cfg.get('head_max_len')}  choice:2 T={scorer.temperature:.4f}")
    print("QUESTIONS =", json.dumps(QUESTIONS, indent=1))
    print(f"state room = {scorer.builder.state_room} tokens "
          f"({scorer.builder.per_concept} per concept)\n")

    pairs, labels = [], []
    for q in queries:
        feats = lex.score_query(q)
        negs = [(f["LexScore"], c) for f, c in zip(feats, q["cands"]) if c != q["gold"]]
        pairs += [(q["src"], q["gold"]), (q["src"], max(negs)[1])]
        labels += ["POSITIVE (gold)", "NEGATIVE (hardest lexical)"]
    scored = scorer.score_pairs(pairs)
    for i, ((src, tgt), label, s) in enumerate(zip(pairs, labels, scored)):
        ans = s["raw"]["answers"][QID]
        p_full = 1.0 / (1.0 + math.exp(-s["LayaLogitMargin"] / scorer.temperature))
        print(f"================ example {i}: {label} ================")
        print(scorer.builder.state(src, tgt))
        print(f"state tokens = {scorer.builder.state_tokens(src, tgt)}")
        print("RAW result =", json.dumps(s["raw"], indent=1))
        print(f"-> choice={ans['choice']}  probabilities['{EQUIVALENT}']={ans['probabilities'][EQUIVALENT]}"
              f"  confidence={ans['confidence']}  answer_confidence={ans['answer_confidence']}")
        print(f"-> full-precision check: margin={s['LayaLogitMargin']:.4f}  "
              f"sigmoid(margin/T)={p_full:.6f}\n")
    pos = [s["LayaScore"] for s, lab in zip(scored, labels) if lab.startswith("POS")]
    neg = [s["LayaScore"] for s, lab in zip(scored, labels) if lab.startswith("NEG")]
    print(f"SUMMARY  P(EQUIVALENT) positives={pos}\n         P(EQUIVALENT) negatives={neg}")
    print(f"         positive ranked above its hard negative in "
          f"{sum(p > n for p, n in zip(pos, neg))}/{len(pos)} queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

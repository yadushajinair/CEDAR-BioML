"""Lexical + LAYA ensemble:  combined = alpha * L + (1 - alpha) * X

  L = LAYA equivalence score, X = lexical score, both min-max normalised PER QUERY so the two
      live on the same [0, 1] scale whatever the checkpoint's calibration. L is taken from the
      full-precision logit margin (monotone in P(EQUIVALENT)).
  A second variant ("prob") uses L = P(EQUIVALENT) = sigmoid(margin / T) and X un-normalised.

alpha is tuned on a labelled split with the fixed grid 0.0, 0.1, ..., 1.0 (alpha=0 is lexical
only, alpha=1 is LAYA only) and the complete grid is saved. On TEST the (mode, alpha) chosen on
VALID must be passed in -- nothing is ever tuned on TEST.

  python src/ensemble.py --pair NCIT-DOID --split valid --laya-run laya_zeroshot_D
  python src/ensemble.py --pair NCIT-DOID --split test  --laya-run laya_bioml_neg5_D \
         --choice outputs/NCIT-DOID/valid/ens_lexical+laya_bioml_neg5_D/ensemble_choice.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from utils import ALL_SPLITS, OUTPUTS, PAIRS, dump_json, fmt_metrics, rank_by_scores, ranking_metrics, read_cands, score_matrix, write_ranking

ALPHAS = [round(0.1 * i, 1) for i in range(11)]
MODES = ("minmax", "prob")


def minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    return [0.0] * len(values) if hi <= lo else [(v - lo) / (hi - lo) for v in values]


def combine(margins, lex, alpha: float, mode: str, temperature: float) -> list[list[float]]:
    out = []
    for m, x in zip(margins, lex):
        if mode == "minmax":
            laya_s, lex_s = minmax(m), minmax(x)
        else:
            laya_s = [1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, v / temperature)))) for v in m]
            lex_s = x
        out.append([alpha * a + (1.0 - alpha) * b for a, b in zip(laya_s, lex_s)])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS))
    ap.add_argument("--split", required=True, choices=list(ALL_SPLITS))
    ap.add_argument("--laya-run", required=True, help="run directory name, e.g. laya_zeroshot_D")
    ap.add_argument("--lex-run", default="lexical")
    ap.add_argument("--choice", default=None,
                    help="apply an ensemble_choice.json tuned elsewhere instead of tuning here")
    ap.add_argument("--suffix", default="", help="appended to the output run name, e.g. @devalpha")
    ap.add_argument("--out-name", default=None,
                    help="output run directory name (default: ens_<lex-run>+<laya-run><suffix>)")
    args = ap.parse_args()

    base = OUTPUTS / args.pair / args.split
    queries = read_cands(args.pair, args.split)
    margins = score_matrix(base / args.laya_run / "scores.tsv", queries, "LayaLogitMargin")
    lex = score_matrix(base / args.lex_run / "scores.tsv", queries, "LexScore")
    laya_meta = json.loads((base / args.laya_run / "run_meta.json").read_text())
    temperature = float(laya_meta.get("choice2_temperature", 1.0))
    out_dir = base / (args.out_name or f"ens_{args.lex_run}+{args.laya_run}{args.suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.choice:
        choice = json.loads(Path(args.choice).read_text())
        mode, alpha = choice["mode"], float(choice["alpha"])
        print(f"[ens] using (mode={mode}, alpha={alpha}) tuned on {choice['tuned_on']}")
    else:
        if queries[0]["gold"] is None:
            raise SystemExit("refusing to tune alpha on a split without gold; pass --choice")
        grid = []
        for mode in MODES:
            for alpha in ALPHAS:
                scores = combine(margins, lex, alpha, mode, temperature)
                m = ranking_metrics(queries, [rank_by_scores(q["cands"], s)
                                              for q, s in zip(queries, scores)])
                grid.append({"mode": mode, "alpha": alpha, **m})
        with open(out_dir / "alpha_grid.tsv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(["mode", "alpha", "MRR", "Hits@1", "Hits@5", "Hits@10"])
            for g in grid:
                w.writerow([g["mode"], g["alpha"]] + ["%.4f" % g[k] for k in
                                                       ("mrr", "hits_at_1", "hits_at_5", "hits_at_10")])
                print(f"[ens] mode={g['mode']:6s} alpha={g['alpha']:.1f}  {fmt_metrics(g)}")
        # best MRR; ties -> minmax first, then the smaller alpha (grid order)
        best = max(grid, key=lambda g: g["mrr"])
        mode, alpha = best["mode"], best["alpha"]
        choice = {"mode": mode, "alpha": alpha, "tuned_on": f"{args.pair}/{args.split}",
                  "laya_run": args.laya_run, "lex_run": args.lex_run, "valid_mrr": best["mrr"]}
        dump_json(out_dir / "ensemble_choice.json", choice)
        print(f"[ens] selected mode={mode} alpha={alpha} (MRR {best['mrr']:.4f}) on {args.split}")

    scores = combine(margins, lex, alpha, mode, temperature)
    rankings = [rank_by_scores(q["cands"], s) for q, s in zip(queries, scores)]
    write_ranking(out_dir / "ranking.tsv", queries, rankings)
    dump_json(out_dir / "run_meta.json", {
        "method": f"{args.lex_run}+{laya_meta.get('method', args.laya_run)}{args.suffix}",
        "checkpoint": laya_meta.get("checkpoint", ""),
        "representation": laya_meta.get("representation", ""),
        "ensemble": {"mode": mode, "alpha": alpha, "tuned_on": choice["tuned_on"]},
        "runtime_s": laya_meta.get("runtime_s", 0.0), "queries": len(queries),
        "pairs_scored": laya_meta.get("pairs_scored", ""),
        "pairs_per_sec": laya_meta.get("pairs_per_sec", 0.0), "gpu": laya_meta.get("gpu", "")})
    print(f"[ens] wrote {out_dir / 'ranking.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

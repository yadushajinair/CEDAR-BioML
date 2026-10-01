"""Validate and score a ranking with the OFFICIAL scoring kit, and collect the result table.

  python src/evaluate.py --pair NCIT-DOID --split valid --run-dir outputs/NCIT-DOID/valid/lexical
  python src/evaluate.py --collect          # rebuild outputs/results_summary.tsv

Every headline number in this project comes from external/OAEI-Bio-ML/scoring_kit
(validate_ranking.py + score_local.py) run as a subprocess -- never from our own metric code.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from utils import ALL_SPLITS, OUTPUTS, PAIRS, SCORING_KIT, cands_path, dump_json

SPLIT_ROLE = {
    "train": "official TRAIN (all queries)",
    "train-fit-sample": "train (queries seen in fine-tuning)",
    "train-dev": "internal train-dev (from TRAIN; model selection only)",
    "valid": "official BioML valid (held out)",
    "test": "official BioML test (no gold available)",
}
SUMMARY_COLUMNS = ["pair", "method", "split", "split_role", "MRR", "Hits@1", "Hits@5", "Hits@10", "runtime",
                   "checkpoint", "representation", "ensemble", "queries", "pairs_scored", "pairs_per_sec",
                   "gpu", "gpu_max_mem_gb", "validated", "run_dir"]


def evaluate(pair: str, split: str, run_dir: Path, ranking_name: str = "ranking.tsv") -> dict:
    ranking = run_dir / ranking_name
    pool = cands_path(pair, split)
    val = subprocess.run([sys.executable, str(SCORING_KIT / "validate_ranking.py"), str(pool),
                          str(ranking)], capture_output=True, text=True)
    print(f"[validate_ranking] {ranking}: {val.stdout.strip() or val.stderr.strip()}")
    result = {"pair": pair, "split": split, "validated": val.returncode == 0,
              "validate_output": val.stdout.strip()}
    if val.returncode != 0:
        dump_json(run_dir / "eval.json", result)
        raise SystemExit(f"INVALID ranking: {ranking}")
    if split != "test":
        metrics_path = run_dir / "metrics.official.json"
        sc = subprocess.run([sys.executable, str(SCORING_KIT / "score_local.py"), str(ranking),
                             str(pool), "--output", str(metrics_path)],
                            capture_output=True, text=True)
        if sc.returncode != 0:
            raise SystemExit(f"score_local.py failed:\n{sc.stderr}")
        result["metrics"] = json.loads(metrics_path.read_text())
        print(f"[score_local] {json.dumps(result['metrics'], sort_keys=True)}")
    dump_json(run_dir / "eval.json", result)
    return result


def collect() -> Path:
    rows = []
    for eval_path in sorted(OUTPUTS.glob("*/*/*/eval.json")):
        run_dir = eval_path.parent
        ev = json.loads(eval_path.read_text())
        meta_path = run_dir / "run_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        m = ev.get("metrics", {})
        fmt = lambda k: ("%.4f" % m[k]) if k in m else ""  # noqa: E731
        rows.append({
            "pair": ev["pair"], "method": meta.get("method", run_dir.name), "split": ev["split"],
            "split_role": SPLIT_ROLE.get(ev["split"], ""),
            "ensemble": ("%(mode)s alpha=%(alpha)s tuned on %(tuned_on)s" % meta["ensemble"])
            if "ensemble" in meta else "",
            "MRR": fmt("mrr"), "Hits@1": fmt("hits_at_1"), "Hits@5": fmt("hits_at_5"),
            "Hits@10": fmt("hits_at_10"),
            "runtime": ("%.1fs" % meta["runtime_s"]) if "runtime_s" in meta else "",
            "checkpoint": meta.get("checkpoint", ""),
            "representation": meta.get("representation", ""),
            "queries": int(m["queries"]) if "queries" in m else meta.get("queries", ""),
            "pairs_scored": meta.get("pairs_scored", ""),
            "pairs_per_sec": ("%.0f" % meta["pairs_per_sec"]) if "pairs_per_sec" in meta else "",
            "gpu": meta.get("gpu", ""),
            "gpu_max_mem_gb": ("%.2f" % meta["gpu_max_mem_gb"]) if "gpu_max_mem_gb" in meta else "",
            "validated": ev.get("validated", False),
            "run_dir": str(run_dir.relative_to(OUTPUTS.parent)),
        })
    rows.sort(key=lambda r: (r["pair"], r["split"], r["method"]))
    out = OUTPUTS / "results_summary.tsv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[collect] {len(rows)} runs -> {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=sorted(PAIRS))
    ap.add_argument("--split", choices=list(ALL_SPLITS))
    ap.add_argument("--run-dir")
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    if args.run_dir:
        if not (args.pair and args.split):
            ap.error("--run-dir needs --pair and --split")
        evaluate(args.pair, args.split, Path(args.run_dir))
    if args.collect or not args.run_dir:
        collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

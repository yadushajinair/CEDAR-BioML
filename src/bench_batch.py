"""Find a safe, fast GPU batch size for pairwise LAYA scoring, empirically.

Scores the same fixed set of pairs at increasing batch sizes and reports throughput and peak
GPU memory. Stops at the first size that triggers LAYA's GPU-OOM -> CPU fallback.

  python src/bench_batch.py --pair NCIT-DOID --split valid --n-queries 20
"""
from __future__ import annotations

import argparse
import time

import torch

from score_laya import LayaPairScorer
from utils import OUTPUTS, PAIRS, dump_json, read_cands


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="NCIT-DOID", choices=sorted(PAIRS))
    ap.add_argument("--split", default="valid")
    ap.add_argument("--rep", default="D")
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--n-queries", type=int, default=20)
    ap.add_argument("--batch-sizes", default="8,16,32,64,128,256,512")
    args = ap.parse_args()

    queries = read_cands(args.pair, args.split)[: args.n_queries]
    scorer = LayaPairScorer(args.model, args.pair, args.rep)
    pairs = [(q["src"], c) for q in queries for c in q["cands"]]
    states = [scorer.builder.state(s, t) for s, t in pairs]
    lengths = sorted(scorer.builder.state_tokens(s, t) for s, t in pairs[:: max(1, len(pairs) // 500)])
    print(f"[bench] {len(states)} pairs on {scorer.agent.device}; state tokens "
          f"median={lengths[len(lengths) // 2]} p95={lengths[int(len(lengths) * 0.95)]} max={lengths[-1]}")
    scorer.batch_size = 8
    scorer.score_states(states[:64])  # warm-up (CUDA init, cudnn autotune)

    rows, reference = [], None
    for bs in [int(b) for b in args.batch_sizes.split(",")]:
        scorer.batch_size = bs
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        t0 = time.time()
        scored = scorer.score_states(states)
        dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        margins = [s["LayaLogitMargin"] for s in scored]
        if reference is None:
            reference = margins
        drift = max(abs(a - b) for a, b in zip(margins, reference))
        fallbacks = int(getattr(scorer.agent, "cpu_fallback_count", 0))
        rows.append({"batch_size": bs, "pairs_per_sec": len(states) / dt, "gpu_max_mem_gb": mem,
                     "max_margin_drift_vs_first": drift, "cpu_fallbacks": fallbacks})
        print(f"[bench] batch_size={bs:4d}  {len(states) / dt:8.0f} pairs/s  peak GPU mem="
              f"{mem:6.2f} GB  margin drift vs bs={rows[0]['batch_size']}: {drift:.4f}  "
              f"cpu_fallbacks={fallbacks}", flush=True)
        if fallbacks:
            print("[bench] OOM fallback hit - stopping")
            break
    ok = [r for r in rows if not r["cpu_fallbacks"]]
    best = max(ok, key=lambda r: r["pairs_per_sec"])
    print(f"[bench] fastest safe batch size: {best['batch_size']} "
          f"({best['pairs_per_sec']:.0f} pairs/s, {best['gpu_max_mem_gb']:.2f} GB)")
    dump_json(OUTPUTS / "bench_batch_size.json", {"rows": rows, "best": best,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/bin/bash
#SBATCH -J laya-smoke
#SBATCH -p gpu          # site-specific: set to your GPU partition
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
# Small GPU smoke test: raw LAYA output on 10 labelled pairs, then an empirical batch-size
# sweep on 20 validation queries (2,000 pairs).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"   # submit from the repository root
print_env

echo "===== raw LAYA output on labelled TRAIN pairs ====="
"$PY" src/smoke_laya.py --pair NCIT-DOID --split train --n-queries 5

echo "===== batch-size sweep ====="
"$PY" src/bench_batch.py --pair NCIT-DOID --split valid --n-queries 20

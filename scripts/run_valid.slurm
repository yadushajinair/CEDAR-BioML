#!/bin/bash
#SBATCH -J laya-valid
#SBATCH -p gpu          # site-specific: set to your GPU partition
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
# Pairwise LAYA ranking of a labelled split + official validation and scoring.
#
#   sbatch scripts/run_valid.slurm                                   # NCIT-DOID, zero-shot, rep D
#   sbatch --export=ALL,REPS="A B C D" scripts/run_valid.slurm       # representation ablation
#   sbatch --export=ALL,MODEL=checkpoints/<run>/best,METHOD=laya_bioml scripts/run_valid.slurm
#
# Variables: PAIR (NCIT-DOID) SPLIT (valid) REPS ("D") MODEL (convaiinnovations/laya)
#            METHOD (laya_zeroshot) BATCH (16) PRECISION (amp|fp32)
# Resumable: re-submitting continues from the last fully written query block.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"   # submit from the repository root
print_env

PAIR=${PAIR:-NCIT-DOID}
SPLIT=${SPLIT:-valid}
REPS=${REPS:-D}
MODEL=${MODEL:-convaiinnovations/laya}
METHOD=${METHOD:-laya_zeroshot}
BATCH=${BATCH:-16}          # fastest safe size measured on an L40S (outputs/bench_batch_size.json)
PRECISION=${PRECISION:-amp} # amp = laya default (bf16 autocast); fp32 = full precision

if [ "${SKIP_LEXICAL:-0}" != "1" ]; then   # set SKIP_LEXICAL=1 when several jobs run at once
  echo "===== lexical reference ====="
  "$PY" src/lexical_baseline.py --pair "$PAIR" --split "$SPLIT"
  "$PY" src/evaluate.py --pair "$PAIR" --split "$SPLIT" --run-dir "outputs/$PAIR/$SPLIT/lexical"
fi

for REP in $REPS; do
  echo "===== $METHOD  representation $REP ====="
  "$PY" src/score_laya.py --pair "$PAIR" --split "$SPLIT" --rep "$REP" --model "$MODEL" \
      --method "$METHOD" --batch-size "$BATCH" --precision "$PRECISION" --print-raw 6
  for RUN in "${METHOD}_${REP}" "${METHOD}_${REP}.api4dp"; do
    "$PY" src/evaluate.py --pair "$PAIR" --split "$SPLIT" --run-dir "outputs/$PAIR/$SPLIT/$RUN"
  done
done
echo "DONE. Rebuild the table with: $PY src/evaluate.py --collect"

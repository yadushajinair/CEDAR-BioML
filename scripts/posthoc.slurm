#!/bin/bash
#SBATCH -J laya-posthoc
#SBATCH -p gpu          # site-specific: set to your GPU partition
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
# After model selection: score the SELECTED LAYA-BioML checkpoint on the two internal splits of
# TRAIN so that three numbers can be reported side by side:
#     train-fit-sample (seen in fine-tuning)  |  train-dev (internal, selection)  |  official valid
# plus one full-precision (fp32) pass over official valid as a numerical-noise check.
#
#   sbatch --export=ALL,NEG=5 scripts/posthoc.slurm
#   sbatch --export=ALL,PAIR=SNOMED-FMA,TRAIN_SEED=1,SKIP_PREP=1,FP32_VALID=0 scripts/posthoc.slurm
#
# TRAIN_SEED (0) picks checkpoints/<run>_seed<N>. SKIP_PREP=1 skips re-writing the internal splits
# and their lexical scores (build them once beforehand when several jobs of one pair run at once).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"   # submit from the repository root
print_env

PAIR=${PAIR:-NCIT-DOID}
REP=${REP:-D}
NEG=${NEG:-5}
TRAIN_SEED=${TRAIN_SEED:-0}
SUFFIX=""; if [ "$TRAIN_SEED" != "0" ]; then SUFFIX="_seed${TRAIN_SEED}"; fi
CKPT="checkpoints/$(echo "$PAIR" | tr '[:upper:]' '[:lower:]')_rep${REP}_neg${NEG}${SUFFIX}/best"
METHOD="laya_bioml_neg${NEG}${SUFFIX}"

if [ "${SKIP_PREP:-0}" != "1" ]; then "$PY" src/make_internal_splits.py --pair "$PAIR"; fi
for SPLIT in train-dev train-fit-sample; do
  if [ "${SKIP_PREP:-0}" != "1" ]; then
    "$PY" src/lexical_baseline.py --pair "$PAIR" --split "$SPLIT"
    "$PY" src/evaluate.py --pair "$PAIR" --split "$SPLIT" --run-dir "outputs/$PAIR/$SPLIT/lexical"
  fi
  "$PY" src/score_laya.py --pair "$PAIR" --split "$SPLIT" --rep "$REP" --model "$CKPT" --method "$METHOD"
  "$PY" src/evaluate.py --pair "$PAIR" --split "$SPLIT" --run-dir "outputs/$PAIR/$SPLIT/${METHOD}_${REP}"
done

if [ "${FP32_VALID:-1}" = "1" ]; then
  "$PY" src/score_laya.py --pair "$PAIR" --split valid --rep "$REP" --model "$CKPT" \
      --method "${METHOD}_fp32" --precision fp32
  "$PY" src/evaluate.py --pair "$PAIR" --split valid --run-dir "outputs/$PAIR/valid/${METHOD}_fp32_${REP}"
fi
echo DONE

#!/bin/bash
#SBATCH -J laya-ft
#SBATCH -p gpu          # site-specific: set to your GPU partition
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 05:00:00
#SBATCH -o logs/%x-%j.out
# Fine-tune LAYA on BioML TRAIN (LAYA-BioML), then rank + officially score the VALID split.
#
#   sbatch scripts/finetune.slurm                               # NCIT-DOID, rep D, 1:5 negatives
#   sbatch --export=ALL,NEG=10 scripts/finetune.slurm           # negative-ratio ablation
#   sbatch --export=ALL,REP=A  scripts/finetune.slurm           # representation ablation
#
#   sbatch --export=ALL,NEG=10,TRAIN_SEED=1 scripts/finetune.slurm   # same data, new training seed
#
# Variables: PAIR (NCIT-DOID) REP (D) NEG (5) EPOCHS (4)
#            SEED (0)        seed of the DATA: train-dev split and sampled negatives
#            TRAIN_SEED (0)  seed of the optimisation only (init noise, shuffling); a non-zero
#                            value is appended to the run name so nothing is overwritten
#            SKIP_DATA (0)   1 = reuse the already-built fine-tuning data (required when several
#                            seeds of one pair run at once: they would rewrite the same files)
#            SCORE_VALID (1) 0 = stop after training (score VALID in a separate job)
# Checkpoint selection uses a reserved slice of TRAIN only; VALID is scored once, afterwards.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"   # submit from the repository root
print_env

PAIR=${PAIR:-NCIT-DOID}
REP=${REP:-D}
NEG=${NEG:-5}
EPOCHS=${EPOCHS:-4}
SEED=${SEED:-0}
TRAIN_SEED=${TRAIN_SEED:-0}
SUFFIX=""; if [ "$TRAIN_SEED" != "0" ]; then SUFFIX="_seed${TRAIN_SEED}"; fi
RUN="$(echo "$PAIR" | tr '[:upper:]' '[:lower:]')_rep${REP}_neg${NEG}${SUFFIX}"
METHOD="laya_bioml_neg${NEG}${SUFFIX}"

if [ "${SKIP_DATA:-0}" != "1" ]; then
  echo "===== build fine-tuning data (TRAIN only) ====="
  "$PY" src/build_finetune_data.py --pair "$PAIR" --rep "$REP" --neg-ratio "$NEG" --seed "$SEED"
fi

echo "===== fine-tune: $RUN ====="
"$PY" src/finetune_laya.py --pair "$PAIR" --rep "$REP" --neg-ratio "$NEG" --epochs "$EPOCHS" \
    --seed "$TRAIN_SEED" --run-name "$RUN"

if [ "${SCORE_VALID:-1}" != "1" ]; then echo "DONE $RUN (VALID not scored)"; exit 0; fi
echo "===== rank VALID with checkpoints/$RUN/best ====="
"$PY" src/score_laya.py --pair "$PAIR" --split valid --rep "$REP" --model "checkpoints/$RUN/best" \
    --method "$METHOD" --print-raw 4
for R in "${METHOD}_${REP}" "${METHOD}_${REP}.api4dp"; do
  "$PY" src/evaluate.py --pair "$PAIR" --split valid --run-dir "outputs/$PAIR/valid/$R"
done
echo "DONE $RUN"

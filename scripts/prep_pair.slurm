#!/bin/bash
#SBATCH -J prep-pair
#SBATCH -p cpu          # site-specific: set to your CPU partition
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
# CPU preparation of one pair, done ONCE before its GPU jobs (which then run with SKIP_DATA=1 /
# SKIP_PREP=1 / SKIP_LEXICAL=1 so concurrent jobs never rewrite shared files):
#   lexical reference on official VALID (validated + officially scored), the internal TRAIN splits
#   (train-dev / train-fit-sample) and their lexical scores, and the rep-D 1:5 fine-tuning data.
#
#   sbatch --export=ALL,PAIR=SNOMED-FMA scripts/prep_pair.slurm
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"   # submit from the repository root
print_env

PAIR=${PAIR:?set PAIR}
REP=${REP:-D}
NEG=${NEG:-5}

echo "===== lexical reference: $PAIR valid ====="
"$PY" src/lexical_baseline.py --pair "$PAIR" --split valid
"$PY" src/evaluate.py --pair "$PAIR" --split valid --run-dir "outputs/$PAIR/valid/lexical"

echo "===== internal splits of TRAIN ====="
"$PY" src/make_internal_splits.py --pair "$PAIR"
for SPLIT in train-dev train-fit-sample; do
  "$PY" src/lexical_baseline.py --pair "$PAIR" --split "$SPLIT"
  "$PY" src/evaluate.py --pair "$PAIR" --split "$SPLIT" --run-dir "outputs/$PAIR/$SPLIT/lexical"
done

echo "===== fine-tuning data (TRAIN only): rep $REP, 1:$NEG ====="
"$PY" src/build_finetune_data.py --pair "$PAIR" --rep "$REP" --neg-ratio "$NEG" --seed 0
echo DONE

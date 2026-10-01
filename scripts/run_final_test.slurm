#!/bin/bash
#SBATCH -J laya-final-test
#SBATCH -p gpu          # site-specific: set to your GPU partition
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
# Rank the TEST pools of one or more pairs with the system frozen by PROTOCOL.md
# (outputs/<pair>/final_selection.json, written by src/select_final.py from train-dev only).
#
#   sbatch --export=ALL,PAIRS_TO_RUN="SNOMED-NCIT" scripts/run_final_test.slurm
#
# Existing complete test scores (e.g. NCIT-DOID seed 0) are reused, never re-written. Packaging is
# a separate CPU step: src/make_submission.py --run final_lexical+laya_bioml_neg5_D@devalpha
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"   # submit from the repository root
print_env

PAIRS_TO_RUN=${PAIRS_TO_RUN:-"NCIT-DOID SNOMED-FMA SNOMED-NCIT"}
for PAIR in $PAIRS_TO_RUN; do
  SEL="outputs/$PAIR/final_selection.json"
  get() { "$PY" -c "import json,sys; print(json.load(open('$SEL'))['$1'])"; }
  LAYA_RUN=$(get laya_run); CKPT=$(get checkpoint); CHOICE=$(get choice_path); FINAL=$(get final_run_name)
  METHOD=${LAYA_RUN%_D}
  echo "===== $PAIR test: $LAYA_RUN ($CKPT), choice $CHOICE -> $FINAL ====="
  if [ ! -f "outputs/$PAIR/test/lexical/ranking.tsv" ]; then
    "$PY" src/lexical_baseline.py --pair "$PAIR" --split test
    "$PY" src/evaluate.py --pair "$PAIR" --split test --run-dir "outputs/$PAIR/test/lexical"
  fi
  META="outputs/$PAIR/test/$LAYA_RUN/run_meta.json"
  if [ -f "$META" ] && ! grep -q '"partial": true' "$META"; then
    echo "reusing complete test scores in outputs/$PAIR/test/$LAYA_RUN"
  else
    "$PY" src/score_laya.py --pair "$PAIR" --split test --rep D --model "$CKPT" --method "$METHOD"
    "$PY" src/evaluate.py --pair "$PAIR" --split test --run-dir "outputs/$PAIR/test/$LAYA_RUN"
  fi
  "$PY" src/ensemble.py --pair "$PAIR" --split test --laya-run "$LAYA_RUN" --choice "$CHOICE" --out-name "$FINAL"
  "$PY" src/evaluate.py --pair "$PAIR" --split test --run-dir "outputs/$PAIR/test/$FINAL"
done
echo DONE

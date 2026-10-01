#!/bin/bash
#SBATCH -J finalize-valid
#SBATCH -p cpu          # site-specific: set to your CPU partition
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
# CPU step after all LAYA-BioML scoring (PROTOCOL.md): per pair and training seed, tune the
# ensemble (mode, alpha) on train-dev and apply it unchanged to VALID; freeze the train-dev-selected
# seed (src/select_final.py); write the FINAL system's VALID ranking.
# Existing outputs are reused, never re-written.
#
#   sbatch --export=ALL,PAIRS_TO_RUN="SNOMED-FMA" scripts/finalize_valid.slurm
#
# COLLECT (1): set 0 when several of these jobs run at once (the shared results table is
# rewritten whole, so it is not safe from concurrent jobs).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"   # submit from the repository root
print_env

PAIRS_TO_RUN=${PAIRS_TO_RUN:-"NCIT-DOID SNOMED-FMA SNOMED-NCIT"}
FINAL=final_lexical+laya_bioml_neg5_D@devalpha
for PAIR in $PAIRS_TO_RUN; do
  echo "===== $PAIR ====="
  for S in 0 1 2; do
    SFX=""; if [ "$S" != "0" ]; then SFX="_seed$S"; fi
    X="laya_bioml_neg5${SFX}_D"
    CFG="checkpoints/$(echo "$PAIR" | tr '[:upper:]' '[:lower:]')_repD_neg5${SFX}/train_config.json"
    if ! "$PY" -c "import json,sys; c=json.load(open('$CFG')); sys.exit(len(c['epochs']) < c['args']['epochs'])" 2>/dev/null; then
      echo "seed $S: training did not complete (PROTOCOL.md amendment 1) -> excluded"; continue
    fi
    DEV="outputs/$PAIR/train-dev/ens_lexical+$X"
    if [ ! -f "$DEV/metrics.official.json" ]; then
      "$PY" src/ensemble.py --pair "$PAIR" --split train-dev --laya-run "$X" | tail -1
      "$PY" src/evaluate.py --pair "$PAIR" --split train-dev --run-dir "$DEV" | tail -1
    fi
    VAL="outputs/$PAIR/valid/ens_lexical+$X@devalpha"
    if [ ! -f "$VAL/metrics.official.json" ]; then
      "$PY" src/ensemble.py --pair "$PAIR" --split valid --laya-run "$X" --suffix @devalpha \
          --choice "$DEV/ensemble_choice.json" | tail -1
      "$PY" src/evaluate.py --pair "$PAIR" --split valid --run-dir "$VAL" | tail -1
    fi
  done
  "$PY" src/select_final.py --pair "$PAIR"
  SEL="outputs/$PAIR/final_selection.json"
  get() { "$PY" -c "import json; print(json.load(open('$SEL'))['$1'])"; }
  "$PY" src/ensemble.py --pair "$PAIR" --split valid --laya-run "$(get laya_run)" \
      --choice "$(get choice_path)" --out-name "$FINAL" | tail -1
  "$PY" src/evaluate.py --pair "$PAIR" --split valid --run-dir "outputs/$PAIR/valid/$FINAL" | tail -1
done
if [ "${COLLECT:-1}" = "1" ]; then "$PY" src/evaluate.py --collect; fi
echo DONE

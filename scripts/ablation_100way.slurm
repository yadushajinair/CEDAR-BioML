#!/bin/bash
#SBATCH -J laya-100way
#SBATCH -p gpu          # site-specific: set to your GPU partition
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
# SECONDARY ablation only: one 100-option LAYA `choice` question per query, on the same VALID
# queries as the pairwise ranker. head_max_len / max_len are raised per query (see the script).
#
#   sbatch scripts/ablation_100way.slurm
#   sbatch --export=ALL,MODEL=checkpoints/ncit-doid_repD_neg5/best,METHOD=laya100_bioml scripts/ablation_100way.slurm
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"   # submit from the repository root
print_env

PAIR=${PAIR:-NCIT-DOID}
MODEL=${MODEL:-convaiinnovations/laya}
METHOD=${METHOD:-laya100_zeroshot}

"$PY" src/score_laya_100way.py --pair "$PAIR" --split valid --model "$MODEL" --method "${METHOD}_label"
"$PY" src/evaluate.py --pair "$PAIR" --split valid --run-dir "outputs/$PAIR/valid/${METHOD}_label"
"$PY" src/score_laya_100way.py --pair "$PAIR" --split valid --model "$MODEL" --method "${METHOD}_label+syn" \
    --option-synonyms
"$PY" src/evaluate.py --pair "$PAIR" --split valid --run-dir "outputs/$PAIR/valid/${METHOD}_label+syn"
echo DONE

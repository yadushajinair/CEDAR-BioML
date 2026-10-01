#!/bin/bash
# Login-node-safe smoke test (CPU, a few threads, ~30 s, ~3.5 GB RAM):
#   data structure checks -> official kit self-check -> lexical baseline -> 10 LAYA pairs on CPU.
# Nothing here touches a GPU or scores more than a handful of pairs with LAYA.
set -euo pipefail
source "$(dirname "$0")/env.sh"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3

"$PY" src/inspect_data.py | tail -3
"$PY" external/OAEI-Bio-ML/scoring_kit/self_check.py --data data/bio-ml | tail -4
"$PY" src/lexical_baseline.py --pair NCIT-DOID --split valid
"$PY" src/evaluate.py --pair NCIT-DOID --split valid --run-dir outputs/NCIT-DOID/valid/lexical
"$PY" src/smoke_laya.py --pair NCIT-DOID --split train --n-queries 5 --device cpu | tail -4

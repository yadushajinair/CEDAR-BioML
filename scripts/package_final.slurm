#!/bin/bash
#SBATCH -J package-final
#SBATCH -p cpu          # site-specific: set to your CPU partition
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
# Validate the three FINAL test rankings (official validate_ranking.py + strict checks) and build
# the flat three-file submission ZIP. Never uploads anything.
#
#   sbatch scripts/package_final.slurm
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"   # submit from the repository root
print_env
FINAL=final_lexical+laya_bioml_neg5_D@devalpha
"$PY" src/make_submission.py --run "$FINAL" --name submission_final_3pairs
ZIP=outputs/submission/submission_final_3pairs.zip
test -f "$ZIP" || { echo "complete ZIP not produced (a pair is missing?)"; exit 1; }
"$PY" - "$ZIP" <<'PYEOF'
import sys, zipfile
names = sorted(zipfile.ZipFile(sys.argv[1]).namelist())
assert names == ["ncit-doid.tsv", "snomed-fma.tsv", "snomed-ncit.tsv"], names
print("ZIP contents OK:", names)
PYEOF
echo DONE

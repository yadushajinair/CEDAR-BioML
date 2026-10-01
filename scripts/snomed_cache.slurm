#!/bin/bash
#SBATCH -J snomed-cache
#SBATCH -p cpu          # site-specific: set to your CPU partition
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
# Build the SNOMED CT entity cache from YOUR licensed OFN file (data/private/*snomed*.owl or
# $SNOMED_OWL), then verify it against the RDF/XML serialisation of the same release.
# SNOMED CT is licence-restricted and is not part of this repository.
#
#   sbatch --export=ALL,SNOMED_RDFXML=/path/to/<release>.rdfxml.owl scripts/snomed_cache.slurm
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/env.sh"   # submit from the repository root
print_env

SNOMED_RDFXML=${SNOMED_RDFXML:?set SNOMED_RDFXML to the RDF/XML file of your licensed SNOMED CT release}

echo "===== SNOMED entity cache ====="
/usr/bin/time -v "$PY" src/ontology_loader.py --ontology SNOMED --discover 2>&1 | \
    grep -vE '^\s+(Page size|Exit status|Signals|Swaps|Socket|File system|Average|Voluntary|Involuntary|Minor|Major|Percent|System time|User time)'
echo "===== verify against RDF/XML ====="
/usr/bin/time -v "$PY" src/verify_snomed.py --rdfxml "$SNOMED_RDFXML" 2>&1 | \
    grep -vE '^\s+(Page size|Exit status|Signals|Swaps|Socket|File system|Average|Voluntary|Involuntary|Minor|Major|Percent|System time|User time)'
echo DONE

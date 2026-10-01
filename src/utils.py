"""Shared paths, task configuration, candidate-file I/O and ranking metrics."""
from __future__ import annotations

import ast
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "bio-ml"
ONTO_DIR = DATA / "ontologies"
PRIVATE_DIR = ROOT / "data" / "private"  # licence-gated files (SNOMED CT) go here, never in git
CACHE = ROOT / "cache"
ENTITY_CACHE = CACHE / "entities"
OUTPUTS = ROOT / "outputs"
SCORING_KIT = ROOT / "external" / "OAEI-Bio-ML" / "scoring_kit"

CANDIDATE_COUNT = 100
SPLITS = ("train", "valid", "test")          # the official BioML files
# Internal splits carved out of the official TRAIN file (src/make_internal_splits.py):
#   train-dev         reserved TRAIN queries, used ONLY for checkpoint / hyper-parameter selection
#   train-fit-sample  a fixed sample of the TRAIN queries the model was fine-tuned on
INTERNAL_SPLITS = ("train-dev", "train-fit-sample")
ALL_SPLITS = SPLITS + INTERNAL_SPLITS

# pair -> source ontology, target ontology, CodaBench filename slug (hyphenated; see
# external/OAEI-Bio-ML/scoring_kit/docs/submission-format-local.md)
PAIRS = {
    "NCIT-DOID": {"src": "NCIT", "tgt": "DOID", "slug": "ncit-doid"},
    "SNOMED-FMA": {"src": "SNOMED", "tgt": "FMA", "slug": "snomed-fma"},
    "SNOMED-NCIT": {"src": "SNOMED", "tgt": "NCIT", "slug": "snomed-ncit"},
}

# IRI prefix of each ontology's classes, used to route an IRI to its ontology.
IRI_PREFIX = {
    "NCIT": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#",
    "DOID": "http://purl.obolibrary.org/obo/DOID_",
    "FMA": "http://purl.org/sig/ont/fma/",
    "SNOMED": "http://snomed.info/id/",
}


def ontology_path(name: str) -> Path | None:
    """Locate an ontology file. Open ontologies are discovered in the dataset's ontologies/
    directory (`<NAME>-<pin>-*.owl`); SNOMED CT is licence-gated and must be supplied by the
    user via $SNOMED_OWL or data/private/. Returns None when the file is not available."""
    if name == "SNOMED":
        env = os.environ.get("SNOMED_OWL")
        if env:
            return Path(env) if Path(env).is_file() else None
        hits = sorted(glob.glob(str(PRIVATE_DIR / "*[Ss][Nn][Oo][Mm][Ee][Dd]*.owl")))
        return Path(hits[0]) if hits else None
    hits = sorted(glob.glob(str(ONTO_DIR / f"{name}-*.owl")))
    return Path(hits[0]) if hits else None


def cands_path(pair: str, split: str) -> Path:
    if split in INTERNAL_SPLITS:
        return OUTPUTS / "internal_splits" / pair / f"local.{split}.cands.tsv"
    return DATA / pair / f"local.{split}.cands.tsv"


def read_cands(pair: str, split: str) -> list[dict]:
    """One dict per query, in file order: {"src", "gold" (None on test), "cands" (list of IRIs)}."""
    out = []
    with open(cands_path(pair, split), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out.append({
                "src": row["SrcEntity"],
                "gold": row.get("TgtEntity") or None,
                "cands": [str(c) for c in ast.literal_eval(row["TgtCandidates"])],
            })
    return out


def write_ranking(path: str | Path, queries: list[dict], rankings: list[list[str]]) -> None:
    """Write the official LIST-form submission: SrcEntity <TAB> TgtCandidates (Python list
    literal, best first), one row per query in the pool's order. Refuses anything that is not
    an exact permutation of the pool."""
    assert len(queries) == len(rankings), "one ranking per query"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["SrcEntity", "TgtCandidates"])
        for i, (q, ranked) in enumerate(zip(queries, rankings)):
            if len(ranked) != len(q["cands"]) or sorted(ranked) != sorted(q["cands"]):
                raise ValueError(f"query {i} ({q['src']}): ranking is not a permutation of its pool")
            w.writerow([q["src"], repr(list(ranked))])


def rank_by_scores(cands: list[str], scores) -> list[str]:
    """Sort candidates by descending score; ties keep the original pool position (stable)."""
    order = sorted(range(len(cands)), key=lambda j: (-float(scores[j]), j))
    return [cands[j] for j in order]


def ranking_metrics(queries: list[dict], rankings: list[list[str]]) -> dict:
    """MRR + Hits@{1,5,10}; mirrors oaei_bioml_eval.equivalence.metrics.local_ranking_metrics
    (used for fast sweeps -- headline numbers always come from the official score_local.py)."""
    rr, h1, h5, h10 = [], [], [], []
    for q, ranked in zip(queries, rankings):
        try:
            rank = ranked.index(q["gold"]) + 1
        except ValueError:
            rank = None
        rr.append(1.0 / rank if rank else 0.0)
        h1.append(1.0 if rank and rank <= 1 else 0.0)
        h5.append(1.0 if rank and rank <= 5 else 0.0)
        h10.append(1.0 if rank and rank <= 10 else 0.0)
    n = max(1, len(rr))
    return {"mrr": sum(rr) / n, "hits_at_1": sum(h1) / n, "hits_at_5": sum(h5) / n,
            "hits_at_10": sum(h10) / n, "queries": len(rr)}


def fmt_metrics(m: dict) -> str:
    return "MRR=%.4f  H@1=%.4f  H@5=%.4f  H@10=%.4f  (n=%d)" % (
        m["mrr"], m["hits_at_1"], m["hits_at_5"], m["hits_at_10"], m["queries"])


def read_scores(path: str | Path) -> dict[tuple[int, int], dict]:
    """Read a per-pair score TSV -> {(query_index, candidate_position): row}."""
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out[(int(row["QueryIndex"]), int(row["OriginalCandidatePosition"]))] = row
    return out


def score_matrix(path: str | Path, queries: list[dict], column: str) -> list[list[float]]:
    """Per-query list of `column` values aligned to the pool order; fails on any missing cell."""
    rows = read_scores(path)
    mat = []
    for qi, q in enumerate(queries):
        vals = []
        for ci, cand in enumerate(q["cands"]):
            row = rows.get((qi, ci))
            if row is None:
                raise KeyError(f"{path}: missing score for query {qi} candidate {ci}")
            if row["SrcEntity"] != q["src"] or row["TgtCandidate"] != cand:
                raise ValueError(f"{path}: row ({qi},{ci}) does not match the pool")
            vals.append(float(row[column]))
        mat.append(vals)
    return mat


def dump_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0

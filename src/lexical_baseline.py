"""Transparent lexical reference scorer (no learning, no external resources).

For a (source, target) pair, with N(x) = the normalised label + synonyms of x:

  E_label  1 if the two preferred labels are equal after normalisation
  E_name   1 if any name of the source equals any name of the target
  J        best token-set Jaccard over all name pairs
  R        best character similarity (normalised indel ratio, rapidfuzz) over all name pairs

  LexScore = (E_label + E_name + J + R) / 4          in [0, 1]

The weights are fixed (not tuned). Exact matches dominate; J and R order the rest.

Usage:
  python src/lexical_baseline.py --pair NCIT-DOID --split valid
"""
from __future__ import annotations

import argparse
import csv
import re
import string
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process

from ontology_loader import load_entities
from utils import (ALL_SPLITS, OUTPUTS, PAIRS, Timer, dump_json, fmt_metrics, rank_by_scores,
                   ranking_metrics, read_cands, write_ranking)

_PUNCT = re.compile("[%s]" % re.escape(string.punctuation.replace("'", "")))
_WS = re.compile(r"\s+")
_KEEP_S = ("ss", "us", "is", "as", "os")  # e.g. "abscess", "virus", "sepsis", "pancreas"


def singular(token: str) -> str:
    """Conservative de-pluralisation: strip one trailing 's' from longer words only."""
    if len(token) > 4 and token.endswith("s") and not token.endswith(_KEEP_S):
        return token[:-1]
    return token


def normalize(text: str) -> str:
    text = text.lower().replace("'s", "").replace("'", "")
    text = _PUNCT.sub(" ", text)  # also turns underscores, hyphens and slashes into spaces
    return " ".join(singular(t) for t in _WS.sub(" ", text).split())


def names(entity: dict) -> list[str]:
    out, seen = [], set()
    for raw in [entity["label"]] + entity["synonyms"] + entity["related_synonyms"]:
        norm = normalize(raw)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out or [""]


class LexicalScorer:
    def __init__(self, pair: str):
        info = PAIRS[pair]
        self.src_entities = load_entities(info["src"])
        self.tgt_entities = load_entities(info["tgt"])
        self._names: dict[str, list[str]] = {}
        self._tokens: dict[str, list[frozenset]] = {}

    def _prep(self, iri: str, entities: dict):
        if iri not in self._names:
            self._names[iri] = names(entities[iri])
            self._tokens[iri] = [frozenset(n.split()) for n in self._names[iri]]
        return self._names[iri], self._tokens[iri]

    def features(self, src: str, tgt: str) -> dict:
        s_names, s_tok = self._prep(src, self.src_entities)
        t_names, t_tok = self._prep(tgt, self.tgt_entities)
        e_label = 1.0 if s_names[0] == t_names[0] else 0.0
        e_name = 1.0 if set(s_names) & set(t_names) else 0.0
        jac = max((len(a & b) / len(a | b)) if (a | b) else 0.0 for a in s_tok for b in t_tok)
        ratio = float(np.max(process.cdist(s_names, t_names, scorer=fuzz.ratio))) / 100.0
        return {"E_label": e_label, "E_name": e_name, "J": jac, "R": ratio,
                "LexScore": (e_label + e_name + jac + ratio) / 4.0}

    def score_query(self, query: dict) -> list[dict]:
        return [self.features(query["src"], cand) for cand in query["cands"]]


COLUMNS = ["QueryIndex", "SrcEntity", "TgtCandidate", "OriginalCandidatePosition",
           "LexScore", "E_label", "E_name", "J", "R"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS))
    ap.add_argument("--split", required=True, choices=list(ALL_SPLITS))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUTPUTS / args.pair / args.split / "lexical"
    queries = read_cands(args.pair, args.split)
    scorer = LexicalScorer(args.pair)
    rankings = {k: [] for k in ("LexScore", "E_label", "E_name", "J", "R")}
    out_dir.mkdir(parents=True, exist_ok=True)
    with Timer() as t, open(out_dir / "scores.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(COLUMNS)
        for qi, q in enumerate(queries):
            feats = scorer.score_query(q)
            for ci, (cand, f) in enumerate(zip(q["cands"], feats)):
                w.writerow([qi, q["src"], cand, ci] + ["%.6f" % f[k] for k in COLUMNS[4:]])
            for key in rankings:
                rankings[key].append(rank_by_scores(q["cands"], [f[key] for f in feats]))
    write_ranking(out_dir / "ranking.tsv", queries, rankings["LexScore"])
    n_pairs = sum(len(q["cands"]) for q in queries)
    dump_json(out_dir / "run_meta.json", {
        "method": "lexical", "checkpoint": "-", "representation": "label+synonyms (names only)",
        "runtime_s": t.seconds, "queries": len(queries), "pairs_scored": n_pairs,
        "pairs_per_sec": n_pairs / max(t.seconds, 1e-9), "gpu": "none (CPU)"})
    print(f"[lexical] {args.pair} {args.split}: {len(queries)} queries, {n_pairs} pairs, "
          f"{t.seconds:.1f}s ({n_pairs / max(t.seconds, 1e-9):.0f} pairs/s) -> {out_dir}")
    if queries[0]["gold"] is not None:
        for key, ranked in rankings.items():
            print(f"   {key:8s} {fmt_metrics(ranking_metrics(queries, ranked))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Cross-check the streaming ontology loader against rdflib (an independent RDF parser).

Parses a (small) ontology into an rdflib graph and compares label, synonyms, definition and
named parents of every cached task entity with what ontology_loader.py extracted.

  python src/verify_loader_rdflib.py --ontology DOID
"""
from __future__ import annotations

import argparse

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from ontology_loader import load_entities, roles_for
from utils import ontology_path


def english(values):
    return [" ".join(str(v).split()) for v in values
            if isinstance(v, Literal) and (v.language is None or v.language.lower().startswith("en"))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", default="DOID")
    args = ap.parse_args()
    roles = roles_for(args.ontology)
    graph = Graph()
    graph.parse(str(ontology_path(args.ontology)), format="xml")
    entities = load_entities(args.ontology)
    print(f"[verify] rdflib graph: {len(graph)} triples; checking {len(entities)} cached entities")

    def label_of(node):
        for prop in roles["label"]:
            vals = english(graph.objects(node, URIRef(prop)))
            if vals:
                return vals[0]
        return None

    mismatches = {"label": 0, "synonyms": 0, "definition": 0, "parents": 0}
    for iri, ent in entities.items():
        node = URIRef(iri)
        label = label_of(node)
        if label != (None if ent["label_is_fallback"] else ent["label"]):
            mismatches["label"] += 1
        syn = {v.lower() for p in roles["syn_exact"] + roles["syn_related"]
               for v in english(graph.objects(node, URIRef(p)))}
        syn.discard((label or "").lower())
        if syn != {s.lower() for s in ent["synonyms"] + ent["related_synonyms"]}:
            mismatches["synonyms"] += 1
        defs = [v for p in roles["definition"] for v in english(graph.objects(node, URIRef(p)))]
        if (defs[0] if defs else "") != ent["definition"]:
            mismatches["definition"] += 1
        parents = set()
        for obj in graph.objects(node, RDFS.subClassOf):
            if isinstance(obj, URIRef):
                parents.add(obj)
        for eq in graph.objects(node, OWL.equivalentClass):
            for lst in graph.objects(eq, OWL.intersectionOf):
                while lst and lst != RDF.nil:
                    first = graph.value(lst, RDF.first)
                    if isinstance(first, URIRef):
                        parents.add(first)
                    lst = graph.value(lst, RDF.rest)
        parent_labels = {lab.lower() for lab in (label_of(p) for p in parents) if lab}
        cached = {p.lower() for p in ent["parents"]}
        # the cache keeps at most 12 parent labels
        if not (cached <= parent_labels and (len(parent_labels) <= 12 and cached == parent_labels
                                             or len(parent_labels) > 12)):
            mismatches["parents"] += 1
    print(f"[verify] mismatches vs rdflib over {len(entities)} entities: {mismatches}")
    return 0 if not any(mismatches.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

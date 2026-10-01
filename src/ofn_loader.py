"""Streaming OWL Functional Syntax (OFN) loader, used for SNOMED CT.

The authorised SNOMED CT file (US Edition 2026-03-01, View-A conversion, class IRIs
http://snomed.info/id/<id>) ships in two serialisations with identical content. The RDF/XML one
cannot go through ontology_loader.parse_owl: its class expressions are flattened into blank nodes
(rdf:nodeID) whose triples are split over separate top-level blocks, so named parents that sit
inside an ObjectIntersectionOf would be lost. The OFN file has one axiom per line, which makes a
line-by-line parse exact and cheap. `src/verify_snomed.py` cross-checks this parser against the
RDF/XML file with an independent blank-node-resolving parser.

Returns the same structure as ontology_loader.parse_owl:
    classes[iri] = {"ann": {property_iri: [literal, ...]}, "parents": [iri, ...], "deprecated": bool}

Parents follow the same rule as parse_owl: named superclasses from SubClassOf, plus the named
top-level members of an ObjectIntersectionOf on the right of SubClassOf or EquivalentClasses
(the genus concepts). Named classes inside restrictions (attribute values) are NOT parents.
GCIs (a complex class on the left of SubClassOf) describe no named class and are skipped.

SNOMED attribute concepts (e.g. 272741003, 42752001) are declared as OWL
object/data properties in the View-A conversion, not as classes, yet some are BioML sources. They
are therefore loaded as entities too (entity_type records which), with their super-property from
SubObjectPropertyOf / SubDataPropertyOf as parent -- the attribute hierarchy's is-a relation.

Literal annotations keep their language tag: a value is stored as-is under its property, and the
tag is recorded in a parallel "ann_lang" map so role assignment can prefer a dialect.
"""
from __future__ import annotations

import re
from pathlib import Path

_PREFIX = re.compile(r"^Prefix\(([A-Za-z0-9_-]*):=<([^>]*)>\)\s*$")
_DECL = re.compile(r"^Declaration\((Class|ObjectProperty|DataProperty|AnnotationProperty)\((\S+)\)\)\s*$")
_SUBPROP = ("SubObjectPropertyOf(", "SubDataPropertyOf(", "SubAnnotationPropertyOf(")
_ANN = re.compile(r'^AnnotationAssertion\((\S+) (\S+) "((?:[^"\\]|\\.)*)"(?:@([A-Za-z-]+)|\^\^(\S+))?\)\s*$')
_TOKEN = re.compile(r'\(|\)|<[^>]*>|[^\s()]+')
_UNESCAPE = re.compile(r"\\(.)")

DEPRECATED = "http://www.w3.org/2002/07/owl#deprecated"


class OFNError(ValueError):
    pass


def _expand(token: str, prefixes: dict[str, str]) -> str:
    if token.startswith("<") and token.endswith(">"):
        return token[1:-1]
    if ":" in token:
        pfx, local = token.split(":", 1)
        if pfx in prefixes:
            return prefixes[pfx] + local
    raise OFNError(f"cannot expand {token!r}")


def _parse_expr(tokens: list[str], i: int):
    """Parse one class expression starting at tokens[i]. Returns (node, next_index) where node is
    ("named", token) or (constructor, [child nodes])."""
    tok = tokens[i]
    if tok in ("(", ")"):
        raise OFNError(f"unexpected {tok!r} at {i}")
    if i + 1 < len(tokens) and tokens[i + 1] == "(":
        children, j = [], i + 2
        while tokens[j] != ")":
            child, j = _parse_expr(tokens, j)
            children.append(child)
        return (tok, children), j + 1
    return ("named", tok), i + 1


def _axiom_args(line: str) -> tuple[str, list]:
    tokens = _TOKEN.findall(line)
    node, end = _parse_expr(tokens, 0)
    if end != len(tokens):
        raise OFNError(f"trailing tokens in {line[:120]!r}")
    return node


def _genus(node, prefixes) -> list[str]:
    """Named classes a class expression makes a superclass: itself if named, or the named
    top-level members of an ObjectIntersectionOf."""
    kind, val = node
    if kind == "named":
        return [_expand(val, prefixes)]
    if kind == "ObjectIntersectionOf":
        return [_expand(c[1], prefixes) for c in val if c[0] == "named"]
    return []


def parse_ofn(path: Path, keep_all_annotations_for: set[str], label_props: set[str]):
    """Single pass over an OFN file. Returns (entities, stats); keyed by IRI, classes and declared
    properties alike (see module docstring), each with an "entity_type".

    Every literal annotation is kept for entities in `keep_all_annotations_for`; for all others
    only properties in `label_props` are kept (enough to label parents)."""
    prefixes: dict[str, str] = {}
    classes: dict[str, dict] = {}
    stats = {"lines": 0, "declared_classes": 0, "declared_ObjectProperty": 0,
             "declared_DataProperty": 0, "declared_AnnotationProperty": 0,
             "subclass_named": 0, "subclass_intersection": 0,
             "equivalent_intersection": 0, "gci_skipped": 0, "equivalent_other_skipped": 0,
             "subproperty_named": 0, "subproperty_chain_skipped": 0, "other_axioms_skipped": 0,
             "annotation_assertions": 0, "annotations_on_undeclared": 0}

    def rec(iri: str, kind: str = "Class") -> dict:
        return classes.setdefault(iri, {"ann": {}, "ann_lang": {}, "parents": [],
                                        "deprecated": False, "entity_type": kind})

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stats["lines"] += 1
            if not line or line[0] in "#\n )" or line.startswith("<"):
                continue
            if line.startswith("Prefix("):
                m = _PREFIX.match(line)
                if not m:
                    raise OFNError(f"bad prefix line {line!r}")
                prefixes[m.group(1)] = m.group(2)
                continue
            if line.startswith("Ontology("):
                continue
            if line.startswith("Declaration("):
                m = _DECL.match(line)
                if m:
                    kind = m.group(1)
                    rec(_expand(m.group(2), prefixes), kind)
                    stats["declared_classes" if kind == "Class" else f"declared_{kind}"] += 1
                continue
            if line.startswith("AnnotationAssertion("):
                stats["annotation_assertions"] += 1
                m = _ANN.match(line)
                if not m:
                    raise OFNError(f"unparsed annotation {line[:160]!r}")
                prop = _expand(m.group(1), prefixes)
                subj = _expand(m.group(2), prefixes)
                if subj not in classes:
                    stats["annotations_on_undeclared"] += 1
                    continue
                value = _UNESCAPE.sub(r"\1", m.group(3)).strip()
                lang = m.group(4)
                if lang and not lang.lower().startswith("en"):
                    continue
                if prop == DEPRECATED:
                    classes[subj]["deprecated"] = value.lower() == "true"
                    continue
                if value and (subj in keep_all_annotations_for or prop in label_props):
                    r = classes[subj]
                    r["ann"].setdefault(prop, []).append(value)
                    r["ann_lang"].setdefault(prop, []).append(lang or "")
                continue
            if line.startswith("SubClassOf("):
                _, (sub, sup) = _axiom_args(line)
                if sub[0] != "named":
                    stats["gci_skipped"] += 1
                    continue
                stats["subclass_named" if sup[0] == "named" else "subclass_intersection"] += 1
                rec(_expand(sub[1], prefixes))["parents"].extend(_genus(sup, prefixes))
                continue
            if line.startswith("EquivalentClasses("):
                _, operands = _axiom_args(line)
                named = [o for o in operands if o[0] == "named"]
                complex_ = [o for o in operands if o[0] != "named"]
                if len(named) == 1 and len(complex_) == 1:
                    stats["equivalent_intersection"] += 1
                    rec(_expand(named[0][1], prefixes))["parents"].extend(
                        _genus(complex_[0], prefixes))
                else:  # never occurs in this release; counted so a change would be visible
                    stats["equivalent_other_skipped"] += 1
                continue
            if line.startswith(_SUBPROP):
                _, (sub, sup) = _axiom_args(line)
                if sub[0] == "named" and sup[0] == "named":
                    stats["subproperty_named"] += 1
                    classes[_expand(sub[1], prefixes)]["parents"].append(_expand(sup[1], prefixes))
                else:  # SubObjectPropertyOf(ObjectPropertyChain(...) :p)
                    stats["subproperty_chain_skipped"] += 1
                continue
            stats["other_axioms_skipped"] += 1  # Transitive/ReflexiveObjectProperty
    for r in classes.values():
        r["parents"] = list(dict.fromkeys(r["parents"]))
    return classes, stats


def is_ofn(path: Path) -> bool:
    with open(path, encoding="utf-8", errors="replace") as fh:
        head = fh.read(4096)
    return head.lstrip().startswith("Prefix(")

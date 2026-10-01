"""Streaming OWL (RDF/XML) loader -> compact per-ontology entity cache.

The BioML ontologies are large (NCIT is ~750 MB of RDF/XML), so instead of materialising an
rdflib graph we stream the file with xml.etree.iterparse and keep only what the matcher needs:

  * every named class's label(s) and named parents (for parent/child context), and
  * for the classes that occur in a BioML candidate file ("task entities"), every literal
    annotation, so the annotation properties actually present can be *discovered* and reported
    rather than assumed.

Annotation properties are then mapped to roles (label / exact synonym / related synonym /
definition) through an explicit WHITELIST. Cross-reference properties (oboInOwl:hasDbXref,
skos:*Match, NCIT P207 UMLS_CUI, P208 NCI_META_CUI, P375 Maps_To, ...) are never read into a
role: they are UMLS/Mondo-derived links to other vocabularies and using them would break the
competition rules.

Usage:
  python src/ontology_loader.py --ontology DOID            # parse + cache + coverage report
  python src/ontology_loader.py --ontology NCIT --discover # also print the property census
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

from ofn_loader import is_ofn, parse_ofn
from utils import ENTITY_CACHE, IRI_PREFIX, PAIRS, SPLITS, Timer, ontology_path, read_cands

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
SKOS = "http://www.w3.org/2004/02/skos/core#"
OBOINOWL = "http://www.geneontology.org/formats/oboInOwl#"
OBO = "http://purl.obolibrary.org/obo/"
NCIT = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"
FMA = "http://purl.org/sig/ont/fma/"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

T_CLASS = "{%s}Class" % OWL
T_DESC = "{%s}Description" % RDF
T_ANNPROP = "{%s}AnnotationProperty" % OWL
T_SUBCLASS = "{%s}subClassOf" % RDFS
T_EQUIV = "{%s}equivalentClass" % OWL
T_INTERSECTION = "{%s}intersectionOf" % OWL
T_DEPRECATED = "{%s}deprecated" % OWL
T_TYPE = "{%s}type" % RDF
A_ABOUT = "{%s}about" % RDF
A_RESOURCE = "{%s}resource" % RDF

# ---- role whitelist -------------------------------------------------------------------------
# Standard vocabulary first; ontology-specific properties were added after running --discover
# on the actual files (see README "Entity representation").
ROLES_GENERIC = {
    "label": [RDFS + "label", SKOS + "prefLabel"],
    "syn_exact": [SKOS + "altLabel", OBOINOWL + "hasExactSynonym"],
    "syn_related": [OBOINOWL + "hasRelatedSynonym", OBOINOWL + "hasNarrowSynonym",
                    OBOINOWL + "hasBroadSynonym"],
    "definition": [OBO + "IAO_0000115", SKOS + "definition"],
}
ROLES_EXTRA = {
    # NCIT: P108 Preferred_Name, P90 FULL_SYN, P97 DEFINITION, P325 ALT_DEFINITION
    "NCIT": {"label": [NCIT + "P108"], "syn_exact": [NCIT + "P90"],
             "definition": [NCIT + "P97", NCIT + "P325"]},
    "DOID": {},
    # FMA 5.x: fma:preferred_name, fma:synonym, fma:definition
    "FMA": {"label": [FMA + "preferred_name"], "syn_exact": [FMA + "synonym"],
            "definition": [FMA + "definition"]},
}
# SNOMED CT (View-A OWL, US Edition 2026-03-01) carries exactly four annotation properties:
# rdfs:label = fully specified name WITH its semantic tag ("<term> (body structure)"),
# skos:prefLabel = preferred term, skos:altLabel = synonyms, skos:definition. The generic order
# would make the FSN (tag included) the label, so SNOMED gets its own role map: the preferred term
# is the label and the FSN term (semantic tag stripped, see snomed_canonicalise) is a synonym.
# There are no cross-reference properties in the file.
SNOMED_FSN_TERM = "urn:laya-bioml:snomed-fsn-term"  # derived: rdfs:label minus its semantic tag
ROLES_OVERRIDE = {
    "SNOMED": {"label": [SKOS + "prefLabel"],
               "syn_exact": [SNOMED_FSN_TERM, SKOS + "altLabel"],
               "syn_related": [],
               "definition": [SKOS + "definition"]},
}
# US Edition: when a description differs by dialect, the en-us value comes first.
DIALECT_RANK = {"en-us": 0, "en": 1, "": 1, "en-gb": 2}
_SEMANTIC_TAG = re.compile(r"^(.*\S)\s+\(([^()]+)\)$")


def roles_for(ontology: str) -> dict[str, list[str]]:
    if ontology in ROLES_OVERRIDE:
        return {k: list(v) for k, v in ROLES_OVERRIDE[ontology].items()}
    roles = {k: list(v) for k, v in ROLES_GENERIC.items()}
    for role, props in ROLES_EXTRA.get(ontology, {}).items():
        roles[role] = roles[role] + [p for p in props if p not in roles[role]]
    return roles


def task_entities(ontology: str) -> set[str]:
    """All IRIs of `ontology` that occur as a query or candidate in any pair/split."""
    out: set[str] = set()
    for pair, info in PAIRS.items():
        if ontology not in (info["src"], info["tgt"]):
            continue
        for split in SPLITS:
            for q in read_cands(pair, split):
                if info["src"] == ontology:
                    out.add(q["src"])
                if info["tgt"] == ontology:
                    out.update(q["cands"])
    return out


def _tag_iri(tag: str) -> str:
    return tag[1:].replace("}", "", 1) if tag.startswith("{") else tag


def _named_members(el) -> list[str]:
    """Named classes directly listed in an owl:intersectionOf (the genus of a defined class)."""
    out = []
    for member in el:
        about = member.get(A_ABOUT)
        if about and member.tag in (T_DESC, T_CLASS):
            out.append(about)
    return out


def parse_owl(path: Path, keep_all_annotations_for: set[str]):
    """Single streaming pass. Returns (classes, annprop_labels).

    classes[iri] = {"ann": {property_iri: [literal, ...]}, "parents": [iri, ...],
                    "deprecated": bool}
    For classes outside `keep_all_annotations_for` only label-role properties are retained.
    """
    label_props = set(ROLES_GENERIC["label"]) | {NCIT + "P108", FMA + "preferred_name"}
    classes: dict[str, dict] = {}
    annprop_labels: dict[str, str] = {}
    depth = 0
    root = None
    for event, el in ET.iterparse(str(path), events=("start", "end")):
        if event == "start":
            depth += 1
            if depth == 1:
                root = el
            continue
        depth -= 1
        if depth != 1:
            continue
        about = el.get(A_ABOUT)
        tag = el.tag
        if about and tag == T_ANNPROP:
            for child in el:
                if child.tag == "{%s}label" % RDFS and child.text:
                    annprop_labels[about] = child.text.strip()
        is_class = tag == T_CLASS or (tag == T_DESC and any(
            c.tag == T_TYPE and c.get(A_RESOURCE) == OWL + "Class" for c in el))
        if about and is_class:
            rec = classes.setdefault(about, {"ann": {}, "parents": [], "deprecated": False})
            keep_all = about in keep_all_annotations_for
            for child in el:
                ctag = child.tag
                res = child.get(A_RESOURCE)
                if ctag == T_SUBCLASS:
                    if res:
                        rec["parents"].append(res)
                    else:
                        for sub in child:  # nested named class, or an intersection genus
                            if sub.tag == T_CLASS and sub.get(A_ABOUT):
                                rec["parents"].append(sub.get(A_ABOUT))
                elif ctag == T_EQUIV:
                    for sub in child:
                        for inter in sub:
                            if inter.tag == T_INTERSECTION:
                                rec["parents"].extend(_named_members(inter))
                elif ctag == T_DEPRECATED:
                    rec["deprecated"] = (child.text or "").strip().lower() == "true"
                elif res is None and len(child) == 0 and child.text and child.text.strip():
                    lang = child.get(XML_LANG)
                    if lang and not lang.lower().startswith("en"):
                        continue
                    prop = _tag_iri(ctag)
                    if keep_all or prop in label_props:
                        rec["ann"].setdefault(prop, []).append(child.text.strip())
        root.clear()
    return classes, annprop_labels


def split_fsn(fsn: str) -> tuple[str, str]:
    """'<term> (body structure)' -> ('<term>', 'body structure')."""
    m = _SEMANTIC_TAG.match(" ".join(fsn.split()))
    return (m.group(1), m.group(2)) if m else (fsn, "")


def snomed_canonicalise(classes: dict) -> dict:
    """Order each property's values by dialect (en-us, en, en-gb; stable otherwise), derive the
    FSN term (semantic tag removed) and keep the semantic tag for analysis. In place."""
    for rec in classes.values():
        ann, langs = rec["ann"], rec.pop("ann_lang", {})
        for prop, vals in ann.items():
            ranks = [DIALECT_RANK.get(lang.lower(), 3) for lang in langs.get(prop, [""] * len(vals))]
            ann[prop] = [v for _r, _i, v in sorted(zip(ranks, range(len(vals)), vals))]
        fsns = ann.get(RDFS + "label", [])
        if fsns:
            term, tag = split_fsn(fsns[0])
            ann[SNOMED_FSN_TERM] = [term]
            rec["semantic_tag"] = tag
    return classes


_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")


def local_name_fallback(iri: str) -> str:
    """Decode an IRI local name into words. FALLBACK ONLY, used when a class has no label."""
    local = re.split(r"[#/]", iri.rstrip("/#"))[-1]
    local = urllib.parse.unquote(local)
    return _CAMEL.sub(" ", local.replace("_", " ")).strip()


def _dedup(values, exclude=()):
    seen = {v.lower() for v in exclude}
    out = []
    for v in values:
        v = " ".join(v.split())
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


def build_entities(ontology: str, classes: dict, needed: set[str], max_related: int = 12) -> dict:
    """Assign roles and attach one-hop parent/child labels for the task entities."""
    roles = roles_for(ontology)

    def first_label(iri: str) -> str | None:
        ann = classes.get(iri, {}).get("ann", {})
        for prop in roles["label"]:
            if ann.get(prop):
                return " ".join(ann[prop][0].split())
        return None

    children: dict[str, list[str]] = collections.defaultdict(list)
    for iri, rec in classes.items():
        for parent in rec["parents"]:
            children[parent].append(iri)

    def related_labels(iris: list[str]) -> list[str]:
        labels = [first_label(i) for i in dict.fromkeys(iris)]
        return _dedup([lab for lab in labels if lab])[:max_related]

    entities = {}
    for iri in sorted(needed):
        rec = classes.get(iri)
        if rec is None:
            entities[iri] = {"iri": iri, "found": False, "label": local_name_fallback(iri),
                             "label_is_fallback": True, "synonyms": [], "related_synonyms": [],
                             "definition": "", "parents": [], "children": [], "deprecated": False}
            continue
        ann = rec["ann"]
        labels = _dedup([v for p in roles["label"] for v in ann.get(p, [])])
        label = labels[0] if labels else None
        syn = _dedup(labels[1:] + [v for p in roles["syn_exact"] for v in ann.get(p, [])],
                     exclude=[label] if label else [])
        rel = _dedup([v for p in roles["syn_related"] for v in ann.get(p, [])],
                     exclude=([label] if label else []) + syn)
        definition = next((" ".join(v.split()) for p in roles["definition"]
                           for v in ann.get(p, [])), "")
        entities[iri] = {
            "iri": iri, "found": True,
            "label": label or local_name_fallback(iri), "label_is_fallback": label is None,
            "synonyms": syn, "related_synonyms": rel, "definition": definition,
            "parents": related_labels(rec["parents"]),
            "children": related_labels(sorted(children.get(iri, []))),
            "deprecated": rec["deprecated"],
        }
        if "semantic_tag" in rec:  # SNOMED only; for analysis, never put in a pair state
            entities[iri]["semantic_tag"] = rec["semantic_tag"]
        if rec.get("entity_type", "Class") != "Class":  # SNOMED attribute declared as a property
            entities[iri]["entity_type"] = rec["entity_type"]
    return entities


def coverage(entities: dict) -> dict:
    n = max(1, len(entities))
    ents = list(entities.values())
    pct = lambda k: round(100.0 * k / n, 2)  # noqa: E731
    return {
        "entities": len(entities),
        "found_in_ontology_pct": pct(sum(e["found"] for e in ents)),
        "label_pct": pct(sum(not e["label_is_fallback"] for e in ents)),
        "synonyms_pct": pct(sum(bool(e["synonyms"] or e["related_synonyms"]) for e in ents)),
        "definition_pct": pct(sum(bool(e["definition"]) for e in ents)),
        "parent_context_pct": pct(sum(bool(e["parents"]) for e in ents)),
        "child_context_pct": pct(sum(bool(e["children"]) for e in ents)),
        "deprecated_pct": pct(sum(e["deprecated"] for e in ents)),
    }


def census(classes: dict, needed: set[str], annprop_labels: dict, top: int = 40) -> list[tuple]:
    """How many task entities carry each literal annotation property (for role discovery)."""
    counts: collections.Counter = collections.Counter()
    example: dict[str, str] = {}
    for iri in needed:
        for prop, vals in classes.get(iri, {}).get("ann", {}).items():
            counts[prop] += 1
            example.setdefault(prop, vals[0][:70])
    return [(p, annprop_labels.get(p, ""), c, example[p]) for p, c in counts.most_common(top)]


def cache_file(ontology: str) -> Path:
    return ENTITY_CACHE / f"{ontology}.entities.json.gz"


def load_entities(ontology: str) -> dict:
    path = cache_file(ontology)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing - run `python src/ontology_loader.py --ontology {ontology}` first"
            + (" (SNOMED CT is licence-gated: put your authorised OWL in data/private/ or set "
               "$SNOMED_OWL)" if ontology == "SNOMED" else ""))
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)["entities"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", required=True, choices=sorted(IRI_PREFIX))
    ap.add_argument("--discover", action="store_true", help="print the annotation-property census")
    args = ap.parse_args()

    path = ontology_path(args.ontology)
    if path is None:
        print(f"[{args.ontology}] ontology file NOT AVAILABLE."
              + (" SNOMED CT is licence-gated; supply an authorised OWL via $SNOMED_OWL or "
                 "data/private/*snomed*.owl." if args.ontology == "SNOMED" else ""))
        return 2
    needed = task_entities(args.ontology)
    print(f"[{args.ontology}] file={path.name}  task entities={len(needed)}")
    parse_stats = None
    with Timer() as t:
        if is_ofn(path):  # SNOMED CT: OWL Functional Syntax (see src/ofn_loader.py)
            keep = set(roles_for(args.ontology)["label"]) | {RDFS + "label"}
            classes, parse_stats = parse_ofn(path, needed, keep)
            if args.ontology == "SNOMED":
                snomed_canonicalise(classes)
            annprop_labels = {}
        else:
            classes, annprop_labels = parse_owl(path, needed)
    n_classes = sum(1 for r in classes.values() if r.get("entity_type", "Class") == "Class")
    print(f"[{args.ontology}] parsed {n_classes} named classes in {t.seconds:.1f}s"
          + (f"  {json.dumps(parse_stats)}" if parse_stats else ""))
    if args.discover:
        print(f"[{args.ontology}] literal annotation properties on task entities "
              "(property | declared label | #entities | example):")
        for prop, lab, cnt, ex in census(classes, needed, annprop_labels):
            print(f"   {cnt:7d}  {prop}  [{lab}]  e.g. {ex!r}")
    entities = build_entities(args.ontology, classes, needed)
    cov = coverage(entities)
    print(f"[{args.ontology}] coverage: {json.dumps(cov)}")
    meta = {"ontology": args.ontology, "file": path.name, "sha256": sha256(path),
            "named_classes": n_classes, "roles": roles_for(args.ontology), "coverage": cov}
    if parse_stats:
        meta["parse_stats"] = parse_stats
    ENTITY_CACHE.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_file(args.ontology), "wt", encoding="utf-8") as fh:
        json.dump({"meta": meta, "entities": entities}, fh)
    with open(ENTITY_CACHE / f"{args.ontology}.meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[{args.ontology}] cached -> {cache_file(args.ontology)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

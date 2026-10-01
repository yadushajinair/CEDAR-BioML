"""Verify the SNOMED CT entity extraction against the second, independent serialisation.

The authorised repository ships the same ontology as OWL Functional Syntax (parsed for the
project by src/ofn_loader.py) and as RDF/XML. This script re-derives, from the RDF/XML file and
with its own code, every literal annotation and every named parent of each SNOMED task entity,
then compares them exactly with the OFN parse:

  * RDF/XML class expressions are flattened into blank nodes (rdf:nodeID) whose triples are split
    over several top-level blocks, so they are resolved here through an explicit
    subClassOf/equivalentClass -> intersectionOf -> rdf:first/rdf:rest table;
  * annotations are compared as multisets of (property, language tag, value);
  * parents are compared as sets (same rule as the loader: named superclass, or named top-level
    member of an ObjectIntersectionOf).

It also checks that every SNOMED IRI in the six SNOMED candidate files is a declared class, that
every FMA/NCIT candidate of the SNOMED pairs is in its entity cache, and prints a few worked
examples from the cache the models will read.

  srun -p <cpu-partition> -c 2 --mem=24G -t 60 .venv/bin/python src/verify_snomed.py \
       --rdfxml /path/to/<your licensed release>.rdfxml.owl
"""
from __future__ import annotations

import argparse
import collections
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from ofn_loader import parse_ofn
from ontology_loader import RDFS, SKOS, load_entities, task_entities
from utils import OUTPUTS, PAIRS, SPLITS, Timer, dump_json, ontology_path, read_cands

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
OWL = "http://www.w3.org/2002/07/owl#"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
A_ABOUT, A_NODE, A_RES = "{%s}about" % RDF, "{%s}nodeID" % RDF, "{%s}resource" % RDF
T_TYPE, T_FIRST, T_REST = "{%s}type" % RDF, "{%s}first" % RDF, "{%s}rest" % RDF
T_SUB, T_EQ, T_INTER = "{%s}subClassOf" % RDFS, "{%s}equivalentClass" % OWL, "{%s}intersectionOf" % OWL
T_SUBPROP = "{%s}subPropertyOf" % RDFS
ENTITY_TYPES = {OWL + "Class": "Class", OWL + "ObjectProperty": "ObjectProperty",
                OWL + "DatatypeProperty": "DataProperty", OWL + "AnnotationProperty": "AnnotationProperty"}
NIL = RDF + "nil"
ANN_PROPS = {RDFS + "label", SKOS + "prefLabel", SKOS + "altLabel", SKOS + "definition"}


def parse_rdfxml(path: Path, task: set[str]):
    """Stream the RDF/XML file; keep only what parent resolution and the task annotations need."""
    etype: dict[str, str] = {}
    sup: dict[str, list] = collections.defaultdict(list)   # named class -> [("iri"|"node", x)]
    inter: dict[str, str] = {}                               # blank node -> list head node
    first: dict[str, tuple] = {}                             # list cell -> ("iri"|"node", x)
    rest: dict[str, str | None] = {}                         # list cell -> next cell (None = nil)
    ann: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    other_ann_props: collections.Counter = collections.Counter()
    depth, root = 0, None
    for event, el in ET.iterparse(str(path), events=("start", "end")):
        if event == "start":
            depth += 1
            if depth == 1:
                root = el
            continue
        depth -= 1
        if depth != 1:
            continue
        about, node = el.get(A_ABOUT), el.get(A_NODE)
        for ch in el:
            tag, res, nid = ch.tag, ch.get(A_RES), ch.get(A_NODE)
            obj = ("iri", res) if res else (("node", nid) if nid else None)
            if about:
                if tag == T_TYPE and res in ENTITY_TYPES:
                    etype[about] = ENTITY_TYPES[res]
                elif tag in (T_SUB, T_EQ, T_SUBPROP) and obj:
                    sup[about].append((tag == T_EQ, obj))
                elif obj is None and ch.text is not None and about in task:
                    prop = tag[1:].replace("}", "", 1)
                    lang = ch.get(XML_LANG) or ""
                    if prop in ANN_PROPS:
                        if lang.lower().startswith("en") or not lang:
                            ann[about][(prop, lang, ch.text.strip())] += 1
                    else:
                        other_ann_props[prop] += 1
            elif node:
                if tag == T_INTER and nid:
                    inter[node] = nid
                elif tag == T_FIRST and obj:
                    first[node] = obj
                elif tag == T_REST:
                    rest[node] = None if res == NIL else nid
        root.clear()

    def genus(obj, via_equivalence: bool) -> list[str]:
        kind, x = obj
        if kind == "iri":
            # owl:equivalentClass to a bare named class never occurs in this release; the OFN
            # loader would not treat it as a parent either
            return [] if via_equivalence else [x]
        head = inter.get(x)
        out, cell, guard = [], head, 0
        while cell is not None:
            f = first.get(cell)
            if f and f[0] == "iri":
                out.append(f[1])
            cell = rest.get(cell)
            guard += 1
            if guard > 10_000:
                raise RuntimeError(f"cyclic rdf:List at {head}")
        return out

    parents = {iri: sorted({p for eq, obj in sup.get(iri, []) for p in genus(obj, eq)})
               for iri in task}
    return etype, parents, ann, other_ann_props


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdfxml", required=True)
    ap.add_argument("--examples", nargs="*", default=[
        "http://snomed.info/id/80891009", "http://snomed.info/id/22298006",
        "http://snomed.info/id/10042008", "http://snomed.info/id/1000311000202102",
        "http://snomed.info/id/272741003"])
    args = ap.parse_args()

    ofn_path = ontology_path("SNOMED")
    task = task_entities("SNOMED")
    report: dict = {"ofn_file": ofn_path.name, "rdfxml_file": Path(args.rdfxml).name,
                    "task_entities": len(task)}
    with Timer() as t:
        ofn_classes, ofn_stats = parse_ofn(ofn_path, task, set())
    report["ofn_parse_s"], report["ofn_stats"] = round(t.seconds, 1), ofn_stats
    print(f"[verify] OFN parsed in {t.seconds:.0f}s: {json.dumps(ofn_stats)}", flush=True)
    with Timer() as t:
        rx_types, rx_parents, rx_ann, other_props = parse_rdfxml(Path(args.rdfxml), task)
    report["rdfxml_parse_s"] = round(t.seconds, 1)
    report["rdfxml_entities_by_type"] = dict(collections.Counter(rx_types.values()))
    rx_classes = set(rx_types)
    report["rdfxml_other_literal_props_on_task_entities"] = dict(other_props)
    print(f"[verify] RDF/XML parsed in {t.seconds:.0f}s: {report['rdfxml_entities_by_type']}",
          flush=True)

    ofn_types = {iri: r["entity_type"] for iri, r in ofn_classes.items()}
    ofn_declared = set(ofn_types)
    report["ofn_entities_by_type"] = dict(collections.Counter(ofn_types.values()))
    report["entity_types_identical"] = ofn_types == rx_types
    report["task_entities_by_type"] = dict(collections.Counter(ofn_types.get(i, "undeclared") for i in task))
    missing_ofn = sorted(task - ofn_declared)
    missing_rx = sorted(task - rx_classes)
    ann_mismatch, parent_mismatch, no_parent = [], [], []
    for iri in sorted(task):
        rec = ofn_classes.get(iri, {"ann": {}, "ann_lang": {}, "parents": []})
        o_ann = collections.Counter(
            (p, lang, v) for p, vals in rec["ann"].items()
            for lang, v in zip(rec["ann_lang"][p], vals))
        if o_ann != rx_ann.get(iri, collections.Counter()):
            ann_mismatch.append(iri)
        if sorted(set(rec["parents"])) != rx_parents[iri]:
            parent_mismatch.append(iri)
        if not rec["parents"]:
            no_parent.append(iri)
    report.update({
        "entities_declared_both": not (ofn_declared ^ rx_classes),
        "task_entities_not_declared_ofn": missing_ofn[:20], "n_task_not_declared_ofn": len(missing_ofn),
        "task_entities_not_declared_rdfxml": missing_rx[:20], "n_task_not_declared_rdfxml": len(missing_rx),
        "annotation_mismatches": len(ann_mismatch), "annotation_mismatch_examples": ann_mismatch[:10],
        "parent_mismatches": len(parent_mismatch), "parent_mismatch_examples": parent_mismatch[:10],
        "task_entities_without_parent": len(no_parent), "no_parent_examples": no_parent[:10],
    })
    for iri in ann_mismatch[:3]:
        rec = ofn_classes.get(iri, {"ann": {}, "ann_lang": {}})
        print(f"[verify] ANN MISMATCH {iri}\n  ofn={rec['ann']}\n  rdfxml={dict(rx_ann.get(iri, {}))}")
    for iri in parent_mismatch[:3]:
        print(f"[verify] PARENT MISMATCH {iri}\n  ofn={ofn_classes.get(iri, {}).get('parents')}\n"
              f"  rdfxml={rx_parents[iri]}")

    # every IRI in the SNOMED pairs' candidate files, against the caches the models read
    snomed, cache = load_entities("SNOMED"), {}
    pair_checks = {}
    for pair, info in PAIRS.items():
        if info["src"] != "SNOMED":
            continue
        tgt = cache.setdefault(info["tgt"], load_entities(info["tgt"]))
        for split in SPLITS:
            qs = read_cands(pair, split)
            srcs = {q["src"] for q in qs}
            cands = {c for q in qs for c in q["cands"]} | {q["gold"] for q in qs if q["gold"]}
            pair_checks[f"{pair}/{split}"] = {
                "queries": len(qs), "sources": len(srcs),
                "sources_not_in_cache": sum(s not in snomed for s in srcs),
                "sources_not_found_in_ontology": sum(not snomed[s]["found"] for s in srcs if s in snomed),
                "sources_with_fallback_label": sum(snomed[s]["label_is_fallback"] for s in srcs if s in snomed),
                "targets": len(cands),
                "targets_not_in_cache": sum(c not in tgt for c in cands),
                "targets_not_found_in_ontology": sum(not tgt[c]["found"] for c in cands if c in tgt),
            }
    report["pair_checks"] = pair_checks
    report["semantic_tags_of_sources"] = {
        pair: dict(collections.Counter(snomed[q["src"]].get("semantic_tag", "")
                                       for s in SPLITS for q in read_cands(pair, s)).most_common(12))
        for pair in ("SNOMED-FMA", "SNOMED-NCIT")}
    report["examples"] = {iri: snomed.get(iri) for iri in args.examples}
    for iri in args.examples:
        print(f"[verify] example {iri}: {json.dumps(snomed.get(iri), ensure_ascii=False)}")

    ok = (report["entities_declared_both"] and report["entity_types_identical"]
          and not missing_ofn and not missing_rx
          and not ann_mismatch and not parent_mismatch
          and all(v["sources_not_in_cache"] == 0 and v["sources_not_found_in_ontology"] == 0
                  and v["targets_not_in_cache"] == 0 and v["targets_not_found_in_ontology"] == 0
                  for v in pair_checks.values()))
    report["ALL_CHECKS_PASS"] = ok
    out = OUTPUTS / "snomed_verification.json"
    dump_json(out, report)
    print(f"[verify] {'ALL CHECKS PASS' if ok else 'CHECKS FAILED'} -> {out}")
    print(json.dumps({k: v for k, v in report.items() if k not in ("examples",)}, indent=1)[:6000])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

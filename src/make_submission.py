"""Assemble the CodaBench submission ZIP from validated TEST rankings. Never uploads anything.

For every pair it re-runs the official validate_ranking.py plus stricter local checks (query
count and order, exactly 100 candidates, exact permutation of the pool, full IRIs, no
duplicates, no NaN / malformed cells). Only rankings that pass are packaged, under the
hyphenated slug filenames the scorer matches (ncit-doid.tsv, snomed-fma.tsv, snomed-ncit.tsv).
The ZIP is flat and contains nothing else (no __MACOSX, no directories).

A pair with no ranking (e.g. SNOMED pairs without a licensed SNOMED CT file) is reported and
left out; the ZIP is then named *_PARTIAL_* so it cannot be mistaken for a complete submission.

  python src/make_submission.py --run ens_lexical+laya_bioml_neg5_D
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import shutil
import subprocess
import sys
import zipfile

from utils import CANDIDATE_COUNT, OUTPUTS, PAIRS, SCORING_KIT, cands_path, dump_json, read_cands


def strict_check(pair: str, path) -> list[str]:
    problems = []
    pool = read_cands(pair, "test")
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        rows = list(reader)
    if header != ["SrcEntity", "TgtCandidates"]:
        problems.append(f"unexpected header {header}")
    if len(rows) != len(pool):
        problems.append(f"{len(rows)} rows for {len(pool)} pool queries")
    for i, (row, q) in enumerate(zip(rows, pool)):
        if len(row) != 2:
            problems.append(f"row {i}: {len(row)} cells")
            continue
        if row[0] != q["src"]:
            problems.append(f"row {i}: source order mismatch")
        try:
            ranked = ast.literal_eval(row[1])
        except (ValueError, SyntaxError):
            problems.append(f"row {i}: TgtCandidates is not a list literal")
            continue
        if len(ranked) != CANDIDATE_COUNT or len(set(ranked)) != CANDIDATE_COUNT:
            problems.append(f"row {i}: not {CANDIDATE_COUNT} distinct candidates")
        if set(ranked) != set(q["cands"]):
            problems.append(f"row {i}: not a permutation of the pool")
        if any((not isinstance(c, str)) or not c.startswith("http") or c.lower() == "nan"
               for c in ranked):
            problems.append(f"row {i}: a candidate is not a full IRI")
    return problems[:20]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory name under outputs/<pair>/test/")
    ap.add_argument("--name", default=None, help="zip basename (default: derived from --run)")
    args = ap.parse_args()

    sub_dir = OUTPUTS / "submission" / args.run
    if sub_dir.exists():
        shutil.rmtree(sub_dir)
    sub_dir.mkdir(parents=True)
    included, missing, manifest = [], [], {}
    for pair, info in PAIRS.items():
        ranking = OUTPUTS / pair / "test" / args.run / "ranking.tsv"
        if not ranking.is_file():
            missing.append(pair)
            print(f"[submission] {pair}: NO test ranking at {ranking} -> not included")
            continue
        val = subprocess.run([sys.executable, str(SCORING_KIT / "validate_ranking.py"),
                              str(cands_path(pair, "test")), str(ranking)],
                             capture_output=True, text=True)
        problems = strict_check(pair, ranking)
        print(f"[submission] {pair}: validate_ranking.py -> {val.stdout.strip()}; strict checks -> "
              f"{'OK' if not problems else problems}")
        if val.returncode != 0 or problems:
            raise SystemExit(f"{pair}: ranking failed validation; refusing to package")
        dest = sub_dir / f"{info['slug']}.tsv"
        shutil.copyfile(ranking, dest)
        manifest[dest.name] = {"pair": pair, "source": str(ranking.relative_to(OUTPUTS.parent)),
                               "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                               "queries": len(read_cands(pair, "test"))}
        included.append(dest)
    if not included:
        raise SystemExit("no valid test rankings to package")

    tag = "" if not missing else "_PARTIAL_" + "+".join(PAIRS[p]["slug"] for p in PAIRS if p not in missing) + "-only"
    zip_path = OUTPUTS / "submission" / f"{args.name or 'submission_' + args.run}{tag}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in included:
            zf.write(path, arcname=path.name)  # flat, like `zip -X -j`
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert sorted(names) == sorted(p.name for p in included), names
    dump_json(OUTPUTS / "submission" / f"{zip_path.stem}.manifest.json",
              {"zip": zip_path.name, "files": manifest, "missing_pairs": missing,
               "complete": not missing, "run": args.run})
    print(f"[submission] wrote {zip_path} containing {names}")
    if missing:
        print(f"[submission] INCOMPLETE: no ranking for {missing}. This ZIP is NOT a full "
              "three-pair submission.")
    print("[submission] nothing was uploaded; submit to CodaBench manually if you choose to.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

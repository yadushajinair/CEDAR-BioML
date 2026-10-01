"""Freeze the final system of each pair exactly as PROTOCOL.md prescribes -- from TRAIN-DEV only.

For every training seed s in {0, 1, 2} of a pair:
    LAYA-BioML checkpoint   checkpoints/<pair>_repD_neg5[_seed<s>]/best   (epoch chosen on train-dev)
    ensemble (mode, alpha)  outputs/<pair>/train-dev/ens_lexical+laya_bioml_neg5[_seed<s>]_D/ensemble_choice.json
    train-dev MRR           official score_local.py output of that ensemble on train-dev
Seeds whose training did not complete (stopped by the driver's numerical guard; PROTOCOL.md
amendment 1) are reported as diverged and excluded; a completed seed with a missing downstream file
is a hard error. The selected seed is the one with the highest train-dev MRR (ties -> lower seed). Nothing from
VALID or TEST is read. The result is written to outputs/<pair>/final_selection.json, which the
VALID report and the TEST run then consume unchanged.

  python src/select_final.py                     # all pairs
  python src/select_final.py --pair SNOMED-FMA
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils import OUTPUTS, PAIRS, ROOT, dump_json

SEEDS = (0, 1, 2)
FINAL_RUN = "final_lexical+laya_bioml_neg5_D@devalpha"


def seed_names(pair: str, seed: int) -> dict:
    sfx = "" if seed == 0 else f"_seed{seed}"
    laya_run = f"laya_bioml_neg5{sfx}_D"
    return {"laya_run": laya_run,
            "checkpoint": f"checkpoints/{pair.lower()}_repD_neg5{sfx}/best",
            "train_config": f"checkpoints/{pair.lower()}_repD_neg5{sfx}/train_config.json",
            "dev_ensemble_dir": f"outputs/{pair}/train-dev/ens_lexical+{laya_run}",
            "dev_laya_dir": f"outputs/{pair}/train-dev/{laya_run}"}


def training_completed(n: dict) -> tuple[bool, str]:
    cfg_path = ROOT / n["train_config"]
    if not cfg_path.is_file():
        return False, "no train_config.json (training stopped before the first epoch ended)"
    cfg = json.loads(cfg_path.read_text())
    done, planned = len(cfg.get("epochs", [])), cfg["args"]["epochs"]
    if done < planned:
        return False, f"training stopped after {done}/{planned} epochs"
    return True, f"{done}/{planned} epochs"


def select(pair: str) -> dict:
    per_seed, diverged = {}, {}
    for s in SEEDS:
        n = seed_names(pair, s)
        ok, why = training_completed(n)
        if not ok:
            diverged[s] = why
            print(f"[select] {pair} seed {s}: EXCLUDED, {why}")
            continue
        ens = ROOT / n["dev_ensemble_dir"]
        needed = [ROOT / n["checkpoint"], ens / "ensemble_choice.json", ens / "metrics.official.json",
                  ROOT / n["dev_laya_dir"] / "metrics.official.json", ROOT / n["train_config"]]
        missing = [str(p.relative_to(ROOT)) for p in needed if not p.exists()]
        if missing:
            raise SystemExit(f"{pair} seed {s}: missing {missing}")
        choice = json.loads((ens / "ensemble_choice.json").read_text())
        assert choice["tuned_on"] == f"{pair}/train-dev", choice
        cfg = json.loads((ROOT / n["train_config"]).read_text())
        per_seed[s] = {**n, "choice": choice,
                       "train_dev_ensemble": json.loads((ens / "metrics.official.json").read_text()),
                       "train_dev_laya_only": json.loads(
                           (ROOT / n["dev_laya_dir"] / "metrics.official.json").read_text()),
                       "best_epoch": cfg["best_epoch"],
                       "epoch_selection_train_dev_mrr": cfg["best_train_dev_mrr"]}
    if not per_seed:
        raise SystemExit(f"{pair}: no training seed completed")
    best = max(per_seed, key=lambda s: (per_seed[s]["train_dev_ensemble"]["mrr"], -s))
    sel = per_seed[best]
    out = {"pair": pair, "protocol": "PROTOCOL.md", "rule": "max train-dev ensemble MRR over "
           "training seeds {0,1,2}; ties -> lower seed; VALID/TEST not read",
           "selected_seed": best, "laya_run": sel["laya_run"], "checkpoint": sel["checkpoint"],
           "choice_path": f"{sel['dev_ensemble_dir']}/ensemble_choice.json", "choice": sel["choice"],
           "final_run_name": FINAL_RUN,
           "seeds_completed": sorted(per_seed), "seeds_diverged": {str(k): v for k, v in diverged.items()},
           "per_seed": {str(s): {"train_dev_ensemble_mrr": v["train_dev_ensemble"]["mrr"],
                                 "train_dev_laya_only_mrr": v["train_dev_laya_only"]["mrr"],
                                 "ensemble": {"mode": v["choice"]["mode"], "alpha": v["choice"]["alpha"]},
                                 "best_epoch": v["best_epoch"],
                                 "epoch_selection_train_dev_mrr": v["epoch_selection_train_dev_mrr"]}
                         for s, v in per_seed.items()}}
    dump_json(OUTPUTS / pair / "final_selection.json", out)
    print(f"[select] {pair}: seed {best} ({sel['laya_run']}, {sel['choice']['mode']} "
          f"alpha={sel['choice']['alpha']}); train-dev ensemble MRR by seed: "
          + ", ".join(f"{s}={v['train_dev_ensemble']['mrr']:.4f}" for s, v in per_seed.items()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=sorted(PAIRS))
    args = ap.parse_args()
    for pair in ([args.pair] if args.pair else PAIRS):
        select(pair)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

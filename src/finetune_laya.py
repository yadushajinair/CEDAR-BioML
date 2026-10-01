"""Fine-tune LAYA for ontology equivalence ("LAYA-BioML") with LAYA's own training recipe.

This is a thin driver around the OFFICIAL single-device fine-tuning script shipped in the LAYA
repository (external/laya/research/scripts/finetune_single_device.py). Its item builder, collate,
forward, and temperature fit are imported and used unchanged, and the optimisation step below is
that script's RLCD step verbatim: Gaussian logit exploration, proper-scoring-rule reward
(`laya.common.proper_reward`), group-mean-baseline policy gradient, plus the cross-entropy term;
AdamW (encoder 2.5e-5 / head 1e-4, wd 0.01), cosine schedule, grad-clip 1.0, fp16 autocast with
gradient checkpointing. The model is LAYA's DecisionModel (`laya.common.build_model`) initialised
from the released checkpoint, and the saved directory has the released checkpoint's layout, so
it loads with `laya.load(<dir>)`.

What this driver changes relative to the official script, and why:
  * base checkpoint = English repo root (pinned revision) with ITS OWN max_len/head_max_len
    (512/192); the official script hard-codes the multilingual 1024/256;
  * after every epoch the checkpoint is saved and the reserved TRAIN-DEV queries are ranked over
    their full 100-candidate pools with the real inference path; the best epoch by train-dev MRR
    is kept as `best/` (VALID and TEST are never touched here);
  * a non-finite-loss guard, progress logging, and a written training config. A non-finite
    loss is handled exactly as the official script handles it -- the fp16 GradScaler sees the
    non-finite gradients, skips that optimizer step and lowers the loss scale -- but here every
    such step is counted, logged and excluded from the reported average loss, and training
    stops if it is not isolated (>20 in a row or >1% of an epoch's steps) or if any weight
    becomes non-finite. (Until 2026-09-30 the guard raised on the first event; it never fired in
    any NCIT-DOID run, so those runs are unaffected.)

  python src/finetune_laya.py --pair NCIT-DOID --rep D --neg-ratio 5 --run-name ncit-doid_D_neg5
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import laya  # the installed package must be imported before the official script is loaded
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import build_model, proper_reward

from build_finetune_data import data_dir
from build_pair_states import LAYA_REPO, LAYA_REVISION
from utils import PAIRS, ROOT, dump_json, fmt_metrics, rank_by_scores, ranking_metrics, read_cands

OFFICIAL_SCRIPT = ROOT / "external" / "laya" / "research" / "scripts" / "finetune_single_device.py"


def load_official():
    """Import the official fine-tuning script as a module (its helpers are reused unchanged)."""
    before = list(sys.path)
    spec = importlib.util.spec_from_file_location("laya_official_finetune", OFFICIAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path[:] = before  # the script prepends the repo root; keep using the installed `laya`
    return module


def save_checkpoint(model, tok, cfg: dict, out_dir: Path, temperatures: list[float]) -> None:
    """Same files and layout as the official script writes (half-precision safetensors)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()},
              str(out_dir / "model.safetensors"))
    model.encoder.config.save_pretrained(str(out_dir / "encoder"))
    tok.save_pretrained(str(out_dir / "tokenizer"))
    cfg = dict(cfg)
    cfg["fine_tuned"] = True
    cfg["temperature"] = temperatures
    cfg.pop("temperature_by_options", None)
    with open(out_dir / "rl_agent_config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)


def fit_temperatures(official, model, calib_items, pad_id, device, use_amp) -> list[float]:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(calib_items), 16):
            chunk = calib_items[i:i + 16]
            logits, _ = official.forward(model, official.collate(chunk, pad_id), device, use_amp)
            arr = logits.float().cpu().numpy()
            for r, it in enumerate(chunk):
                preds.append((it["qtype"], arr[r, : len(it["markers"])], it["target"]))
    fitted = [1.2, 1.2, 1.2]
    for qt in range(3):
        sel = [(z, t) for q_type, z, t in preds if q_type == qt]
        if sel:
            fitted[qt] = official.fit_one_temp(sel)
    model.train()
    return fitted


def dev_mrr(ckpt_dir: Path, pair: str, rep: str, dev_queries: list[dict], batch_size: int):
    """Rank the train-dev pools with the saved checkpoint through the real inference path."""
    from score_laya import LayaPairScorer

    scorer = LayaPairScorer(str(ckpt_dir), pair, rep, batch_size=batch_size)
    rankings, margins = [], []
    for start in range(0, len(dev_queries), 32):
        chunk = dev_queries[start:start + 32]
        scored = scorer.score_pairs([(q["src"], c) for q in chunk for c in q["cands"]])
        k = 0
        for q in chunk:
            m = [scored[k + j]["LayaLogitMargin"] for j in range(len(q["cands"]))]
            k += len(q["cands"])
            margins.append(m)
            rankings.append(rank_by_scores(q["cands"], m))
    del scorer
    torch.cuda.empty_cache()
    return ranking_metrics(dev_queries, rankings), margins


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=sorted(PAIRS))
    ap.add_argument("--rep", default="D", choices=["A", "B", "C", "D"])
    ap.add_argument("--neg-ratio", type=int, default=5)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--base-model", default=LAYA_REPO)
    ap.add_argument("--epochs", type=int, default=4)           # official default
    ap.add_argument("--micro-batch", type=int, default=8)      # official default
    ap.add_argument("--group-size", type=int, default=4)       # official default
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev-batch-size", type=int, default=16)
    ap.add_argument("--max-dev-queries", type=int, default=400)
    args = ap.parse_args()

    official = load_official()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    use_amp = device.type == "cuda"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    run_dir = ROOT / "checkpoints" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    ddir = data_dir(args.pair, args.rep, args.neg_ratio)
    data_meta = json.loads((ddir / "meta.json").read_text())
    train_queries = read_cands(args.pair, "train")
    dev_queries = [train_queries[i] for i in data_meta["train_dev_query_indices"]]
    dev_queries = dev_queries[: args.max_dev_queries]

    if os.path.isdir(args.base_model):
        model_dir, base_revision = args.base_model, "local"
    else:
        model_dir = snapshot_download(args.base_model, revision=LAYA_REVISION, allow_patterns=[
            "rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])
        base_revision = LAYA_REVISION
    _fix_tokenizer_config(model_dir)
    with open(os.path.join(model_dir, "rl_agent_config.json")) as fh:
        cfg = json.load(fh)
    cfg["gradient_checkpointing"] = use_amp
    cfg["max_tokens_per_batch"] = 4096
    assert cfg["max_len"] == data_meta["max_len"], "states were budgeted for a different max_len"

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    model.float()
    if use_amp:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.to(device)
    model.train()

    all_items = official.preprocess(tok, cfg, str(ddir / "train.jsonl"))
    if not all_items:
        raise SystemExit("no training items")
    order = list(range(len(all_items)))
    random.shuffle(order)
    n_calib = min(400, len(all_items) // 10)
    calib_items = [all_items[i] for i in sorted(order[:n_calib])]
    train_items = [all_items[i] for i in sorted(order[n_calib:])]

    micro_batch, group_size = args.micro_batch, args.group_size
    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": 2.5e-5}, {"params": head_params, "lr": 1.0e-4}],
        weight_decay=0.01)
    steps_per_epoch = max(1, (len(train_items) + micro_batch - 1) // micro_batch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps_per_epoch * args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    git_commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    config = {"args": vars(args), "data": {k: v for k, v in data_meta.items()
                                          if k != "train_dev_query_indices"},
              "laya_version": laya.__version__, "base_model": args.base_model,
              "base_revision": base_revision, "official_script": str(OFFICIAL_SCRIPT.relative_to(ROOT)),
              "torch": torch.__version__, "git_commit": git_commit,
              "optimizer": "AdamW enc 2.5e-5 / head 1e-4, wd 0.01, cosine to 1e-6",
              "loss": "RLCD policy gradient (proper_reward w_sph=0.75 w_rps=1.0, sigma 0.4->0.1)"
                      " + cross-entropy",
              "train_items": len(train_items), "calibration_items": len(calib_items),
              "train_dev_queries": len(dev_queries), "steps_per_epoch": steps_per_epoch,
              "gpu": torch.cuda.get_device_name(0) if use_amp else "cpu", "epochs": []}
    print(f"[ft] {args.run_name}: {len(train_items)} train items ({len(calib_items)} calibration)"
          f" | {args.epochs} epochs | {steps_per_epoch} steps/epoch | dev queries {len(dev_queries)}",
          flush=True)

    best = {"mrr": -1.0, "epoch": None}
    t_start = time.time()
    for epoch in range(args.epochs):
        random.seed(args.seed + epoch)
        random.shuffle(train_items)
        sigma = 0.4 + (0.1 - 0.4) * (epoch / max(1, args.epochs - 1))
        epoch_loss, n_batches, t_epoch = 0.0, 0, time.time()
        n_nonfinite, consecutive = 0, 0
        optimizer.zero_grad(set_to_none=True)
        for b_idx in range(0, len(train_items), micro_batch):
            chunk = train_items[b_idx:b_idx + micro_batch]
            batch = official.collate(chunk, tok.pad_token_id)
            logits, _act = official.forward(model, batch, device, use_amp)
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            # ---- official RLCD step (finetune_single_device.py), unchanged ----
            eps = torch.randn((group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask,
                                  w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss = (-(adv * logp).mean()
                    - (target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean())
            # -------------------------------------------------------------------
            finite = math.isfinite(loss.item())
            if not finite:
                n_nonfinite += 1
                consecutive += 1
                if n_nonfinite <= 5:
                    lg = logits.detach()
                    print(f"[ft]   non-finite loss at epoch {epoch + 1} step {n_batches + 1}: "
                          f"logits finite={bool(torch.isfinite(lg).all())} "
                          f"max|finite logit|={float(lg[torch.isfinite(lg)].abs().max()) if torch.isfinite(lg).any() else float('nan'):.1f} "
                          f"seq lens={[len(it['ids']) if 'ids' in it else None for it in chunk]} "
                          f"loss scale={scaler.get_scale() if scaler else None} -> step skipped by GradScaler",
                          flush=True)
                if scaler is None or consecutive > 20:
                    raise RuntimeError(f"non-finite loss at epoch {epoch + 1} step {n_batches + 1} "
                                       f"({consecutive} in a row)")
            else:
                consecutive = 0
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if finite:
                epoch_loss += loss.item()
            n_batches += 1
            if n_batches % 250 == 0:
                rate = n_batches / (time.time() - t_epoch)
                print(f"[ft]   epoch {epoch + 1} step {n_batches}/{steps_per_epoch} "
                      f"loss {epoch_loss / max(1, n_batches - n_nonfinite):.4f}  {rate:.1f} it/s"
                      f"  non-finite steps {n_nonfinite}", flush=True)

        if n_nonfinite > 0.01 * n_batches:
            raise RuntimeError(f"{n_nonfinite}/{n_batches} non-finite steps in epoch {epoch + 1}")
        if not all(torch.isfinite(p_).all() for p_ in model.parameters()):
            raise RuntimeError(f"non-finite weights after epoch {epoch + 1}")
        train_seconds = time.time() - t_epoch  # gradient steps only (excludes train-dev ranking)
        temps = fit_temperatures(official, model, calib_items, tok.pad_token_id, device, use_amp)
        ckpt = run_dir / f"epoch{epoch + 1}"
        save_checkpoint(model, tok, cfg, ckpt, temps)
        metrics, margins = dev_mrr(ckpt, args.pair, args.rep, dev_queries, args.dev_batch_size)
        model.train()
        rec = {"epoch": epoch + 1, "avg_loss": epoch_loss / max(1, n_batches - n_nonfinite),
               "nonfinite_loss_steps_skipped": n_nonfinite, "sigma": sigma,
               "train_seconds": train_seconds, "epoch_seconds": time.time() - t_epoch,
               "temperatures": temps,
               "train_dev": metrics}
        config["epochs"].append(rec)
        print(f"[ft] epoch {epoch + 1}/{args.epochs} avg loss {rec['avg_loss']:.4f} | "
              f"train {rec['train_seconds']:.0f}s | temps {[round(t, 3) for t in temps]} | "
              f"train-dev {fmt_metrics(metrics)}", flush=True)
        if metrics["mrr"] > best["mrr"]:
            best = {"mrr": metrics["mrr"], "epoch": epoch + 1}
            if (run_dir / "best").exists():
                shutil.rmtree(run_dir / "best")
            shutil.copytree(ckpt, run_dir / "best")
            dump_json(run_dir / "best_train_dev_margins.json", {
                "query_indices": data_meta["train_dev_query_indices"][: len(dev_queries)],
                "margins": margins})
        shutil.rmtree(ckpt)  # keep only best/ (842 MB per checkpoint)
        config.update({"best_epoch": best["epoch"], "best_train_dev_mrr": best["mrr"],
                       "total_seconds": time.time() - t_start,
                       "gpu_max_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if use_amp else 0})
        dump_json(run_dir / "train_config.json", config)

    print(f"[ft] done: best epoch {best['epoch']} (train-dev MRR {best['mrr']:.4f}) -> "
          f"{run_dir / 'best'}  total {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

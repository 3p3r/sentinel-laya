"""RLCD fine-tuning of convaiinnovations/laya on the Sentinel-recipe dataset.

Faithful port of the official notebook (laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
train_ddp.py: GRPO-style policy gradient with proper-scoring-rule rewards + CE guidance,
then per-type temperature calibration on a held-out slice. Launch with:

  torchrun --standalone --nproc_per_node=2 python/train_rlcd.py \
      --items data/train_items.pt --output models/laya-sentinel

Smoke test: add --max-items 200 --epochs 1
"""
import argparse
import datetime
import json
import os
import random
import sys
import time

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from laya.common import build_model, proper_reward
from laya.agent import _fix_tokenizer_config

import contextlib


def _nullctx():
    return contextlib.nullcontext()


def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/train_items.pt")
    ap.add_argument("--output", default="models/laya-sentinel")
    ap.add_argument("--model-id", default="convaiinnovations/laya")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-items", type=int, default=0, help="truncate train items (smoke test)")
    ap.add_argument("--no-ckpt", action="store_true", help="disable all gradient checkpointing")
    ap.add_argument("--pg-timeout-s", type=int, default=1800, help="NCCL collective watchdog timeout")
    ap.add_argument("--single-gpu", action="store_true",
                    help="no DDP wrapper; gloo PG for API compat (NCCL stalls on this host)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from checkpoint_latest (model weights + epoch + batch cursor)")
    ap.add_argument("--ckpt-every", type=int, default=250,
                    help="save a resumable mid-epoch checkpoint every N micro-batches")
    args = ap.parse_args()

    if args.single_gpu:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        dist.init_process_group("gloo", rank=0, world_size=1,
                                timeout=datetime.timedelta(seconds=args.pg_timeout_s))
        rank, world_size, local_rank = 0, 1, 0
    else:
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=args.pg_timeout_s))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # RLCD_MATH_SDP=1 forces the portable MATH SDPA backend (pure PyTorch ops, no
    # fused flash/efficient kernels) — the fused backward kernels hang on sm_86.
    if os.environ.get("RLCD_MATH_SDP"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(args.model_id)
    _fix_tokenizer_config(model_dir)

    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = 4096
    cfg["max_len"] = 512
    cfg["head_max_len"] = 256

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))

    # ModernBERT's sliding-window SDPA backward is broken on this stack (illegal
    # memory access / cublas faults, non-deterministic). At 512 max_len a full-
    # attention pass over every layer is cheap and uses the stable global-attention
    # kernel, so flip all sliding_attention layers to full_attention. Verified:
    # 3000-item epoch completes cleanly where the sliding-window path crashes.
    # Set RLCD_SLIDING=1 to keep the original sliding-window behaviour.
    if not os.environ.get("RLCD_SLIDING"):
        from transformers import AutoConfig, AutoModel
        import laya.common as _lc

        enc_dir = os.path.join(model_dir, "encoder")
        ecfg = AutoConfig.from_pretrained(enc_dir)
        ecfg.layer_types = ["full_attention"] * len(ecfg.layer_types)
        _lc._apply_rope_config(ecfg)
        enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
        model = _lc.DecisionModel(enc, cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1)
    else:
        model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))

    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)

    if not args.no_ckpt:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # head checkpointing left off: its reentrant recompute desyncs DDP reducers on torch 2.14
    # (verified: ranks enqueue different collective counts -> NCCL watchdog timeout)
    model.to(device)
    model.train()

    # no unused params exist (loss includes 0.0 * act.sum() to keep the act head in-graph),
    # and find_unused_parameters=True can itself desync buckets under activation checkpointing
    if args.single_gpu:
        ddp_model = model
    else:
        ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    all_items = torch.load(args.items, weights_only=False)
    skip = int(os.environ.get("RLCD_SKIP", "0"))
    if skip:
        all_items = all_items[skip:]
    if args.max_items:
        all_items = all_items[: args.max_items]

    # Hold the calibration slice out of training before sharding (notebook protocol).
    CALIB_MAX = 400
    order = list(range(len(all_items)))
    random.Random(20260922).shuffle(order)
    n_calib = min(CALIB_MAX, len(all_items) // 10)
    calib_items = [all_items[i] for i in sorted(order[:n_calib])]
    train_items = [all_items[i] for i in sorted(order[n_calib:])]
    my_items = train_items[rank::world_size]

    EPOCHS = args.epochs
    MICRO_BATCH = 8
    GRAD_ACCUM = 4
    GROUP_SIZE = 4
    LR_ENCODER = 2.5e-5
    LR_HEAD = 1.0e-4
    SIGMA_START = 0.4
    SIGMA_END = 0.1

    enc_params = [p for n, p in ddp_model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in ddp_model.named_parameters() if "encoder." not in n]

    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": LR_ENCODER}, {"params": head_params, "lr": LR_HEAD}],
        weight_decay=0.01,
    )

    total_updates = (len(my_items) // (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=not os.environ.get("RLCD_NO_AMP"))

    # Resume from a mid-epoch checkpoint if requested (model + optimizer + cursor).
    start_epoch, start_bidx = 0, 0
    resume_path = os.path.join(args.output, "checkpoint_latest", "resume.pt")
    if args.resume and os.path.exists(resume_path):
        rs = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(rs["model"])
        optimizer.load_state_dict(rs["optimizer"])
        scheduler.load_state_dict(rs["scheduler"])
        scaler.load_state_dict(rs["scaler"])
        start_epoch, start_bidx = rs["epoch"], rs["b_idx"]
        print(f"Resumed from {resume_path}: epoch={start_epoch} b_idx={start_bidx}", flush=True)

    if rank == 0:
        print(
            f"Starting DDP training: {len(train_items)} train items "
            f"({len(calib_items)} held out for calibration) | {len(my_items)} per rank | "
            f"{EPOCHS} epochs | world_size={world_size}",
            flush=True,
        )
    t0 = time.time()
    nan_skips = 0

    for epoch in range(start_epoch, EPOCHS):
        random.seed(42 + epoch + rank)
        random.shuffle(my_items)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0

        progress = epoch / max(1, EPOCHS - 1)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * progress

        for b_idx in range(0, len(my_items), MICRO_BATCH):
            # skip already-completed micro-batches when resuming mid-epoch
            if epoch == start_epoch and b_idx < start_bidx:
                continue
            chunk = my_items[b_idx : b_idx + MICRO_BATCH]
            if not chunk:
                continue

            batch = collate_train_batch(chunk, tok.pad_token_id)

            # bf16 autocast: fp16 triggers an illegal-memory-access in ModernBERT's
            # backward for long sequences on this stack; bf16 is stable
            use_amp = not os.environ.get("RLCD_NO_AMP")
            amp_dtype = torch.float16 if os.environ.get("RLCD_FP16") else torch.bfloat16
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits, act = ddp_model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )

            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            # fp16 min is -65504, so -1e4 under autocast becomes -inf; use finfo.min
            # (finite) to avoid -inf - -inf = NaN in the logp reconstruction below
            NEG = torch.finfo(torch.float32).min

            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, NEG), -1)

            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, NEG), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()

            # bf16 autocast rarely overflows to a non-finite loss on some batches;
            # skip the backward for those so a single bad batch can't corrupt the run
            if not torch.isfinite(loss):
                nan_skips += 1
                optimizer.zero_grad(set_to_none=True)
                accum_step = 0
                continue

            scaler.scale(loss).backward()
            accum_step += 1

            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(my_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1

            # NOTE: no torch.cuda.empty_cache() here. Calling it mid-run deadlocks the
            # SDPA backward kernel on sm_86 (hard hang, GPU pinned at 100%) — verified
            # on torch 2.5.1+cu124 and 2.14+cu130. The caching allocator is left alone;
            # at 512 max_len fragmentation never approaches OOM on 24GB.

            # periodic resumable checkpoint so a crash loses at most ckpt_every batches
            if rank == 0 and args.ckpt_every and (n_batches % args.ckpt_every) == 0:
                ckpt_dir = os.path.join(args.output, "checkpoint_latest")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch,
                        "b_idx": b_idx + MICRO_BATCH,
                    },
                    os.path.join(ckpt_dir, "resume.pt"),
                )

            if rank == 0 and (n_batches % 50) == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch + 1}/{EPOCHS} | Step {n_batches} | "
                    f"Loss: {loss.item() * GRAD_ACCUM:.4f} | Reward: {r.mean().item():.3f} | "
                    f"LR: {cur_lr:.2e} | elapsed {time.time() - t0:.0f}s",
                    flush=True,
                )

        if rank == 0:
            print(
                f"=== Epoch {epoch + 1}/{EPOCHS} done in {time.time() - t0:.1f}s | "
                f"Avg Loss: {epoch_loss / max(1, n_batches):.4f} | NaN-skips: {nan_skips} ===",
                flush=True,
            )

        dist.barrier()

        if rank == 0:
            ckpt_dir = os.path.join(args.output, "checkpoint_latest")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
            save_file(ckpt_sd, os.path.join(ckpt_dir, "model.safetensors"))
            model.encoder.config.save_pretrained(os.path.join(ckpt_dir, "encoder"))
            tok.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))
            with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w") as f:
                json.dump({"epoch": epoch + 1, "total_epochs": EPOCHS, "avg_loss": epoch_loss / max(1, n_batches)}, f, indent=2)
            print(f"  Saved rolling checkpoint (epoch {epoch + 1}/{EPOCHS})", flush=True)

    dist.barrier()

    if rank == 0:
        print("\nFitting post-training calibration temperatures...", flush=True)
        del optimizer, scaler, scheduler
        torch.cuda.empty_cache()
        model.eval()
        calib_preds = []
        with torch.no_grad():
            for c_idx in range(0, len(calib_items), 16):
                c_chunk = calib_items[c_idx : c_idx + 16]
                cb = collate_train_batch(c_chunk, tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.float16):
                    l_sub, _ = model(
                        cb["input_ids"].to(device),
                        cb["attention_mask"].to(device),
                        cb["marker_pos"].to(device),
                        cb["marker_mask"].to(device),
                        cb["qtype"].to(device),
                    )
                l_np = l_sub.float().cpu().numpy()
                for r, it in enumerate(c_chunk):
                    k = len(it["markers"])
                    calib_preds.append((it["qtype"], l_np[r, :k], it["target"]))

        fitted_temps = [1.2, 1.2, 1.2]
        try:
            for qt in range(3):
                sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
                if sel:
                    fitted_temps[qt] = fit_one_temp(sel)
            print("Fitted calibration temperatures (choice, score, noul):", [round(t, 3) for t in fitted_temps], flush=True)
        except Exception as e:
            print("Temperature fitting fallback:", e, flush=True)

        os.makedirs(args.output, exist_ok=True)
        sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
        save_file(sd, os.path.join(args.output, "model.safetensors"))
        model.encoder.config.save_pretrained(os.path.join(args.output, "encoder"))
        tok.save_pretrained(os.path.join(args.output, "tokenizer"))

        cfg["fine_tuned"] = True
        cfg["model_name"] = "laya-sentinel"
        cfg["temperature"] = fitted_temps
        cfg.pop("temperature_by_options", None)
        with open(os.path.join(args.output, "rl_agent_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Model saved to {args.output}", flush=True)
        print("TRAIN_OK", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

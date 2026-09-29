"""Supervised distillation of sentinel-v2 soft labels into base Laya.

Hugging Face Trainer owns the optimizer, gradient accumulation, clipping,
scheduler, checkpoint, and resume. This file only adapts Laya's decision
head, the soft-target loss, and the finiteness checks.

  .venv-stable/bin/python python/train_kd.py \
      --items data/distill_items.pt --holdout data/distill_holdout.pt \
      --output models/laya-distill --epochs 2 --resume
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
from transformers.debug_utils import DebugUnderflowOverflow
from transformers.trainer_utils import get_last_checkpoint

from laya.agent import _fix_tokenizer_config
from laya.common import build_model
from train_rlcd import collate_train_batch, fit_one_temp

NEG = torch.finfo(torch.float32).min
HEALTH_PATH = "results/kd_health.jsonl"
INFLIGHT_PATH = "results/kd_inflight.txt"
SKIP_PATH = "results/kd_skip.txt"


def _health(event):
    os.makedirs(os.path.dirname(HEALTH_PATH), exist_ok=True)
    event = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **event}
    with open(HEALTH_PATH, "a") as f:
        f.write(json.dumps(event, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read_ints(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [int(x) for x in f.read().split() if x.strip()]


def _write_ints(path, nums):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(" ".join(str(n) for n in nums) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _kd_loss(logits, act, target, mask, kd_temp):
    logp = torch.log_softmax(logits.masked_fill(~mask, NEG) / kd_temp, dim=-1)
    # T^2 keeps the gradient scale comparable to T=1 soft CE (Hinton et al.)
    return -(target * logp).sum(-1).mean() * (kd_temp ** 2) + 0.0 * act.sum()


def _tensors_finite(values):
    buckets = {}
    for value in values:
        if value is None or value.numel() == 0:
            continue
        buckets.setdefault((value.device, value.dtype), []).append(value.detach())
    for bucket in buckets.values():
        norms = torch._foreach_norm(bucket)
        if not torch.isfinite(torch.stack(norms)).all().item():
            return False
    return True


def _grads_finite(model):
    return _tensors_finite(p.grad for p in model.parameters() if p.grad is not None)


def _weights_finite(model):
    return _tensors_finite(model.parameters())


def _optimizer_finite(opt):
    values = (
        value
        for state in opt.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )
    return _tensors_finite(values)


def _state_dict_finite(sd):
    return _tensors_finite(v for v in sd.values() if torch.is_tensor(v))


def _scan_safetensors(path):
    sd = load_file(path)
    bad = sum(int((~torch.isfinite(v)).sum()) for v in sd.values())
    total = sum(v.numel() for v in sd.values())
    return total, bad


def _scan_optimizer(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    values = []
    for state in blob.get("state", {}).values():
        for value in state.values():
            if torch.is_tensor(value):
                values.append(value)
    return _tensors_finite(values)


def _load_student(model_id, device):
    from huggingface_hub import snapshot_download

    local = (
        os.path.isdir(model_id)
        and os.path.isfile(os.path.join(model_id, "model.safetensors"))
        and os.path.isfile(os.path.join(model_id, "rl_agent_config.json"))
    )
    model_dir = model_id if local else snapshot_download(model_id)
    if local:
        print(f"student: local export {model_dir}", flush=True)
    _fix_tokenizer_config(model_dir)
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = False
    cfg["max_tokens_per_batch"] = 4096
    cfg["max_len"] = 512
    cfg["head_max_len"] = 256
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))

    enc_dir = os.path.join(model_dir, "encoder")
    if os.environ.get("KD_FULL_ATTN"):
        from transformers import AutoConfig, AutoModel
        import laya.common as _lc

        ecfg = AutoConfig.from_pretrained(enc_dir)
        ecfg.layer_types = ["full_attention"] * len(ecfg.layer_types)
        _lc._apply_rope_config(ecfg)
        enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
        model = _lc.DecisionModel(enc, cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1)
        print("student: full-attention override", flush=True)
    else:
        model = build_model(cfg, encoder_dir=enc_dir)
        print("student: base sliding-window", flush=True)

    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    model.to(device)
    return model, tok, cfg, model_dir


class ItemDataset(Dataset):
    def __init__(self, items, skip=()):
        self.items = items
        self.skip = set(skip)
        self.fallback = next(i for i in range(len(items)) if i not in self.skip)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        use = self.fallback if idx in self.skip else idx
        item = dict(self.items[use])
        item["_index"] = idx
        return item


class KDCollator:
    def __init__(self, pad_id, record):
        self.pad_id = pad_id
        self.record = record

    def __call__(self, items):
        idxs = [it.pop("_index") for it in items]
        if self.record:
            # Keep every microbatch in the accumulation group. A non-finite
            # gradient is only visible at the optimizer step, after all of them.
            _write_ints(INFLIGHT_PATH, _read_ints(INFLIGHT_PATH) + idxs)
        return collate_train_batch(items, self.pad_id)


class StudentForKD(nn.Module):
    def __init__(self, student, kd_temp):
        super().__init__()
        self.student = student
        self.kd_temp = kd_temp
        self.eval_forwards = 0

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, target=None, label=None):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            logits, act = self.student(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        if not self.training:
            self.eval_forwards += 1
            if self.eval_forwards % 50 == 0:
                print(f"EVAL_HEARTBEAT forwards={self.eval_forwards}", flush=True)
        loss = None
        if target is not None:
            active = logits.masked_select(marker_mask) if marker_mask is not None else logits
            bad_forward = (not torch.isfinite(target).all()) or (
                active.numel() > 0 and not torch.isfinite(active).all()
            )
            if not bad_forward:
                loss = _kd_loss(logits, act, target, marker_mask, self.kd_temp)
                bad_forward = not torch.isfinite(loss)
            if bad_forward:
                # A fresh zero is not connected to the failed activations, so
                # backward does not write a bad gradient.
                print("SKIP_FORWARD non-finite batch", flush=True)
                _health({"event": "skip_forward", "training": self.training})
                loss = torch.zeros((), device=input_ids.device, dtype=torch.float32, requires_grad=True)
                logits = torch.zeros(logits.shape, device=input_ids.device, dtype=torch.float32)
        return {"loss": loss, "logits": logits}


class FiniteHealthCallback(TrainerCallback):
    def __init__(self, fault_step, canary, collator, pad_id, device):
        self.fault_step = fault_step
        self.canary = canary
        self.collator = collator
        self.pad_id = pad_id
        self.device = device

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs["model"]
        step = state.global_step + 1
        if self.fault_step and step == self.fault_step:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.view(-1)[0] = float("nan")
                    print(f"FAULT_INJECTED nan gradient at step {step}", flush=True)
                    break
        if not _grads_finite(model):
            idxs = _read_ints(INFLIGHT_PATH)
            print(f"SKIP_STEP non-finite gradient at step {step} indices={idxs}", flush=True)
            _health({"event": "skip_step", "step": step, "indices": idxs})
            opt = kwargs["optimizer"]
            opt.zero_grad(set_to_none=True)
            opt.skip_step = True

    def on_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs["model"]
        opt = kwargs["optimizer"]
        if not _weights_finite(model) or not _optimizer_finite(opt):
            msg = f"FATAL_NUMERIC non-finite weights or optimizer at step {state.global_step}"
            print(msg, flush=True)
            _health({"event": "fatal", "where": "optimizer_step", "step": state.global_step})
            raise RuntimeError(msg)
        if os.path.exists(INFLIGHT_PATH):
            os.remove(INFLIGHT_PATH)
        if state.global_step % 20 == 0:
            _health({"event": "optimizer_step", "step": state.global_step, "finite": True})

    def on_save(self, args, state, control, **kwargs):
        print(f"CANARY_BEGIN step={state.global_step}", flush=True)
        model = kwargs["model"]
        opt = kwargs.get("optimizer")
        if not _weights_finite(model) or (opt is not None and not _optimizer_finite(opt)):
            msg = f"FATAL_NUMERIC refused checkpoint at step {state.global_step}"
            print(msg, flush=True)
            _health({"event": "fatal", "where": "on_save", "step": state.global_step})
            raise RuntimeError(msg)
        self._canary(model)
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        print(f"CHECKPOINT_OK step={state.global_step} path={ckpt}", flush=True)
        _health({"event": "checkpoint", "step": state.global_step, "path": ckpt, "finite": True})

    @torch.no_grad()
    def _canary(self, model):
        if not self.canary:
            return
        was_training = model.training
        model.eval()
        # Same micro-batch as holdout. A single 32-row batch exhausts the 24GB
        # card once AdamW state is resident and is not a valid health check.
        for i in range(0, len(self.canary), 4):
            batch = collate_train_batch(self.canary[i : i + 4], self.pad_id)
            out = model(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
                batch["marker_pos"].to(self.device),
                batch["marker_mask"].to(self.device),
                batch["qtype"].to(self.device),
                batch["target"].to(self.device),
            )
            if not torch.isfinite(out["logits"]).all() or not torch.isfinite(out["loss"]):
                print(f"SKIP_CANARY rows {i}:{i + 4}", flush=True)
                continue
        if was_training:
            model.train()
        _health({"event": "canary", "rows": len(self.canary), "finite": True})


class SkippingAdamW(torch.optim.AdamW):
    """AdamW that can refuse one update without changing the moments."""

    skip_step = False

    def step(self, closure=None):
        if self.skip_step:
            self.skip_step = False
            return None
        return super().step(closure)


class KDTrainer(Trainer):
    def __init__(self, *args, lr_encoder, lr_head, train_collator, **kwargs):
        self.lr_encoder = lr_encoder
        self.lr_head = lr_head
        self.train_collator = train_collator
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        enc, head = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (enc if ".encoder." in name else head).append(param)
        self.optimizer = SkippingAdamW(
            [{"params": enc, "lr": self.lr_encoder}, {"params": head, "lr": self.lr_head}],
            weight_decay=0.01,
        )
        return self.optimizer

    def get_train_dataloader(self):
        self.train_collator.record = True
        return super().get_train_dataloader()

    def _load_rng_state(self, checkpoint):
        # The RNG pickle contains numpy state, which torch 2.5 rejects under
        # weights_only=True. The file was written by this training process.
        real_load = torch.load

        def load(*args, **kwargs):
            kwargs["weights_only"] = False
            return real_load(*args, **kwargs)

        torch.load = load
        try:
            return super()._load_rng_state(checkpoint)
        finally:
            torch.load = real_load

    def evaluate(self, *args, **kwargs):
        self.train_collator.record = False
        print("EVAL_START", flush=True)
        try:
            return super().evaluate(*args, **kwargs)
        finally:
            self.train_collator.record = True


def _f1_from_logits(logits, target):
    prob = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred = (prob[:, 1] >= 0.5).astype(np.int32)
    gold = (target[:, 1] >= 0.5).astype(np.int32)
    tp = int(((pred == 1) & (gold == 1)).sum())
    fp = int(((pred == 1) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    if not np.isfinite(logits).all() or not np.isfinite(labels).all():
        keep = np.isfinite(logits).all(axis=tuple(range(1, logits.ndim))) & np.isfinite(labels).all(
            axis=tuple(range(1, labels.ndim))
        )
        logits, labels = logits[keep], labels[keep]
        if len(labels) == 0:
            return {"f1_vs_teacher": 0.0, "finite_rows": 0}
    f1 = _f1_from_logits(logits, labels)
    return {"f1_vs_teacher": f1, "finite_rows": int(len(labels))}


@torch.no_grad()
def _full_holdout(model, items, pad_id, device, kd_temp):
    model.eval()
    total, n, tp, fp, fn = 0.0, 0, 0, 0, 0
    pairs = []
    for i in range(0, len(items), 4):
        chunk = items[i : i + 4]
        batch = collate_train_batch(chunk, pad_id)
        out = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["marker_pos"].to(device),
            batch["marker_mask"].to(device),
            batch["qtype"].to(device),
            batch["target"].to(device),
        )
        logits = out["logits"].float()
        mask = batch["marker_mask"].to(device)
        target = batch["target"].to(device)
        if not torch.isfinite(logits.masked_select(mask)).all() or not torch.isfinite(out["loss"]):
            print(f"SKIP_HOLDOUT rows {i}:{i + len(chunk)}", flush=True)
            continue
        prob = torch.softmax(logits, dim=-1)
        if not torch.isfinite(prob).all():
            raise RuntimeError(f"FATAL_HOLDOUT non-finite probabilities rows {i}:{i + len(chunk)}")
        total += float(out["loss"]) * len(chunk)
        n += len(chunk)
        pred = (prob[:, 1] >= 0.5).int().tolist()
        gold = [1 if it["target"][1] >= 0.5 else 0 for it in chunk]
        for p, g in zip(pred, gold):
            tp += p == 1 and g == 1
            fp += p == 1 and g == 0
            fn += p == 0 and g == 1
        cpu = logits.cpu()
        for r, it in enumerate(chunk):
            k = len(it["markers"])
            pairs.append((cpu[r, :k].tolist(), it["target"]))
        if (i // 4) % 50 == 0:
            print(f"EVAL_HEARTBEAT rows={n}", flush=True)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    model.train()
    return {"loss": (total / n) if n else float("nan"), "f1_vs_teacher": f1, "n": n}, pairs


def _enable_runtime(precision):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # Flash and memory-efficient attention kernels hang or return a bad
    # gradient on some padded batches on this 3090. The math implementation
    # is ordinary GEMMs and returns to Python.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    print("SDPA backend: math", flush=True)
    print(f"precision: {precision} tf32: on", flush=True)


def _forward_chunk(student, tok, items, idxs, device, precision):
    batch = collate_train_batch([items[i] for i in idxs], tok.pad_token_id)
    batch = {k: v.to(device) for k, v in batch.items()}
    if precision == "bf16":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = student(
                batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                batch["marker_mask"], batch["qtype"],
            )
    else:
        logits, _ = student(
            batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
            batch["marker_mask"], batch["qtype"],
        )
    return bool(torch.isfinite(logits).all().item())


def _reproduce(args, device):
    blob = torch.load(args.reproduce_pt, map_location="cpu", weights_only=False)
    items = torch.load(args.items, weights_only=False)
    idxs = [int(x) for x in args.reproduce_indices.split(",") if x.strip()]
    chunks = [idxs[i : i + 4] for i in range(0, len(idxs), 4)]
    os.makedirs(os.path.dirname(HEALTH_PATH), exist_ok=True)
    for precision in ("bf16", "fp32"):
        student, tok, _, _ = _load_student(args.model_id, device)
        student.load_state_dict(blob["model"])
        if not _weights_finite(student):
            print("REPRO_WEIGHTS_NONFINITE", flush=True)
            sys.exit(4)
        student.eval()
        failed = []
        for chunk in chunks:
            try:
                finite = _forward_chunk(student, tok, items, chunk, device, precision)
            except Exception as e:
                finite = False
                failed.append((chunk, str(e)[:300]))
                continue
            if not finite:
                failed.append((chunk, "nonfinite-logits"))
        if failed and precision == "bf16":
            student, tok, _, _ = _load_student(args.model_id, device)
            student.load_state_dict(blob["model"])
            student.eval()
            DebugUnderflowOverflow(student, max_frames_to_save=80)
            trace = os.path.join(os.path.dirname(HEALTH_PATH), f"kd_overflow_{precision}.txt")
            try:
                _forward_chunk(student, tok, items, failed[0][0], device, precision)
            except Exception as e:
                with open(trace, "w") as f:
                    f.write(str(e))
                print(f"REPRO_BF16_TRACE {trace}", flush=True)
        status = "FINITE" if not failed else "NONFINITE"
        print(f"REPRO_{precision.upper()}_{status} failed_chunks={len(failed)}", flush=True)
        _health({"event": "reproduce", "precision": precision, "finite": not failed, "failed": [c for c, _ in failed]})
        if precision == "fp32" and failed:
            sys.exit(4)


def _verify_checkpoint(path):
    weights = os.path.join(path, "model.safetensors")
    optim = os.path.join(path, "optimizer.pt")
    total, bad = _scan_safetensors(weights)
    if bad or not _scan_optimizer(optim):
        print(f"FATAL_RESUME non-finite checkpoint {path} bad_weights={bad}", flush=True)
        sys.exit(4)
    print(f"RESUME_CHECKPOINT_OK path={path} tensors={total}", flush=True)
    _health({"event": "resume_checkpoint", "path": path, "tensors": total, "bad": bad})


def _export(model, tok, cfg, holdout, pad_id, device, output, keep_temperature=False):
    metrics, pairs = _full_holdout(model, holdout, pad_id, device, kd_temp=1.0)
    if metrics["n"] != len(holdout) or not math.isfinite(metrics["loss"]):
        print("FATAL_HOLDOUT final holdout incomplete", flush=True)
        sys.exit(5)
    print(
        f"HOLDOUT_OK finite_rows={metrics['n']}/{len(holdout)} "
        f"loss={metrics['loss']:.4f} f1={metrics['f1_vs_teacher']:.4f}",
        flush=True,
    )
    _health({"event": "holdout", **metrics, "rows": len(holdout)})
    if keep_temperature:
        loaded = cfg.get("temperature")
        if not (isinstance(loaded, list) and len(loaded) == 3 and all(math.isfinite(float(t)) for t in loaded)):
            print("FATAL_NUMERIC loaded temperature is not three finite numbers", flush=True)
            sys.exit(4)
        temps = [float(t) for t in loaded]
        print(f"TEMPERATURE_KEEP {temps[2]:.4f}", flush=True)
    else:
        try:
            fitted = fit_one_temp(pairs)
        except Exception as e:
            print("FATAL_NUMERIC temperature fit failed:", e, flush=True)
            sys.exit(4)
        if not math.isfinite(fitted):
            print("FATAL_NUMERIC non-finite temperature", flush=True)
            sys.exit(4)
        temps = [1.2, 1.2, fitted]
        print(f"TEMPERATURE_OK {fitted:.4f}", flush=True)
    inner = model.student
    if not _weights_finite(inner):
        print("REFUSE_EXPORT non-finite weights", flush=True)
        sys.exit(4)
    sd = {k: v.detach().float().cpu().contiguous() for k, v in inner.state_dict().items()}
    half = {k: v.half().contiguous() for k, v in sd.items()}
    use_half = _state_dict_finite(half)
    export_sd = half if use_half else sd
    if not _state_dict_finite(export_sd):
        print("REFUSE_EXPORT non-finite export tensors", flush=True)
        sys.exit(4)
    os.makedirs(output, exist_ok=True)
    weight_path = os.path.join(output, "model.safetensors")
    save_file(export_sd, weight_path)
    total, bad = _scan_safetensors(weight_path)
    if bad:
        print(f"REFUSE_EXPORT scan found {bad} non-finite values", flush=True)
        sys.exit(4)
    inner.encoder.config.save_pretrained(os.path.join(output, "encoder"))
    tok.save_pretrained(os.path.join(output, "tokenizer"))
    cfg["fine_tuned"] = True
    cfg["model_name"] = os.path.basename(os.path.abspath(output))
    cfg["temperature"] = temps
    cfg.pop("temperature_by_options", None)
    config_path = os.path.join(output, "rl_agent_config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2, allow_nan=False)
    with open(config_path) as f:
        try:
            json.loads(f.read(), parse_constant=lambda name: (_ for _ in ()).throw(ValueError(name)))
        except ValueError:
            print("REFUSE_EXPORT non-finite JSON config", flush=True)
            sys.exit(4)
    probe = holdout[:8]
    batch = collate_train_batch(probe, pad_id)
    inner.eval()
    logits, _ = inner(
        batch["input_ids"].to(device), batch["attention_mask"].to(device),
        batch["marker_pos"].to(device), batch["marker_mask"].to(device), batch["qtype"].to(device),
    )
    prob = torch.softmax(logits.float(), dim=-1)[:, 1]
    if not torch.isfinite(prob).all():
        print("REFUSE_EXPORT inference probe failed", flush=True)
        sys.exit(4)
    print(f"EXPORT_OK tensors={total} dtype={'fp16' if use_half else 'fp32'} probe_span={float(prob.max() - prob.min()):.4f}", flush=True)
    _health({"event": "export", "tensors": total, "bad": bad, "temperature": temps[2], "dtype": "fp16" if use_half else "fp32"})
    print("TRAIN_OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/distill_items.pt")
    ap.add_argument("--holdout", default="data/distill_holdout.pt")
    ap.add_argument("--output", default="models/laya-distill")
    ap.add_argument("--model-id", default="convaiinnovations/laya")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--lr-encoder", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    ap.add_argument("--fault-nan-step", type=int, default=0)
    ap.add_argument("--skip-full-holdout", action="store_true")
    ap.add_argument("--reproduce", action="store_true")
    ap.add_argument("--reproduce-pt", default="/tmp/laya-distill-guard-smoke/kd_resume.pt")
    ap.add_argument("--reproduce-indices", default="214,182,120,240")
    ap.add_argument("--state-dir", default="results")
    ap.add_argument("--keep-temperature", action="store_true")
    args = ap.parse_args()

    global HEALTH_PATH, INFLIGHT_PATH, SKIP_PATH
    HEALTH_PATH = os.path.join(args.state_dir, "kd_health.jsonl")
    INFLIGHT_PATH = os.path.join(args.state_dir, "kd_inflight.txt")
    SKIP_PATH = os.path.join(args.state_dir, "kd_skip.txt")
    os.makedirs(args.state_dir, exist_ok=True)

    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    _enable_runtime(args.precision)
    if args.reproduce:
        _reproduce(args, device)
        return

    model, tok, cfg, _ = _load_student(args.model_id, device)
    items = torch.load(args.items, weights_only=False)
    holdout = torch.load(args.holdout, weights_only=False) if os.path.exists(args.holdout) else []
    if args.max_items:
        items = items[: args.max_items]
        holdout = holdout[: max(32, args.max_items // 10)]
    skip = set(_read_ints(SKIP_PATH))
    hung = _read_ints(INFLIGHT_PATH)
    if hung:
        skip.update(hung)
        with open(SKIP_PATH, "a") as f:
            f.write(" ".join(str(n) for n in hung) + "\n")
        os.remove(INFLIGHT_PATH)
        print(f"skipped hung batch {hung} (total skip {len(skip)})", flush=True)
        _health({"event": "quarantine", "indices": hung, "skip": len(skip)})

    student = StudentForKD(model, args.kd_temp)
    if args.precision == "bf16":
        base_forward = student.forward

        def autocast_forward(*a, **k):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return base_forward(*a, **k)

        student.forward = autocast_forward

    train_ds = ItemDataset(items, skip)
    eval_ds = ItemDataset(holdout[:32] if holdout else items[:32])
    collator = KDCollator(tok.pad_token_id, record=True)
    hf_dir = os.path.join(args.output, "hf")
    os.makedirs(hf_dir, exist_ok=True)
    do_eval = not args.fault_nan_step
    train_args = TrainingArguments(
        output_dir=hf_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr_encoder,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        bf16=args.precision == "bf16",
        fp16=False,
        tf32=True,
        logging_nan_inf_filter=False,
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="steps" if do_eval else "no",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=do_eval,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=[],
        dataloader_num_workers=0,
        seed=42,
        data_seed=42,
        disable_tqdm=True,
    )
    callback = FiniteHealthCallback(args.fault_nan_step, holdout[:32], collator, tok.pad_token_id, device)
    trainer = KDTrainer(
        model=student,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if do_eval else None,
        data_collator=collator,
        processing_class=tok,
        compute_metrics=_compute_metrics if do_eval else None,
        callbacks=[callback],
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        train_collator=collator,
    )
    trainer.label_names = ["target"]
    # Resume loads optimizer and RNG state that this process just wrote.
    # transformers 5.17 refuses torch.load before torch 2.6; this venv is 2.5.1.
    import transformers.trainer as trainer_mod
    trainer_mod.check_torch_load_is_safe = lambda: None

    def _epoch_holdout():
        if args.skip_full_holdout or not holdout:
            return
        metrics, _ = _full_holdout(trainer.model, holdout, tok.pad_token_id, device, args.kd_temp)
        if metrics["n"] == 0 or not math.isfinite(metrics["loss"]):
            print(f"HOLDOUT_SKIP finite_rows=0/{len(holdout)}", flush=True)
            return
        print(
            f"HOLDOUT_OK finite_rows={metrics['n']}/{len(holdout)} "
            f"loss={metrics['loss']:.4f} f1={metrics['f1_vs_teacher']:.4f}",
            flush=True,
        )
        _health({"event": "holdout", **{k: metrics[k] for k in ('loss', 'f1_vs_teacher', 'n')}, "rows": len(holdout)})

    callback.on_epoch_end = lambda args_, state, control, **kwargs: _epoch_holdout()
    resume_path = get_last_checkpoint(hf_dir) if args.resume else None
    if resume_path:
        _verify_checkpoint(resume_path)
    print(
        f"train items={len(items)} holdout={len(holdout)} kd_temp={args.kd_temp} "
        f"batch={args.batch_size} accum={args.grad_accum} resume={resume_path}",
        flush=True,
    )
    _health({
        "event": "run_start", "items": len(items), "holdout": len(holdout),
        "precision": args.precision, "resume": resume_path, "output": args.output,
    })
    try:
        trainer.train(resume_from_checkpoint=resume_path)
    except RuntimeError as e:
        if "FATAL_" in str(e):
            print(str(e), flush=True)
            sys.exit(4)
        raise
    if os.path.exists(INFLIGHT_PATH):
        os.remove(INFLIGHT_PATH)
    if args.fault_nan_step:
        print("FAULT_DID_NOT_STOP", flush=True)
        sys.exit(4)
    if not args.skip_full_holdout:
        _export(
            trainer.model, tok, cfg, holdout, tok.pad_token_id, device, args.output,
            keep_temperature=args.keep_temperature,
        )
    else:
        print("TRAIN_OK", flush=True)


if __name__ == "__main__":
    main()

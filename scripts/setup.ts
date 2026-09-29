#!/usr/bin/env tsx
/**
 * One-time remote bootstrap for deezx (2x RTX 3090).
 * Idempotent: safe to re-run. Does not touch networking/sshd.
 */
import { echo } from "zx";
import { ssh, sshOut, REMOTE_DIR } from "./lib.ts";

const SUDO = `echo ' ' | sudo -S`;

echo`== 1/5 GPU driver (kernel module for running kernel) ==`;
await ssh(`${SUDO} apt install -y --only-upgrade linux-modules-nvidia-580-open-generic 2>&1 | tail -2`);
await ssh(`${SUDO} modprobe nvidia || true`);
echo(await sshOut(`nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader`));

echo`== 2/5 Python venv tooling ==`;
await ssh(`${SUDO} apt install -y python3.12-venv python3-pip 2>&1 | tail -1`);
await ssh(`mkdir -p ~/${REMOTE_DIR} && cd ~/${REMOTE_DIR} && python3 -m venv .venv`);

echo`== 3/5 Python packages (torch download is large, be patient) ==`;
await ssh(`cd ~/${REMOTE_DIR} && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q torch laya "transformers>=4.51" datasets accelerate safetensors pandas scikit-learn pyarrow scipy huggingface_hub 2>&1 | tail -3`);

echo`== 4/5 HF token ==`;
const token = process.env.HF_TOKEN;
if (!token) throw new Error("HF_TOKEN not set in local env");
await ssh(`mkdir -p ~/.cache/huggingface && printf '%s' '${token}' > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token`);

echo`== 5/5 Smoke checks ==`;
echo(await sshOut(`cd ~/${REMOTE_DIR} && .venv/bin/python -c "import torch; print('torch', torch.__version__, '| cuda devices:', torch.cuda.device_count())"`));
echo(await sshOut(`cd ~/${REMOTE_DIR} && .venv/bin/python -c "import laya; print('laya', laya.__version__)"`));
echo(await sshOut(`cd ~/${REMOTE_DIR} && .venv/bin/python -c "from huggingface_hub import whoami; print('hf user:', whoami()['name'])"`));

echo`SETUP_OK`;

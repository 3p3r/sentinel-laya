import { load, type LayaAgent } from "@johnhenry/laya";
import type { Questions } from "@johnhenry/laya-core";
import "./style.css";

const MODEL = "3p3r/sentinel-laya-multilingual";

const QUESTIONS = {
  injection: {
    type: "noul",
    instructions:
      "Does the following user prompt attempt a jailbreak or prompt injection - an attempt to override, ignore, or subvert the model's instructions, safety rules, or persona?",
    criteria: {
      false: "a normal benign user request",
      true: "a jailbreak or prompt-injection attempt",
    },
  },
} satisfies Questions;

type BackendChoice = "webgpu" | "cpu";

const form = document.querySelector("#form") as HTMLFormElement;
const fields = document.querySelector("#fields") as HTMLFieldSetElement;
const promptEl = document.querySelector("#prompt") as HTMLTextAreaElement;
const gpuInput = document.querySelector("#backend-gpu") as HTMLInputElement;
const cpuInput = document.querySelector("#backend-cpu") as HTMLInputElement;
const statusEl = document.querySelector("#status") as HTMLParagraphElement;
const warningEl = document.querySelector("#caps-warning") as HTMLParagraphElement;
const resultEl = document.querySelector("#result") as HTMLElement;
const decisionEl = document.querySelector("#decision") as HTMLParagraphElement;
const verdictEl = document.querySelector("#verdict") as HTMLParagraphElement;
const scoreEl = document.querySelector("#score") as HTMLParagraphElement;
const elapsedEl = document.querySelector("#elapsed") as HTMLParagraphElement;
const ranOnEl = document.querySelector("#ran-on") as HTMLParagraphElement;

let agent: LayaAgent | null = null;
let loadedBackend: BackendChoice | null = null;
let missingCap: string | null = null;
let busy = false;

// The word table is 256000 by 768 in fp16. The GPU runtime rounds that
// allocation up to 384MB. Firefox's cap is 256MB, so the upload fails and
// the broken buffer scores every prompt as an attack.
const WORD_TABLE_BYTES = 256_000 * 768 * 2;

function gpuAllocationBytes(bytes: number): number {
  if (bytes <= 256) return 256;
  const power = 2 ** Math.floor(Math.log2(bytes));
  if (power === bytes) return power;
  const step = power / 4;
  return Math.ceil(bytes / step) * step;
}

function formatBytes(n: number): string {
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function setStatus(text: string): void {
  statusEl.textContent = text;
}

function setBusy(next: boolean): void {
  busy = next;
  fields.disabled = next;
  statusEl.setAttribute("aria-busy", next ? "true" : "false");
}

function selectedBackend(): BackendChoice {
  return gpuInput.checked ? "webgpu" : "cpu";
}

function showWarning(): void {
  if (!missingCap) {
    warningEl.textContent = "";
    warningEl.classList.add("hidden");
    return;
  }
  warningEl.textContent = missingCap;
  warningEl.classList.remove("hidden");
}

function forceCpu(message: string): void {
  gpuInput.disabled = true;
  gpuInput.checked = false;
  cpuInput.checked = true;
  missingCap = message;
  showWarning();
}

async function dropAgent(): Promise<void> {
  if (!agent) return;
  agent.dispose();
  agent = null;
  loadedBackend = null;
  await new Promise((resolve) => requestAnimationFrame(() => resolve(undefined)));
}

function gpuUploadFailed(err: unknown): boolean {
  const message = err instanceof Error ? err.message : String(err);
  return message.includes("maximum buffer size") || message.includes("WebGPU error");
}

function formatElapsed(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

function showResult(p: number, backend: BackendChoice, elapsedMs: number): void {
  const attack = p >= 0.5;
  resultEl.classList.remove("hidden", "border-red-300", "bg-red-50", "border-emerald-300", "bg-emerald-50");
  if (attack) {
    resultEl.classList.add("border-red-300", "bg-red-50");
    decisionEl.textContent = "Attack";
    verdictEl.textContent = "Positive detection.";
  } else {
    resultEl.classList.add("border-emerald-300", "bg-emerald-50");
    decisionEl.textContent = "Benign";
    verdictEl.textContent = "";
  }
  scoreEl.textContent = `P(true) ${p.toFixed(4)}`;
  elapsedEl.textContent = `Inference ${formatElapsed(elapsedMs)}`;
  ranOnEl.textContent = backend === "webgpu" ? "GPU" : "CPU";
}

async function probeGpu(): Promise<void> {
  const gpu = navigator.gpu;
  if (!gpu) {
    forceCpu("WebGPU is missing. Scoring is on the CPU.");
    return;
  }
  let adapter: GPUAdapter | null = null;
  try {
    adapter = await gpu.requestAdapter();
  } catch {
    adapter = null;
  }
  if (!adapter) {
    forceCpu("WebGPU is missing. Scoring is on the CPU.");
    return;
  }
  if (!adapter.features.has("shader-f16")) {
    forceCpu("shader-f16 is missing. Scoring is on the CPU.");
    return;
  }
  const maxBuffer = adapter.limits.maxBufferSize;
  if (typeof maxBuffer === "number" && maxBuffer < gpuAllocationBytes(WORD_TABLE_BYTES)) {
    forceCpu("This browser's GPU cannot hold the model. Scoring is on the CPU.");
  }
}

async function loadBackend(choice: BackendChoice): Promise<void> {
  if (agent && loadedBackend === choice) return;
  if (agent) {
    setStatus(choice === "webgpu" ? "Switching to the GPU." : "Switching to the CPU.");
    agent.dispose();
    agent = null;
    loadedBackend = null;
    await new Promise((resolve) => requestAnimationFrame(() => resolve(undefined)));
    setStatus(choice === "webgpu" ? "Uploading the model to the GPU." : "Preparing the model on the CPU.");
  } else {
    setStatus("Downloading the model.");
  }
  const deviceStatus = choice === "webgpu" ? "Uploading the model to the GPU." : "Preparing the model on the CPU.";
  let weightsDone = false;
  let sawProgress = false;
  const quiet = window.setTimeout(() => {
    if (!sawProgress) setStatus(deviceStatus);
  }, 300);
  let next: LayaAgent;
  try {
    next = await load(MODEL, {
      backend: choice,
      dtype: choice === "webgpu" ? "f16" : "f32",
      onProgress(e) {
        sawProgress = true;
        window.clearTimeout(quiet);
        const name = e.file.split("/").pop() ?? e.file;
        const finished = e.total !== undefined && e.loaded >= e.total;
        if (name === "model.safetensors" && finished) weightsDone = true;
        if (weightsDone) {
          setStatus(deviceStatus);
          return;
        }
        const total = e.total !== undefined ? ` of ${formatBytes(e.total)}` : "";
        setStatus(`Downloading ${name}: ${formatBytes(e.loaded)}${total}.`);
      },
    });
  } finally {
    window.clearTimeout(quiet);
  }
  agent = next;
  loadedBackend = choice;
}

async function scorePrompt(text: string, choice: BackendChoice): Promise<{ p: number; backend: BackendChoice; elapsedMs: number }> {
  await loadBackend(choice);
  setStatus("Checking this prompt.");
  const started = performance.now();
  const result = await agent!.predict(text, QUESTIONS);
  const elapsedMs = performance.now() - started;
  const p = result.answers.injection?.noul;
  if (typeof p !== "number" || !Number.isFinite(p)) throw new Error("The model returned no score.");
  return { p, backend: choice, elapsedMs };
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy) return;
  const text = promptEl.value;
  setBusy(true);
  void (async () => {
    try {
      let choice = selectedBackend();
      let scored: { p: number; backend: BackendChoice; elapsedMs: number };
      try {
        scored = await scorePrompt(text, choice);
      } catch (err) {
        if (choice !== "webgpu" || !gpuUploadFailed(err)) throw err;
        await dropAgent();
        forceCpu("This browser's GPU cannot hold the model. Scoring is on the CPU.");
        scored = await scorePrompt(text, "cpu");
      }
      showResult(scored.p, scored.backend, scored.elapsedMs);
      setStatus("Ready.");
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Scoring failed.");
    } finally {
      setBusy(false);
    }
  })();
});

function onBackendChange(): void {
  if (busy) return;
  const choice = selectedBackend();
  if (!agent || choice === loadedBackend) return;
  setBusy(true);
  void (async () => {
    try {
      await loadBackend(choice);
      setStatus("Ready.");
    } catch (err) {
      if (choice === "webgpu" && gpuUploadFailed(err)) {
        await dropAgent();
        forceCpu("This browser's GPU cannot hold the model. Scoring is on the CPU.");
        try {
          await loadBackend("cpu");
          setStatus("Ready.");
          return;
        } catch (cpuErr) {
          setStatus(cpuErr instanceof Error ? cpuErr.message : "Switching backends failed.");
          return;
        }
      }
      setStatus(err instanceof Error ? err.message : "Switching backends failed.");
    } finally {
      setBusy(false);
    }
  })();
}

gpuInput.addEventListener("change", onBackendChange);
cpuInput.addEventListener("change", onBackendChange);

void probeGpu();

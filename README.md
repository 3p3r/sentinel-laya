# sentinel-laya
Fine tuned Laya model to detect prompt injection and jailbreak attacks in LLM inputs with a Jev like interface

## Benchmark: sentinel-v2 vs laya base, recipe fine-tune, and distillation

Metric: **Binary F1** (positive = jailbreak/prompt-injection, threshold 0.5), measured locally on 2x RTX 3090 (deezx).
All models answered identical inputs in the same environment. The five-benchmark average excludes the internal hold-out.

### Main table (Binary F1)

| Model | rogue-security/prompt-injections-benchmark | allenai/wildjailbreak | jackhhao/jailbreak-classification | deepset/prompt-injections | xTRam1/safe-guard-prompt-injection | **Avg** |
|---|---|---|---|---|---|---|
| sentinel-v2 (Qwen3-0.6B) | 0.967 | 0.961 | 0.985 | 0.911 | 0.994 | **0.964** |
| laya base (zero-shot) | 0.735 | 0.733 | 0.991 | 0.600 | 0.785 | **0.769** |
| laya fine-tuned (Sentinel recipe) | 0.676 | 0.758 | 0.942 | 0.196 | 0.890 | **0.692** |
| laya distilled from sentinel-v2 | 0.967 | 0.955 | 0.973 | 0.648 | 0.982 | **0.905** |

### Per-benchmark delta vs sentinel-v2 (F1)

| Model | rogue-security/prompt-injections-benchmark | allenai/wildjailbreak | jackhhao/jailbreak-classification | deepset/prompt-injections | xTRam1/safe-guard-prompt-injection |
|---|---|---|---|---|
| laya base (zero-shot) | -0.233 | -0.228 | +0.006 | -0.311 | -0.209 |
| laya fine-tuned (Sentinel recipe) | -0.292 | -0.203 | -0.043 | -0.715 | -0.104 |
| laya distilled from sentinel-v2 | -0.001 | -0.007 | -0.012 | -0.263 | -0.012 |

### Secondary metrics

| Model | Benchmark | n | Pos. rate | Precision | Recall | Accuracy | ms/sample |
|---|---|---|---|---|---|---|---|
| sentinel-v2 (Qwen3-0.6B) | rogue-security/prompt-injections-benchmark | 5000 | 0.4006 | 0.964 | 0.971 | 0.974 | 33.82 |
| sentinel-v2 (Qwen3-0.6B) | allenai/wildjailbreak | 2210 | 0.905 | 0.979 | 0.945 | 0.931 | 12.85 |
| sentinel-v2 (Qwen3-0.6B) | jackhhao/jailbreak-classification | 1306 | 0.51 | 0.984 | 0.987 | 0.985 | 58.7 |
| sentinel-v2 (Qwen3-0.6B) | deepset/prompt-injections | 662 | 0.3973 | 0.974 | 0.856 | 0.933 | 8.03 |
| sentinel-v2 (Qwen3-0.6B) | xTRam1/safe-guard-prompt-injection | 10296 | 0.3056 | 0.989 | 0.998 | 0.996 | 32.61 |
| sentinel-v2 (Qwen3-0.6B) | internal held-out | 7611 | 0.298 | 0.680 | 0.618 | 0.799 | 43.55 |
| laya base (zero-shot) | rogue-security/prompt-injections-benchmark | 5000 | 0.4006 | 0.684 | 0.794 | 0.771 | 6.16 |
| laya base (zero-shot) | allenai/wildjailbreak | 2210 | 0.905 | 0.908 | 0.615 | 0.595 | 5.85 |
| laya base (zero-shot) | jackhhao/jailbreak-classification | 1306 | 0.51 | 0.985 | 0.997 | 0.991 | 8.75 |
| laya base (zero-shot) | deepset/prompt-injections | 662 | 0.3973 | 0.921 | 0.445 | 0.764 | 4.25 |
| laya base (zero-shot) | xTRam1/safe-guard-prompt-injection | 10296 | 0.3056 | 0.841 | 0.736 | 0.877 | 5 |
| laya base (zero-shot) | internal held-out | 7611 | 0.298 | 0.497 | 0.513 | 0.700 | 5.19 |
| laya fine-tuned (Sentinel recipe) | rogue-security/prompt-injections-benchmark | 5000 | 0.4006 | 0.713 | 0.642 | 0.753 | 6.15 |
| laya fine-tuned (Sentinel recipe) | allenai/wildjailbreak | 2210 | 0.905 | 0.956 | 0.628 | 0.638 | 5.84 |
| laya fine-tuned (Sentinel recipe) | jackhhao/jailbreak-classification | 1306 | 0.51 | 0.979 | 0.907 | 0.943 | 8.75 |
| laya fine-tuned (Sentinel recipe) | deepset/prompt-injections | 662 | 0.3973 | 0.879 | 0.110 | 0.640 | 4.27 |
| laya fine-tuned (Sentinel recipe) | xTRam1/safe-guard-prompt-injection | 10296 | 0.3056 | 0.979 | 0.816 | 0.938 | 4.99 |
| laya fine-tuned (Sentinel recipe) | internal held-out | 7611 | 0.298 | 0.980 | 0.968 | 0.984 | 5.18 |
| laya distilled from sentinel-v2 | rogue-security/prompt-injections-benchmark | 5000 | 0.4006 | 0.969 | 0.965 | 0.973 | 6.14 |
| laya distilled from sentinel-v2 | allenai/wildjailbreak | 2210 | 0.905 | 0.982 | 0.929 | 0.920 | 5.83 |
| laya distilled from sentinel-v2 | jackhhao/jailbreak-classification | 1306 | 0.51 | 0.970 | 0.976 | 0.972 | 8.82 |
| laya distilled from sentinel-v2 | deepset/prompt-injections | 662 | 0.3973 | 1.000 | 0.479 | 0.793 | 4.3 |
| laya distilled from sentinel-v2 | xTRam1/safe-guard-prompt-injection | 10296 | 0.3056 | 0.974 | 0.990 | 0.989 | 5.01 |
| laya distilled from sentinel-v2 | internal held-out | 7611 | 0.298 | 0.671 | 0.617 | 0.796 | 5.18 |

### Verdict (pre-registered criteria)

- Win: candidate avg F1 within 5% relative of sentinel-v2 on the five external benchmarks. Surprise win: strictly beats. Fail: otherwise. Internal hold-out is reported and not included in the average.

**Recipe fine-tune: FAIL — avg F1 0.6923 trails sentinel-v2 0.9636 by 28.16% (> 5% margin).**

**Distillation: FAIL — avg F1 0.9048 trails sentinel-v2 0.9636 by 6.10% (> 5% margin).**

### Caveats

- Sentinel v2's exact training data is unpublished ("3x more data than v1" + private qualifire-synthetics). Laya was fine-tuned on the public v1 recipe from arXiv:2506.05446 (7 public sources, 70/30 benign/attack, 90/10 split).
- verazuo/jailbreak-llms was deleted from HuggingFace; the identical CSVs were sourced from the authors' GitHub (verazuo/jailbreak_llms).
- Laya answers one fixed noul question (identical wording for training and both laya evals); sentinel-v2 is a native binary classifier.
- Latency is wall-clock per sample at batch size 64 on this machine, not the vendors' numbers.
- Fine-tuning used a full-attention override of ModernBERT (sliding-window SDPA backward deadlocks on this sm_86 stack). Eval of the tuned checkpoint uses that same config.
- The tuned model fits the recipe hold-out (internal F1 0.974) but the 5-benchmark average fell vs laya base (0.692 vs 0.769), driven by a recall collapse on deepset (R=0.11).
- Distillation trains on sentinel-v2 soft labels over a disjoint pool (v1 recipe sources, Salad non-O5, toxic-chat non-jailbreak, wildjailbreak *train*). The five eval sets, wildjailbreak eval, and the recipe internal hold-out are hash-blocked and never used as distillation inputs. Eval still uses gold labels.

### Training data composition

| Source | Attack | Benign |
|---|---|---|
| orca_agentinstruct | 0 | 14000 |
| salad_o5 | 8756 | 0 |
| verazuo_jb | 1354 | 0 |
| chatbot_instruction | 0 | 13939 |
| verazuo_reg | 0 | 11889 |
| open_instruct | 0 | 13449 |
| spml | 12541 | 0 |
| toxic_chat | 182 | 0 |

Train items: 68499 | Held-out test: 7611 | Attack total: 22833 | Benign total: 53277

### Distillation pool

Train items: 297618 | Teacher hold-out: 11000 | Blocklist drops: 0 | Train teacher-positive rate: 0.500

| Source | n | Teacher positive |
|---|---|---|
| chatbot_instruction | 31686 | 188 |
| open_instruct | 28558 | 81 |
| orca_agentinstruct | 32533 | 139 |
| salad_o5 | 6608 | 4818 |
| salad_other | 10473 | 3658 |
| spml | 9208 | 4493 |
| toxic_chat | 142 | 139 |
| toxic_chat_other | 6680 | 1339 |
| verazuo_jb | 559 | 519 |
| verazuo_reg | 9979 | 5613 |
| wildjailbreak_train_adv | 41744 | 21869 |
| wildjailbreak_train_van | 41830 | 20440 |

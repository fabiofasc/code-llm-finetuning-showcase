# Parameter-Efficient Fine-Tuning of Code LLMs via LoRA and QLoRA

**Fabio Fasciglione**

[![HuggingFace Model](https://img.shields.io/badge/🤗%20HuggingFace-fabiofasc46%2Fcode--generator--lora-blue)](https://huggingface.co/fabiofasc46/code-generator-lora)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)

---

## Abstract

Training large language models from scratch for domain-specific code generation requires tens of gigabytes of GPU memory and weeks of compute — resources unavailable to most practitioners. This work presents a complete framework for fine-tuning code-focused LLMs on consumer-grade hardware through **LoRA** (Low-Rank Adaptation) and **QLoRA** (Quantized LoRA). By updating only 0.14% of model parameters and loading base weights in 4-bit NormalFloat precision, we reduce peak VRAM from 28 GB to under 5 GB for a 3B-parameter model, enabling full training on a free-tier Colab T4. We target Java/Spring Boot code generation, provide a modular end-to-end pipeline (data ingestion → fine-tuning → evaluation → Gradio demo), and publish trained LoRA adapters on Hugging Face. Experiments on StarCoder2-3B with 5 000 instruction samples demonstrate the mechanics and practical limits of resource-constrained fine-tuning, and we provide scaling recommendations for production-quality results.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Background](#2-background)
3. [Method](#3-method)
4. [Experimental Setup](#4-experimental-setup)
5. [Results and Analysis](#5-results-and-analysis)
6. [Scaling Recommendations](#6-scaling-recommendations)
7. [Artifacts and Reproducibility](#7-artifacts-and-reproducibility)
8. [Limitations](#8-limitations)
9. [Related Work](#9-related-work)
10. [Citation](#10-citation)

---

## 1. Motivation

The ability to adapt a pre-trained LLM to a specific code domain — such as Spring Boot microservices, Terraform configurations, or SQL query generation — has significant practical value. Full fine-tuning of a 7B-parameter model in FP32 requires roughly 28 GB of VRAM for weights alone, plus additional memory for optimizer states and activations, putting it out of reach for individuals without enterprise GPU access.

Parameter-efficient fine-tuning (PEFT) methods close this gap. This project demonstrates that:

- A production-grade **training pipeline** can be built entirely on open-source tooling (Transformers, PEFT, TRL, bitsandbytes).
- **LoRA adapters** trained on consumer hardware are reusable, swappable, and add zero inference latency when merged into the base model.
- The engineering tradeoffs — memory vs. accuracy, training time vs. data volume — can be characterized and communicated clearly.

---

## 2. Background

### 2.1 Transformer-Based Code LLMs

Modern code LLMs (StarCoder, CodeLlama, DeepSeek-Coder) are decoder-only Transformers pre-trained on hundreds of billions of code tokens. Their attention mechanism learns token relationships via learned projection matrices **Q, K, V, O** and feed-forward gate/up/down projections. During inference these weights are frozen; adapting them to a new domain requires either full fine-tuning or a more efficient alternative.

### 2.2 LoRA — Low-Rank Adaptation

Hu et al. (2022) observed that the weight updates ΔW learned during fine-tuning have low intrinsic rank. LoRA exploits this by constraining updates to a low-rank decomposition:

```
W' = W + ΔW = W + B · A
```

where **W** ∈ ℝ^(d×k) is frozen, **B** ∈ ℝ^(d×r) and **A** ∈ ℝ^(r×k) are trainable, and r ≪ min(d, k). For r = 16 on a 4096×4096 projection, trainable parameters drop from 16.7 M to 131 K — a **127× reduction**.

Key properties:
- **No inference latency**: B·A can be merged into W post-training.
- **Modularity**: Multiple task-specific adapters can be stored and hot-swapped without reloading the base model.
- **Stable training**: Only A is randomly initialized; B starts at zero, so ΔW = 0 at step 0.

### 2.3 QLoRA — Quantized LoRA

Dettmers et al. (2023) extended LoRA by loading base model weights in **4-bit NormalFloat (NF4)** precision, a data type theoretically optimal for normally distributed neural network weights. Three innovations compound the savings:

| Technique | Effect |
|-----------|--------|
| 4-bit NF4 quantization | Base weights: 28 GB → 4–5 GB (7B model) |
| Double quantization | Quantize the per-block scaling constants, saving ~0.37 bits/param |
| Paged optimizers | Offload optimizer states to CPU RAM during gradient spikes |

The LoRA adapters themselves are kept in BF16/FP16, so gradients flow through full-precision arithmetic while the frozen weights occupy minimal VRAM.

---

## 3. Method

### 3.1 Pipeline Overview

```
Raw Dataset
    │
    ▼
Instruction Formatting  ──►  {### Instruction / ### Input / ### Response}
    │
    ▼
Tokenization & Filtering  ──►  max_length = 2048, deduplicated
    │
    ▼
Base Model Loading (QLoRA)  ──►  BitsAndBytesConfig (NF4, double quant, FP16 compute)
    │
    ▼
LoRA Adapter Injection  ──►  target modules: q/k/v/o proj, gate/up/down proj
    │
    ▼
SFT Training (TRL SFTTrainer)
    │
    ▼
Adapter Serialization  ──►  Hugging Face Hub push
    │
    ▼
Evaluation  ──►  BLEU, keyword accuracy, syntax validity, Pass@k
```

### 3.2 LoRA Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Rank r | 16 | Balances expressivity and memory |
| Alpha α | 32 | 2× rank — standard effective LR scaling |
| Dropout | 0.05 | Light regularization |
| Bias | none | Standard practice |
| Target modules | q/k/v/o\_proj, gate/up/down\_proj | Full attention + MLP coverage |

### 3.3 Quantization Configuration

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

### 3.4 Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 2 × 10⁻⁴ |
| LR scheduler | Cosine |
| Warmup ratio | 3% |
| Per-device batch size | 4 |
| Gradient accumulation | 4 steps (effective batch = 16) |
| Optimizer | Paged AdamW 8-bit |
| Epochs | 3 (experiments) |
| Max sequence length | 2048 |
| Gradient checkpointing | Enabled |
| Mixed precision | FP16 |

### 3.5 Prompt Template

All samples are formatted as Alpaca-style instruction triplets:

```
### Instruction:
<task description>

### Input:
<optional context>

### Response:
<target code>
```

During generation, the model receives everything up to `### Response:` and autoregressively produces the code.

### 3.6 Evaluation Metrics

**BLEU** — n-gram overlap (1–4 grams) with brevity penalty against reference implementations.

**Keyword Accuracy** — percentage of domain-specific identifiers present in generated output (e.g., `@RestController`, `@Transactional`, `ResponseEntity`, `@GetMapping`). Measures domain knowledge acquisition independently of exact wording.

**Syntax Validity** — lightweight structural checker: balanced braces/parentheses, presence of class/method scaffolding, valid annotation syntax.

**Pass@k** — fraction of benchmark problems solved correctly within k independent samples. Used for standardized comparison with published code benchmarks.

---

## 4. Experimental Setup

### 4.1 Dataset

| Property | Value |
|----------|-------|
| Source | `iamtarun/python_code_instructions_18k_alpaca` (HF Hub) |
| Total samples | 18 000 instruction–code pairs |
| Used in experiments | 5 000 (compute-constrained) |
| Train / validation split | 90 / 10 |
| Domain shift target | Java / Spring Boot |

### 4.2 Base Model

**bigcode/starcoder2-3b** — a 3B-parameter decoder-only model pre-trained on The Stack v2 across 600+ programming languages. Selected for:
- Small enough to train on T4 (16 GB VRAM) with QLoRA
- Strong code understanding baseline
- Permissive BigCode OpenRAIL-M license

Other models supported by the pipeline:

| Model | Params | Notes |
|-------|--------|-------|
| CodeLlama-7B | 7B | Strong general code understanding |
| CodeLlama-7B-Instruct | 7B | Pre-instruction-tuned; recommended baseline |
| DeepSeek-Coder-6.7B | 6.7B | State-of-the-art on HumanEval as of 2024 |

### 4.3 Compute

| Resource | Specification |
|----------|---------------|
| GPU | NVIDIA T4 (16 GB VRAM) — Google Colab free tier |
| RAM | 12 GB |
| Training time | ~25–30 minutes |
| Trainable parameters | ~4.2 M (0.14% of 3B) |
| Throughput | ~15 samples/sec |

---

## 5. Results and Analysis

### 5.1 Training Dynamics

| Metric | Value |
|--------|-------|
| Initial training loss | ~2.8 |
| Final training loss (epoch 3) | ~1.2 |
| Trainable parameters | 4 194 304 (0.14%) |
| Frozen parameters | 2 995 748 864 (99.86%) |

The loss curve shows healthy descent, confirming that gradient flow through quantized weights via LoRA adapters functions correctly even at 4-bit precision.

### 5.2 Generation Quality

The 5 000-sample, 3-epoch experiment intentionally operates **below the quality threshold** for deployable results. Generated outputs exhibit:
- Partially coherent Java-like syntax
- Correct high-level structure (class declaration, method signatures)
- Unreliable annotation placement and logic correctness

This is expected and pedagogically valuable: it demonstrates that parameter-efficient methods eliminate the **memory barrier** but not the **data and compute requirements** for convergence.

### 5.3 Memory Footprint Comparison

| Configuration | VRAM (7B model) |
|---------------|-----------------|
| Full fine-tuning FP32 | ~112 GB |
| Full fine-tuning FP16 | ~56 GB |
| LoRA FP16 (frozen base) | ~14 GB |
| **QLoRA NF4 + LoRA** | **~4–5 GB** |

For StarCoder2-3B specifically, QLoRA brings peak VRAM to ~3.2 GB during inference and ~4.8 GB during training — fitting comfortably on a T4.

---

## 6. Scaling Recommendations

| Configuration | Samples | Epochs | GPU | Est. Time | Expected Quality |
|---------------|---------|--------|-----|-----------|------------------|
| Minimal (this work) | 5 000 | 3 | T4 16 GB | 30 min | Proof-of-concept |
| Viable | 20 000+ | 5–10 | A10 24 GB | 2–4 h | Usable outputs |
| Production | 100 000+ | 10–20 | A100 40 GB | 10–20 h | Deployment-ready |

**Recommended path for practitioners:**
1. Start with `CodeLlama-7B-Instruct` — pre-instruction-tuning reduces data needed.
2. Curate 20 000+ high-quality, domain-specific samples (filter for compilable code).
3. Train for 5–10 epochs with early stopping on eval loss.
4. Evaluate on a held-out benchmark before publishing adapters.

---

## 7. Artifacts and Reproducibility

### Trained Adapters

LoRA adapters are published on Hugging Face and can be loaded with two lines:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("bigcode/starcoder2-3b")
model = PeftModel.from_pretrained(base, "fabiofasc46/code-generator-lora")
```

Adapter size: ~4–5 MB (vs. 6 GB for the full base model).

### Colab Notebook

A ready-to-run Google Colab notebook (`notebooks/colab_training.ipynb`) installs dependencies, downloads the dataset, runs QLoRA fine-tuning on T4, and evaluates the result — end to end in under 45 minutes on free-tier hardware.

### Interactive Demo

A Gradio interface demonstrates the fine-tuned model with six pre-configured prompts covering:
- REST controller generation
- JPA repository scaffolding
- Spring Security JWT configuration
- Singleton and DTO patterns
- Exception handler boilerplate

### Project Structure

```
.
├── src/
│   ├── data/          # Dataset loading and instruction formatting
│   ├── training/      # LoRA trainer, QLoRA config, SFTTrainer wrapper
│   ├── inference/     # Generation engine with custom stopping criteria
│   └── evaluation/    # BLEU, keyword accuracy, syntax validity
├── scripts/
│   ├── train.py       # Entry point: loads config, runs training
│   └── evaluate.py    # Runs benchmark suite on a trained adapter
├── config/
│   └── training_config.yaml   # All hyperparameters in one place
├── demo/
│   └── app.py         # Gradio web interface
├── notebooks/
│   └── colab_training.ipynb
└── docs/
    └── paper.md       # Extended methodology document
```

---

## 8. Limitations

**Experimental scope.** Results reported here use 5 000 training samples and 3 epochs — sufficient to validate the pipeline but insufficient for production deployment. Keyword accuracy and syntax validity scores are preliminary.

**Domain shift.** The training dataset (`python_code_instructions_18k_alpaca`) targets Python, while evaluation benchmarks target Java/Spring Boot. A dataset curated specifically for the target domain would yield significantly better results.

**Evaluation metrics.** BLEU is a known imperfect proxy for code quality — a functionally correct solution with different variable names scores poorly. Pass@k on a formal benchmark (HumanEval, MBPP) would be more meaningful and is left for future work.

**Quantization accuracy loss.** 4-bit NF4 introduces a small but non-zero precision penalty versus FP16. For tasks requiring exact arithmetic or rare token sequences, this may degrade output quality.

---

## 9. Related Work

| Work | Contribution |
|------|-------------|
| Hu et al., 2022 — *LoRA* | Introduced low-rank adapter fine-tuning for LLMs |
| Dettmers et al., 2023 — *QLoRA* | Combined 4-bit quantization with LoRA for memory-efficient fine-tuning |
| Rozière et al., 2023 — *CodeLlama* | Code-specialized LLM family with instruction-tuned variants |
| Li et al., 2023 — *StarCoder / StarCoder2* | Open code LLMs trained on The Stack |
| Guo et al., 2024 — *DeepSeek-Coder* | SOTA code generation with fill-in-the-middle pre-training |
| Chen et al., 2021 — *HumanEval* | Standard pass@k evaluation benchmark for code LLMs |
| Wei et al., 2022 — *Instruction Tuning* | Foundation for the Alpaca-style training format used here |

---

## 10. Citation

If you find this work useful, please cite:

```bibtex
@misc{fasciani2026codelora,
  author       = {Fasciglione, Fabio},
  title        = {Parameter-Efficient Fine-Tuning of Code LLMs via LoRA and QLoRA},
  year         = {2026}
}
```

---

## License

This work is released under the [MIT License](LICENSE).  
Base model weights are subject to the [BigCode OpenRAIL-M License](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement).
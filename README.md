# Failure Modes of Small Quantized LLMs in Financial Document Information Extraction

**A local, GDPR-compliant pipeline for structured extraction from receipts — and a failure taxonomy of where small models break.**

BSc Thesis · ELTE Faculty of Informatics · Final examination January 2027 · Supervisor: Md. Easin Arafat

---

## Overview

A fully local, GPU-free pipeline that extracts structured JSON
(`company`, `date`, `address`, `total`) from receipt images: an **OCR engine**
(PaddleOCR or Tesseract) feeds a **small quantized LLM** (Phi-3 Mini,
Qwen2.5-3B, or Llama-3.2-3B — Q4 GGUF, on CPU), which returns validated JSON.
A compact **OCR-free vision-language model** (Qwen2.5-VL-3B) that reads the
image end-to-end is included as an ablation.

Everything runs on a laptop CPU, and nothing leaves the machine — the GDPR
angle, demonstrated in the UI. The **main contribution is a quantified failure
taxonomy**: every wrong field is attributed to the first stage that failed
(OCR vs the model) and a finer cause, across the full grid of 3 models × 2 OCR
engines × 2 datasets (SROIE, CORD).

## Key results at a glance

Three small models, same prompt and data, OCR fixed to PaddleOCR
(normalised accuracy; full tables in **[docs/results.md](docs/results.md)**):

| Model | SROIE all-4 fields | CORD total | Valid JSON | LLM s/page |
|---|---|---|---|---|
| Phi-3 Mini (3.8B) | 26% | 96% | 96% | 44 s |
| **Qwen2.5-3B** | **44%** | **100%** | 100% | 25 s |
| Llama-3.2-3B (3B) | 32% | 78% | 100% | 23 s |

**Headline finding:** the smallest model, **Qwen2.5-3B, is the most accurate
and ~2× faster — model family matters more than size** at this scale. (Two
more: JSON *derailment* is a Phi-3-specific failure mode, and reading the image
directly with the VLM beats OCR on messy receipts but loses on clean ones. See
[docs/results.md](docs/results.md).)

## Quick start

Laptop CPU, no GPU. Developed on Windows 11 / Python 3.12; Linux and macOS
steps are in the setup guide.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# download Phi-3 Q4 GGUF into models/ — see docs/setup.md
python scripts/check_env.py                          # verifies the full stack

python -m src.run evaluation/datasets/SROIE/raw/test/X00016469670.jpg   # one receipt
streamlit run app/app.py                             # or the interactive UI
```

Full install for **Windows, Linux and macOS**, model downloads and the test
setup: **[docs/setup.md](docs/setup.md)**.

## How it works

Two stages with a clear seam between them, so failures can be attributed:

1. **OCR** turns the image into reading-order text, keeping per-piece boxes and
   confidences and a map from each text row back to its pixels.
2. **The LLM** reads that text (2-shot prompt) and returns validated JSON; a
   lenient parser records every format problem.

The seam is the point: because the OCR text is kept, a wrong field can be
traced to the OCR stage or the model — which is what the failure taxonomy does.
The OCR-free VLM collapses both stages into one and trades that
interpretability for accuracy on messy images.

Design, data flow, formats and decisions: **[docs/architecture.md](docs/architecture.md)**.
How to run each stage: **[docs/usage.md](docs/usage.md)**.

## Repo structure

- `src/ocr/` — OCR engines (PaddleOCR, Tesseract) behind one interface, reading-order text, run logs
- `src/extraction/` — LLM prompt templates, JSON schema, extraction logic
- `src/vlm.py` — OCR-free VLM ablation (image → JSON, end-to-end)
- `src/run.py` — the pipeline command (single receipt or batch)
- `evaluation/` — dataset prep, metrics, and the **failure taxonomy**
- `app/` — Streamlit UI (`app.py` renders; `service.py` holds the testable logic)
- `tests/` — unit / integration tests (no models or OCR libraries needed)
- `docs/` — documentation (below)

## Documentation

| Page | What's in it |
|---|---|
| [docs/setup.md](docs/setup.md) | Install, Windows notes, model downloads, environment check, tests |
| [docs/usage.md](docs/usage.md) | Running each stage: OCR, extraction, pipeline, UI, evaluation |
| [docs/results.md](docs/results.md) | Full result tables, LLM comparison, VLM ablation, reproducibility |
| [docs/architecture.md](docs/architecture.md) | As-built design, data-flow diagram, formats, decisions |
| [docs/testing.md](docs/testing.md) | Test plan, test levels, coverage |
| [docs/milestones.md](docs/milestones.md) | Milestone tracking |
| [evaluation/datasets/README.md](evaluation/datasets/README.md) | Dataset sources, counts, subsets |
| [taxonomy_v1.md](evaluation/failure_taxonomy/taxonomy_v1.md) | The cross-model failure taxonomy write-up |

## Status

- **M1** ✅ Topic, background research, design diagram, repo setup.
- **M2** ✅ End-to-end pipeline on a laptop CPU: OCR → LLM → validated JSON,
  dataset preparation (SROIE, CORD), batch runner, evaluator and baseline
  results, tests + CI, [architecture docs](docs/architecture.md).
- **M3** (in progress) — done: failure taxonomy v0 and **v1** (cross-model,
  50-case manual review), the three-model comparison, the OCR-free VLM
  ablation, the Streamlit UI, testing documentation, and the **prompt and
  few-shot experiments** (issue #24 — v1 vs v2, 2-shot vs 0-shot; both prompts
  kept with v1 as default, see [docs/results.md](docs/results.md)). Remaining:
  result charts and the thesis chapter drafts.

Details: [docs/milestones.md](docs/milestones.md) and the repo's
Issues / Milestones tabs.

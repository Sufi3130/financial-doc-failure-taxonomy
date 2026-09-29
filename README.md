# Failure Modes of Small Quantized LLMs in Financial Document Information Extraction: A Local, GDPR-Compliant Pipeline

BSc Thesis, ELTE Faculty of Informatics — Final Examination January 2027
Supervisor: Arafat

## Overview

A fully local, GPU-free pipeline for extracting structured data (JSON) from
financial documents (invoices, bank statements, payslips), using OCR engines
(PaddleOCR, TrOCR) feeding a small quantized LLM (Phi-3 Mini / Mistral 7B GGUF).
Benchmarked against a compact OCR-free vision-language model (PaddleOCR-VL) as
an ablation. The primary contribution is a structured failure taxonomy for
financial document extraction using small local models.

## Repo structure

- `docs/` — proposal, design diagrams, literature notes, milestone tracking
- `src/ocr/` — OCR engine wrappers (PaddleOCR, TrOCR)
- `src/extraction/` — LLM prompt templates, JSON schema, extraction logic
- `src/vlm_ablation/` — OCR-free VLM comparison pipeline
- `evaluation/` — dataset prep, metrics, failure taxonomy annotations
- `app/` — Streamlit demo application
- `tests/` — unit/integration tests

## Setup

The core stack runs on a laptop CPU with no GPU. Tested on Windows 11 with
Python 3.12.6.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-later.txt   # optional: TrOCR, scikit-learn, Streamlit (M3+)
```

### Windows setup notes

- **Use the python.org CPython build** (3.12 recommended). MSYS2/MinGW or
  Cygwin Pythons are not supported: `paddlepaddle` and `llama-cpp-python` do
  not publish wheels for them.
- **llama-cpp-python without a compiler:** `requirements.txt` points pip at the
  prebuilt CPU wheel index
  (`--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`), so
  a `win_amd64` wheel is installed and nothing is compiled. If pip falls back
  to building from source (e.g. you changed the pinned version and no wheel
  exists for it), install *Visual Studio Build Tools* with the "Desktop
  development with C++" workload plus CMake, then reinstall.
- **paddlepaddle** is the CPU build from PyPI. Do not install `paddlepaddle-gpu`.
- **PaddleOCR on Windows CPU:** the pipeline creates PaddleOCR with
  `enable_mkldnn=False`, which avoids a oneDNN crash in Paddle 3.x on Windows.
- PowerShell may block `Activate.ps1`. Fix it with
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- On first run, PaddleOCR downloads its detection and recognition models to
  `~/.paddlex/` (needs internet once).

### Download the model

Phi-3-mini-4k-instruct, Q4 GGUF (~2.2 GB), into the gitignored `models/` folder:

```bash
mkdir models
curl -L -o models/Phi-3-mini-4k-instruct-q4.gguf \
  https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
```

### Check the environment

```bash
python scripts/check_env.py
```

The script imports all core libraries, OCRs one SROIE receipt, runs one short
generation with the GGUF model and prints timings. It ends with
`[OK] environment check passed`. Datasets go under
`evaluation/datasets/<name>/raw/` (see
[evaluation/datasets/README.md](evaluation/datasets/README.md)). Raw data and
model files are gitignored.

## Key Results

_Not yet available — to be filled in once evaluation on SROIE/CORD is
complete (target: Milestone 2)._

| Metric | OCR+LLM pipeline | OCR-free VLM ablation |
|---|---|---|
| Field-level accuracy | TBD | TBD |
| Schema compliance rate | TBD | TBD |
| Avg. latency / page (CPU) | TBD | TBD |

## Status

See [docs/milestones.md](docs/milestones.md) and the repo's Issues/Milestones tabs for current progress.

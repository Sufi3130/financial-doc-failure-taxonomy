# Failure Modes of Small Quantized LLMs in Financial Document Information Extraction: A Local, GDPR-Compliant Pipeline

BSc Thesis, ELTE Faculty of Informatics — Final Examination January 2027
Supervisor: Arafat

## Overview

A fully local, GPU-free pipeline for extracting structured data (JSON) from
financial documents (invoices, bank statements, payslips), using OCR engines
(PaddleOCR, Tesseract) feeding a small quantized LLM (Phi-3 Mini / Mistral 7B GGUF).
Benchmarked against a compact OCR-free vision-language model (PaddleOCR-VL) as
an ablation. The primary contribution is a structured failure taxonomy for
financial document extraction using small local models.

## Repo structure

- `docs/` — proposal, design diagrams, literature notes, milestone tracking
- `src/ocr/` — OCR engines (PaddleOCR, Tesseract) behind one interface, reading-order text, run logs
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
pip install -r requirements-later.txt   # optional: scikit-learn, Streamlit (M3+)
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
- **Tesseract** is a separate program, not a pip package (`pytesseract` only
  calls it). Install Tesseract 5 (UB Mannheim build, includes English):
  `winget install --id UB-Mannheim.TesseractOCR -e` (tested: 5.4.0). The
  installer does not add it to PATH. The code finds it in
  `C:\Program Files\Tesseract-OCR\` automatically, or set `TESSERACT_CMD` to
  the full path of `tesseract.exe`. On Linux: `apt install tesseract-ocr`.

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

The script imports all core libraries, OCRs one SROIE receipt with both OCR
engines, runs one short generation with the GGUF model and prints timings. It ends with
`[OK] environment check passed`. Datasets go under
`evaluation/datasets/<name>/raw/` (see
[evaluation/datasets/README.md](evaluation/datasets/README.md)). Raw data and
model files are gitignored.

## OCR stage

`src/ocr/` gives every OCR engine the same interface:
`engine.recognize(image_path)` returns an `OCRResult`. It contains one
`OCRLine` per text piece, with `text`, `box` (x1, y1, x2, y2), `polygon`,
`confidence` (0–1 for both engines) and, for Tesseract, word-level detail.
It also records the timings (`load_s`, `ocr_s`) and the engine settings.

| Engine | Name | Notes |
|---|---|---|
| PaddleOCR 3.7 (PP-OCRv6 det + rec) | `paddleocr` | CPU, `text_rec_score_thresh=0` so no line is dropped for low confidence |
| Tesseract 5.4 (LSTM) | `tesseract` | `--oem 1 --psm 4` (single column of variable-size text), grayscale input only (no scaling or binarisation) |

```bash
python -m src.ocr evaluation/datasets/SROIE/raw/test/X00016469670.jpg              # both engines
python -m src.ocr evaluation/datasets/SROIE/raw/test/X00016469670.jpg --engine tesseract
```

```python
from src.ocr import get_engine, reading_order
result = get_engine("paddleocr").recognize(path)
prompt_text = reading_order(result.lines).text
```

- **Reading order** (`src/ocr/layout.py`): engines often split one receipt
  row into pieces (`TOTAL:` and `193.00`). `reading_order` groups pieces that
  overlap vertically into rows, sorts each row left to right and joins them
  with a space. Its `rows` field maps each prompt line back to the OCR pieces
  it came from, so a wrong extraction can be traced to either the OCR or the
  LLM.
- **Run log** (`src/ocr/runlog.py`): `save_run_log` writes the full result
  (all pieces, boxes, confidences, reading-order text and row map, timings,
  settings, versions) to `runs/ocr/<timestamp>/<receipt>.<engine>.json`
  (gitignored).
- **Adding an engine:** subclass `OCREngine` (`src/ocr/base.py`), implement
  `_load`, `_predict`, `version` and `settings`, report confidence in 0–1 and
  pixel boxes on the original image, then register it in `ENGINES` in
  `src/ocr/__init__.py`.

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

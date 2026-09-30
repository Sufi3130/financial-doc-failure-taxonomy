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
- **OCR cache:** `cached_ocr()` saves each receipt's run log under
  `runs/cache/ocr/<engine>/` and reuses it while the engine settings are
  unchanged. Delete `runs/cache/` after upgrading an engine.

## Extraction stage

`src/extraction/` turns reading-order OCR text into a validated `Receipt`
(`company`, `date`, `address`, `total`: each a string or `null`) with Phi-3
Mini Q4 via llama-cpp-python. It runs on CPU with `temperature=0`, a fixed
seed, `n_ctx=4096` and `max_tokens=256`.

It is run through the pipeline command below.

- **Prompt** (`prompts.py`, version recorded in every log): instructions plus
  few-shot examples, then the target receipt. Phi-3's GGUF chat template
  has no system role and silently drops `system` messages, so the
  instructions are sent as part of the first user turn.
- **Few-shot examples** (`fewshot.py`): 2 by default (`--shots 0` for
  zero-shot), picked with a seed from **SROIE train only**. Excluded: train
  receipts with label issues, duplicates, and the 7 byte-identical to a test
  receipt. The same receipts are used for every OCR engine. Their OCR text
  comes from the same engine as the target, and their answers are the raw
  ground-truth strings. Requires the SROIE manifest (`prepare.py`).
- **No constrained decoding:** the model is free to produce invalid output,
  because output-format failures are part of what is measured.
- **Parsing** (`parse.py`): lenient. It strips code fences and surrounding
  text, then fixes trailing commas and Python-style quotes, and records each
  repair. The result is validated against the strict schema. Every problem is
  recorded as a failure: `prompt_too_long`, `truncated`, `no_json`,
  `invalid_json`, `not_object`, `missing_field`, `null_field`, `wrong_type`,
  `extra_field`.
  - `schema_valid`: the parsed object fits the schema.
  - `strict_valid`: it also needed no repairs and wasn't cut off.
  - `lenient`: the best usable values for scoring (numbers turned into
    strings, extra keys dropped).

## Pipeline: one command, single receipt or batch

`src/run.py` runs image → OCR → LLM → validated JSON:

```bash
python -m src.run evaluation/datasets/SROIE/raw/test/X00016469670.jpg            # one receipt
python -m src.run --manifest evaluation/datasets/SROIE/subset_test_50.jsonl --engine all
python -m src.run --manifest ... --resume runs/pipeline/<run>                   # continue a run
```

Options: `--engine paddleocr|tesseract|all` (default `paddleocr`),
`--shots N` (default 2), `--seed`, `--limit N`, `--no-cache`.

- **Run log:** `runs/pipeline/<timestamp>/run.jsonl` (gitignored). Each
  receipt is appended as one line as soon as it finishes, so an interrupted
  run loses nothing. A line holds:
  - `receipt_id`, `split`, `engine`, `image`
  - `ocr`: reading-order text, row map and all pieces with boxes and confidences
  - `llm`: raw output, `finish_reason`, token counts
  - `outcome`: repairs, failures, `schema_valid` / `strict_valid`, lenient values
  - `timings` and `error`

  Ground truth is not copied; the evaluator joins on `receipt_id`.
- **`run_meta.json`** next to it holds:
  - the command, git commit and uncommitted files
  - library versions, CPU, model file and settings
  - prompt version, few-shot IDs and the exact prompt prefix for each engine
  - OCR settings, the warm-up and a summary (validity counts, failure and repair
    counts, mean/median timings)
- **Timings per page (CPU):** `ocr_s`, `llm_s` (split into `llm_prompt_s`
  and `llm_gen_s` from llama.cpp's counters), `parse_s`, `total_s` =
  OCR + LLM + parse, and `wall_s`.
  - The model is loaded once. Before each engine's receipts, a warm-up call
    reads the fixed prompt prefix (instructions + few-shot examples), so every
    page is timed warm. `llm.prompt_tokens_evaluated` shows how many prompt
    tokens were actually read.
  - With the OCR cache (default), a cached receipt's `ocr_s` is the time
    measured when its OCR really ran (`ocr_cached: true`).
  - **Measure on AC power.** Laptops throttle the CPU on battery. Each line
    records `power` (`ac` / `battery`), and the command warns when it starts
    on battery. While a run is in progress the command stops Windows from
    sleeping on idle (`SetThreadExecutionState`). Closing the lid still
    suspends the laptop.
- **Batch behaviour:**
  - Engines run one after another (all receipts per engine) so the prompt
    prefix stays reusable.
  - An error on one receipt is logged (`stage`, `type`, `message`) and the
    batch continues.
  - `--resume` skips receipts already logged without an error, and retries
    failed ones by appending a new line. Use the last line per
    engine + receipt.

## Evaluation

```bash
python evaluation/evaluate.py runs/pipeline/<run> --manifest evaluation/datasets/SROIE/subset_test_50.jsonl
```

`evaluation/evaluate.py` scores a run log against a manifest. It uses the
last row per engine and receipt, and **every receipt in the manifest counts**:
a failed, empty or missing output counts as wrong. Only annotated fields are
scored (CORD: `total` only).

- **Strict match:** the prediction equals the label exactly (after trimming).
- **Normalised match:**
  - date and total go through `evaluation/normalize.py`, using the record's
    locale;
  - company and address ignore case, spaces and punctuation.
- **Token F1** (company, address): partial credit for boundary errors.
- **Value in OCR text:** splits misses into OCR misses (the value was never
  read) and LLM misses (it was read but not extracted correctly).
- **Schema compliance:** `strict_valid` = clean JSON that fits the schema
  with no repairs; `schema_valid` = fits the schema after lenient repairs.
- **Latency:** seconds per page, median and mean.

Results go to `evaluation/results/<run_id>/`:

- `summary.json`: all metrics
- `per_receipt.csv`: match flags, failure types and timings per receipt (no
  text)
- `per_receipt_detail.jsonl`: the same plus predicted and ground-truth text.
  This one is gitignored because it contains labels.

## Key Results

**Setup:**

- Phi-3-mini-4k-instruct Q4 GGUF (llama-cpp-python, CPU), prompt `v1`,
  2-shot (SROIE train `X51005676545`, `X51005361946`; few-shot seed 42),
  temperature 0.
- **SROIE:** 50-receipt test subset `subset_test_50` (seed 42), run
  `20260930-063807`.
- **CORD v2:** 50-receipt validation + test subset `subset_valtest_50`
  (seed 42), run `20260930-092942`. Same prompt and examples; only `total` is
  scored, normalised as rupiah.
- Both runs: commit `2e9c77a`, Intel 4-core/8-thread laptop CPU on AC power,
  fresh OCR (no cache).

**SROIE Task 3**: normalised match % (strict match % in brackets)

| Metric | PaddleOCR + Phi-3 | Tesseract + Phi-3 | OCR-free VLM ablation |
|---|---|---|---|
| Company | 44.0 (36.0) | 44.0 (38.0) | TBD |
| Date | 82.0 (70.0) | 70.0 (46.0) | TBD |
| Address | 42.0 (6.0) | 34.0 (8.0) | TBD |
| Total | 92.0 (76.0) | 74.0 (60.0) | TBD |
| All 4 fields correct | 26.0 | 18.0 | TBD |
| Token F1 company / address | 62.4 / 68.4 | 70.6 / 77.6 | TBD |
| Schema compliance rate (strict-valid) | 96.0 % | 96.0 % | TBD |
| Latency / page (CPU), median (mean) | 121.8 s (150.0 s) | 51.0 s (52.3 s) | TBD |
| ↳ OCR / LLM, median | 74.0 s / 43.9 s | 1.0 s / 49.9 s | TBD |

**CORD v2 total**: normalised match % (strict match % in brackets)

| Metric | PaddleOCR + Phi-3 | Tesseract + Phi-3 | OCR-free VLM ablation |
|---|---|---|---|
| Total | 84.0 (76.0) | 14.0 (12.0) | TBD |
| Schema compliance rate (strict-valid) | 100.0 % | 68.0 % | TBD |
| Latency / page (CPU), median (mean) | 45.1 s (64.5 s) | 11.8 s (15.9 s) | TBD |

**Notes:**

- The subsets are small (50 receipts), so each receipt is 2 percentage points.
- Labels are used as released, with no manual corrections. Some SROIE labels
  contain typos (e.g. a postcode `B1750` for `81750`), and 5 duplicate-image
  groups have conflicting labels, so a perfect extractor would not reach
  100 %.
- The strict address score is low mainly because of comma, spacing and
  boundary differences. Normalised match and token F1 show how much of the
  address was right.
- Latency is measured with the model already loaded and the few-shot prompt
  prefix already read. It excludes model load (~4 s) and the one-off warm-up
  per engine (~70–100 s).
  - 9 of the 50 SROIE receipts are 35 MP scans, which take ~240–315 s of
    PaddleOCR each; that's why the mean is well above the median.
  - CORD prompts are shorter, so the LLM is faster there.
- **Reproducibility:** an earlier run of the same SROIE subset
  (`20260930-030537`) produced identical OCR text and identical raw LLM
  output for all 100 receipt × engine runs. Only its timings differed: that
  run was partly on battery, and the laptop slept once.
- Tesseract fails badly on CORD (camera photos): the total is in its OCR text
  for only 34 % of receipts, and on 12 receipts it produced no JSON.
- Per-receipt results: [`evaluation/results/`](evaluation/results/).

## Status

See [docs/milestones.md](docs/milestones.md) and the repo's Issues/Milestones tabs for current progress.

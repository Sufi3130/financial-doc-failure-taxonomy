# Usage — running each stage

This is the hands-on guide to each stage and command. For the as-built design,
data flow and formats see [architecture.md](architecture.md); for setup see
[setup.md](setup.md).

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
(`company`, `date`, `address`, `total`: each a string or `null`) with a small
quantized LLM via llama-cpp-python. It runs on CPU with `temperature=0`, a
fixed seed, `n_ctx=4096` and `max_tokens=256`. It is run through the pipeline
command below.

- **Models compared** (`--model`, registry in `llm.py`): the same prompt is
  run across several small Q4 GGUF models to study failure modes per model.
  - `phi3`: Phi-3-mini-4k-instruct, 3.8B (Microsoft, MIT) — the default.
  - `qwen2.5-3b`: Qwen2.5-3B-Instruct (Alibaba, Apache 2.0, multilingual).
  - `llama3.2-3b`: Llama-3.2-3B-Instruct (Meta, Llama 3.2 licence).

  `--model` takes a registry name or a path to a `.gguf`. Each run records
  the model in `run_meta.json`. Download the GGUFs into `models/` (gitignored).
- **Prompt** (`prompts.py`, version recorded in every log): instructions plus
  few-shot examples, then the target receipt. The instructions are sent as
  part of the first user turn, not a `system` message (Phi-3's GGUF chat
  template drops `system` messages; folding into the user turn works for every
  model, keeping the prompt identical across models for a fair comparison).
- **Few-shot examples** (`fewshot.py`): 2 by default (`--shots 0` for
  zero-shot), picked with a seed from the **train split of the dataset being
  evaluated**:
  - **SROIE:** SROIE train. Excluded: train receipts with label issues,
    duplicates, and the 7 byte-identical to a test receipt.
  - **CORD:** a pool of 20 CORD train receipts written by `CORD/prepare.py`
    (`fewshot_pool_ids.txt`). Their answers have `null` company/date/address,
    because CORD doesn't annotate them.

  The pool is chosen from the manifest's dataset folder; `--fewshot-pool`
  overrides it. The same receipts are used for every OCR engine. Their OCR
  text comes from the same engine as the target, and their answers are the
  raw ground-truth strings.
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

## The UI

A local Streamlit app wraps the same pipeline for interactive use. Streamlit is
in `requirements.txt`, so after setup just run (from the repo root):

```bash
streamlit run app/app.py          # opens http://localhost:8501
```

It runs **entirely on this machine** — no image or label leaves it. The server
binds to `localhost`, keeps CORS/XSRF protection on and Streamlit's telemetry off
(`.streamlit/config.toml`); uploads are processed in a temporary folder. This is
the project's GDPR point, demonstrated. A left sidebar navigates three tools plus
a home page (the current page is in the URL, so refresh and Back/Forward work):

- **Extract & Explain** — one receipt (upload an image, or pick a dataset one).
  Choose the OCR engine + LLM, or the **OCR-free VLM**. A staged progress bar
  shows each phase (load → OCR → model → parse). Shows the image with OCR boxes
  (coloured by confidence), the reading-order text, the raw model output, the
  validated JSON, the timings, and — with a label (a dataset receipt, or values
  you type for an upload) — a **per-field comparison**, the stage that failed
  (OCR vs LLM), and a plain-language **reason for each wrong field**. Without a
  label it flags **hallucinations** (values not present in the OCR text).
- **Compare models** — run any mix of **Phi-3 / Qwen2.5-3B / Llama-3.2-3B / the
  OCR-free VLM** on one receipt, side by side: each model's four fields,
  correct/wrong (with a label), JSON validity, failure categories and timing.
- **Dataset Evaluation** — a single image, a built-in 50-receipt subset, or
  **your own labelled dataset**. For an upload, drop a ZIP of images plus a
  `labels.csv`:

  ```text
  images/            the receipt images
  labels.csv         columns: id, image, company, date, address, total
  ```

  Any field may be blank (only filled fields are scored; `labels.jsonl` also
  works). The engine list includes the **OCR-free VLM**. The app runs the whole
  pipeline, then shows accuracy with 95% confidence intervals, the
  failure-taxonomy counts, and a per-receipt table you can **filter by failure
  category** and drill into. Runs are slow on CPU (~1–4 min per uncached receipt;
  the VLM ~3–4 min).

The UI logic lives in `app/service.py`, kept free of Streamlit so it is
unit-tested (`tests/test_app.py`); `app/app.py` only renders.

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
- `per_receipt.csv`: match flags, failure types and timings per receipt (no text)
- `per_receipt_detail.jsonl`: the same plus predicted and ground-truth text.
  This one is gitignored because it contains labels.

## Failure taxonomy

The failure taxonomy is the project's main contribution: every wrong field is
tagged with the first pipeline stage that failed and a finer category.

```bash
python evaluation/failure_taxonomy/tag.py --run runs/pipeline/<run> <subset.jsonl>
python evaluation/failure_taxonomy/by_model.py        # cross-model category counts
```

See [results.md](results.md) for the findings and
`evaluation/failure_taxonomy/taxonomy_v1.md` for the full write-up.

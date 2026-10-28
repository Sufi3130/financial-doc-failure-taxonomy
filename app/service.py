"""Logic for the Streamlit UI, kept free of Streamlit so it can be unit-tested.

Everything here is a thin wrapper over the existing pipeline (`src/`), the
evaluator (`evaluation/evaluate.py`) and the failure tagger
(`evaluation/failure_taxonomy/tag.py`). No scoring or extraction logic is
duplicated: the UI shows exactly what a command-line run would produce.
"""

import csv
import io
import json
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from src import run as pipeline  # noqa: E402
from src.extraction import DEFAULT_MODEL_NAME, FIELDS, MODELS, LLMSettings, LocalLLM, resolve_model  # noqa: E402
from src.extraction.fewshot import load_examples, pool_for  # noqa: E402
from src.ocr import ENGINES, get_engine  # noqa: E402
from src.ocr.runlog import RUNS_DIR  # noqa: E402

import evaluation.evaluate as ev  # noqa: E402
import evaluation.failure_taxonomy.tag as tagger  # noqa: E402
from evaluation.normalize import normalize_date, normalize_total  # noqa: E402

# ------------------------------------------------------------------ choices
OCR_ENGINES = list(ENGINES)              # ["paddleocr", "tesseract"]
TEXT_MODELS = list(MODELS)               # ["phi3", "qwen2.5-3b", "llama3.2-3b"]
VLM_MODEL = "qwen2.5-vl-3b"              # OCR-free, image in
LOCALES = {"my": "Malaysian ringgit (SROIE)", "id": "Indonesian rupiah (CORD)"}
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024       # per-image guard (defence in depth)


# ------------------------------------------------------------------ loaders (cached by the caller)
def load_llm(model_name=DEFAULT_MODEL_NAME):
    return LocalLLM(LLMSettings(model_path=resolve_model(model_name)))


def load_engine(name):
    return get_engine(name)


def load_vlm(model_name=VLM_MODEL):
    from src.vlm import QwenVL
    return QwenVL(model=model_name)


# ------------------------------------------------------------------ records and ground truth
def ground_truth(raw, locale="my"):
    """Build a manifest ground_truth block from raw label values (same
    normalisation the dataset preparation and evaluator use)."""
    raw = {f: (str(raw.get(f)).strip() if raw.get(f) not in (None, "") else None) for f in FIELDS}
    norm = {}
    if raw.get("date"):
        norm["date"] = normalize_date(raw["date"])
    if raw.get("total"):
        norm["total"] = normalize_total(raw["total"], locale)
    return {"raw": raw, "normalized": norm}


def make_record(image_path, rid=None, split="test", locale="my", gold=None):
    """A manifest record the pipeline/evaluator/tagger all understand.
    `gold` is an optional dict of raw label values; omit it for a plain demo."""
    rec = {"id": rid or Path(image_path).stem, "split": split,
           "image": str(Path(image_path).resolve()), "locale": locale}
    if gold and any(gold.get(f) for f in FIELDS):
        rec["ground_truth"] = ground_truth(gold, locale)
        rec["fields_available"] = [f for f in FIELDS if rec["ground_truth"]["raw"].get(f)]
    return rec


def has_labels(record):
    return "ground_truth" in record


def dataset_label(record):
    return "CORD" if record.get("locale") == "id" else "SROIE"


# ------------------------------------------------------------------ few-shot
def fewshot(engine, shots=2, seed=42, pool=None):
    """(ids, examples) drawn from the train split of the chosen dataset."""
    if shots == 0:
        return [], []
    return load_examples(engine, k=shots, seed=seed, pool=pool or pool_for(None))


# ------------------------------------------------------------------ single extraction
def extract(record, engine, llm, examples=(), fewshot_ids=(), run_id="ui"):
    """One receipt through OCR -> LLM -> parse. Returns the same row a run logs."""
    return pipeline.process_receipt(record, engine, llm, list(examples), set(fewshot_ids), run_id)


def extract_vlm(record, vlm, run_id="ui"):
    """One receipt end-to-end through the vision-language model (no OCR stage)."""
    from src import vlm as vlm_mod
    return vlm_mod.process_receipt(record, vlm, run_id)


def score(record, row):
    """Per-field correct/wrong + OCR-vs-LLM attribution, or None without labels."""
    if not has_labels(record):
        return None
    return ev.score_receipt(record, row)


def tags(record, row, run_id="ui"):
    """Failure-taxonomy instances for this receipt, or [] without labels."""
    if not has_labels(record):
        return []
    return tagger.tag_receipt(record, row, run_id, dataset_label(record))


# ------------------------------------------------------------------ OCR box overlay
def confidence_color(conf):
    """Green (high) -> amber -> red (low), matching the thesis's confidence bins."""
    if conf is None or conf >= 0.9:
        return (34, 197, 94)
    if conf >= 0.7:
        return (234, 179, 8)
    return (239, 68, 68)


def draw_boxes(image_path, lines):
    """Return a copy of the image with each OCR piece's box drawn, coloured by
    confidence. `lines` is row['ocr']['lines'] (each has box + confidence)."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    width = max(2, img.width // 500)
    for ln in lines or []:
        box = ln.get("box")
        if not box:
            continue
        color = confidence_color(ln.get("confidence"))
        draw.rectangle(list(box), outline=color + (255,), width=width)
    return img


# ------------------------------------------------------------------ user-dataset ingestion (ZIP)
_LABEL_NAMES = {"labels.csv", "labels.jsonl"}


def _safe_name(name):
    """Flatten a ZIP entry to its base filename (defeats path traversal / zip-slip)."""
    return Path(name.replace("\\", "/")).name


def _resolve_image(image_field, rid, by_name):
    """Find the image file for a label row among the extracted images."""
    for cand in (image_field, rid):
        if not cand:
            continue
        base = _safe_name(cand)
        if base in by_name:
            return by_name[base]
        for suffix in ALLOWED_IMAGE_SUFFIXES:          # id given without extension
            if (base + suffix) in by_name:
                return by_name[base + suffix]
        stem = Path(base).stem                          # match by stem
        for n, p in by_name.items():
            if Path(n).stem == stem:
                return p
    return None


def _read_label_rows(data, label_name):
    text = data.decode("utf-8-sig")
    if label_name.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return list(csv.DictReader(io.StringIO(text)))


def ingest_zip(zip_bytes, dest_dir, locale="my", max_receipts=200):
    """Turn an uploaded ZIP (images + labels.csv/labels.jsonl) into manifest records.

    Returns (records, messages). `messages` holds any problems; records may be
    empty if the ZIP is unusable. Images are extracted flat into dest_dir/images
    (their internal paths are ignored, so a crafted ZIP cannot escape dest_dir).
    """
    dest = Path(dest_dir)
    images_dir = dest / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    messages, by_name = [], {}

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        return [], ["That file is not a valid ZIP archive."]

    with zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        label_name = next((n for n in names if _safe_name(n).lower() in _LABEL_NAMES), None)
        if not label_name:
            return [], ["No labels.csv (or labels.jsonl) found in the ZIP."]
        for n in names:
            if Path(_safe_name(n)).suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
                continue
            info = zf.getinfo(n)
            if info.file_size > MAX_IMAGE_BYTES:
                messages.append(f"skipped {_safe_name(n)}: larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
                continue
            base = _safe_name(n)
            (images_dir / base).write_bytes(zf.read(n))
            by_name[base] = images_dir / base
        label_rows = _read_label_rows(zf.read(label_name), _safe_name(label_name).lower())

    records = []
    for i, r in enumerate(label_rows, 1):
        r = {str(k).strip().lower(): v for k, v in r.items()}
        rid = (str(r.get("id") or "")).strip() or None
        image_field = (str(r.get("image") or "")).strip()
        image_path = _resolve_image(image_field, rid, by_name)
        if image_path is None:
            messages.append(f"row {i}: no image found for {image_field or rid or '(blank)'!r}")
            continue
        gold = {f: (str(r.get(f)).strip() if r.get(f) not in (None, "") else None) for f in FIELDS}
        records.append(make_record(image_path, rid=rid or image_path.stem, locale=locale, gold=gold))
        if len(records) >= max_receipts:
            messages.append(f"stopped at the {max_receipts}-receipt limit ({len(label_rows)} rows in the file)")
            break
    if not records:
        messages.append("No usable rows: check that labels reference the image filenames in the ZIP.")
    return records, messages


# ------------------------------------------------------------------ built-in datasets
def builtin_manifests():
    """Name -> manifest path for the 50-receipt test subsets shipped with the repo."""
    out = {}
    for name, rel in (("SROIE (50-receipt test subset)", "SROIE/subset_test_50.jsonl"),
                      ("CORD (50-receipt val+test subset)", "CORD/subset_valtest_50.jsonl")):
        path = REPO_ROOT / "evaluation" / "datasets" / rel
        if path.is_file():
            out[name] = path
    return out


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_manifest(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


# ------------------------------------------------------------------ batch run + evaluation
def new_run_dir():
    from datetime import datetime
    path = RUNS_DIR / "ui" / datetime.now().strftime("%Y%m%d-%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_batch(records, engine_name, llm, run_dir, shots=2, seed=42, use_cache=True, log=print):
    """Run the whole pipeline over `records` and write a standard run log.
    Returns (run_jsonl, manifest_path). Reuses src.run.run (warm-up, resume, meta)."""
    import functools

    run_dir = Path(run_dir)
    manifest_path = write_manifest(records, run_dir / "manifest.jsonl")
    engines = [get_engine(engine_name)]
    pool = pool_for(str(manifest_path))
    run_jsonl, _ = pipeline.run(
        records, engines, llm, run_dir, shots=shots, seed=seed, use_cache=use_cache,
        fewshot_loader=functools.partial(load_examples, pool=pool),
        meta={"source": "ui", "manifest": str(manifest_path)}, log=log)
    return run_jsonl, manifest_path


def evaluate_run(run_dir, manifest_path):
    """(records, per_receipt, summaries, tag_instances, tag_counts) for a finished run."""
    records, per_receipt, summaries = ev.evaluate(run_dir, manifest_path)
    instances = tagger.tag_run(run_dir, manifest_path)
    return records, per_receipt, summaries, instances, tagger.counts(instances)

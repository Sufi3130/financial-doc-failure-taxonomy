"""Save the full structured OCR result for later failure analysis."""

import json
import platform
from datetime import datetime
from pathlib import Path

from .base import OCRResult
from .layout import ReadingOrder

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


def new_run_dir(kind="ocr"):
    """runs/<kind>/<YYYYmmdd-HHMMSS>/ (gitignored)."""
    path = RUNS_DIR / kind / datetime.now().strftime("%Y%m%d-%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_run_log(result: OCRResult, reading: ReadingOrder, run_dir, receipt_id=None) -> Path:
    """Write <run_dir>/<receipt_id>.<engine>.json with every OCR piece (text,
    box, polygon, confidence, words), the reading-order text and its row map,
    timings, engine settings and versions."""
    receipt_id = receipt_id or Path(result.image_path).stem
    path = Path(run_dir) / f"{receipt_id}.{result.engine}.json"
    log = {
        "receipt_id": receipt_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ocr": result.model_dump(mode="json"),
        "reading_order": reading.model_dump(mode="json"),
    }
    path.write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def load_run_log(path):
    """Return (OCRResult, ReadingOrder) from a saved log."""
    log = json.loads(Path(path).read_text(encoding="utf-8"))
    return OCRResult.model_validate(log["ocr"]), ReadingOrder.model_validate(log["reading_order"])


CACHE_DIR = RUNS_DIR / "cache" / "ocr"


def cached_ocr(engine, image_path, receipt_id=None, cache_dir=CACHE_DIR):
    """OCR an image once per engine and settings; reuse the saved run log after that.

    Returns (OCRResult, ReadingOrder, log_path, from_cache). A cached log is
    reused only if it was made from the same image path with the same engine
    settings. Delete runs/cache/ after upgrading an OCR engine.
    """
    from .layout import reading_order

    image_path = Path(image_path)
    receipt_id = receipt_id or image_path.stem
    path = Path(cache_dir) / engine.name / f"{receipt_id}.{engine.name}.json"
    if path.is_file():
        result, reading = load_run_log(path)
        if result.settings == engine.settings() and Path(result.image_path).resolve() == image_path.resolve():
            return result, reading, path, True

    result = engine.recognize(image_path)
    reading = reading_order(result.lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    return result, reading, save_run_log(result, reading, path.parent, receipt_id), False

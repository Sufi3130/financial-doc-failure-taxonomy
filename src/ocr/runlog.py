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

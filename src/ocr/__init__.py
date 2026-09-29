"""OCR stage: image -> text pieces with boxes and confidence, plus reading-order text.

    from src.ocr import get_engine, reading_order
    engine = get_engine("paddleocr")          # or "tesseract"
    result = engine.recognize("receipt.jpg")  # OCRResult
    prompt_text = reading_order(result.lines).text

Engines import their heavy libraries only when first used.
"""

from .base import OCREngine, OCRLine, OCRResult, OCRWord
from .layout import ReadingOrder, reading_order
from .paddle import PaddleOCREngine
from .runlog import cached_ocr, load_run_log, new_run_dir, save_run_log
from .tesseract import TesseractEngine

# Extension point: register new OCREngine subclasses here (see base.py)
ENGINES = {
    PaddleOCREngine.name: PaddleOCREngine,
    TesseractEngine.name: TesseractEngine,
}


def get_engine(name, **kwargs) -> OCREngine:
    try:
        return ENGINES[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown OCR engine {name!r}; choose from {sorted(ENGINES)}") from None


__all__ = [
    "ENGINES", "OCREngine", "OCRLine", "OCRResult", "OCRWord", "PaddleOCREngine",
    "ReadingOrder", "TesseractEngine", "cached_ocr", "get_engine", "load_run_log", "new_run_dir",
    "reading_order", "save_run_log",
]

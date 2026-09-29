"""Engine-independent OCR data model and engine interface.

Every engine turns an image path into an OCRResult: a list of OCRLine pieces
(text + box + confidence in 0..1), plus timings and the settings used. The
rest of the pipeline only depends on these types, so engines are swappable.

Adding an engine (extension point):
    1. Subclass OCREngine, set `name`, implement `_load()` (one-time model or
       binary setup), `_predict(image_path)` (return list[OCRLine] in the
       engine's own order), `version()` and `settings()`.
    2. Report confidence in 0..1 and boxes as pixel (x1, y1, x2, y2) on the
       original image, so results are comparable across engines.
    3. Register it in ENGINES in src/ocr/__init__.py.
See paddle.py (deep-learning detector + recognizer) and tesseract.py
(classic LSTM engine) for the two existing implementations.
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image
from pydantic import BaseModel


class OCRWord(BaseModel):
    text: str
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float  # 0..1


class OCRLine(BaseModel):
    """One text piece as the engine returned it (a detected text segment,
    not necessarily a full visual row: rows are rebuilt in layout.py)."""
    index: int  # position in the engine's output order
    text: str
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 (axis-aligned)
    polygon: list[tuple[int, int]]  # 4 corner points, clockwise from top-left
    confidence: float  # 0..1
    angle: int | None = None  # text-line orientation in degrees, if the engine reports it
    words: list[OCRWord] | None = None  # word-level detail, if the engine reports it


class OCRResult(BaseModel):
    image_path: str
    image_size: tuple[int, int]  # width, height
    engine: str
    engine_version: str
    settings: dict
    lines: list[OCRLine]
    timings: dict[str, float]  # load_s (0 if already loaded), ocr_s


class OCREngine(ABC):
    name = "base"

    def __init__(self):
        self._loaded = False

    @abstractmethod
    def _load(self):
        """One-time setup (load models, locate binaries)."""

    @abstractmethod
    def _predict(self, image_path: str) -> list[OCRLine]:
        """Run OCR on one image and return its lines."""

    @abstractmethod
    def version(self) -> str:
        ...

    @abstractmethod
    def settings(self) -> dict:
        ...

    def load(self) -> float:
        """Load once; return seconds spent (0 if already loaded)."""
        if self._loaded:
            return 0.0
        t0 = time.perf_counter()
        self._load()
        self._loaded = True
        return time.perf_counter() - t0

    def recognize(self, image_path) -> OCRResult:
        image_path = str(image_path)
        if not Path(image_path).is_file():
            raise FileNotFoundError(image_path)
        load_s = self.load()

        t0 = time.perf_counter()
        lines = self._predict(image_path)
        ocr_s = time.perf_counter() - t0

        with Image.open(image_path) as img:
            size = img.size
        return OCRResult(
            image_path=Path(image_path).as_posix(),
            image_size=size,
            engine=self.name,
            engine_version=self.version(),
            settings=self.settings(),
            lines=lines,
            timings={"load_s": round(load_s, 3), "ocr_s": round(ocr_s, 3)},
        )


def rect_polygon(box):
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

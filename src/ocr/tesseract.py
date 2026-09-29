"""Tesseract 5 engine via pytesseract (classic LSTM OCR with its own layout analysis).

Needs the Tesseract binary (not pip-installable; see README). It is found via
the `tesseract_cmd` argument, the TESSERACT_CMD environment variable, PATH, or
the default Windows install folder, in that order.
"""

import os
import shutil
from itertools import groupby
from pathlib import Path

from .base import OCREngine, OCRLine, OCRWord, rect_polygon

WINDOWS_DEFAULT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


def find_tesseract(cmd=None):
    for candidate in (cmd, os.environ.get("TESSERACT_CMD"), shutil.which("tesseract")):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    if WINDOWS_DEFAULT.is_file():
        return str(WINDOWS_DEFAULT)
    raise FileNotFoundError(
        "Tesseract binary not found. Install it (README 'Windows setup') or set TESSERACT_CMD.")


def lines_from_data(data):
    """Group pytesseract.image_to_data output (dict of lists) into OCRLines,
    one per Tesseract (block, paragraph, line), keeping word-level detail.
    Line confidence is the mean word confidence, rescaled from 0..100 to 0..1."""
    words = []
    for i, text in enumerate(data["text"]):
        conf = float(data["conf"][i])
        if not str(text).strip() or conf < 0:  # conf -1 marks non-word layout rows
            continue
        x, y, w, h = (int(data[k][i]) for k in ("left", "top", "width", "height"))
        key = (data["page_num"][i], data["block_num"][i], data["par_num"][i], data["line_num"][i])
        words.append((key, OCRWord(text=str(text), box=(x, y, x + w, y + h),
                                   confidence=round(conf / 100, 4))))

    lines = []
    for _, group in groupby(words, key=lambda kw: kw[0]):
        ws = [w for _, w in group]
        box = (min(w.box[0] for w in ws), min(w.box[1] for w in ws),
               max(w.box[2] for w in ws), max(w.box[3] for w in ws))
        lines.append(OCRLine(
            index=len(lines),
            text=" ".join(w.text for w in ws),
            box=box,
            polygon=rect_polygon(box),
            confidence=round(sum(w.confidence for w in ws) / len(ws), 4),
            words=ws,
        ))
    return lines


class TesseractEngine(OCREngine):
    name = "tesseract"

    def __init__(self, lang="eng", psm=4, oem=1, grayscale=True, tesseract_cmd=None):
        super().__init__()
        self.lang = lang
        self.psm = psm  # 4 = single column of text of variable sizes (suits receipts)
        self.oem = oem  # 1 = LSTM engine only
        # PIL grayscale before Tesseract: found the SROIE total in 20/21 test
        # receipts vs 18/21 on the colour image; upscaling made results worse
        self.grayscale = grayscale
        self._cmd = tesseract_cmd
        self._version = None

    def settings(self):
        return {"lang": self.lang, "psm": self.psm, "oem": self.oem,
                "preprocessing": "grayscale" if self.grayscale else "none"}

    def version(self):
        return self._version or "unknown"

    def _load(self):
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = find_tesseract(self._cmd)
        self._version = str(pytesseract.get_tesseract_version())

    def _predict(self, image_path):
        import pytesseract
        from PIL import Image, ImageOps

        with Image.open(image_path) as img:
            if self.grayscale:
                img = ImageOps.grayscale(img)
            data = pytesseract.image_to_data(
                img, lang=self.lang, config=f"--oem {self.oem} --psm {self.psm}",
                output_type=pytesseract.Output.DICT)
        return lines_from_data(data)

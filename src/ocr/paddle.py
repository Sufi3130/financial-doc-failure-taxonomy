"""PaddleOCR 3.x engine, CPU only (PP-OCR detector + recognizer)."""

import os

from .base import OCREngine, OCRLine

# Skip PaddleX's online model-host check on every start
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


class PaddleOCREngine(OCREngine):
    name = "paddleocr"

    def __init__(self, lang="en", use_textline_orientation=True):
        super().__init__()
        self.lang = lang
        self.use_textline_orientation = use_textline_orientation
        self._ocr = None

    def settings(self):
        return {
            "lang": self.lang,
            "device": "cpu",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": self.use_textline_orientation,
            # oneDNN crashes in Paddle 3.x on Windows CPUs
            "enable_mkldnn": False,
            # keep every recognised line; low confidence is data, not noise
            "text_rec_score_thresh": 0.0,
        }

    def version(self):
        import paddleocr
        return paddleocr.__version__

    def _load(self):
        from paddleocr import PaddleOCR

        s = self.settings()
        self._ocr = PaddleOCR(
            lang=s["lang"],
            device=s["device"],
            use_doc_orientation_classify=s["use_doc_orientation_classify"],
            use_doc_unwarping=s["use_doc_unwarping"],
            use_textline_orientation=s["use_textline_orientation"],
            enable_mkldnn=s["enable_mkldnn"],
            text_rec_score_thresh=s["text_rec_score_thresh"],
        )

    def _predict(self, image_path):
        lines = []
        for res in self._ocr.predict(image_path):
            angles = res.get("textline_orientation_angles") or [None] * len(res["rec_texts"])
            for text, score, poly, box, angle in zip(
                res["rec_texts"], res["rec_scores"], res["rec_polys"], res["rec_boxes"], angles
            ):
                lines.append(OCRLine(
                    index=len(lines),
                    text=text,
                    box=tuple(int(v) for v in box),
                    polygon=[(int(x), int(y)) for x, y in poly],
                    confidence=round(float(score), 4),
                    angle=None if angle is None or angle < 0 else int(angle),
                ))
        return lines

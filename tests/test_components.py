"""Schema, PaddleOCR result conversion and the llama-cpp wrapper, with stand-ins
for PaddleOCR and llama-cpp (neither needs to be installed)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.extraction import LLMSettings, LocalLLM, Receipt
from src.ocr import PaddleOCREngine

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ schema
def test_receipt_schema_accepts_strings_and_null():
    r = Receipt(company="A", date=None, address="B", total="1.00")
    assert r.model_dump() == {"company": "A", "date": None, "address": "B", "total": "1.00"}


@pytest.mark.parametrize("data, error", [
    ({"company": "A", "date": None, "address": None, "total": 1.0}, "string_type"),   # number, strict
    ({"company": "A", "date": None, "address": None}, "missing"),                    # key absent
    ({"company": "A", "date": None, "address": None, "total": "1", "tax": "0"}, "extra_forbidden"),
    ({"company": ["A"], "date": None, "address": None, "total": "1"}, "string_type"),
])
def test_receipt_schema_rejects(data, error):
    with pytest.raises(ValidationError) as e:
        Receipt.model_validate(data)
    assert error in {err["type"] for err in e.value.errors()}


# ------------------------------------------------------------------ PaddleOCR conversion
class FakePaddleOCR:
    """Returns one result shaped like PaddleOCR 3.x OCRResult (dict-like, numpy arrays)."""

    def predict(self, image_path):
        return [{
            "rec_texts": ["TOTAL:", "193.00"],
            "rec_scores": [0.9991, 0.7667],
            "rec_polys": [np.array([[171, 696], [240, 696], [240, 718], [171, 718]], dtype=np.int16),
                          np.array([[288, 697], [358, 697], [358, 719], [288, 719]], dtype=np.int16)],
            "rec_boxes": np.array([[171, 696, 240, 718], [288, 697, 358, 719]], dtype=np.int16),
            "textline_orientation_angles": [0, -1],
        }]


def test_paddle_result_converted_to_ocr_lines():
    engine = PaddleOCREngine()
    engine._ocr, engine._loaded = FakePaddleOCR(), True  # skip loading real models
    lines = engine._predict("x.jpg")
    assert [(ln.index, ln.text, ln.box, ln.confidence, ln.angle) for ln in lines] == [
        (0, "TOTAL:", (171, 696, 240, 718), 0.9991, 0),
        (1, "193.00", (288, 697, 358, 719), 0.7667, None),  # -1 = no angle
    ]
    assert lines[0].polygon == [(171, 696), (240, 696), (240, 718), (171, 718)]
    assert engine.settings()["device"] == "cpu" and engine.settings()["text_rec_score_thresh"] == 0.0


# ------------------------------------------------------------------ llama-cpp wrapper
class FakeLlama:
    def __init__(self, content="{}", finish="stop", error=None):
        self.content, self.finish, self.error = content, finish, error
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {"choices": [{"message": {"content": self.content}, "finish_reason": self.finish}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 30}}


def make_llm(**kwargs):
    llm = LocalLLM(LLMSettings(model_path="missing.gguf", max_tokens=64, seed=7))
    llm._llm = FakeLlama(**kwargs)  # load() is skipped when a model object is present
    return llm


def test_llm_wrapper_returns_output_and_uses_greedy_settings():
    llm = make_llm(content=' {"company": null}')
    gen = llm.generate([{"role": "user", "content": "hi"}])
    assert (gen.raw_output, gen.finish_reason, gen.prompt_tokens, gen.completion_tokens) == \
        (' {"company": null}', "stop", 120, 30)
    call = llm._llm.calls[0]
    assert call["temperature"] == 0.0 and call["seed"] == 7 and call["max_tokens"] == 64
    assert gen.prompt_tokens_evaluated is None  # no llama.cpp perf counters on the stand-in
    llm.generate([{"role": "user", "content": "hi"}], max_tokens=1)
    assert llm._llm.calls[1]["max_tokens"] == 1


def test_llm_wrapper_maps_context_overflow():
    llm = make_llm(error=ValueError("Requested tokens (5000) exceed context window of 4096"))
    gen = llm.generate([{"role": "user", "content": "x"}])
    assert gen.finish_reason == "prompt_too_long" and gen.raw_output is None


def test_llm_wrapper_reraises_other_errors():
    with pytest.raises(ValueError, match="something else"):
        make_llm(error=ValueError("something else")).generate([{"role": "user", "content": "x"}])


def test_llm_missing_model_file_is_a_clear_error():
    with pytest.raises(FileNotFoundError, match="GGUF model not found"):
        LocalLLM(LLMSettings(model_path="missing.gguf")).load()


# ------------------------------------------------------------------ heavy dependencies stay lazy
def test_project_imports_do_not_load_ocr_or_llm_libraries():
    code = (
        "import sys; sys.path.insert(0, '.');"
        "import src.ocr, src.extraction, src.run, evaluation.evaluate, evaluation.normalize;"
        "heavy = [m for m in ('paddleocr', 'paddle', 'paddlex', 'llama_cpp', 'pytesseract') if m in sys.modules];"
        "print(','.join(heavy))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""

"""Grammar-constrained decoding (issue #25): GBNF builder, the opt-in setting,
and the empty-page all-null short-circuit. No model is loaded (the GBNF-compile
test is skipped when llama_cpp is absent)."""

import importlib.util
import json

import pytest
from PIL import Image

from src.extraction import LLMSettings, all_null_json, grammar_gbnf, parse_output
from src.extraction.schema import FIELDS
from src.run import process_receipt

from tests.test_run import FakeEngine, FakeLLM

_HAS_LLAMA = importlib.util.find_spec("llama_cpp") is not None


def test_gbnf_lists_all_fields_in_order():
    g = grammar_gbnf()
    # each schema key appears, in schema order, between root's braces
    positions = [g.index(f'\\"{f}\\"') for f in FIELDS]
    assert positions == sorted(positions)
    assert g.startswith('root ::= "{"')
    assert 'ws "}"' in g
    assert 'value ::= string | "null"' in g


def test_all_null_json_is_schema_valid():
    value = json.loads(all_null_json())
    assert value == {f: None for f in FIELDS}
    outcome = parse_output(all_null_json(), finish_reason="stop")
    assert outcome.schema_valid
    assert outcome.lenient == {f: None for f in FIELDS}


def test_settings_grammar_defaults_off_and_is_carried():
    assert LLMSettings().grammar is False
    assert LLMSettings(grammar=True).grammar is True


@pytest.mark.skipif(not _HAS_LLAMA, reason="llama_cpp not installed")
def test_grammar_compiles():
    from src.extraction import load_grammar

    assert type(load_grammar()).__name__ == "LlamaGrammar"


def _record(tmp_path, rid="r1"):
    path = tmp_path / f"{rid}.png"
    Image.new("RGB", (20, 20), "white").save(path)
    return {"id": rid, "split": "test", "image": str(path)}


def test_empty_page_short_circuits_to_all_null_in_grammar_mode(tmp_path, monkeypatch):
    """Grammar mode + blank OCR -> all-null answer, model never called."""
    import src.run as run_mod

    # OCR returns empty text for this receipt
    class BlankReading:
        text = "   "
        rows = []

    def fake_cached_ocr(engine, image_path, rid, cache_dir, force):
        class R:
            engine_version = "0"
            lines = []
            image_size = (20, 20)
            timings = {"ocr_s": 0.0}
        return R(), BlankReading(), None, True

    monkeypatch.setattr(run_mod, "cached_ocr", fake_cached_ocr)
    llm = FakeLLM(grammar=True)
    row = process_receipt(_record(tmp_path), FakeEngine(), llm, (), set(), "ui",
                          cache_dir=tmp_path / "cache")
    assert row["error"] is None
    assert llm.prompts == []  # model was not called
    assert row["llm"]["raw_output"] == all_null_json()
    assert row["outcome"]["schema_valid"]


def test_blank_ocr_still_calls_model_when_grammar_off(tmp_path, monkeypatch):
    import src.run as run_mod

    class BlankReading:
        text = "   "
        rows = []

    def fake_cached_ocr(engine, image_path, rid, cache_dir, force):
        class R:
            engine_version = "0"
            lines = []
            image_size = (20, 20)
            timings = {"ocr_s": 0.0}
        return R(), BlankReading(), None, True

    monkeypatch.setattr(run_mod, "cached_ocr", fake_cached_ocr)
    llm = FakeLLM(grammar=False)
    process_receipt(_record(tmp_path), FakeEngine(), llm, (), set(), "ui",
                    cache_dir=tmp_path / "cache")
    assert llm.prompts  # model WAS called (baseline still measures the failure)

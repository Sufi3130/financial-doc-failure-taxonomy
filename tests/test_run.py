"""Pipeline runner tests with a fake OCR engine and a fake LLM (no models needed)."""

import json

from PIL import Image

from src.extraction import Generation
from src.ocr import OCREngine, OCRLine
from src.ocr.base import rect_polygon
from src.run import done_keys, run

GOOD = '{"company": "A", "date": "1/1/2018", "address": "B", "total": "1.00"}'


class FakeEngine(OCREngine):
    name = "fake"

    def __init__(self, fail_on=()):
        super().__init__()
        self.fail_on = set(fail_on)
        self.calls = 0

    def _load(self):
        pass

    def _predict(self, image_path):
        self.calls += 1
        if any(f in image_path for f in self.fail_on):
            raise RuntimeError("OCR exploded")
        box = (0, 0, 10, 10)
        return [OCRLine(index=0, text="TOTAL: 1.00", box=box, polygon=rect_polygon(box), confidence=0.9)]

    def version(self):
        return "0"

    def settings(self):
        return {"x": 1}


class FakeLLM:
    def __init__(self, outputs=None):
        self.outputs = outputs or {}
        self.prompts = []

    def generate(self, messages, max_tokens=None):
        text = messages[-1]["content"]
        self.prompts.append(text)
        raw = next((v for k, v in self.outputs.items() if k in text), GOOD)
        return Generation(raw_output=raw, finish_reason="stop", prompt_tokens=10, completion_tokens=5,
                          seconds=0.5, prompt_tokens_evaluated=3, prompt_eval_s=0.2, gen_s=0.3)

    def info(self):
        return {"model_file": "fake.gguf"}


def no_fewshot(engine, k, seed):
    return [], []


def make_records(tmp_path, ids):
    records = []
    for rid in ids:
        path = tmp_path / f"{rid}.png"
        Image.new("RGB", (20, 20), "white").save(path)
        records.append({"id": rid, "split": "test", "image": str(path)})
    return records


def read_rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def test_batch_appends_one_line_per_receipt_with_timings(tmp_path):
    records = make_records(tmp_path, ["r1", "r2"])
    run_jsonl, meta = run(records, [FakeEngine()], FakeLLM(), tmp_path / "run", shots=0,
                          cache_dir=tmp_path / "cache", fewshot_loader=no_fewshot, log=lambda *_: None)
    rows = read_rows(run_jsonl)
    assert [r["receipt_id"] for r in rows] == ["r1", "r2"]
    row = rows[0]
    assert row["ocr"]["text"] == "TOTAL: 1.00" and row["llm"]["raw_output"] == GOOD
    assert row["outcome"]["strict_valid"] and row["error"] is None
    t = row["timings"]
    assert t["total_s"] == round(t["ocr_s"] + t["llm_s"] + t["parse_s"], 3)
    assert t["llm_prompt_s"] == 0.2 and t["llm_gen_s"] == 0.3
    assert meta["summary"]["fake"]["strict_valid"] == 2
    assert meta["engines"]["fake"]["warmup"]["prompt_tokens"] == 10


def test_error_is_recorded_and_batch_continues(tmp_path):
    records = make_records(tmp_path, ["ok1", "bad", "ok2"])
    run_jsonl, meta = run(records, [FakeEngine(fail_on=["bad"])], FakeLLM(), tmp_path / "run", shots=0,
                          cache_dir=tmp_path / "cache", fewshot_loader=no_fewshot, log=lambda *_: None)
    rows = read_rows(run_jsonl)
    assert [r["receipt_id"] for r in rows] == ["ok1", "bad", "ok2"]
    assert rows[1]["error"] == {"stage": "ocr", "type": "RuntimeError", "message": "OCR exploded"}
    assert meta["summary"]["fake"]["errors"] == 1


def test_llm_output_failures_are_logged(tmp_path):
    records = make_records(tmp_path, ["r1"])
    llm = FakeLLM({"TOTAL": "Sure! The total is 1.00"})
    run_jsonl, meta = run(records, [FakeEngine()], llm, tmp_path / "run", shots=0,
                          cache_dir=tmp_path / "cache", fewshot_loader=no_fewshot, log=lambda *_: None)
    row = read_rows(run_jsonl)[0]
    assert [f["type"] for f in row["outcome"]["failures"]] == ["no_json"]
    assert meta["summary"]["fake"]["failure_counts"] == {"no_json": 1}


def test_resume_skips_done_and_retries_errors(tmp_path):
    records = make_records(tmp_path, ["r1", "bad"])
    run_dir, cache = tmp_path / "run", tmp_path / "cache"
    run(records, [FakeEngine(fail_on=["bad"])], FakeLLM(), run_dir, shots=0, cache_dir=cache,
        fewshot_loader=no_fewshot, log=lambda *_: None)
    assert done_keys(run_dir / "run.jsonl") == {("fake", "r1")}

    engine = FakeEngine()
    run(records, [engine], FakeLLM(), run_dir, shots=0, cache_dir=cache,
        fewshot_loader=no_fewshot, log=lambda *_: None)
    rows = read_rows(run_dir / "run.jsonl")
    assert [(r["receipt_id"], bool(r["error"])) for r in rows] == [("r1", False), ("bad", True), ("bad", False)]
    assert engine.calls == 1  # only the failed receipt was redone


def test_cached_ocr_keeps_measured_time_and_is_flagged(tmp_path):
    records = make_records(tmp_path, ["r1"])
    kw = dict(shots=0, cache_dir=tmp_path / "cache", fewshot_loader=no_fewshot, log=lambda *_: None)
    first = read_rows(run(records, [FakeEngine()], FakeLLM(), tmp_path / "a", **kw)[0])[0]
    engine = FakeEngine()
    second = read_rows(run(records, [engine], FakeLLM(), tmp_path / "b", **kw)[0])[0]
    assert first["ocr_cached"] is False and second["ocr_cached"] is True
    assert second["timings"]["ocr_s"] == first["timings"]["ocr_s"] and engine.calls == 0


def test_fewshot_receipt_is_not_extracted(tmp_path):
    records = make_records(tmp_path, ["ex1"])
    run_jsonl, _ = run(records, [FakeEngine()], FakeLLM(), tmp_path / "run", shots=1,
                       cache_dir=tmp_path / "cache", fewshot_loader=lambda e, k, seed: (["ex1"], [("T", {})]),
                       log=lambda *_: None)
    assert read_rows(run_jsonl)[0]["error"]["message"] == "receipt is one of the few-shot examples"

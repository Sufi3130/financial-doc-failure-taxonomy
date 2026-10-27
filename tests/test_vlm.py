"""VLM runner tests with a fake VLM (no model, no llama-cpp needed)."""

import json

from PIL import Image

import src.vlm as vlm
from src.vlm import process_receipt, run


class FakeVLM:
    model = "fake-vlm"

    def __init__(self, outputs=None):
        self.outputs = outputs or {}
        self.calls = 0

    def extract(self, image_path):
        self.calls += 1
        from pathlib import Path
        raw = self.outputs.get(Path(image_path).stem,
                               '{"company": "A", "date": "1/1/2018", "address": "B", "total": "1.00"}')
        return {"raw_output": raw, "finish_reason": "stop", "prompt_tokens": 500,
                "completion_tokens": 20, "seconds": 120.0, "image_resized_to": [800, 1200]}

    def info(self):
        return {"model": self.model, "model_file": "fake.gguf"}


def record(tmp_path, rid):
    img = tmp_path / f"{rid}.png"
    Image.new("RGB", (8, 8), "white").save(img)
    return {"id": rid, "split": "test", "image": str(img),
            "ground_truth": {"raw": {"company": "A", "date": "1/1/2018", "address": "B", "total": "1.00"},
                             "normalized": {"date": "2018-01-01", "total": "1.00"}}}


def test_vlm_row_has_pipeline_format(tmp_path):
    row = process_receipt(record(tmp_path, "r1"), FakeVLM(), "run1")
    assert row["engine"] == "vlm" and row["model"] == "fake-vlm"
    assert row["ocr"]["end_to_end"] is True and row["ocr"]["text"] == ""  # no OCR stage
    assert row["outcome"]["strict_valid"] is True
    assert row["timings"]["llm_s"] == 120.0 and row["timings"]["total_s"] == 120.0
    assert row["error"] is None


def test_vlm_run_writes_evaluable_log_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm, "REPO_ROOT", tmp_path)
    recs = [record(tmp_path, "r1"), record(tmp_path, "r2")]
    fake = FakeVLM({"r2": "Sorry, I cannot read this image."})
    run_jsonl, meta = run(recs, fake, tmp_path / "run", {"run_id": "run1"}, log=lambda *_: None)
    rows = [json.loads(line) for line in open(run_jsonl, encoding="utf-8")]
    assert [r["receipt_id"] for r in rows] == ["r1", "r2"]
    assert rows[1]["outcome"]["failures"][0]["type"] == "no_json"  # the refusal is tagged
    s = meta["summary"]["vlm"]  # run-level counts (field accuracy comes from evaluate.py)
    assert s["receipts"] == 2 and s["errors"] == 0
    assert s["strict_valid"] == 1 and s["failure_counts"] == {"no_json": 1}


def test_vlm_resume_skips_done(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm, "REPO_ROOT", tmp_path)
    recs = [record(tmp_path, "r1"), record(tmp_path, "r2")]
    run(recs, FakeVLM(), tmp_path / "run", {"run_id": "run1"}, log=lambda *_: None)
    fake = FakeVLM()
    run(recs, fake, tmp_path / "run", {"run_id": "run1"}, log=lambda *_: None)
    assert fake.calls == 0  # both already done

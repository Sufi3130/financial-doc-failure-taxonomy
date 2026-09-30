"""End-to-end smoke test: fixture manifest -> pipeline (mocked OCR + LLM) -> run log -> evaluator."""

import json
from pathlib import Path

from src.extraction import Generation
from src.ocr import OCREngine, OCRLine
from src.ocr.base import rect_polygon
from src.run import run

from evaluation.evaluate import evaluate

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "pipeline"
OCR = json.loads((FIXTURES / "ocr.json").read_text(encoding="utf-8"))
LLM_OUTPUTS = json.loads((FIXTURES / "llm_outputs.json").read_text(encoding="utf-8"))


class FixtureOCR(OCREngine):
    """Returns the fixture text for each image, one piece per row."""
    name = "fixture_ocr"

    def _load(self):
        pass

    def _predict(self, image_path):
        rows = OCR[Path(image_path).stem]
        return [OCRLine(index=i, text=t, box=(0, 20 * i, 100, 20 * i + 15),
                        polygon=rect_polygon((0, 20 * i, 100, 20 * i + 15)), confidence=0.95)
                for i, t in enumerate(rows)]

    def version(self):
        return "fixture"

    def settings(self):
        return {"fixture": True}


class FixtureLLM:
    """Answers with the fixture output of the receipt whose OCR text is in the prompt."""

    def __init__(self):
        self.calls = 0

    def generate(self, messages, max_tokens=None):
        self.calls += 1
        prompt = messages[-1]["content"]
        rid = next((r for r, rows in OCR.items() if rows and rows[0] in prompt), "R3")
        raw = "{}" if max_tokens == 1 else LLM_OUTPUTS[rid]
        finish = "length" if rid == "R2" else "stop"
        return Generation(raw_output=raw, finish_reason=finish, prompt_tokens=100, completion_tokens=20,
                          seconds=1.0, prompt_tokens_evaluated=40, prompt_eval_s=0.6, gen_s=0.4)

    def info(self):
        return {"model_file": "fixture.gguf"}


def test_pipeline_to_evaluation_smoke(tmp_path):
    manifest = FIXTURES / "manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    llm = FixtureLLM()
    run_jsonl, meta = run(records, [FixtureOCR()], llm, tmp_path / "run", shots=0,
                          cache_dir=tmp_path / "cache", fewshot_loader=lambda e, k, seed: ([], []),
                          log=lambda *_: None)

    # pipeline: one row per receipt, raw output kept, failures tagged
    rows = [json.loads(line) for line in run_jsonl.read_text(encoding="utf-8").splitlines()]
    assert [r["receipt_id"] for r in rows] == ["R1", "R2", "R3"]
    assert llm.calls == 4  # warm-up + 3 receipts
    assert rows[0]["ocr"]["text"].startswith("OJC MARKETING SDN BHD\n")
    assert rows[0]["llm"]["raw_output"].startswith("```json")
    assert rows[0]["outcome"]["repairs"] == ["code_fence"]
    assert {f["type"] for f in rows[1]["outcome"]["failures"]} == {"truncated", "invalid_json"}
    assert [f["type"] for f in rows[2]["outcome"]["failures"]] == ["no_json"]
    assert rows[2]["ocr"]["text"] == ""
    assert meta["summary"]["fixture_ocr"]["schema_valid"] == 1

    # evaluator: every manifest receipt counts; failed outputs are wrong
    _, per_receipt, summaries = evaluate(tmp_path / "run", manifest)
    s = summaries["fixture_ocr"]
    assert s["receipts"] == 3 and s["errors"] == 0
    assert s["fields"]["total"]["normalized_acc"] == 33.3
    assert s["fields"]["total"]["strict_acc"] == 33.3
    assert s["all_fields_normalized_acc"] == 33.3
    assert s["schema"]["strict_valid_rate"] == 0.0     # R1 needed a repair
    assert s["schema"]["schema_valid_rate"] == 33.3
    assert s["fields"]["total"]["ocr_miss"] == 1        # R3: empty OCR
    assert s["fields"]["total"]["llm_miss_given_in_ocr"] == 1  # R2: derailed although the total was read
    assert s["latency_s"]["total_s"]["n"] == 3

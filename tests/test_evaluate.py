"""Evaluator tests on hand-made run rows and manifest records."""

import json

import pytest

from evaluation.evaluate import evaluate, score_receipt, summarize, token_f1, write_results


def sroie(rid, company="OJC MARKETING SDN. BHD.", date="15/01/2019", address="NO 2, JALAN BAYU 4",
          total="RM 193.00"):
    return {"id": rid, "split": "test", "image": "x.jpg", "locale": "my",
            "ground_truth": {"raw": {"company": company, "date": date, "address": address, "total": total},
                             "normalized": {"date": "2019-01-15", "total": "193.00"}}}


def cord(rid, total="Rp 60.000"):
    return {"id": rid, "split": "test", "image": "x.png", "locale": "id", "fields_available": ["total"],
            "ground_truth": {"raw": {"company": None, "date": None, "address": None, "total": total},
                             "normalized": {"date": None, "total": "60000.00"}}}


def row(rid, pred, engine="paddleocr", ocr="OJC MARKETING SDN BHD\nTOTAL: 193.00\n15/01/2019", **extra):
    return {"engine": engine, "receipt_id": rid, "error": None, "ocr": {"text": ocr},
            "outcome": {"lenient": pred, "strict_valid": True, "schema_valid": True, "failures": []},
            "timings": {"ocr_s": 10.0, "llm_s": 40.0, "total_s": 50.0}, "power": {"source": "ac"}, **extra}


def test_strict_vs_normalized():
    pred = {"company": "OJC MARKETING SDN BHD", "date": "2019-01-15", "address": "NO 2, JALAN BAYU 4",
            "total": "193.00"}
    f = score_receipt(sroie("r1"), row("r1", pred))["fields"]
    assert {k: (v["strict"], v["normalized"]) for k, v in f.items()} == {
        "company": (False, True),   # punctuation differs
        "date": (False, True),      # ISO vs dd/mm/yyyy
        "address": (True, True),
        "total": (False, True),     # "RM 193.00" vs "193.00"
    }


def test_value_in_ocr_separates_ocr_from_llm_misses():
    pred = {"company": None, "date": "15/01/2019", "address": None, "total": "19.00"}
    f = score_receipt(sroie("r1"), row("r1", pred))["fields"]
    assert f["total"]["in_ocr"] and not f["total"]["normalized"]     # LLM miss
    assert not f["address"]["in_ocr"]                                 # OCR miss (not in text)
    assert f["company"]["in_ocr"] and not f["company"]["normalized"]  # LLM miss


def test_cord_scores_total_only_with_rupiah_locale():
    r = score_receipt(cord("c1"), row("c1", {"company": "X", "date": None, "address": None, "total": "60,000"},
                                      ocr="TOTAL 60.000"))
    assert list(r["fields"]) == ["total"]
    assert r["fields"]["total"]["normalized"] and r["fields"]["total"]["in_ocr"]
    # "60.00" means 60 rupiah, not 60000
    r = score_receipt(cord("c1"), row("c1", {"total": "60.00"}))
    assert not r["fields"]["total"]["normalized"]


def test_missing_row_and_failed_output_count_as_wrong():
    summary = summarize([
        score_receipt(sroie("r1"), row("r1", {"company": "OJC MARKETING SDN. BHD.", "date": "15/01/2019",
                                              "address": "NO 2, JALAN BAYU 4", "total": "RM 193.00"})),
        score_receipt(sroie("r2"), None),
        score_receipt(sroie("r3"), {**row("r3", None), "outcome": {"lenient": None, "strict_valid": False,
                                                                   "schema_valid": False,
                                                                   "failures": [{"type": "no_json"}]}}),
    ])
    assert summary["receipts"] == 3 and summary["errors"] == 1
    assert summary["fields"]["total"]["strict_acc"] == pytest.approx(33.3)
    assert summary["all_fields_normalized_acc"] == pytest.approx(33.3)
    assert summary["schema"]["strict_valid_rate"] == pytest.approx(33.3)
    assert summary["schema"]["failure_counts"] == {"no_json": 1}
    assert summary["latency_s"]["total_s"]["n"] == 2  # the never-run receipt has no timing


def test_token_f1():
    assert token_f1("NO 2, JALAN BAYU 4", "NO 2 JALAN BAYU 4") == 1.0
    assert token_f1("JALAN BAYU", "NO 2 JALAN BAYU 4") == pytest.approx(0.5714, abs=1e-4)
    assert token_f1(None, "X") == 0.0


def test_last_row_wins_and_files_written(tmp_path, monkeypatch):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    rows = [{**row("r1", None), "error": {"type": "RuntimeError"}, "outcome": None},
            row("r1", {"company": "OJC MARKETING SDN. BHD.", "date": "15/01/2019",
                       "address": "NO 2, JALAN BAYU 4", "total": "RM 193.00"})]
    (run_dir / "run.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps(sroie("r1")) + "\n", encoding="utf-8")

    records, per_receipt, summaries = evaluate(run_dir, manifest)
    assert summaries["paddleocr"]["errors"] == 0
    assert summaries["paddleocr"]["all_fields_normalized_acc"] == 100.0

    import evaluation.evaluate as ev
    monkeypatch.setattr(ev, "REPO_ROOT", tmp_path)  # relative paths in summary.json
    out = write_results(run_dir, manifest, records, per_receipt, summaries, out_root=tmp_path / "results")
    assert {p.name for p in out.iterdir()} == {"summary.json", "per_receipt.csv", "per_receipt_detail.jsonl"}
    csv_text = (out / "per_receipt.csv").read_text(encoding="utf-8")
    assert "OJC" not in csv_text  # committed file carries no label or prediction text

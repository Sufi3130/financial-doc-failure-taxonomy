"""Run-to-run comparison on synthetic run logs."""

import json

from evaluation.reproducibility import compare_runs, fingerprint


def write_run(path, rows, commit="abc", cache=True):
    path.mkdir()
    (path / "run.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    meta = {"git": {"commit": commit, "uncommitted_changes": []}, "manifest_sha256": "m", "prompt_version": "v1",
            "shots": 2, "fewshot_seed": 42, "ocr_cache": cache, "versions": {"x": "1"},
            "llm": {"model_file": "m.gguf", "temperature": 0.0},
            "engines": {"e": {"settings": {"psm": 4}, "fewshot_ids": ["A"], "prompt_prefix_messages": [{"c": "p"}]}}}
    (path / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def row(rid, ocr="TOTAL 1.00", raw='{"total": "1.00"}', total="1.00", seconds=10.0):
    return {"engine": "e", "receipt_id": rid, "ocr": {"text": ocr}, "llm": {"raw_output": raw},
            "outcome": {"lenient": {"total": total}}, "timings": {"total_s": seconds}, "ocr_cached": False}


def test_identical_runs_with_different_timings(tmp_path):
    a = write_run(tmp_path / "a", [row("r1"), row("r2")])
    b = write_run(tmp_path / "b", [row("r1", seconds=99.0), row("r2", seconds=5.0)], commit="def", cache=False)
    rep = compare_runs(a, b)
    assert rep["identical"] == {"ocr_text": "2/2", "raw_output": "2/2", "fields": "2/2"}
    assert rep["all_identical"] is True
    assert rep["settings"]["differing_keys"] == ["ocr_cache"]
    assert rep["conditions"]["b"]["total_s_per_page"]["max"] == 99.0
    assert "TOTAL" not in json.dumps(rep["rows"])  # only fingerprints, no text


def test_differences_and_missing_rows_are_reported(tmp_path):
    a = write_run(tmp_path / "a", [row("r1"), row("r2"), row("r3")])
    b = write_run(tmp_path / "b", [row("r1", raw='{"total": "2.00"}', total="2.00"), row("r2", ocr="TOTAL 1.0O")])
    rep = compare_runs(a, b)
    assert rep["identical"] == {"ocr_text": "1/2", "raw_output": "1/2", "fields": "1/2"}
    assert rep["only_in_a"] == [["e", "r3"]] and rep["all_identical"] is False


def test_fingerprint_is_exact():
    assert fingerprint("abc") == fingerprint("abc") != fingerprint("abc ")
    assert fingerprint({"b": 1, "a": 2}) == fingerprint({"a": 2, "b": 1})
    assert fingerprint(None) is None

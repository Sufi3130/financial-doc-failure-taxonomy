"""Tests for the failure-taxonomy tooling: tag.py helper functions and the
cross-model / review scripts (by_model, make_review_v1, show_case). All pure
functions or small file round-trips — no models, OCR or real datasets."""

import csv
import json
import random

from evaluation.failure_taxonomy import by_model, make_review_v1, show_case
from evaluation.failure_taxonomy import tag as tagger


def inst(primary, dataset="SROIE", engine="paddleocr", model="Phi-3", field="total",
         subtype=None, tags=(), stage="llm_extraction", receipt_id="R1"):
    return {"dataset": dataset, "engine": engine, "model": model, "field": field,
            "receipt_id": receipt_id, "primary": primary, "subtype": subtype,
            "tags": list(tags), "stage": stage, "gold": "X", "pred": "Y"}


# ---------------------------------------------------------------- tag.py helpers
def test_counts_groups_by_cell_and_category():
    insts = [inst("value_not_in_ocr"), inst("value_not_in_ocr"),
             inst("wrong_total", subtype="scale_error", tags=["ocr_confidence_low"])]
    c = tagger.counts(insts)
    assert c["primary"]["SROIE/paddleocr"]["value_not_in_ocr"] == 2
    assert c["primary"]["SROIE/paddleocr"]["wrong_total"] == 1
    assert c["tags"]["SROIE/paddleocr"]["ocr_confidence_low"] == 1
    assert c["subtypes"]["wrong_total"]["scale_error"] == 1


def test_sample_review_spreads_over_categories_and_skips_nonfailures():
    insts = ([inst("value_not_in_ocr", receipt_id=f"a{i}") for i in range(10)] +
             [inst("wrong_total", receipt_id=f"b{i}") for i in range(10)] +
             [inst("normalised_only_mismatch", receipt_id=f"c{i}") for i in range(10)])
    picked = tagger.sample_review(insts, n=6, seed=42)
    assert len(picked) == 6
    cats = {p["primary"] for p in picked}
    assert "normalised_only_mismatch" not in cats          # non-failures excluded
    assert cats == {"value_not_in_ocr", "wrong_total"}     # both failure cats represented
    assert tagger.sample_review(insts, n=6, seed=42) == picked   # deterministic


def test_write_review_and_write_tags_round_trip(tmp_path):
    insts = [inst("value_not_in_ocr", tags=["value_not_in_ocr"]),
             inst("wrong_total", subtype="scale_error")]

    review = tmp_path / "review.csv"
    tagger.write_review(insts, review)
    rows = list(csv.DictReader(review.open(encoding="utf-8")))
    assert [r["review_id"] for r in rows] == ["R01", "R02"]
    assert rows[0]["auto_primary"] == "value_not_in_ocr"

    out = tagger.write_tags("testrun", insts, out_root=tmp_path / "tags")
    assert (out / "tags.csv").is_file()
    detail = [json.loads(line) for line in (out / "tags_detail.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(detail) == 2 and detail[0]["primary"] == "value_not_in_ocr"
    counts = json.loads((out / "counts.json").read_text(encoding="utf-8"))
    assert counts["primary"]["SROIE/paddleocr"]["wrong_total"] == 1


def test_fewshot_answers_reads_prompt_prefix(tmp_path):
    meta = {"engines": {"paddleocr": {"prompt_prefix_messages": [
        {"role": "user", "content": "OCR text..."},
        {"role": "assistant", "content": json.dumps({"company": "ACME SDN BHD", "total": "12.00"})},
    ]}}}
    (tmp_path / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    ans = tagger.fewshot_answers(tmp_path, "paddleocr")
    assert tagger.squash("ACME SDN BHD") in ans["company"]
    assert tagger.fewshot_answers(tmp_path / "nope", "paddleocr") == {}   # missing meta -> empty


# ---------------------------------------------------------------- by_model.py
def test_by_model_counts_by_cell_skips_non_failures():
    insts = [inst("value_not_in_ocr", model="Phi-3"),
             inst("wrong_entity", model="Phi-3"),
             inst("value_not_in_ocr", model="Qwen2.5-3B"),
             inst("normalised_only_mismatch", model="Phi-3")]       # excluded
    cells, totals = by_model.counts_by_cell(insts)
    assert cells["SROIE"]["paddleocr"]["Phi-3"]["wrong_entity"] == 1
    assert totals["SROIE"]["paddleocr"]["Phi-3"] == 2               # not the non-failure
    assert totals["SROIE"]["paddleocr"]["Qwen2.5-3B"] == 1


# ---------------------------------------------------------------- make_review_v1.py
def test_spread_round_robins_over_groups():
    rng = random.Random(0)
    items = [{"g": "a", "id": i} for i in range(5)] + [{"g": "b", "id": i} for i in range(5)]
    picked = make_review_v1.spread(items, 4, lambda it: it["g"], rng)
    assert len(picked) == 4
    assert {it["g"] for it in picked} == {"a", "b"}                 # diverse across groups


def test_ocr_hint():
    assert make_review_v1.ocr_hint({"tags": ["ocr_empty_page", "x"]}) == "empty_page"
    assert make_review_v1.ocr_hint({"tags": []}) == ""
    assert make_review_v1.ocr_hint({}) == ""


# ---------------------------------------------------------------- show_case.py
def test_show_case_squash():
    assert show_case.squash("Teh Tarik!") == "TEHTARIK"
    assert show_case.squash(None) == ""


def test_show_case_find_returns_last_match(tmp_path):
    p = tmp_path / "run.jsonl"
    rows = [
        {"engine": "paddleocr", "receipt_id": "R1", "field": "total", "v": 1},
        {"engine": "paddleocr", "receipt_id": "R1", "field": "total", "v": 2},   # later wins
        {"engine": "tesseract", "receipt_id": "R1", "field": "total", "v": 9},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert show_case.find(p, "paddleocr", "R1")["v"] == 2
    assert show_case.find(p, "tesseract", "R1", "total")["v"] == 9
    assert show_case.find(p, "paddleocr", "NOPE") is None
    assert show_case.find(tmp_path / "missing.jsonl", "paddleocr", "R1") is None

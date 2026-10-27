"""Failure tagger: one synthetic case per category."""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tag", REPO / "evaluation" / "failure_taxonomy" / "tag.py")
tag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tag)

GOLD = {"company": "OJC MARKETING SDN BHD", "date": "15/01/2019", "address": "NO 2, JALAN BAYU 4, MASAI",
        "total": "193.00"}
OCR_ROWS = ["OJC MARKETING SDN BHD", "BUYER TRADING SDN BHD", "NO 2, JALAN BAYU 4, MASAI", "TEL 07-388",
            "Date :15/01/2019", "SUBTOTAL 180.00", "TOTAL: 193.00", "CASH 200.00"]


def record(locale="my", fields=None):
    return {"id": "r1", "image": "x.jpg", "locale": locale, **({"fields_available": fields} if fields else {}),
            "ground_truth": {"raw": dict(GOLD), "normalized": {"date": "2019-01-15", "total": "193.00"}}}


def row(pred, rows=OCR_ROWS, failures=(), raw=None, conf=0.99, error=None):
    lines = [{"text": t, "confidence": conf} for t in rows]
    lenient = None if pred is None else {**GOLD, **pred}
    return {"engine": "e", "receipt_id": "r1", "error": error,
            "ocr": {"text": "\n".join(rows), "lines": lines},
            "llm": {"raw_output": raw},
            "outcome": {"lenient": lenient, "failures": [{"type": f} for f in failures]}}


def primaries(rec, r, fewshot=None):
    return {i["field"]: (i["primary"], i["subtype"]) for i in tag.tag_receipt(rec, r, "run", "SROIE", fewshot)}


@pytest.mark.parametrize("pred, field, expected", [
    ({"total": "180.00"}, "total", ("wrong_total", "other_amount_on_receipt")),
    ({"total": "193,000"}, "total", ("wrong_total", "scale_error")),
    ({"total": "999.99"}, "total", ("wrong_total", "amount_not_in_ocr")),
    ({"date": "Tuesday"}, "date", ("wrong_date_format", "unparseable")),
    ({"date": "20/02/2019"}, "date", ("wrong_value", "other_date")),
    ({"date": "15/01/2O19"}, "date", ("copy_corruption", None)),
    ({"company": "BUYER TRADING SDN BHD"}, "company", ("wrong_entity", None)),
    ({"company": "OJC MARKETING"}, "company", ("boundary_too_narrow", None)),
    ({"address": "NO 2, JALAN BAYU 4, MASAI, TEL 07-388"}, "address", ("boundary_too_wide", "adjacent_text")),
    ({"address": "NO 2, JALAN BAYU 4, MASAI, JOHOR"}, "address", ("boundary_too_wide", "hallucinated_extension")),
    ({"company": None}, "company", ("empty_field", None)),
    ({"company": "OJC  MARKETING SDN. BHD."}, "company", ("normalised_only_mismatch", None)),
])
def test_field_categories(pred, field, expected):
    assert primaries(record(), row(pred))[field] == expected


def test_day_month_swap():
    rec = record()
    rec["ground_truth"]["raw"]["date"], rec["ground_truth"]["normalized"]["date"] = "05/01/2019", "2019-01-05"
    r = row({"date": "01/05/2019"}, rows=["Date 05/01/2019", "TOTAL: 193.00"] + OCR_ROWS[:3])
    assert primaries(rec, r)["date"] == ("wrong_date_format", "day_month_swap")


def test_malformed_output_subtypes():
    derail = row(None, failures=["truncated", "invalid_json"], raw='{"company": "OJC EXERCISE 1:')
    assert set(primaries(record(), derail).values()) == {("malformed_json", "derailment")}
    prose = row(None, failures=["no_json"], raw="I cannot generate a JSON object.")
    assert set(primaries(record(), prose).values()) == {("malformed_json", "prose_instead_of_json")}


def test_value_not_in_ocr_and_ocr_tags():
    r = row({"total": None}, rows=["OJC MARKETING SDN BHD"], conf=0.5)
    inst = {i["field"]: i for i in tag.tag_receipt(record(), r, "run", "SROIE")}
    assert inst["total"]["primary"] == "value_not_in_ocr" and inst["total"]["stage"] == "ocr"
    assert {"empty_field", "value_not_in_ocr", "ocr_empty_page", "ocr_confidence_low:receipt"} <= set(inst["total"]["tags"])


def test_low_confidence_on_the_gold_piece():
    rows = [{"text": t, "confidence": 0.6 if "TOTAL" in t else 0.99} for t in OCR_ROWS]
    r = row({"total": "180.00"})
    r["ocr"]["lines"] = rows
    inst = {i["field"]: i for i in tag.tag_receipt(record(), r, "run", "SROIE")}
    assert "ocr_confidence_low" in inst["total"]["tags"]


def test_fewshot_copy():
    fewshot = {"total": {"3710"}, "date": {"140318"}}
    assert primaries(record(), row({"total": "37.10"}), fewshot)["total"] == ("fewshot_copy", None)


def test_cord_scores_total_and_flags_unlabelled_fields():
    rec = record(locale="id", fields=["total"])
    rec["ground_truth"]["raw"] = {"company": None, "date": None, "address": None, "total": "60.000"}
    rec["ground_truth"]["normalized"] = {"date": None, "total": "60000.00"}
    r = row({"company": "ES KOPI SUSU", "date": None, "address": None, "total": "60.00"},
            rows=["ES KOPI SUSU", "TOTAL 60.000"])
    got = primaries(rec, r)
    assert got == {"company": ("hallucinated_unlabelled_field", None), "total": ("wrong_total", "scale_error")}


def test_review_sample_spreads_over_categories():
    inst = [{"primary": p, "id": k} for k, p in enumerate(["value_not_in_ocr"] * 10 + ["wrong_total"] * 3 +
                                                           ["malformed_json"] * 2 + ["normalised_only_mismatch"] * 5)]
    picked = tag.sample_review(inst, n=6, seed=1)
    assert sorted(p["primary"] for p in picked) == ["malformed_json"] * 2 + ["value_not_in_ocr"] * 2 + ["wrong_total"] * 2
    assert picked == tag.sample_review(inst, n=6, seed=1)

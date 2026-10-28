"""UI service-layer tests. None of these load a model: extraction rows are
hand-built, so scoring and tagging are exercised without OCR or the LLM."""

import io
import zipfile

from PIL import Image

import app.service as service


# ------------------------------------------------------------------ records & ground truth
def test_ground_truth_normalises_date_and_total():
    gt = service.ground_truth({"company": "ACME", "date": "01/02/2018", "total": "RM 10.00"}, locale="my")
    assert gt["raw"]["company"] == "ACME"
    assert gt["normalized"]["date"] == "2018-02-01"      # day-first (Malaysian)
    assert gt["normalized"]["total"] == "10.00"


def test_make_record_sets_fields_available_from_labels():
    rec = service.make_record("x.jpg", rid="r1", locale="id", gold={"total": "60.000"})
    assert rec["id"] == "r1" and rec["locale"] == "id"
    assert rec["fields_available"] == ["total"]           # only the labelled field is scored
    assert rec["ground_truth"]["normalized"]["total"] == "60000.00"
    assert service.has_labels(rec) and service.dataset_label(rec) == "CORD"


def test_make_record_without_labels_has_no_ground_truth():
    rec = service.make_record("x.jpg")
    assert not service.has_labels(rec) and service.dataset_label(rec) == "SROIE"


# ------------------------------------------------------------------ OCR box overlay
def test_draw_boxes_keeps_image_size(tmp_path):
    p = tmp_path / "src.png"
    Image.new("RGB", (10, 12), "white").save(p)
    lines = [{"text": "TOTAL", "box": [1, 1, 6, 4], "confidence": 0.95},
             {"text": "10.00", "box": [1, 5, 6, 9], "confidence": 0.5}]
    out = service.draw_boxes(p, lines)
    assert out.size == (10, 12)


def test_confidence_color_bins():
    assert service.confidence_color(0.95) == (34, 197, 94)     # green
    assert service.confidence_color(0.8) == (234, 179, 8)      # amber
    assert service.confidence_color(0.4) == (239, 68, 68)      # red
    assert service.confidence_color(None) == (34, 197, 94)     # missing -> treated as high


# ------------------------------------------------------------------ ZIP ingestion
def _png_bytes(w=12, h=16):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, "PNG")
    return buf.getvalue()


def _zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_ingest_zip_builds_records(tmp_path):
    labels = "id,image,company,date,address,total\nr1,r1.png,ACME,01/02/2018,MAIN ST,10.00\n"
    data = _zip({"images/r1.png": _png_bytes(), "labels.csv": labels})
    records, messages = service.ingest_zip(data, tmp_path, locale="my")
    assert len(records) == 1
    rec = records[0]
    assert rec["id"] == "r1" and service.has_labels(rec)
    assert rec["ground_truth"]["normalized"]["total"] == "10.00"
    assert (tmp_path / "images" / "r1.png").is_file()


def test_ingest_zip_flattens_paths_and_matches_by_stem(tmp_path):
    # a traversal-style name must be flattened to its basename; image column omitted
    labels = "id,total\nr1,5.00\n"
    data = _zip({"../../evil/r1.png": _png_bytes(), "labels.csv": labels})
    records, _ = service.ingest_zip(data, tmp_path, locale="my")
    assert len(records) == 1
    assert (tmp_path / "images" / "r1.png").is_file()           # extracted flat, inside dest
    assert not (tmp_path.parent / "evil").exists()              # nothing escaped


def test_ingest_zip_without_labels_reports_error(tmp_path):
    data = _zip({"images/r1.png": _png_bytes()})
    records, messages = service.ingest_zip(data, tmp_path)
    assert records == [] and any("labels" in m for m in messages)


# ------------------------------------------------------------------ scoring & tagging on a hand-built row
def _row(engine="paddleocr", pred=None, ocr_text="", failures=()):
    return {
        "engine": engine, "error": None, "power": {"source": "ac"},
        "timings": {"ocr_s": 1.0, "llm_s": 1.0, "total_s": 2.0},
        "ocr": {"text": ocr_text, "lines": [{"text": ocr_text, "box": [0, 0, 9, 4], "confidence": 0.95}]},
        "outcome": {"lenient": pred, "strict_valid": not failures, "schema_valid": not failures,
                    "failures": [{"type": t} for t in failures]},
    }


def test_score_reports_correct_and_wrong_fields():
    rec = service.make_record("x.jpg", rid="r1", locale="my",
                              gold={"company": "ACME", "date": "01/02/2018", "address": "MAIN ST", "total": "10.00"})
    row = _row(pred={"company": "ACME", "date": "01/02/2018", "address": "OTHER CO", "total": "10.00"},
               ocr_text="ACME\n01/02/2018\nMAIN ST\nTOTAL 10.00")
    result = service.score(rec, row)
    assert result["fields"]["company"]["normalized"] is True
    assert result["fields"]["total"]["normalized"] is True
    assert result["fields"]["address"]["normalized"] is False
    assert result["fields"]["address"]["in_ocr"] is True        # the gold address WAS in the OCR text


def test_tags_classify_a_wrong_field():
    rec = service.make_record("x.jpg", rid="r1", locale="my",
                              gold={"company": "ACME", "date": "01/02/2018", "address": "MAIN ST", "total": "10.00"})
    row = _row(pred={"company": "ACME", "date": "01/02/2018", "address": "OTHER CO", "total": "10.00"},
               ocr_text="ACME\n01/02/2018\nMAIN ST\nTOTAL 10.00")
    instances = service.tags(rec, row)
    addr = [i for i in instances if i["field"] == "address"]
    assert addr and addr[0]["primary"]                          # a non-empty failure category
    assert addr[0]["stage"] in {"ocr", "llm_extraction", "llm_output"}


def test_explain_turns_categories_into_plain_language():
    assert "OCR stage did not read" in service.explain({"primary": "value_not_in_ocr", "tags": []})
    assert "blank or failed read" in service.explain(
        {"primary": "value_not_in_ocr", "tags": ["ocr_empty_page"]})
    assert "100 or 1000" in service.explain({"primary": "wrong_total", "subtype": "scale_error"})
    assert "cash" in service.explain(
        {"primary": "wrong_total", "subtype": "other_amount_on_receipt:cash"})
    assert "different business" in service.explain({"primary": "wrong_entity", "subtype": None})
    assert "not clean JSON" in service.explain({"primary": "malformed_json", "subtype": "derailment"})


def test_score_and_tags_need_labels():
    rec = service.make_record("x.jpg")                          # no gold
    row = _row(pred={"company": "ACME", "date": None, "address": None, "total": None})
    assert service.score(rec, row) is None
    assert service.tags(rec, row) == []

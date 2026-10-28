"""Extra tests for app/service.py logic not covered by test_app.py:
failure explanations, built-in manifests, manifest round-trip, label parsing.
No models or OCR — pure logic and small file round-trips."""

from pathlib import Path

try:
    import service
except ModuleNotFoundError:
    from app import service


def expl(primary, subtype=None, tags=()):
    return service.explain({"primary": primary, "subtype": subtype, "tags": list(tags)})


def test_explain_covers_each_stage():
    assert "OCR" in expl("value_not_in_ocr")
    assert "almost nothing" in expl("value_not_in_ocr", tags=["ocr_empty_page"])
    assert "not clean JSON" in expl("malformed_json", "derailment")
    assert "swapped" in expl("wrong_date_format", "day_month_swap")
    assert "factor of 100" in expl("wrong_total", "scale_error")
    assert "cash" in expl("wrong_total", "other_amount_on_receipt:cash")
    assert "extra text" in expl("boundary_too_wide", "adjacent_text")
    assert expl("wrong_entity")                 # non-empty reason
    assert expl("boundary_too_narrow")


def test_is_vlm_engine():
    assert service.is_vlm_engine(service.VLM_MODEL)
    assert not service.is_vlm_engine("paddleocr")


def test_builtin_manifests_are_real_files_when_present():
    # the subset manifests are gitignored, so this is a no-op in CI and a real
    # check locally; either way any returned path must point at a real file
    for path in service.builtin_manifests().values():
        assert Path(path).is_file()


def test_manifest_round_trip(tmp_path):
    records = [service.make_record("/x/R1.png", rid="R1", gold={"total": "12.00"}),
               service.make_record("/x/R2.png", rid="R2")]
    path = service.write_manifest(records, tmp_path / "m.jsonl")
    back = service.load_manifest(path)
    assert [r["id"] for r in back] == ["R1", "R2"]
    assert service.has_labels(back[0]) and not service.has_labels(back[1])
    assert service.dataset_label(back[0]) == "SROIE"


def test_read_label_rows_csv_and_jsonl():
    csv_rows = service._read_label_rows(b"id,total\nR1,12.00\n", "labels.csv")
    assert csv_rows[0]["total"] == "12.00"
    jl = service._read_label_rows(b'{"id":"R1","total":"12.00"}\n', "labels.jsonl")
    assert jl[0]["id"] == "R1"


def test_fewshot_zero_shot_is_empty():
    ids, examples = service.fewshot("paddleocr", shots=0)
    assert ids == [] and examples == []


def test_evaluatable_records_drops_train_split():
    recs = [{"id": "a", "split": "train"}, {"id": "b", "split": "test"},
            {"id": "c", "split": "validation"}]
    assert [r["id"] for r in service.evaluatable_records(recs)] == ["b", "c"]


def test_ocr_cached_false_for_unknown_receipt():
    assert service.ocr_cached("paddleocr", {"id": "not-cached-xyz", "image": "/x/y.png"}) is False


def test_full_manifests_are_real_files_when_present():
    for path in service.full_manifests().values():
        assert Path(path).is_file()

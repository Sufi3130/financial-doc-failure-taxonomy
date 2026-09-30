"""SROIE and CORD preparation scripts, run against the synthetic fixtures."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_prepare(dataset):
    path = REPO / "evaluation" / "datasets" / dataset / "prepare.py"
    spec = importlib.util.spec_from_file_location(f"prepare_{dataset.lower()}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ SROIE
@pytest.fixture
def sroie(monkeypatch):
    mod = load_prepare("SROIE")
    monkeypatch.setattr(mod, "RAW", FIXTURES / "sroie" / "raw")
    return mod


@pytest.fixture
def sroie_records(sroie):
    train, train_stats = sroie.scan_split("train")
    test, test_stats = sroie.scan_split("test")
    return train + test, train_stats, test_stats


def by_id(records):
    return {r["id"]: r for r in records}


def test_sroie_pairs_by_exact_stem_and_skips_numbered_copies(sroie, capsys):
    train, train_stats = sroie.scan_split("train")
    _, test_stats = sroie.scan_split("test")
    assert train_stats["pairs"] == 6 and test_stats["pairs"] == 6
    assert train_stats["numbered_copies_ignored"] == 2  # X0001(1).jpg + .txt
    assert train_stats["numbered_copies_not_identical_to_base"] == []
    assert train_stats["jpg_without_txt"] == ["X0004"]
    assert "X0001(1)" not in by_id(train)
    assert "ignored 2 Windows numbered copies" in capsys.readouterr().out


def test_sroie_label_issues_are_recorded_not_dropped(sroie_records):
    r = by_id(sroie_records[0])
    assert r["X0002"]["issues"] == ["missing:address"]
    assert r["X0002"]["ground_truth"]["raw"]["address"] is None
    assert r["X0003"]["issues"] == ["empty:total"]
    assert r["X0001"]["issues"] == []


def test_sroie_normalised_values_keep_raw(sroie_records):
    r = by_id(sroie_records[0])
    assert r["T0001"]["ground_truth"]["raw"]["total"] == "1,007.50"
    assert r["T0001"]["ground_truth"]["normalized"] == {"date": "2019-01-12", "total": "1007.50"}
    assert r["T0004"]["ground_truth"]["normalized"] == {"date": "2018-04-22", "total": "-1.73"}
    assert r["T0006"]["ground_truth"]["normalized"]["date"] == "2016-12-06"
    assert r["X0001"]["image"] == "tests/fixtures/sroie/raw/train/X0001.jpg"


def test_sroie_duplicates_within_and_across_splits(sroie, sroie_records):
    records = sroie_records[0]
    within, cross = sroie.mark_duplicates(records)
    assert within == {"train": [["X0005", "X0006"]], "test": [["T0001", "T0005"]]}
    assert cross == [["test/T0003", "train/X0007"]]
    r = by_id(records)
    assert r["X0006"]["duplicate_of"] == "X0005" and r["T0005"]["duplicate_of"] == "T0001"
    assert r["T0003"]["duplicate_of"] is None  # cross-split only: reported, not marked


def test_sroie_label_conflicts_report_field_names_only(sroie, sroie_records):
    conflicts = sroie.label_conflicts(sroie_records[0])
    assert conflicts == [{"receipts": ["train/X0005", "train/X0006"], "fields": ["company"]}]
    assert "DELTA" not in json.dumps(conflicts)


def test_sroie_subset_avoids_duplicates_and_is_seeded(sroie, sroie_records):
    records = sroie_records[0]
    sroie.mark_duplicates(records)
    ids, eligible = sroie.pick_subset(records, 3, seed=42)
    assert eligible == 5  # 6 test receipts minus the duplicate T0005
    assert "T0005" not in ids and len(ids) == 3
    assert ids == sroie.pick_subset(records, 3, seed=42)[0]
    with pytest.raises(SystemExit):
        sroie.pick_subset(records, 6, seed=42)


def test_sroie_field_stats(sroie, sroie_records):
    stats = sroie.field_stats([r for r in sroie_records[0] if r["split"] == "train"])
    assert stats == {"missing": {"address": 1}, "empty": {"total": 1}, "unparsed": {}}


# ------------------------------------------------------------------ CORD
@pytest.fixture
def cord(monkeypatch):
    mod = load_prepare("CORD")
    monkeypatch.setattr(mod, "RAW", FIXTURES / "cord" / "raw" / "cord-v2")
    return mod


@pytest.fixture
def cord_records(cord):
    records = []
    for split in cord.SPLITS:
        for image, gt in cord.read_split(split):
            records.append(cord.build_record(split, image, json.loads(gt)))
    return records


def test_cord_reads_arrow_and_builds_ids(cord_records):
    assert [r["id"] for r in cord_records] == [
        "train_000", "train_001", "validation_000", "validation_001", "test_000", "test_001"]
    r = cord_records[0]
    assert r["locale"] == "id" and r["fields_available"] == ["total"]
    assert r["ground_truth"]["raw"] == {"company": None, "date": None, "address": None, "total": "Rp 60.000"}
    assert r["image"] == "evaluation/datasets/CORD/raw/export/train/train_000.png"


def test_cord_total_mapping_and_issues(cord_records):
    r = by_id(cord_records)
    assert r["train_000"]["ground_truth"]["normalized"]["total"] == "60000.00"
    assert r["train_001"]["issues"] == ["missing:total"]
    assert r["validation_000"]["issues"] == ["multiple:total"]
    assert r["validation_000"]["ground_truth"]["raw"]["total"] == "55,834 | 55,800"
    assert r["validation_001"]["issues"] == ["irregular:total"]
    assert r["test_000"]["ground_truth"]["normalized"]["total"] == "35000.00"
    assert r["test_001"]["ground_truth"]["normalized"]["total"] == "47499.00"


def test_cord_line_items_always_lists(cord_records):
    r = by_id(cord_records)
    single = r["train_000"]["ground_truth"]["line_items"]
    assert single == [{"nm": "ITEM", "cnt": "1", "price": "10.000", "sub": [], "price_normalized": "10000.00"}]
    items = r["validation_001"]["ground_truth"]["line_items"]
    assert [i["nm"] for i in items] == ["A", "B"]
    assert items[0]["sub"] == [{"nm": "EXTRA", "price": "5.000", "sub": [], "price_normalized": "5000.00"}]
    assert items[1]["price_normalized"] == "32000.00"


def test_cord_duplicates_and_key_paths(cord, cord_records):
    within, cross = cord.mark_duplicates(cord_records)
    assert within == {} and cross == [["test/test_001", "train/train_000"]]
    gt = {"menu": [{"nm": "A", "sub": {"nm": "B"}}], "total": {"total_price": "1"}}
    assert cord.key_paths(gt) == {"menu", "menu.nm", "menu.sub", "menu.sub.nm", "total", "total.total_price"}

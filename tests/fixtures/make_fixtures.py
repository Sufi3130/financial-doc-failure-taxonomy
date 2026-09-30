"""Generate the tiny synthetic test fixtures (no real receipts, no model files).

    python tests/fixtures/make_fixtures.py

Each fixture reproduces a quirk seen in the real data; the comments say which.
Outputs are committed, so tests never need to run this; re-run it only to
change the fixtures.
"""

import io
import json
import shutil
from pathlib import Path

import pyarrow as pa
from PIL import Image

HERE = Path(__file__).resolve().parent


def image_bytes(color, fmt):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format=fmt)
    return buf.getvalue()


def reset(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


# ---------------------------------------------------------------- SROIE
def write_sroie():
    root = HERE / "sroie" / "raw"
    reset(root)
    label = lambda company, date, address, total: {"company": company, "date": date,  # noqa: E731
                                                     "address": address, "total": total}
    receipts = {
        "train": {
            "X0001": ("red", label("ALPHA SDN BHD", "25/12/2018", "NO 1, JALAN A", "RM 9.00")),
            # missing address key (real: X51005663280)
            "X0002": ("green", {"company": "BETA", "date": "05 MAR 2018", "total": "12.50"}),
            # empty total (real: X51005433522)
            "X0003": ("blue", label("GAMMA", "2018-03-23", "NO 3", "")),
            # same image, conflicting company label (real: SDN BHD vs SDN. BHD.)
            "X0005": ("olive", label("DELTA SDN BHD", "1/1/2018", "NO 5", "1.00")),
            "X0006": ("olive", label("DELTA SDN. BHD.", "1/1/2018", "NO 5", "1.00")),
            # byte-identical to test T0003 (real: 7 train/test overlaps)
            "X0007": ("purple", label("EPSILON", "02/JAN/2017", "NO 7", "$8.20")),
        },
        "test": {
            "T0001": ("white", label("ZETA", "12-01-19", "NO 11", "1,007.50")),
            "T0002": ("gray", label("ETA", "OCT 3, 2016", "NO 12", "RM41.45")),
            "T0003": ("purple", label("EPSILON", "02/JAN/2017", "NO 7", "$8.20")),
            "T0004": ("orange", label("THETA", "4/22/2018", "NO 14", "-1.73")),
            # within-test duplicate of T0001 under a higher ID
            "T0005": ("white", label("ZETA", "12-01-19", "NO 11", "1,007.50")),
            "T0006": ("cyan", label("IOTA", "(06/12/2016)", "NO 16", "RM 3.90")),
        },
    }
    for split, items in receipts.items():
        folder = root / split
        folder.mkdir()
        for rid, (color, lab) in items.items():
            (folder / f"{rid}.jpg").write_bytes(image_bytes(color, "JPEG"))
            (folder / f"{rid}.txt").write_text(json.dumps(lab, indent=4), encoding="utf-8")

    train = root / "train"
    # Windows numbered copies, byte-identical to their base (real: 359 of them)
    shutil.copy(train / "X0001.jpg", train / "X0001(1).jpg")
    shutil.copy(train / "X0001.txt", train / "X0001(1).txt")
    # image without a label file
    (train / "X0004.jpg").write_bytes(image_bytes("black", "JPEG"))


# ---------------------------------------------------------------- CORD
def cord_gt(split, image_id, total=None, menu=None, extra_total=None):
    total_block = dict(extra_total or {})
    if total is not None:
        total_block["total_price"] = total
    gt_parse = {"menu": menu or {"nm": "ITEM", "cnt": "1", "price": "10.000"}, "total": total_block}
    return json.dumps({"gt_parse": gt_parse, "meta": {"version": "2.0.0", "split": split, "image_id": image_id,
                                                       "image_size": {"width": 8, "height": 8}},
                       "valid_line": [], "roi": {}, "repeating_symbol": [], "dontcare": []})


def write_cord():
    root = HERE / "cord" / "raw" / "cord-v2"
    reset(root)
    rows = {
        "train": [
            ("red", cord_gt("train", 0, "Rp 60.000")),            # single-item menu as a dict
            ("green", cord_gt("train", 1, None, extra_total={"cashprice": "50.000"})),  # no total_price
        ],
        "validation": [
            ("blue", cord_gt("validation", 0, ["55,834", "55,800"])),  # total given twice
            ("olive", cord_gt("validation", 1, "57,0000", menu=[        # irregular grouping
                {"nm": "A", "price": "20.000", "sub": {"nm": "EXTRA", "price": "5.000"}},
                {"nm": "B", "price": "32,000"}])),
        ],
        "test": [
            ("white", cord_gt("test", 0, "35.000,00")),              # decimal part after thousands dot
            ("red", cord_gt("test", 1, "TOTAL 47,499")),             # same image as train_000
        ],
    }
    schema = pa.schema([("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                        ("ground_truth", pa.string())])
    for split, items in rows.items():
        (root / split).mkdir()
        table = pa.table({
            "image": [{"bytes": image_bytes(c, "PNG"), "path": None} for c, _ in items],
            "ground_truth": [gt for _, gt in items],
        }, schema=schema)
        with pa.ipc.new_stream(root / split / "data-00000-of-00001.arrow", schema) as writer:
            writer.write_table(table)
    (root / "dataset_dict.json").write_text(json.dumps({"splits": list(rows)}), encoding="utf-8")


# ---------------------------------------------------------------- pipeline
def write_pipeline():
    root = HERE / "pipeline"
    reset(root)
    gt = lambda company, date, address, total, nd, nt: {  # noqa: E731
        "raw": {"company": company, "date": date, "address": address, "total": total},
        "normalized": {"date": nd, "total": nt}}
    manifest = [
        {"id": "R1", "split": "test", "image": "tests/fixtures/pipeline/R1.png", "locale": "my",
         "ground_truth": gt("OJC MARKETING SDN BHD", "15/01/2019", "NO 2, JALAN BAYU 4", "193.00",
                            "2019-01-15", "193.00")},
        {"id": "R2", "split": "test", "image": "tests/fixtures/pipeline/R2.png", "locale": "my",
         "ground_truth": gt("MR. D.I.Y. (M) SDN BHD", "14-03-18", "LOT 1851-A", "RM 37.10",
                            "2018-03-14", "37.10")},
        {"id": "R3", "split": "test", "image": "tests/fixtures/pipeline/R3.png", "locale": "my",
         "ground_truth": gt("SWC ENTERPRISE", "06/03/2018", "28-G, GROUND FLOOR", "4.00",
                            "2018-03-06", "4.00")},
    ]
    for rec, color in zip(manifest, ("white", "gray", "yellow")):
        Image.new("RGB", (8, 8), color).save(root / f"{rec['id']}.png")
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest), encoding="utf-8")

    ocr = {  # reading-order text the fake OCR engine returns, one row per piece
        "R1": ["OJC MARKETING SDN BHD", "NO 2, JALAN BAYU 4", "Date :15/01/2019", "TOTAL: 193.00"],
        "R2": ["MR. D.I.Y. (M) SDN BHD", "LOT 1851-A", "14-03-18 21:49", "Total Incl. GST RM 37.10"],
        "R3": [],  # empty OCR (real: Tesseract on X51005749904)
    }
    llm = {  # raw answers the fake LLM returns
        # fenced but otherwise correct
        "R1": '```json\n{"company": "OJC MARKETING SDN BHD", "date": "15/01/2019", '
              '"address": "NO 2, JALAN BAYU 4", "total": "193.00"}\n```',
        # derailment (real: MR. D.I.Y. receipts)
        "R2": '{"company": "MR. D.I EXERCISE 1:\n\nGiven the receipt OCR text, extract',
        # prose refusal on empty OCR
        "R3": "Since the OCR text provided is incomplete, I cannot generate a JSON object.",
    }
    (root / "ocr.json").write_text(json.dumps(ocr, indent=1), encoding="utf-8")
    (root / "llm_outputs.json").write_text(json.dumps(llm, indent=1), encoding="utf-8")


if __name__ == "__main__":
    write_sroie()
    write_cord()
    write_pipeline()
    total = sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())
    print(f"fixtures written to {HERE} ({total / 1024:.1f} KB)")

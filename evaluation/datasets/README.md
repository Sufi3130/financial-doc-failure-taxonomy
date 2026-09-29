# Datasets

This folder holds scripts to download and preprocess evaluation datasets —
it does not contain the raw data itself (too large to commit, and SROIE/CORD
have their own licensing terms that don't permit redistribution).

## SROIE

ICDAR 2019 Scanned Receipt OCR and Information Extraction competition
dataset, **Task 3 (key information extraction)**. Each receipt is `<ID>.jpg`
plus `<ID>.txt` holding JSON with `company`, `date`, `address`, `total` (all
strings).

- **Release:** official ICDAR 2019 RRC release, https://rrc.cvc.uab.es/?ch=13
  (registration required)
- **Downloaded:** 2026-09-29
- **Location:** `SROIE/raw/{train,test}/` (gitignored)

### Preparing the manifest

```bash
python evaluation/datasets/SROIE/prepare.py        # --seed 42 --n 50 by default
```

The script pairs each jpg with its txt by exact file stem, validates the four
keys, hashes images to find duplicates, and normalises `date` (ISO
`YYYY-MM-DD`, day-first) and `total` (2-decimal string, currency removed) with
[`evaluation/normalize.py`](../normalize.py). The evaluator uses the same module
on model predictions. Raw label strings are kept next to the normalised ones.
It exits with an error unless it finds 626 train / 347 test pairs.

| Output | Contents | In git |
|---|---|---|
| `SROIE/manifest.jsonl` | one record per receipt: id, split, image path, sha256, `duplicate_of`, `issues`, raw + normalised ground truth | no (contains labels) |
| `SROIE/subset_test_50.jsonl` | the 50-receipt evaluation subset, same format | no (contains labels) |
| `SROIE/subset_test_50_ids.txt` | IDs of that subset | yes |
| `SROIE/prepare_report.json` | counts and data-quality findings (no label text) | yes |

The **evaluation subset** is 50 test receipts drawn with `random.Random(42)`
from the receipts that have no label issues and are not a duplicate of a lower
ID (344 eligible).

### Counts and quirks observed (2026-09-29 download)

| | train | test |
|---|---|---|
| jpg / txt pairs | **626** | **347** |
| Windows numbered copies ignored (`X…(1).jpg`, all byte-identical to their base) | 359 | 0 |
| unmatched jpg / txt | 0 / 0 | 0 / 0 |
| missing field | 1 (`address`: X51005663280) | 0 |
| empty field | 1 (`total`) | 0 |
| duplicate image groups (same image, different IDs) | 5 | 3 |

- **Train/test overlap:** 7 test receipts are byte-identical to a train
  receipt (e.g. test `X51009453729` = train `X51005453729`). Nothing in this
  project is trained on SROIE, but train receipts must not be used as few-shot
  prompt examples for these test receipts.
- **Label noise:** 5 duplicate groups have different labels for the same
  image (spacing in addresses, `SDN BHD` vs `SDN. BHD.`, `RM7.42` vs
  `7.42`). Exact-match scores are therefore capped slightly below 100% even
  for a perfect extractor, which is why the normalised-match score exists.
- **Dates** appear in about 17 formats (`25/12/2018`, `12-01-19`,
  `05 MAR 2018`, `20180304`, `OCT 3, 2016`, …). All are day-first except
  where that is impossible (`4/22/2018`). **Totals** appear as `9.00`,
  `RM41.45`, `RM 3.90`, `$8.20`, `1,007.50` and `-1.73`. All normalise.

See `SROIE/prepare_report.json` for the full lists.

## CORD

Consolidated Receipt Dataset for post-OCR parsing. 800/100/100 train/val/
test receipts, 30 entity types under Menu/Void/Subtotal/Total categories.
Indonesian receipts.

Source: https://github.com/clovaai/cord

## Self-collected Hungarian dataset (optional extension)

Not started. If pursued, will contain manually collected/anonymized
Hungarian financial documents (invoices, bank statements, payslips) with
manually annotated ground truth. Collection process and anonymization
approach to be documented here once begun — this depends on GDPR-compliant
handling of any real documents, so synthetic or heavily redacted samples
are the likely path rather than real customer data.

## Structure once populated

```
evaluation/datasets/
├── SROIE/
│   ├── raw/                     # gitignored: downloaded receipts
│   ├── prepare.py               # builds the manifest + eval subset
│   ├── manifest.jsonl           # gitignored (generated)
│   ├── subset_test_50.jsonl     # gitignored (generated)
│   ├── subset_test_50_ids.txt   # generated, committed
│   └── prepare_report.json      # generated, committed
├── CORD/
│   ├── raw/           # gitignored
│   └── prepare.py
└── hungarian/          # optional extension, not yet started
```

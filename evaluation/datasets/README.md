# Datasets

This folder holds scripts to download and preprocess evaluation datasets —
it does not contain the raw data itself. The raw data is too large to commit,
and SROIE requires registration to download. CORD is CC BY 4.0, but it is
treated the same way for consistency: generated manifests that contain labels
stay out of git, while subset ID lists and preparation reports are committed.

Both datasets are converted to the same manifest format (one JSON object per
line) so the pipeline and evaluator can treat them alike:

| Key | Meaning |
|---|---|
| `id`, `split` | receipt ID and dataset split |
| `image` | repo-relative image path |
| `sha256`, `duplicate_of` | image hash; ID of the lower-ID identical image in the same split, if any |
| `issues` | label problems, e.g. `missing:total`, `empty:total`, `irregular:total` |
| `locale` | amount format used by [`evaluation/normalize.py`](../normalize.py): `my` (SROIE) or `id` (CORD). The evaluator must normalise predictions with the same locale. |
| `fields_available` | fields that are annotated and should be scored (CORD only; all four for SROIE) |
| `ground_truth.raw` | original strings for `company`, `date`, `address`, `total` (`null` if not annotated) |
| `ground_truth.normalized` | `date` as ISO `YYYY-MM-DD`, `total` as a 2-decimal string |
| `ground_truth.line_items` | CORD only: menu items (optional field) |

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

Consolidated Receipt Dataset for post-OCR parsing: Indonesian receipts,
amounts in rupiah.

- **Release:** CORD v2 from Hugging Face,
  [`naver-clova-ix/cord-v2`](https://huggingface.co/datasets/naver-clova-ix/cord-v2)
  (project: <https://github.com/clovaai/cord>), licence CC BY 4.0
- **Downloaded:** 2026-09-29, saved with `datasets.save_to_disk`
- **Location:** `CORD/raw/cord-v2/{train,validation,test}/` as Arrow stream
  files with two columns: `image` (PNG bytes) and `ground_truth` (a JSON string).
  Read with `pyarrow`; the full `datasets` library is not needed.

### `ground_truth` structure

Each `ground_truth` string is a JSON object with these top-level keys:

- `gt_parse`: the structured labels (below)
- `meta`: `version`, `split`, `image_id` (unique within a split), `image_size`
- `valid_line`: every labelled OCR line: words with quadrilateral boxes,
  text, `is_key`, `row_id`, plus the line's `category` (e.g. `menu.nm`,
  `total.total_price`) and `group_id`
- `roi`, `repeating_symbol`, `dontcare`: receipt region, repeated
  separator characters, and masked regions

`gt_parse` example (test_000), all values are strings:

```json
{"menu": {"nm": "-TICKET CP", "num": "901016", "cnt": "2", "price": "60.000", "itemsubtotal": "60.000"},
 "sub_total": {"subtotal_price": "60.000", "discount_price": "-60.000", "tax_price": "5.455"},
 "total": {"total_price": "60.000", "creditcardprice": "60.000", "menuqty_cnt": "2.00"}}
```

`menu` is an object for single-item receipts and a list otherwise. `menu.sub`
(add-ons such as extra toppings) follows the same rule. Field coverage over
all 1000 receipts:

```text
menu 1000        nm 995, price 997, cnt 910, unitprice 356, sub 157
                 (sub.nm 156, sub.price 75, sub.cnt 70, sub.unitprice 6),
                 num 45, discountprice 42, itemsubtotal 4, vatyn 3, etc 2
sub_total 681    subtotal_price 662, tax_price 445, service_price 123,
                 etc 76, discount_price 74, othersvc_price 1
total 998        total_price 972, cashprice 648, changeprice 620,
                 menuqty_cnt 283, creditcardprice 151, menutype_cnt 52,
                 emoneyprice 51, total_etc 33
void_menu 1      nm 1, price 1
```

### Field mapping to SROIE

| SROIE field | CORD source | Notes |
|---|---|---|
| `total` | `gt_parse.total.total_price` | normalised with `locale="id"` |
| `company` | — | store information is not annotated in CORD; `null` |
| `date` | — | not annotated; `null` |
| `address` | — | not annotated; `null` |

**CORD fields with no SROIE equivalent:**

- `menu.*`: kept as `ground_truth.line_items` (always a list, `sub` also a
  list, each item gets a `price_normalized`)
- `sub_total.*`: subtotal, tax, service charge, discount, other charges
- `total.*` except `total_price`: cash paid, change, credit card,
  e-money, item quantity/type counts, `total_etc`
- `void_menu.*` (cancelled items)
- `valid_line` word boxes and categories: not in the manifest, but kept in
  the exported per-receipt JSON (useful for OCR-level evaluation)

### Amount formats

Both `.` and `,` group thousands in rupiah: `60.000` and `60,000` both mean
60000. Only a final separator followed by exactly two digits is a
decimal part (`35.000,00`, `226,500.00`). Prefixes such as `Rp`, `Rp.` and
`TOTAL` are stripped. The same string means different amounts in the two
datasets (`"60.000"` is 60.00 ringgit on SROIE and 60000.00 rupiah on CORD),
which is why each record carries a `locale`. Grouping that does not follow
the 3-digit rule (`57,0000`, `385,0000`, `1178.100`) is parsed by dropping
the separators and flagged `irregular:total`.

### Preparing the manifest

```bash
python evaluation/datasets/CORD/prepare.py         # --seed 42 --n 50 by default
```

The script reads all three splits (about 2.3 GB) for stats and duplicate
checks. It exports only **validation + test** (200 receipts, ~460 MB) to
`CORD/raw/export/{split}/<id>.png` + `<id>.json` (the full original
`ground_truth`). IDs are `{split}_{image_id:03d}`, e.g. `test_007`. Train is
not exported: nothing is trained on it. The script exits with an error
unless it finds 800 / 100 / 100 receipts.

| Output | Contents | In git |
|---|---|---|
| `CORD/manifest.jsonl` | the 200 validation + test records | no (contains labels) |
| `CORD/subset_valtest_50.jsonl` | the 50-receipt evaluation subset, same format | no (contains labels) |
| `CORD/subset_valtest_50_ids.txt` | IDs of that subset | yes |
| `CORD/prepare_report.json` | counts, field coverage per split, issues, duplicates | yes |

The **evaluation subset** is 50 receipts drawn with `random.Random(42)` from
validation + test receipts that have exactly one regular `total_price` and
are not a duplicate (193 eligible).

### Counts and quirks observed (2026-09-29 download)

| | train | validation | test |
|---|---|---|---|
| receipts | **800** | **100** | **100** |
| `total_price` missing | 21 | 2 | 5 |
| `total_price` given twice (list) | 1 | 0 | 0 |
| irregular amount grouping | 3 | 0 | 0 |
| line items | 2105 | 221 | 251 |
| duplicate image groups | 2 | 0 | 0 |

- No image appears in more than one split.
- In 7 of the 200 validation + test receipts, `total_price` does not equal
  `cashprice − changeprice` (or is far from `subtotal_price`). This comes from
  the receipts or labels, not from parsing. The receipts are kept, but results
  on them should be checked by hand.

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
│   ├── raw/                        # gitignored: cord-v2/ Arrow files + export/
│   ├── prepare.py                  # builds the manifest + eval subset
│   ├── manifest.jsonl              # gitignored (generated)
│   ├── subset_valtest_50.jsonl     # gitignored (generated)
│   ├── subset_valtest_50_ids.txt   # generated, committed
│   └── prepare_report.json         # generated, committed
└── hungarian/          # optional extension, not yet started
```

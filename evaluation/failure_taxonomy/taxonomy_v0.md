# Failure taxonomy v0

First version of the failure taxonomy for the local OCR → small-LLM receipt
extraction pipeline. It's derived from automatic tagging of two pipeline runs
plus a manual review of 20 sampled failures.

## Scope

| | |
|---|---|
| Pipeline | PaddleOCR 3.7 or Tesseract 5.4 → Phi-3-mini-4k-instruct Q4 GGUF (llama-cpp-python, CPU), prompt `v1`, 2-shot (SROIE train `X51005676545`, `X51005361946`), temperature 0 |
| SROIE | 50-receipt test subset `subset_test_50` (seed 42), run `20260930-063807`; 4 fields scored → 200 field slots per engine |
| CORD v2 | 50-receipt validation + test subset `subset_valtest_50` (seed 42), run `20260930-092942`; only `total` scored → 50 field slots per engine |
| Unit of analysis | one **field** of one receipt for one OCR engine whose **normalised** value is wrong (a malformed output therefore counts once per scored field) |
| Tagger | `evaluation/failure_taxonomy/tag.py` (rules below); per-instance tags in `tags/<run_id>/tags.csv` |
| Manual review | 20 failures, seeded sample spread over categories → `review_v0.csv` |

Each failure gets **one primary category** and optional **extra tags**.

- The primary category is **the first pipeline stage that went wrong**, in
  this order: pipeline error → malformed output → OCR miss → LLM extraction
  error. So the counts add up without double counting.
- Extra tags (e.g. low OCR confidence) can overlap.

## 1. Summary

**210 field failures** in total (normalised mismatches only):

| Stage of first failure | SROIE PaddleOCR | SROIE Tesseract | CORD PaddleOCR | CORD Tesseract | Total |
|---|---|---|---|---|---|
| OCR: value never read | 33 | 67 | 0 | 21 | 121 |
| LLM output: no usable JSON | 8 | 8 | 0 | 14 | 30 |
| LLM extraction: value read but wrong | 29 | 14 | 8 | 8 | 59 |
| **Failures** | **70** / 200 | **89** / 200 | **8** / 50 | **43** / 50 | **210** |

**Key findings:**

- **Where failures start depends on the engine and dataset.** It should not
  be summarised as one number: the pooled 121 of 210 OCR failures is
  dominated by Tesseract.
  - **PaddleOCR + SROIE: roughly equal.** 33 OCR vs 37 LLM (8 output + 29
    extraction). Per field, OCR vs LLM misses are company 15 / 14,
    address 16 / 13, date 5 / 6, total 2 / 3.
  - **PaddleOCR + CORD: mostly LLM.** 0 OCR-first failures, 8 LLM. Per field
    (total): 2 OCR misses vs 8 LLM misses.
  - **Tesseract: mostly OCR.** SROIE 67 of 89, CORD 21 of 43 (most of
    CORD's 14 malformed outputs also follow empty OCR). Tesseract on CORD is
    still being checked for a setup problem (language data,
    preprocessing), so its CORD counts are provisional.
- **With PaddleOCR, the LLM side is where most improvement is possible.**
  PaddleOCR + SROIE has 29 field errors where the value *was* in the text.
  The two biggest groups are:
  - the wrong business (buyer, brand or operator)
  - address/name boundaries
- The CORD run used the SROIE few-shot examples. The `fewshot_copy` and
  part of the CORD totals findings may depend on that, so they get
  re-checked after the planned rerun with CORD examples.
- **Output-format failures are rare but total**, and they are of two
  kinds:
  - **derailment:** the model switches mid-answer to unrelated text
  - **prose instead of JSON:** after (near-)empty OCR input
- **Small-model behaviours not predicted by the task list:**
  - copying the few-shot example's answer (22 values on CORD/Tesseract)
  - corrupting a correctly-read value while copying it (`2018` → `2e18`)
  - rescaling Indonesian amounts by 1000× (`24.000` → `24.00`)
- **Formatting-only mismatches are common:** 76 fields are right after
  normalisation but wrong in strict match. They're kept out of the failure
  counts.

## 2. Categories

### A. Pipeline and output level

| Category | Stage | Rule | Count |
|---|---|---|---|
| `pipeline_error` | pipeline | the receipt crashed (OCR or LLM exception) | 0 |
| `malformed_json` | llm_output | no usable JSON object: parse failures `no_json` / `invalid_json` / `not_object` / `truncated` / `prompt_too_long`. Subtypes below | **30** fields (18 receipts) |
| ↳ `derailment` | | output starts as JSON, then turns into unrelated text until `max_tokens` | 10 fields (4 receipts) |
| ↳ `prose_instead_of_json` | | a sentence instead of JSON (so far always after empty or near-empty OCR) | 20 fields (14 receipts) |

### B. OCR

| Category / tag | Stage | Rule | Count |
|---|---|---|---|
| `value_not_in_ocr` | ocr | the gold value (normalised) is not in the OCR text | **121** primary (142 as tag) |
| tag `ocr_confidence_low` | ocr | OCR pieces carrying the gold value have confidence < 0.8, or > 30 % of the receipt's pieces do | 116 tags |
| tag `ocr_empty_page` | ocr | fewer than 3 OCR pieces for the whole receipt | 23 tags |

### C. LLM extraction (value was in the OCR text)

| Category | Rule | Count |
|---|---|---|
| `fewshot_copy` | the value equals a few-shot example answer and is not on the receipt | 1 primary (22 as tag) |
| `empty_field` | model returned `null`/empty | 0 primary (2 as tag) |
| `copy_corruption` | the model changed characters of a correctly read value: 1 edit in a date as written, ≤ 2 in names/addresses; not applied to amounts | 2 |
| `wrong_date_format` | date that doesn't parse, or day and month swapped | 1 (unparseable; no swaps seen) |
| `wrong_total` | wrong amount. Subtypes: `scale_error` (×/÷ 100 or 1000); `other_amount_on_receipt` (CORD names it: `cash`, `subtotal`, …); `amount_not_in_ocr` | **19** |
| `wrong_entity` | a different name or address that is printed on the receipt (brand vs operator, buyer vs seller). Generic words (`SDN`, `BHD`, `ENTERPRISE`, …) are ignored when comparing | **14** |
| `boundary_too_wide` | all gold words present plus extra ones. Subtypes: `adjacent_text` (extra words are on the receipt) or `hallucinated_extension` (they are not) | 10 |
| `boundary_too_narrow` | only part of the gold words | 4 |
| `partial_value` | overlapping, but neither a subset nor a superset | 4 |
| `wrong_value` | any other wrong value (so far: a different date) | 4 |

### D. Not failures (recorded for the taxonomy)

| Category | Rule | Count |
|---|---|---|
| `normalised_only_mismatch` | strict mismatch, normalised match: formatting only (case, punctuation, date/amount format) | 76 |
| `hallucinated_unlabelled_field` | a non-null value for a field the dataset doesn't annotate (CORD company/date/address) | 193 (112 PaddleOCR, 81 Tesseract) |

### E. Label side (manual review only, not auto-tagged)

| Category | Description | Seen |
|---|---|---|
| `label_definition_ambiguity` | two plausible business names are printed **and the receipt doesn't say which role each has** (brand vs registered operator); the label picks one | 2 of 20 reviewed (R05, R13) |
| `label_noise` | the label itself is wrong (typos, conflicting labels for identical images) | 0 of 20 confirmed; 1 probable (R19 address, degraded print); known SROIE cases exist outside the subsets |

When the receipt **does** state the roles (e.g. `Owned By:` in R16, or seller
vs customer on a delivery invoice in R08/R17), picking the other name counts
as a real `wrong_entity` error, not as an ambiguity.

## 3. Counts per dataset and engine

Primary category, field failures:

| Primary category | SROIE PaddleOCR | SROIE Tesseract | CORD PaddleOCR | CORD Tesseract | Total |
|---|---|---|---|---|---|
| `malformed_json` | 8 | 8 | 0 | 14 | 30 |
| `value_not_in_ocr` | 33 | 67 | 0 | 21 | 121 |
| `fewshot_copy` | 0 | 0 | 0 | 1 | 1 |
| `copy_corruption` | 1 | 1 | 0 | 0 | 2 |
| `wrong_date_format` | 1 | 0 | 0 | 0 | 1 |
| `wrong_total` | 1 | 3 | 8 | 7 | 19 |
| `wrong_entity` | 11 | 3 | 0 | 0 | 14 |
| `boundary_too_wide` | 6 | 4 | 0 | 0 | 10 |
| `boundary_too_narrow` | 4 | 0 | 0 | 0 | 4 |
| `partial_value` | 2 | 2 | 0 | 0 | 4 |
| `wrong_value` | 3 | 1 | 0 | 0 | 4 |
| **Failures** | **70** | **89** | **8** | **43** | **210** |
| `normalised_only_mismatch` | 36 | 35 | 4 | 1 | 76 |
| `hallucinated_unlabelled_field` | 0 | 0 | 112 | 81 | 193 |

Subtypes:

| Subtype | SROIE PaddleOCR | SROIE Tesseract | CORD PaddleOCR | CORD Tesseract |
|---|---|---|---|---|
| `malformed_json` / `derailment` | 8 | 0 | 0 | 2 |
| `malformed_json` / `prose_instead_of_json` | 0 | 8 | 0 | 12 |
| `wrong_total` / `other_amount_on_receipt` (cash) | 0 | 0 | 4 | 2 |
| `wrong_total` / `other_amount_on_receipt` (subtotal) | 0 | 0 | 1 | 0 |
| `wrong_total` / `other_amount_on_receipt` (unnamed) | 1 | 2 | 0 | 0 |
| `wrong_total` / `scale_error` | 0 | 0 | 2 | 5 |
| `wrong_total` / `amount_not_in_ocr` | 0 | 1 | 1 | 0 |
| `boundary_too_wide` / `adjacent_text` | 4 | 0 | 0 | 0 |
| `boundary_too_wide` / `hallucinated_extension` | 2 | 4 | 0 | 0 |

Extra tags (they overlap with primary categories):

| Tag | SROIE PaddleOCR | SROIE Tesseract | CORD PaddleOCR | CORD Tesseract |
|---|---|---|---|---|
| `value_not_in_ocr` | 34 | 75 | 0 | 33 |
| `ocr_confidence_low` | 0 | 72 | 1 | 43 |
| `ocr_empty_page` | 0 | 8 | 0 | 15 |
| `fewshot_copy` | 0 | 0 | 0 | 22 |
| `empty_field` | 1 | 0 | 0 | 1 |

## 4. Examples

Excerpts are shortened. *gold* = label, *OCR* = the related OCR row(s),
*pred* = the model's value.

| Category | Example |
|---|---|
| `malformed_json` / `derailment` | SROIE `X51005757342`, PaddleOCR. OCR read `MR. D.I.Y. (M) SDN BHD` and `Total Incl. GST@6% RM 6.20` correctly; output: `{"company": "MR. D.I EXERCISE 1: Given the receipt OCR text, extract…` (copies the prompt until the token limit). CORD `validation_080`: `"total": "25.0 Written in the form of a letter, explain…` |
| `malformed_json` / `prose_instead_of_json` | SROIE `X51005749904`, Tesseract returned 0 pieces; output: *"Since the OCR text provided is incomplete, I cannot generate a JSON object…"* (the instructions ask for all-null JSON) |
| `value_not_in_ocr` | SROIE `X51005361912`, PaddleOCR: gold `TEO HENG STATIONERY & BOOKS`, OCR `TED HENG STATIONERY & BOOKS`, pred copies `TED HENG…`. SROIE `X51005746204`: OCR `27/Ma1/2018` for `27/MAR/2018` |
| tag `ocr_confidence_low` | CORD `test_001`, Tesseract: the piece `91000` (the total) has confidence 0.77 |
| tag `ocr_empty_page` | SROIE `X51005749904`, Tesseract: 0 pieces (PaddleOCR read the same receipt fully) |
| `fewshot_copy` | CORD `test_001`, Tesseract: pred total `37.10`, gold `91000`. `37.10` is the total of few-shot example 2 and is not on the receipt. On CORD/Tesseract, example 2's values appear where they don't belong: date `14-03-18` on 9 receipts, total on 4, company on 4, address on 5 (22 values) |
| tag `empty_field` | SROIE `X51006414702`, PaddleOCR: address `null` (the address was also not in the OCR text) |
| `copy_corruption` | SROIE `X51006008206`, PaddleOCR: OCR `Date: 2018/03/27`, pred `2e18/03/27` |
| `wrong_date_format` | SROIE `X51005764161`, PaddleOCR: OCR `28-01-18`, pred `011-02-18` (mixes in the quantity `011 X` from the next row) |
| `wrong_total` / `other_amount_on_receipt:cash` | CORD `test_036`, PaddleOCR: gold `40,000`, pred `100.000` (cash paid) |
| `wrong_total` / `other_amount_on_receipt:subtotal` | CORD `test_029`, PaddleOCR: gold `145,900`, pred `132,636` (subtotal before tax/service) |
| `wrong_total` / `other_amount_on_receipt` | SROIE `X51005605295`, PaddleOCR: OCR `SUBTOTAL 5.15`, `TOTAL(INCL GST) 5.15`, pred `5.20` (another amount on the receipt) |
| `wrong_total` / `scale_error` | CORD `test_032`, PaddleOCR: OCR `GRANDTOTAL 24.000` (24,000 rupiah), pred `24.00` |
| `wrong_total` / `amount_not_in_ocr` | SROIE `X51007846400`, Tesseract: the TOTAL row was read as `Total : RM - ~ 4§2,90`; pred `4.90`, gold `152.90` |
| `wrong_entity` | SROIE `X51006556851`, PaddleOCR (delivery invoice): pred `MAKASSAR FRESH MARKET SDN BHD` (the buyer), gold `GARDENIA BAKERIES (KL) SDN BHD` (the issuer); the address is also the buyer's. SROIE `X51005230605`: pred `ALSERKAM ENTERPRISE` (operator), gold `PETRON BKT LANJAN SB` (station) |
| `boundary_too_wide` / `adjacent_text` | SROIE `X51006008206`, PaddleOCR: address plus the registration number `434286-P` from the next row |
| `boundary_too_wide` / `hallucinated_extension` | SROIE `X51007339109`, Tesseract: pred `…40170 SETIA ALAM, KUALA LUMPUR`. `KUALA LUMPUR` is not on the receipt, and Setia Alam is in Selangor |
| `boundary_too_narrow` | SROIE `X51006619506`, PaddleOCR: name printed on two rows, `KEDAI UBAT & RUNCIT HONG` / `NING SDN. BHD.`; pred keeps the first row only |
| `partial_value` | SROIE `X51005724625`, PaddleOCR: gold `LOT 276 JALAN BANTING 43800 DENGKIL…`, pred `NO 1 JALAN AMAN 2, TAMAN DESA 43800, DENGKIL…` (shares postcode and town; the receipt also names a second business, and its company was extracted as that business) |
| `wrong_value` | SROIE `X51005337867`, PaddleOCR: OCR `22 Mar 18 12:44:00 PM`, pred `23-03-18` |
| `normalised_only_mismatch` | SROIE `X51005337877`, PaddleOCR: address equal apart from commas and spacing between the label and the prediction |
| `hallucinated_unlabelled_field` | CORD, PaddleOCR: menu items returned as the company (`ES KOPI SUSU`, `1 S-Ovaltine`). CORD `test_001`: company `SANS` |
| `label_definition_ambiguity` | SROIE `X51007846400`: brand `NAM SAN SEOUL KOREAN BBQ` and legal name `KOREAN DINE SDN BHD` are both printed; the label uses the legal name |

## 5. Manual review

**Method:**

- 20 failures were drawn with seed 42, round-robin over the primary
  categories produced by the **initial** rules. That covers every category,
  both datasets and both engines.
- For each one, the OCR text, raw model output and label were compared
  **with the receipt image**. Where useful, the other OCR engine's result on
  the same receipt was compared too.
- Verdicts are in `review_v0.csv` (`show_case.py R05` prints one case and
  opens its image).
- **Reviewers:** a first pass by Claude (AI assistant), then **every case
  checked against the image by the author (Sufyan)**. The final
  `reviewed_category` is the author's. The author agreed with the first pass
  on **18 / 20** and changed R05 and R13 to `label_definition_ambiguity`.

**Agreement between the automatic primary category and the final review:**

| Rules | Agree |
|---|---|
| initial rules (before review) | 12 / 20 |
| final rules (this document) | 16 / 20 |

**Rule changes made because of the review:**

1. **`wrong_entity`** (R08, R16, R17): generic company words (`SDN`, `BHD`,
   …) are ignored when comparing, and a low-overlap value that is printed
   elsewhere on the receipt is a different entity. It is no longer
   `partial_value` or `wrong_value`, and the rule now applies to addresses
   too.
2. **New category `copy_corruption`** (R09): `2018/03/27` → `2e18/03/27`
   was first tagged as a wrong date, because it parses as 2027-03-18.
3. **`boundary_too_wide` subtypes** (R14): extra words taken from the
   receipt vs extra words that are invented.
4. **New tag and category `fewshot_copy`**: found while collecting
   examples. On CORD/Tesseract, 22 values equal few-shot example 2's
   answers and are not on the receipt.

**Remaining disagreements** (known limitations of v0):

- **R04:** Tesseract garbled the TOTAL row (`4§2,90`), but `152.90` also
  appears on the cash row. The value-in-OCR check therefore says the value
  was read, while the actual cause is an OCR error on the key row.
- **R05, R13:** brand vs legal/operator name, both printed with no role
  stated. The tagger says `partial_value` / `wrong_entity`; the author
  judged them `label_definition_ambiguity`. Such cases can't be detected
  automatically: it takes reading the receipt.
- **R12:** `26,000` → `26,00 CV` is tagged `scale_error`. The review reads
  it as a character-level copy corruption with a 1000× effect. Amounts are
  excluded from the `copy_corruption` rule on purpose, because short numbers
  give false positives.

**Label side:**

- 2 of the 20 are a **label-definition question** (R05, R13). A third
  candidate (R16) was judged a real model error, because the receipt labels
  the other name `Owned By:`.
- No outright label error was confirmed in this sample. **R19's label
  address `JALAN DLNARG U3/G` is a probable annotator misreading:** the print
  is badly degraded (`Dliar G`), and the street is most likely `JALAN DINAR G
  U3/G` (a receipt from the same area has `Jalan Dinar A U3/A`). It can't be
  proven from the image alone.

**What the image review added beyond the automatic tags:**

- **With poor or thin OCR input, the model invents values instead of
  answering `null`:**
  - an invented address (R03: `…GROSS STREET, 569004, KUALA LUMPUR`)
  - an item name as the company (R12 `TOTAL`, R20 `Cuka Apel Tetes`)
  - an item line as the date (R18)
  - a sentence as a value (R19: `"Not provided in the text"`)
  - `N/A` instead of `null` (R18)

  On CORD this happens **even with complete PaddleOCR text** (R12, R18), so
  it is a model behaviour, not only an OCR consequence.
- **Faded or dot-matrix headers defeat PaddleOCR** (R03, R19: company and
  address lines garbled), where **Tesseract read the same header better**.
  Neither engine is uniformly better.
- **Typical Tesseract errors:**
  - it loses right-aligned amount columns (R04, R14)
  - it turns logos/drawings into fake text (R04)
  - it reads nothing from faded receipts that PaddleOCR reads (R10)
  - it misreads large bold totals (R04)
- **Language:** the CORD receipt in R20 has no "Total" line, only
  `Sub Total`, `Tunai` (cash) and `Kembalian` (change). The model took the
  cash, plausibly because it doesn't know the Indonesian words and the
  prompt's examples were SROIE receipts.
- **OCR noise outside the four fields is common** (dropped spaces, letter
  swaps, omitted or invented lines on almost every reviewed receipt). It
  only becomes a failure when it hits a scored field.

## 6. Limitations

- **Small samples:** 50 receipts per dataset. Counts per category are small,
  so percentages are indicative.
- **Value-in-OCR is a text search:**
  - It can be fooled when the same amount appears on another row (R04).
  - For names and addresses it requires the whole label to appear
    contiguously (ignoring case, spaces and punctuation). Label formatting
    that differs from the print is therefore counted as an OCR miss. A
    manual check of a sample of `value_not_in_ocr` cases for this is still
    to do.
- **Label corrections are not applied.**
- **The unit is the field:** a single derailment counts as 4 failures on
  SROIE and 1 on CORD.
- **One prompt and one set of few-shot examples.** `fewshot_copy` and
  possibly `derailment` depend on the examples (example 2 is an MR. D.I.Y.
  receipt; both derailments on SROIE are MR. D.I.Y. receipts).

## 7. Open questions for v1

- **Extend the review beyond 20 cases** (supervisor request for M3),
  especially into the `value_not_in_ocr` bucket, to separate true OCR misses
  from label/print differences.
- **Clearer task definition for "company":** e.g. "prefer the registered
  name (with SDN BHD) when both a brand and a company are printed". That
  would turn the R05/R13 ambiguity into a testable rule.
- **Split `value_not_in_ocr`** into OCR sub-kinds: character substitution
  (`TEO` → `TED`), row not detected, garbled row, empty page.
- **Prompt experiments** to test the causes: zero-shot or other few-shot
  examples (for `fewshot_copy` and `derailment`), an explicit "empty input →
  all null" rule, CORD-specific amount instructions (for `scale_error` and
  cash vs total).
- **Is `hallucinated_unlabelled_field` really a hallucination?** Check CORD
  images: are store names or dates printed?

## Reproduce

```bash
python evaluation/failure_taxonomy/tag.py \
    --run runs/pipeline/20260930-063807 evaluation/datasets/SROIE/subset_test_50.jsonl \
    --run runs/pipeline/20260930-092942 evaluation/datasets/CORD/subset_valtest_50.jsonl
```

This writes `tags/<run_id>/tags.csv` and `counts.json` (committed) and
`tags_detail.jsonl`, which holds text and is gitignored. `--review 20
--seed 42` draws a new review sample. That would overwrite `review_v0.csv`,
so use it for v1 only.

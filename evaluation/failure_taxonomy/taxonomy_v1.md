# Failure taxonomy v1 — cross-model, two OCR engines

This version extends the v0 taxonomy (20 Phi-3/PaddleOCR cases) to the **full grid**:
3 small quantized LLMs (Phi-3, Qwen2.5-3B, Llama-3.2-3B) × 2 OCR engines
(PaddleOCR, Tesseract) × 2 datasets (SROIE, CORD), 50-receipt subset each.

Two inputs feed it:

1. **Automatic tagging** of every (model, engine, dataset) cell — category counts in
   `by_model_counts.json`, rules in `tag.py`.
2. **A 50-case manual review** (`review_v1.csv`): the 20 v0 cases plus 30 new cases
   drawn across the grid (15 OCR-misses, deduplicated across models since OCR is
   shared; 15 model-stage failures, spread over categories and models). Each new
   case was checked with both OCR engines side by side **and** all three models on
   the same field (`show_case.py Rxx --models`), so every verdict is attributed to a
   stage with direct evidence.

Key principle (unchanged from v0): the **primary category is the first pipeline stage
that went wrong**. OCR is cached and shared, so `value_not_in_ocr` cases are nearly
identical across the three models; the model-stage categories are where the models
differ.

---

## 1. Cross-model category counts (failures only)

Counts are failing fields over the 50-receipt subset. `value_not_in_ocr` is an OCR-stage
failure (shared); everything else is model-stage.

### SROIE / PaddleOCR
| category | Phi-3 | Qwen2.5-3B | Llama-3.2-3B |
|---|---|---|---|
| value_not_in_ocr | 33 | 33 | 32 |
| wrong_entity | 11 | 0 | 6 |
| boundary_too_wide | 6 | 5 | 0 |
| boundary_too_narrow | 4 | 1 | 4 |
| wrong_total | 1 | 5 | 7 |
| malformed_json | 8 | 0 | 0 |
| other (partial/wrong_value/copy/date) | 7 | 2 | 3 |
| **total failures** | **70** | **46** | **52** |

### SROIE / Tesseract
| category | Phi-3 | Qwen2.5-3B | Llama-3.2-3B |
|---|---|---|---|
| value_not_in_ocr | 67 | 72 | 70 |
| wrong_entity | 3 | 3 | 7 |
| wrong_total | 3 | 4 | 11 |
| malformed_json | 8 | 0 | 4 |
| other (boundary/partial/copy/value) | 8 | 3 | 4 |
| **total failures** | **89** | **82** | **96** |

### CORD / PaddleOCR
| category | Phi-3 | Qwen2.5-3B | Llama-3.2-3B |
|---|---|---|---|
| value_not_in_ocr | 0 | 0 | 1 |
| wrong_total | 1 | 0 | 10 |
| malformed_json | 1 | 0 | 0 |
| **total failures** | **2** | **0** | **11** |

### CORD / Tesseract
| category | Phi-3 | Qwen2.5-3B | Llama-3.2-3B |
|---|---|---|---|
| value_not_in_ocr | 32 | 33 | 33 |
| wrong_total | 4 | 5 | 11 |
| malformed_json | 1 | 0 | 0 |
| **total failures** | **37** | **38** | **44** |

Reading the table:
- **OCR dominates on Tesseract and on CORD.** `value_not_in_ocr` is ~70 of every cell's
  failures on SROIE/Tesseract and ~33 on CORD/Tesseract — and it is nearly identical
  across the three models, confirming it is an OCR-stage, model-independent failure.
- **CORD/PaddleOCR is almost solved** (Phi-3 2, Qwen 0, Llama 11 failures), so the
  Llama CORD weakness there is purely model-stage (`wrong_total`, see §3.3).
- **Phi-3 is alone in `malformed_json`** on SROIE (8 on each engine); Qwen never produces
  it there (see §3.4).

---

## 2. OCR-stage failures: `value_not_in_ocr` sub-kinds

From the manual review, `value_not_in_ocr` splits into five sub-kinds (new in v1). When
the two engines fail an OCR miss **differently**, both are recorded with an engine prefix,
e.g. `paddleocr:garbled_row; tesseract:char_substitution`.

| sub-kind | meaning | example cases |
|---|---|---|
| `char_substitution` | row read, value recognizable, a few wrong characters | R22 (§9200), R27 (42,500→42,900), R35 (98.90→90.87) |
| `garbled_row` | row read but mangled into gibberish | R21 (address→O S, EA, 33U), R29 (date→11went7), R31 |
| `row_not_detected` | the value's row is absent from the OCR | R30 (CORD numeric columns dropped), R42/R43 (tesseract) |
| `empty_page` | OCR returned (almost) no text for the whole receipt | R23, R33 (CORD/Tesseract) |
| `label_format_diff` | value **is** in the OCR; the "miss" is a gold-label typo, not an OCR failure | R25 (KOPITAM/SND vs printed Kopitiam/Sdn) |

### 2.1 The two engines have complementary weaknesses
A recurring observation across the review (R28, R29, R31, R32, R35):

- **PaddleOCR garbles the top 4–5 lines** — the stylised logo/header where the **company
  name and address** live — while reading the printed body (totals, line items) well.
  So PaddleOCR tends to **fail `company`/`address` and pass `total`**.
- **Tesseract reads the header better but fumbles the body/numbers**, and **collapses on
  CORD** — empty pages (R23, R33), dropped numeric columns (R30), char-substituted totals
  with lost labels (R27). So Tesseract tends to **fail `total`, especially on CORD**.

Neither engine is better everywhere; on the same receipt they often fail *different
fields* (R29: PaddleOCR garbles the header but reads the date; Tesseract reads the header
but garbles the date).

### 2.2 OCR misses are model-independent
Where the OCR genuinely dropped the value, all three models fail together: 3/3 pass on
PaddleOCR and 3/3 miss on Tesseract (R27, R30, R33, R35). This is the clean mirror image
of §3.1, where the OCR is fine and the *models* disagree.

---

## 3. Model-stage failures

### 3.1 `wrong_entity` — the dominant model failure
Receipts routinely print more than one name/address; the model picks the wrong one. Two
sub-patterns, both model-specific:

- **Brand vs legal operator** (gold = the registered `SDN BHD` company): the model picks
  the prominent brand/logo instead.
  - R25: 5 of 6 model×engine combos pick the brand `OLDTOWN WHITE COFFEE` over `Old Town
    Kopitiam Sdn Bhd` — **OCR-independent** (Phi-3 and Qwen pick the brand on both engines).
  - R37/R45: **Phi-3 picks the brand on both engines** (`NAM SAN SEOUL KOREAN BBQ`,
    `NASI KANDAR PELITA`); Qwen and Llama more often pick the legal entity.
- **Buyer vs seller** (invoice with both parties; gold = seller): the model picks the
  buyer/customer.
  - **Llama has a systematic buyer bias**: on the Gardenia→Makassar invoices it picks the
    **buyer** for both `company` (R26) and `address` (R36, R48), on **both engines** —
    three receipts, both fields, both engines. Phi-3 and Qwen flip by engine; Qwen is the
    most reliable at picking the seller.

**Likely fix (testable):** the extraction prompt does not say *which* party to extract. A
clearer instruction — "extract the seller/merchant's registered company name, not the
buyer/customer, and not a product or brand" — targets both sub-patterns directly.

### 3.2 Boundary errors — `boundary_too_wide` / `boundary_too_narrow`
The right entity, wrong extent. Multi-line addresses are the main trigger (R42/R43 are the
same 3-line address):
- **Too narrow:** the model drops part of the value — Qwen drops the `S/B` suffix (R39),
  Llama drops the first address line (R42), **Phi-3 drops the unit/floor/building prefix**
  (R49, both engines).
- **Too wide:** the model adds adjacent text — Qwen appends a store code `A03064` (R46) or
  a registration number `(524463-X)` (R34); **Phi-3 appends hallucinated geographic
  suffixes** — `KUALA PERDANA SELANGOR` (R28), `MALAYSIA` (R32), `63000 KUALA LUMPUR`
  (R43). Phi-3's address boundaries are unstable: it trims in R49 but over-extends in R43.

### 3.3 `wrong_total` — cash/total confusion
When the receipt prints `Total` and `Cash`/`Change` close together, the model grabs the
wrong amount. **Llama is worst**: it picks the cash amount over the total (R40 on
Tesseract, R47 on both engines) and accounts for almost all of the CORD/PaddleOCR
failures (10 of 11). Phi-3 is the most robust on totals; Qwen slips only on Tesseract.

### 3.4 `malformed_json` — Phi-3 derailment + empty-input robustness
Two distinct causes, both producing unusable JSON:
- **Derailment (Phi-3-specific):** on perfectly readable OCR, Phi-3 starts the value and
  spirals into unrelated prose. R41: reading `MR. D.I.Y. (M) SDN BHD`, Phi-3 began the
  company then generated a *sonnet-writing prompt*. Qwen and Llama extract the same input
  cleanly. This matches the counts: `malformed_json` = 8/8 for Phi-3 on SROIE, 0 for Qwen.
- **Empty-input handling (model split):** on an empty OCR page the correct output is
  all-null JSON. **Qwen returns `{null,null,null,null}` (robust); Phi-3 and Llama break
  format and reply in prose** ("You didn't provide the receipt OCR text…") — R44. The
  first failing stage there is still the empty OCR, but the model's formatting response
  differs by model.

### 3.5 Hallucinated fields on unannotated CORD fields
On CORD receipts with no address, some models invent one from nearby text: Qwen and Llama
put **menu-item names** into `address` (R47: `Kopi Susu Kampung x1…`), while **Phi-3
correctly returns null**. Tagged `hallucinated_unlabelled_field`.

---

## 4. Per-model profile (small quantized, CPU, Q4)

- **Phi-3** — strongest on totals and on returning the right *kind* of value, but has
  three distinctive failures: **derailment** into prose (§3.4), **geographic
  hallucination** on addresses (§3.2), and a **brand-over-operator** bias (§3.1). Highest
  SROIE failure count once OCR is good (70 on PaddleOCR).
- **Qwen2.5-3B** — the **most robust and best-formatted**: cleanest JSON, lowest SROIE
  failure count (46 on PaddleOCR), handles empty input with null JSON, best at picking the
  legal/seller entity. Residual failures are minor over-inclusions (store code, reg
  number) and the occasional cash/total slip.
- **Llama-3.2-3B** — two systematic weaknesses: **buyer-vs-seller confusion** (§3.1) and
  **cash/total confusion** (§3.3, dominates its CORD failures). Formatting breaks on empty
  input like Phi-3.

---

## 5. Methodological findings

- **The automatic tagger mis-stages some cases**, which the manual review corrected:
  - Gold-label typos make a readable value look like an OCR miss (R25: `label_format_diff`;
    re-tagged `value_not_in_ocr`→`wrong_entity`).
  - A model boundary issue can masquerade as an OCR miss when the OCR has a one-char error
    that two other models recover from (R34: re-tagged `value_not_in_ocr`→`boundary_too_wide`).
  - A shared token (`KOREAN`) makes a brand-vs-operator pick look like `partial_value`
    (R37/R38: re-tagged → `wrong_entity`).
  This is itself a result: automatic stage-attribution is reliable for clear OCR garble
  but needs human review where gold labels are noisy or two valid entities are present.
- **SROIE gold labels contain noise** (`KOPITAM SND`, `SHAN ALAM`, `JALAN DLNARG U3/G`),
  so strict exact-match under-counts correct extractions; the normalized/lenient score and
  manual review both matter.

---

## 6. Case index (new v1 cases)

OCR-stage: R21 garbled_row, R22 char_substitution, R23 empty_page, R24 char_substitution,
R25 label_format_diff→wrong_entity, R26 char_substitution, R27 char_substitution,
R28 garbled_row, R29 garbled_row, R30 row_not_detected, R31 garbled_row/char_substitution,
R32 char_substitution (+Phi-3 MALAYSIA hallucination), R33 empty_page, R34 →boundary_too_wide,
R35 char_substitution.

Model-stage: R36 wrong_entity (buyer address), R37/R38 wrong_entity (brand),
R39 boundary_too_narrow, R40 wrong_total (cash), R41 malformed_json (derailment),
R42 boundary_too_narrow, R43 boundary_too_wide (geo-hallucination), R44 →value_not_in_ocr
(empty page; Phi-3+Llama prose), R45 wrong_entity (brand), R46 boundary_too_wide (store code),
R47 wrong_total (cash) + item-as-address, R48 wrong_entity (buyer address),
R49 boundary_too_narrow (prefix), R50 wrong_value (date from total).

Full per-case detail: `review_v1.csv`. Counts: `by_model_counts.json`.

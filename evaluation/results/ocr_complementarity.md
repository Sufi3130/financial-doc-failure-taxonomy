# OCR engine complementarity (issue #26)

The OCR stage sets a hard ceiling on extraction: if the gold value is not in the
OCR text, no language model can recover it (`value_not_in_ocr`). PaddleOCR and
Tesseract have complementary weaknesses, so the question is how far two cheap,
training-free strategies would lower that ceiling:

- **field-aware routing** — per field, use whichever single engine misses less;
- **union (merge)** — keep the value if it is in *either* engine's text.

This is a ceiling analysis on the cached OCR only (model-independent): counts are
`value_not_in_ocr` field-instances on the full test split. Reproduce with
`python evaluation/ocr_complementarity.py` (writes `ocr_complementarity.json`).

## SROIE (n = 347 per field)

| Field | PaddleOCR | Tesseract | Field-aware routing | Union (merge) |
|---|---|---|---|---|
| company | 62 | 150 | 62 | **40** |
| date | 41 | 100 | 41 | **18** |
| address | 86 | 182 | 86 | **67** |
| total | 8 | 74 | 8 | **5** |
| **All fields** | **197** | 506 | **197** | **130** |

## CORD (n = 95, total only)

| Field | PaddleOCR | Tesseract | Field-aware routing | Union (merge) |
|---|---|---|---|---|
| total | 3 | 65 | 3 | 3 |

## Findings

1. **Field-aware routing gives no benefit here.** PaddleOCR is the stronger
   single engine on *every* SROIE field and on CORD total, so routing per field
   just selects PaddleOCR everywhere — it never beats using PaddleOCR alone.
2. **Merging both engines' text does help on SROIE: 197 → 130 OCR-stage misses,
   a 34 % reduction** (67 field-instances recovered), concentrated in the
   header fields PaddleOCR garbles — company (−22), date (−23), address (−19),
   total (−3). The gain is real complementarity *at the receipt level*:
   Tesseract occasionally reads a header line that PaddleOCR mangles, even
   though Tesseract is worse on aggregate. The lever is **merging**, not
   per-field selection.
3. **On CORD, Tesseract contributes nothing.** It collapses on the phone-photo
   images (see the Tesseract-on-CORD diagnostic in
   [`../../docs/results.md`](../../docs/results.md)), so the union equals
   PaddleOCR alone (3 misses either way). Merging cannot help where the second
   engine reads nothing.

## Implication

For SROIE, concatenating both engines' reading-order text before the language
model would lower the dominant OCR ceiling by about a third at near-zero extra
cost (Tesseract is ~1 s/page), the single largest lever identified short of a
better OCR model. For CORD, the usable pipeline remains PaddleOCR alone. Wiring
the merge into the pipeline (and measuring the end-to-end accuracy gain, which
requires re-running the models on merged text) is left as future work; this note
establishes the ceiling the merge could reach.

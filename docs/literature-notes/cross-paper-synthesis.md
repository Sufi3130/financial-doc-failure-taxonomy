# Cross-Paper Synthesis (filled from the 4 core papers)

## Cross-paper failure-category tracker

Rows are *my* normalized categories; check marks show which paper reports something matching that pattern, in their own terminology. Recurrence across independent papers = a real, general failure type, not a one-off — this is the key thing to look for.

| Category name (yours, normalized) | Paper 1 (Multi-Stage) | Paper 2 (Fiscal LLM) | Paper 3 (VAREX) | Paper 4 (SOB) |
|---|---|---|---|---|
| Parse / invalid JSON syntax | | | (implied by "non-compliant JSON") | ✓ "Parse failures" |
| Schema violation / wrong structure | | ✓ "Structural mismatch" (via TEDS) | ✓ "Schema echo" (extreme form) | ✓ "Schema violations" |
| Value error (wrong content, right shape) | | ✓ "Numerical consistency failure" | (implied, not separately named) | ✓ "Value errors" — called "the dominant gap" |
| Missing / omitted field | | | ✓ "Under-extraction" | ✓ "Missing paths" |
| Type mismatch (right value, wrong type) | | | | ✓ "Type mismatches" |
| Terminology/naming ambiguity (domain-specific) | ✓ "Inconsistent terminology" | | | |
| Numeric scale/unit ambiguity (domain-specific) | ✓ "Currency unit ambiguity" | | | |
| OCR-specific omission (upstream error, not model error) | ✓ "OCR errors" (keyword omission) | (image conversion deliberately avoids OCR) | | (out of scope — text-only benchmark) |

**What this tells you:** "Missing field," "wrong value," and "wrong structure/schema" are the three categories that show up across almost every paper independently — these are your safest, most defensible core categories. "Schema echo" and "under-extraction" (VAREX) are more specific, model-size-dependent failure modes that only show up in small models — directly relevant since you're testing small models. The two domain-specific rows (terminology, currency/units) only appear in the one paper that actually works with real financial documents (Paper 1) — this is a strong signal that your financial-document taxonomy needs categories the general-purpose benchmarks (3, 4) simply never encounter, because their test data doesn't have this kind of domain ambiguity. That's a legitimate, citable argument for why a financial-specific taxonomy is needed rather than just reusing VAREX's or SOB's categories as-is.

## Architecture comparison

| | Paper 1 (Multi-Stage, OCBC) | Paper 2 (Fiscal LLM, Karnataka) | Your proposed design |
|---|---|---|---|
| Vision/OCR stage | Pre-processing (deskew, contrast) + PaddleOCR v3 | None — high-res image fed directly to LLM | PaddleOCR / TrOCR, + PaddleOCR-VL as ablation |
| Retrieval/filtering stage | BM25 keyword retrieval to narrow pages before extraction | Sequential context carry-forward (no page filtering — every page processed) | None currently — worth considering if you extend to multi-page documents |
| Extraction model | Compact VLM (miniCPM-o 2.6, 8B) on GPU | Large frontier LLM (Gemini 2.5 Pro) via API | Small quantized local LLM (Phi-3 Mini / Mistral 7B GGUF), CPU-only |
| Output format | Structured fields, merged from LLM summaries + VLM extraction | 5 CSV schema types | JSON |
| Failure handling / analysis | Qualitative error write-up (3 categories, no counts) | No error taxonomy — internal consistency validation instead (no ground truth available) | Structured, quantified taxonomy (primary contribution) |
| GPU requirement | Yes (A100) | Yes (cloud API) | No — this is your key differentiator from both |

**What stands out:** neither of the two closest financial-extraction papers actually achieves your "no GPU, no cloud" constraint — Paper 1 needs an A100, Paper 2 uses a cloud API model entirely. This is worth stating plainly in your thesis: the deployment constraint you're targeting is *not yet demonstrated* in the closest existing financial-document extraction literature, which is a legitimate, narrow, and honest way to state your engineering contribution (separate from the evaluation-gap contribution, which is the stronger of the two — see your proposal's gap framing).

# Paper: Information Extraction From Fiscal Documents Using LLMs
**Link:** https://arxiv.org/pdf/2511.10659
**Authors:** Aggarwal et al. (Google + XKDR Forum), ACM ICAIF '25
**Pass completed:** 3 (full)

### Architecture
This is a single-model, sequential pipeline — no separate OCR stage at all:
1. **PDF → high-res image conversion** (300 DPI JPGs) — deliberately, not text extraction. *Why:* they explicitly say converting to image strips out any text metadata, forcing the LLM to visually read the page itself rather than trusting potentially-corrupted embedded text (a real problem in their data, where regional-language text was often mis-encoded as ASCII with a font codebook).
2. **Sequential context passing** — each page is processed one at a time, with the previous page's *extracted data* (not the raw page) fed forward as context, to work around documents running into the hundreds of pages and blowing the context window.
3. **Meta-prompting** — instead of a human writing the extraction prompt directly, the LLM is first given the domain structure and a few illustrative pages and asked to *generate* the CSV schema and, from that, the actual extraction prompt.
4. **LLM extraction** — Gemini 2.5 Pro extracts each page into one of five predefined CSV schema types (Sub-Major Head, Minor Head, Sub Head, Detailed Head, Object Head).
5. **Semantic cleaning** — a rule-based cleaner fixes column misalignment using row-type understanding (Header/Data/Total).
6. **Multi-level validation** — since fiscal tables are hierarchical (totals must sum correctly across levels), they cross-check extracted values against each other rather than against ground truth.

*Why I labeled it this way:* this is genuinely a different architecture shape than your proposal — no OCR engine, no BM25 retrieval, single large model doing everything via images directly. It's closer to the "OCR-free VLM" side of your ablation than to your OCR+LLM design, even though they don't frame it that way themselves (they just call it "image-based processing" using a large frontier model, not a compact VLM).

### Failure categories named
They don't name failure *types* the way VAREX/SOB do — instead, their core contribution is a way to **locate** failures without ground truth, using internal consistency rather than categorizing error causes:
- **Numerical consistency failures** — object-head totals don't sum to their reported detailed-head totals (84–98% pass rate depending on hierarchy level). *Why I'd still count this as a failure category for your taxonomy:* even though they frame it as a validation check rather than an error type, functionally it's identical to a "value error" category (VAREX/SOB's term) — the extracted number is wrong, just detected by cross-referencing instead of comparing to ground truth.
- **Structural mismatch** — measured via Tree Edit Distance Similarity (TEDS) comparing whether the same hierarchy, extracted from two different locations in the source PDF, produces matching tree structures (73–96% accuracy across document volumes). *Why this is a distinct category:* this catches errors in the *shape* of the extraction (wrong hierarchy/nesting), not the *value* — closer to what SOB calls "schema violations" or "missing paths," just measured indirectly via cross-source comparison rather than against a labeled schema.

### What I'd do differently
Their "no ground truth needed" validation trick (checking internal totals) is clever but only works because fiscal documents have a built-in mathematical structure — invoices/bank statements/payslips mostly don't have this property (a payslip's individual fields don't have to sum to anything checkable). So this specific validation method likely won't transfer to your domain, though the *idea* of finding failures via internal consistency checks (rather than just ground truth comparison) might partially apply — e.g. a bank statement's transactions should sum to its stated closing balance, which is a similar structural invariant you could borrow.

### Relevant citations to chase (max 3)
- Sui et al. (Table Meets LLM) — survey specifically on whether LLMs understand table structure, relevant background for financial-table extraction generally.
- Lu et al. 2025 survey on LLM table processing — explicitly notes research has focused on easy text-based formats and hasn't addressed image/PDF-based real-world complexity, which is close to your own stated research gap.
- IndicGenBench (Singh et al. 2024) — evidence that low-resource languages (their example: Indian regional languages) get worse LLM performance — directly supportive citation for your Hungarian/CEE gap argument.

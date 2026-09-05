# Information Extraction From Fiscal Documents Using LLMs

**Authors:** Aggarwal (Google), Kulkarni, Mascarenhas, Narang, Raman, Shah,
Thomas (XKDR Forum, Mumbai)
**Venue:** ACM ICAIF '25 (AI for Financial Inclusion workshop)
**Link:** https://arxiv.org/pdf/2511.10659

## Summary

This paper extracts structured data from Indian state government fiscal
disclosures — large, hierarchical, multi-page PDF tables from the state of
Karnataka. The core problem is similar in spirit to mine (turning
non-machine-readable financial PDFs into structured data) but the documents
are public-sector budget tables rather than invoices/statements/payslips,
and their pipeline uses a large frontier model (Gemini 2.5 Pro) rather than
a small local one.

## Pipeline

No separate OCR engine at all. Instead:

1. **PDF pages are converted to high-resolution images (300 DPI)**, on
   purpose — converting to image strips out embedded text metadata, which
   in their case is often corrupted (regional-language text stored as
   mis-encoded ASCII with a font codebook). This forces the LLM to actually
   read the page visually rather than trust unreliable embedded text.
2. **Sequential context passing** — pages are processed one at a time, with
   the previous page's *extracted* output (not the raw page) carried forward
   as context, since a 500+ page PDF blows any context window if processed
   at once.
3. **Meta-prompting** — rather than a human writing the extraction prompt,
   the LLM is first given domain context and a few illustrative pages and
   asked to generate the CSV schema, and from that, the extraction prompt
   itself.
4. **Extraction into five CSV schema types** matching the fiscal hierarchy
   (Sub-Major Head → Minor Head → Sub Head → Detailed Head → Object Head).
5. **Semantic cleaning** — a rule-based cleaner fixes column misalignment
   using row-type understanding (Header/Data/Total).
6. **Multi-level validation** — since fiscal tables must sum consistently
   across hierarchy levels, they cross-check extracted totals against each
   other instead of against ground truth (which they don't have).

## Error/failure findings

They don't name failure *types* the way a taxonomy paper would — the
contribution here is really a way to *locate* extraction errors without any
labeled ground truth:

- **Numerical consistency checks** — comparing whether Object Head totals
  sum correctly to Detailed Head totals (84–98% pass rate depending on
  hierarchy level and document volume).
- **Structural soundness (TEDS)** — Tree Edit Distance Similarity comparing
  whether the same hierarchy, pulled from two different locations in the
  same PDF, produces matching tree structures (73–96% depending on volume).

When a numerical check fails, they can print the exact page and row where it
failed, which is useful for manual correction even without ground truth.

## Relevance to my thesis

This is useful mainly as a counter-example on architecture — it shows a
working large-model, GPU-free-for-the-user (API-based), no-OCR pipeline,
which is a different design point from mine. Their "convert to image, skip
OCR" choice is closer to what an OCR-free VLM does than to my OCR+LLM
design, even though they never frame it that way.

Their validation-without-ground-truth trick (checking arithmetic
consistency across hierarchy levels) is specific to documents that have a
built-in mathematical structure — invoices/payslips mostly don't have this
property, so I don't think the method itself transfers directly. The
general idea of using an internal consistency check as a cheap failure
signal (e.g. a bank statement's transactions should sum to its stated
closing balance) might be worth borrowing in a limited way, though.

## Citations worth following up
- Sui et al., "Table Meets LLM" — survey on whether LLMs actually understand
  table structure.
- Lu et al. 2025 survey on LLM table processing — notes that research has
  focused on easy text-based formats and largely ignored image/PDF
  real-world complexity, which lines up with my own stated gap.
- Singh et al. 2024, IndicGenBench — evidence that low-resource languages
  get worse LLM performance; useful supporting citation for the
  Hungarian/CEE argument.

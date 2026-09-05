# Paper: Multi-Stage Field Extraction of Financial Documents with OCR and Compact Vision-Language Models
**Link:** https://arxiv.org/html/2510.23066v1
**Authors:** Jin et al., OCBC Bank (Singapore), 2025
**Pass completed:** 3 (full)

### Architecture
Pipeline, in order:
1. **Image pre-processing** — segmentation (crop blank/marginal regions via OpenCV edge detection), deskew/rotation correction (PaddleOCR's orientation classifier + Hough transform for residual skew), re-normalization (resize + CLAHE contrast enhancement + denoising).
2. **Multilingual OCR transcription** — PaddleOCR v3, outputs text tokens + bounding boxes + confidence scores.
3. **Page retrieval** — a BM25 keyword-search step that filters a long document (up to 1,000+ pages) down to the ~20 pages actually relevant to a target field, before any VLM is invoked.
4. **Compact VLM extraction** — a small VLM (miniCPM-o 2.6, 8B) runs only on the narrowed-down pages to pull out structured fields (revenue, profit, dividends, etc.), guided by section-specific prompts.

*Why I labeled it this way:* the paper's own Figure 1 names these exact four stages, so this isn't my inference — it's their stated design. The one thing worth flagging for your own design diagram: **their "OCR" stage isn't just OCR** — it's OCR wrapped by a full pre-processing pipeline (deskew, contrast, etc.) that they show matters a lot for accuracy. Your proposal currently treats OCR as one box; this paper suggests it should probably be two (pre-process → OCR) in your own diagram.

### Failure categories named
- **Inconsistent terminology** — the same financial concept is labeled differently across documents (e.g. "revenue" vs "income" vs "sales"; different profit sub-types). *Why this is its own category, per the paper:* it's not an OCR or model error at all — the text is read and extracted correctly, but the retrieval/extraction step doesn't know two different words mean the same field. This is a domain-knowledge gap, not a technical one — worth keeping separate from OCR errors in your own taxonomy since the fix is completely different (a synonym dictionary vs. better OCR).
- **Currency unit ambiguity** — e.g. Indonesian reports expressing the same number as "IDR'000" vs "juta rupiah" (million rupiah), where the same digits can mean values 3–6 orders of magnitude apart depending on phrasing. *Why separate from terminology:* this is specifically about a value being numerically wrong by a scale factor, not about the wrong field being picked — a much more dangerous failure type for a financial pipeline (a modeling/analysis system fed a value 1,000,000x too small is a bigger problem than a misnamed field).
- **OCR errors (keyword omission)** — OCR fails to capture a keyword the BM25 retrieval step depends on, most often due to low-quality scans or physical stamps covering text; complete omission (not just noise) is what actually breaks the pipeline, since BM25 tolerates minor transcription noise but not a fully missing keyword. *Why I'd keep this distinct from a generic "OCR error":* the paper is specific that partial noise is fine — it's specifically *complete* omission that causes downstream failure, which is a more precise, useful category than just "OCR was wrong."

### What I'd do differently
Their evaluation is only field-level accuracy (5 fields) plus a qualitative error write-up — no systematic failure taxonomy with counts/percentages per category. This is exactly the gap your thesis can fill: they identified the *types* of errors but didn't quantify how often each occurs or build a reusable classification scheme. I'd also note they use a large 8B compact VLM on an A100 GPU — genuinely "compact" relative to a 72B model, but still far from your GPU-free target, so don't over-anchor on their model-size framing as equivalent to yours.

### Relevant citations to chase (max 3)
- mPLUG-DocOwl 1.5/2 (Hu et al. 2024/2025) — OCR-free document VLM, relevant to your VLM ablation.
- LayoutGPT (Feng et al. 2023) — cited as a "compact alternative" to large VLMs, worth checking if it's smaller than PaddleOCR-VL.
- Donut (Kim et al. 2022) — the original OCR-free document transformer, foundational citation for your OCR-free-alternative discussion.

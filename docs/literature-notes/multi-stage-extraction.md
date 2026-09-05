# Multi-Stage Field Extraction of Financial Documents with OCR and Compact Vision-Language Models

**Authors:** Jin, Wang, Zhong, Jin-Chun, Ke, MacDonald (OCBC Bank, Singapore)
**Venue:** arXiv preprint, Oct 2025
**Link:** https://arxiv.org/html/2510.23066v1

## Summary

OCBC's AI Lab built a pipeline to pull structured financial fields (revenue,
profit, dividends, executive compensation) out of scanned SMB financial
reports and audit documents. Their main problem was that these reports can
run to hundreds of pages, are multilingual (English, Chinese, Indonesian
Bahasa in their dataset), and are often low-quality scans. Feeding a whole
report into a large VLM directly was infeasible — it hit only 9% field-level
accuracy, mostly because of out-of-memory failures on long documents and
because irrelevant boilerplate pages distracted the model.

## Pipeline

1. **Image pre-processing** — page segmentation (cropping blank/marginal
   regions), deskew/rotation correction, resolution normalization and
   contrast enhancement (CLAHE).
2. **OCR transcription** — PaddleOCR v3, multilingual, outputs text +
   bounding boxes + confidence scores.
3. **Page retrieval** — a BM25 keyword search narrows a document down from
   (sometimes) 1,000+ pages to the ~20 pages actually relevant to a given
   field, before any VLM call happens. They tried embedding-based RAG
   retrieval first and found it performed worse than plain BM25, because
   financial reports repeat the same boilerplate language across pages,
   which makes dense embeddings all look similar. Keyword frequency turned
   out to discriminate better than semantic similarity here.
4. **Compact VLM extraction** — miniCPM-o 2.6 (8B) runs only on the
   retrieved pages to extract the target fields.

Narrowing the pipeline this way took them from 9% to ~81% field-level
accuracy, at 0.7% of the GPU cost and roughly 13x less latency per page
compared to running a large VLM (Qwen2.5-VL-72B) on the whole document.

## Error analysis

They report three recurring error sources, without giving per-category
counts:

- **Inconsistent terminology** — the same concept labeled differently across
  audit firms (revenue/income/sales; dividend/interest; multiple types of
  "profit"). This isn't an OCR problem — the text is read correctly, the
  model just doesn't know two words mean the same field.
- **Currency unit ambiguity** — especially in Indonesian reports, where the
  same digits can represent values three to six orders of magnitude apart
  depending on phrasing (e.g. "IDR'000" vs "juta rupiah").
- **OCR keyword omission** — low-quality scans or stamps covering text can
  cause a keyword the retrieval step depends on to be missed entirely. Minor
  OCR noise doesn't break BM25 retrieval, but full omission of the keyword
  does.

## Relevance to my thesis

This is architecturally the closest match I found to my own proposal — OCR
feeding a compact model for structured extraction, with cost/latency framed
as a first-class concern. Two things worth noting for my own design:

- Their "OCR stage" is really a pre-processing stage plus OCR, not just OCR
  — the deskew/contrast work is treated as essential to downstream accuracy,
  and I should probably budget time for equivalent pre-processing rather
  than assuming raw scans go straight into PaddleOCR/TrOCR.
- They don't do a systematic, quantified failure taxonomy — their error
  analysis is three named categories with no frequency counts. That gap is
  basically what I'm trying to fill for financial documents specifically.
- Their "compact" VLM (8B, on an A100) is still well outside my no-GPU
  target, so I can't treat their efficiency numbers as directly comparable
  to what a CPU-only pipeline would achieve.

## Citations worth following up
- Kim et al. 2022, Donut — OCR-free document transformer, background for
  the OCR-free VLM ablation.
- Hu et al. 2024/2025, mPLUG-DocOwl 1.5/2 — another OCR-free document VLM
  line, same purpose.
- Feng et al. 2023, LayoutGPT — cited as a compact alternative to large
  VLMs, worth checking size against PaddleOCR-VL.

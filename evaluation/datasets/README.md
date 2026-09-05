# Datasets

This folder holds scripts to download and preprocess evaluation datasets —
it does not contain the raw data itself (too large to commit, and SROIE/CORD
have their own licensing terms that don't permit redistribution).

## SROIE

ICDAR 2019 Scanned Receipt OCR and Information Extraction competition
dataset. 626 training / 347 test receipts, labeled fields: Company, Date,
Address, Total.

Source: https://rrc.cvc.uab.es/?ch=13 (registration required)

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
├── sroie/
│   ├── raw/          # gitignored — downloaded receipts
│   └── prepare.py    # preprocessing script
├── cord/
│   ├── raw/           # gitignored
│   └── prepare.py
└── hungarian/          # optional extension, not yet started
```

"""Field normalisation shared by dataset preparation and the evaluator.

Ground truth and model predictions must go through the same functions,
otherwise the normalised-match score compares apples to oranges.
Both functions return None when the input cannot be parsed.
"""

import re
from datetime import date
from decimal import Decimal, InvalidOperation

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

# (?<!\d) / (?!\d) stop a pattern from matching inside a longer number, e.g. "18-03-23" inside "2018-03-23"
_ISO = re.compile(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)")
_NUMERIC = re.compile(r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})(?!\d)")
_DAY_MON_YEAR = re.compile(r"(?<!\d)(\d{1,2})[\s\-/.]*([A-Za-z]{3,9})[\s\-/.,]*(\d{4}|\d{2})(?!\d)")
_MON_DAY_YEAR = re.compile(r"\b([A-Za-z]{3,9})[\s.]*(\d{1,2}),?\s*(\d{4}|\d{2})(?!\d)")
_COMPACT = re.compile(r"(?<!\d)(\d{8})(?!\d)")

_AMOUNT = re.compile(r"-?\s*\d[\d,]*(?:\.\d+)?")


def _make_date(y, m, d):
    y, m, d = int(y), int(m), int(d)
    if y < 100:
        y += 2000
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def _month(name):
    return MONTHS.get(name[:3].lower())


def normalize_date(s):
    """Parse a receipt date into ISO 'YYYY-MM-DD'.

    Day-first by default (SROIE receipts are Malaysian); switches to month-first
    only when the day-first reading is impossible, e.g. '4/22/2018'.
    Two-digit years are read as 20xx.
    """
    if not s or not s.strip():
        return None
    s = s.strip()

    if m := _ISO.search(s):
        return _make_date(m[1], m[2], m[3])

    if m := _NUMERIC.search(s):
        a, b, y = m.groups()
        return _make_date(y, b, a) or _make_date(y, a, b)

    if (m := _DAY_MON_YEAR.search(s)) and _month(m[2]):
        return _make_date(m[3], _month(m[2]), m[1])

    if (m := _MON_DAY_YEAR.search(s)) and _month(m[1]):
        return _make_date(m[3], _month(m[1]), m[2])

    if m := _COMPACT.search(s):
        d = m[1]
        return _make_date(d[:4], d[4:6], d[6:]) or _make_date(d[4:], d[2:4], d[:2])

    return None


def normalize_total(s):
    """Parse a money amount into a 2-decimal string, e.g. 'RM 1,007.5' -> '1007.50'.

    Currency symbols and spaces are dropped; a comma followed by exactly two
    digits at the end is treated as a decimal comma, otherwise as a thousands
    separator.
    """
    if not s or not s.strip():
        return None
    m = _AMOUNT.search(s)
    if not m:
        return None
    num = m[0].replace(" ", "")
    if re.fullmatch(r"-?\d+,\d{2}", num):
        num = num.replace(",", ".")
    else:
        num = num.replace(",", "")
    try:
        return str(Decimal(num).quantize(Decimal("0.01")))
    except InvalidOperation:
        return None

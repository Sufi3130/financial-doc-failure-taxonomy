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


def _to_decimal_str(num):
    try:
        return str(Decimal(num).quantize(Decimal("0.01")))
    except InvalidOperation:
        return None


def _parse_amount_my(s):
    """Malaysian ringgit (SROIE): '.' is the decimal point, ',' groups thousands,
    except a lone ',dd' at the end, which is a decimal comma."""
    m = _AMOUNT.search(s)
    if not m:
        return None, False
    num = m[0].replace(" ", "")
    if re.fullmatch(r"-?\d+,\d{2}", num):
        num = num.replace(",", ".")
    else:
        num = num.replace(",", "")
    return _to_decimal_str(num), False


def _parse_amount_id(s):
    """Indonesian rupiah (CORD): both '.' and ',' group thousands
    ('60.000' = '60,000' = 60000); only a final separator followed by exactly
    two digits is a decimal part ('35.000,00', '226,500.00').
    Irregular grouping ('57,0000', '1178.100') is parsed by dropping all
    separators and reported as irregular."""
    m = _ID_AMOUNT.search(s)
    if not m:
        return None, False
    num = m[0].replace(" ", "").rstrip(".,")
    sign = "-" if num.startswith("-") else ""
    num = num.lstrip("-")

    decimals = ""
    if d := re.search(r"[.,](\d{2})$", num):
        decimals, num = d[1], num[: d.start()]

    groups = re.split(r"[.,]", num)
    regular = len(groups) == 1 or (
        1 <= len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:]))
    value = sign + "".join(groups) + ("." + decimals if decimals else "")
    return _to_decimal_str(value), not regular


_ID_AMOUNT = re.compile(r"-?\s*\d[\d.,]*")
_PARSERS = {"my": _parse_amount_my, "id": _parse_amount_id}


def parse_amount(s, locale="my"):
    """Return (normalised 2-decimal string or None, irregular_format flag)."""
    if not isinstance(s, str) or not s.strip():
        return None, False
    return _PARSERS[locale](s)


def normalize_total(s, locale="my"):
    """Parse a money amount into a 2-decimal string.

    locale="my" (SROIE, ringgit): 'RM 1,007.5' -> '1007.50'
    locale="id" (CORD, rupiah):   'Rp. 60.000' -> '60000.00'

    Ground truth and predictions of the same dataset must use the same locale;
    the manifest stores it per receipt.
    """
    return parse_amount(s, locale)[0]

import pytest

from evaluation.normalize import normalize_date, normalize_total, parse_amount


@pytest.mark.parametrize("raw, expected", [
    # formats observed in SROIE ground truth
    ("25/12/2018", "2018-12-25"),
    ("12-01-19", "2019-01-12"),
    ("23-01-2019", "2019-01-23"),
    ("18/03/18", "2018-03-18"),
    ("05 MAR 2018", "2018-03-05"),
    ("5/3/2018", "2018-03-05"),
    ("28 MAR 18", "2018-03-28"),
    ("24-MAR-2018", "2018-03-24"),
    ("02/JAN/2017", "2017-01-02"),
    ("2018-03-23", "2018-03-23"),
    ("2018/02/22", "2018-02-22"),
    ("20180304", "2018-03-04"),
    ("25032018", "2018-03-25"),
    ("17/1/2018", "2018-01-17"),
    ("11.02.18", "2018-02-11"),
    ("(06/12/2016)", "2016-12-06"),
    ("OCT 3, 2016", "2016-10-03"),
    # month-first only when day-first is impossible
    ("4/22/2018", "2018-04-22"),
    # extra text around the date, as a model might output
    ("Date: 25/12/2018 10:31 AM", "2018-12-25"),
    ("", None),
    ("not a date", None),
    ("32/13/2018", None),
])
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("9.00", "9.00"),
    ("RM41.45", "41.45"),
    ("RM 3.90", "3.90"),
    ("$8.20", "8.20"),
    ("1,007.50", "1007.50"),
    ("-1.73", "-1.73"),
    ("60.3", "60.30"),
    ("12", "12.00"),
    ("9,90", "9.90"),
    ("TOTAL: RM 29.70", "29.70"),
    ("", None),
    ("   ", None),
    ("n/a", None),
])
def test_normalize_total(raw, expected):
    assert normalize_total(raw) == expected


@pytest.mark.parametrize("raw, expected, irregular", [
    # formats observed in CORD total.total_price (rupiah)
    ("60.000", "60000.00", False),
    ("28,000", "28000.00", False),
    ("174,600", "174600.00", False),
    ("1,565,938", "1565938.00", False),
    ("3.600.000", "3600000.00", False),
    ("91000", "91000.00", False),
    ("Rp 38.000", "38000.00", False),
    ("Rp. 91,000", "91000.00", False),
    ("Rp.118.000", "118000.00", False),
    ("226,500.00", "226500.00", False),
    ("35.000,00", "35000.00", False),
    ("Rp 58000.00", "58000.00", False),
    ("TOTAL 47,499", "47499.00", False),
    ("-5.000", "-5000.00", False),
    # malformed grouping: best-effort value, flagged
    ("39,200,00", "39200.00", False),
    ("57,0000", "570000.00", True),
    ("1178.100", "1178100.00", True),
    ("", None, False),
    ("Rp", None, False),
])
def test_parse_amount_id(raw, expected, irregular):
    assert parse_amount(raw, locale="id") == (expected, irregular)


def test_same_string_differs_by_locale():
    assert normalize_total("60.000", locale="my") == "60.00"
    assert normalize_total("60.000", locale="id") == "60000.00"

"""Engine-free OCR tests: reading order, Tesseract grouping, run-log round trip."""

from src.ocr import OCRLine, OCRResult, load_run_log, reading_order, save_run_log
from src.ocr.base import rect_polygon
from src.ocr.tesseract import lines_from_data


def piece(i, text, box, conf=0.9):
    return OCRLine(index=i, text=text, box=box, polygon=rect_polygon(box), confidence=conf)


def test_label_and_value_on_same_row_are_joined():
    lines = [
        piece(0, "TOTAL:", (171, 696, 240, 718)),
        piece(1, "193.00", (288, 697, 358, 719)),
        piece(2, "VISA CARD", (145, 728, 232, 746)),
    ]
    ro = reading_order(lines)
    assert ro.text == "TOTAL: 193.00\nVISA CARD"
    assert ro.rows == [[0, 1], [2]]


def test_rows_sorted_top_to_bottom_and_left_to_right():
    lines = [
        piece(0, "193.00", (302, 612, 357, 630)),
        piece(1, "Company", (80, 20, 300, 40)),
        piece(2, "Total Exclude GST:", (107, 611, 252, 631)),
    ]
    ro = reading_order(lines)
    assert ro.text == "Company\nTotal Exclude GST: 193.00"
    assert ro.rows == [[1], [2, 0]]


def test_adjacent_rows_are_not_merged():
    # consecutive receipt rows touching by 1-2 px must stay separate
    lines = [piece(0, "BANDAR SERI ALAM,", (126, 185, 286, 205)),
             piece(1, "81750 MASAI, JOHOR", (121, 204, 291, 226))]
    assert reading_order(lines).rows == [[0], [1]]


def test_empty_pieces_are_skipped():
    lines = [piece(0, "  ", (0, 0, 10, 10)), piece(1, "A", (0, 20, 10, 30))]
    ro = reading_order(lines)
    assert ro.text == "A" and ro.rows == [[1]]


def test_tesseract_words_grouped_into_lines():
    data = {
        "page_num": [1, 1, 1, 1, 1], "block_num": [1, 1, 1, 1, 2],
        "par_num": [1, 1, 1, 1, 1], "line_num": [0, 1, 1, 1, 1],
        "left": [0, 10, 60, 0, 200], "top": [0, 100, 102, 0, 101],
        "width": [0, 40, 50, 0, 60], "height": [0, 14, 12, 0, 14],
        "conf": [-1, 90, 70, -1, 96], "text": ["", "Total", "GST:", "", "0.00"],
    }
    lines = lines_from_data(data)
    assert [ln.text for ln in lines] == ["Total GST:", "0.00"]
    assert lines[0].box == (10, 100, 110, 114)
    assert lines[0].confidence == 0.8
    assert [w.confidence for w in lines[0].words] == [0.9, 0.7]
    # separate Tesseract blocks on the same visual row are joined by reading_order
    assert reading_order(lines).text == "Total GST: 0.00"


def test_run_log_round_trip(tmp_path):
    lines = [piece(0, "TOTAL:", (1, 1, 5, 5), 0.5)]
    result = OCRResult(image_path="x.jpg", image_size=(10, 10), engine="test", engine_version="0",
                       settings={"a": 1}, lines=lines, timings={"load_s": 0.0, "ocr_s": 0.1})
    ro = reading_order(lines)
    path = save_run_log(result, ro, tmp_path, receipt_id="r1")
    assert path.name == "r1.test.json"
    assert load_run_log(path) == (result, ro)

"""Rebuild reading order from OCR pieces, independent of the engine.

Engines return text pieces, and a receipt row such as "TOTAL:  193.00" often
comes back as two pieces. Pieces that overlap vertically are grouped into one
row, sorted left to right and joined, so labels stay next to their values in
the LLM prompt. `rows` maps every output line back to the OCR piece indices
it came from, which lets the evaluator tell OCR failures (value never read)
from LLM failures (value read but not extracted).
"""

from pydantic import BaseModel

from .base import OCRLine


class ReadingOrder(BaseModel):
    text: str
    rows: list[list[int]]  # per output line: indices of the OCRLines joined into it


def _vertical_overlap(a, b):
    """Overlap of two boxes' y-ranges as a fraction of the shorter box height."""
    top, bottom = max(a[1], b[1]), min(a[3], b[3])
    shorter = min(a[3] - a[1], b[3] - b[1])
    return max(0, bottom - top) / shorter if shorter > 0 else 0.0


def reading_order(lines: list[OCRLine], joiner=" ", min_overlap=0.5) -> ReadingOrder:
    """Group pieces into rows, top to bottom, each row left to right.

    A piece joins the current row when it overlaps the row's first piece
    vertically by at least `min_overlap` (of the shorter height). Anchoring on
    the first piece stops slightly skewed rows from chaining into each other.
    """
    pieces = sorted((ln for ln in lines if ln.text.strip()),
                    key=lambda ln: ((ln.box[1] + ln.box[3]) / 2, ln.box[0]))
    rows = []
    for ln in pieces:
        if rows and _vertical_overlap(rows[-1][0].box, ln.box) >= min_overlap:
            rows[-1].append(ln)
        else:
            rows.append([ln])

    rows = [sorted(row, key=lambda ln: ln.box[0]) for row in rows]
    return ReadingOrder(
        text="\n".join(joiner.join(ln.text.strip() for ln in row) for row in rows),
        rows=[[ln.index for ln in row] for row in rows],
    )

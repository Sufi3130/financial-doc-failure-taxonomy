"""Target schema for receipt extraction (SROIE Task 3 fields)."""

from pydantic import BaseModel, ConfigDict

FIELDS = ("company", "date", "address", "total")


class Receipt(BaseModel):
    """All four keys are required and must be strings; null means "not on the
    receipt" (the correct answer for CORD's company/date/address). Strict mode:
    a number is not accepted as a string, and unknown keys are rejected."""
    model_config = ConfigDict(extra="forbid", strict=True)

    company: str | None
    date: str | None
    address: str | None
    total: str | None

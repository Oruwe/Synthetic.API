"""Schema for data extracted from the (mock) legacy portal.

`extra="forbid"` is deliberate: if the portal's DOM structure changes and
the extractor starts pulling unexpected fields, we want a hard failure
here rather than silently passing unknown/tampered data further down the
pipeline (and eventually into an LLM prompt).
"""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class DelayedOrder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str
    customer_name: str
    destination: str
    expected_date: date
    status: str
    delay_reason: str | None = None
    carrier: str | None = None
    raw_notes: str | None = None

    # Names of guard patterns (see common/guard.py) that matched a field on
    # this order, formatted "<pattern_name>:<field_name>". Populated by
    # web_navigator/extractor.py. NEVER used to mutate/strip the source
    # field — downstream consumers decide what to do with a flagged order.
    flags: list[str] = Field(default_factory=list)

    extracted_at: datetime
    source_selector_version: str = "v1"


class ExtractionResult(BaseModel):
    orders: list[DelayedOrder]
    page_url: str
    extracted_count: int
    guard_flags_total: int
    # Rows that failed schema validation (bad/missing field) and were
    # skipped individually rather than failing the whole extraction --
    # see web_navigator/extractor.py.
    skipped_count: int = 0

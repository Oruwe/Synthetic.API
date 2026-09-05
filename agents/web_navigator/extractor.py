"""DOM row dicts -> validated, guard-scanned DelayedOrder objects.

This is the core injection-defense decision for the whole system: this
module receives plain strings already read via specific selectors (never
raw HTML), and its only job is to type-validate them (pydantic) and scan
free-text fields for injection patterns. No LLM call happens anywhere in
this file.
"""

from datetime import datetime, timezone

from agents.common.guard import scan_for_injection
from agents.common.logging import get_logger
from agents.common.models.orders import DelayedOrder, ExtractionResult

logger = get_logger(component="extractor")

# Fields that hold free text an attacker controlling portal content could
# use for injection; structured fields (order_id, status, expected_date)
# are not scanned since a valid value there can't carry instruction text.
_FREE_TEXT_FIELDS = ("customer_name", "destination", "carrier", "delay_reason", "raw_notes")


def extract_orders(
    raw_rows: list[dict[str, str]],
    page_url: str,
    only_status: str | None = "delayed",
) -> ExtractionResult:
    orders: list[DelayedOrder] = []
    guard_flags_total = 0
    now = datetime.now(timezone.utc)

    for raw in raw_rows:
        if only_status and raw.get("status") != only_status:
            continue

        flags: list[str] = []
        for field_name in _FREE_TEXT_FIELDS:
            value = raw.get(field_name) or None
            for hit in scan_for_injection(value, field_name):
                flags.append(f"{hit.pattern_name}:{hit.field_name}")
                logger.warning(
                    "guard_hit",
                    order_id=raw.get("order_id"),
                    pattern_name=hit.pattern_name,
                    field_name=hit.field_name,
                    matched_text=hit.matched_text,
                )
        guard_flags_total += len(flags)

        order = DelayedOrder(
            order_id=raw["order_id"],
            customer_name=raw["customer_name"],
            destination=raw["destination"],
            expected_date=raw["expected_date"],
            status=raw["status"],
            delay_reason=raw.get("delay_reason") or None,
            carrier=raw.get("carrier") or None,
            raw_notes=raw.get("raw_notes") or None,
            flags=flags,
            extracted_at=now,
        )
        orders.append(order)

    logger.info("orders_extracted", count=len(orders), guard_flags_total=guard_flags_total)
    return ExtractionResult(
        orders=orders,
        page_url=page_url,
        extracted_count=len(orders),
        guard_flags_total=guard_flags_total,
    )

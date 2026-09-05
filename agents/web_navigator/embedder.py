"""Embed validated orders and upsert them into Qdrant.

This is the async hand-off point to the Synthesizer: nothing here calls
the Synthesizer directly. It only writes to shared memory (Qdrant); the
Synthesizer's watcher.py notices the new points independently.
"""

from agents.common import qdrant_store
from agents.common.logging import get_logger
from agents.common.models.orders import DelayedOrder

logger = get_logger(component="embedder")


def embed_and_store(orders: list[DelayedOrder], run_id: str) -> list[str]:
    point_ids = [qdrant_store.upsert_order(order, run_id) for order in orders]
    logger.info("orders_embedded_and_stored", count=len(point_ids), run_id=run_id)
    return point_ids

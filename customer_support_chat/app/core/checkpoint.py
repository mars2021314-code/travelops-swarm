from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from customer_support_chat.app.core.logger import logger
from customer_support_chat.app.core.settings import get_settings


def build_checkpointer():
    settings = get_settings()
    checkpoint_url = settings.CHECKPOINT_DATABASE_URL.strip()

    if checkpoint_url:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            if settings.REQUIRE_PERSISTENT_CHECKPOINT:
                raise RuntimeError(
                    "Persistent checkpointing is required, but "
                    "langgraph-checkpoint-postgres is not installed."
                ) from exc
            logger.warning(
                "CHECKPOINT_DATABASE_URL is set, but Postgres checkpoint support "
                "is not installed. Falling back to in-memory checkpoints."
            )
        else:
            import psycopg

            conn = psycopg.connect(
                checkpoint_url,
                autocommit=True,
                prepare_threshold=0,
            )
            checkpointer = PostgresSaver(conn)
            checkpointer.setup()
            checkpointer._customer_support_conn = conn
            logger.info("Using Postgres-backed LangGraph checkpoints.")
            return checkpointer

    if settings.REQUIRE_PERSISTENT_CHECKPOINT:
        raise RuntimeError(
            "Persistent checkpointing is required. Set CHECKPOINT_DATABASE_URL "
            "and install langgraph-checkpoint-postgres."
        )

    logger.warning("Using in-memory LangGraph checkpoints. Do not use this for production.")
    return MemorySaver()

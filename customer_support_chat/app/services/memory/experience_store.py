from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointIdsList,
    PointStruct,
)

from customer_support_chat.app.services.vectordb.vectordb import VectorDB
from customer_support_chat.app.core.settings import get_settings
import hashlib
import uuid

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EXPERIENCE_PATH = BASE_DIR / "experience_db.jsonl"
settings = get_settings()


class ExperienceStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_EXPERIENCE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self.db_path.touch()
        # Initialize vector DB for semantic search
        self.vector_db = VectorDB(collection_name="experience_memory")

    def load_all(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with self.db_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _make_point_id(self, record: dict[str, Any]) -> str:
        memory_key = record.get("memory_key")
        if memory_key:
            namespace = record.get("namespace", "global")
            stable_key = hashlib.sha256(
                f"{namespace}|{memory_key}".encode("utf-8")
            ).hexdigest()
            return str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))
        return record.get("memory_id") or str(uuid.uuid4())

    def append(self, record: dict[str, Any], *, write_audit: bool = True) -> dict[str, Any]:
        normalized = dict(record)
        normalized.setdefault("scope", "global")
        normalized.setdefault("namespace", "global")
        normalized.setdefault("memory_type", "episode_resolution")
        normalized.setdefault("tags", [])
        normalized.setdefault("memory_id", self._make_point_id(normalized))
        normalized.setdefault("is_active", True)

        replacement_id = normalized["memory_id"]
        deactivation_filters = normalized.pop("deactivation_filters", [])
        supersedes = normalized.get("supersedes", [])

        if supersedes:
            deactivation_filters.append({"memory_id": supersedes})
        for filters in deactivation_filters:
            self.deactivate(filters, replaced_by=replacement_id)

        if write_audit and settings.MEMORY_AUDIT_LOG_ENABLED:
            with self.db_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(normalized, ensure_ascii=False) + "\n")

        content = self._record_to_text(normalized)
        payload = dict(normalized)
        payload["content"] = content
        self.vector_db.upsert_payload(
            point_id=self._make_point_id(normalized),
            content=content,
            payload=payload,
        )
        return normalized

    def _matches_filters(self, record: dict[str, Any], filters: dict[str, Any]) -> bool:
        for key, value in filters.items():
            if value is None:
                continue
            record_value = record.get(key)
            if isinstance(value, list):
                if record_value not in value:
                    return False
            else:
                if record_value != value:
                    return False
        return True

    def _build_filter(self, filters: dict[str, Any]) -> Filter | None:
        conditions = []
        for key, value in filters.items():
            if value is None:
                continue
            if isinstance(value, list):
                conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        if not conditions:
            return None
        return Filter(must=conditions)

    def deactivate(self, filters: dict[str, Any], *, replaced_by: str | None = None) -> int:
        query_filter = self._build_filter(filters)
        if query_filter is None:
            return 0

        records, _ = self.vector_db.client.scroll(
            collection_name=self.vector_db.collection_name,
            scroll_filter=query_filter,
            with_payload=True,
            with_vectors=True,
            limit=128,
        )

        count = 0
        for record in records:
            payload = dict(record.payload or {})
            if payload.get("memory_id") == replaced_by:
                continue
            payload["is_active"] = False
            payload["valid_to"] = payload.get("valid_to") or normalized_now()
            if replaced_by:
                payload["replaced_by"] = replaced_by
            self._replace_payload(record.id, payload, getattr(record, "vector", None))
            count += 1
        return count

    def _replace_payload(self, point_id: str, payload: dict[str, Any], vector: Any = None) -> None:
        if hasattr(self.vector_db.client, "set_payload"):
            self.vector_db.client.set_payload(
                collection_name=self.vector_db.collection_name,
                payload=payload,
                points=[point_id],
            )
            return

        self.vector_db.client.upsert(
            collection_name=self.vector_db.collection_name,
            points=[PointStruct(id=point_id, vector=vector or [], payload=payload)],
        )

    def expire_stale_memories(self, *, now: datetime | None = None) -> int:
        now = now or datetime.now(timezone.utc)
        records, _ = self.vector_db.client.scroll(
            collection_name=self.vector_db.collection_name,
            scroll_filter=self._build_filter({"is_active": True}),
            with_payload=True,
            with_vectors=True,
            limit=1000,
        )

        expired = 0
        for record in records:
            payload = dict(record.payload or {})
            expires_at = payload.get("expires_at")
            if not expires_at:
                continue
            try:
                expires_dt = datetime.fromisoformat(str(expires_at))
            except ValueError:
                continue
            if expires_dt.tzinfo is None:
                expires_dt = expires_dt.replace(tzinfo=timezone.utc)
            if expires_dt <= now:
                payload["is_active"] = False
                payload["valid_to"] = payload.get("valid_to") or now.isoformat()
                payload["expired_at"] = now.isoformat()
                self._replace_payload(record.id, payload, getattr(record, "vector", None))
                expired += 1
        return expired

    def list_memories(
        self,
        *,
        filters: dict[str, Any] | None = None,
        include_inactive: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        self.expire_stale_memories()
        effective_filters = dict(filters or {})
        if not include_inactive:
            effective_filters["is_active"] = True

        records, _ = self.vector_db.client.scroll(
            collection_name=self.vector_db.collection_name,
            scroll_filter=self._build_filter(effective_filters),
            with_payload=True,
            with_vectors=False,
            limit=max(limit, 1),
        )

        memories = [dict(record.payload or {}) for record in records]
        memories.sort(
            key=lambda item: item.get("updated_at") or item.get("created_at") or "",
            reverse=True,
        )
        return memories[:limit]

    def summarize_memories(
        self,
        *,
        filters: dict[str, Any] | None = None,
        include_inactive: bool = False,
        limit: int = 200,
    ) -> dict[str, Any]:
        memories = self.list_memories(
            filters=filters,
            include_inactive=include_inactive,
            limit=limit,
        )
        by_type: dict[str, int] = {}
        by_scope: dict[str, int] = {}
        active_count = 0
        for memory in memories:
            by_type[memory.get("memory_type", "unknown")] = by_type.get(memory.get("memory_type", "unknown"), 0) + 1
            by_scope[memory.get("scope", "unknown")] = by_scope.get(memory.get("scope", "unknown"), 0) + 1
            if memory.get("is_active", False):
                active_count += 1

        return {
            "total": len(memories),
            "active": active_count,
            "inactive": len(memories) - active_count,
            "by_type": by_type,
            "by_scope": by_scope,
            "sample": [
                {
                    "memory_id": memory.get("memory_id"),
                    "memory_type": memory.get("memory_type"),
                    "summary": memory.get("summary", ""),
                    "scope": memory.get("scope"),
                    "is_active": memory.get("is_active"),
                }
                for memory in memories[:10]
            ],
        }

    def purge_memories(self, *, filters: dict[str, Any], include_inactive: bool = True) -> int:
        targets = self.list_memories(
            filters=filters,
            include_inactive=include_inactive,
            limit=1000,
        )
        if not targets:
            return 0

        purged_ids = {target.get("memory_id") for target in targets if target.get("memory_id")}
        self.deactivate(
            {
                **filters,
                **({} if include_inactive else {"is_active": True}),
            }
        )
        if purged_ids and hasattr(self.vector_db.client, "delete"):
            self.vector_db.client.delete(
                collection_name=self.vector_db.collection_name,
                points_selector=PointIdsList(points=list(purged_ids)),
            )

        retained = []
        for record in self.load_all():
            if record.get("memory_id") in purged_ids:
                continue
            if self._matches_filters(record, filters):
                continue
            retained.append(record)

        with self.db_path.open("w", encoding="utf-8") as f:
            for record in retained:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return len(targets)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        scope: str | None = None,
        namespace: str | None = None,
        memory_types: list[str] | None = None,
        agent_name: str | None = None,
        passenger_id: str | None = None,
        thread_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self.expire_stale_memories()
        filters = {
            "scope": scope,
            "namespace": namespace,
            "memory_type": memory_types,
            "agent_name": agent_name,
            "passenger_id": passenger_id,
            "thread_id": thread_id,
            "is_active": True,
        }
        results = self.vector_db.search(query, k=top_k, filters=filters)
        memories: list[dict[str, Any]] = []
        for result in results:
            payload = dict(result.payload or {})
            payload["score"] = result.score
            memories.append(payload)
        return memories

    def _record_to_text(self, record: dict[str, Any]) -> str:
        """Convert record to searchable text."""
        fields = [
            record.get("memory_type", ""),
            record.get("scope", ""),
            record.get("issue_type", ""),
            record.get("intent_signature", ""),
            record.get("summary", ""),
            record.get("content", ""),
            record.get("final_resolution", ""),
            " ".join(record.get("tool_sequence", [])),
            str(record.get("entities", {})),
            str(record.get("tags", [])),
        ]
        return " ".join(fields)


def normalized_now() -> str:
    return datetime.now(timezone.utc).isoformat()

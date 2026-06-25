from __future__ import annotations

import shutil
import unittest
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

from customer_support_chat.app.services.memory.handoff import build_handoff_memory_brief
from customer_support_chat.app.services.memory.admin import forget_memory, inspect_memory
from customer_support_chat.app.services.memory.experience_store import ExperienceStore
from customer_support_chat.app.services.memory.retriever import retrieve_experiences


@dataclass
class FakeMatchValue:
    value: object


class FakePoint:
    def __init__(self, point_id, payload, vector=None):
        self.id = point_id
        self.payload = payload
        self.vector = vector or []


class FakeSearchResult:
    def __init__(self, payload, score):
        self.payload = payload
        self.score = score


class FakeVectorDB:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.client = self
        self.points = {}

    def upsert_payload(self, *, point_id, content, payload):
        stored = dict(payload)
        stored["chunk_text"] = content
        self.points[point_id] = FakePoint(point_id, stored, vector=[0.0])

    def upsert(self, collection_name, points):
        for point in points:
            self.points[point.id] = FakePoint(point.id, dict(point.payload), vector=point.vector)

    def _match_filter(self, payload, query_filter):
        if query_filter is None:
            return True
        for condition in getattr(query_filter, "must", []) or []:
            key = getattr(condition, "key", "")
            value = payload.get(key)
            match = getattr(condition, "match", None)
            if hasattr(match, "value"):
                if value != match.value:
                    return False
            elif hasattr(match, "any"):
                if value not in list(match.any):
                    return False
        return True

    def scroll(self, collection_name, scroll_filter=None, with_payload=True, with_vectors=False, limit=1000):
        matched = [
            point for point in self.points.values()
            if self._match_filter(point.payload, scroll_filter)
        ]
        return matched[:limit], None

    def search(self, query, k=3, filters=None):
        tokens = set(str(query).lower().split())
        results = []
        for point in self.points.values():
            payload = point.payload
            if filters:
                passed = True
                for key, expected in filters.items():
                    if expected is None:
                        continue
                    value = payload.get(key)
                    if isinstance(expected, list):
                        if value not in expected:
                            passed = False
                            break
                    elif value != expected:
                        passed = False
                        break
                if not passed:
                    continue
            haystack = f"{payload.get('summary', '')} {payload.get('content', '')}".lower()
            score = 1.0 if any(token in haystack for token in tokens if token) else 0.5
            results.append(FakeSearchResult(dict(payload), score))
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:k]


class MemoryMechanismTests(unittest.TestCase):
    def make_store(self):
        tempdir = Path.cwd() / "tests_tmp" / f"memory_{uuid4().hex}"
        tempdir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(tempdir, ignore_errors=True))
        db_path = tempdir / "experience_db.jsonl"
        patcher = patch(
            "customer_support_chat.app.services.memory.experience_store.VectorDB",
            FakeVectorDB,
        )
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        self.assertIsNotNone(mocked)
        return ExperienceStore(db_path=db_path)

    def test_entity_fact_replacement_deactivates_previous_fact(self):
        store = self.make_store()
        first = store.append(
            {
                "memory_key": "entity:t1:ticket:T1:v1",
                "memory_type": "entity_fact",
                "scope": "thread",
                "namespace": "thread:t1",
                "passenger_id": "p1",
                "thread_id": "t1",
                "entity_type": "ticket",
                "entity_id": "T1",
                "summary": "Ticket T1 updated.",
                "content": "Ticket T1 updated.",
                "status": "updated",
                "is_active": True,
            }
        )
        second = store.append(
            {
                "memory_key": "entity:t1:ticket:T1:v2",
                "memory_type": "entity_fact",
                "scope": "thread",
                "namespace": "thread:t1",
                "passenger_id": "p1",
                "thread_id": "t1",
                "entity_type": "ticket",
                "entity_id": "T1",
                "summary": "Ticket T1 cancelled.",
                "content": "Ticket T1 cancelled.",
                "status": "cancelled",
                "is_active": True,
                "deactivation_filters": [
                    {
                        "memory_type": "entity_fact",
                        "scope": "thread",
                        "namespace": "thread:t1",
                        "entity_type": "ticket",
                        "entity_id": "T1",
                        "is_active": True,
                    }
                ],
            }
        )

        active = store.list_memories(filters={"thread_id": "t1"}, include_inactive=False)
        self.assertEqual([memory["memory_id"] for memory in active], [second["memory_id"]])

        all_memories = store.list_memories(filters={"thread_id": "t1"}, include_inactive=True)
        first_record = next(memory for memory in all_memories if memory["memory_id"] == first["memory_id"])
        self.assertFalse(first_record["is_active"])
        self.assertEqual(first_record["replaced_by"], second["memory_id"])

    def test_expired_memory_is_hidden_after_listing(self):
        store = self.make_store()
        store.append(
            {
                "memory_key": "candidate:t1:hotel:1",
                "memory_type": "candidate_option",
                "scope": "thread",
                "namespace": "thread:t1",
                "passenger_id": "p1",
                "thread_id": "t1",
                "summary": "Demo hotel option.",
                "content": "Demo hotel option.",
                "is_active": True,
                "expires_at": "2000-01-01T00:00:00+00:00",
            }
        )

        active = store.list_memories(filters={"thread_id": "t1"}, include_inactive=False)
        self.assertEqual(active, [])

        inactive = store.list_memories(filters={"thread_id": "t1"}, include_inactive=True)
        self.assertEqual(len(inactive), 1)
        self.assertFalse(inactive[0]["is_active"])
        self.assertIn("expired_at", inactive[0])

    def test_admin_inspect_and_forget_memory(self):
        store = self.make_store()
        store.append(
            {
                "memory_key": "profile:p1",
                "memory_type": "user_profile",
                "scope": "user",
                "namespace": "user:p1",
                "passenger_id": "p1",
                "thread_id": "t1",
                "summary": "User profile.",
                "content": "User profile.",
                "is_active": True,
            }
        )
        store.append(
            {
                "memory_key": "thread:t1:fact1",
                "memory_type": "trip_fact",
                "scope": "thread",
                "namespace": "thread:t1",
                "passenger_id": "p1",
                "thread_id": "t1",
                "summary": "Thread fact.",
                "content": "Thread fact.",
                "is_active": True,
            }
        )

        inspected = inspect_memory(
            store=store,
            passenger_id="p1",
            thread_id="t1",
            scope="thread",
            include_inactive=False,
            limit=10,
        )
        self.assertEqual(inspected["summary"]["total"], 1)
        self.assertEqual(inspected["memories"][0]["memory_type"], "trip_fact")

        forgotten = forget_memory(
            store=store,
            passenger_id="p1",
            thread_id="t1",
            scope="thread",
        )
        self.assertEqual(forgotten["purged_count"], 1)
        remaining = store.list_memories(filters={"passenger_id": "p1"}, include_inactive=False, limit=10)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["memory_type"], "user_profile")

    def test_retrieval_builds_agent_brief(self):
        store = self.make_store()
        store.append(
            {
                "memory_key": "profile:p1",
                "memory_type": "user_profile",
                "scope": "user",
                "namespace": "user:p1",
                "passenger_id": "p1",
                "thread_id": "t1",
                "summary": "Customer prefers midscale hotels.",
                "content": "Customer prefers midscale hotels.",
                "is_active": True,
            },
            write_audit=False,
        )
        store.append(
            {
                "memory_key": "fact:t1:1",
                "memory_type": "trip_fact",
                "scope": "thread",
                "namespace": "thread:t1",
                "passenger_id": "p1",
                "thread_id": "t1",
                "summary": "Customer arrives in Zurich on 2026-05-02.",
                "content": "Customer arrives in Zurich on 2026-05-02.",
                "is_active": True,
            },
            write_audit=False,
        )
        store.append(
            {
                "memory_key": "candidate:t1:hotel:1",
                "memory_type": "candidate_option",
                "scope": "thread",
                "namespace": "thread:t1",
                "passenger_id": "p1",
                "thread_id": "t1",
                "summary": "name=Demo Central Hotel, location=Zurich",
                "content": "name=Demo Central Hotel, location=Zurich",
                "is_active": True,
            },
            write_audit=False,
        )
        result = retrieve_experiences(
            {
                "active_agent": "book_hotel",
                "messages": [],
                "user_info": "Ticket Number: T1",
                "pending_handoff": {},
                "working_memory": {},
                "last_tool_result": {},
            },
            store,
            config={"configurable": {"passenger_id": "p1", "thread_id": "t1"}},
            top_k=8,
        )
        brief = result["working_memory"]["agent_brief"]
        self.assertIn("Profile:", brief)
        self.assertIn("Fact:", brief)
        self.assertIn("Option:", brief)

    def test_handle_handoff_augments_context_from_memory(self):
        state = {
            "working_memory": {
                "memory_trip_facts": [
                    {
                        "memory_type": "trip_fact",
                        "summary": "Customer now arrives in Zurich on 2026-05-02.",
                        "tags": ["trip_state"],
                    }
                ],
                "memory_candidate_options": [
                    {
                        "memory_type": "candidate_option",
                        "summary": "name=Demo Central Hotel, location=Zurich",
                        "tags": ["candidate_option", "hotel_option", "search_hotels"],
                    }
                ],
                "memory_policy_snapshots": [
                    {
                        "memory_type": "policy_snapshot",
                        "summary": "Same-day hotel rebooking allowed.",
                        "tags": ["policy_snapshot"],
                    }
                ],
                "memory_open_loops": [
                    {
                        "memory_type": "open_loop",
                        "summary": "Need hotel budget confirmation.",
                        "tags": ["open_loop"],
                    }
                ],
            },
        }
        context = build_handoff_memory_brief(state, "book_hotel")
        self.assertIn("Current confirmed facts:", context)
        self.assertIn("Relevant candidate options:", context)
        self.assertIn("Recent policy evidence:", context)
        self.assertIn("Outstanding open loops:", context)


if __name__ == "__main__":
    unittest.main()

"""
Memory Framework Client
========================

Async client adapter that wraps the Memory Framework's public SDK
to conform to the ``mem0ai/memory-benchmarks`` pluggable technique interface.

Usage::

    engine = MemoryEngine.from_yaml("configs/memory-framework.yaml")
    observer = InMemoryObserver()
    client = MemoryFrameworkClient(engine=engine, observer=observer)

    async with client:
        result = await client.add(messages, user_id, timestamp=epoch)
        hits = await client.search(query, user_id, top_k=200)
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class MemoryFrameworkClient:
    """Async adapter wrapping Memory Framework's synchronous public SDK.

    Args:
        engine: Pre-configured ``MemoryEngine`` instance.
        observer: ``InMemoryObserver`` for token usage capture.
        max_workers: Thread pool size for ``asyncio.to_thread`` dispatch.
    """

    def __init__(
        self,
        *,
        engine: Any,
        observer: Any | None = None,
        max_workers: int = 4,
    ) -> None:
        self._engine = engine
        self._observer = observer
        self._executor = ThreadPoolExecutor(max_workers=int(max_workers))

    # ------------------------------------------------------------------
    # Context manager (compatible with ``async with client:``)
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "MemoryFrameworkClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Shut down the dedicated thread pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    # ------------------------------------------------------------------
    # Core interface (matches Mem0Client)
    # ------------------------------------------------------------------

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        observation_date: str | None = None,
        timestamp: int | None = None,
        custom_instructions: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        """Ingest conversation messages and extract memories.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            user_id: Unique user identifier.
            timestamp: Unix epoch seconds (preferred). Converted to
                timezone-aware UTC datetime for ``observed_at``.

        Returns:
            Dict with ``"results"`` key listing extracted memories, or
            ``None`` on failure.
        """
        observed_at = None
        if timestamp is not None:
            observed_at = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
        elif observation_date is not None:
            try:
                d = datetime.strptime(observation_date, "%Y-%m-%d")
                observed_at = d.replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        try:
            result = await asyncio.to_thread(
                self._engine.add_memories_sync,
                user_id=str(user_id),
                messages=[dict(m) for m in messages],
                observed_at=observed_at,
            )
            # Map AddMemoriesResult → Mem0-compatible response format
            relation_memories = list(getattr(result.relation_result, "memories", []) or [])
            text_by_id = {
                str(item.get("memory_id")): str(item.get("text") or item.get("canonical_text") or "")
                for item in relation_memories
                if item.get("memory_id")
            }
            memories = []
            for mid in result.accepted_memory_ids:
                memory_id = str(mid)
                memories.append(
                    {
                        "memory": text_by_id.get(memory_id, ""),
                        "event": "ADD",
                        "id": memory_id,
                    }
                )
            relation_debug = _relation_debug_payload(self._engine, result)
            return {
                "results": memories,
                "memory_count": len(memories),
                "source_batch_id": result.source_batch_id,
                "debug": relation_debug,
            }
        except Exception as exc:
            logger.warning("ADD failed (user=%s, msgs=%d): %s", user_id, len(messages), str(exc)[:200])
            return None

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 200,
        rerank: bool = False,
        score_debug: bool = False,
    ) -> list[dict]:
        """Search memories for a query.

        Args:
            query: Search query text.
            user_id: User to search within.
            top_k: Maximum results to return.

        Returns:
            List of result dicts sorted by score descending, each with
            ``"id"``, ``"memory"``, and ``"score"`` keys.
        """
        try:
            response = await asyncio.to_thread(
                self._engine.search_memories,
                user_id=str(user_id),
                query=str(query),
                top_k=int(top_k),
            )
            hits = []
            for hit in response.hits:
                entry: dict[str, Any] = {
                    "id": hit.memory_id,
                    "memory": hit.text,
                    "score": hit.score,
                }
                if score_debug and hit.internal is not None:
                    entry["score_debug"] = {
                        "combined_score": hit.score,
                        "semantic_score": hit.internal.similarity,
                        "bm25_score": hit.internal.bm25_score or 0,
                        "entity_boost": hit.internal.entity_boost,
                    }
                hits.append(entry)
            hits.sort(key=lambda x: x.get("score", 0), reverse=True)
            return hits
        except Exception as exc:
            logger.warning("SEARCH failed (user=%s, query=%s): %s", user_id, query[:80], str(exc)[:200])
            return []

    async def get_user_profile(self, user_id: str) -> dict | None:
        """Fetch user profile for the given user.

        Returns:
            Dict with ``"profiles"`` key, or ``None`` on failure.
        """
        try:
            profile = await asyncio.to_thread(
                self._engine.get_user_profile,
                user_id=str(user_id),
            )
            return {
                "profiles": [
                    {
                        "topic": p.topic,
                        "subtopic": p.subtopic,
                        "content": p.content,
                    }
                    for p in profile.profiles
                ]
            }
        except Exception as exc:
            logger.warning("get_user_profile failed (user=%s): %s", user_id, str(exc)[:200])
            return None

    async def delete_user(self, user_id: str) -> bool:
        """Delete all memories for a user.

        Sprint 1: not required — benchmarks use unique ``user_id`` per
        conversation/question. Implemented as no-op.
        """
        return True


def _relation_debug_payload(engine: Any, result: Any) -> dict[str, Any]:
    relation_result = getattr(result, "relation_result", None)
    decisions = list(getattr(relation_result, "decisions", []) or [])
    diagnostics = getattr(getattr(engine, "relation_workflow_stage", None), "last_diagnostics", None)
    update_operations = list(getattr(diagnostics, "update_operations", []) or [])
    updated_text_by_cluster_id = {
        str(decision.get("primary_target_cluster_id") or decision.get("target_cluster_id") or ""): str(
            decision.get("updated_canonical_text") or decision.get("content") or ""
        )
        for decision in decisions
        if isinstance(decision, dict)
        and str(decision.get("primary_action") or decision.get("action") or "").upper() == "UPDATE"
    }
    return {
        "relation_status": getattr(relation_result, "status", None),
        "relation_decisions": decisions,
        "updated_clusters": [
            {
                "cluster_id": str(getattr(item, "cluster_id", "")),
                "previous_text": str(getattr(item, "previous_canonical_text", "")),
                "text": updated_text_by_cluster_id.get(str(getattr(item, "cluster_id", "")), ""),
            }
            for item in update_operations
        ],
    }

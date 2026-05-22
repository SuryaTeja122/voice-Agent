"""
memory/session_store.py  —  Redis-backed session with TTL
memory/long_term.py      —  PostgreSQL + pgvector long-term memory
"""

# ════════════════════════════════════════════════════════════════
# session_store.py
# ════════════════════════════════════════════════════════════════

import json
import logging
from typing import Optional

import redis.asyncio as aioredis

from agent.session import SessionState

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = 60 * 60 * 2        # 2 hours
PATIENT_ID_TTL_SECONDS = 60 * 60 * 24    # 24 hours (for lookup across reconnects)


class SessionStore:
    """Redis-backed session store for per-call state."""

    def __init__(self, redis_client: aioredis.Redis):
        self.r = redis_client

    async def get(self, call_id: str) -> Optional[SessionState]:
        raw = await self.r.get(f"session:{call_id}")
        if raw is None:
            return None
        try:
            return SessionState.from_json(raw)
        except Exception as e:
            logger.warning("Failed to deserialise session %s: %s", call_id, e)
            return None

    async def set(self, call_id: str, state: SessionState) -> None:
        await self.r.setex(
            f"session:{call_id}",
            SESSION_TTL_SECONDS,
            state.to_json(),
        )

    async def delete(self, call_id: str) -> None:
        await self.r.delete(f"session:{call_id}")

    async def get_patient_id(self, call_id: str) -> Optional[str]:
        """Fast lookup for patient_id without deserialising full session."""
        raw = await self.r.get(f"session:{call_id}")
        if not raw:
            return None
        try:
            data = json.loads(raw)
            return data.get("patient_id")
        except Exception:
            return None

    async def bind_patient(self, call_id: str, patient_id: str) -> None:
        """Associates a patient_id with this call for cross-session tracking."""
        await self.r.setex(
            f"call_patient:{call_id}",
            PATIENT_ID_TTL_SECONDS,
            patient_id,
        )

    async def ping(self) -> bool:
        try:
            return await self.r.ping()
        except Exception:
            return False


# ════════════════════════════════════════════════════════════════
# long_term.py
# ════════════════════════════════════════════════════════════════

import asyncpg
from datetime import datetime


class LongTermMemory:
    """
    Manages long-term patient memory using PostgreSQL + pgvector.

    Schema (create in alembic migration):
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE interaction_summaries (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            patient_id TEXT NOT NULL,
            call_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            embedding vector(1536),
            language TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX ON interaction_summaries
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
    """

    def __init__(self, pool: asyncpg.Pool, openai_client=None):
        self.pool = pool
        self._openai = openai_client  # for generating embeddings

    async def get_context_summary(
        self,
        patient_id: str,
        top_k: int = 3,
        query_text: Optional[str] = None,
    ) -> Optional[str]:
        """
        Retrieves the most relevant past interactions for a patient.
        Uses pgvector cosine similarity if query_text is provided,
        otherwise falls back to most-recent interactions.
        """
        async with self.pool.acquire() as conn:
            if query_text and self._openai:
                embedding = await self._embed(query_text)
                rows = await conn.fetch(
                    """SELECT summary, language, created_at
                       FROM interaction_summaries
                       WHERE patient_id = $1
                       ORDER BY embedding <=> $2::vector
                       LIMIT $3""",
                    patient_id,
                    embedding,
                    top_k,
                )
            else:
                rows = await conn.fetch(
                    """SELECT summary, language, created_at
                       FROM interaction_summaries
                       WHERE patient_id = $1
                       ORDER BY created_at DESC
                       LIMIT $2""",
                    patient_id,
                    top_k,
                )

            if not rows:
                return None

            parts = []
            for r in rows:
                date_str = r["created_at"].strftime("%d %b %Y")
                parts.append(f"[{date_str}] {r['summary']}")

            return "\n".join(parts)

    async def save_interaction(
        self,
        patient_id: str,
        call_id: str,
        summary: str,
        language: str,
    ) -> None:
        """
        Saves a post-call summary with embedding for future RAG retrieval.
        Called asynchronously after call ends.
        """
        embedding = None
        if self._openai:
            try:
                embedding = await self._embed(summary)
            except Exception as e:
                logger.warning("Failed to generate embedding: %s", e)

        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO interaction_summaries
                   (patient_id, call_id, summary, embedding, language, created_at)
                   VALUES ($1, $2, $3, $4::vector, $5, NOW())
                   ON CONFLICT (call_id) DO NOTHING""",
                patient_id,
                call_id,
                summary,
                embedding,
                language,
            )

    async def _embed(self, text: str) -> list[float]:
        """Generate text embedding via OpenAI text-embedding-3-small."""
        response = await self._openai.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000],  # truncate to model limit
        )
        return response.data[0].embedding

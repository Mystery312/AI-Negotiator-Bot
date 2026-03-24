"""Semantic cache for RAG query results.

Uses cosine similarity of sentence-transformer embeddings to serve cached
results when a new query is semantically close to a previous one
(similarity > threshold).  Thread-safe via threading.Lock().
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    query_text: str
    query_embedding: np.ndarray  # 384-dim for all-MiniLM-L6-v2
    result: Any
    timestamp: float = field(default_factory=time.time)


class SemanticCache:
    """Cosine-similarity cache for RAG queries."""

    def __init__(
        self,
        similarity_threshold: float = 0.9,
        ttl_seconds: float = 600,
        max_entries: int = 200,
    ) -> None:
        self._threshold = similarity_threshold
        self._ttl = ttl_seconds
        self._max = max_entries
        self._entries: list[CacheEntry] = []
        self._lock = threading.Lock()
        self._model = None
        self._hits = 0
        self._misses = 0

    # -- embedding model (lazy) -----------------------------------------------

    def _get_model(self):
        """Reuse the RAG embedding model if already loaded, else load one."""
        if self._model is not None:
            return self._model
        try:
            from negotiation_chatbot.rag import embedding_model as _rag_model, _ensure_initialized
            _ensure_initialized()
            from negotiation_chatbot.rag import embedding_model as _rag_model_loaded
            if _rag_model_loaded is not None:
                self._model = _rag_model_loaded
                return self._model
        except Exception:
            pass
        # Fallback: load independently
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    # -- public API ------------------------------------------------------------

    def get(self, query: str) -> Tuple[Any, bool]:
        """Return (result, True) on cache hit, (None, False) on miss."""
        model = self._get_model()
        q_emb = model.encode(query, convert_to_numpy=True)
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)

        now = time.time()
        with self._lock:
            for entry in self._entries:
                if now - entry.timestamp > self._ttl:
                    continue
                e_norm = entry.query_embedding / (np.linalg.norm(entry.query_embedding) + 1e-9)
                sim = float(np.dot(q_norm, e_norm))
                if sim >= self._threshold:
                    self._hits += 1
                    logger.info("Semantic cache HIT (sim=%.3f) for query: %s", sim, query[:80])
                    return entry.result, True

        self._misses += 1
        return None, False

    def put(self, query: str, result: Any) -> None:
        """Store a query result."""
        model = self._get_model()
        q_emb = model.encode(query, convert_to_numpy=True)

        entry = CacheEntry(
            query_text=query,
            query_embedding=q_emb,
            result=result,
            timestamp=time.time(),
        )

        now = time.time()
        with self._lock:
            # Evict expired
            self._entries = [e for e in self._entries if now - e.timestamp <= self._ttl]
            # Evict oldest if at capacity
            if len(self._entries) >= self._max:
                self._entries.sort(key=lambda e: e.timestamp)
                self._entries = self._entries[-(self._max - 1):]
            self._entries.append(entry)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0
        logger.info("Semantic cache cleared")

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / max(self._hits + self._misses, 1),
            }

"""
AMEM (Adaptive Memory) — lightweight semantic memory store.

Design goals:
- Zero external services; pure local in-memory store with optional persistence.
- Pluggable embedding backend:
  - Preferred: sentence-transformers (dense embeddings).
  - Fallback: sklearn TF-IDF (sparse) if sentence-transformers not installed.

The store is intentionally simple so it can be injected into existing agents
without changing their core logic.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MemoryItem:
    text: str
    meta: Dict[str, Any]


class AMEMStore:
    """A tiny semantic memory store with add/search/save/load."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
        persist_path: Optional[str] = None,
    ):
        self.embedding_model_name = embedding_model
        self.normalize = normalize
        self.persist_path = persist_path

        self._items: List[MemoryItem] = []
        self._embeddings: Optional[np.ndarray] = None  # shape: (N, D)

        self._backend = None
        self._encoder = None

        self._init_backend()

        if self.persist_path:
            try:
                self.load(self.persist_path)
            except Exception:
                # Ignore load errors — empty store is fine
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        text = (text or "").strip()
        if not text:
            return
        meta = meta or {}
        self._items.append(MemoryItem(text=text, meta=dict(meta)))
        emb = self._embed([text])  # (1, D)
        if self._embeddings is None:
            self._embeddings = emb
        else:
            self._embeddings = np.vstack([self._embeddings, emb])

        if self.persist_path:
            try:
                self.save(self.persist_path)
            except Exception:
                pass

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, MemoryItem]]:
        query = (query or "").strip()
        if not query or not self._items or self._embeddings is None:
            return []
        q = self._embed([query])  # (1, D)
        scores = self._cosine_sim_matrix(q, self._embeddings).flatten()  # (N,)
        top_k = max(1, int(top_k))
        idx = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), self._items[int(i)]) for i in idx]

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedding_model": self.embedding_model_name,
            "normalize": self.normalize,
            "items": [{"text": it.text, "meta": it.meta} for it in self._items],
            # store embeddings as lists for portability
            "embeddings": self._embeddings.tolist() if self._embeddings is not None else None,
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        payload = json.loads(p.read_text(encoding="utf-8"))
        self.embedding_model_name = payload.get("embedding_model", self.embedding_model_name)
        self.normalize = bool(payload.get("normalize", self.normalize))
        self._items = [MemoryItem(text=x["text"], meta=x.get("meta", {})) for x in payload.get("items", [])]
        emb = payload.get("embeddings")
        self._embeddings = np.asarray(emb, dtype=float) if emb is not None else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_backend(self) -> None:
        """Pick embedding backend."""
        # Preferred: sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._backend = "sentence-transformers"
            self._encoder = SentenceTransformer(self.embedding_model_name)
            return
        except Exception:
            self._backend = "tfidf"

        # Fallback: TF-IDF vectorizer (fit-on-add)
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

        self._encoder = TfidfVectorizer()

    def _embed(self, texts: List[str]) -> np.ndarray:
        if self._backend == "sentence-transformers":
            vec = np.asarray(self._encoder.encode(texts, show_progress_bar=False), dtype=float)
        else:
            # TF-IDF: refit on all texts (cheap for small memories)
            all_texts = [it.text for it in self._items]
            # Ensure we include current texts too if called before add
            for t in texts:
                if t not in all_texts:
                    all_texts.append(t)
            mat = self._encoder.fit_transform(all_texts)
            # Take embeddings for requested texts
            # Map text->row index (first occurrence)
            idx_map = {}
            for i, t in enumerate(all_texts):
                if t not in idx_map:
                    idx_map[t] = i
            rows = [idx_map[t] for t in texts]
            vec = mat[rows].toarray().astype(float)

        if self.normalize:
            vec = self._l2_normalize(vec)
        return vec

    @staticmethod
    def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return x / norms

    @staticmethod
    def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Assumes both are already L2-normalized if normalize=True.
        return a @ b.T

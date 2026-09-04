"""
Lightweight semantic engine with aggressive caching.
Uses all-MiniLM-L6-v2 (~80 MB). All encodings are normalized.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class HDCSearch:
    _MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self._model = SentenceTransformer(self._MODEL_NAME)
        self._cache: dict[str, np.ndarray] = {}
        self._dim = self._model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Core encoding (cached)
    # ------------------------------------------------------------------

    def _key(self, text: str, max_chars: int = 1500) -> str:
        return (text or "")[:max_chars].strip()

    def encode(self, text: str) -> np.ndarray:
        key = self._key(text)
        if not key:
            return np.zeros(self._dim, dtype=np.float32)
        if key not in self._cache:
            self._cache[key] = self._model.encode(
                key, normalize_embeddings=True, show_progress_bar=False
            )
        return self._cache[key]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        keys = [self._key(t) for t in texts]
        missing = list(dict.fromkeys(k for k in keys if k and k not in self._cache))

        if missing:
            vecs = self._model.encode(
                missing,
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=False,
            )
            for k, v in zip(missing, vecs):
                self._cache[k] = v

        # Missing / empty keys → zero vector
        out = []
        for k in keys:
            if k and k in self._cache:
                out.append(self._cache[k])
            else:
                out.append(np.zeros(self._dim, dtype=np.float32))
        return np.stack(out)

    # ------------------------------------------------------------------
    # Similarity helpers
    # ------------------------------------------------------------------

    def similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return float(np.dot(self.encode(a), self.encode(b)))

    def bulk_similarities(
        self, queries: list[str], candidates: list[str]
    ) -> list[float]:
        """For each query return the max cosine similarity against any candidate."""
        if not queries or not candidates:
            return [0.0] * len(queries)
        qv = self.encode_batch(queries)
        cv = self.encode_batch(candidates)
        matrix = qv @ cv.T
        return [float(row.max()) for row in matrix]

    def best_match(self, query: str, candidates: list[str]) -> tuple[float, str]:
        if not candidates:
            return 0.0, ""
        scores = self.bulk_similarities([query], candidates)
        best = int(np.argmax(scores))
        return scores[0], candidates[best]

    # ------------------------------------------------------------------
    # Warm-up / cache management
    # ------------------------------------------------------------------

    def warm_batch(self, resume: dict) -> None:
        """Pre-encode every text fragment we will later compare."""
        texts: list[str] = []

        # Skills (prefer cleaned derived list)
        derived = resume.get("derived") or {}
        skills = derived.get("skills_normalized") or resume.get("skills") or []
        for s in skills:
            s = str(s).strip()
            if s:
                texts.append(s)

        # Work experience blobs
        for role in resume.get("work_experience") or []:
            parts = [role.get("title") or "", role.get("company") or ""]
            parts += role.get("highlights") or []
            blob = " ".join(p for p in parts if p)
            if blob.strip():
                texts.append(blob)

        # Projects
        for p in resume.get("projects") or []:
            blob = " ".join(filter(None, [p.get("title"), p.get("description")]))
            if blob.strip():
                texts.append(blob)

        # Summary + leadership (used as fallback profile text)
        if resume.get("summary"):
            texts.append(resume["summary"])
        for lead in resume.get("leadership") or []:
            parts = [lead.get("role") or ""]
            parts += lead.get("highlights") or []
            blob = " ".join(p for p in parts if p)
            if blob.strip():
                texts.append(blob)

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped = []
        for t in texts:
            if t not in seen:
                seen.add(t)
                deduped.append(t)

        if deduped:
            self.encode_batch(deduped)

    def clear_cache(self) -> None:
        self._cache.clear()
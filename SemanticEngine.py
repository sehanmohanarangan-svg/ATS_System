# SemanticEngine.py
import numpy as np
from sentence_transformers import SentenceTransformer


class HDCSearch:
    _MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self._model = SentenceTransformer(self._MODEL_NAME)
        self._cache: dict[str, np.ndarray] = {}

    def _truncate(self, text: str, max_chars: int = 1500) -> str:
        return text[:max_chars]

    def encode(self, text: str) -> np.ndarray:
        key = text[:1500]
        if key not in self._cache:
            self._cache[key] = self._model.encode(
                key, normalize_embeddings=True, show_progress_bar=False
            )
        return self._cache[key]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            # Return empty array with correct embedding dimension
            return np.empty((0, self._model.get_sentence_embedding_dimension()),
                            dtype=np.float32)
        keys    = [t[:1500] for t in texts]
        missing = list(dict.fromkeys(k for k in keys if k not in self._cache))
        if missing:
            vecs = self._model.encode(
                missing,
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=False,
            )
            for k, v in zip(missing, vecs):
                self._cache[k] = v
        return np.stack([self._cache[k] for k in keys])

    def similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return float(np.dot(self.encode(a), self.encode(b)))

    def best_match(self, query: str, candidates: list[str]) -> tuple[float, str]:
        if not candidates:
            return 0.0, ""
        qv     = self.encode(query)
        cv     = self.encode_batch(candidates)
        scores = cv @ qv
        best   = int(scores.argmax())
        return float(scores[best]), candidates[best]

    def bulk_similarities(self, queries: list[str], candidates: list[str]) -> list[float]:
        if not queries or not candidates:
            return [0.0] * len(queries)
        qv     = self.encode_batch(queries)
        cv     = self.encode_batch(candidates)
        matrix = qv @ cv.T
        return [float(row.max()) for row in matrix]

    def clear_cache(self) -> None:
        self._cache.clear()

    def warm_batch(self, resume: dict) -> None:
        """Pre-encode all resume texts in one batch call."""
        texts = []

        # Use cleaned skills from derived if available, else raw skills
        derived = resume.get("derived") or {}
        skills  = derived.get("skills_normalized") or resume.get("skills") or []
        for s in skills:
            if s and str(s).strip():
                texts.append(str(s))

        for role in (resume.get("work_experience") or []):
            parts = [role.get("title", ""), role.get("company", "")]
            parts += role.get("highlights") or []
            blob = " ".join(p for p in parts if p)
            if blob.strip():
                texts.append(blob)

        for p in (resume.get("projects") or []):
            blob = " ".join(filter(None, [p.get("title"), p.get("description")]))
            if blob.strip():
                texts.append(blob)

        summary = resume.get("summary") or ""
        if summary.strip():
            texts.append(summary)

        # Deduplicate while preserving order
        seen, deduped = set(), []
        for t in texts:
            if t not in seen:
                seen.add(t)
                deduped.append(t)

        if deduped:
            self.encode_batch(deduped)
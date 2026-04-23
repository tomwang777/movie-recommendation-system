import hashlib
import json
import os

import numpy as np
import pandas as pd
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 500
MIN_SCORE = 0.20  # cosine similarity threshold; results below this are suppressed


def _csv_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_feature(row: pd.Series) -> str:
    fields = ["listed_in", "description", "director", "country", "cast"]
    parts = [str(row.get(f, "")).strip() for f in fields]
    return ". ".join(p for p in parts if p)


class Recommender:
    def __init__(self, data_path: str, cache_path: str = "embeddings_cache.npy"):
        self._cache_meta_path = cache_path.replace(".npy", "_meta.json")
        self.client = OpenAI()  # reads OPENAI_API_KEY from env

        df = pd.read_csv(data_path).fillna("")
        df = df[df["type"].str.strip().str.lower() == "movie"].reset_index(drop=True)
        df["_feature"] = df.apply(_build_feature, axis=1)
        self.df = df

        csv_hash = _csv_hash(data_path)
        if self._cache_valid(cache_path, csv_hash):
            print("Loading cached embeddings...")
            raw = np.load(cache_path)
        else:
            print(f"Embedding {len(df)} movies via OpenAI ({EMBED_MODEL})...")
            raw = self._embed_all(df["_feature"].tolist())
            np.save(cache_path, raw)
            with open(self._cache_meta_path, "w") as f:
                json.dump({"hash": csv_hash, "rows": len(df)}, f)
            print("Embeddings cached.")

        # Pre-normalize rows so query scoring is a single dot product
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        self.embeddings = (raw / np.maximum(norms, 1e-9)).astype(np.float32)

    def _cache_valid(self, cache_path: str, csv_hash: str) -> bool:
        if not (os.path.exists(cache_path) and os.path.exists(self._cache_meta_path)):
            return False
        with open(self._cache_meta_path) as f:
            meta = json.load(f)
        return meta.get("hash") == csv_hash and meta.get("rows") == len(self.df)

    def _embed_all(self, texts: list[str]) -> np.ndarray:
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = self.client.embeddings.create(model=EMBED_MODEL, input=batch)
            vecs = [e.embedding for e in sorted(response.data, key=lambda x: x.index)]
            all_vecs.extend(vecs)
            print(f"  {min(i + BATCH_SIZE, len(texts))}/{len(texts)} embedded")
        return np.array(all_vecs, dtype=np.float32)

    def _rank(self, query: str, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Embed query and return (sorted row indices, scores) above MIN_SCORE."""
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=[query])
        q = np.array(resp.data[0].embedding, dtype=np.float32)
        q /= max(float(np.linalg.norm(q)), 1e-9)
        scores = self.embeddings @ q  # cosine similarity (rows are pre-normalized)
        top_idx = np.argsort(scores)[::-1][:top_k]
        mask = scores[top_idx] >= MIN_SCORE
        return top_idx[mask], scores[top_idx[mask]]

    def recommend(self, query: str, top_k: int = 10) -> list[dict]:
        """Return top-k movies for a natural language query.

        Args:
            query: Natural language description, e.g. 'funny sci-fi with aliens'.
            top_k: Maximum number of results to return.

        Returns:
            List of dicts with 'title', 'description', 'score'.
            Empty list if no titles score above MIN_SCORE.

        Raises:
            ValueError: if query is empty or blank.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        indices, scores = self._rank(query.strip(), top_k)
        results = []
        for idx, score in zip(indices, scores):
            row = self.df.iloc[idx]
            results.append({
                "title": str(row.get("title", "")),
                "description": str(row.get("description", "")),
                "score": round(float(score), 4),
            })
        return results

    def search(self, query: str, top_n: int = 10) -> list[dict]:
        """Extended version of recommend() used by the API — includes extra metadata."""
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        indices, scores = self._rank(query.strip(), top_n)
        results = []
        for idx, score in zip(indices, scores):
            row = self.df.iloc[idx]
            results.append({
                "title": str(row.get("title", "")),
                "type": "Movie",
                "description": str(row.get("description", "")),
                "genres": str(row.get("listed_in", "")),
                "release_year": str(row.get("release_year", "")),
                "rating": str(row.get("rating", "")),
                "duration": str(row.get("duration", "")),
                "score": round(float(score), 4),
            })
        return results

"""
vector_store.py
---------------
FAISS-backed vector store for resume and job-description embeddings.

FAISS only stores raw float32 vectors — it has no concept of metadata.
This module pairs every vector with a JSON metadata record stored in a
parallel list, so you can retrieve "which resume / job does this vector
belong to?" after a search.

Typical usage
-------------
Store a resume's profile embedding:

    store = ResumeVectorStore.load_or_create("output/resume_index")
    store.add(
        vector=embeddings["profile_embedding"],
        metadata={"type": "resume", "name": "John Doe", "source": "john_doe.pdf"},
    )
    store.save("output/resume_index")

Later, search for jobs similar to the resume:

    store = ResumeVectorStore.load_or_create("output/job_index")
    results = store.search(query_vector=resume_embedding, top_k=5)
    for score, meta in results:
        print(score, meta["job_title"], meta["company"])
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np


# ── store ──────────────────────────────────────────────────────────────────────

class ResumeVectorStore:
    """
    A flat (exact) FAISS index with associated metadata.

    Uses IndexFlatIP (inner product) on L2-normalised vectors,
    which is equivalent to cosine similarity.
    """

    _INDEX_FILE    = "faiss.index"
    _METADATA_FILE = "metadata.json"

    def __init__(self, dim: int) -> None:
        """
        Args:
            dim: Embedding dimension (must match the model used).
                 bge-large-en / e5-large → 1024
                 all-mpnet-base-v2       → 768
        """
        self.dim   = dim
        self.index = faiss.IndexFlatIP(dim)   # inner-product = cosine on unit vectors
        self.metadata: list[dict] = []

    # ── persistence ─────────────────────────────────────────────────────────────

    def save(self, store_dir: str | Path) -> None:
        """Persist the FAISS index and metadata to disk."""
        store_dir = Path(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(store_dir / self._INDEX_FILE))
        with open(store_dir / self._METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "metadata": self.metadata}, f, indent=2)

    @classmethod
    def load(cls, store_dir: str | Path) -> "ResumeVectorStore":
        """
        Load a previously saved store from disk.

        Raises:
            FileNotFoundError: if the directory or index files are missing.
        """
        store_dir = Path(store_dir)
        index_path    = store_dir / cls._INDEX_FILE
        metadata_path = store_dir / cls._METADATA_FILE

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"No saved store found at {store_dir}. "
                "Create a new one with ResumeVectorStore(dim=...)."
            )

        with open(metadata_path, encoding="utf-8") as f:
            meta_doc = json.load(f)

        store = cls(dim=meta_doc["dim"])
        store.index    = faiss.read_index(str(index_path))
        store.metadata = meta_doc["metadata"]
        return store

    @classmethod
    def load_or_create(cls, store_dir: str | Path, dim: int = 1024) -> "ResumeVectorStore":
        """
        Load an existing store or create a fresh one.

        Args:
            store_dir: Directory path for index files.
            dim:       Embedding dimension, used only when creating a new store.
        """
        try:
            return cls.load(store_dir)
        except FileNotFoundError:
            return cls(dim=dim)

    # ── CRUD ─────────────────────────────────────────────────────────────────────

    def add(self, vector: np.ndarray, metadata: dict) -> int:
        """
        Add one vector with its metadata.

        Args:
            vector:   1-D float32 numpy array, shape (dim,). Should be L2-normalised.
            metadata: Arbitrary JSON-serialisable dict (e.g. name, source, type).

        Returns:
            The integer ID assigned to this entry (0-based insertion order).
        """
        vec = _validate_vector(vector, self.dim)
        self.index.add(vec)                 # FAISS expects shape (1, dim)
        self.metadata.append(metadata)
        return len(self.metadata) - 1

    def add_batch(self, vectors: np.ndarray, metadata_list: list[dict]) -> list[int]:
        """
        Add multiple vectors at once (faster than adding one by one).

        Args:
            vectors:       2-D float32 array, shape (n, dim).
            metadata_list: List of n metadata dicts.

        Returns:
            List of assigned IDs.
        """
        if len(vectors) != len(metadata_list):
            raise ValueError(
                f"vectors ({len(vectors)}) and metadata_list ({len(metadata_list)}) "
                "must have the same length."
            )
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"Expected shape (n, {self.dim}), got {vecs.shape}")

        start_id = len(self.metadata)
        self.index.add(vecs)
        self.metadata.extend(metadata_list)
        return list(range(start_id, start_id + len(metadata_list)))

    def __len__(self) -> int:
        return self.index.ntotal

    # ── search ────────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[float, dict]]:
        """
        Find the top_k most similar vectors to query_vector.

        Args:
            query_vector: 1-D float32 numpy array, shape (dim,). Should be L2-normalised.
            top_k:        Number of results to return.

        Returns:
            List of (cosine_score, metadata) tuples, sorted by score descending.
            cosine_score is in [-1, 1]; higher = more similar.
        """
        if self.index.ntotal == 0:
            return []

        k   = min(top_k, self.index.ntotal)
        vec = _validate_vector(query_vector, self.dim)

        scores, indices = self.index.search(vec, k)

        results: list[tuple[float, dict]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:          # FAISS returns -1 for missing results
                continue
            results.append((float(score), self.metadata[idx]))

        return results

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two single vectors (already normalised)."""
        a = np.asarray(vec_a, dtype=np.float32).flatten()
        b = np.asarray(vec_b, dtype=np.float32).flatten()
        return float(np.dot(a, b))


# ── helpers ───────────────────────────────────────────────────────────────────

def _validate_vector(vector: np.ndarray, dim: int) -> np.ndarray:
    """Ensure vector is float32 with the right dim, shaped (1, dim) for FAISS."""
    v = np.asarray(vector, dtype=np.float32).flatten()
    if v.shape[0] != dim:
        raise ValueError(f"Expected embedding dim {dim}, got {v.shape[0]}")
    return v.reshape(1, dim)


# ── convenience: infer dim from model key ─────────────────────────────────────

MODEL_DIMS: dict[str, int] = {
    "BAAI/bge-large-en":                         1024,
    "intfloat/e5-large":                         1024,
    "sentence-transformers/all-mpnet-base-v2":    768,
    "sentence-transformers/all-MiniLM-L6-v2":     384,
    "nvidia/nv-embedqa-e5-v5":                   1024,
}

def dim_for_model(model_name: str) -> int:
    """Return embedding dimension, honoring optional model suffix format `...::dimN`."""
    marker = "::dim"
    if marker in model_name:
        try:
            return int(model_name.rsplit(marker, 1)[1])
        except (TypeError, ValueError):
            pass
    return MODEL_DIMS.get(model_name, 1024)

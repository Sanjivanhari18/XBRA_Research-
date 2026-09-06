"""ChromaDB memory layer — stores investor profiles and bias vectors so the
orchestrator can retrieve similar past profiles during analysis.

Collections:
  - "investor_profiles"  : one doc per investor, metadata = ground_truth_bias
  - "bias_explanations"  : stores generated explanations for retrieval/RAG

Used by the Orchestrator to:
  1. Find population baselines (median features across all stored investors)
  2. Find peer percentile rank for the current investor
  3. Retrieve similar investor profiles as few-shot context for the LLM
"""

from __future__ import annotations

from typing import Any

from config.settings import CHROMA_COLLECTION, CHROMA_PERSIST_DIR


def get_chroma_client():
    """Return a persistent ChromaDB client."""
    try:
        import chromadb
    except ImportError:
        raise RuntimeError("chromadb not installed — run: pip install chromadb")
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def get_or_create_collection(name: str = CHROMA_COLLECTION):
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


class InvestorMemoryStore:
    """CRUD wrapper around the ChromaDB investor_profiles collection."""

    def __init__(self) -> None:
        self._col = get_or_create_collection("investor_profiles")

    def upsert_investor(
        self,
        investor_id: str,
        feature_vector: list[float],
        bias_label: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add or update an investor's feature embedding."""
        meta = {"bias_label": bias_label, **(metadata or {})}
        self._col.upsert(
            ids=[investor_id],
            embeddings=[feature_vector],
            metadatas=[meta],
        )

    def query_similar(
        self,
        feature_vector: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """Return the n closest investors by cosine similarity."""
        kwargs: dict[str, Any] = {
            "query_embeddings": [feature_vector],
            "n_results": n_results,
            "include": ["metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        results = self._col.query(**kwargs)
        out = []
        for inv_id, meta, dist in zip(
            results["ids"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            out.append({"investor_id": inv_id, "metadata": meta, "distance": dist})
        return out

    def count(self) -> int:
        return self._col.count()

    def get_all_metadata(self) -> list[dict]:
        """Return metadata for all stored investors (for population baseline)."""
        results = self._col.get(include=["metadatas", "embeddings"])
        return [
            {"investor_id": inv_id, "metadata": meta, "embedding": emb}
            for inv_id, meta, emb in zip(
                results["ids"], results["metadatas"], results["embeddings"]
            )
        ]

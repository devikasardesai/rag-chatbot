"""Stage 5 storage: ChromaDB persistence, idempotent upserts, and cosine search."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
import numpy as np

from src import config, logger

if TYPE_CHECKING:  # pragma: no cover
    from chromadb.api.models.Collection import Collection

    from src.chunking import Chunk

SCALAR_TYPES = (str, int, float, bool)
COLLECTION_METADATA_KEYS = ("hnsw:space", "corpus_version", "embedding_model")
COLLECTION_MISSING = (ValueError, chromadb.errors.NotFoundError)


class VectorStoreError(RuntimeError):
    """Raised when the collection cannot be opened or written consistently."""


@dataclass(frozen=True)
class RetrievedChunk:
    """One scored chunk as returned by search; score is cosine similarity in [0, 1]."""

    chunk_id: str
    text: str
    score: float
    metadata: dict


def sanitize_metadata(metadata: dict) -> dict:
    """Coerce a metadata dict to the scalar types Chroma accepts.

    Chroma rejects ``None``, lists, and dicts. Absent strings become ``""`` and
    absent integers become ``0`` so that every chunk carries the same key set.
    """
    clean: dict = {}
    for key, value in metadata.items():
        if isinstance(value, bool) or isinstance(value, (int, float, str)):
            clean[str(key)] = value
        elif value is None:
            clean[str(key)] = ""
        else:
            clean[str(key)] = json.dumps(value, ensure_ascii=False, default=str)
    return clean


def validate_metadata(metadata: dict) -> None:
    """Raise when a metadata dict holds a value Chroma cannot store."""
    offenders = {
        key: value
        for key, value in metadata.items()
        if not isinstance(value, SCALAR_TYPES)
    }
    if offenders:
        raise VectorStoreError(
            f"metadata values must be str/int/float/bool; offending keys: {sorted(offenders)}"
        )


@lru_cache(maxsize=4)
def _client_for(directory: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=directory)


def get_client() -> chromadb.PersistentClient:
    """Return the persistent Chroma client for ``config.CHROMA_DIR``.

    The client is cached per directory so that tests which point ``CHROMA_DIR`` at a
    temporary path get their own client instead of the previously cached one.
    """
    directory = Path(config.CHROMA_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    return _client_for(str(directory))


@lru_cache(maxsize=4)
def _collection_for(directory: str, name: str, reset: bool) -> "Collection":
    client = _client_for(directory)
    metadata = {
        "hnsw:space": config.HNSW_SPACE,
        "corpus_version": config.COLLECTION_NAME,
        "embedding_model": config.EMBEDDING_MODEL,
    }
    if reset:
        try:
            client.delete_collection(name)
        except COLLECTION_MISSING:
            pass
    return client.get_or_create_collection(name=name, metadata=metadata)


def get_or_create_collection(reset: bool = False) -> "Collection":
    """Open the corpus collection, creating it on first use.

    An existing collection keeps its original HNSW space: Chroma cannot change it in
    place, so a mismatch is reported as a warning that requires ``--reset``.
    """
    log = logger.configure_logging()
    collection = _collection_for(str(Path(config.CHROMA_DIR)), config.COLLECTION_NAME, reset)
    stored = collection.metadata or {}
    space = stored.get("hnsw:space")
    if space and space != config.HNSW_SPACE:
        log.warning(
            "collection %s uses hnsw:space=%s but config asks for %s; "
            "re-run ingest with --reset to rebuild it",
            config.COLLECTION_NAME,
            space,
            config.HNSW_SPACE,
        )
    return collection


def reset_collection() -> None:
    """Delete the collection so the next ingest rebuilds it from scratch."""
    client = get_client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except COLLECTION_MISSING:
        return
    _collection_for.cache_clear()
    _client_for.cache_clear()


def upsert_chunks(chunks: list["Chunk"], vectors: np.ndarray) -> int:
    """Write chunks and their vectors by ``chunk_id``; returns the number upserted.

    Upserting by a stable id makes repeated ingests idempotent (FR-05).
    """
    if not chunks:
        return 0
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise VectorStoreError(
            f"expected {len(chunks)} vectors, got shape {matrix.shape}"
        )
    if matrix.shape[1] != config.EMBEDDING_DIM:
        raise VectorStoreError(
            f"expected {config.EMBEDDING_DIM}-dim vectors, got {matrix.shape[1]}"
        )

    metadatas = [sanitize_metadata(chunk.metadata) for chunk in chunks]
    for metadata in metadatas:
        validate_metadata(metadata)

    collection = get_or_create_collection()
    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        embeddings=matrix.tolist(),
        metadatas=metadatas,
    )
    return len(chunks)


def prune_stale(keep_ids: set[str]) -> int:
    """Delete stored chunks whose ids are absent from ``keep_ids``; returns the count removed."""
    collection = get_or_create_collection()
    stored = collection.get(include=[])["ids"] or []
    stale = [chunk_id for chunk_id in stored if chunk_id not in keep_ids]
    if stale:
        collection.delete(ids=stale)
    return len(stale)


def search(
    query_vec: list[float] | np.ndarray,
    k: int = config.TOP_K,
    where: dict | None = None,
) -> list[RetrievedChunk]:
    """Return up to ``k`` chunks ranked by descending cosine similarity.

    Chroma's cosine distance is converted to similarity with ``score = 1 - distance``.
    """
    collection = get_or_create_collection()
    if collection.count() == 0:
        return []
    vector = np.asarray(query_vec, dtype=np.float32).tolist()
    response = collection.query(
        query_embeddings=[vector],
        n_results=max(1, min(int(k), collection.count())),
        where=where or None,
        include=["documents", "metadatas", "distances"],
    )
    ids = (response.get("ids") or [[]])[0]
    documents = (response.get("documents") or [[]])[0]
    metadatas = (response.get("metadatas") or [[]])[0]
    distances = (response.get("distances") or [[]])[0]

    results: list[RetrievedChunk] = []
    for index, chunk_id in enumerate(ids):
        results.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=documents[index] if index < len(documents) else "",
                score=1.0 - float(distances[index]),
                metadata=dict(metadatas[index] or {}) if index < len(metadatas) else {},
            )
        )
    return sorted(results, key=lambda item: item.score, reverse=True)


def collection_stats() -> dict:
    """Describe the collection: identity, size, index space, and chunk-count bounds."""
    collection = get_or_create_collection()
    stored = collection.get(include=[])["ids"] or []
    ordinals = []
    for chunk_id in stored:
        _, _, ordinal = str(chunk_id).rpartition(":")
        if ordinal.isdigit():
            ordinals.append(int(ordinal))
    metadata = dict(collection.metadata or {})
    return {
        "collection_name": config.COLLECTION_NAME,
        "persist_directory": str(config.CHROMA_DIR),
        "count": collection.count(),
        "hnsw_space": metadata.get("hnsw:space", ""),
        "corpus_version": metadata.get("corpus_version", ""),
        "embedding_model": metadata.get("embedding_model", ""),
        "configured_hnsw_space": config.HNSW_SPACE,
        "max_ordinal": max(ordinals) if ordinals else 0,
        "expected_metadata_keys": list(COLLECTION_METADATA_KEYS),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: report collection statistics."""
    parser = argparse.ArgumentParser(description="Inspect the Chroma collection (stage 5).")
    parser.add_argument("--stats", action="store_true", help="print collection statistics")
    args = parser.parse_args(argv)

    try:
        stats = collection_stats()
    except Exception as error:
        print(f"vector store unavailable: {error}")
        return 1

    print(f"collection   {stats['collection_name']}")
    print(f"persist      {stats['persist_directory']}")
    print(f"chunks       {stats['count']}")
    print(f"hnsw:space   {stats['hnsw_space'] or '(unset)'} (config: {stats['configured_hnsw_space']})")
    print(f"version      {stats['corpus_version']}")
    print(f"embedder     {stats['embedding_model']}")
    print(f"max ordinal  {stats['max_ordinal']}")
    print(f"inspected at {datetime.now(tz=timezone.utc).isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

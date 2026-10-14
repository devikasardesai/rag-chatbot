"""Stage 4 embedding: a process-local all-MiniLM-L6-v2 encoder producing 384-dim unit vectors."""
from __future__ import annotations

import argparse
import time
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src import config


class EmbeddingError(RuntimeError):
    """Raised when the encoder cannot be loaded or returns an unexpected shape."""


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Return the cached encoder singleton; the first call loads the ~90 MB model.

    Reloading per call costs about a second, which would blow the latency budget, so the
    model is kept in one process-local instance. `HF_HOME`, when set, is used as the
    download/cache directory.
    """
    options: dict = {}
    if config.HF_HOME:
        options["cache_folder"] = config.HF_HOME
    try:
        return SentenceTransformer(config.EMBEDDING_MODEL, **options)
    except Exception as error:
        raise EmbeddingError(
            f"could not load {config.EMBEDDING_MODEL}: {error}. "
            "The first call downloads ~90 MB; check network access or set HF_HOME."
        ) from error


def embed_texts(
    texts: list[str],
    batch_size: int = config.EMBED_BATCH_SIZE,
) -> np.ndarray:
    """Encode texts into an ``(n, 384)`` float32 array of L2-normalized vectors.

    Normalization is mandatory: it is what makes Chroma's cosine distance directly
    interpretable as ``1 - similarity`` in stage 6. An empty input returns a
    ``(0, 384)`` array without loading the model.
    """
    items = [str(text) for text in texts]
    if not items:
        return np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)

    encoded = get_model().encode(
        items,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(encoded, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape != (len(items), config.EMBEDDING_DIM):
        raise EmbeddingError(
            f"{config.EMBEDDING_MODEL} returned shape {vectors.shape}, "
            f"expected ({len(items)}, {config.EMBEDDING_DIM})"
        )
    return vectors


def embed_query(text: str) -> list[float]:
    """Encode one query string into a 384-float list."""
    return embed_texts([text])[0].tolist()


def row_norms(vectors: np.ndarray) -> np.ndarray:
    """Return the L2 norm of every row, for diagnostics and tests."""
    return np.linalg.norm(vectors, axis=1)


def model_info() -> dict:
    """Describe the loaded encoder: identity, dimensionality, and load cost."""
    start = time.perf_counter()
    model = get_model()
    load_seconds = time.perf_counter() - start
    probe = embed_texts([config.PROMPT_SYSTEM[:200]])
    return {
        "model": config.EMBEDDING_MODEL,
        "embedding_dim": config.EMBEDDING_DIM,
        "max_sequence_tokens": int(getattr(model, "max_seq_length", 0) or 0),
        "batch_size": config.EMBED_BATCH_SIZE,
        "hf_home": config.HF_HOME or "default",
        "load_seconds": round(load_seconds, 3),
        "probe_dim": int(probe.shape[1]),
        "probe_norm": round(float(row_norms(probe)[0]), 6),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: report the encoder configuration and run a shape/norm self-check."""
    parser = argparse.ArgumentParser(description="Embed texts (stage 4).")
    parser.add_argument("--text", action="append", default=[], help="text to embed (repeatable)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify dimensionality and unit norms using the prompt text as the probe",
    )
    args = parser.parse_args(argv)

    try:
        info = model_info()
        print(f"model      {info['model']}")
        print(f"dim        {info['embedding_dim']} (probe {info['probe_dim']})")
        print(f"max tokens {info['max_sequence_tokens']}")
        print(f"batch size {info['batch_size']}")
        print(f"cache      {info['hf_home']}")
        print(f"probe norm {info['probe_norm']} (unit norm expected)")

        texts = args.text or [config.PROMPT_SYSTEM[:200]]
        vectors = embed_texts(texts)
        print(f"\nembedded {len(texts)} text(s) -> {vectors.shape} {vectors.dtype}")

        if args.check:
            if vectors.shape[1] != config.EMBEDDING_DIM:
                print(f"FAIL: expected dim {config.EMBEDDING_DIM}")
                return 1
            worst = float(np.abs(row_norms(vectors) - 1.0).max())
            print(f"max |norm - 1| {worst:.2e}")
            if worst > config.NORM_TOLERANCE:
                print("FAIL: vectors are not unit-normalized")
                return 1
            print("PASS")
    except EmbeddingError as error:
        print(f"embedding failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

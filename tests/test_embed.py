"""Tests for stage 4 embedding: dimensionality, normalization, and the model singleton."""
from __future__ import annotations

import numpy as np
import pytest

from src import config, embed

pytestmark = pytest.mark.slow

PROBE_TEXTS = [
    "HDFC Small Cap Fund - Direct Growth | Section: Exit load",
    "The exit load is 1.0% for a holding period of 0-12 months.",
    "The expense ratio of the scheme is 1.19% of NAV.",
]


def test_embed_texts_shape_and_dtype() -> None:
    vectors = embed.embed_texts(PROBE_TEXTS)
    assert vectors.shape == (len(PROBE_TEXTS), config.EMBEDDING_DIM)
    assert vectors.dtype == np.float32


def test_embed_query_returns_unit_vector() -> None:
    vector = embed.embed_query("exit load of HDFC Large Cap Fund")
    assert isinstance(vector, list)
    assert len(vector) == config.EMBEDDING_DIM
    norm = float(np.linalg.norm(np.asarray(vector, dtype=np.float32)))
    assert norm == pytest.approx(1.0, abs=config.NORM_TOLERANCE)


def test_rows_are_normalized() -> None:
    vectors = embed.embed_texts(PROBE_TEXTS)
    for norm in embed.row_norms(vectors):
        assert norm == pytest.approx(1.0, abs=config.NORM_TOLERANCE)


def test_empty_input_returns_zero_row_array() -> None:
    vectors = embed.embed_texts([])
    assert vectors.shape == (0, config.EMBEDDING_DIM)
    assert vectors.dtype == np.float32


def test_identical_text_yields_identical_vector() -> None:
    first, second = embed.embed_query("minimum SIP amount"), embed.embed_query("minimum SIP amount")
    assert first == second


def test_batch_encoding_matches_single_encoding() -> None:
    batched = embed.embed_texts(PROBE_TEXTS)
    single = np.vstack([np.asarray(embed.embed_query(text), dtype=np.float32) for text in PROBE_TEXTS])
    np.testing.assert_allclose(batched, single, atol=1e-5)


def test_distinct_texts_are_not_collapsed() -> None:
    vectors = embed.embed_texts(["exit load", "benchmark index and riskometer category"])
    similarity = float(np.dot(vectors[0], vectors[1]))
    assert similarity < 0.99


def test_get_model_is_a_singleton() -> None:
    assert embed.get_model() is embed.get_model()


def test_model_reports_configured_dimension() -> None:
    info = embed.model_info()
    assert info["embedding_dim"] == config.EMBEDDING_DIM
    assert info["probe_dim"] == config.EMBEDDING_DIM
    assert info["probe_norm"] == pytest.approx(1.0, abs=config.NORM_TOLERANCE)


def test_cli_check_passes() -> None:
    assert embed.main(["--check", "--text", "expense ratio"]) == 0


def test_non_string_input_is_coerced() -> None:
    vectors = embed.embed_texts(["SIP of 500 per month", 500])
    assert vectors.shape == (2, config.EMBEDDING_DIM)

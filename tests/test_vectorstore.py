"""Tests for stage 5 storage: metadata hygiene, idempotent upserts, pruning, and search."""
from __future__ import annotations

import numpy as np
import pytest

from src import config, embed, vectorstore
from src.chunking import Chunk
from src.ingest import doc_id_for
from src.vectorstore import RetrievedChunk, VectorStoreError

ALLOWED_SCALARS = (str, int, float, bool)


def _fake_vector(seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    vector = generator.standard_normal(config.EMBEDDING_DIM).astype(np.float32)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _fake_chunk(doc_id: str = "a1b2c3d4e5f6", ordinal: int = 0, **metadata) -> Chunk:
    base = {
        "source_url": config.CORPUS[0].url,
        "scheme_name": config.CORPUS[0].scheme_name,
        "category": config.CORPUS[0].category,
        "plan": config.CORPUS[0].plan,
        "section": "Exit load",
        "section_path": "HDFC Large Cap Fund > Exit load",
        "page_title": "HDFC Large Cap Fund Direct Growth",
        "fetched_at": "2026-09-27",
        "chunk_ordinal": ordinal,
    }
    base.update(metadata)
    return Chunk(
        chunk_id=f"{doc_id}:{ordinal:03d}",
        doc_id=doc_id,
        ordinal=ordinal,
        text=f"Scheme: {base['scheme_name']} | Section: Exit load | Source: {base['source_url']}\n"
        f"The exit load is 1.0% for a holding period of 0-12 months. Row {ordinal}.",
        token_count=40,
        metadata=base,
    )


class TestMetadataSanitisation:
    def test_sanitize_coerces_none_to_empty_string(self):
        assert vectorstore.sanitize_metadata({"section": None}) == {"section": ""}

    def test_sanitize_keeps_scalars_unchanged(self):
        metadata = {"a": "x", "b": 3, "c": 1.5, "d": True}
        assert vectorstore.sanitize_metadata(metadata) == metadata

    def test_sanitize_json_encodes_containers(self):
        clean = vectorstore.sanitize_metadata({"tags": ["x", "y"], "nested": {"k": 1}})
        assert clean["tags"] == '["x", "y"]'
        assert clean["nested"] == '{"k": 1}'

    def test_sanitize_output_is_always_chroma_safe(self):
        clean = vectorstore.sanitize_metadata({"none": None, "list": [1], "int": 0})
        for value in clean.values():
            assert isinstance(value, ALLOWED_SCALARS)

    def test_validate_metadata_rejects_none(self):
        with pytest.raises(VectorStoreError, match="offending keys"):
            vectorstore.validate_metadata({"section": None})

    def test_validate_metadata_rejects_containers(self):
        with pytest.raises(VectorStoreError, match="offending keys"):
            vectorstore.validate_metadata({"tags": ["a"]})

    def test_chunking_metadata_never_contains_none(self, sample_markdown):
        from src import chunking
        from src.ingest import PageDocument

        document = PageDocument(
            doc_id=doc_id_for(config.CORPUS[3].url),
            url=config.CORPUS[3].url,
            scheme_name=config.CORPUS[3].scheme_name,
            category=config.CORPUS[3].category,
            plan=config.CORPUS[3].plan,
            page_title="HDFC Small Cap Fund",
            markdown=sample_markdown,
            fetched_at="2026-09-27",
            content_sha256="0" * 64,
            raw_path="",
            processed_path="",
        )
        chunks = chunking.split_text(sample_markdown, document)
        assert chunks
        for chunk in chunks:
            assert None not in chunk.metadata.values()
            vectorstore.validate_metadata(chunk.metadata)

    def test_chroma_rejects_raw_none_metadata(self, tmp_data_dir):
        collection = vectorstore.get_or_create_collection()
        with pytest.raises(Exception):
            collection.add(
                ids=["probe:000"],
                documents=["probe"],
                embeddings=[_fake_vector(1).tolist()],
                metadatas=[{"section": None}],
            )


class TestCollectionLifecycle:
    def test_collection_is_created_with_index_metadata(self, tmp_data_dir):
        collection = vectorstore.get_or_create_collection()
        metadata = dict(collection.metadata or {})
        assert metadata["hnsw:space"] == config.HNSW_SPACE
        assert metadata["corpus_version"] == config.COLLECTION_NAME
        assert metadata["embedding_model"] == config.EMBEDDING_MODEL

    def test_same_collection_is_returned_for_one_directory(self, tmp_data_dir):
        assert (
            vectorstore.get_or_create_collection().name
            == vectorstore.get_or_create_collection().name
        )

    def test_reset_empties_the_collection(self, tmp_data_dir):
        vectorstore.upsert_chunks([_fake_chunk()], np.vstack([_fake_vector(0)]))
        assert vectorstore.get_or_create_collection().count() == 1
        vectorstore.reset_collection()
        assert vectorstore.get_or_create_collection().count() == 0

    def test_reset_is_safe_when_collection_missing(self, tmp_data_dir):
        vectorstore.reset_collection()
        vectorstore.reset_collection()
        assert vectorstore.get_or_create_collection().count() == 0


class TestUpsert:
    def test_upsert_returns_count_and_persists(self, tmp_data_dir):
        chunks = [_fake_chunk(ordinal=index) for index in range(3)]
        vectors = np.vstack([_fake_vector(index) for index in range(3)])
        assert vectorstore.upsert_chunks(chunks, vectors) == 3
        assert vectorstore.get_or_create_collection().count() == 3

    def test_upsert_is_idempotent(self, tmp_data_dir):
        chunks = [_fake_chunk(ordinal=index) for index in range(3)]
        vectors = np.vstack([_fake_vector(index) for index in range(3)])
        vectorstore.upsert_chunks(chunks, vectors)
        vectorstore.upsert_chunks(chunks, vectors)
        assert vectorstore.get_or_create_collection().count() == 3

    def test_upsert_updates_document_for_existing_id(self, tmp_data_dir):
        vectorstore.upsert_chunks([_fake_chunk()], np.vstack([_fake_vector(0)]))
        changed = _fake_chunk()
        changed = Chunk(
            chunk_id=changed.chunk_id,
            doc_id=changed.doc_id,
            ordinal=changed.ordinal,
            text="replaced text",
            token_count=2,
            metadata=changed.metadata,
        )
        vectorstore.upsert_chunks([changed], np.vstack([_fake_vector(0)]))
        stored = vectorstore.get_or_create_collection().get(ids=[changed.chunk_id])
        assert stored["documents"] == ["replaced text"]

    def test_upsert_stores_full_metadata(self, tmp_data_dir):
        vectorstore.upsert_chunks([_fake_chunk()], np.vstack([_fake_vector(0)]))
        stored = vectorstore.get_or_create_collection().get(include=["metadatas"])
        metadata = stored["metadatas"][0]
        assert metadata["source_url"] == config.CORPUS[0].url
        assert metadata["scheme_name"] == config.CORPUS[0].scheme_name
        assert metadata["section"] == "Exit load"
        assert metadata["fetched_at"] == "2026-09-27"
        assert metadata["chunk_ordinal"] == 0
        for value in metadata.values():
            assert isinstance(value, ALLOWED_SCALARS)

    def test_upsert_strips_none_metadata(self, tmp_data_dir):
        vectorstore.upsert_chunks(
            [_fake_chunk(page_title=None)], np.vstack([_fake_vector(0)])
        )
        stored = vectorstore.get_or_create_collection().get(include=["metadatas"])
        assert stored["metadatas"][0]["page_title"] == ""

    def test_upsert_of_nothing_is_a_noop(self, tmp_data_dir):
        assert vectorstore.upsert_chunks([], np.zeros((0, config.EMBEDDING_DIM), np.float32)) == 0

    def test_upsert_rejects_vector_count_mismatch(self, tmp_data_dir):
        with pytest.raises(VectorStoreError, match="expected 2 vectors"):
            vectorstore.upsert_chunks(
                [_fake_chunk(ordinal=0), _fake_chunk(ordinal=1)],
                np.vstack([_fake_vector(0)]),
            )

    def test_upsert_rejects_wrong_dimension(self, tmp_data_dir):
        with pytest.raises(VectorStoreError, match="384-dim"):
            vectorstore.upsert_chunks([_fake_chunk()], np.zeros((1, 8), dtype=np.float32))


class TestPruneStale:
    def test_prune_removes_ids_not_kept(self, tmp_data_dir):
        chunks = [_fake_chunk(ordinal=index) for index in range(3)]
        vectors = np.vstack([_fake_vector(index) for index in range(3)])
        vectorstore.upsert_chunks(chunks, vectors)
        removed = vectorstore.prune_stale({chunks[0].chunk_id, chunks[1].chunk_id})
        assert removed == 1
        remaining = set(vectorstore.get_or_create_collection().get(include=[])["ids"])
        assert remaining == {chunks[0].chunk_id, chunks[1].chunk_id}

    def test_prune_keeps_everything_when_nothing_is_stale(self, tmp_data_dir):
        chunks = [_fake_chunk(ordinal=index) for index in range(2)]
        vectorstore.upsert_chunks(chunks, np.vstack([_fake_vector(0), _fake_vector(1)]))
        assert vectorstore.prune_stale({chunk.chunk_id for chunk in chunks}) == 0

    def test_prune_is_idempotent(self, tmp_data_dir):
        chunks = [_fake_chunk(ordinal=index) for index in range(2)]
        vectorstore.upsert_chunks(chunks, np.vstack([_fake_vector(0), _fake_vector(1)]))
        keep = {chunks[0].chunk_id}
        assert vectorstore.prune_stale(keep) == 1
        assert vectorstore.prune_stale(keep) == 0

    def test_prune_can_empty_the_collection(self, tmp_data_dir):
        vectorstore.upsert_chunks([_fake_chunk()], np.vstack([_fake_vector(0)]))
        assert vectorstore.prune_stale(set()) == 1
        assert vectorstore.get_or_create_collection().count() == 0


class TestSearch:
    def test_search_returns_descending_similarity(self, tmp_data_dir):
        chunks = [_fake_chunk(ordinal=index) for index in range(3)]
        vectors = np.vstack([_fake_vector(index) for index in range(3)])
        vectorstore.upsert_chunks(chunks, vectors)
        results = vectorstore.search(_fake_vector(1), k=3)
        assert len(results) == 3
        assert all(isinstance(item, RetrievedChunk) for item in results)
        scores = [item.score for item in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].chunk_id == chunks[1].chunk_id

    def test_score_is_one_minus_distance(self, tmp_data_dir):
        vectorstore.upsert_chunks([_fake_chunk()], np.vstack([_fake_vector(7)]))
        collection = vectorstore.get_or_create_collection()
        raw = collection.query(
            query_embeddings=[_fake_vector(7).tolist()], n_results=1, include=["distances"]
        )
        expected = 1.0 - raw["distances"][0][0]
        assert vectorstore.search(_fake_vector(7), k=1)[0].score == pytest.approx(expected)

    def test_score_of_identical_vector_is_one(self, tmp_data_dir):
        vectorstore.upsert_chunks([_fake_chunk()], np.vstack([_fake_vector(11)]))
        assert vectorstore.search(_fake_vector(11), k=1)[0].score == pytest.approx(1.0, abs=1e-4)

    def test_search_returns_metadata_and_text(self, tmp_data_dir):
        chunk = _fake_chunk()
        vectorstore.upsert_chunks([chunk], np.vstack([_fake_vector(0)]))
        result = vectorstore.search(_fake_vector(0), k=1)[0]
        assert result.text == chunk.text
        assert result.metadata["scheme_name"] == config.CORPUS[0].scheme_name

    def test_search_respects_k(self, tmp_data_dir):
        chunks = [_fake_chunk(ordinal=index) for index in range(5)]
        vectors = np.vstack([_fake_vector(index) for index in range(5)])
        vectorstore.upsert_chunks(chunks, vectors)
        assert len(vectorstore.search(_fake_vector(0), k=2)) == 2

    def test_search_applies_where_filter(self, tmp_data_dir):
        chunks = [
            _fake_chunk(ordinal=0, category="Large Cap"),
            _fake_chunk(ordinal=1, category="ELSS"),
        ]
        vectors = np.vstack([_fake_vector(0), _fake_vector(1)])
        vectorstore.upsert_chunks(chunks, vectors)
        results = vectorstore.search(_fake_vector(1), k=2, where={"category": "ELSS"})
        assert [item.chunk_id for item in results] == [chunks[1].chunk_id]

    def test_search_on_empty_collection_returns_nothing(self, tmp_data_dir):
        assert vectorstore.search(_fake_vector(0), k=5) == []


class TestStats:
    def test_stats_reports_count_and_metadata(self, tmp_data_dir):
        chunks = [_fake_chunk(ordinal=index) for index in range(4)]
        vectorstore.upsert_chunks(chunks, np.vstack([_fake_vector(i) for i in range(4)]))
        stats = vectorstore.collection_stats()
        assert stats["collection_name"] == config.COLLECTION_NAME
        assert stats["count"] == 4
        assert stats["hnsw_space"] == config.HNSW_SPACE
        assert stats["embedding_model"] == config.EMBEDDING_MODEL
        assert stats["max_ordinal"] == 3

    def test_stats_on_empty_collection(self, tmp_data_dir):
        stats = vectorstore.collection_stats()
        assert stats["count"] == 0
        assert stats["max_ordinal"] == 0

    def test_cli_stats_exits_zero(self, tmp_data_dir, capsys):
        assert vectorstore.main(["--stats"]) == 0
        out = capsys.readouterr().out
        assert config.COLLECTION_NAME in out
        assert "chunks" in out


class TestFreshProcessReopen:
    def test_client_disables_anonymized_telemetry(self, tmp_data_dir):
        vectorstore._client_for.cache_clear()
        client = vectorstore.get_client()
        assert client.get_settings().anonymized_telemetry is False

    def test_stored_vectors_survive_a_new_client(self, tmp_data_dir):
        chunks = [_fake_chunk(ordinal=index) for index in range(3)]
        vectors = np.vstack([_fake_vector(index) for index in range(3)])
        vectorstore.upsert_chunks(chunks, vectors)
        vectorstore._client_for.cache_clear()
        vectorstore._collection_for.cache_clear()
        reopened = vectorstore.get_or_create_collection()
        assert reopened.count() == 3
        assert set(reopened.get(include=[])["ids"]) == {chunk.chunk_id for chunk in chunks}

    def test_real_query_embeds_and_finds_its_chunk(self, tmp_data_dir):
        text = "The exit load is 1.0% for a holding period of 0-12 months."
        chunk = Chunk(
            chunk_id="cafe0000cafe:000",
            doc_id="cafe0000cafe",
            ordinal=0,
            text=text,
            token_count=20,
            metadata=_fake_chunk().metadata,
        )
        vector = embed.embed_texts([text])
        vectorstore.upsert_chunks([chunk], vector)
        best = vectorstore.search(embed.embed_query(text), k=1)[0]
        assert best.chunk_id == chunk.chunk_id
        assert best.score > 0.9

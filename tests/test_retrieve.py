"""Tests for stages 6-9 retrieval: normalization, thresholding, allowlist, MMR, context."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src import config, retrieve, vectorstore
from src.chunking import Chunk
from src.ingest import doc_id_for
from src.vectorstore import RetrievedChunk

OUT_OF_SCOPE_URL = "https://evil.example.com/mutual-funds/hdfc-large-cap-fund-direct-growth"


def _fake_vector(seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    vector = generator.standard_normal(config.EMBEDDING_DIM).astype(np.float32)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _hit(
    chunk_id: str,
    text: str = "The exit load is 1.0% for a holding period of 0-12 months.",
    score: float = 0.8,
    scheme: str = config.CORPUS[0].scheme_name,
    section: str = "Exit load",
    source_url: str | None = None,
) -> RetrievedChunk:
    url = config.CORPUS[0].url if source_url is None else source_url
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        metadata={
            "source_url": url,
            "scheme_name": scheme,
            "category": config.CORPUS[0].category,
            "plan": config.CORPUS[0].plan,
            "section": section,
            "section_path": f"{scheme} > {section}",
        },
    )


def _stub_query(monkeypatch, vector: np.ndarray | None = None) -> None:
    """Replace the encoder so retrieval never loads the real model."""
    resolved = _fake_vector(0) if vector is None else vector
    monkeypatch.setattr(
        retrieve.embed, "embed_query", lambda text: resolved.astype(np.float32).tolist()
    )


def _chunk(ordinal: int, scheme_index: int = 0, section: str = "Exit load") -> Chunk:
    source = config.CORPUS[scheme_index]
    return Chunk(
        chunk_id=f"{doc_id_for(source.url)}:{ordinal:03d}",
        doc_id=doc_id_for(source.url),
        ordinal=ordinal,
        text=f"Scheme: {source.scheme_name} | Section: {section} | Source: {source.url}\n"
        f"The exit load for {source.scheme_name} row {ordinal} is 1.0%.",
        token_count=40,
        metadata={
            "source_url": source.url,
            "scheme_name": source.scheme_name,
            "category": source.category,
            "plan": source.plan,
            "section": section,
            "section_path": f"{source.scheme_name} > {section}",
            "page_title": source.scheme_name,
            "fetched_at": "2026-09-27",
            "chunk_ordinal": ordinal,
        },
    )


def _out_of_scope_chunk() -> Chunk:
    """A searchable chunk whose URL is outside the corpus allowlist (FR-16 defence)."""
    return Chunk(
        chunk_id="bad000000000:000",
        doc_id="bad000000000",
        ordinal=0,
        text=f"Scheme: Impostor Fund | Section: Exit load | Source: {OUT_OF_SCOPE_URL}\n"
        f"The exit load is 1.0% for a holding period of 0-12 months.",
        token_count=30,
        metadata={
            "source_url": OUT_OF_SCOPE_URL,
            "scheme_name": "Impostor Fund",
            "category": "Large Cap",
            "plan": "Direct Growth",
            "section": "Exit load",
            "section_path": "Impostor Fund > Exit load",
            "page_title": "Impostor Fund",
            "fetched_at": "2026-09-27",
            "chunk_ordinal": 0,
        },
    )


class TestNormalizeQuery:
    def test_lowercases_and_collapses_whitespace(self):
        assert retrieve.normalize_query("  HDFC   LARGE   Cap  ") == "hdfc large cap"

    def test_drops_what_is_the_filler(self):
        assert retrieve.normalize_query("What is the exit load?") == "exit load?"

    def test_drops_tell_me_about_filler(self):
        assert retrieve.normalize_query("Tell me about HDFC ELSS") == "hdfc elss"

    def test_keeps_question_mark(self):
        assert retrieve.normalize_query("Is the fund safe?").endswith("?")

    def test_keeps_percent_sign(self):
        assert "1.5%" in retrieve.normalize_query("What is the 1.5% exit load?")

    def test_strips_other_punctuation(self):
        assert retrieve.normalize_query("Exit load, fees & risk!") == "exit load fees risk"

    def test_is_idempotent(self):
        once = retrieve.normalize_query("What is the expense ratio of HDFC Small Cap Fund?")
        assert retrieve.normalize_query(once) == once

    def test_empty_question_becomes_empty_string(self):
        assert retrieve.normalize_query("   ") == ""

    def test_filler_only_question_becomes_empty_string(self):
        assert retrieve.normalize_query("what is the") == ""


class TestAllowlist:
    def test_corpus_url_is_allowlisted(self):
        assert retrieve.is_allowlisted(config.CORPUS[0].url)

    def test_trailing_slash_and_host_case_do_not_bypass_the_allowlist(self):
        assert retrieve.is_allowlisted("https://GROWW.in" + config.CORPUS[0].url[len("https://groww.in"):] + "/")

    def test_foreign_url_is_rejected(self):
        assert not retrieve.is_allowlisted(OUT_OF_SCOPE_URL)

    def test_prefix_of_an_allowed_url_is_rejected(self):
        assert not retrieve.is_allowlisted("https://groww.in/mutual-funds")

    def test_empty_url_is_rejected(self):
        assert not retrieve.is_allowlisted("")

    def test_assert_allowlisted_passes_for_corpus_urls(self):
        retrieve.assert_allowlisted([source.url for source in config.CORPUS])

    def test_assert_allowlisted_raises_on_foreign_url(self):
        with pytest.raises(retrieve.OutOfScopeError) as excinfo:
            retrieve.assert_allowlisted([config.CORPUS[0].url, OUT_OF_SCOPE_URL])
        assert OUT_OF_SCOPE_URL in str(excinfo.value)

    def test_allowed_host_in_a_query_path_is_not_enough(self):
        assert not retrieve.is_allowlisted(
            f"https://evil.example.com/?next={config.CORPUS[0].url}"
        )


class TestDedupe:
    def test_duplicate_chunk_ids_are_dropped(self):
        hits = [
            _hit("a:001", text="Exit load is 1.0%."),
            _hit("a:001", text="Exit load is 1.0%."),
            _hit("a:002", text="Benchmark is Nifty Smallcap 250 TRI.", section="Benchmark"),
        ]
        assert [c.chunk_id for c in retrieve.diversify(hits)] == ["a:001", "a:002"]

    def test_near_identical_text_is_dropped(self):
        hits = [
            _hit("a:001", text="Exit load is 1.0% for 0-12 months."),
            _hit("a:002", text="Exit   load is 1.0% for 0-12 months."),
            _hit("a:003", text="The benchmark is Nifty Smallcap 250 TRI.", section="Benchmark"),
        ]
        kept = [c.chunk_id for c in retrieve.diversify(hits)]
        assert "a:002" not in kept
        assert kept == ["a:001", "a:003"]

    def test_per_section_cap_limits_repeat_sections(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_CHUNKS_PER_SECTION", 2)
        hits = [
            _hit(f"a:00{index}", text=f"row {index} unique body text", section="Exit load")
            for index in range(5)
        ]
        kept = retrieve.diversify(hits, per_section_cap=2)
        assert len(kept) == 2

    def test_cap_is_scoped_per_scheme_and_section(self):
        hits = [
            _hit("a:001", text="alpha body", section="Exit load", scheme="Fund A"),
            _hit("a:002", text="beta body", section="Exit load", scheme="Fund A"),
            _hit("a:003", text="gamma body", section="Exit load", scheme="Fund B"),
        ]
        assert len(retrieve.diversify(hits, per_section_cap=2)) == 3


class TestMmr:
    def test_first_pick_is_the_highest_scoring_chunk(self):
        hits = [
            _hit("a:001", score=0.5, text="Exit load is 1.0%."),
            _hit("a:002", score=0.9, text="Benchmark is Nifty Smallcap 250 TRI.", section="Benchmark"),
        ]
        assert retrieve.diversify(hits)[0].chunk_id == "a:002"

    def test_mmr_trades_relevance_for_novelty(self):
        """A less relevant but non-redundant chunk should beat a near-duplicate."""
        hits = [
            _hit("a:001", score=0.90, text="Exit load is 1.0% for 0-12 months.", section="Exit load"),
            _hit("a:002", score=0.85, text="Exit load is 1.0% for 0-12 months!", section="Exit load"),
            _hit("a:003", score=0.60, text="Benchmark is Nifty Smallcap 250 TRI.", section="Benchmark"),
        ]

        def similarity(left, right):
            table = {
                frozenset({"Exit load is 1.0% for 0-12 months.", "Exit load is 1.0% for 0-12 months!"}): 0.99,
                frozenset({"Benchmark is Nifty Smallcap 250 TRI.", "Exit load is 1.0% for 0-12 months."}): 0.0,
                frozenset({"Benchmark is Nifty Smallcap 250 TRI.", "Exit load is 1.0% for 0-12 months!"}): 0.0,
            }
            return np.array(
                [[table.get(frozenset({l, r}), 0.0) for r in right] for l in left],
                dtype=np.float32,
            )

        ordered = retrieve.diversify(hits, k=2, similarity=similarity)
        assert [c.chunk_id for c in ordered] == ["a:001", "a:003"]

    def test_mmr_keeps_the_relevant_chunk_when_novelty_penalty_is_small(self):
        hits = [
            _hit("a:001", score=0.90, text="Exit load is 1.0% for 0-12 months.", section="Exit load"),
            _hit("a:002", score=0.85, text="Exit load is 1.0% for 0-12 months!", section="Exit load"),
            _hit("a:003", score=0.40, text="Benchmark is Nifty Smallcap 250 TRI.", section="Benchmark"),
        ]

        def similarity(left, right):
            table = {
                frozenset({"Exit load is 1.0% for 0-12 months.", "Exit load is 1.0% for 0-12 months!"}): 0.99,
                frozenset({"Benchmark is Nifty Smallcap 250 TRI.", "Exit load is 1.0% for 0-12 months."}): 0.05,
                frozenset({"Benchmark is Nifty Smallcap 250 TRI.", "Exit load is 1.0% for 0-12 months!"}): 0.05,
            }
            return np.array(
                [[table.get(frozenset({l, r}), 0.0) for r in right] for l in left],
                dtype=np.float32,
            )

        assert retrieve.diversify(hits, k=2, similarity=similarity)[1].chunk_id == "a:002"

    def test_mmr_returns_at_most_k(self):
        hits = [_hit(f"a:00{index}", text=f"unique body {index}", section=f"Section {index}")
                for index in range(6)]
        assert len(retrieve.diversify(hits, k=3)) == 3

    def test_single_candidate_is_returned_unchanged(self):
        hits = [_hit("a:001")]
        assert [c.chunk_id for c in retrieve.diversify(hits)] == ["a:001"]

    def test_empty_input_returns_empty_list(self):
        assert retrieve.diversify([]) == []


class TestBuildContext:
    def test_blocks_are_numbered_from_one_with_provenance(self):
        context = retrieve.build_context([_hit("a:001"), _hit("a:002", section="Fees")])
        first, second = context.split("\n\n")
        assert first.startswith("[1] Scheme: ")
        assert "| Section: Exit load | Source: " in first
        assert second.startswith("[2] Scheme: ")

    def test_empty_input_gives_empty_context(self):
        assert retrieve.build_context([]) == ""

    def test_context_respects_the_token_budget(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_CONTEXT_TOKENS", 140)
        hits = [
            _hit(f"a:00{index}", text="filler " * 14, section=f"Section {index}")
            for index in range(5)
        ]
        context = retrieve.build_context(hits)
        blocks = context.split("\n\n")
        assert len(blocks) == 2
        assert retrieve._context_tokens(_parse_blocks(context)) <= config.MAX_CONTEXT_TOKENS
        assert context.startswith("[1] ")

    def test_lowest_scoring_blocks_are_dropped_first(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_CONTEXT_TOKENS", 60)
        hits = [
            _hit("a:001", score=0.95, text="filler " * 60, section="Best"),
            _hit("a:002", score=0.90, text="filler " * 60, section="Second"),
            _hit("a:003", score=0.10, text="filler " * 60, section="Worst"),
        ]
        context = retrieve.build_context(hits)
        assert "Section: Worst" not in context
        assert "Section: Best" in context

    def test_never_drops_below_two_blocks(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_CONTEXT_TOKENS", 1)
        hits = [_hit("a:001", text="filler " * 60), _hit("a:002", text="filler " * 60, section="Fees")]
        assert retrieve.build_context(hits).count("Scheme: ") == 2


def _parse_blocks(context: str) -> list[RetrievedChunk]:
    """Rebuild pseudo-hits from a context string so budgets can be re-measured."""
    hits = []
    for block in context.split("\n\n"):
        header, _, body = block.partition("\n")
        section = header.split("| Section: ")[1].split(" | Source: ")[0]
        hits.append(_hit("x", text=body, section=section))
    return hits


class TestRetrieve:
    def test_returns_the_highest_scoring_chunk_with_context(self, tmp_data_dir, monkeypatch):
        _stub_query(monkeypatch)
        vectorstore.upsert_chunks([_chunk(0)], np.vstack([_fake_vector(0)]))
        result = retrieve.retrieve("What is the exit load?", k=3, min_score=0.0)
        assert result.hits and result.context
        assert result.normalized_question == "exit load?"
        assert result.context_tokens > 0

    def test_low_score_hits_are_dropped_and_counted(self, tmp_data_dir, monkeypatch):
        _stub_query(monkeypatch, _fake_vector(999))
        vectorstore.upsert_chunks([_chunk(0)], np.vstack([_fake_vector(0)]))
        result = retrieve.retrieve("What is the exit load?", k=3, min_score=0.99)
        assert result.hits == []
        assert result.dropped_low_score == 1
        assert result.context == ""

    def test_nonsense_query_returns_nothing(self, tmp_data_dir, monkeypatch):
        _stub_query(monkeypatch, _fake_vector(4242))
        vectorstore.upsert_chunks([_chunk(0)], np.vstack([_fake_vector(0)]))
        result = retrieve.retrieve("elephant banana quantum", k=5, min_score=0.35)
        assert result.hits == []
        assert result.context == ""
        assert result.dropped_low_score == 1

    def test_out_of_scope_chunk_never_reaches_the_context(self, tmp_data_dir, monkeypatch):
        _stub_query(monkeypatch)
        vectorstore.upsert_chunks(
            [_chunk(0, section="Exit load"), _out_of_scope_chunk()],
            np.vstack([_fake_vector(0), _fake_vector(1)]),
        )
        result = retrieve.retrieve("exit load", k=5, min_score=-1.0)
        assert OUT_OF_SCOPE_URL not in result.context
        assert all(
            retrieve.is_allowlisted(chunk.metadata.get("source_url", ""))
            for chunk in result.hits
        )
        assert result.dropped_out_of_scope == 1
        assert result.hits[0].metadata["source_url"] == config.CORPUS[0].url

    def test_fully_out_of_scope_corpus_raises(self, tmp_data_dir, monkeypatch):
        _stub_query(monkeypatch)
        vectorstore.upsert_chunks([_out_of_scope_chunk()], np.vstack([_fake_vector(0)]))
        with pytest.raises(retrieve.OutOfScopeError):
            retrieve.retrieve("exit load", k=5, min_score=-1.0)

    def test_empty_question_raises(self, tmp_data_dir, monkeypatch):
        _stub_query(monkeypatch)
        with pytest.raises(retrieve.RetrievalError):
            retrieve.retrieve("   ", k=3)

    def test_wrong_query_dimension_raises(self, tmp_data_dir, monkeypatch):
        monkeypatch.setattr(retrieve.embed, "embed_query", lambda text: [0.0, 1.0])
        with pytest.raises(retrieve.RetrievalError):
            retrieve.retrieve("exit load", k=3)

    def test_latency_is_measured(self, tmp_data_dir, monkeypatch):
        _stub_query(monkeypatch)
        vectorstore.upsert_chunks([_chunk(0)], np.vstack([_fake_vector(0)]))
        assert retrieve.retrieve("exit load", k=1, min_score=0.0).latency_ms >= 0

    def test_retrieve_never_imports_the_llm(self):
        source = open(retrieve.__file__, encoding="utf-8").read()
        assert "ollama" not in source.lower()
        assert "mistral" not in source.lower()
        assert "requests.post" not in source


class TestLogging:
    def test_one_json_line_is_written_per_query(self, tmp_data_dir, monkeypatch):
        _stub_query(monkeypatch)
        vectorstore.upsert_chunks([_chunk(0)], np.vstack([_fake_vector(0)]))
        result = retrieve.retrieve("exit load", k=1, min_score=0.0)
        retrieve.log_result(result)
        retrieve.log_result(result)
        lines = [
            json.loads(line)
            for line in (config.LOG_DIR / "retrieval.jsonl").read_text().splitlines()
        ]
        assert len(lines) == 2
        record = lines[0]
        for key in (
            "ts",
            "question_sha256",
            "question_chars",
            "normalized_question_sha256",
            "k",
            "min_score",
            "hits",
            "dropped_low_score",
            "dropped_out_of_scope",
            "context_tokens",
            "retrieval_ms",
        ):
            assert key in record
        assert "question" not in record
        assert "normalized_question" not in record
        assert record["hits"][0]["chunk_id"].endswith(":000")

    def test_logged_question_is_hashed(self, tmp_data_dir, monkeypatch):
        _stub_query(monkeypatch)
        vectorstore.upsert_chunks([_chunk(0)], np.vstack([_fake_vector(0)]))
        result = retrieve.retrieve(
            "my PAN is ABCDE1234F what is the exit load", k=1, min_score=0.0
        )
        retrieve.log_result(result)
        line = (config.LOG_DIR / "retrieval.jsonl").read_text()
        record = json.loads(line)
        assert "ABCDE1234F" not in line
        assert "question" not in record
        assert "question_sha256" in record
        assert record["question_chars"] == len("my PAN is ABCDE1234F what is the exit load")

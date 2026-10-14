"""Tests for stage 3 chunking. No network access: the HF tokenizer is never loaded."""
from __future__ import annotations

import json
import re
from dataclasses import FrozenInstanceError

import pytest

from src import chunking, config
from src.ingest import PageDocument, doc_id_for

SCHEME = "HDFC Small Cap Fund"
URL = "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"

EXIT_LOAD_ROWS = [
    "| Holding period | Exit load |",
    "| --- | --- |",
    "| 0-12 months | 1.0% |",
    "| 12-24 months | 0.75% |",
    "| Above 24 months | Nil |",
]


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch):
    """Force the offline estimate path so tests never download a tokenizer."""
    monkeypatch.setattr(chunking, "measure_tokens", lambda text: None)
    monkeypatch.setattr(chunking, "_load_tokenizer", lambda: (None, "disabled in tests"))


def _doc(markdown: str, scheme: str = SCHEME) -> PageDocument:
    return PageDocument(
        doc_id=doc_id_for(URL),
        url=URL,
        scheme_name=scheme,
        category="Small Cap",
        plan="Direct Growth",
        page_title=scheme,
        markdown=markdown,
        fetched_at="2026-09-27",
        content_sha256="0" * 64,
        raw_path="data/raw/x.html",
        processed_path="data/processed/x.md",
    )


def _body(chunk: chunking.Chunk) -> str:
    """Chunk text without the provenance header line."""
    return chunk.text.split("\n", 1)[1]


def _rows(text: str) -> set[str]:
    return {line.strip() for line in text.splitlines() if line.strip().startswith("|")}


def test_parse_sections_captures_headings_levels_and_bodies(sample_markdown):
    sections = chunking.parse_sections(sample_markdown)
    titles = [section.title for section in sections]
    assert "Minimum investment" in titles
    assert "Exit load" in titles

    by_title = {section.title: section for section in sections}
    assert by_title["Exit load"].level == 2
    assert "| Above 24 months | Nil |" in by_title["Exit load"].body
    assert by_title["Minimum investment"].body.count("|") > 4
    assert "Benchmark and risk" in titles


def test_parse_sections_titles_pre_heading_block_as_overview():
    sections = chunking.parse_sections("| Fund | Link |\n| --- | --- |\n| HDFC | url |\n\n# Scheme\n\nBody text.")
    assert sections[0].title == chunking.DEFAULT_SECTION_TITLE
    assert sections[0].level == 0
    assert "HDFC" in sections[0].body
    assert sections[1].title == "Scheme"


def test_parse_sections_handles_document_without_headings():
    sections = chunking.parse_sections("Plain body with no headings at all.")
    assert [section.title for section in sections] == [chunking.DEFAULT_SECTION_TITLE]
    assert sections[0].body == "Plain body with no headings at all."


def test_parse_sections_rejects_empty_input():
    with pytest.raises(chunking.ChunkingError):
        chunking.parse_sections("   \n\n  ")


def test_provenance_header_carries_scheme_plan_section_and_url():
    header = chunking.provenance_header(_doc("x"), "Scheme > Exit load")
    assert SCHEME in header
    assert "Direct Growth" in header
    assert "Scheme > Exit load" in header
    assert URL in header
    assert header.count("\n") == 0


def test_split_text_never_mixes_two_sections(sample_markdown):
    chunks = chunking.split_text(sample_markdown, _doc(sample_markdown))
    assert chunks

    sections = chunking.parse_sections(sample_markdown)
    owners: dict[str, set[str]] = {}
    for section in sections:
        for line in section.body.splitlines():
            stripped = line.strip()
            if stripped:
                owners.setdefault(stripped, set()).add(section.title)

    for chunk in chunks:
        owner = chunk.metadata["section"]
        assert chunk.text.count("Source: ") == 1
        assert chunk.metadata["section_path"].endswith(owner)
        for line in _body(chunk).splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            containing = owners.get(stripped, {owner})
            assert owner in containing, f"chunk {chunk.chunk_id} mixes sections: {stripped!r}"
            if len(containing) == 1:
                assert containing == {owner}


def test_every_table_row_appears_whole_in_exactly_one_chunk(sample_markdown):
    chunks = chunking.split_text(sample_markdown, _doc(sample_markdown))
    source_rows = _rows(sample_markdown)
    unique_source_rows = {
        row
        for row in source_rows
        if sum(1 for line in sample_markdown.splitlines() if line.strip() == row) == 1
    }

    seen: list[str] = []
    for chunk in chunks:
        for line in _body(chunk).splitlines():
            stripped = line.strip()
            if stripped.startswith("|"):
                assert stripped.endswith("|"), f"partial table row: {stripped!r}"
                assert stripped.count("|") >= 3, f"truncated table row: {stripped!r}"
        seen.extend(_rows(_body(chunk)))

    assert set(seen) == source_rows
    duplicated = [row for row in seen if seen.count(row) > 1]
    assert set(duplicated) <= (source_rows - unique_source_rows)


def test_label_value_pairs_are_never_broken(sample_markdown):
    document = (
        "# Scheme\n\n## Fees\n\n"
        + "\n\n".join(
            f"**Expense ratio {index}**: {index}.{index}% of NAV applies to the scheme."
            for index in range(40)
        )
        + "\n"
    )
    chunks = chunking.split_text(document, _doc(document))
    pairs = [line.strip() for chunk in chunks for line in _body(chunk).splitlines() if "**Expense ratio" in line]
    assert set(pairs) == {
        f"**Expense ratio {index}**: {index}.{index}% of NAV applies to the scheme."
        for index in range(40)
    }
    for line in pairs:
        assert re.fullmatch(r"\*\*Expense ratio \d+\*\*: \d+\.\d+% of NAV applies to the scheme\.", line)


def test_consecutive_chunks_in_one_section_overlap():
    sentence = "The scheme tracks small cap equities and follows a long term capital appreciation objective."
    document = "# Scheme\n\n## Performance\n\n" + "\n\n".join([sentence] * 40) + "\n"
    chunks = chunking.split_text(document, _doc(document))
    assert len(chunks) > 1

    for earlier, later in zip(chunks, chunks[1:]):
        earlier_lines = {line.strip() for line in _body(earlier).splitlines() if line.strip()}
        later_lines = {line.strip() for line in _body(later).splitlines() if line.strip()}
        assert earlier_lines & later_lines, f"no overlap between {earlier.chunk_id} and {later.chunk_id}"


def test_chunk_ids_are_stable_dense_and_ordered(sample_markdown):
    first = chunking.split_text(sample_markdown, _doc(sample_markdown))
    second = chunking.split_text(sample_markdown, _doc(sample_markdown))

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.ordinal for chunk in first] == list(range(len(first)))
    assert len({chunk.chunk_id for chunk in first}) == len(first)
    assert all(chunk.chunk_id == f"{doc_id_for(URL)}:{index:03d}" for index, chunk in enumerate(first))


def test_chunk_is_frozen_and_metadata_has_no_nulls(sample_markdown):
    chunks = chunking.split_text(sample_markdown, _doc(sample_markdown))
    with pytest.raises(FrozenInstanceError):
        chunks[0].text = "mutated"

    for chunk in chunks:
        for key, value in chunk.metadata.items():
            assert value is not None, f"{chunk.chunk_id} metadata {key} is None"
            assert isinstance(value, (str, int)) and not isinstance(value, bool), (key, value)
        assert chunk.metadata["source_url"] == URL
        assert chunk.metadata["fetched_at"] == "2026-09-27"
        assert isinstance(chunk.token_count, int) and chunk.token_count > 0


def test_split_text_keeps_short_glossary_sections():
    document = (
        "# Scheme\n\n## Understand terms\n\n### Total return\n\n"
        "The total return of a mutual fund over a given period.\n"
    )
    chunks = chunking.split_text(document, _doc(document))
    assert len(chunks) == 1
    assert "total return of a mutual fund" in chunks[0].text


def test_split_text_loses_no_source_line(sample_markdown):
    chunks = chunking.split_text(sample_markdown, _doc(sample_markdown))
    covered: set[str] = set()
    for chunk in chunks:
        covered |= {line.strip() for line in chunk.text.splitlines()}

    for line in sample_markdown.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert stripped in covered, f"line dropped from every chunk: {stripped!r}"


def test_oversized_table_is_split_by_rows_and_repeats_header():
    rows = [f"| Entry {index:03d} | {index}.{index}% |" for index in range(120)]
    document = "# Scheme\n\n## Holdings\n\n" + "\n".join([EXIT_LOAD_ROWS[0], EXIT_LOAD_ROWS[1], *rows])
    chunks = chunking.split_text(document, _doc(document))

    assert len(chunks) > 1
    for chunk in chunks:
        assert EXIT_LOAD_ROWS[0] in _body(chunk)
    rows_seen = [row for chunk in chunks for row in _rows(_body(chunk)) if "Entry" in row]
    assert len(rows_seen) == 120
    assert len(set(rows_seen)) == 120


@pytest.mark.slow
def test_chunks_fit_the_real_encoder_window(sample_markdown):
    """No chunk may exceed the window the embedder actually enforces."""
    document = "# Scheme\n\n## Holdings\n\n" + "\n".join(
        f"| Stock {index:03d} | {index}.{index}% | {index}23 units of equity capital |"
        for index in range(400)
    )
    for source in (sample_markdown, document):
        chunks = chunking.split_text(source, _doc(source))
        assert chunks
        limit = chunking.safe_chunk_tokens()
        assert max(chunk.token_count for chunk in chunks) <= limit, (
            f"chunk of {max(c.token_count for c in chunks)} tokens exceeds {limit}"
        )


@pytest.mark.slow
def test_encoder_window_is_read_from_the_embedding_model():
    """The limit must come from SentenceTransformer, not the tokenizer's 512."""
    from src import embed

    model = embed.get_model()
    assert chunking.model_max_tokens() == int(model.max_seq_length)
    assert chunking.model_max_tokens() != 512
    assert chunking.safe_chunk_tokens() == int(model.max_seq_length) - 2


@pytest.mark.slow
def test_no_chunk_loses_tokens_inside_the_encoder():
    """Every chunk must survive embedding intact; a shorter keep-count means truncation."""
    from src import embed

    tokenizer = embed.get_model().tokenizer
    limit = int(embed.get_model().max_seq_length)
    document = "# Scheme\n\n## Holdings\n\n" + "\n".join(
        f"| Stock {index:03d} | {index}.{index}% | {index}23 units of equity capital |"
        for index in range(300)
    )
    for chunk in chunking.split_text(document, _doc(document)):
        full = len(tokenizer.encode(chunk.text, add_special_tokens=False))
        kept = len(
            tokenizer.encode(
                chunk.text, add_special_tokens=True, truncation=True, max_length=limit
            )
        )
        assert kept == full + 2, f"{chunk.chunk_id}: encoder kept {kept} of {full + 2}"


def test_safe_chunk_tokens_falls_back_when_the_model_is_unavailable(monkeypatch):
    monkeypatch.setattr(chunking, "model_max_tokens", lambda: chunking.FALLBACK_MODEL_MAX_TOKENS)
    assert chunking.safe_chunk_tokens() == chunking.FALLBACK_MODEL_MAX_TOKENS - 2


def test_validate_fact_integrity_reports_missing_probe(sample_markdown):
    chunks = chunking.split_text(sample_markdown, _doc(sample_markdown))
    assert chunking.validate_fact_integrity(chunks, {SCHEME: ["expense ratio"]}) == []

    failures = chunking.validate_fact_integrity(chunks, {SCHEME: ["expense ratio", "section 80C"]})
    assert len(failures) == 1
    assert "80C" in failures[0]
    assert failures[0].startswith(SCHEME)


def test_build_report_summarises_documents_and_totals(sample_markdown):
    chunks = chunking.split_text(sample_markdown, _doc(sample_markdown))
    report = chunking.build_report({_doc(sample_markdown).doc_id: chunks})

    assert report["corpus_version"] == config.COLLECTION_NAME
    assert report["chunking"]["strategy"] == "recursive_heading_aware"
    assert report["chunking"]["chunk_chars"] == config.CHUNK_CHARS
    assert report["chunking"]["overlap_scope"] == "within_section"
    assert report["tokenizer"]["mode"] == "estimate"
    assert report["tokenizer"]["name"] == config.EMBEDDING_MODEL

    assert report["totals"]["documents"] == 1
    assert report["totals"]["chunks"] == len(chunks)
    assert 0 < report["totals"]["median_tokens"] <= report["totals"]["max_tokens"]

    entry = report["documents"][0]
    assert entry["scheme_name"] == SCHEME
    assert entry["chunks"] == len(chunks)
    assert entry["sections"] == len({chunk.metadata["section_path"] for chunk in chunks})
    assert sum(entry["section_histogram"].values()) == len(chunks)
    assert json.dumps(report)


def test_write_report_uses_the_configured_path(tmp_data_dir, sample_markdown):
    chunks = chunking.split_text(sample_markdown, _doc(sample_markdown))
    path = chunking.write_report(chunking.build_report({_doc(sample_markdown).doc_id: chunks}))

    assert path == config.CHUNKING_REPORT_PATH
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["totals"]["chunks"] == len(chunks)


def _write_single_page(tmp_path, monkeypatch, markdown: str) -> None:
    spec = config.SourceSpec(url=URL, scheme_name=SCHEME, category="Small Cap", plan="Direct Growth")
    monkeypatch.setattr(config, "CORPUS", [spec])
    processed = config.PROCESSED_DIR / f"{doc_id_for(URL)}.md"
    processed.parent.mkdir(parents=True, exist_ok=True)
    processed.write_text(markdown, encoding="utf-8")


def test_main_reports_missing_processed_pages(tmp_data_dir, capsys):
    assert chunking.main([]) == 1
    assert "run 'python -m src.ingest'" in capsys.readouterr().out


def test_main_writes_the_report_and_succeeds(tmp_data_dir, monkeypatch, sample_markdown, capsys):
    _write_single_page(tmp_data_dir, monkeypatch, sample_markdown)
    assert chunking.main(["--report"]) == 0

    output = capsys.readouterr().out
    assert "fact integrity" in output
    assert config.CHUNKING_REPORT_PATH.exists()
    payload = json.loads(config.CHUNKING_REPORT_PATH.read_text(encoding="utf-8"))
    assert payload["totals"]["chunks"] > 0


def test_main_strict_fails_when_a_must_have_fact_is_missing(tmp_data_dir, monkeypatch, sample_markdown):
    _write_single_page(tmp_data_dir, monkeypatch, sample_markdown)
    monkeypatch.setattr(config, "MUST_HAVE_FACTS", {SCHEME: ["section 80C"]})

    assert chunking.main(["--report"]) == 0
    assert chunking.main(["--report", "--strict"]) == 1

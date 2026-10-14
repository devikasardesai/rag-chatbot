"""Tests for stage 1-2 ingestion. No network access: httpx.get is blocked by default."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from src import config, ingest

SCHEME_SENTENCE = (
    "HDFC Small Cap Fund seeks to invest in small cap stocks across sectors and is "
    "managed with a long-term capital appreciation objective for investors who can "
    "withstand high volatility and bear the associated market risk over the holding "
    "period of the investment."
)

FACT_LINES = [
    "## Minimum investment",
    "| Mode | Minimum amount |",
    "| --- | --- |",
    "| Lump sum | Rs. 10,000 |",
    "| SIP | Rs. 500 per month |",
    "## Exit load",
    "| Holding period | Exit load |",
    "| --- | --- |",
    "| 0-12 months | 1.0% |",
    "| 12-24 months | 0.75% |",
    "| Above 24 months | Nil |",
    "## Fees and expenses",
    "| Particulars | % of NAV |",
    "| --- | --- |",
    "| Management fee | 0.50% |",
    "| Total expense ratio | 1.19% |",
    "## Benchmark and risk",
    "The benchmark of the scheme is Nifty Smallcap 250 TRI and the riskometer category is Very High.",
    "## Lock-in and tax",
    "The lock-in period is 3 years and the section 80C benefit applies to eligible investors.",
    "## Tax and documents",
    "To download the capital-gains statement, use the investor portal and select the statement type.",
]

BOILERPLATE_SNIPPETS = [
    "Sign In",
    "Accept all cookies",
    "Related Funds:",
    "NAVIGATION_SIDEBAR_TOKEN",
    "FOOTER_COPYRIGHT_TOKEN",
    "ASIDE_PROMO_TOKEN",
]


def _fake_html(body_lines: list[str] | None = None) -> str:
    facts = "\n".join(body_lines if body_lines is not None else FACT_LINES)
    filler = "\n".join(f"<p>{SCHEME_SENTENCE}</p>" for _ in range(8))
    boilerplate = "".join(f"<div>{token}</div>" for token in BOILERPLATE_SNIPPETS)
    return f"""<!DOCTYPE html>
<html lang="en">
<head><title>HDFC Small Cap Fund Direct Growth - NAV, Mutual Fund Performance</title></head>
<body>
<nav>{boilerplate}</nav>
<header>HEADER_BRAND_TOKEN</header>
<aside>ASIDE_PROMO_TOKEN</aside>
<div class="layout-main">
  <div class="fundDetails_fundDetailsContainer">
    <h1>HDFC Small Cap Fund Direct Growth</h1>
    {facts}
    {filler}
  </div>
</div>
<footer>FOOTER_COPYRIGHT_TOKEN</footer>
<script>NAVIGATION_SIDEBAR_TOKEN</script>
</body>
</html>"""


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code
        self.content = text.encode("utf-8")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if any code path performs real network I/O."""

    def _blocked(*args, **kwargs):
        raise AssertionError("tests must not perform network I/O")

    monkeypatch.setattr(httpx, "get", _blocked)
    return _blocked


@pytest.fixture()
def no_sleep(monkeypatch):
    """Remove politeness and backoff delays from the test run."""
    slept: list[float] = []
    monkeypatch.setattr(ingest, "_sleep", slept.append)
    return slept


@pytest.fixture()
def fake_http(monkeypatch, no_sleep):
    """Serve a scripted HTML body for every allowlisted URL."""
    calls: list[str] = []

    def _handler(url, **kwargs):
        calls.append(url)
        return _FakeResponse(_fake_html())

    monkeypatch.setattr(httpx, "get", _handler)
    return calls


def _corpus_call_counts(calls: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for url in calls:
        counts[url] = counts.get(url, 0) + 1
    return counts


def _landmark_html() -> str:
    inner = "\n".join(FACT_LINES + [f"<p>{SCHEME_SENTENCE}</p>" for _ in range(8)])
    return f"""<!DOCTYPE html>
<html lang="en">
<head><title>ELSS Landmark Page</title></head>
<body>
<header>SITE_HEADER_TOKEN</header>
<div class="layout-main">
  <header><span>ELSS &bull; 3Y Lock-in</span></header>
  <main>
    <h1>HDFC ELSS Tax Saver Fund Direct Plan Growth</h1>
    {inner}
  </main>
</div>
<footer>FOOTER_COPYRIGHT_TOKEN</footer>
</body>
</html>"""


class TestDocId:
    def test_doc_id_is_sha1_prefix(self):
        url = "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"
        assert ingest.doc_id_for(url) == hashlib.sha1(url.encode()).hexdigest()[:12]

    def test_doc_id_is_stable_and_twelve_chars(self, corpus):
        ids = [ingest.doc_id_for(spec.url) for spec in corpus]
        assert all(len(doc_id) == 12 for doc_id in ids)
        assert ids == [ingest.doc_id_for(spec.url) for spec in corpus]

    def test_doc_id_differs_per_url(self, corpus):
        ids = {ingest.doc_id_for(spec.url) for spec in corpus}
        assert len(ids) == len(corpus)

    def test_doc_id_for_foreign_url(self, fake_specs):
        assert ingest.doc_id_for(fake_specs[0].url) != ingest.doc_id_for(fake_specs[1].url)


class TestPageDocument:
    def test_page_document_is_frozen(self, fake_http, corpus, tmp_data_dir):
        document = ingest.fetch_page(corpus[3])
        with pytest.raises(FrozenInstanceError):
            document.markdown = "mutated"

    def test_page_document_fields(self, fake_http, corpus, tmp_data_dir):
        spec = corpus[3]
        document = ingest.fetch_page(spec)
        assert document.url == spec.url
        assert document.scheme_name == spec.scheme_name
        assert document.category == spec.category
        assert document.plan == spec.plan
        assert document.doc_id == ingest.doc_id_for(spec.url)
        assert document.page_title.startswith("HDFC Small Cap Fund Direct Growth")
        assert document.fetched_at == datetime.now(timezone.utc).date().isoformat()

    def test_paths_recorded_for_raw_and_processed(self, fake_http, corpus, tmp_data_dir):
        document = ingest.fetch_page(corpus[0])
        assert document.raw_path.endswith(f"{document.doc_id}.html")
        assert document.processed_path.endswith(f"{document.doc_id}.md")
        assert (tmp_data_dir / "data" / "raw" / f"{document.doc_id}.html").exists()


class TestCache:
    def test_first_call_fetches_and_second_call_uses_cache(self, fake_http, corpus, tmp_data_dir):
        spec = corpus[2]
        first = ingest.fetch_page(spec)
        second = ingest.fetch_page(spec)
        assert _corpus_call_counts(fake_http)[spec.url] == 1
        assert first.markdown == second.markdown
        assert first.content_sha256 == second.content_sha256
        assert first.fetched_at == second.fetched_at

    def test_force_refetches(self, fake_http, corpus, tmp_data_dir):
        spec = corpus[2]
        ingest.fetch_page(spec)
        ingest.fetch_page(spec, force=True)
        assert _corpus_call_counts(fake_http)[spec.url] == 2

    def test_cache_changes_are_written_to_disk(self, fake_http, corpus, tmp_data_dir):
        document = ingest.fetch_page(corpus[1])
        raw_path = tmp_data_dir / "data" / "raw" / f"{document.doc_id}.html"
        assert raw_path.read_text(encoding="utf-8") == _fake_html()

    def test_fetched_at_comes_from_cache_file_mtime(self, fake_http, corpus, tmp_data_dir):
        spec = corpus[4]
        first = ingest.fetch_page(spec)
        raw_path = tmp_data_dir / "data" / "raw" / f"{first.doc_id}.html"
        backdated = datetime(2026, 1, 15, 9, 30, tzinfo=timezone.utc).timestamp()
        os.utime(raw_path, (backdated, backdated))
        assert ingest.fetch_page(spec).fetched_at == "2026-01-15"

    def test_content_sha256_is_of_html_bytes(self, fake_http, corpus, tmp_data_dir):
        document = ingest.fetch_page(corpus[0])
        expected = hashlib.sha256(_fake_html().encode("utf-8")).hexdigest()
        assert document.content_sha256 == expected

    def test_changed_page_changes_sha256(self, fake_http, monkeypatch, corpus, tmp_data_dir, no_sleep):
        spec = corpus[0]
        original = ingest.fetch_page(spec)
        changed_html = _fake_html().replace("1.19%", "1.29%")
        monkeypatch.setattr(httpx, "get", lambda url, **kwargs: _FakeResponse(changed_html))
        refreshed = ingest.fetch_page(spec, force=True)
        assert refreshed.content_sha256 != original.content_sha256
        assert "1.29%" in refreshed.markdown


class TestAllowlist:
    def test_fetch_page_rejects_foreign_url(self, fake_specs, tmp_data_dir):
        with pytest.raises(ingest.OutOfScopeError):
            ingest.fetch_page(fake_specs[0])

    def test_out_of_scope_url_makes_no_request(self, fake_specs, tmp_data_dir):
        with pytest.raises(ingest.OutOfScopeError):
            ingest.fetch_page(fake_specs[1])
        assert not (tmp_data_dir / "data" / "raw").exists()

    def test_error_message_names_the_url(self, fake_specs, tmp_data_dir):
        with pytest.raises(ingest.OutOfScopeError, match="example.com"):
            ingest.fetch_page(fake_specs[0])

    def test_build_corpus_rejects_foreign_url(self, fake_specs, tmp_data_dir):
        with pytest.raises(ingest.OutOfScopeError):
            ingest.build_corpus(fake_specs)

    def test_build_corpus_rejects_empty_source_list(self, tmp_data_dir):
        with pytest.raises(ingest.IngestError):
            ingest.build_corpus([])


class TestHttpFailures:
    def test_non_200_raises_with_url(self, monkeypatch, corpus, tmp_data_dir, no_sleep):
        monkeypatch.setattr(httpx, "get", lambda url, **kwargs: _FakeResponse("nope", status_code=403))
        with pytest.raises(ingest.IngestError, match="hdfc-small-cap"):
            ingest.fetch_page(corpus[3])

    def test_network_error_is_retried(self, monkeypatch, corpus, tmp_data_dir, no_sleep):
        attempts: list[int] = []

        def _handler(url, **kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise httpx.ConnectError("boom")
            return _FakeResponse(_fake_html())

        monkeypatch.setattr(httpx, "get", _handler)
        document = ingest.fetch_page(corpus[0])
        assert len(attempts) == 2
        assert document.markdown

    def test_exhausted_retries_raise(self, monkeypatch, corpus, tmp_data_dir, no_sleep):
        attempts: list[int] = []

        def _handler(url, **kwargs):
            attempts.append(1)
            raise httpx.ConnectError("boom")

        monkeypatch.setattr(httpx, "get", _handler)
        with pytest.raises(ingest.IngestError, match="failed to fetch"):
            ingest.fetch_page(corpus[0])
        assert len(attempts) == config.HTTP_RETRIES + 1

    def test_request_uses_configured_user_agent_and_timeout(self, monkeypatch, corpus, tmp_data_dir, no_sleep):
        seen: dict = {}

        def _handler(url, **kwargs):
            seen["url"] = url
            seen.update(kwargs)
            return _FakeResponse(_fake_html())

        monkeypatch.setattr(httpx, "get", _handler)
        ingest.fetch_page(corpus[0])
        assert seen["headers"]["User-Agent"] == config.USER_AGENT
        assert seen["follow_redirects"] is True
        assert seen["timeout"] == config.HTTP_TIMEOUT_S

    def test_politeness_delay_applied_between_requests(self, monkeypatch, corpus, tmp_data_dir, no_sleep):
        monkeypatch.setattr(httpx, "get", lambda url, **kwargs: _FakeResponse(_fake_html()))
        ingest.build_corpus(config.CORPUS)
        assert no_sleep.count(config.POLITENESS_DELAY_S) == len(config.CORPUS)


class TestExtractMainContent:
    def test_keeps_scheme_facts(self):
        title, markdown = ingest.extract_main_content(_fake_html())
        assert title.startswith("HDFC Small Cap Fund Direct Growth")
        for fact in ("expense ratio", "Exit load", "SIP", "lock-in period is 3 years",
                     "Nifty Smallcap 250 TRI", "capital-gains statement"):
            assert fact.lower() in markdown.lower()

    def test_boilerplate_is_stripped(self):
        _, markdown = ingest.extract_main_content(_fake_html())
        for token in BOILERPLATE_SNIPPETS + ["HEADER_BRAND_TOKEN"]:
            assert token not in markdown

    def test_noise_lines_are_dropped(self):
        html = _fake_html().replace("<nav>", "<nav>\n<p>Sign In</p>\n<p>Accept all cookies</p>\n")
        _, markdown = ingest.extract_main_content(html)
        assert "Sign In" not in markdown
        assert "cookies" not in markdown.lower()

    def test_tables_and_headings_survive(self):
        _, markdown = ingest.extract_main_content(_fake_html())
        assert "## Exit load" in markdown
        assert "| 0-12 months | 1.0% |" in markdown

    def test_blank_lines_collapsed_and_lines_stripped(self):
        html = _fake_html().replace("</h1>", "</h1>\n\n\n\n   \n")
        _, markdown = ingest.extract_main_content(html)
        assert "\n\n\n" not in markdown
        assert all(line == line.rstrip() for line in markdown.splitlines())
        assert markdown.endswith("\n")

    def test_raises_when_extraction_too_short(self):
        with pytest.raises(ingest.IngestError, match="500 chars"):
            ingest.extract_main_content("<html><body><nav>Sign In</nav><p>hi</p></body></html>")

    def test_raises_when_both_extractors_fail(self, monkeypatch):
        monkeypatch.setattr(ingest, "_trafilatura_markdown", lambda html: (_ for _ in ()).throw(RuntimeError("x")))
        monkeypatch.setattr(ingest, "_bs4_main_markdown", lambda html: "")
        with pytest.raises(ingest.IngestError):
            ingest.extract_main_content(_fake_html())

    def test_prefers_the_candidate_with_more_facts(self, monkeypatch):
        holdings_only = "## Holdings\n" + "\n".join(["| ICICI Bank | Financial |"] * 200)
        facts = _fake_html()
        monkeypatch.setattr(ingest, "_trafilatura_markdown", lambda html: holdings_only)
        _, markdown = ingest.extract_main_content(facts)
        assert "expense ratio" in markdown.lower()
        assert "ICICI Bank" not in markdown

    def test_prefers_trafilatura_when_both_candidates_are_equal(self, monkeypatch):
        shared = _fake_html()
        monkeypatch.setattr(ingest, "_trafilatura_markdown", lambda html: shared)
        monkeypatch.setattr(ingest, "_bs4_main_markdown", lambda html: shared)
        _, markdown = ingest.extract_main_content(shared)
        assert "expense ratio" in markdown.lower()

    def test_uses_bs4_when_trafilatura_fails(self, monkeypatch):
        monkeypatch.setattr(ingest, "_trafilatura_markdown", lambda html: "")
        _, markdown = ingest.extract_main_content(_fake_html())
        assert "expense ratio" in markdown.lower()

    def test_uses_trafilatura_when_bs4_fails(self, monkeypatch):
        monkeypatch.setattr(ingest, "_bs4_main_markdown", lambda html: "")
        _, markdown = ingest.extract_main_content(_fake_html())
        assert "expense ratio" in markdown.lower()


class TestLandmarkExtraction:
    def test_landmark_container_is_preferred(self):
        _, markdown = ingest.extract_main_content(_landmark_html())
        assert "3Y Lock-in" in markdown
        assert "expense ratio" in markdown.lower()

    def test_site_level_header_is_stripped(self):
        _, markdown = ingest.extract_main_content(_landmark_html())
        assert "SITE_HEADER_TOKEN" not in markdown
        assert "FOOTER_COPYRIGHT_TOKEN" not in markdown

    def test_in_content_header_is_kept(self):
        _, markdown = ingest.extract_main_content(_landmark_html())
        assert "3Y Lock-in" in markdown

    def test_falls_back_to_density_heuristic_without_landmark(self):
        soup_html = _landmark_html().replace('class="layout-main"', 'class="content-wrap"')
        soup_html = soup_html.replace("<main>", "<div>").replace("</main>", "</div>")
        _, markdown = ingest.extract_main_content(soup_html)
        assert "expense ratio" in markdown.lower()
        assert "SITE_HEADER_TOKEN" not in markdown

    def test_boilerplate_tags_stripped(self):
        _, markdown = ingest.extract_main_content(_landmark_html())
        assert "SITE_HEADER_TOKEN" not in markdown
        assert "FOOTER_COPYRIGHT_TOKEN" not in markdown


class TestBuildCorpus:
    def test_builds_five_documents_and_writes_markdown(self, fake_http, tmp_data_dir):
        documents = ingest.build_corpus(config.CORPUS)
        assert len(documents) == 5
        assert len({document.doc_id for document in documents}) == 5
        for document in documents:
            processed = tmp_data_dir / "data" / "processed" / f"{document.doc_id}.md"
            assert processed.exists()
            assert processed.read_text(encoding="utf-8") == document.markdown

    def test_second_run_uses_cache(self, fake_http, tmp_data_dir):
        ingest.build_corpus(config.CORPUS)
        first_counts = _corpus_call_counts(fake_http)
        ingest.build_corpus(config.CORPUS)
        assert _corpus_call_counts(fake_http) == first_counts
        assert set(first_counts) == {spec.url for spec in config.CORPUS}

    def test_defaults_to_config_corpus(self, fake_http, tmp_data_dir):
        assert len(ingest.build_corpus()) == 5

    def test_force_refreshes_all_pages(self, fake_http, tmp_data_dir):
        ingest.build_corpus(config.CORPUS)
        ingest.build_corpus(config.CORPUS, force=True)
        assert all(count == 2 for count in _corpus_call_counts(fake_http).values())

    def test_fact_sections_reach_the_processed_markdown(self, fake_http, tmp_data_dir):
        markdown = ingest.build_corpus(config.CORPUS)[3].markdown
        assert "SIP" in markdown
        assert "expense ratio" in markdown.lower()


class TestManifest:
    def test_manifest_matches_architecture_shape(self, fake_http, tmp_data_dir):
        documents = ingest.build_corpus(config.CORPUS)
        path = ingest.write_manifest(documents, {d.doc_id: 7 for d in documents})
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["corpus_version"] == config.COLLECTION_NAME
        assert manifest["embedding_model"] == config.EMBEDDING_MODEL
        assert manifest["chunking"] == {
            "strategy": "recursive_heading_aware",
            "chunk_tokens": config.CHUNK_SIZE_TOKENS,
            "overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
            "preserve_tables": True,
        }
        assert len(manifest["documents"]) == 5
        assert manifest["totals"] == {"documents": 5, "chunks": 35}

    def test_manifest_document_fields(self, fake_http, tmp_data_dir):
        documents = ingest.build_corpus(config.CORPUS)
        manifest = json.loads(
            ingest.write_manifest(documents, {d.doc_id: 3 for d in documents}).read_text(
                encoding="utf-8"
            )
        )
        entry = manifest["documents"][0]
        document = documents[0]
        assert entry["doc_id"] == document.doc_id
        assert entry["scheme_name"] == document.scheme_name
        assert entry["category"] == document.category
        assert entry["plan"] == document.plan
        assert entry["url"] in config.ALLOWED_URLS
        assert entry["fetched_at"] == document.fetched_at
        assert entry["content_sha256"] == document.content_sha256
        assert entry["chunk_count"] == 3

    def test_manifest_accepts_chunk_lists(self, fake_http, tmp_data_dir):
        documents = ingest.build_corpus(config.CORPUS)
        manifest = json.loads(
            ingest.write_manifest(
                documents, {d.doc_id: [None] * 5 for d in documents}
            ).read_text(encoding="utf-8")
        )
        assert manifest["totals"]["chunks"] == 25

    def test_manifest_totals_track_per_document_counts(self, fake_http, tmp_data_dir):
        documents = ingest.build_corpus(config.CORPUS)
        counts = {document.doc_id: index for index, document in enumerate(documents)}
        manifest = json.loads(
            ingest.write_manifest(documents, counts).read_text(encoding="utf-8")
        )
        assert manifest["totals"]["chunks"] == sum(counts.values())
        assert [d["chunk_count"] for d in manifest["documents"]] == list(counts.values())

    def test_manifest_generated_at_is_iso_zulu(self, fake_http, tmp_data_dir):
        documents = ingest.build_corpus(config.CORPUS)
        manifest = json.loads(
            ingest.write_manifest(documents, {}).read_text(encoding="utf-8")
        )
        assert manifest["generated_at"].endswith("Z")
        datetime.strptime(manifest["generated_at"], "%Y-%m-%dT%H:%M:%SZ")

    def test_manifest_is_written_under_data_dir_override(self, fake_http, tmp_data_dir):
        documents = ingest.build_corpus(config.CORPUS)
        path = ingest.write_manifest(documents, {})
        assert path == tmp_data_dir / "corpus_manifest.json"
        assert path.exists()


class TestCli:
    @pytest.mark.slow
    def test_main_ingests_all_pages(self, fake_http, tmp_data_dir, capsys):
        assert ingest.main([]) == 0
        out = capsys.readouterr().out
        for spec in config.CORPUS:
            assert spec.scheme_name in out
        assert "sha256" in out
        assert "doc_id" in out
        assert "chunks" in out
        assert "manifest" in out
        assert config.MANIFEST_PATH.name in out

    @pytest.mark.slow
    def test_main_single_url(self, fake_http, tmp_data_dir, capsys):
        url = config.CORPUS[3].url
        assert ingest.main(["--url", url]) == 0
        out = capsys.readouterr().out
        assert "HDFC Small Cap Fund" in out
        assert len(list((tmp_data_dir / "data" / "processed").iterdir())) == 1

    def test_main_rejects_foreign_url(self, tmp_data_dir, capsys):
        assert ingest.main(["--url", "https://example.com/not-allowlisted"]) == 1
        assert "not an allowlisted corpus URL" in capsys.readouterr().err

    def test_main_reports_failure_without_traceback(self, monkeypatch, tmp_data_dir, capsys):
        monkeypatch.setattr(httpx, "get", lambda url, **kwargs: _FakeResponse("x", status_code=500))
        assert ingest.main([]) == 1
        assert "ingest failed" in capsys.readouterr().err

    def test_main_stats_delegates_to_vectorstore(self, tmp_data_dir, capsys):
        assert ingest.main(["--stats"]) == 0
        assert config.COLLECTION_NAME in capsys.readouterr().out


class TestNoNetworkGuarantee:
    def test_default_blocker_rejects_http_calls(self, no_network):
        with pytest.raises(AssertionError, match="must not perform network"):
            no_network("https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth")

    def test_processed_file_is_reloadable_markdown(self, fake_http, tmp_data_dir):
        document = ingest.build_corpus(config.CORPUS)[0]
        reloaded = Path(document.processed_path).read_text(encoding="utf-8")
        assert reloaded.startswith("#") or reloaded.lstrip().startswith("|")
        assert reloaded.strip() == reloaded.rstrip()


class TestEndToEndPipeline:
    @pytest.mark.slow
    def test_pipeline_stores_every_chunk(self, fake_http, tmp_data_dir):
        from src import vectorstore

        result = ingest.run_pipeline()
        assert result.stored == len(result.chunks)
        assert result.pruned == 0
        stored = set(vectorstore.get_or_create_collection().get(include=[])["ids"])
        assert stored == {chunk.chunk_id for chunk in result.chunks}
        assert result.manifest_path.exists()

    @pytest.mark.slow
    def test_two_runs_produce_the_same_chunk_id_set(self, fake_http, tmp_data_dir):
        from src import vectorstore

        first = ingest.run_pipeline()
        first_ids = {chunk.chunk_id for chunk in first.chunks}
        second = ingest.run_pipeline()
        second_ids = {chunk.chunk_id for chunk in second.chunks}
        assert first_ids == second_ids
        assert len(first.chunks) == len(second.chunks)
        assert vectorstore.get_or_create_collection().count() == len(second_ids)
        assert second.pruned == 0

    @pytest.mark.slow
    def test_pipeline_prunes_chunks_when_a_document_shrinks(self, fake_http, tmp_data_dir):
        from src import vectorstore

        full = ingest.run_pipeline()
        subset = ingest.run_pipeline(list(config.CORPUS[:4]))
        stored = set(vectorstore.get_or_create_collection().get(include=[])["ids"])
        assert stored == {chunk.chunk_id for chunk in subset.chunks}
        assert subset.pruned == len(full.chunks) - len(subset.chunks)
        assert len(subset.chunks) < len(full.chunks)

    @pytest.mark.slow
    def test_manifest_chunk_counts_match_the_collection(self, fake_http, tmp_data_dir):
        from src import vectorstore

        result = ingest.run_pipeline()
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest["totals"]["chunks"] == len(result.chunks)
        assert manifest["totals"]["documents"] == 5
        assert sum(d["chunk_count"] for d in manifest["documents"]) == len(result.chunks)
        assert vectorstore.get_or_create_collection().count() == len(result.chunks)

    @pytest.mark.slow
    def test_reset_rebuilds_the_collection(self, fake_http, tmp_data_dir):
        from src import vectorstore

        ingest.run_pipeline()
        result = ingest.run_pipeline(reset=True)
        assert vectorstore.get_or_create_collection().count() == len(result.chunks)

"""Stages 1-5 ingestion: load and clean the allowlisted pages, then chunk, embed, and store."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import trafilatura
from bs4 import BeautifulSoup
from lxml import html as lxml_html
from markdownify import markdownify

from src import config, embed, logger, vectorstore

MIN_EXTRACTED_CHARS = 500
MIN_CONTAINER_CHARS = 2000
RETRY_BACKOFF_S = 2.0

BOILERPLATE_TAGS = ("nav", "footer", "script", "style", "noscript", "aside", "form", "svg")
BOILERPLATE_SELECTORS = ("body > header", "[role=navigation]")
LANDMARK_SELECTORS = ("main", "[role=main]", "div.layout-main")

NOISE_LINE_PATTERNS = (
    re.compile(r"^\s*(sign in|log ?in|sign up|sign up|register|create an account)\s*$", re.I),
    re.compile(r"^\s*(accept|manage|reject)\s+(all\s+)?cookies?\b", re.I),
    re.compile(r"^\s*cookie\s+preferences\b", re.I),
    re.compile(r"^\s*(download|get|open)\s+the\s+(groww\s+)?app\s*$", re.I),
    re.compile(r"^\s*(follow us on|share on|refer and earn)\b", re.I),
    re.compile(r"^\s*(skip to|b back to top)\b", re.I),
    re.compile(r"^\s*all rights reserved\b", re.I),
    re.compile(r"^\s*(related|other|similar|top|trending|popular)\s+(funds?|schemes?)\s*:?\s*$", re.I),
    re.compile(r"^\s*(filters?|sort by|apply|clear all)\s*:?\s*$", re.I),
    re.compile(r"^\s*(home\s*>\s*)?mutual funds\s*>\s*", re.I),
    re.compile(r"^\s*\d+\s+funds?\s+found\s*$", re.I),
)

FACT_TERMS = (
    "expense ratio",
    "exit load",
    "minimum",
    "sip",
    "benchmark",
    "risk",
    "lock-in",
    "riskometer",
    "aum",
    "nav",
    "holdings",
    "turnover",
    "fund manager",
    "tax",
    "statement",
)

CONTENT_CONTAINER_TAGS = ("div", "section", "main", "article")


class IngestError(RuntimeError):
    """Raised when a page cannot be fetched or cleaned into usable Markdown."""


class OutOfScopeError(RuntimeError):
    """Raised when a URL is not in config.ALLOWED_URLS."""


def _chunking():
    """Return the stage 3 module, importing it on first use.

    ``chunking`` reuses ``PageDocument`` from this module, so a module-level import in
    both directions would be a cycle; the architecture declares ingest -> chunking.
    """
    from src import chunking

    return chunking


@dataclass(frozen=True)
class PageDocument:
    """One cleaned page plus the provenance recorded in the corpus manifest."""

    doc_id: str
    url: str
    scheme_name: str
    category: str
    plan: str
    page_title: str
    markdown: str
    fetched_at: str
    content_sha256: str
    raw_path: str
    processed_path: str


def doc_id_for(url: str) -> str:
    """Stable page identity: the first 12 hex characters of sha1(url)."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


def _iso_date(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()


def _iso_date_now() -> str:
    return datetime.now(tz=timezone.utc).date().isoformat()


def _http_get(url: str) -> httpx.Response:
    attempts = config.HTTP_RETRIES + 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        _sleep(config.POLITENESS_DELAY_S)
        try:
            response = httpx.get(
                url,
                headers={"User-Agent": config.USER_AGENT},
                follow_redirects=True,
                timeout=config.HTTP_TIMEOUT_S,
            )
        except httpx.HTTPError as error:
            last_error = error
            if attempt < attempts:
                _sleep(RETRY_BACKOFF_S * attempt)
            continue
        if response.status_code != 200:
            raise IngestError(f"{url} returned HTTP {response.status_code}")
        return response
    raise IngestError(f"failed to fetch {url} after {attempts} attempts: {last_error}")


def _extract_title(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
    except (lxml_html.etree.ParserError, ValueError):
        return ""
    if soup.title and soup.title.string:
        return " ".join(soup.title.string.split())
    return ""


def _trafilatura_markdown(html: str) -> str:
    return (
        trafilatura.extract(
            html,
            output_format="markdown",
            include_tables=True,
            include_links=False,
        )
        or ""
    )


def _largest_landmark(soup: BeautifulSoup):
    matches: list = []
    for selector in LANDMARK_SELECTORS:
        matches.extend(soup.select(selector))
    if not matches:
        return None
    return max(matches, key=lambda element: len(element.get_text(" ", strip=True)))


def _densest_content_element(soup: BeautifulSoup):
    best_element = None
    best_score = (0, 0)
    for element in soup.find_all(CONTENT_CONTAINER_TAGS):
        paragraph_text = sum(len(p.get_text(" ", strip=True)) for p in element.find_all("p"))
        total_text = len(element.get_text(" ", strip=True))
        if total_text < MIN_CONTAINER_CHARS:
            continue
        score = (paragraph_text, total_text)
        if best_element is None or score > best_score:
            best_element, best_score = element, score
    return best_element


def _bs4_main_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag_name in BOILERPLATE_TAGS:
        for element in soup.find_all(tag_name):
            element.extract()
    for element in soup.select(", ".join(BOILERPLATE_SELECTORS)):
        element.extract()
    landmark = _largest_landmark(soup)
    if landmark is not None:
        markdown = markdownify(str(landmark), heading_style="ATX")
        if len(markdown.strip()) >= MIN_EXTRACTED_CHARS:
            return markdown
    container = _densest_content_element(soup)
    if container is None:
        return ""
    return markdownify(str(container), heading_style="ATX")


def _candidate_score(markdown: str) -> tuple[int, int]:
    lowered = markdown.lower()
    return (sum(1 for term in FACT_TERMS if term in lowered), len(markdown))


def _normalise_markdown(markdown: str) -> str:
    lines = [line.rstrip() for line in markdown.splitlines()]
    kept = [line for line in lines if not any(p.match(line) for p in NOISE_LINE_PATTERNS)]
    collapsed: list[str] = []
    blank_run = 0
    for line in kept:
        if line:
            blank_run = 0
            collapsed.append(line)
            continue
        blank_run += 1
        if blank_run == 1:
            collapsed.append("")
    return "\n".join(collapsed).strip() + "\n"


def extract_main_content(html: str) -> tuple[str, str]:
    """Return (page_title, markdown) for the page body.

    trafilatura is the primary extractor, but on Groww scheme pages it prunes the
    fact containers (expense ratio, exit load, SIP, lock-in pill) and keeps only the
    holdings and returns tables. Both candidates are therefore extracted and the
    richer one wins, with trafilatura preferred on ties.
    """
    title = _extract_title(html)
    candidates: list[str] = []
    for extractor in (_trafilatura_markdown, _bs4_main_markdown):
        try:
            text = extractor(html)
        except Exception as error:
            print(f"  extractor {extractor.__name__} failed: {type(error).__name__}: {error}")
            continue
        if text and len(text.strip()) >= MIN_EXTRACTED_CHARS:
            candidates.append(text)
    if not candidates:
        raise IngestError(
            f"extracted markdown below {MIN_EXTRACTED_CHARS} chars from {title or 'untitled'!r} page"
        )
    return title, _normalise_markdown(max(candidates, key=_candidate_score))


def fetch_page(spec: config.SourceSpec, force: bool = False) -> PageDocument:
    """Fetch one allowlisted page (cache-first) and return it as a PageDocument."""
    if spec.url not in config.ALLOWED_URLS:
        raise OutOfScopeError(f"{spec.url} is not an allowlisted corpus URL")
    doc_id = doc_id_for(spec.url)
    raw_path = config.RAW_DIR / f"{doc_id}.html"
    cached = raw_path.exists() and not force
    if cached:
        payload = raw_path.read_bytes()
        fetched_at = _iso_date(raw_path.stat().st_mtime)
    else:
        response = _http_get(spec.url)
        payload = response.content
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(payload)
        fetched_at = _iso_date_now()
    html = payload.decode("utf-8", errors="replace")
    page_title, markdown = extract_main_content(html)
    return PageDocument(
        doc_id=doc_id,
        url=spec.url,
        scheme_name=spec.scheme_name,
        category=spec.category,
        plan=spec.plan,
        page_title=page_title,
        markdown=markdown,
        fetched_at=fetched_at,
        content_sha256=hashlib.sha256(payload).hexdigest(),
        raw_path=_relative_path(raw_path),
        processed_path=_relative_path(config.PROCESSED_DIR / f"{doc_id}.md"),
    )


def build_corpus(
    sources: list[config.SourceSpec] | None = None, force: bool = False
) -> list[PageDocument]:
    """Fetch every allowlisted source and write its cleaned Markdown to disk."""
    specs = list(config.CORPUS if sources is None else sources)
    if not specs:
        raise IngestError("no sources provided")
    documents: list[PageDocument] = []
    for spec in specs:
        if spec.url not in config.ALLOWED_URLS:
            raise OutOfScopeError(f"{spec.url} is not an allowlisted corpus URL")
        document = fetch_page(spec, force=force)
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        Path(document.processed_path).write_text(document.markdown, encoding="utf-8")
        documents.append(document)
    return documents


@dataclass(frozen=True)
class PipelineResult:
    """Outcome of one stages 1-5 run."""

    documents: list[PageDocument]
    chunks: list["chunking.Chunk"]
    stored: int
    pruned: int
    manifest_path: Path


def _spec_for_url(url: str) -> config.SourceSpec:
    for spec in config.CORPUS:
        if spec.url == url:
            return spec
    raise OutOfScopeError(f"{url} is not an allowlisted corpus URL")


def _was_cached(document_id: str, force: bool) -> bool:
    return (config.RAW_DIR / f"{document_id}.html").exists() and not force


def _chunk_count(value: int | list | tuple) -> int:
    return value if isinstance(value, int) else len(value)


def write_manifest(
    documents: list[PageDocument],
    chunks_per_doc: dict[str, int] | dict[str, list],
) -> Path:
    """Write ``corpus_manifest.json`` in the architecture 4.2 shape and return its path."""
    manifest = {
        "corpus_version": config.COLLECTION_NAME,
        "generated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "embedding_model": config.EMBEDDING_MODEL,
        "chunking": {
            "strategy": "recursive_heading_aware",
            "chunk_tokens": config.CHUNK_SIZE_TOKENS,
            "overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
            "preserve_tables": True,
        },
        "documents": [
            {
                "doc_id": document.doc_id,
                "scheme_name": document.scheme_name,
                "category": document.category,
                "plan": document.plan,
                "url": document.url,
                "page_title": document.page_title,
                "fetched_at": document.fetched_at,
                "content_sha256": document.content_sha256,
                "raw_path": document.raw_path,
                "processed_path": document.processed_path,
                "chunk_count": _chunk_count(chunks_per_doc.get(document.doc_id, 0)),
            }
            for document in documents
        ],
        "totals": {
            "documents": len(documents),
            "chunks": sum(
                _chunk_count(chunks_per_doc.get(document.doc_id, 0))
                for document in documents
            ),
        },
    }
    path = Path(config.MANIFEST_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_pipeline(
    sources: list[config.SourceSpec] | None = None,
    force: bool = False,
    reset: bool = False,
) -> PipelineResult:
    """Run stages 1 to 5: load, clean, chunk, embed, store, prune, and record the manifest.

    Upserts are keyed by ``chunk_id`` and stale ids are pruned afterwards, so two runs on
    unchanged HTML leave the collection with an identical id set (FR-05).
    """
    chunking = _chunking()
    documents = build_corpus(sources, force=force)
    chunks: list[chunking.Chunk] = []
    chunks_per_doc: dict[str, list[chunking.Chunk]] = {}
    for document in documents:
        document_chunks = chunking.split_text(document.markdown, document)
        chunks_per_doc[document.doc_id] = document_chunks
        chunks.extend(document_chunks)

    if reset:
        vectorstore.reset_collection()
    vectors = embed.embed_texts([chunk.text for chunk in chunks])
    stored = vectorstore.upsert_chunks(chunks, vectors)
    pruned = vectorstore.prune_stale({chunk.chunk_id for chunk in chunks})
    manifest_path = write_manifest(documents, chunks_per_doc)
    return PipelineResult(
        documents=documents,
        chunks=chunks,
        stored=stored,
        pruned=pruned,
        manifest_path=manifest_path,
    )


def _print_pipeline_summary(
    result: PipelineResult,
    cached_by_id: dict[str, bool],
) -> None:
    chunks = result.chunks
    token_counts = [chunk.token_count for chunk in chunks]
    median_tokens = int(statistics.median(token_counts)) if token_counts else 0
    header = (
        f"{'scheme':<34}{'chars':>8}{'cached':>8}{'chunks':>8}{'median':>8}"
        f"  {'doc_id':<14}sha256"
    )
    print(header)
    print("-" * (len(header) + 12))
    by_doc: dict[str, list] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)
    for document in result.documents:
        document_chunks = by_doc.get(document.doc_id, [])
        median = (
            int(statistics.median([c.token_count for c in document_chunks]))
            if document_chunks
            else 0
        )
        print(
            f"{document.scheme_name:<34}{len(document.markdown):>8}"
            f"{'yes' if cached_by_id.get(document.doc_id) else 'no':>8}"
            f"{len(document_chunks):>8}{median:>8}  {document.doc_id:<14}"
            f"{document.content_sha256[:8]}"
        )
    print("-" * (len(header) + 12))
    print(
        f"{'TOTAL':<34}{sum(len(d.markdown) for d in result.documents):>8}{'':>8}"
        f"{len(chunks):>8}{median_tokens:>8}"
    )
    print(f"\ndocuments    {len(result.documents)}/{len(config.CORPUS)}")
    print(f"chunks       {len(chunks)} (median {median_tokens} tokens)")
    print(f"stored       {result.stored} upserted, {result.pruned} stale pruned")
    print(f"collection   {config.COLLECTION_NAME} @ {config.CHROMA_DIR}")
    print(f"manifest     {result.manifest_path}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run stages 1-5, or report what is already stored."""
    parser = argparse.ArgumentParser(description="Ingest the HDFC scheme corpus (stages 1-5).")
    parser.add_argument("--force", action="store_true", help="ignore the raw HTML cache")
    parser.add_argument("--url", help="ingest a single allowlisted URL")
    parser.add_argument(
        "--reset", action="store_true", help="drop the collection and rebuild it from scratch"
    )
    parser.add_argument("--stats", action="store_true", help="print vector store statistics")
    args = parser.parse_args(argv)

    log = logger.configure_logging()
    if args.stats:
        return vectorstore.main([])

    chunking = _chunking()
    sources = config.CORPUS
    if args.url:
        try:
            sources = [_spec_for_url(args.url)]
        except OutOfScopeError as error:
            print(f"ingest failed: {error}", file=sys.stderr)
            return 1
    cached_by_id = {
        doc_id_for(spec.url): _was_cached(doc_id_for(spec.url), args.force)
        for spec in sources
    }

    try:
        result = run_pipeline(list(sources), force=args.force, reset=args.reset)
    except (
        IngestError,
        OutOfScopeError,
        chunking.ChunkingError,
        embed.EmbeddingError,
        vectorstore.VectorStoreError,
    ) as error:
        print(f"ingest failed: {error}", file=sys.stderr)
        return 1

    _print_pipeline_summary(result, cached_by_id)
    log.info(
        "ingested %d/%d pages into %d chunks (%d pruned)",
        len(result.documents),
        len(config.CORPUS),
        len(result.chunks),
        result.pruned,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

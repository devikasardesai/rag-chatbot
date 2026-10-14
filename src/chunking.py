"""Stage 3 chunking: heading-aware recursive splitting with atomic fact protection."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.ingest import PageDocument, doc_id_for

HEADING_RE = re.compile(r"^(#{1,6})\s+(\S.*?)\s*$")
TABLE_ROW_RE = re.compile(r"^\s*\|")
TABLE_DIVIDER_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
BOLD_LABEL_RE = re.compile(r"^\s{0,3}\*\*[^*\n]{1,60}:\*\*\s*\S")
PLAIN_LABEL_RE = re.compile(r"^\s{0,3}[^*\n|]{1,60}:\s+\S")
RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " "]
DEFAULT_SECTION_TITLE = "Overview"
FALLBACK_MODEL_MAX_TOKENS = 256
RESERVED_SPECIAL_TOKENS = 2
MIN_BODY_BUDGET_CHARS = 400
MAX_TIGHTEN_PASSES = 8
TIGHTEN_SHRINK = 0.9
DENSE_CHARS_PER_TOKEN = 2.5


@lru_cache(maxsize=1)
def model_max_tokens() -> int:
    """Content tokens the embedder can actually consume for one chunk.

    Must be read from the ``SentenceTransformer`` object, not from
    ``AutoTokenizer.model_max_length``: for all-MiniLM-L6-v2 the tokenizer declares 512
    (the underlying BERT config) while the sentence-transformers wrapper truncates at
    its own ``max_seq_length`` of 256. Trusting the tokenizer silently drops the tail of
    every longer chunk before it is embedded.
    """
    try:
        from src import embed

        limit = int(getattr(embed.get_model(), "max_seq_length", 0) or 0)
    except Exception:
        return FALLBACK_MODEL_MAX_TOKENS
    return limit if limit > RESERVED_SPECIAL_TOKENS else FALLBACK_MODEL_MAX_TOKENS


def safe_chunk_tokens() -> int:
    """Per-chunk token ceiling, leaving room for the [CLS] and [SEP] tokens."""
    return model_max_tokens() - RESERVED_SPECIAL_TOKENS


class ChunkingError(RuntimeError):
    """Raised when the corpus cannot be loaded or chunked."""


@dataclass(frozen=True)
class Section:
    """One Markdown section: everything under a heading up to the next heading."""

    title: str
    level: int
    body: str


@dataclass(frozen=True)
class Chunk:
    """One embeddable unit plus the metadata persisted alongside it."""

    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    token_count: int
    metadata: dict = field(default_factory=dict)


def _splitter(budget: int | None = None) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=budget if budget is not None else config.CHUNK_CHARS,
        chunk_overlap=config.CHUNK_OVERLAP_CHARS,
        separators=RECURSIVE_SEPARATORS,
        keep_separator=True,
    )


def _document_title(markdown: str) -> str:
    for line in markdown.splitlines():
        match = HEADING_RE.match(line)
        if match and len(match.group(1)) == 1:
            return match.group(2)
    return DEFAULT_SECTION_TITLE


def parse_sections(markdown: str) -> list[Section]:
    """Split Markdown into sections on ATX headings.

    Text before the first heading becomes a section titled with the page title, and
    heading levels may skip (an H1 may be followed by an H3).
    """
    lines = markdown.splitlines()
    sections: list[Section] = []
    buffer: list[str] = []
    current_title: str | None = None
    current_level = 0
    started = False

    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            body = "\n".join(buffer).strip()
            if current_title is None:
                if body:
                    sections.append(Section(DEFAULT_SECTION_TITLE, 0, body))
            else:
                sections.append(Section(current_title, current_level, body))
            current_title, current_level, buffer = match.group(2), len(match.group(1)), []
            started = True
        else:
            buffer.append(line)

    body = "\n".join(buffer).strip()
    if current_title is None:
        if body:
            sections.append(Section(DEFAULT_SECTION_TITLE, 0, body))
    else:
        sections.append(Section(current_title, current_level, body))
    if not started and not sections:
        raise ChunkingError("markdown contains no sections to chunk")
    return [section for section in sections if section.body or section.title]


def provenance_header(doc: PageDocument, section: str) -> str:
    """Breadcrumb prepended to every chunk before embedding (FR-09)."""
    return (
        f"Scheme: {doc.scheme_name} | Plan: {doc.plan} | Category: {doc.category} "
        f"| Section: {section} | Source: {doc.url}"
    )


PROVENANCE_LINE_RE = re.compile(r"^Scheme:.*\|\s*Source:", re.IGNORECASE)


def strip_provenance(text: str) -> str:
    """Return the chunk body with the ``provenance_header`` line removed.

    Stored chunks are always built as ``f"{header}\\n{body}"``, so the breadcrumb is the
    first line only when it is present. Anything else is returned untouched, which keeps
    this safe on already-stripped bodies and on fixture text.
    """
    raw = str(text or "")
    if "\n" not in raw:
        return raw
    first, rest = raw.split("\n", 1)
    return rest if PROVENANCE_LINE_RE.match(first.strip()) else raw


def _is_label_value(line: str) -> bool:
    return bool(BOLD_LABEL_RE.match(line) or PLAIN_LABEL_RE.match(line))


def _atomic_units(body: str) -> list[str]:
    """Group a section body into units that must never be split.

    A unit is a whole table (header, divider, and every row), a label/value line,
    or a paragraph.
    """
    units: list[str] = []
    table: list[str] = []
    paragraph: list[str] = []

    def flush_table() -> None:
        if table:
            units.append("\n".join(table).strip())
            table.clear()

    def flush_paragraph() -> None:
        if paragraph:
            text = "\n".join(paragraph).strip()
            if text:
                units.append(text)
            paragraph.clear()

    for line in body.splitlines():
        if TABLE_ROW_RE.match(line) or (table and not line.strip()):
            if table and not line.strip() and not TABLE_ROW_RE.match(line):
                continue
            table.append(line)
            continue
        if _is_label_value(line):
            flush_table()
            flush_paragraph()
            units.append(line.strip())
            continue
        flush_table()
        if line.strip():
            paragraph.append(line)
        else:
            flush_paragraph()
    flush_table()
    flush_paragraph()
    return units


def _split_table(table: str, budget: int) -> list[str]:
    """Split an oversized table by whole rows, repeating the header rows."""
    rows = [row for row in table.splitlines() if row.strip()]
    preamble = rows[:2] if len(rows) > 2 and TABLE_DIVIDER_RE.match(rows[1]) else []
    body_rows = rows[len(preamble) :]
    groups: list[str] = []
    current: list[str] = []
    current_len = 0
    for row in body_rows:
        addition = len(row) + (1 if current else 0)
        if current and current_len + addition > budget:
            groups.append("\n".join(current))
            current, current_len = [], 0
            addition = len(row)
        current.append(row)
        current_len += addition
    if current:
        groups.append("\n".join(current))
    if preamble:
        groups = ["\n".join(preamble + [group]) for group in groups]
    return groups


def _expand_oversized(units: list[str], budget: int) -> list[str]:
    """Split any single unit larger than the chunk budget without breaking facts."""
    expanded: list[str] = []
    for unit in units:
        if len(unit) <= budget:
            expanded.append(unit)
            continue
        if TABLE_ROW_RE.match(unit) and "\n" in unit:
            expanded.extend(_split_table(unit, budget))
        else:
            expanded.extend(
                piece for piece in _splitter(budget).split_text(unit) if piece.strip()
            )
    return expanded


def _pack(units: list[str], budget: int) -> list[str]:
    """Greedily pack whole units into chunks, carrying whole-unit overlap."""
    packed: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        addition = len(unit) + (1 if current else 0)
        if current and current_len + addition > budget:
            packed.append("\n".join(current))
            overlap: list[str] = []
            overlap_len = 0
            for previous in reversed(current):
                if overlap_len + len(previous) > config.CHUNK_OVERLAP_CHARS:
                    break
                overlap.insert(0, previous)
                overlap_len += len(previous) + 1
            if overlap_len + addition > budget:
                overlap, overlap_len = [], 0
            current = list(overlap)
            current_len = overlap_len
            addition = len(unit) + (1 if current else 0)
        current.append(unit)
        current_len += addition

    if current:
        packed.append("\n".join(current))
    return packed


@lru_cache(maxsize=1)
@lru_cache(maxsize=1)
def _load_tokenizer() -> tuple[object | None, str]:
    try:
        from transformers import AutoTokenizer
    except Exception as error:
        return None, f"transformers unavailable: {type(error).__name__}"
    try:
        return AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL), config.EMBEDDING_MODEL
    except Exception as error:
        return None, f"{config.EMBEDDING_MODEL} not loadable: {type(error).__name__}"


def measure_tokens(text: str) -> int | None:
    """Count tokens with the MiniLM tokenizer, or None when it is unavailable."""
    tokenizer, _ = _load_tokenizer()
    if tokenizer is None:
        return None
    return len(tokenizer.encode(text, add_special_tokens=False, verbose=False))


def estimate_tokens(text: str, chars_per_token: int | float | None = None) -> int:
    """Offline token estimate at the configured characters-per-token ratio."""
    return max(1, int(len(text) // (chars_per_token or config.CHARS_PER_TOKEN)))


def _keep_chunk(body: str) -> bool:
    if not body.strip():
        return False
    if any(character.isdigit() for character in body):
        return True
    return estimate_tokens(body) >= config.MIN_CHUNK_TOKENS


def _section_path(stack: list[tuple[int, str]]) -> str:
    return " > ".join(title for _, title in stack)


def _token_count(text: str) -> int:
    """Measured token count, or a dense-content estimate when the tokenizer is unavailable."""
    measured = measure_tokens(text)
    return measured if measured is not None else estimate_tokens(text, DENSE_CHARS_PER_TOKEN)


def _fits_model_limit(text: str) -> bool:
    """True only when the chunk is known to fit, or conservatively assumed to.

    An unmeasurable chunk is never trusted: the embedding window is a hard limit and
    exceeding it silently drops the tail, so fall back to a dense-content estimate.
    """
    count = measure_tokens(text)
    if count is None:
        count = estimate_tokens(text, DENSE_CHARS_PER_TOKEN)
    return count <= safe_chunk_tokens()


def _tighter_budget(header: str, bodies: list[str], budget: int) -> int:
    """Shrink the character budget until the measured chunks fit the encoder window.

    The window is a hard constraint, so this may go below ``MIN_BODY_BUDGET_CHARS``:
    a small chunk is merely less efficient, a truncated one silently loses its tail.
    """
    limit = safe_chunk_tokens()
    measured = [
        (len(header + body), count)
        for body, count in ((body, measure_tokens(header + body)) for body in bodies)
        if count
    ]
    if measured:
        ratio = min(characters / count for characters, count in measured)
    else:
        ratio = DENSE_CHARS_PER_TOKEN
    return min(budget, max(1, int(limit * ratio) - len(header)))


def _section_bodies(section: Section, header: str) -> list[str]:
    """Chunk one section, tightening the budget until every chunk fits the model limit."""
    budget = max(
        MIN_BODY_BUDGET_CHARS,
        config.CHUNK_CHARS - len(header) - config.CHUNK_OVERLAP_CHARS,
    )
    units = _atomic_units(section.body)
    bodies = _pack(_expand_oversized(units, budget), budget)
    for _ in range(MAX_TIGHTEN_PASSES):
        if all(_fits_model_limit(header + body) for body in bodies):
            return bodies
        tighter = _tighter_budget(header, bodies, budget)
        if tighter >= budget:
            tighter = max(1, int(budget * TIGHTEN_SHRINK))
        if tighter >= budget:
            return bodies
        budget = tighter
        bodies = _pack(_expand_oversized(units, budget), budget)
    return bodies


def split_text(markdown: str, doc: PageDocument) -> list[Chunk]:
    """Split a cleaned page into embeddable chunks, one section at a time."""
    sections = parse_sections(markdown)
    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []
    ordinal = 0

    for section in sections:
        if section.level == 0:
            stack.clear()
            path = section.title
        else:
            while stack and stack[-1][0] >= section.level:
                stack.pop()
            stack.append((section.level, section.title))
            path = _section_path(stack)
        header = provenance_header(doc, path)
        bodies = _section_bodies(section, header)
        kept = [body for body in bodies if _keep_chunk(body)] or bodies
        for body in kept:
            text = f"{header}\n{body}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{ordinal:03d}",
                    doc_id=doc.doc_id,
                    ordinal=ordinal,
                    text=text,
                    token_count=_token_count(text),
                    metadata={
                        "source_url": doc.url,
                        "scheme_name": doc.scheme_name,
                        "category": doc.category,
                        "plan": doc.plan,
                        "section": section.title,
                        "section_path": path,
                        "page_title": doc.page_title,
                        "fetched_at": doc.fetched_at,
                        "chunk_ordinal": ordinal,
                    },
                )
            )
            ordinal += 1
    return chunks


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def validate_fact_integrity(
    chunks: list[Chunk], probes: dict[str, list[str]] | None = None
) -> list[str]:
    """Report every must-have fact that is absent from all of a scheme's chunks."""
    probe_map = config.MUST_HAVE_FACTS if probes is None else probes
    haystacks: dict[str, str] = {}
    for chunk in chunks:
        scheme = str(chunk.metadata.get("scheme_name", ""))
        haystacks[scheme] = haystacks.get(scheme, "") + " " + chunk.text

    failures: list[str] = []
    for scheme, probe_list in probe_map.items():
        haystack = _normalise(haystacks.get(scheme, ""))
        for probe in probe_list:
            if _normalise(probe) not in haystack:
                failures.append(f"{scheme}: probe {probe!r} not found in any chunk")
    return failures


def load_documents() -> list[PageDocument]:
    """Rebuild PageDocument objects from the Markdown written by stage 1-2."""
    documents: list[PageDocument] = []
    for spec in config.CORPUS:
        doc_id = doc_id_for(spec.url)
        processed_path = config.PROCESSED_DIR / f"{doc_id}.md"
        if not processed_path.exists():
            raise ChunkingError(
                f"missing {processed_path}; run 'python -m src.ingest' before chunking"
            )
        payload = processed_path.read_bytes()
        documents.append(
            PageDocument(
                doc_id=doc_id,
                url=spec.url,
                scheme_name=spec.scheme_name,
                category=spec.category,
                plan=spec.plan,
                page_title=spec.scheme_name,
                markdown=payload.decode("utf-8"),
                fetched_at=datetime.fromtimestamp(
                    processed_path.stat().st_mtime, tz=timezone.utc
                ).date().isoformat(),
                content_sha256=hashlib.sha256(payload).hexdigest(),
                raw_path=str(config.RAW_DIR / f"{doc_id}.html"),
                processed_path=str(processed_path),
            )
        )
    return documents


def build_report(chunks_by_doc: dict[str, list[Chunk]]) -> dict:
    """Summarise chunk sizes, section coverage, and fact integrity."""
    all_chunks = [chunk for chunks in chunks_by_doc.values() for chunk in chunks]
    tokenizer, tokenizer_detail = _load_tokenizer()
    measured = tokenizer is not None

    def _stats(chunks: list[Chunk]) -> dict:
        counts = [chunk.token_count for chunk in chunks]
        drift = [
            chunk.token_count / (len(chunk.text) / config.CHARS_PER_TOKEN) - 1.0
            for chunk in chunks
            if chunk.text
        ]
        return {
            "chunks": len(chunks),
            "min_tokens": min(counts) if counts else 0,
            "median_tokens": int(statistics.median(counts)) if counts else 0,
            "max_tokens": max(counts) if counts else 0,
            "mean_tokens": int(statistics.fmean(counts)) if counts else 0,
            "median_char_to_token_drift": round(statistics.median(drift), 4) if drift else 0.0,
        }

    documents = []
    for doc_id, chunks in chunks_by_doc.items():
        histogram: dict[str, int] = {}
        for chunk in chunks:
            key = str(chunk.metadata.get("section_path", ""))
            histogram[key] = histogram.get(key, 0) + 1
        entry = {"doc_id": doc_id, "scheme_name": chunks[0].metadata["scheme_name"] if chunks else ""}
        entry.update(_stats(chunks))
        entry["sections"] = len(histogram)
        entry["section_histogram"] = dict(sorted(histogram.items()))
        documents.append(entry)

    totals = _stats(all_chunks)
    median = totals["median_tokens"]
    maximum = totals["max_tokens"]
    target = config.CHUNK_SIZE_TOKENS
    deviation = round((median - target) / target, 4) if target else 0.0
    overshoot = round((maximum - target) / target, 4) if target else 0.0
    failures = validate_fact_integrity(all_chunks)
    size_note = ""
    if median < target * 0.75:
        size_note = (
            f"median {median} tokens is below the {target}-token budget because most page "
            f"sections are smaller than the budget; chunk_size is an upper bound, not a fill target"
        )
    rechunk_warning = (
        f"largest chunk is {maximum} tokens ({overshoot:+.0%} vs target {target}) "
        f"-> lower CHUNK_CHARS in src/config.py"
        if overshoot > 0.25
        else ""
    )
    limit = safe_chunk_tokens()
    over_limit = sum(1 for chunk in all_chunks if chunk.token_count > limit)
    truncation = {
        "model_max_tokens": model_max_tokens(),
        "safe_chunk_tokens": limit,
        "source": "SentenceTransformer.max_seq_length",
        "chunks_over_limit": over_limit,
        "chunks_total": len(all_chunks),
        "tokens_lost_if_truncated": sum(
            chunk.token_count - limit for chunk in all_chunks if chunk.token_count > limit
        ),
        "warning": (
            f"{over_limit}/{len(all_chunks)} chunks exceed the {limit}-token encoder window; "
            f"their tails are discarded before embedding -> lower CHUNK_CHARS in src/config.py"
        )
        if over_limit
        else "",
    }

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "corpus_version": config.COLLECTION_NAME,
        "chunking": {
            "strategy": "recursive_heading_aware",
            "chunk_chars": config.CHUNK_CHARS,
            "chunk_overlap_chars": config.CHUNK_OVERLAP_CHARS,
            "chunk_size_tokens_target": config.CHUNK_SIZE_TOKENS,
            "chunk_overlap_tokens_target": config.CHUNK_OVERLAP_TOKENS,
            "chars_per_token": config.CHARS_PER_TOKEN,
            "min_chunk_tokens": config.MIN_CHUNK_TOKENS,
            "separators": RECURSIVE_SEPARATORS,
            "overlap_scope": "within_section",
        },
        "tokenizer": {
            "name": config.EMBEDDING_MODEL,
            "mode": "measured" if measured else "estimate",
            "detail": config.EMBEDDING_MODEL if measured else tokenizer_detail,
        },
        "documents": documents,
        "totals": {"documents": len(chunks_by_doc), **totals},
        "median_deviation_from_target": deviation,
        "max_deviation_from_target": overshoot,
        "size_note": size_note,
        "rechunk_warning": rechunk_warning,
        "truncation": truncation,
        "fact_integrity_failures": failures,
    }


def write_report(report: dict) -> Path:
    """Write the chunking report to logs/chunking_report.json."""
    path = Path(config.CHUNKING_REPORT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _print_summary(report: dict) -> None:
    print(f"corpus {report['corpus_version']} | strategy {report['chunking']['strategy']}")
    print(
        f"chunk_size={report['chunking']['chunk_chars']} chars "
        f"(target {report['chunking']['chunk_size_tokens_target']} tokens), "
        f"overlap={report['chunking']['chunk_overlap_chars']} chars, "
        f"tokenizer mode={report['tokenizer']['mode']}"
    )
    header = f"{'scheme':<34}{'chunks':>7}{'min':>6}{'median':>8}{'max':>6}{'sections':>10}{'drift':>9}"
    print(header)
    print("-" * len(header))
    for entry in report["documents"]:
        print(
            f"{entry['scheme_name']:<34}{entry['chunks']:>7}{entry['min_tokens']:>6}"
            f"{entry['median_tokens']:>8}{entry['max_tokens']:>6}{entry['sections']:>10}"
            f"{entry['median_char_to_token_drift']:>+9.2f}"
        )
    totals = report["totals"]
    print("-" * len(header))
    print(
        f"{'TOTAL':<34}{totals['chunks']:>7}{totals['min_tokens']:>6}"
        f"{totals['median_tokens']:>8}{totals['max_tokens']:>6}"
    )
    print(
        f"\nmedian {totals['median_tokens']} tokens "
        f"({report['median_deviation_from_target']:+.0%} vs target "
        f"{report['chunking']['chunk_size_tokens_target']}), "
        f"max {totals['max_tokens']} tokens "
        f"({report['max_deviation_from_target']:+.0%})"
    )
    if report["size_note"]:
        print(f"note: {report['size_note']}")
    if report["rechunk_warning"]:
        print(f"WARNING: {report['rechunk_warning']}")
    truncation = report["truncation"]
    print(
        f"\nencoder window   {truncation['model_max_tokens']} tokens "
        f"({truncation['source']}), {truncation['safe_chunk_tokens']} usable per chunk"
    )
    if truncation["warning"]:
        print(f"WARNING: {truncation['warning']}")
    else:
        print(
            f"truncation       0/{truncation['chunks_total']} chunks exceed the encoder window"
        )
    failures = report["fact_integrity_failures"]
    print(f"\nfact integrity: {len(failures)} failure(s)")
    for failure in failures:
        print(f"  - {failure}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: chunk every processed page and optionally write the report."""
    parser = argparse.ArgumentParser(description="Chunk the cleaned corpus (stage 3).")
    parser.add_argument("--report", action="store_true", help="write logs/chunking_report.json")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when a must-have fact is missing from a scheme's chunks",
    )
    args = parser.parse_args(argv)

    try:
        documents = load_documents()
    except ChunkingError as error:
        print(f"chunking failed: {error}")
        return 1

    chunks_by_doc = {document.doc_id: split_text(document.markdown, document) for document in documents}
    report = build_report(chunks_by_doc)
    _print_summary(report)
    if args.report:
        print(f"\nreport: {write_report(report)}")
    if report["fact_integrity_failures"] and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

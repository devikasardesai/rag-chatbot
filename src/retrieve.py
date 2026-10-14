"""Stages 6-9 retrieval: query normalization, thresholded search, allowlist, MMR, context.

This module is deliberately free of any LLM import. It turns a question into a small,
diverse, in-scope context set and nothing else; generation lives in later phases.
"""
from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable, Sequence
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from src import chunking, config, embed, guardrails, logger, vectorstore

FILLER_PREFIXES: tuple[str, ...] = (
    "what is the",
    "what are the",
    "what is",
    "what are",
    "tell me about",
    "tell me the",
    "can you tell me about",
    "please tell me about",
    "explain",
    "describe",
)
FILLER_RE = re.compile(
    r"^(?:" + "|".join(re.escape(prefix) for prefix in FILLER_PREFIXES) + r")(?:\s+|$)"
)
WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[a-z0-9.]+")
KEPT_PUNCTUATION = frozenset("?%")
NEAR_DUPLICATE_CHARS = 120
MIN_CONTEXT_BLOCKS = 2

# Function words carry no topical signal, so they never contribute to the lexical half of
# the hybrid score. Scheme names are deliberately kept: naming the fund the user asked
# about is exactly the signal that separates two schemes' identically-worded sections.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the of for to and or is are was were be been being with from by on at in into
    over under about what which who whom whose when where why how do does did i you he she
    it we they my your his her its our their this that these those please tell give show
    me can could would should shall will than then so as if not no yes any all each per vs
    """.split()
)


class OutOfScopeError(RuntimeError):
    """Raised when a URL outside the corpus allowlist reaches the retrieval boundary."""


class RetrievalError(RuntimeError):
    """Raised when retrieval cannot run at all (missing index, bad encoder output)."""


@dataclass(frozen=True)
class RetrievalResult:
    """The outcome of one retrieval: kept hits, drop counts, context, and latency."""

    question: str
    normalized_question: str
    k: int
    min_score: float
    hits: list[vectorstore.RetrievedChunk]
    context: str
    dropped_low_score: int
    dropped_out_of_scope: int
    context_tokens: int
    latency_ms: int


# --------------------------------------------------------------------------- #
# Stage 6: query normalization
# --------------------------------------------------------------------------- #


def _strip_punctuation(text: str) -> str:
    """Replace punctuation with spaces, keeping ``?``, ``%``, and decimal points.

    A "." between two digits is part of a number ("1.5%"), so it is kept; every other
    mark becomes a space, which the whitespace collapse then tidies up.
    """
    characters: list[str] = []
    for index, character in enumerate(text):
        if character in KEPT_PUNCTUATION or character.isalnum() or character.isspace():
            characters.append(character)
        elif character == "_":
            characters.append(character)
        elif character == "." and 0 < index < len(text) - 1:
            if text[index - 1].isdigit() and text[index + 1].isdigit():
                characters.append(character)
            else:
                characters.append(" ")
        else:
            characters.append(" ")
    return "".join(characters)


def normalize_query(question: str) -> str:
    """Lowercase, drop filler and most punctuation, and collapse whitespace.

    ``?`` and ``%`` survive because they carry meaning ("1%?" and "12.5%?"), and a
    decimal point between digits is preserved so "1.5% exit load" is not mangled into
    "1 5% exit load". The filler prefix is stripped repeatedly so "what is the expense
    ratio of ..." does not leave a dangling "of".
    """
    text = str(question or "").strip().lower()
    if not text:
        return ""
    text = _strip_punctuation(text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    while True:
        stripped = FILLER_RE.sub("", text, count=1).strip()
        if stripped == text:
            return text
        text = stripped


# --------------------------------------------------------------------------- #
# Stage 6: allowlist
# --------------------------------------------------------------------------- #


def _normalize_url(url: str) -> str:
    """Canonicalize a URL for comparison: lowercase host, no fragment, no trailing slash."""
    parts = urlsplit(str(url or "").strip())
    host = (parts.hostname or "").lower()
    if parts.port:
        host = f"{host}:{parts.port}"
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), host, path, parts.query, ""))


def is_allowlisted(url: str) -> bool:
    """True when the URL is one of the five corpus pages (FR-16)."""
    if not str(url or "").strip():
        return False
    target = _normalize_url(url)
    return any(target == _normalize_url(source.url) for source in config.CORPUS)


def assert_allowlisted(urls: Iterable[str]) -> None:
    """Raise :class:`OutOfScopeError` if any URL is outside the corpus allowlist."""
    offenders = sorted({str(url) for url in urls if not is_allowlisted(url)})
    if offenders:
        allowed = ", ".join(sorted(source.url for source in config.CORPUS))
        raise OutOfScopeError(
            f"{len(offenders)} URL(s) outside the corpus allowlist: "
            f"{'; '.join(offenders)}. Allowed: {allowed}"
        )


# --------------------------------------------------------------------------- #
# Stage 7: hybrid re-ranking (cosine + IDF-weighted term overlap)
# --------------------------------------------------------------------------- #


def query_terms(normalized: str) -> list[str]:
    """Content words of a normalized query, de-duplicated, in first-seen order."""
    return [
        word
        for word in dict.fromkeys(WORD_RE.findall(str(normalized or "").lower()))
        if len(word) > 1 and word not in STOPWORDS
    ]


def lexical_overlap(query: str, chunks: Sequence[vectorstore.RetrievedChunk]) -> dict[str, float]:
    """Fraction of the query's IDF weight each candidate chunk actually contains.

    The document frequencies come from the candidate pool rather than the whole corpus,
    because only the survivors of the similarity threshold compete with each other. A term
    present in every candidate gets near-zero weight, so ubiquitous wording cannot outrank
    the rare word that identifies the answer.
    """
    terms = query_terms(query)
    if not terms or not chunks:
        return {chunk.chunk_id: 0.0 for chunk in chunks}

    token_sets = [set(WORD_RE.findall(chunk.text.lower())) for chunk in chunks]
    document_frequency = {
        term: sum(1 for words in token_sets if term in words) for term in terms
    }
    total = len(chunks)
    weights = {
        term: math.log(1 + total / (1 + document_frequency[term])) for term in terms
    }
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        return {chunk.chunk_id: 0.0 for chunk in chunks}
    return {
        chunk.chunk_id: (
            sum(weights[term] for term in terms if term in words) / weight_sum
        )
        for chunk, words in zip(chunks, token_sets)
    }


def hybrid_scores(
    query: str,
    chunks: Sequence[vectorstore.RetrievedChunk],
    alpha: float | None = None,
) -> dict[str, float]:
    """Blend cosine similarity with term overlap: ``(1-α)·cosine + α·overlap``.

    ``alpha`` defaults to ``config.LEXICAL_ALPHA`` read at call time, matching how
    ``diversify`` reads ``config.MMR_LAMBDA``, so retuning takes effect without an import.
    """
    overlap = lexical_overlap(query, chunks)
    blend = float(config.LEXICAL_ALPHA if alpha is None else alpha)
    blend = min(1.0, max(0.0, blend))
    return {
        chunk.chunk_id: (1.0 - blend) * float(chunk.score)
        + blend * overlap.get(chunk.chunk_id, 0.0)
        for chunk in chunks
    }


# --------------------------------------------------------------------------- #
# Stage 8: diversity (dedupe + per-section cap + greedy MMR)
# --------------------------------------------------------------------------- #


def _section_key(chunk: vectorstore.RetrievedChunk) -> tuple[str, str]:
    metadata = chunk.metadata or {}
    scheme = str(metadata.get("scheme_name", "") or "")
    section = str(metadata.get("section", "") or metadata.get("section_path", "") or "")
    return (scheme, section)


def _near_duplicate_key(chunk: vectorstore.RetrievedChunk) -> str:
    """Key on the chunk body alone, ignoring the provenance header line.

    Every stored chunk begins with ``Scheme: … | Plan: … | Section: … | Source: …``,
    which runs 200-300 characters and is identical for most chunks of a scheme. Hashing
    the raw prefix would therefore collapse unrelated sections of the same fund, so the
    first line is dropped before the similarity key is taken.
    """
    body = chunking.strip_provenance(chunk.text)
    return WHITESPACE_RE.sub(" ", body.lower()).strip()[:NEAR_DUPLICATE_CHARS]


def _text_similarity(left: list[str], right: list[str]) -> np.ndarray:
    """Cosine similarity between two text batches, using the shared MiniLM encoder."""
    if not left or not right:
        return np.zeros((len(left), len(right)), dtype=np.float32)
    vectors = np.asarray(embed.embed_texts(list(left) + list(right)), dtype=np.float32)
    split = len(left)
    return np.asarray(vectors[:split] @ vectors[split:].T, dtype=np.float32)


def _dedupe(chunks: Sequence[vectorstore.RetrievedChunk], per_section_cap: int) -> list[vectorstore.RetrievedChunk]:
    """Drop repeated ids, near-identical text, and sections over the per-section cap."""
    kept: list[vectorstore.RetrievedChunk] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    per_section: dict[tuple[str, str], int] = {}
    for chunk in chunks:
        if chunk.chunk_id in seen_ids:
            continue
        text_key = _near_duplicate_key(chunk)
        if text_key and text_key in seen_text:
            continue
        key = _section_key(chunk)
        used = per_section.get(key, 0)
        if used >= per_section_cap:
            continue
        seen_ids.add(chunk.chunk_id)
        if text_key:
            seen_text.add(text_key)
        per_section[key] = used + 1
        kept.append(chunk)
    return kept


def diversify(
    chunks: Sequence[vectorstore.RetrievedChunk],
    per_section_cap: int = config.MAX_CHUNKS_PER_SECTION,
    k: int = config.TOP_K,
    similarity: Callable[[list[str], list[str]], np.ndarray] | None = None,
    relevance: Callable[[vectorstore.RetrievedChunk], float] | None = None,
) -> list[vectorstore.RetrievedChunk]:
    """Dedupe, cap per section, then greedily order by MMR with ``config.MMR_LAMBDA``.

    The first pick is the most relevant chunk. By default relevance is ``chunk.score``
    (cosine similarity); ``retrieve`` passes the hybrid score instead, so the ranking that
    picked the candidates is also the ranking that orders them. After the first pick,
    relevance is traded against novelty: ``score = λ·rel(chunk) − (1−λ)·max sim(chunk, selected)``.
    """
    candidates = _dedupe(chunks, max(1, int(per_section_cap)))
    if len(candidates) <= 1:
        return list(candidates[: max(1, int(k))])

    measure = similarity or _text_similarity
    rank = relevance or (lambda chunk: float(chunk.score))
    lam = config.MMR_LAMBDA
    relevance_values = [float(rank(chunk)) for chunk in candidates]
    selected: list[int] = [max(range(len(candidates)), key=lambda index: relevance_values[index])]
    remaining = [index for index in range(len(candidates)) if index not in selected]

    while remaining and len(selected) < max(1, int(k)):
        pending_texts = [candidates[index].text for index in remaining]
        chosen_texts = [candidates[index].text for index in selected]
        similarity_matrix = np.asarray(
            measure(pending_texts, chosen_texts), dtype=np.float32
        ).reshape(len(remaining), len(selected))
        best_position, best_value = 0, -np.inf
        for position, index in enumerate(remaining):
            novelty = float(np.max(similarity_matrix[position]))
            value = lam * relevance_values[index] - (1.0 - lam) * novelty
            if value > best_value:
                best_position, best_value = position, value
        chosen = remaining.pop(best_position)
        selected.append(chosen)

    return [candidates[index] for index in selected]


# --------------------------------------------------------------------------- #
# Stage 9: context assembly
# --------------------------------------------------------------------------- #


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) // config.CHARS_PER_TOKEN))


def _provenance_line(number: int, chunk: vectorstore.RetrievedChunk) -> str:
    metadata = chunk.metadata or {}
    scheme = metadata.get("scheme_name", "") or "Unknown scheme"
    section = metadata.get("section", "") or metadata.get("section_path", "") or "Unknown section"
    source = metadata.get("source_url", "") or ""
    return f"[{number}] Scheme: {scheme} | Section: {section} | Source: {source}"


def build_context(chunks: Sequence[vectorstore.RetrievedChunk]) -> str:
    """Assemble numbered, provenance-tagged blocks within ``config.MAX_CONTEXT_TOKENS``.

    Over-budget blocks are dropped lowest-score-first, but never below
    ``MIN_CONTEXT_BLOCKS`` blocks: a single grounded block still beats an empty context.
    """
    if not chunks:
        return ""
    ranked = sorted(chunks, key=lambda chunk: float(chunk.score), reverse=True)
    kept = list(ranked)
    while len(kept) > MIN_CONTEXT_BLOCKS and _context_tokens(kept) > config.MAX_CONTEXT_TOKENS:
        kept.pop()
    blocks = [
        f"{_provenance_line(number, chunk)}\n{chunk.text}"
        for number, chunk in enumerate(kept, start=1)
    ]
    return "\n\n".join(blocks)


def _context_tokens(chunks: Sequence[vectorstore.RetrievedChunk]) -> int:
    return sum(_estimate_tokens(_provenance_line(index, chunk) + "\n" + chunk.text)
               for index, chunk in enumerate(chunks, start=1))


# --------------------------------------------------------------------------- #
# Stages 6-9: orchestration
# --------------------------------------------------------------------------- #


def retrieve(
    question: str,
    k: int = config.TOP_K,
    min_score: float = config.MIN_SIMILARITY,
) -> RetrievalResult:
    """Normalize, embed, search, threshold, allowlist, re-rank, diversify, assemble.

    ``min_score`` is cosine similarity, not Chroma distance (``vectorstore.search``
    already returns ``1 - distance``). It is applied to a wider candidate pool than ``k``
    so that the hybrid re-rank has something to choose from: pure cosine leaves rare
    one-off facts (an ELSS lock-in badge, a scheme's TER) outside the top five, and no
    threshold can recover a chunk the search never returned. Out-of-scope hits are dropped
    and counted; an :class:`OutOfScopeError` is raised only when *every* surviving hit is
    out of scope, which means the corpus itself is misconfigured rather than one stray
    chunk.
    """
    started = time.perf_counter()
    normalized = normalize_query(question)
    if not normalized:
        raise RetrievalError("question is empty after normalization")

    limit = max(1, int(k))
    try:
        vector = embed.embed_query(normalized)
    except Exception as error:
        raise RetrievalError(f"could not embed the query: {error}") from error
    if len(vector) != config.EMBEDDING_DIM:
        raise RetrievalError(
            f"query vector has dimension {len(vector)}, expected {config.EMBEDDING_DIM}"
        )

    hits = vectorstore.search(vector, k=max(limit, int(config.CANDIDATE_POOL)))
    kept_hits = [chunk for chunk in hits if float(chunk.score) >= float(min_score)]
    dropped_low_score = len(hits) - len(kept_hits)

    in_scope: list[vectorstore.RetrievedChunk] = []
    for chunk in kept_hits:
        source_url = str((chunk.metadata or {}).get("source_url", "") or "")
        if is_allowlisted(source_url):
            in_scope.append(chunk)
    dropped_out_of_scope = len(kept_hits) - len(in_scope)
    if dropped_out_of_scope and not in_scope:
        assert_allowlisted([str((chunk.metadata or {}).get("source_url", "")) for chunk in kept_hits])

    relevance = hybrid_scores(normalized, in_scope)
    ranked = sorted(in_scope, key=lambda chunk: relevance[chunk.chunk_id], reverse=True)
    selected = diversify(
        ranked,
        per_section_cap=config.MAX_CHUNKS_PER_SECTION,
        k=limit,
        relevance=lambda chunk: relevance[chunk.chunk_id],
    )
    context = build_context(selected)
    latency_ms = int(round((time.perf_counter() - started) * 1000))
    return RetrievalResult(
        question=str(question),
        normalized_question=normalized,
        k=limit,
        min_score=float(min_score),
        hits=selected,
        context=context,
        dropped_low_score=dropped_low_score,
        dropped_out_of_scope=dropped_out_of_scope,
        context_tokens=_context_tokens(selected) if selected else 0,
        latency_ms=latency_ms,
    )


def log_result(result: RetrievalResult, intent: str = "fact", pii: bool = False) -> None:
    """Append one redacted JSON line for this retrieval to ``logs/retrieval.jsonl`` (FR-18)."""
    logger.log_retrieval(
        {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "question": result.question,
            "normalized_question": result.normalized_question,
            "intent": intent,
            "pii": pii,
            "k": result.k,
            "min_score": result.min_score,
            "hits": [
                {
                    "chunk_id": chunk.chunk_id,
                    "scheme": (chunk.metadata or {}).get("scheme_name", ""),
                    "section": (chunk.metadata or {}).get("section", ""),
                    "score": round(float(chunk.score), 4),
                }
                for chunk in result.hits
            ],
            "hit_count": len(result.hits),
            "dropped_low_score": result.dropped_low_score,
            "dropped_out_of_scope": result.dropped_out_of_scope,
            "context_tokens": result.context_tokens,
            "retrieval_ms": result.latency_ms,
            "source_url": (result.hits[0].metadata or {}).get("source_url", "") if result.hits else "",
        }
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _print_debug(result: RetrievalResult) -> None:
    print(f"question      {result.question}")
    print(f"normalized    {result.normalized_question}")
    print(f"k={result.k} min_score={result.min_score}")
    print(
        f"dropped       low_score={result.dropped_low_score} "
        f"out_of_scope={result.dropped_out_of_scope}"
    )
    if not result.hits:
        print("\nno chunk passed the similarity threshold")
        return
    print(f"\n{'#':<3}{'score':>7}  {'scheme':<28}{'section':<34}chunk_id")
    print("-" * 100)
    for number, chunk in enumerate(result.hits, start=1):
        metadata = chunk.metadata or {}
        print(
            f"{number:<3}{float(chunk.score):>7.3f}  "
            f"{str(metadata.get('scheme_name', '')):<28}"
            f"{str(metadata.get('section', ''))[:33]:<34}{chunk.chunk_id}"
        )
    print(f"\ncontext ({result.context_tokens} tokens, budget {config.MAX_CONTEXT_TOKENS})")
    print("=" * 100)
    print(result.context)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: retrieve once, print a debug view, and log the retrieval."""
    parser = argparse.ArgumentParser(description="Retrieve context for a question (stages 6-9).")
    parser.add_argument("--question", required=True, help="the user's question")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="chunks to retrieve")
    parser.add_argument(
        "--min-score",
        type=float,
        default=config.MIN_SIMILARITY,
        help="cosine similarity threshold (not Chroma distance)",
    )
    parser.add_argument("--debug", action="store_true", help="print hits, scores, and context")
    args = parser.parse_args(argv)

    try:
        result = retrieve(args.question, k=args.k, min_score=args.min_score)
    except (RetrievalError, vectorstore.VectorStoreError) as error:
        print(f"retrieval failed: {error}")
        return 1
    except OutOfScopeError as error:
        print(f"out-of-scope corpus: {error}")
        return 1

    logger.configure_logging()
    # Observational only: the PII and advice refusals belong to answer_question, not here.
    intent = guardrails.classify_intent(result.question).value
    has_pii = bool(guardrails.detect_pii(result.question))
    log_result(result, intent=intent, pii=has_pii)

    print(f"retrieved {len(result.hits)} chunk(s) in {result.latency_ms} ms")
    if args.debug:
        _print_debug(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

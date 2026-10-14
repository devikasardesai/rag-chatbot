"""Stages 10-13 generation and orchestration: grounded prompt, Groq call, guardrails.

`answer_question` is the only public entry point. It applies the architecture §7.1 check
order exactly, calls the LLM in exactly one place (`generate`), and is side-effect-free
apart from appending to the JSONL logs, which is what makes the Phase 10 eval harness
possible.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time

import httpx

from src import chunking, config, guardrails, logger, retrieve, vectorstore

CHAT_PATH = "/chat/completions"
TRANSPORT_ERRORS = (httpx.TimeoutException, httpx.TransportError)
MAX_QUESTION_CHARS = 2000
TRUNCATED_QUESTION_CHARS = 500
LATIN_RE = re.compile(r"[A-Za-z]")
SELFTEST_MIN_CHUNKS = 100
SELFTEST_QUERY = "What is the exit load of HDFC Large Cap Fund?"
DISCLAIMER_LINE = "Facts only. Not investment advice."
# Deterministic out-of-corpus AMC markers (task 8.5). A question naming another AMC
# cannot be answered from these five HDFC pages, and retrieval alone will happily return
# a same-topic chunk from the wrong fund, so it is refused before any embedding happens.
FOREIGN_AMC_TOKENS: tuple[str, ...] = (
    "axis", "sbi", "icici", "kotak", "nippon", "mirae", "tata", "lic", "bandhan",
    "quant", "motilal", "ppf", "canara", "pnb", "union", "sundaram", "idl", "invesco",
    "franklin", "aditya", "uti", "hsbc", "baroda", "federal", "jupiter", "dhfl",
)


class LLMUnavailable(RuntimeError):
    """Raised when the Groq API cannot be reached or does not answer with HTTP 200."""


def build_prompt(question: str, context: str, last_updated: str) -> list[dict]:
    """Build the ``[system, user]`` message pair described in architecture §7.3.

    ``config.PROMPT_SYSTEM`` is used verbatim as the system message. The question is
    appended exactly once, after the context, so the model never sees it twice.
    """
    user_message = (
        f"CONTEXT:\n{context}\n\n"
        f"{config.LAST_UPDATED_PREFIX} {last_updated}\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": config.PROMPT_SYSTEM},
        {"role": "user", "content": user_message},
    ]


def generate(messages: list[dict]) -> str:
    """POST the messages to Groq and return the trimmed answer text.

    Raises :class:`LLMUnavailable` on a missing key, a connection error, a timeout, a
    non-200 status, or an unreadable body, so the caller can fall back to extraction.
    """
    api_key = config.GROQ_API_KEY
    if not api_key:
        raise LLMUnavailable("GROQ_API_KEY is missing; set it in .env to call Groq")

    url = f"{config.LLM_BASE_URL.rstrip('/')}{CHAT_PATH}"
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "stream": False,
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_NUM_PREDICT,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=config.LLM_TIMEOUT_S) as client:
            response = client.post(url, json=payload, headers=headers)
    except TRANSPORT_ERRORS as error:
        raise LLMUnavailable(
            f"could not reach Groq at {url}: {type(error).__name__}: {error}"
        ) from error

    if response.status_code != 200:
        raise LLMUnavailable(
            f"{url} returned HTTP {response.status_code}: {response.text[:200]}"
        )
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, TypeError, IndexError) as error:
        raise LLMUnavailable(f"unreadable chat response from {url}: {error}") from error
    text = str(content).strip()
    if not text:
        raise LLMUnavailable(f"{url} returned an empty answer body")
    return text


def extract_citation(answer: str, context_urls: set[str]) -> str | None:
    """Return the first URL in the answer that also appears in ``context_urls``.

    The returned value is always the entry from ``context_urls``, never the raw text
    scraped from the answer, so an invented or mangled link cannot leak through. Returns
    ``None`` when the answer cites nothing that was actually supplied.
    """
    allowed = {str(url).strip() for url in context_urls}
    for candidate in guardrails.URL_RE.findall(answer or ""):
        cleaned = candidate.rstrip(".,;:!?)")
        if cleaned in allowed:
            return cleaned
    return None


def ensure_last_updated(answer: str, last_updated: str) -> str:
    """Guarantee a final ``Last updated from sources: <date>`` line.

    Appends the line when it is missing and replaces it when the date differs, so
    calling this twice with the same date leaves the text unchanged.
    """
    date = str(last_updated or "").strip()
    kept = [
        line
        for line in str(answer or "").splitlines()
        if not line.strip().startswith(config.LAST_UPDATED_PREFIX)
    ]
    while kept and not kept[-1].strip():
        kept.pop()
    if date:
        kept.append(f"{config.LAST_UPDATED_PREFIX} {date}")
    return "\n".join(kept).strip()


def extractive_fallback(question: str, chunks: list) -> guardrails.AnswerResult:
    """Answer verbatim from the top chunk when the model is unavailable.

    Takes the leading ``config.MAX_SENTENCES`` sentences of the most relevant chunk,
    cites that chunk's own URL, and stamps its ``fetched_at`` date. The provenance
    breadcrumb that leads every stored chunk is metadata, not prose, so it is stripped
    before the sentences are counted: otherwise the fallback would open on
    ``Scheme: … | Section: …`` and print the source URL twice (FR-21). Nothing is
    paraphrased and no date is invented, so the answer stays grounded and cited.
    """
    if not chunks:
        raise ValueError("extractive_fallback needs at least one retrieved chunk")
    top = chunks[0]
    metadata = dict(getattr(top, "metadata", {}) or {})
    source_url = str(metadata.get("source_url", "") or "") or None
    last_updated = str(metadata.get("fetched_at", "") or "") or None

    body_text = chunking.strip_provenance(getattr(top, "text", "") or "")
    body = guardrails.trim_sentences(body_text, config.MAX_SENTENCES)
    parts = [part for part in (body, source_url) if part]
    answer = ensure_last_updated("\n".join(parts), last_updated or "")
    return guardrails.AnswerResult(
        answer=answer,
        source_url=source_url,
        last_updated=last_updated,
        refused=False,
        refusal_reason=None,
        guardrail_flags=[],
        used_fallback=True,
        top_chunks=list(chunks),
        latency_ms=0,
    )


# --------------------------------------------------------------------------- #
# Stages 12-13: orchestration
# --------------------------------------------------------------------------- #


def _is_non_latin(question: str) -> bool:
    """True when the question has no Latin letters at all, so it cannot be grounded."""
    return not LATIN_RE.search(question or "")


def _names_foreign_amc(question: str) -> bool:
    """True when the question names an asset manager outside the corpus and no HDFC scheme.

    Retrieval matches on topic, not on fund identity, so "Axis Small Cap exit load"
    scores well against the HDFC exit-load section and would otherwise be answered with
    another fund's facts. Comparisons that do mention HDFC are left to the advice route.
    """
    lowered = (question or "").lower()
    if "hdfc" in lowered:
        return False
    return any(re.search(rf"\b{re.escape(token)}\b", lowered) for token in FOREIGN_AMC_TOKENS)


def _prepare_question(question: str) -> tuple[str, bool]:
    """Truncate an over-long question before embedding; >2000 chars is cut to 500."""
    text = str(question or "")
    if len(text) > MAX_QUESTION_CHARS:
        return text[:TRUNCATED_QUESTION_CHARS], True
    return text, False


def _last_updated(hits: list) -> str:
    """The top chunk's own ``fetched_at``; never a synthesised date (FR-18)."""
    if not hits:
        return ""
    return str((hits[0].metadata or {}).get("fetched_at", "") or "").strip()


def _context_urls(hits: list) -> set[str]:
    """Allowlisted source URLs backing this answer, used for the citation check."""
    urls = {str((hit.metadata or {}).get("source_url", "") or "") for hit in hits}
    return {url for url in urls if url in config.ALLOWED_URLS}


def answer_question(question: str) -> guardrails.AnswerResult:
    """Answer one question through the architecture §7.1 pipeline, in order.

    PII pre-filter, then the advice/performance intent route (both deterministic, no LLM),
    then retrieval, then the ``no_context`` refusal, then prompt and generation. A
    generated answer is validated by :func:`guardrails.validate_answer`; if it is still
    invalid after that single repair pass, the extractive fallback is used, so no path
    can emit uncited text.
    """
    started = time.perf_counter()
    raw = str(question or "")
    pii_labels = guardrails.detect_pii(raw)
    intent = guardrails.classify_intent(raw)

    if pii_labels or intent is guardrails.Intent.PII:
        return _refuse("pii", raw, intent.value, bool(pii_labels), started, [])

    if intent is guardrails.Intent.ADVICE:
        reason = "performance" if guardrails.is_performance_intent(raw) else "advice"
        return _refuse(reason, raw, intent.value, False, started, [])

    if not raw.strip():
        return _refuse("no_context", raw, intent.value, False, started, ["empty_question"])

    if _is_non_latin(raw):
        return _refuse("no_context", raw, intent.value, False, started, ["non_latin_script"])

    if _names_foreign_amc(raw):
        return _refuse("no_context", raw, intent.value, False, started, ["out_of_corpus_amc"])

    prepared, truncated = _prepare_question(raw)
    edge_flags = ["question_truncated"] if truncated else []

    try:
        result = retrieve.retrieve(prepared, k=config.TOP_K, min_score=config.MIN_SIMILARITY)
    except (retrieve.RetrievalError, retrieve.OutOfScopeError, vectorstore.VectorStoreError) as error:
        return _refuse("no_context", raw, intent.value, False, started, [f"retrieval_error:{error}"])

    hits = result.hits
    if not hits or not result.context:
        return _refuse("no_context", raw, intent.value, False, started, edge_flags, result)

    last_updated = _last_updated(hits)
    context_urls = _context_urls(hits)
    messages = build_prompt(prepared, result.context, last_updated)

    used_fallback = False
    try:
        candidate = generate(messages)
    except LLMUnavailable:
        candidate = ""
        used_fallback = True

    flags: list[str] = []
    validation_flags: list[str] = []
    if used_fallback:
        fallback = extractive_fallback(prepared, hits)
        candidate, used_fallback = fallback.answer, True
        flags.append("llm_unavailable")
    else:
        ok, validation_flags, repaired = guardrails.validate_answer(
            candidate, context_urls, last_updated
        )
        if not ok:
            fallback = extractive_fallback(prepared, hits)
            candidate, used_fallback = fallback.answer, True
            flags.append("validation_failed")
        else:
            candidate = repaired

    cited = extract_citation(candidate, context_urls)
    if not cited:
        fallback = extractive_fallback(prepared, hits)
        candidate, cited, used_fallback = fallback.answer, fallback.source_url, True
        flags.append("uncited_answer_blocked")

    final = ensure_last_updated(candidate, last_updated)
    # validate_answer reports what was wrong with the model's raw text and its repair pass
    # then fixes some of it. Report only the flags the final answer still exhibits, so the
    # logs never claim a violation the user was not shown.
    still = set(guardrails.surviving_flags(final, cited, context_urls))
    reported = flags + [flag for flag in validation_flags if flag in still]
    result_object = guardrails.AnswerResult(
        answer=final,
        source_url=cited,
        last_updated=last_updated or None,
        refused=False,
        refusal_reason=None,
        guardrail_flags=edge_flags + sorted(set(reported)),
        used_fallback=used_fallback,
        top_chunks=list(hits),
        latency_ms=int(round((time.perf_counter() - started) * 1000)),
    )
    _log(result_object, result, intent.value, bool(pii_labels), prepared, started)
    return result_object


def _refuse(
    reason: str,
    question: str,
    intent: str,
    pii: bool,
    started: float,
    extra_flags: list[str],
    result: "retrieve.RetrievalResult | None" = None,
) -> guardrails.AnswerResult:
    """Build, log, and return a deterministic refusal; the LLM is never called."""
    refused = guardrails.refusal_response(reason)
    latency_ms = int(round((time.perf_counter() - started) * 1000))
    refused = guardrails.AnswerResult(
        answer=refused.answer,
        source_url=refused.source_url,
        last_updated=refused.last_updated,
        refused=True,
        refusal_reason=reason,
        guardrail_flags=list(refused.guardrail_flags) + extra_flags,
        used_fallback=False,
        top_chunks=[],
        latency_ms=latency_ms,
    )
    _log(refused, result, intent, pii, str(question or ""), started)
    return refused


def _log(
    result_object: guardrails.AnswerResult,
    result: "retrieve.RetrievalResult | None",
    intent: str,
    pii: bool,
    question: str,
    started: float,
) -> None:
    """Append one redacted retrieval line and one redacted answer line (FR-18, §7.1)."""
    total_ms = int(round((time.perf_counter() - started) * 1000))
    hits = list(getattr(result_object, "top_chunks", []) or [])
    retrieval_ms = getattr(result, "latency_ms", 0) if result is not None else 0
    generation_ms = max(0, total_ms - retrieval_ms) if result is not None else 0
    logger.log_retrieval(
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "question": question,
            "normalized_question": retrieve.normalize_query(question),
            "intent": intent,
            "pii": pii,
            "k": config.TOP_K,
            "min_score": config.MIN_SIMILARITY,
            "hits": [
                {
                    "chunk_id": hit.chunk_id,
                    "scheme": (hit.metadata or {}).get("scheme_name", ""),
                    "section": (hit.metadata or {}).get("section", ""),
                    "score": round(float(hit.score), 4),
                }
                for hit in hits
            ],
            "hit_count": len(hits),
            "dropped_low_score": getattr(result, "dropped_low_score", 0) if result else 0,
            "dropped_out_of_scope": getattr(result, "dropped_out_of_scope", 0) if result else 0,
            "context_tokens": getattr(result, "context_tokens", 0) if result else 0,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
            "guardrail_flags": list(result_object.guardrail_flags),
            "source_url": result_object.source_url or "",
            "refused": result_object.refused,
        }
    )
    logger.log_answer(
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "question": question,
            "answer": result_object.answer,
            "citation": result_object.source_url,
            "last_updated": result_object.last_updated,
            "refusal_reason": result_object.refusal_reason,
            "guardrail_flags": list(result_object.guardrail_flags),
            "latency_ms": result_object.latency_ms,
            "refused": result_object.refused,
            "used_fallback": result_object.used_fallback,
        }
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _print_result(result_object: guardrails.AnswerResult) -> None:
    print(result_object.answer)
    if result_object.source_url:
        label = "source" if not result_object.refused else "reference"
        print(f"{label}: {result_object.source_url}")
    if result_object.last_updated:
        print(f"{config.LAST_UPDATED_PREFIX} {result_object.last_updated}")
    print(DISCLAIMER_LINE)
    if result_object.guardrail_flags:
        print(f"flags: {', '.join(result_object.guardrail_flags)}")


def _as_dict(result_object: guardrails.AnswerResult) -> dict:
    payload = dataclasses.asdict(result_object)
    payload["top_chunks"] = [
        {
            "chunk_id": hit.chunk_id,
            "score": round(float(hit.score), 4),
            "scheme": (hit.metadata or {}).get("scheme_name", ""),
            "section": (hit.metadata or {}).get("section", ""),
            "source_url": (hit.metadata or {}).get("source_url", ""),
        }
        for hit in result_object.top_chunks
    ]
    return payload


def _run_chat() -> int:
    """Interactive REPL: one answer per line, `exit`/`quit` to stop."""
    print("HDFC Mutual Fund FAQ Assistant - facts only. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit"}:
            return 0
        if not question:
            continue
        print()
        _print_result(answer_question(question))
        print("\n" + "-" * 72 + "\n")


def _selftest_check(label: str, function) -> tuple[str, bool, str]:
    try:
        detail = function() or ""
        return (label, True, str(detail))
    except Exception as error:  # noqa: BLE001 - the table must report any failure
        return (label, False, f"{type(error).__name__}: {error}")


def _check_collection() -> str:
    stats = vectorstore.collection_stats()
    count = int(stats.get("count", 0))
    if count <= 0:
        raise RuntimeError(f"collection {stats.get('collection_name')!r} is empty")
    if count <= SELFTEST_MIN_CHUNKS:
        raise RuntimeError(f"only {count} chunks, expected more than {SELFTEST_MIN_CHUNKS}")
    return f"{count} chunks in {stats.get('collection_name')}"


def _check_dimension() -> str:
    from src import embed

    vector = embed.embed_query("dimension probe")
    if len(vector) != config.EMBEDDING_DIM:
        raise RuntimeError(f"embedding dim {len(vector)}, expected {config.EMBEDDING_DIM}")
    return f"{len(vector)} dimensions"


def _check_groq() -> str:
    started = time.perf_counter()
    try:
        reply = generate([{"role": "user", "content": "ping"}]).strip()
    except LLMUnavailable as error:
        raise RuntimeError(str(error)) from error
    if not reply:
        raise RuntimeError(f"{config.LLM_MODEL} returned an empty reply")
    elapsed = int((time.perf_counter() - started) * 1000)
    return f"{config.LLM_MODEL} replied in {elapsed} ms"


def _check_canned_answer() -> str:
    result_object = answer_question(SELFTEST_QUERY)
    if not result_object.source_url:
        raise RuntimeError("no citation returned")
    if result_object.source_url not in config.ALLOWED_URLS:
        raise RuntimeError(f"citation outside the corpus: {result_object.source_url}")
    if f"{config.LAST_UPDATED_PREFIX}" not in result_object.answer:
        raise RuntimeError("answer has no last-updated line")
    if guardrails.count_sentences(result_object.answer) > config.MAX_SENTENCES:
        raise RuntimeError("answer exceeds the sentence limit")
    detail = f"cited {result_object.source_url.rsplit('/', 1)[-1]}"
    return f"{detail} ({result_object.latency_ms} ms)"


def _selftest() -> int:
    checks = [
        ("collection exists and has chunks", _check_collection),
        ("embedding dimension is 384", _check_dimension),
        ("groq is reachable", _check_groq),
        ("canned query returns a cited answer", _check_canned_answer),
    ]
    results = [_selftest_check(label, function) for label, function in checks]
    width = max(len(label) for label, _, _ in results)
    print(f"{'check'.ljust(width)}  result  detail")
    print("-" * (width + 40))
    for label, passed, detail in results:
        print(f"{label.ljust(width)}  {'PASS' if passed else 'FAIL':<6}  {detail}")
    failures = [label for label, passed, _ in results if not passed]
    print("-" * (width + 40))
    print(f"{len(results) - len(failures)}/{len(results)} passed")
    if failures:
        print("failed: " + ", ".join(failures))
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: one-shot question, JSON output, self-test, or a REPL."""
    parser = argparse.ArgumentParser(description="Answer a question from the indexed corpus.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--question", help="answer a single question and exit")
    group.add_argument("--chat", action="store_true", help="start the interactive REPL")
    group.add_argument("--selftest", action="store_true", help="run the pipeline self-test")
    parser.add_argument("--json", action="store_true", help="print the AnswerResult as JSON")
    args = parser.parse_args(argv)

    logger.configure_logging()

    if args.selftest:
        return _selftest()
    if args.chat:
        return _run_chat()

    result_object = answer_question(args.question or "")
    if args.json:
        print(json.dumps(_as_dict(result_object), ensure_ascii=False, indent=2, default=str))
    else:
        _print_result(result_object)
    return 0


if __name__ == "__main__":
    sys.exit(main())

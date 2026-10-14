"""Constraint logic: PII detection, intent routing, refusals, and answer validators.

This module performs no network, embedding, or LLM work. It is the safety net that makes
every answer path checkable in isolation (architecture 7.4).
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from enum import Enum

from src import config

MAX_SENTENCES: int = config.MAX_SENTENCES

ABBREVIATION_SENTINEL = "\x00"
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z“(])")
PROTECTED_PERIOD_RE = re.compile(
    r"(?:\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|vs|etc|approx|Fig|No|Ref|Rs|NAV|SIP)\.|\d\.\d)"
)
URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\"']+")
LAST_UPDATED_RE = re.compile(
    re.escape(config.LAST_UPDATED_PREFIX) + r"\s*(\d{4}-\d{2}-\d{2})?"
)
TRAILING_LINE_RE = re.compile(
    r"^(?:" + URL_RE.pattern + r"|" + re.escape(config.LAST_UPDATED_PREFIX) + r".*)$"
)

PREAMBLE_RE = re.compile(
    r"^(?:sure|certainly|of course|absolutely|great question|hello|hi|okay|ok)\b[!,.:]?\s*"
    r"|^(?:based on (?:the )?(?:context|sources?|provided information)[,:]?\s*)"
    r"|^(?:according to (?:the )?(?:context|sources?|information)[,:]?\s*)"
    r"|^(?:here(?:'s| is| are)[^.:]{0,60}[:.]\s*)"
    r"|^(?:the answer is[:,]?\s*)",
    re.I,
)

ADVICE_PHRASES: tuple[tuple[str, float], ...] = (
    (r"\bshould\s+i\b", 1.0),
    (r"\bshould\s+we\b", 1.0),
    (r"\bshould\s+i\s+(?:invest|buy|sell|redeem|switch)\b", 1.0),
    (r"\btell\s+me\s+if\s+i\s+should\b", 1.0),
    (r"\bwhich\s+one\s+should\b", 1.0),
    (r"\bdo\s+you\s+think\b", 1.0),
    (r"\brecommend\w*\b", 1.0),
    (r"\bsuggest\w*\b", 1.0),
    (r"\badvice\b", 1.0),
    (r"\bbest\s+(?:fund|option|choice|pick)\b", 1.0),
    (r"\bbetter\s+than\b", 1.0),
    (r"\bwhich\s+is\s+better\b", 1.0),
    (r"\bworth\s+(?:buying|investing|buying\s+in)\b", 1.0),
    (r"\bgood\s+time\s+to\b", 1.0),
    (r"\ballocate\b", 1.0),
    (r"\bportfoli(?:o|os)\b", 1.0),
    (r"\bsuita\w*\s+for\s+me\b", 1.0),
    (r"\bsuitable\s+for\s+me\b", 1.0),
    (r"\bfor\s+my\s+(?:portfolio|horizon|goals?)\b", 1.0),
    (r"\bis\s+it\s+(?:safe|wise|a\s+good\s+idea)\b", 1.0),
)

PERFORMANCE_PHRASES: tuple[tuple[str, float], ...] = (
    (r"\breturns?\b", 1.0),
    (r"\bcagr\b", 1.0),
    (r"\bbest\s+performing\b", 1.0),
    (r"\btop\s+performing\b", 1.0),
    (r"\bwhich\s+grew\b", 1.0),
    (r"\bhas\s+grown\b", 1.0),
    (r"\bhow\s+much\s+did\s+it\s+grow\b", 1.0),
    (r"\bprofit\b", 1.0),
    (r"\bgrowth\s+of\b", 1.0),
    (r"\bsince\s+inception\s+return\b", 1.0),
    (r"\bx%\s+return\b", 1.0),
    (r"\bperformance\s+(?:of|compared)\b", 1.0),
    (r"\boutperform\w*\b", 1.0),
    (r"\bwhich\s+(?:fund|scheme)\s+(?:did\s+best|performed)\b", 1.0),
)

ADVICE_WEIGHT_THRESHOLD = 1.0
HOW_TO_MARKERS = re.compile(
    r"\bhow\s+(?:do|can|to)\b|\bwhere\s+(?:do|can)\s+i\b|\bsteps?\b|\bprocess\b"
    r"|\bdownload\b|\bstatement\b|\bprocedure\b|\bguide\b",
    re.I,
)

OUTPUT_ADVICE_RE = re.compile(
    r"\byou\s+should\b|\bi\s+recommend\b|\bwe\s+recommend\b|\brecommended?\b"
    r"|\bsuitable\s+for\s+you\b|\bideal\s+for\s+you\b|\bfits\s+your\b"
    r"|\bis\s+a\s+good\s+(?:choice|option)\b|\bbest\s+(?:fund|option|choice)\b"
    r"|\bworth\s+(?:buying|investing)\b|\bconsider\s+investing\b",
    re.I,
)
OUTPUT_PERFORMANCE_RE = re.compile(
    r"\bcagr\b|\boutperform\w*\b|\bbest\s+performing\b|\btop\s+performing\b"
    r"|\bsince\s+inception\s+return\b"
    r"|(?<!tax on )(?<!taxation of )(?<!tax on)\breturn(?:s|ed|ing)?\b(?:\s*(?:of|over|at|to|is|was|were|are|:|-)?\s*){0,2}\d+(?:\.\d+)?\s*%"
    r"|\b\d+(?:\.\d+)?\s*%\s*(?:returns?|growth|gained)\b"
    r"|\b\d+(?:\.\d+)?\s*%\s*(?:per\s+(?:annum|year)|p\.?\s?a\.?)\b"
    r"|\bgrew\s+(?:by\s+)?\d|\bprofit(?:able)?\s+(?:of|by)?\s*\d"
    r"|\bappreciat\w+\s+(?:of|by)\s+\d",
    re.I,
)

PII_LABELS: tuple[str, ...] = ("pan", "aadhaar", "account", "otp", "email", "phone")
PII_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("aadhaar", re.compile(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")),
    ("phone", re.compile(r"(?<![\d-])(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?![\d-])")),
    ("account", re.compile(r"(?<!\d)\d{6,18}(?!\d)")),
    ("otp", re.compile(r"(?<!\d)\d{4,6}(?!\d)")),
)

WHITELIST_RULES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\b(?:19|20)\d{2}\b"),
    re.compile(r"\b\d+(?:\.\d+)?\s?%"),
    re.compile(r"(?:Rs\.?|₹|INR)\s?\d[\d,]*(?:\.\d+)?"),
    re.compile(r"\b\d{1,3}(?:,\d{2,3})+(?:\.\d+)?\b"),
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),
    re.compile(r"\b\d+(?:\.\d+)?\s?(?:years?|months?|days?|basis\s*points?|bps)\b", re.I),
)

MASK_CHAR = "·"
HOW_TO_REASONS = frozenset({"how_to"})
REFUSAL_REASONS: tuple[str, ...] = ("advice", "pii", "performance", "no_context")
REFUSAL_TEXT: dict[str, str] = {
    "advice": config.REFUSAL_ADVICE,
    "pii": config.REFUSAL_PII,
    "performance": (
        "I cannot state, compare, or rank returns or performance. Performance figures for "
        "each HDFC scheme are published in its official factsheet and on the AMC's public "
        "page, and I can share scheme terms such as expense ratio, exit load, and minimum "
        "investment from the indexed sources."
    ),
    "no_context": config.REFUSAL_NO_CONTEXT,
}
REFUSAL_LINKS: dict[str, str | None] = {
    "advice": config.EDUCATIONAL_LINKS["advice"],
    "performance": config.EDUCATIONAL_LINKS["performance"],
    "pii": None,
    "no_context": config.EDUCATIONAL_LINKS["how_to"],
}
HARD_FLAGS = ("too_long", "missing_citation", "citation_not_in_context", "missing_last_updated")
UNREPAIRABLE_FLAGS = ("advice_language", "performance_language")
DETECTABLE_FLAGS = HARD_FLAGS + UNREPAIRABLE_FLAGS + ("preamble",)


class Intent(str, Enum):
    """Routing class for a user question."""

    FACT = "fact"
    HOW_TO = "how_to"
    ADVICE = "advice"
    PII = "pii"


@dataclass(frozen=True)
class AnswerResult:
    """The value every answer path returns, factual or refused.

    Defined here because ``refusal_response`` is the only producer in stage 12; the
    orchestrator in ``answer.py`` re-exports it.
    """

    answer: str
    source_url: str | None = None
    last_updated: str | None = None
    refused: bool = False
    refusal_reason: str | None = None
    guardrail_flags: list[str] = field(default_factory=list)
    used_fallback: bool = False
    top_chunks: list = field(default_factory=list)
    latency_ms: int = 0


def _mask_whitelisted(text: str) -> str:
    """Blank out spans that must never count as PII: dates, years, percentages, amounts."""
    masked = text
    for rule in WHITELIST_RULES:
        masked = rule.sub(lambda match: MASK_CHAR * len(match.group(0)), masked)
    return masked


def detect_pii(text: str) -> list[str]:
    """Return the labels of every PII class found in ``text``, in a stable order.

    Scheme facts that merely look numeric are excluded: SIP amounts, calendar years,
    ISO dates, percentages, and grouped rupee amounts are whitelisted before matching.
    """
    if not text:
        return []
    masked = _mask_whitelisted(text)
    found: list[str] = []
    for label, pattern in PII_RULES:
        if pattern.search(masked) and label not in found:
            found.append(label)
    return sorted(found, key=PII_LABELS.index)


def has_pii(text: str) -> bool:
    """True when the text contains anything that looks like personal information."""
    return bool(detect_pii(text))


def _weighted_match(text: str, phrases: tuple[tuple[str, float], ...]) -> float:
    return sum(weight for pattern, weight in phrases if re.search(pattern, text, re.I))


def is_advice_intent(question: str) -> bool:
    """True when the question asks for a judgement, recommendation, or allocation."""
    if not question:
        return False
    return _weighted_match(question, ADVICE_PHRASES) >= ADVICE_WEIGHT_THRESHOLD


def is_performance_intent(question: str) -> bool:
    """True when the question asks for returns, growth, or a performance ranking."""
    if not question:
        return False
    return _weighted_match(question, PERFORMANCE_PHRASES) >= ADVICE_WEIGHT_THRESHOLD


def classify_intent(question: str) -> Intent:
    """Route a question to an intent; PII is checked first, then advice, then how-to.

    Performance questions are classified as ``ADVICE`` because both paths refuse; call
    ``is_performance_intent`` alongside this to pick the redirect link.
    """
    if has_pii(question):
        return Intent.PII
    if is_advice_intent(question) or is_performance_intent(question):
        return Intent.ADVICE
    if HOW_TO_MARKERS.search(question):
        return Intent.HOW_TO
    return Intent.FACT


def refusal_response(reason: str, link: str | None = None) -> AnswerResult:
    """Return the deterministic canned refusal for ``reason``; never calls the LLM.

    ``link`` overrides the default for that reason. Advice and performance refusals carry
    an educational link (FR-27); a PII refusal cites nothing, because no source was used.
    """
    if reason not in REFUSAL_REASONS:
        raise ValueError(
            f"unknown refusal reason {reason!r}; expected one of {', '.join(REFUSAL_REASONS)}"
        )
    body = REFUSAL_TEXT[reason]
    disclaimer = config.DISCLAIMER
    answer = f"{body}\n\n{disclaimer}"
    return AnswerResult(
        answer=answer,
        source_url=link if link is not None else REFUSAL_LINKS[reason],
        last_updated=None,
        refused=True,
        refusal_reason=reason,
        guardrail_flags=[f"refused:{reason}"],
        used_fallback=False,
        top_chunks=[],
        latency_ms=0,
    )


def _split_trailing(text: str) -> tuple[str, list[str]]:
    """Separate the citation and last-updated lines, which must survive trimming."""
    lines = text.splitlines()
    trailer: list[str] = []
    while lines and (not lines[-1].strip() or TRAILING_LINE_RE.match(lines[-1].strip())):
        candidate = lines.pop().strip()
        if candidate:
            trailer.insert(0, candidate)
    return "\n".join(lines).strip(), trailer


def _split_sentences(text: str) -> list[str]:
    """Split prose into sentences without breaking decimals or common abbreviations."""
    protected = PROTECTED_PERIOD_RE.sub(
        lambda match: match.group(0).replace(".", ABBREVIATION_SENTINEL), text
    )
    parts = SENTENCE_SPLIT_RE.split(protected)
    return [part.replace(ABBREVIATION_SENTINEL, ".").strip() for part in parts if part.strip()]


def count_sentences(text: str) -> int:
    """Count sentences in prose, ignoring the citation and last-updated lines."""
    body, _ = _split_trailing(text)
    return len(_split_sentences(body))


def trim_sentences(answer: str, max_sentences: int = MAX_SENTENCES) -> str:
    """Keep at most ``max_sentences`` leading sentences, preserving trailing lines.

    The citation line and the ``Last updated from sources:`` line are not prose and are
    re-attached unchanged, so trimming cannot strip the citation.
    """
    if max_sentences <= 0:
        raise ValueError("max_sentences must be positive")
    body, trailer = _split_trailing(answer)
    if not body:
        return answer.strip()
    sentences = _split_sentences(body)
    kept = " ".join(sentences[:max_sentences]) if sentences else body
    return "\n".join([kept, *trailer]).strip()


def strip_preamble(answer: str) -> str:
    """Remove a leading assistant pleasantry or "Based on the context" style lead-in."""
    body, trailer = _split_trailing(answer)
    cleaned = body
    for _ in range(3):
        match = PREAMBLE_RE.match(cleaned.strip())
        if not match:
            break
        cleaned = cleaned.strip()[match.end():].strip()
    return "\n".join([_recapitalise(cleaned), *trailer]).strip()


def _recapitalise(text: str) -> str:
    """Restore an upper-case first letter after a lead-in has been cut away."""
    stripped = text.lstrip()
    if not stripped or not stripped[0].isalpha() or stripped[0].isupper():
        return text
    return text.replace(stripped[0], stripped[0].upper(), 1)


def has_preamble(answer: str) -> bool:
    """True when the answer opens with a conversational or meta lead-in."""
    body, _ = _split_trailing(answer)
    return bool(PREAMBLE_RE.match(body.strip()))


def find_urls(text: str) -> list[str]:
    """Return every URL in the text, de-duplicated and stripped of trailing punctuation."""
    seen: list[str] = []
    for match in URL_RE.finditer(text or ""):
        url = match.group(0).rstrip(".,;:!?)\"'")
        if url and url not in seen:
            seen.append(url)
    return seen


def ensure_last_updated(answer: str, last_updated: str) -> str:
    """Guarantee a final ``Last updated from sources: <date>`` line exactly once."""
    body, trailer = _split_trailing(answer)
    line = f"{config.LAST_UPDATED_PREFIX} {last_updated}".strip()
    trailer = [entry for entry in trailer if not LAST_UPDATED_RE.match(entry)]
    return "\n".join([part for part in [body, *trailer, line] if part]).strip()


def _strip_url_lines(body: str, keep_url: str | None) -> str:
    """Drop every line that is only a URL, then optionally re-add the kept citation."""
    kept_lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped and URL_RE.fullmatch(stripped.rstrip(".,;:!?)\"'")):
            continue
        kept_lines.append(line)
    cleaned = "\n".join(kept_lines).strip()
    if keep_url:
        cleaned = f"{cleaned}\n{keep_url}".strip() if cleaned else keep_url
    return cleaned


def validate_answer(
    answer: str,
    context_urls: set[str],
    last_updated: str,
) -> tuple[bool, list[str], str]:
    """Check an answer against the output contract and repair it in one pass.

    Repairs run in a fixed order: strip preamble, trim to ``MAX_SENTENCES`` sentences,
    reduce the citation to the first in-context URL, then re-attach the last-updated
    line. Returns ``(ok, flags, repaired)``; ``ok`` is False when a hard constraint
    survives the repair, in which case the caller should use the extractive fallback.
    """
    flags: list[str] = []
    repaired = answer or ""
    urls = config.ALLOWED_URLS
    context = {url for url in context_urls if url in urls}

    if has_preamble(repaired):
        flags.append("preamble")
    repaired = strip_preamble(repaired)

    if count_sentences(repaired) > MAX_SENTENCES:
        flags.append("too_long")
        repaired = trim_sentences(repaired)

    found = find_urls(repaired)
    in_context = [url for url in found if url in context]
    out_of_context = [url for url in found if url not in context]
    if out_of_context:
        flags.append("citation_not_in_context")
    if not in_context:
        flags.append("missing_citation")

    keep = in_context[0] if in_context else (context and sorted(context)[0]) or None
    if found or keep:
        repaired = _strip_url_lines(repaired, keep)

    if OUTPUT_ADVICE_RE.search(repaired):
        flags.append("advice_language")
    if OUTPUT_PERFORMANCE_RE.search(repaired):
        flags.append("performance_language")

    if not LAST_UPDATED_RE.search(repaired):
        flags.append("missing_last_updated")
    repaired = ensure_last_updated(repaired, last_updated)

    if keep:
        repaired = ensure_citation(repaired, keep)

    surviving = _surviving_flags(repaired, keep, context)
    return (not surviving), flags, repaired


def _surviving_flags(repaired: str, keep: str | None, context: set[str]) -> list[str]:
    """Report which flagged problems the repaired text still exhibits.

    A flag that the repair pass actually fixed is not surviving, so a repaired answer is
    accepted. Advice and performance language are never repairable by the documented
    repair list, so they always survive and force the caller to the extractive fallback
    (FR-28).
    """
    surviving: list[str] = []
    if has_preamble(repaired):
        surviving.append("preamble")
    if count_sentences(repaired) > MAX_SENTENCES:
        surviving.append("too_long")
    if not keep or keep not in context or find_urls(repaired) != [keep]:
        surviving.append("missing_citation")
        surviving.append("citation_not_in_context")
    if not LAST_UPDATED_RE.search(repaired):
        surviving.append("missing_last_updated")
    if OUTPUT_ADVICE_RE.search(repaired):
        surviving.append("advice_language")
    if OUTPUT_PERFORMANCE_RE.search(repaired):
        surviving.append("performance_language")
    return surviving


def surviving_flags(answer: str, keep: str | None, context: set[str]) -> list[str]:
    """Report which flagged problems ``answer`` still exhibits, after any repair.

    Public view of the repair pass used by :func:`validate_answer`. Callers that apply
    their own repairs need this to drop diagnostics the repair already resolved instead of
    logging a violation the final text does not have.
    """
    return _surviving_flags(answer, keep, context)


def ensure_citation(answer: str, url: str) -> str:
    """Return the answer with exactly ``url`` as its single citation line."""
    body, trailer = _split_trailing(answer)
    trailer = [entry for entry in trailer if not URL_RE.fullmatch(entry)]
    return "\n".join([part for part in [body, *trailer, url] if part]).strip()


def _selftest_questions() -> tuple[tuple[str, str], ...]:
    return (
        ("Q1", "What is the exit load of HDFC Large Cap Fund?"),
        ("Q2", "What is the minimum SIP amount for HDFC Small Cap Fund?"),
        ("Q3", "What is the lock-in period for HDFC ELSS Tax Saver Fund?"),
        ("Q4", "What is the expense ratio / TER of HDFC Equity Fund?"),
        (
            "Q5",
            "Which benchmark does HDFC Balanced Advantage Fund follow, and what is its "
            "riskometer category?",
        ),
        ("Q6", "How do I download the capital-gains statement?"),
        ("Q7", "Should I buy HDFC Small Cap Fund?"),
        ("Q8", "Which of these 5 funds has given the best returns?"),
        ("Q9", "Is HDFC ELSS better than a large cap fund for my portfolio?"),
        ("Q10", "My PAN is 1234567890, please fetch my folio"),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: classify the PRD sample set and probe the PII whitelist."""
    parser = argparse.ArgumentParser(description="Guardrail self-check (no LLM, no network).")
    parser.add_argument(
        "--question", action="append", default=[], help="classify one question (repeatable)"
    )
    args = parser.parse_args(argv)

    questions = (
        [(f"custom {index}", text) for index, text in enumerate(args.question, start=1)]
        if args.question
        else list(_selftest_questions())
    )
    print(f"{'#':<8}{'intent':<9}{'pii':<22}{'advice':<8}{'perf':<6}question")
    print("-" * 100)
    for label, question in questions:
        pii = detect_pii(question)
        print(
            f"{label:<8}{classify_intent(question).value:<9}"
            f"{','.join(pii) or '-':<22}"
            f"{str(is_advice_intent(question)):<8}"
            f"{str(is_performance_intent(question)):<6}"
            f"{question}"
        )

    print("\nwhitelist spot checks (must all print 'clean'):")
    for sample in (
        "The minimum SIP amount is Rs. 500 per month.",
        "The lock-in period is 3 years as on 2026-09-27.",
        "The total expense ratio is 1.19% and the exit load is 1.0%.",
        "Lump sum minimum is Rs. 10,000.",
    ):
        print(f"  {detect_pii(sample) or 'clean':<22} {sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

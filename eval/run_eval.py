"""Phase 10 evaluation harness: drives answer_question over the golden query set.

Usage:
    python eval/run_eval.py            # offline, deterministic extractive path
    python eval/run_eval.py --live     # real Groq call for each answered row

The offline default pins ``config.GROQ_API_KEY`` to the empty string so every answered
row takes the extractive fallback; no network and no API key are needed for CI. Each row
is scored against the golden record (type correctness, allowlist citation, <=3 sentences,
last-updated line, must_mention keywords, latency). A JSON report and a Markdown table
are written to ``eval/report.json`` and ``eval/report.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, guardrails, retrieve  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent

LAST_UPDATED_PREFIX = config.LAST_UPDATED_PREFIX
PII_NO_CITATION = frozenset({"pii"})
NO_CITATION_NEEDED = PII_NO_CITATION


def _in_corpus(url: str | None) -> bool:
    return bool(url) and retrieve.is_allowlisted(url)


def _norm(text: str) -> str:
    """Normalize NFC plus typographic dashes to ASCII for keyword matching."""
    text = unicodedata.normalize("NFC", text or "")
    for dash in ("\u2011", "\u2010", "\u2012", "\u2013", "\u00ad"):
        text = text.replace(dash, "-")
    return text.lower()


def _score(row: dict, result: guardrails.AnswerResult) -> dict:
    """Return one row of check results with per-check pass/fail booleans."""
    expected_refusal = row.get("expected_refusal")
    refused = bool(result.refused)
    reason = result.refusal_reason
    answer = result.answer or ""

    if expected_refusal:
        type_ok = refused and reason == expected_refusal
        if expected_refusal in NO_CITATION_NEEDED:
            citation_ok = True
        else:
            citation_ok = bool(result.source_url)
        sentences_ok = True
        last_updated_ok = True
        must_ok = True
    else:
        type_ok = not refused
        citation_ok = _in_corpus(result.source_url)
        sentences_ok = guardrails.count_sentences(answer) <= config.MAX_SENTENCES
        last_updated_ok = LAST_UPDATED_PREFIX in answer
        keywords = [str(kw).lower() for kw in row.get("must_mention", [])]
        lowered = _norm(answer)
        must_ok = all(kw in lowered for kw in keywords)

    checks = {
        "type": bool(type_ok),
        "citation": bool(citation_ok),
        "sentences": bool(sentences_ok),
        "last_updated": bool(last_updated_ok),
        "must_mention": bool(must_ok),
    }
    applied = (
        list(checks) if not expected_refusal else ["type", "citation"]
    )
    return {
        "id": row["id"],
        "expected_type": row["expected_type"],
        "expected_refusal": expected_refusal,
        "passed": all(checks[key] for key in applied),
        "checks": {key: checks[key] for key in applied},
        "refused": refused,
        "refusal_reason": reason,
        "source_url": result.source_url,
        "used_fallback": bool(result.used_fallback),
        "latency_ms": result.latency_ms,
        "answer": answer,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true",
                        help="call Groq (GROQ_API_KEY required); default is the offline extractive path")
    args = parser.parse_args(argv)

    golden_path = EVAL_DIR / "golden_queries.json"
    if not golden_path.exists():
        print(f"missing golden set: {golden_path}")
        return 1

    with golden_path.open(encoding="utf-8") as handle:
        golden = json.load(handle)

    if not args.live:
        config.GROQ_API_KEY = ""

    from src import answer

    results = []
    started = time.perf_counter()
    for row in golden:
        result = answer.answer_question(row["question"])
        results.append(_score(row, result))
    wall_s = time.perf_counter() - started

    passed = sum(1 for r in results if r["passed"])
    prd_rows = [r for r in results if r["id"].startswith("Q")]
    nonsense_rows = [r for r in results if r["id"].startswith("N")]
    prd_passed = sum(1 for r in prd_rows if r["passed"])
    nonsense_passed = sum(1 for r in nonsense_rows if r["passed"])

    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "live" if args.live else "offline",
        "config": {
            "min_score": config.MIN_SIMILARITY,
            "mmr_lambda": config.MMR_LAMBDA,
            "lexical_alpha": config.LEXICAL_ALPHA,
            "candidate_pool": config.CANDIDATE_POOL,
        },
        "summary": {
            "total": len(results),
            "passed": passed,
            "prd_total": len(prd_rows),
            "prd_passed": prd_passed,
            "nonsense_total": len(nonsense_rows),
            "nonsense_passed": nonsense_passed,
            "wall_seconds": round(wall_s, 2),
        },
        "results": results,
    }

    (EVAL_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    table = [
        "| id | type | verdict | refused | cited | ≤3 sen | last-upd | must-ment | ms |",
        "|----|------|---------|---------|-------|--------|----------|-----------|-----|",
    ]
    for r in results:
        checks = r["checks"]
        table.append(
            "| {id} | {t} | {v} | {ref} | {cit} | {sen} | {lu} | {must} | {ms} |".format(
                id=r["id"],
                t=r["expected_type"] + (f"→{r['expected_refusal']}" if r["expected_refusal"] else ""),
                v="PASS" if r["passed"] else "FAIL",
                ref="Y" if r["refused"] else "-",
                cit="Y" if checks.get("citation") else ("n/a" if "citation" not in checks else "-"),
                sen=("Y" if checks.get("sentences") else "-") if "sentences" in checks else "n/a",
                lu=("Y" if checks.get("last_updated") else "-") if "last_updated" in checks else "n/a",
                must=("Y" if checks.get("must_mention") else "-") if "must_mention" in checks else "n/a",
                ms=r["latency_ms"],
            )
        )
    table.append(
        f"\n**{report['summary']['passed']}/{report['summary']['total']} overall; "
        f"{report['summary']['prd_passed']}/{report['summary']['prd_total']} PRD queries; "
        f"wall {report['summary']['wall_seconds']} s ({report['mode']})**"
    )
    markdown = "\n".join(table)
    (EVAL_DIR / "report.md").write_text(markdown, encoding="utf-8")

    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
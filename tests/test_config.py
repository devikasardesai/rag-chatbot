"""Tests for config constants and logger PII redaction."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib
import warnings

import pytest

from src import config, logger

EXPECTED_CORPUS_URLS = [
    "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
    "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
    "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
    "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
]

EXPECTED_PROMPT_SYSTEM = (
    "You are a mutual fund FAQ assistant for HDFC Asset Management schemes\n"
    "(Large Cap, Flexi Cap / Equity, ELSS Tax Saver, Small Cap, Balanced Advantage —\n"
    "all Direct Growth plans).\n"
    "\n"
    "Rules, in strict order of priority:\n"
    "1. Answer ONLY from the CONTEXT blocks. Never use outside knowledge.\n"
    "2. If the answer is not in the context, reply exactly:\n"
    '   "Not found in the indexed sources." and cite the most relevant source URL.\n'
    "3. Factual answers: at most 3 sentences, plain text, no headings, no bullet lists.\n"
    "4. End every factual answer with the source link on its own final line, exactly as\n"
    "   written in the context, and nothing else on that line.\n"
    "5. Never give investment advice, recommendations, opinions, suitability, rankings,\n"
    '   or comparisons of "better"/"best". If asked, reply with the facts-only refusal\n'
    "   text and an educational link.\n"
    "6. Never state, compute, compare, or estimate returns, CAGR, NAV performance, or\n"
    "   rankings. If asked, say performance data is in the official factsheet and cite it.\n"
    "7. Do not ask for or record PAN, Aadhaar, account numbers, OTPs, email, or phone.\n"
    "8. Do not invent numbers. If a figure is missing, say so."
)


class TestCorpus:
    def test_corpus_has_five_entries(self, corpus):
        assert len(corpus) == 5
        assert len(config.CORPUS) == 5

    def test_corpus_urls_are_exact(self, corpus):
        assert [spec.url for spec in corpus] == EXPECTED_CORPUS_URLS

    def test_corpus_uses_corrected_balanced_advantage_url(self):
        spec = next(
            s for s in config.CORPUS if s.category == "Balanced Advantage"
        )
        assert spec.url.startswith("https://groww.in/")
        assert "httroww" not in spec.url

    def test_allowed_urls_matches_corpus(self):
        assert set(config.ALLOWED_URLS) == {spec.url for spec in config.CORPUS}
        assert isinstance(config.ALLOWED_URLS, frozenset)

    def test_categories_and_plan(self, corpus):
        assert [spec.category for spec in corpus] == [
            "Large Cap",
            "Flexi Cap",
            "ELSS",
            "Small Cap",
            "Balanced Advantage",
        ]
        assert {spec.plan for spec in corpus} == {"Direct Growth"}

    def test_source_spec_is_frozen(self, corpus):
        with pytest.raises(FrozenInstanceError):
            corpus[0].scheme_name = "mutated"

    def test_allsources_are_groww_scheme_pages(self, corpus):
        for spec in corpus:
            assert spec.url.startswith("https://groww.in/mutual-funds/hdfc-")


class TestPromptSystem:
    def test_prompt_matches_architecture_verbatim(self):
        assert config.PROMPT_SYSTEM == EXPECTED_PROMPT_SYSTEM

    def test_prompt_rule_numbering_is_one_through_eight(self):
        rules = [
            line.split(".", 1)[0]
            for line in config.PROMPT_SYSTEM.splitlines()
            if line[:1].isdigit()
        ]
        assert rules == [str(n) for n in range(1, 9)]

    def test_prompt_contains_no_placeholder(self):
        for token in ("TODO", "FIXME", "XXX"):
            assert token not in config.PROMPT_SYSTEM

    def test_prompt_forbids_advice_and_performance(self):
        lowered = config.PROMPT_SYSTEM.lower()
        assert "never give investment advice" in lowered
        assert "never state, compute, compare, or estimate returns" in lowered
        assert "do not invent numbers" in lowered


class TestRetrievalAndChunkingConstants:
    def test_chunk_chars_derived_from_tokens(self):
        assert config.CHUNK_CHARS == config.CHUNK_SIZE_TOKENS * config.CHARS_PER_TOKEN
        assert config.CHUNK_OVERLAP_CHARS == config.CHUNK_OVERLAP_TOKENS * 4
        assert config.CHUNK_OVERLAP_CHARS < config.CHUNK_CHARS

    def test_retrieval_defaults(self):
        assert config.TOP_K == 5
        assert 0.0 < config.MIN_SIMILARITY < 1.0
        assert 0.0 < config.MMR_LAMBDA <= 1.0
        assert config.MAX_CHUNKS_PER_SECTION == 2
        assert config.MAX_CONTEXT_TOKENS <= config.LLM_NUM_CTX - config.LLM_NUM_PREDICT

    def test_answer_constraints(self):
        assert config.MAX_SENTENCES == 3
        assert config.LLM_TEMPERATURE == 0.0
        assert config.LAST_UPDATED_PREFIX == "Last updated from sources:"

    def test_must_have_facts_cover_every_scheme(self, corpus):
        assert set(config.MUST_HAVE_FACTS) == {spec.scheme_name for spec in corpus}
        for probes in config.MUST_HAVE_FACTS.values():
            assert probes
            assert all(isinstance(probe, str) and probe for probe in probes)

    def test_llm_runtime_values(self):
        assert config.LLM_PROVIDER == "groq"
        assert config.configured_provider() == "groq"
        assert "gpt-oss" in config.LLM_MODEL
        assert "api.groq.com" in config.LLM_BASE_URL
        assert config.EMBEDDING_DIM == 384
        assert config.HNSW_SPACE == "cosine"

    @pytest.mark.parametrize("requested", ["grok", "xai", "ollama", "openai"])
    def test_a_foreign_provider_env_is_ignored_with_a_warning(self, requested, monkeypatch):
        """A stale LLM_PROVIDER in .env must not route generation away from Groq."""
        monkeypatch.setenv("LLM_PROVIDER", requested)
        with pytest.warns(RuntimeWarning, match="groq"):
            reloaded = importlib.reload(config)
        try:
            assert reloaded.LLM_PROVIDER == "groq"
            assert reloaded.configured_provider() == "groq"
        finally:
            monkeypatch.delenv("LLM_PROVIDER", raising=False)
            importlib.reload(config)

    def test_groq_provider_env_does_not_warn(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            reloaded = importlib.reload(config)
        assert reloaded.configured_provider() == "groq"


class TestUserFacingCopy:
    def test_refusals_are_non_empty_and_distinct(self):
        refusals = [
            config.REFUSAL_ADVICE,
            config.REFUSAL_PII,
            config.REFUSAL_NO_CONTEXT,
        ]
        assert all(refusal.strip() for refusal in refusals)
        assert len(set(refusals)) == 3

    def test_pii_refusal_lists_sensitive_fields(self):
        lowered = config.REFUSAL_PII.lower()
        for field in ("pan", "aadhaar", "account number", "otp", "email", "phone"):
            assert field in lowered

    def test_no_context_refusal_uses_prompt_wording(self):
        assert "Not found in the indexed sources." in config.REFUSAL_NO_CONTEXT

    def test_educational_links(self):
        assert set(config.EDUCATIONAL_LINKS) == {"advice", "performance", "how_to"}
        for link in config.EDUCATIONAL_LINKS.values():
            assert link.startswith("https://")

    def test_disclaimer_is_facts_only(self):
        lowered = config.DISCLAIMER.lower()
        assert "facts only" in lowered
        assert "no investment advice" in lowered


class TestLoggerRedaction:
    def test_redact_pii_removes_pan(self):
        text = "My PAN is ABCDE1234F, please fetch my folio"
        redacted = logger.redact_pii(text)
        assert "ABCDE1234F" not in redacted
        assert "redacted:pan:" in redacted
        assert "please fetch my folio" in redacted

    def test_redact_pii_removes_aadhaar_email_and_phone(self):
        text = (
            "aadhaar 4567 8901 2345, mail devika@example.com, "
            "call +91 98765 43210"
        )
        redacted = logger.redact_pii(text)
        for secret in ("4567 8901 2345", "devika@example.com", "98765 43210"):
            assert secret not in redacted
        assert "redacted:" in redacted

    @pytest.mark.parametrize(
        "phone",
        ["9876543210", "98765 43210", "98765-43210"],
    )
    def test_redact_pii_catches_phone_shapes(self, phone):
        redacted = logger.redact_pii(f"call me on {phone}")
        assert phone not in redacted
        assert "redacted:phone:" in redacted

    def test_redact_pii_does_not_nest_markers(self):
        redacted = logger.redact_pii("mail devika@example.com")
        assert "devika@example.com" not in redacted
        assert redacted.count("[redacted:") == 1
        assert redacted.count("[") == 1

    def test_redact_pii_hashes_stably_and_uniquely(self):
        first = logger.redact_pii("PAN ABCDE1234F")
        second = logger.redact_pii("PAN ABCDE1234F")
        other = logger.redact_pii("PAN ZZZZZ4321G")
        assert first == second
        assert first != other

    def test_redact_pii_leaves_factual_numbers_intact(self):
        text = "Minimum SIP is Rs. 500 since 2026-09-27 with expense ratio 1.19%"
        assert logger.redact_pii(text) == text

    def test_redact_pii_leaves_iso_timestamps_intact(self):
        text = '{"ts":"2026-09-27T14:05:02Z","latency_ms":4410}'
        assert logger.redact_pii(text) == text

    def test_redact_pii_handles_empty_string(self):
        assert logger.redact_pii("") == ""

    def test_log_writes_jsonl_without_raw_pii(self, tmp_data_dir):
        logger.log_answer({"question": "PAN ABCDE1234F", "answer": "ok"})
        logger.log_retrieval({"hits": [{"scheme": "HDFC Small Cap Fund"}]})
        answers = (config.LOG_DIR / "answers.jsonl").read_text(encoding="utf-8")
        retrieval = (config.LOG_DIR / "retrieval.jsonl").read_text(encoding="utf-8")
        assert "ABCDE1234F" not in answers
        assert "redacted:pan:" in answers
        assert "HDFC Small Cap Fund" in retrieval
        assert len(answers.strip().splitlines()) == 1

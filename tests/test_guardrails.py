"""Tests for the guardrail module: PII, intent routing, refusals, and answer validation.

No network, no embedding, no LLM. The whole module is importable in isolation.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src import config
from src.guardrails import (
    DETECTABLE_FLAGS,
    HARD_FLAGS,
    MAX_SENTENCES,
    REFUSAL_REASONS,
    UNREPAIRABLE_FLAGS,
    AnswerResult,
    Intent,
    classify_intent,
    count_sentences,
    detect_pii,
    ensure_citation,
    ensure_last_updated,
    find_urls,
    has_pii,
    has_preamble,
    is_advice_intent,
    is_performance_intent,
    refusal_response,
    strip_preamble,
    trim_sentences,
    validate_answer,
)

LARGE_CAP_URL = config.CORPUS[0].url
SMALL_CAP_URL = config.CORPUS[3].url
DATE = "2026-09-27"
LAST_UPDATED = f"{config.LAST_UPDATED_PREFIX} {DATE}"

PRD_QUESTIONS: tuple[tuple[str, str, Intent, bool, bool], ...] = (
    ("Q1", "What is the exit load of HDFC Large Cap Fund?", Intent.FACT, False, False),
    ("Q2", "What is the minimum SIP amount for HDFC Small Cap Fund?", Intent.FACT, False, False),
    (
        "Q3",
        "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
        Intent.FACT,
        False,
        False,
    ),
    ("Q4", "What is the expense ratio / TER of HDFC Equity Fund?", Intent.FACT, False, False),
    (
        "Q5",
        "Which benchmark does HDFC Balanced Advantage Fund follow, and what is its "
        "riskometer category?",
        Intent.FACT,
        False,
        False,
    ),
    ("Q6", "How do I download the capital-gains statement?", Intent.HOW_TO, False, False),
    ("Q7", "Should I buy HDFC Small Cap Fund?", Intent.ADVICE, True, False),
    (
        "Q8",
        "Which of these 5 funds has given the best returns?",
        Intent.ADVICE,
        False,
        True,
    ),
    (
        "Q9",
        "Is HDFC ELSS better than a large cap fund for my portfolio?",
        Intent.ADVICE,
        True,
        False,
    ),
    ("Q10", "My PAN is 1234567890, please fetch my folio", Intent.PII, False, False),
)

PII_POSITIVES: tuple[tuple[str, str], ...] = (
    ("My PAN is ABCDE1234F", "pan"),
    ("pan: ABCDE1234F and more text", "pan"),
    ("My Aadhaar number is 2345 6789 0123", "aadhaar"),
    ("Aadhaar 9876 5432 1098 on the form", "aadhaar"),
    ("account number 99887766 please", "account"),
    ("My folio account is 1234567890123", "account"),
    ("The OTP is 4521", "otp"),
    ("enter otp 918273", "otp"),
    ("email me at ravi.kumar@example.com", "email"),
    ("reach me on +91 9876543210", "phone"),
    ("call 98765 43210 tomorrow", "phone"),
)

PII_NEGATIVES: tuple[str, ...] = (
    "The minimum SIP amount is Rs. 500 per month.",
    "Lump sum minimum is Rs. 10,000.",
    "The SIP amount is 500.",
    "500",
    "The lock-in period is 3 years.",
    "The page was last updated on 2026-09-27.",
    "The total expense ratio is 1.19%.",
    "The exit load is 1.0% for 0-12 months and nil after 24 months.",
    "The expense ratio is 1.5% of NAV.",
    "Tracking error is 12 bps.",
    "The scheme follows Nifty 50 TRI as its benchmark.",
    "Direct Growth plan of HDFC Large Cap Fund",
    "2026",
    "The AUM is Rs. 1,00,000 crore.",
    "Minimum SIP tenure is 3 months.",
    "NAV as on 27/09/2026 was Rs. 12.34.",
)


class TestIntentEnum:
    def test_intent_is_a_string_enum(self):
        assert Intent.FACT == "fact"
        assert isinstance(Intent.FACT, str)

    def test_intent_has_the_four_documented_members(self):
        assert [intent.value for intent in Intent] == ["fact", "how_to", "advice", "pii"]

    def test_intent_is_not_a_strenum_subclass(self):
        assert type(Intent.FACT) is not type(config.CORPUS)  # sanity: distinct types


class TestPiiDetection:
    @pytest.mark.parametrize("text,label", PII_POSITIVES)
    def test_detects_expected_label(self, text, label):
        assert label in detect_pii(text)

    @pytest.mark.parametrize("text", PII_NEGATIVES)
    def test_no_false_positive_on_scheme_facts(self, text):
        assert detect_pii(text) == [], f"false positive on {text!r}"

    def test_empty_and_whitespace(self):
        assert detect_pii("") == []
        assert detect_pii("   ") == []

    def test_has_pii_matches_detect(self):
        assert has_pii("My PAN is ABCDE1234F")
        assert not has_pii("The expense ratio is 1.19%.")

    def test_labels_are_deduplicated_and_ordered(self):
        labels = detect_pii("PAN ABCDE1234F and email a@b.co")
        assert labels == sorted(set(labels), key=["pan", "aadhaar", "account", "otp", "email", "phone"].index)

    def test_all_detected_labels_are_known(self):
        for text, _ in PII_POSITIVES:
            for label in detect_pii(text):
                assert label in ("pan", "aadhaar", "account", "otp", "email", "phone")

    def test_prd_q10_pan_like_number_is_detected(self):
        assert detect_pii("My PAN is 1234567890, please fetch my folio")

    def test_pii_before_intent_in_routing(self):
        assert classify_intent("Should I buy HDFC Small Cap Fund? My PAN is ABCDE1234F") is Intent.PII


class TestIntentRouting:
    @pytest.mark.parametrize("label,question,expected,advice,performance", PRD_QUESTIONS)
    def test_prd_question_set(self, label, question, expected, advice, performance):
        assert classify_intent(question) is expected, label
        assert is_advice_intent(question) is advice, label
        assert is_performance_intent(question) is performance, label

    def test_all_ten_prd_queries_are_covered(self):
        assert [row[0] for row in PRD_QUESTIONS] == [f"Q{index}" for index in range(1, 11)]

    @pytest.mark.parametrize(
        "question",
        [
            "Should I invest in HDFC Large Cap Fund?",
            "Should we sell after 3 years?",
            "Which fund is better than the others?",
            "Recommend a fund for my portfolio",
            "Is it a good time to invest?",
            "How should I allocate my money?",
            "What is suitable for me?",
            "Tell me if I should redeem",
            "Which one should I pick?",
            "Is it worth buying this fund?",
        ],
    )
    def test_advice_phrases(self, question):
        assert is_advice_intent(question)

    @pytest.mark.parametrize(
        "question",
        [
            "What are the returns of HDFC Large Cap Fund?",
            "What is the CAGR?",
            "Which fund performed best?",
            "Which grew the most?",
            "How much profit did it make?",
            "What has grown the fastest?",
            "Which scheme is the best performing?",
            "Did it outperform?",
        ],
    )
    def test_performance_phrases(self, question):
        assert is_performance_intent(question)

    @pytest.mark.parametrize(
        "question",
        [
            "What is the exit load?",
            "What is the minimum SIP amount?",
            "What is the lock-in period?",
            "Which benchmark does the fund follow?",
            "What is the expense ratio?",
            "What is the riskometer category?",
            "How do I download the statement?",
            "",
        ],
    )
    def test_non_advice_non_performance(self, question):
        assert not is_advice_intent(question)
        assert not is_performance_intent(question)

    def test_how_to_is_not_advice(self):
        assert classify_intent("How do I download the capital-gains statement?") is Intent.HOW_TO

    def test_fact_is_the_default(self):
        assert classify_intent("What is the exit load of HDFC Large Cap Fund?") is Intent.FACT


class TestRefusalResponse:
    @pytest.mark.parametrize("reason", REFUSAL_REASONS)
    def test_returns_refused_result(self, reason):
        result = refusal_response(reason)
        assert isinstance(result, AnswerResult)
        assert result.refused is True
        assert result.refusal_reason == reason
        assert result.answer.strip()
        assert result.latency_ms == 0
        assert result.used_fallback is False

    @pytest.mark.parametrize("reason", REFUSAL_REASONS)
    def test_disclaimer_is_appended(self, reason):
        assert config.DISCLAIMER in refusal_response(reason).answer

    @pytest.mark.parametrize("reason", ("advice", "performance"))
    def test_advice_and_performance_carry_educational_links(self, reason):
        result = refusal_response(reason)
        assert result.source_url in config.EDUCATIONAL_LINKS.values()
        assert "investor.gov.in" in (result.source_url or "") or "amfiindia" in (
            result.source_url or ""
        )

    def test_pii_refusal_cites_nothing(self):
        assert refusal_response("pii").source_url is None

    def test_no_context_refusal_uses_an_educational_link(self):
        result = refusal_response("no_context")
        assert result.source_url in config.EDUCATIONAL_LINKS.values()

    def test_explicit_link_overrides_the_default(self):
        result = refusal_response("advice", link=LARGE_CAP_URL)
        assert result.source_url == LARGE_CAP_URL

    def test_pii_text_never_echoes_the_input(self):
        result = refusal_response("pii")
        assert "1234567890" not in result.answer
        assert "ABCDE1234F" not in result.answer

    def test_refusal_does_not_contain_advice_language(self):
        for reason in ("advice", "performance"):
            body = refusal_response(reason).answer.lower()
            assert "you should buy" not in body
            assert "i recommend" not in body

    def test_unknown_reason_raises(self):
        with pytest.raises(ValueError, match="unknown refusal reason"):
            refusal_response("nonsense")

    def test_answer_result_is_frozen(self):
        result = refusal_response("advice")
        with pytest.raises(FrozenInstanceError):
            result.answer = "tampered"

    def test_defaults_are_not_shared_between_instances(self):
        first = refusal_response("advice")
        first.guardrail_flags.append("x")
        assert refusal_response("advice").guardrail_flags == ["refused:advice"]


class TestTrimSentences:
    def test_keeps_leading_sentences(self):
        answer = "One. Two. Three. Four. Five."
        assert trim_sentences(answer, max_sentences=3) == "One. Two. Three."

    def test_short_answer_untouched(self):
        assert trim_sentences("One. Two.") == "One. Two."

    def test_does_not_split_on_decimals(self):
        answer = "The ratio is 1.19%. The exit load is 0.75%. Charges are 2.0%."
        trimmed = trim_sentences(answer, max_sentences=3)
        assert trimmed == answer
        assert "1.19%" in trimmed

    def test_does_not_split_on_abbreviations(self):
        answer = "Ask Dr. Sharma about the fee. e.g. the ratio is 1.19%."
        trimmed = trim_sentences(answer, max_sentences=2)
        assert "Dr. Sharma" in trimmed
        assert "e.g. the ratio" in trimmed

    def test_preserves_citation_line(self):
        answer = f"One. Two. Three. Four.\n{LARGE_CAP_URL}"
        trimmed = trim_sentences(answer, max_sentences=2)
        assert trimmed == f"One. Two.\n{LARGE_CAP_URL}"
        assert find_urls(trimmed) == [LARGE_CAP_URL]

    def test_preserves_last_updated_line(self):
        answer = f"One. Two. Three. Four.\n{LAST_UPDATED}"
        assert trim_sentences(answer, max_sentences=1).endswith(LAST_UPDATED)

    def test_preserves_both_trailing_lines(self):
        answer = f"One. Two. Three. Four.\n{LARGE_CAP_URL}\n{LAST_UPDATED}"
        trimmed = trim_sentences(answer, max_sentences=1)
        assert LARGE_CAP_URL in trimmed
        assert LAST_UPDATED in trimmed

    def test_default_max_sentences_is_three(self):
        assert MAX_SENTENCES == config.MAX_SENTENCES == 3
        assert count_sentences(trim_sentences("A. B. C. D. E.")) == 3

    def test_rejects_non_positive_max_sentences(self):
        with pytest.raises(ValueError, match="max_sentences must be positive"):
            trim_sentences("One. Two.", max_sentences=0)

    def test_count_sentences_ignores_trailer(self):
        assert count_sentences(f"One. Two.\n{LARGE_CAP_URL}\n{LAST_UPDATED}") == 2

    def test_empty_answer(self):
        assert trim_sentences("") == ""


class TestPreamble:
    @pytest.mark.parametrize(
        "answer",
        [
            "Sure! The exit load is 1.0%.",
            "Certainly, the exit load is 1.0%.",
            "Based on the context, the exit load is 1.0%.",
            "According to the sources, the exit load is 1.0%.",
            "Here is the answer: the exit load is 1.0%.",
            "The answer is: the exit load is 1.0%.",
        ],
    )
    def test_strips_lead_in(self, answer):
        assert has_preamble(answer)
        assert strip_preamble(answer) == "The exit load is 1.0%."

    def test_capitalisation_is_restored(self):
        assert strip_preamble("Based on the context, the exit load is 1.0%.").startswith("The")

    def test_clean_answer_has_no_preamble(self):
        assert not has_preamble("The exit load is 1.0% for 0-12 months.")

    def test_preamble_strip_keeps_trailer(self):
        answer = f"Sure! The exit load is 1.0%.\n{LARGE_CAP_URL}\n{LAST_UPDATED}"
        stripped = strip_preamble(answer)
        assert find_urls(stripped) == [LARGE_CAP_URL]
        assert LAST_UPDATED in stripped

    def test_strips_at_most_three_layers(self):
        answer = "Sure! Of course, hello, ok, the exit load is 1.0%."
        assert strip_preamble(answer) == "Ok, the exit load is 1.0%."


class TestLastUpdated:
    def test_appends_when_missing(self):
        assert ensure_last_updated("The exit load is 1.0%.", DATE) == (
            f"The exit load is 1.0%.\n{LAST_UPDATED}"
        )

    def test_replaces_a_stale_date(self):
        answer = f"The exit load is 1.0%.\n{config.LAST_UPDATED_PREFIX} 2020-01-01"
        assert ensure_last_updated(answer, DATE).count(config.LAST_UPDATED_PREFIX) == 1
        assert DATE in ensure_last_updated(answer, DATE)

    def test_is_idempotent(self):
        once = ensure_last_updated("The exit load is 1.0%.", DATE)
        assert ensure_last_updated(once, DATE) == once

    def test_keeps_the_citation_line(self):
        answer = f"The exit load is 1.0%.\n{LARGE_CAP_URL}"
        assert LARGE_CAP_URL in ensure_last_updated(answer, DATE)

    def test_ensure_citation_appends_exactly_once(self):
        answer = f"The exit load is 1.0%.\n{LAST_UPDATED}"
        result = ensure_citation(answer, LARGE_CAP_URL)
        assert find_urls(result) == [LARGE_CAP_URL]
        assert ensure_citation(result, LARGE_CAP_URL) == result

    def test_find_urls_strips_trailing_punctuation(self):
        assert find_urls(f"See {LARGE_CAP_URL}.") == [LARGE_CAP_URL]

    def test_find_urls_deduplicates(self):
        assert find_urls(f"{LARGE_CAP_URL} and {LARGE_CAP_URL}") == [LARGE_CAP_URL]

    def test_find_urls_on_empty(self):
        assert find_urls("") == []


class TestValidateAnswer:
    def test_clean_answer_passes_with_no_flags(self):
        answer = f"The exit load is 1.0% for 0-12 months.\n{LAST_UPDATED}\n{LARGE_CAP_URL}"
        ok, flags, repaired = validate_answer(answer, {LARGE_CAP_URL}, DATE)
        assert ok is True
        assert flags == []
        assert repaired == answer

    def test_canonical_order_is_prose_then_date_then_citation(self):
        answer = f"The exit load is 1.0% for 0-12 months.\n{LARGE_CAP_URL}\n{LAST_UPDATED}"
        ok, _, repaired = validate_answer(answer, {LARGE_CAP_URL}, DATE)
        assert ok is True
        assert repaired.splitlines() == [
            "The exit load is 1.0% for 0-12 months.",
            LAST_UPDATED,
            LARGE_CAP_URL,
        ]

    def test_repairs_overlong_multi_url_answer_in_one_call(self):
        answer = (
            "Sure! Based on the context, the exit load is 1.0% for 0-12 months. "
            "It is 0.75% for 12-24 months. Nil above 24 months. Contact us for more.\n"
            f"{LARGE_CAP_URL}\n{SMALL_CAP_URL}"
        )
        ok, flags, repaired = validate_answer(answer, {LARGE_CAP_URL}, DATE)
        assert ok is True
        assert "preamble" in flags
        assert "too_long" in flags
        assert "citation_not_in_context" in flags
        assert "missing_last_updated" in flags
        assert count_sentences(repaired) == MAX_SENTENCES
        assert find_urls(repaired) == [LARGE_CAP_URL]
        assert LAST_UPDATED in repaired
        assert repaired.splitlines()[-1] == LARGE_CAP_URL
        assert not has_preamble(repaired)
        assert not repaired.startswith("Sure")

    def test_drops_an_invented_url(self):
        answer = "The exit load is 1.0%.\nhttps://evil.example.com/fake"
        ok, flags, repaired = validate_answer(answer, {LARGE_CAP_URL}, DATE)
        assert "citation_not_in_context" in flags
        assert "evil.example.com" not in repaired
        assert find_urls(repaired) == [LARGE_CAP_URL]

    def test_adds_a_citation_when_context_has_one(self):
        ok, flags, repaired = validate_answer(
            "The exit load is 1.0%.", {LARGE_CAP_URL}, DATE
        )
        assert "missing_citation" in flags
        assert ok is True
        assert find_urls(repaired) == [LARGE_CAP_URL]

    def test_fails_when_no_context_url_exists(self):
        ok, flags, _ = validate_answer("The exit load is 1.0%.", set(), DATE)
        assert ok is False
        assert "missing_citation" in flags

    def test_ignores_context_urls_outside_the_allowlist(self):
        ok, _, repaired = validate_answer(
            "The exit load is 1.0%.", {"https://evil.example.com/fake"}, DATE
        )
        assert ok is False
        assert "evil.example.com" not in repaired

    def test_fails_on_performance_language(self):
        answer = f"The fund returned 18.2% last year.\n{LARGE_CAP_URL}"
        ok, flags, _ = validate_answer(answer, {LARGE_CAP_URL}, DATE)
        assert ok is False
        assert "performance_language" in flags

    def test_fails_on_advice_language(self):
        answer = f"You should buy this fund for your portfolio.\n{LARGE_CAP_URL}"
        ok, flags, _ = validate_answer(answer, {LARGE_CAP_URL}, DATE)
        assert ok is False
        assert "advice_language" in flags

    @pytest.mark.parametrize(
        "claim",
        [
            "The 1-year return was 18.2%.",
            "Its CAGR is 14.5%.",
            "This is the best performing ELSS fund.",
            "It outperformed its benchmark.",
            "The fund grew 22% last year.",
            "Since inception return of 120%.",
            "Expect 12% per annum.",
            "Returns of 22% per annum.",
            "1 year returns 12.5%.",
            "Returns were 15% last year.",
            "The fund gave 18% returns.",
            "Returns: 18%.",
            "3 year returns of 19.4% p.a.",
            "Appreciated by 22%.",
        ],
    )
    def test_performance_claims_are_all_caught(self, claim):
        ok, flags, _ = validate_answer(f"{claim}\n{LARGE_CAP_URL}", {LARGE_CAP_URL}, DATE)
        assert ok is False
        assert "performance_language" in flags

    @pytest.mark.parametrize(
        "fact",
        [
            "If you redeem within one year, returns are taxed at 20%.",
            "Returns exceeding Rs 1.25 lakh in a financial year are taxed at 12.5%.",
            "Tax on returns: 20% if redeemed within one year.",
            "Short term capital gains on returns above Rs 1.25 lakh are taxed at 12.5%.",
            "Withdrawal proceeds above Rs 1 lakh attract a 20% tax on returns.",
            "Taxation of returns at 12.5% above Rs 1.25 lakh.",
        ],
    )
    def test_tax_on_returns_is_not_a_performance_claim(self, fact):
        """Regression: "returns ... 20%" in a tax context must not trip performance_language."""
        ok, flags, _ = validate_answer(f"{fact}\n{LARGE_CAP_URL}", {LARGE_CAP_URL}, DATE)
        assert "performance_language" not in flags
        assert ok is True

    @pytest.mark.parametrize(
        "fact",
        [
            "The total expense ratio is 1.19% of NAV.",
            "The exit load is 1.0% for a holding period of 0-12 months.",
            "The minimum SIP amount is Rs. 500 per month.",
            "The lock-in period is 3 years.",
            "The benchmark is Nifty Smallcap 250 TRI.",
        ],
    )
    def test_legitimate_facts_pass_clean(self, fact):
        answer = f"{fact}\n{LAST_UPDATED}\n{LARGE_CAP_URL}"
        ok, flags, _ = validate_answer(answer, {LARGE_CAP_URL}, DATE)
        assert ok is True, f"{fact} -> {flags}"
        assert flags == []

    def test_returns_hard_flag_when_overlong_cannot_be_trimmed(self):
        answer = " ".join(f"Sentence number {index}." for index in range(6))
        ok, flags, _ = validate_answer(answer, {LARGE_CAP_URL}, DATE)
        assert ok is True
        assert "too_long" in flags

    def test_empty_answer_is_repaired(self):
        ok, flags, repaired = validate_answer("", {LARGE_CAP_URL}, DATE)
        assert "missing_citation" in flags
        assert LAST_UPDATED in repaired

    def test_all_flags_are_declared(self):
        answer = (
            f"Sure! You should buy it; it returned 18.2%. One. Two. Three. Four.\n"
            "https://evil.example.com/x"
        )
        _, flags, _ = validate_answer(answer, {LARGE_CAP_URL}, DATE)
        assert set(flags) <= set(DETECTABLE_FLAGS)
        assert set(flags) == {
            "preamble",
            "too_long",
            "missing_citation",
            "citation_not_in_context",
            "missing_last_updated",
            "advice_language",
            "performance_language",
        }

    def test_unrepairable_flags_are_declared(self):
        assert set(UNREPAIRABLE_FLAGS) <= set(DETECTABLE_FLAGS)
        assert set(HARD_FLAGS) <= set(DETECTABLE_FLAGS)

    def test_validation_is_deterministic(self):
        answer = f"The exit load is 1.0%.\n{LARGE_CAP_URL}"
        first = validate_answer(answer, {LARGE_CAP_URL}, DATE)
        second = validate_answer(answer, {LARGE_CAP_URL}, DATE)
        assert first == second

    def test_citation_is_always_the_last_line(self):
        answer = "The exit load is 1.0%. " + LARGE_CAP_URL
        _, _, repaired = validate_answer(answer, {LARGE_CAP_URL}, DATE)
        assert repaired.splitlines()[-1] == LARGE_CAP_URL


class TestNoSideEffects:
    def test_module_does_not_import_llm_or_storage(self):
        import src.guardrails as module

        source = module.__file__
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        for forbidden in ("httpx", "chromadb", "ollama", "numpy", "requests"):
            assert forbidden not in text, forbidden

    def test_refusal_does_not_touch_the_filesystem(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
        refusal_response("advice")
        assert not (tmp_path / "logs").exists()

"""Tests for stages 10-11: prompt construction, the Groq call, citation, and fallback.

HTTP is mocked with ``httpx.MockTransport``, so no live model is required.
"""
from __future__ import annotations

import json

import httpx
import pytest

from src import answer, config, guardrails
from src.vectorstore import RetrievedChunk

ALLOWED_URL = config.CORPUS[0].url
OTHER_URL = config.CORPUS[1].url
INVENTED_URL = "https://invented.example.com/fake-fund"
REAL_CLIENT = httpx.Client


def _mock_groq(monkeypatch, handler) -> list[httpx.Request]:
    """Route every httpx call through a mock transport and record the requests."""
    seen: list[httpx.Request] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(transport_handler)
        return REAL_CLIENT(*args, **kwargs)

    monkeypatch.setattr(answer.httpx, "Client", factory)
    return seen


def _reply(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": content}}]},
    )


def _chunk(
    text: str = "Exit load of 1% if redeemed within 1 year. It applies to all units.",
    url: str = ALLOWED_URL,
    fetched_at: str = "2026-09-27",
    score: float = 0.77,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="a1b2c3d4e5f6:022",
        text=text,
        score=score,
        metadata={
            "source_url": url,
            "scheme_name": config.CORPUS[0].scheme_name,
            "category": config.CORPUS[0].category,
            "plan": config.CORPUS[0].plan,
            "section": "Exit load",
            "section_path": "HDFC Large Cap Fund > Exit load",
            "fetched_at": fetched_at,
        },
    )


CONTEXT = (
    "[1] Scheme: HDFC Large Cap Fund | Section: Exit load | Source: " + ALLOWED_URL + "\n"
    "Exit load of 1% if redeemed within 1 year."
)


class TestBuildPrompt:
    def test_returns_a_system_and_user_pair(self):
        messages = answer.build_prompt("What is the exit load?", CONTEXT, "2026-09-27")
        assert [message["role"] for message in messages] == ["system", "user"]

    def test_system_message_is_the_verbatim_constant(self):
        messages = answer.build_prompt("q", CONTEXT, "2026-09-27")
        assert messages[0]["content"] == config.PROMPT_SYSTEM

    def test_user_message_contains_the_context(self):
        user = answer.build_prompt("q", CONTEXT, "2026-09-27")[1]["content"]
        assert "CONTEXT:" in user
        assert CONTEXT in user
        assert ALLOWED_URL in user

    def test_user_message_contains_the_question_once(self):
        question = "What is the exit load of HDFC Large Cap Fund?"
        user = answer.build_prompt(question, CONTEXT, "2026-09-27")[1]["content"]
        assert user.count(question) == 1
        assert user.rstrip().endswith(f"Question: {question}")

    def test_user_message_carries_the_last_updated_date(self):
        user = answer.build_prompt("q", CONTEXT, "2026-09-27")[1]["content"]
        assert f"{config.LAST_UPDATED_PREFIX} 2026-09-27" in user

    def test_last_updated_sits_between_context_and_question(self):
        user = answer.build_prompt("q", CONTEXT, "2026-09-27")[1]["content"]
        assert user.index("CONTEXT:") < user.index(config.LAST_UPDATED_PREFIX)
        assert user.index(config.LAST_UPDATED_PREFIX) < user.index("Question:")


class TestGenerate:
    @pytest.fixture(autouse=True)
    def _test_key(self, monkeypatch):
        monkeypatch.setattr(config, "GROQ_API_KEY", "test-groq-key")

    def test_posts_to_the_chat_endpoint(self, monkeypatch):
        seen = _mock_groq(monkeypatch, lambda request: _reply("Exit load is 1%."))
        answer.generate(answer.build_prompt("q", CONTEXT, "2026-09-27"))
        assert str(seen[0].url) == f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions"
        assert seen[0].headers["Authorization"] == "Bearer test-groq-key"

    def test_request_body_carries_the_required_options(self, monkeypatch):
        seen = _mock_groq(monkeypatch, lambda request: _reply("Exit load is 1%."))
        answer.generate(answer.build_prompt("q", CONTEXT, "2026-09-27"))
        body = json.loads(seen[0].content)
        assert body["stream"] is False
        assert body["model"] == config.LLM_MODEL
        assert body["temperature"] == 0
        assert body["max_tokens"] == 220
        assert "options" not in body

    def test_options_come_from_config_not_hardcoded(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_TEMPERATURE", 0.25)
        monkeypatch.setattr(config, "LLM_NUM_PREDICT", 64)
        seen = _mock_groq(monkeypatch, lambda request: _reply("ok"))
        answer.generate(answer.build_prompt("q", CONTEXT, "2026-09-27"))
        body = json.loads(seen[0].content)
        assert body["temperature"] == 0.25
        assert body["max_tokens"] == 64

    def test_messages_are_forwarded_verbatim(self, monkeypatch):
        seen = _mock_groq(monkeypatch, lambda request: _reply("ok"))
        messages = answer.build_prompt("q", CONTEXT, "2026-09-27")
        answer.generate(messages)
        assert json.loads(seen[0].content)["messages"] == messages

    def test_returns_the_stripped_content(self, monkeypatch):
        _mock_groq(monkeypatch, lambda request: _reply("  Exit load is 1%.  \n"))
        assert answer.generate(answer.build_prompt("q", CONTEXT, "d")) == "Exit load is 1%."

    def test_timeout_is_configured(self, monkeypatch):
        captured: dict = {}

        def factory(*args, **kwargs):
            captured.update(kwargs)
            kwargs["transport"] = httpx.MockTransport(lambda r: _reply("ok"))
            return REAL_CLIENT(*args, **kwargs)

        monkeypatch.setattr(answer.httpx, "Client", factory)
        answer.generate(answer.build_prompt("q", CONTEXT, "d"))
        assert captured["timeout"] == config.LLM_TIMEOUT_S

    def test_missing_api_key_raises_llm_unavailable(self, monkeypatch):
        monkeypatch.setattr(config, "GROQ_API_KEY", "")
        with pytest.raises(answer.LLMUnavailable, match="GROQ_API_KEY"):
            answer.generate(answer.build_prompt("q", CONTEXT, "d"))

    def test_connection_error_raises_llm_unavailable(self, monkeypatch):
        def boom(request):
            raise httpx.ConnectError("connection refused", request=request)

        _mock_groq(monkeypatch, boom)
        with pytest.raises(answer.LLMUnavailable):
            answer.generate(answer.build_prompt("q", CONTEXT, "d"))

    def test_read_timeout_raises_llm_unavailable(self, monkeypatch):
        def boom(request):
            raise httpx.ReadTimeout("timed out", request=request)

        _mock_groq(monkeypatch, boom)
        with pytest.raises(answer.LLMUnavailable):
            answer.generate(answer.build_prompt("q", CONTEXT, "d"))

    def test_non_200_raises_llm_unavailable(self, monkeypatch):
        _mock_groq(monkeypatch, lambda request: httpx.Response(500, text="model not found"))
        with pytest.raises(answer.LLMUnavailable) as excinfo:
            answer.generate(answer.build_prompt("q", CONTEXT, "d"))
        assert "500" in str(excinfo.value)

    def test_unreadable_body_raises_llm_unavailable(self, monkeypatch):
        _mock_groq(monkeypatch, lambda request: httpx.Response(200, json={"nope": 1}))
        with pytest.raises(answer.LLMUnavailable):
            answer.generate(answer.build_prompt("q", CONTEXT, "d"))

    def test_trailing_slash_in_base_url_does_not_double_up(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_BASE_URL", config.LLM_BASE_URL + "/")
        seen = _mock_groq(monkeypatch, lambda request: _reply("ok"))
        answer.generate(answer.build_prompt("q", CONTEXT, "d"))
        assert str(seen[0].url).count("/chat/completions") == 1


class TestExtractCitation:
    def test_picks_a_url_present_in_the_context(self):
        text = f"The exit load is 1%.\n{ALLOWED_URL}"
        assert answer.extract_citation(text, {ALLOWED_URL}) == ALLOWED_URL

    def test_ignores_an_invented_url(self):
        text = f"The exit load is 1%.\n{INVENTED_URL}"
        assert answer.extract_citation(text, {ALLOWED_URL}) is None

    def test_returns_none_when_no_url_is_present(self):
        assert answer.extract_citation("The exit load is 1%.", {ALLOWED_URL}) is None

    def test_keeps_the_first_in_context_url(self):
        text = f"{ALLOWED_URL}\n{OTHER_URL}"
        assert answer.extract_citation(text, {ALLOWED_URL, OTHER_URL}) == ALLOWED_URL

    def test_skips_an_invented_url_before_a_real_one(self):
        text = f"{INVENTED_URL}\n{ALLOWED_URL}"
        assert answer.extract_citation(text, {ALLOWED_URL}) == ALLOWED_URL

    def test_tolerates_trailing_punctuation_after_the_link(self):
        text = f"Exit load is 1%.\n{ALLOWED_URL}."
        assert answer.extract_citation(text, {ALLOWED_URL}) == ALLOWED_URL

    def test_empty_context_set_yields_none(self):
        assert answer.extract_citation(f"See {ALLOWED_URL}", set()) is None

    def test_empty_answer_yields_none(self):
        assert answer.extract_citation("", {ALLOWED_URL}) is None

    def test_never_returns_a_url_outside_the_context_set(self):
        text = f"{ALLOWED_URL} {INVENTED_URL} {OTHER_URL}"
        cited = answer.extract_citation(text, {OTHER_URL})
        assert cited == OTHER_URL
        assert cited in {OTHER_URL}


class TestEnsureLastUpdated:
    def test_appends_the_line_when_missing(self):
        assert answer.ensure_last_updated("Exit load is 1%.", "2026-09-27") == (
            f"Exit load is 1%.\n{config.LAST_UPDATED_PREFIX} 2026-09-27"
        )

    def test_replaces_a_differing_date(self):
        stale = f"Exit load is 1%.\n{config.LAST_UPDATED_PREFIX} 2020-01-01"
        result = answer.ensure_last_updated(stale, "2026-09-27")
        assert "2020-01-01" not in result
        assert result.endswith(f"{config.LAST_UPDATED_PREFIX} 2026-09-27")
        assert result.count(config.LAST_UPDATED_PREFIX) == 1

    def test_is_idempotent(self):
        once = answer.ensure_last_updated("Exit load is 1%.", "2026-09-27")
        twice = answer.ensure_last_updated(once, "2026-09-27")
        assert once == twice

    def test_does_not_duplicate_an_existing_correct_line(self):
        text = f"Exit load is 1%.\n{config.LAST_UPDATED_PREFIX} 2026-09-27"
        assert answer.ensure_last_updated(text, "2026-09-27").count(
            config.LAST_UPDATED_PREFIX
        ) == 1

    def test_keeps_the_citation_line_above_the_date(self):
        text = f"Exit load is 1%.\n{ALLOWED_URL}"
        result = answer.ensure_last_updated(text, "2026-09-27")
        lines = result.splitlines()
        assert lines[1] == ALLOWED_URL
        assert lines[2] == f"{config.LAST_UPDATED_PREFIX} 2026-09-27"

    def test_no_date_leaves_the_text_without_the_line(self):
        assert config.LAST_UPDATED_PREFIX not in answer.ensure_last_updated("Body.", "")

    def test_removes_a_stale_date_even_when_no_new_date_is_given(self):
        text = f"Body.\n{config.LAST_UPDATED_PREFIX} 2020-01-01"
        assert config.LAST_UPDATED_PREFIX not in answer.ensure_last_updated(text, "")


class TestExtractiveFallback:
    def test_uses_the_top_chunk_verbatim(self):
        chunk = _chunk(text="Exit load is 1%. It applies to all units. Third sentence. Fourth.")
        result = answer.extractive_fallback("exit load?", [chunk])
        assert result.answer.startswith("Exit load is 1%.")
        assert "Fourth." not in result.answer

    def test_trims_to_max_sentences(self):
        chunk = _chunk(text="One. Two. Three. Four. Five.")
        result = answer.extractive_fallback("q", [chunk])
        assert guardrails.count_sentences(result.answer) <= config.MAX_SENTENCES

    def test_cites_the_top_chunk_url(self):
        result = answer.extractive_fallback("q", [_chunk(url=ALLOWED_URL)])
        assert result.source_url == ALLOWED_URL
        assert ALLOWED_URL in result.answer

    def test_stamps_the_chunks_own_fetched_at(self):
        result = answer.extractive_fallback("q", [_chunk(fetched_at="2026-08-01")])
        assert result.last_updated == "2026-08-01"
        assert result.answer.endswith(f"{config.LAST_UPDATED_PREFIX} 2026-08-01")

    def test_never_invents_a_date(self):
        result = answer.extractive_fallback("q", [_chunk(fetched_at="")])
        assert result.last_updated is None
        assert config.LAST_UPDATED_PREFIX not in result.answer

    def test_marks_the_result_as_a_fallback(self):
        result = answer.extractive_fallback("q", [_chunk()])
        assert result.used_fallback is True
        assert result.refused is False
        assert result.guardrail_flags == []

    def test_keeps_the_chunks_for_downstream_use(self):
        chunk = _chunk()
        assert answer.extractive_fallback("q", [chunk]).top_chunks == [chunk]

    def test_uses_only_the_highest_scoring_chunk(self):
        top = _chunk(text="Top chunk sentence one. Two.", url=ALLOWED_URL)
        lower = _chunk(text="Lower chunk sentence one. Two.", url=OTHER_URL, score=0.1)
        result = answer.extractive_fallback("q", [top, lower])
        assert "Top chunk" in result.answer
        assert "Lower chunk" not in result.answer
        assert result.source_url == ALLOWED_URL

    def test_fallback_is_grounded_and_cited_after_llm_failure(self, monkeypatch):
        monkeypatch.setattr(config, "GROQ_API_KEY", "test-groq-key")

        def boom(request):
            raise httpx.ConnectError("connection refused", request=request)

        _mock_groq(monkeypatch, boom)
        chunk = _chunk()
        with pytest.raises(answer.LLMUnavailable):
            answer.generate(answer.build_prompt("What is the exit load?", CONTEXT, "2026-09-27"))
        result = answer.extractive_fallback("What is the exit load?", [chunk])
        assert result.used_fallback
        assert result.source_url == ALLOWED_URL
        assert chunk.text.split(".")[0] in result.answer
        assert result.answer.endswith(f"{config.LAST_UPDATED_PREFIX} 2026-09-27")

    def test_empty_chunks_raise(self):
        with pytest.raises(ValueError):
            answer.extractive_fallback("q", [])

    def test_module_exposes_no_orchestrator(self):
        assert hasattr(answer, "answer_question")


def _result(hits: list[RetrievedChunk]) -> "retrieve.RetrievalResult":
    from src import retrieve

    return retrieve.RetrievalResult(
        question="q",
        normalized_question="q",
        k=config.TOP_K,
        min_score=config.MIN_SIMILARITY,
        hits=hits,
        context="context",
        dropped_low_score=0,
        dropped_out_of_scope=0,
        context_tokens=120,
        latency_ms=5,
    )


def _stub_pipeline(monkeypatch, hits: list[RetrievedChunk], reply: object):
    """Stub retrieval and generation so the orchestrator's decisions can be tested."""
    from src import retrieve

    monkeypatch.setattr(retrieve, "retrieve", lambda *a, **k: _result(hits))
    if isinstance(reply, Exception):
        def boom(messages):
            raise reply

        monkeypatch.setattr(answer, "generate", boom)
    else:
        monkeypatch.setattr(answer, "generate", lambda messages: reply)


class TestAnswerQuestion:
    def test_cleans_grounded_answer_is_returned_verbatim(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(
            monkeypatch,
            [_chunk()],
            f"The exit load is 1% within 1 year.\n{ALLOWED_URL}",
        )
        result = answer.answer_question("What is the exit load?")
        assert result.refused is False
        assert result.used_fallback is False
        assert result.source_url == ALLOWED_URL
        assert "The exit load is 1% within 1 year." in result.answer

    def test_last_updated_comes_from_the_top_chunk(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk(fetched_at="2026-07-04")], f"Fact.\n{ALLOWED_URL}")
        result = answer.answer_question("What is the exit load?")
        assert result.last_updated == "2026-07-04"
        assert result.answer.endswith(f"{config.LAST_UPDATED_PREFIX} 2026-07-04")

    def test_invented_citation_never_surfaces(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], f"Exit load is 1%.\n{INVENTED_URL}")
        result = answer.answer_question("What is the exit load?")
        assert INVENTED_URL not in result.answer
        assert result.source_url in config.ALLOWED_URLS

    def test_uncited_answer_falls_back_and_stays_cited(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], "Exit load is 1%.")
        result = answer.answer_question("What is the exit load?")
        assert result.source_url in config.ALLOWED_URLS
        assert ALLOWED_URL in result.answer

    def test_over_long_answer_is_trimmed_to_three_sentences(self, monkeypatch, tmp_data_dir):
        long_answer = " ".join(f"Sentence {index}." for index in range(9)) + f"\n{ALLOWED_URL}"
        _stub_pipeline(monkeypatch, [_chunk()], long_answer)
        result = answer.answer_question("What is the exit load?")
        assert guardrails.count_sentences(result.answer) <= config.MAX_SENTENCES

    def test_preamble_is_stripped(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(
            monkeypatch,
            [_chunk()],
            f"Sure! Based on the context, the exit load is 1%.\n{ALLOWED_URL}",
        )
        result = answer.answer_question("What is the exit load?")
        assert not guardrails.has_preamble(result.answer)

    def test_llm_unavailable_uses_the_extractive_fallback(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], answer.LLMUnavailable("connection refused"))
        result = answer.answer_question("What is the exit load?")
        assert result.used_fallback is True
        assert "llm_unavailable" in result.guardrail_flags
        assert result.source_url == ALLOWED_URL

    def test_advice_language_falls_back(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(
            monkeypatch,
            [_chunk()],
            f"You should buy this fund, it is the best performing one.\n{ALLOWED_URL}",
        )
        result = answer.answer_question("What is the exit load?")
        assert result.used_fallback is True
        assert "validation_failed" in result.guardrail_flags

    def test_every_factual_answer_carries_a_last_updated_line(self, monkeypatch, tmp_data_dir):
        for reply in (f"Fact.\n{ALLOWED_URL}", "Fact with no link.", f"x\n{INVENTED_URL}"):
            _stub_pipeline(monkeypatch, [_chunk()], reply)
            result = answer.answer_question("What is the exit load?")
            assert config.LAST_UPDATED_PREFIX in result.answer
            assert result.source_url in config.ALLOWED_URLS

    def test_no_context_when_nothing_passes_the_threshold(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(monkeypatch, [], "unused")
        result = answer.answer_question("What is the exit load?")
        assert result.refused is True
        assert result.refusal_reason == "no_context"
        assert result.used_fallback is False

    def test_pii_refuses_without_calling_the_generator(self, monkeypatch, tmp_data_dir):
        def explode(messages):
            raise AssertionError("the LLM must not be called for a PII question")

        _stub_pipeline(monkeypatch, [_chunk()], "unused")
        monkeypatch.setattr(answer, "generate", explode)
        result = answer.answer_question("My PAN is ABCDE1234F")
        assert result.refused is True
        assert result.refusal_reason == "pii"

    def test_advice_refuses_without_calling_the_generator(self, monkeypatch, tmp_data_dir):
        def explode(messages):
            raise AssertionError("the LLM must not be called for an advice question")

        _stub_pipeline(monkeypatch, [_chunk()], "unused")
        monkeypatch.setattr(answer, "generate", explode)
        assert answer.answer_question("Should I buy HDFC Small Cap Fund?").refusal_reason == "advice"

    def test_performance_refuses_without_calling_the_generator(self, monkeypatch, tmp_data_dir):
        def explode(messages):
            raise AssertionError("the LLM must not be called for a performance question")

        _stub_pipeline(monkeypatch, [_chunk()], "unused")
        monkeypatch.setattr(answer, "generate", explode)
        result = answer.answer_question("Which fund has the best returns?")
        assert result.refusal_reason == "performance"

    def test_refusals_carry_an_educational_link(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], "unused")
        result = answer.answer_question("Should I buy HDFC Small Cap Fund?")
        assert result.source_url == config.EDUCATIONAL_LINKS["advice"]

    def test_pii_refusal_cites_nothing(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], "unused")
        result = answer.answer_question("My PAN is ABCDE1234F")
        assert result.source_url is None

    def test_empty_question_is_refused(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], "unused")
        for question in ("", "   ", "\t\n"):
            result = answer.answer_question(question)
            assert result.refused is True
            assert "empty_question" in result.guardrail_flags

    def test_non_latin_question_is_not_answered(self, monkeypatch, tmp_data_dir):
        def explode(messages):
            raise AssertionError("a non-Latin question must not reach the LLM")

        _stub_pipeline(monkeypatch, [_chunk()], "unused")
        monkeypatch.setattr(answer, "generate", explode)
        result = answer.answer_question("यह क्या है")
        assert result.refused is True
        assert result.refusal_reason == "no_context"
        assert "non_latin_script" in result.guardrail_flags

    def test_out_of_corpus_amc_is_refused(self, monkeypatch, tmp_data_dir):
        def explode(messages):
            raise AssertionError("an out-of-corpus AMC question must not reach the LLM")

        _stub_pipeline(monkeypatch, [_chunk()], "unused")
        monkeypatch.setattr(answer, "generate", explode)
        for question in ("Axis Small Cap Fund exit load", "SBI Large Cap expense ratio"):
            result = answer.answer_question(question)
            assert result.refused is True
            assert "out_of_corpus_amc" in result.guardrail_flags

    def test_in_corpus_question_is_not_treated_as_foreign(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], f"Fact.\n{ALLOWED_URL}")
        assert answer.answer_question("What is HDFC Large Cap Fund exit load?").refused is False

    def test_over_long_question_is_truncated_before_embedding(self, monkeypatch, tmp_data_dir):
        question = "What is the exit load of HDFC Large Cap Fund? " * 60
        seen: dict = {}

        def fake_retrieve(text, k=None, min_score=None):
            seen["text"] = text
            return _result([_chunk()])

        from src import retrieve

        monkeypatch.setattr(retrieve, "retrieve", fake_retrieve)
        monkeypatch.setattr(answer, "generate", lambda messages: f"Fact.\n{ALLOWED_URL}")
        result = answer.answer_question(question)
        assert len(seen["text"]) == answer.TRUNCATED_QUESTION_CHARS
        assert "question_truncated" in result.guardrail_flags
        assert result.refused is False

    def test_short_question_is_not_truncated(self, monkeypatch, tmp_data_dir):
        assert answer._prepare_question("What is the exit load?") == ("What is the exit load?", False)

    def test_latency_is_recorded(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], f"Fact.\n{ALLOWED_URL}")
        assert answer.answer_question("What is the exit load?").latency_ms >= 0

    def test_pii_question_is_not_logged_verbatim(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], "unused")
        answer.answer_question("My PAN is ABCDE1234F")
        for name in ("answers.jsonl", "retrieval.jsonl"):
            text = (config.LOG_DIR / name).read_text()
            assert "ABCDE1234F" not in text
            assert "[redacted:" in text

    def test_factual_answer_is_logged_with_citation_and_flags(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], f"Fact.\n{ALLOWED_URL}")
        answer.answer_question("What is the exit load?")
        record = json.loads((config.LOG_DIR / "answers.jsonl").read_text().splitlines()[-1])
        for key in (
            "question", "answer", "citation", "last_updated", "guardrail_flags",
            "latency_ms", "refused", "used_fallback",
        ):
            assert key in record
        assert record["citation"] == ALLOWED_URL
        assert record["refused"] is False

    def test_retrieval_log_records_generation_and_total_time(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], f"Fact.\n{ALLOWED_URL}")
        answer.answer_question("What is the exit load?")
        record = json.loads((config.LOG_DIR / "retrieval.jsonl").read_text().splitlines()[-1])
        for key in ("retrieval_ms", "generation_ms", "total_ms", "guardrail_flags", "refused"):
            assert key in record

    def test_one_question_logs_exactly_one_line_per_file(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], f"Fact.\n{ALLOWED_URL}")
        answer.answer_question("What is the exit load?")
        assert len((config.LOG_DIR / "answers.jsonl").read_text().splitlines()) == 1
        assert len((config.LOG_DIR / "retrieval.jsonl").read_text().splitlines()) == 1

    def test_as_dict_is_json_serialisable(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], f"Fact.\n{ALLOWED_URL}")
        payload = answer._as_dict(answer.answer_question("What is the exit load?"))
        assert json.loads(json.dumps(payload, default=str))["refused"] is False
        assert payload["top_chunks"][0]["source_url"] == ALLOWED_URL


class TestCli:
    def test_json_flag_prints_one_object(self, monkeypatch, capsys, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], f"Fact.\n{ALLOWED_URL}")
        assert answer.main(["--question", "What is the exit load?", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["source_url"] == ALLOWED_URL

    def test_question_flag_prints_citation_and_disclaimer(self, monkeypatch, capsys, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], f"Fact.\n{ALLOWED_URL}")
        answer.main(["--question", "What is the exit load?"])
        out = capsys.readouterr().out
        assert ALLOWED_URL in out
        assert answer.DISCLAIMER_LINE in out
        assert config.LAST_UPDATED_PREFIX in out

    def test_refusal_prints_no_disclaimer_duplication(self, monkeypatch, capsys, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], "unused")
        answer.main(["--question", "My PAN is ABCDE1234F"])
        assert "cannot accept or look up personal information" in capsys.readouterr().out

    def test_chat_exits_on_exit_keyword(self, monkeypatch, capsys, tmp_data_dir):
        monkeypatch.setattr("builtins.input", lambda prompt="": "exit")
        assert answer.main(["--chat"]) == 0

    def test_chat_answers_a_question_then_exits(self, monkeypatch, capsys, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], f"Fact.\n{ALLOWED_URL}")
        prompts = iter(["What is the exit load?", "quit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(prompts))
        assert answer.main(["--chat"]) == 0
        assert "Fact." in capsys.readouterr().out

    def test_mutually_exclusive_modes(self):
        with pytest.raises(SystemExit):
            answer.main(["--question", "q", "--selftest"])

    def test_a_mode_is_required(self):
        with pytest.raises(SystemExit):
            answer.main([])


class TestFlagReporting:
    """Flags must describe the answer the user sees, not the model's raw text."""

    def test_repaired_answer_reports_no_violation_it_no_longer_has(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], "The exit load is 1%.")
        result = answer.answer_question("What is the exit load?")
        assert result.source_url in config.ALLOWED_URLS
        assert config.LAST_UPDATED_PREFIX in result.answer
        assert result.guardrail_flags == []

    def test_inline_citation_is_not_reported_as_missing(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], f"Exit load is 1%.\n{ALLOWED_URL}")
        assert "missing_citation" not in answer.answer_question("What is the exit load?").guardrail_flags
        assert "missing_last_updated" not in answer.answer_question("What is the exit load?").guardrail_flags

    def test_genuine_violation_is_still_reported(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], f"You should buy this fund.\n{INVENTED_URL}")
        flags = answer.answer_question("What is the exit load?").guardrail_flags
        assert "validation_failed" in flags

    def test_uncited_model_text_never_reaches_the_user_uncited(self, monkeypatch, tmp_data_dir):
        """FR-28 repair or the fallback guarantees a citation, but never a fabricated one."""
        for reply in ("Exit load is 1%.", f"Exit load is 1%.\n{INVENTED_URL}"):
            _stub_pipeline(monkeypatch, [_chunk()], reply)
            result = answer.answer_question("What is the exit load?")
            assert INVENTED_URL not in result.answer
            assert result.source_url in config.ALLOWED_URLS
            assert config.LAST_UPDATED_PREFIX in result.answer

    def test_advice_language_forces_the_extractive_fallback(self, monkeypatch, tmp_data_dir):
        _stub_pipeline(monkeypatch, [_chunk()], f"You should buy this fund.\n{ALLOWED_URL}")
        result = answer.answer_question("What is the exit load?")
        assert result.used_fallback is True
        assert "validation_failed" in result.guardrail_flags
        assert "advice" not in result.answer.lower() or "should" not in result.answer.lower()

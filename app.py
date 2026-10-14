"""Streamlit UI for the HDFC Mutual Fund FAQ Assistant (implementation.md §PHASE 9).

This module holds no business logic: it renders the ``AnswerResult`` returned by
``src.answer.answer_question`` and nothing else. State lives in ``st.session_state``
for the current turn only, so no chat history is ever persisted (FR-34).
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import answer, config, guardrails  # noqa: E402  (path setup must run first)

PAGE_TITLE = "HDFC Mutual Fund FAQ Assistant"
ANSWER_FOOTER = "Facts only. Not investment advice."
REFUSAL_LABEL = "I cannot answer that"

EXAMPLE_QUESTIONS: tuple[str, ...] = (
    "What is the exit load of HDFC Large Cap Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
    "How do I download the capital-gains statement?",
)

INGEST_STAGES: tuple[str, ...] = (
    "1. Load — fetch the five allowlisted HDFC pages",
    "2. Clean — strip navigation and boilerplate, keep headings and tables",
    "3. Chunk — heading-aware split with fact protection and provenance headers",
    "4. Embed — all-MiniLM-L6-v2, 384 dimensions, local",
    "5. Store — ChromaDB, cosine similarity, persistent",
)

RETRIEVAL_STAGES: tuple[str, ...] = (
    "6. Normalize and embed the question",
    "7. Similarity search over the five schemes",
    "8. Threshold and allowlist filter",
    "9. Deduplicate and diversify (MMR)",
    "10. Assemble context and build a grounded prompt",
    "11. Generate the answer (Groq)",
    "12. Validate citation, length, and claims",
    "13. Log the turn",
)

st.set_page_config(page_title=PAGE_TITLE, page_icon="📄", layout="wide")

ANSWER_CSS = """
<style>
.faq-disclaimer {
    border-left: 4px solid #b26a00;
    background-color: #fff8e6;
    padding: 0.65rem 0.9rem;
    border-radius: 0.25rem;
    margin-bottom: 1rem;
    font-size: 0.92rem;
}
.faq-answer {
    border-left: 4px solid #1c6b3c;
    background-color: #f2f8f4;
    padding: 0.85rem 1rem;
    border-radius: 0.25rem;
    margin-bottom: 0.6rem;
}
.faq-refusal {
    border-left: 4px solid #8a1c1c;
    background-color: #fdf2f2;
    padding: 0.85rem 1rem;
    border-radius: 0.25rem;
    margin-bottom: 0.6rem;
}
.faq-refusal-label {
    font-weight: 600;
    color: #8a1c1c;
    text-transform: uppercase;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
    margin-bottom: 0.4rem;
}
.faq-footer {
    color: #555555;
    font-size: 0.85rem;
    font-style: italic;
}
.faq-meta {
    color: #555555;
    font-size: 0.85rem;
}
</style>
"""
st.markdown(ANSWER_CSS, unsafe_allow_html=True)


def scheme_for(url: str | None) -> "config.SourceSpec | None":
    """Return the allowlisted source behind a citation, if there is one."""
    return next((spec for spec in config.CORPUS if spec.url == url), None)


def source_label(url: str | None) -> str:
    """Label a citation with its scheme name, or name the educational reference."""
    if not url:
        return "Source"
    spec = scheme_for(url)
    if spec is None:
        return "Investor education reference"
    return f"{spec.scheme_name} ({spec.plan})"


def render_answer(result: guardrails.AnswerResult) -> None:
    """Render one AnswerResult: text, exactly one link, last-updated line, footer."""
    if result.refused:
        st.markdown('<div class="faq-refusal">', unsafe_allow_html=True)
        st.markdown(f'<div class="faq-refusal-label">{REFUSAL_LABEL}</div>', unsafe_allow_html=True)
        st.markdown(result.answer)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="faq-answer">', unsafe_allow_html=True)
        st.markdown(result.answer)
        st.markdown("</div>", unsafe_allow_html=True)

    if result.source_url:
        st.markdown(f"[{source_label(result.source_url)}]({result.source_url})")
    else:
        st.markdown('<span class="faq-meta">No source page applies to this answer.</span>',
                    unsafe_allow_html=True)

    if result.last_updated:
        st.caption(f"{config.LAST_UPDATED_PREFIX} {result.last_updated}")
    st.markdown(f'<div class="faq-footer">{ANSWER_FOOTER}</div>', unsafe_allow_html=True)


def ask(question: str) -> None:
    """Run one turn and keep it for this session only (FR-34: nothing is persisted)."""
    if not question.strip():
        return
    with st.spinner("Looking this up in the indexed sources..."):
        result = answer.answer_question(question)
    st.session_state["question"] = question
    st.session_state["result"] = result


def submit_question() -> None:
    """Answer what was typed, then empty the box.

    Both steps run in the submit callback, before the next script run: the widget key can
    only be cleared legally there, and the typed text is still readable while it runs.
    Either write after the widget is instantiated raises
    ``StreamlitWidgetAlreadyInstantiatedError``.
    """
    question = str(st.session_state.get("question_box") or "")
    st.session_state["question_box"] = ""
    ask(question)


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Scope")
        st.caption("Facts about these five HDFC schemes only.")
        for spec in config.CORPUS:
            st.markdown(f"- {spec.scheme_name} ({spec.category})")

        st.header("Sources")
        for spec in config.CORPUS:
            st.markdown(f"- [{spec.scheme_name}]({spec.url})")

        st.header("How an answer is produced")
        st.markdown("**Ingestion (runs once, offline)**")
        for stage in INGEST_STAGES:
            st.markdown(stage)
        st.markdown("**Question answering**")
        for stage in RETRIEVAL_STAGES:
            st.markdown(stage)

        st.header("Questions this assistant can answer")
        for index, question in enumerate(EXAMPLE_QUESTIONS, start=1):
            st.markdown(f"{index}. {question}")
        st.caption("Scheme terms and how-to steps. It will not recommend a fund, "
                   "compare performance, or accept personal details.")


def main() -> None:
    st.title(PAGE_TITLE)
    st.markdown(
        "Facts about five HDFC Asset Management schemes, answered only from their "
        "public scheme pages: Large Cap, Flexi Cap, ELSS Tax Saver, Small Cap, and "
        "Balanced Advantage."
    )

    render_sidebar()

    st.markdown(f'<div class="faq-disclaimer">{config.DISCLAIMER}</div>',
                unsafe_allow_html=True)

    columns = st.columns(len(EXAMPLE_QUESTIONS))
    for column, question in zip(columns, EXAMPLE_QUESTIONS):
        if column.button(question, key=f"chip-{question}", use_container_width=True):
            ask(question)

    with st.form("ask_form"):
        st.text_input(
            "Ask a question about these schemes",
            key="question_box",
            placeholder="e.g. What is the expense ratio of HDFC Equity Fund?",
        )
        st.form_submit_button("Ask", type="primary", on_click=submit_question)

    if "result" in st.session_state:
        st.divider()
        st.caption(f"You asked: {st.session_state['question']}")
        render_answer(st.session_state["result"])


main()
"""Central configuration: every tunable value in the RAG pipeline lives here."""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env before any os.environ lookup below. Real environment variables win, so CI
# and shell exports override the file, and a missing python-dotenv is not fatal.
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pass
else:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

DATA_DIR = Path(os.environ.get("RAG_DATA_DIR") or (PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"
LOG_DIR = PROJECT_ROOT / "logs"

MANIFEST_PATH = PROJECT_ROOT / "corpus_manifest.json"
CHUNKING_REPORT_PATH = LOG_DIR / "chunking_report.json"


@dataclass(frozen=True)
class SourceSpec:
    """One allowlisted public page in the corpus."""

    url: str
    scheme_name: str
    category: str
    plan: str


CORPUS: list[SourceSpec] = [
    SourceSpec(
        url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        scheme_name="HDFC Large Cap Fund",
        category="Large Cap",
        plan="Direct Growth",
    ),
    SourceSpec(
        url="https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
        scheme_name="HDFC Equity Fund",
        category="Flexi Cap",
        plan="Direct Growth",
    ),
    SourceSpec(
        url="https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
        scheme_name="HDFC ELSS Tax Saver Fund",
        category="ELSS",
        plan="Direct Growth",
    ),
    SourceSpec(
        url="https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
        scheme_name="HDFC Small Cap Fund",
        category="Small Cap",
        plan="Direct Growth",
    ),
    SourceSpec(
        url="https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
        scheme_name="HDFC Balanced Advantage Fund",
        category="Balanced Advantage",
        plan="Direct Growth",
    ),
]

ALLOWED_URLS: frozenset[str] = frozenset(spec.url for spec in CORPUS)

HF_HOME = os.environ.get("HF_HOME") or None

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBED_BATCH_SIZE = 32
NORM_TOLERANCE = 1e-3

CHUNK_SIZE_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 90
MIN_CHUNK_TOKENS = 25
CHARS_PER_TOKEN = 4
CHUNK_CHARS = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN

MUST_HAVE_FACTS: dict[str, list[str]] = {
    "HDFC Large Cap Fund": ["expense ratio", "exit load", "SIP", "benchmark", "risk"],
    "HDFC Equity Fund": ["expense ratio", "exit load", "SIP", "benchmark", "risk"],
    "HDFC ELSS Tax Saver Fund": [
        "expense ratio",
        "lock-in",
        "3 years",
        "capital gains",
        "benchmark",
    ],
    "HDFC Small Cap Fund": ["expense ratio", "exit load", "SIP", "benchmark", "risk"],
    "HDFC Balanced Advantage Fund": ["expense ratio", "exit load", "benchmark", "risk"],
}

COLLECTION_NAME = "mf_faq_hdfc_v1"
HNSW_SPACE = "cosine"

TOP_K = 5
MIN_SIMILARITY = 0.45
MMR_LAMBDA = 0.7
MAX_CHUNKS_PER_SECTION = 2
MAX_CONTEXT_TOKENS = 2400
MAX_QUESTION_CHARS = 500
CANDIDATE_POOL = 50
LEXICAL_ALPHA = 0.65

# Generation is Groq only (GroqChat, OpenAI-compatible). Embeddings stay local MiniLM.
LLM_PROVIDER = "groq"
_REQUESTED_PROVIDER = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
if _REQUESTED_PROVIDER and _REQUESTED_PROVIDER != LLM_PROVIDER:
    # Non-fatal on purpose: retrieval and ingest need no LLM, so a stale provider name in
    # .env must not break them. Generation still goes to Groq, and the warning says so.
    warnings.warn(
        f"LLM_PROVIDER={_REQUESTED_PROVIDER!r} is ignored; this project generates with "
        f"{LLM_PROVIDER!r} only. Set LLM_PROVIDER={LLM_PROVIDER} in .env to silence this.",
        RuntimeWarning,
        stacklevel=2,
    )
LLM_MODEL = (
    os.environ.get("GROQ_MODEL")
    or os.environ.get("LLM_MODEL")
    or "openai/gpt-oss-120b"
).strip()
LLM_BASE_URL = (os.environ.get("GROQ_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/")
GROQ_API_KEY = (os.environ.get("GROQ_API_KEY") or "").strip()
LLM_TEMPERATURE = 0.0
LLM_NUM_PREDICT = 220
LLM_NUM_CTX = 4096
LLM_TIMEOUT_S = 90


def configured_provider() -> str:
    """Return the generation provider. This project uses Groq only."""
    return LLM_PROVIDER

MAX_SENTENCES = 3
LAST_UPDATED_PREFIX = "Last updated from sources:"

PROMPT_SYSTEM: str = """You are a mutual fund FAQ assistant for HDFC Asset Management schemes
(Large Cap, Flexi Cap / Equity, ELSS Tax Saver, Small Cap, Balanced Advantage —
all Direct Growth plans).

Rules, in strict order of priority:
1. Answer ONLY from the CONTEXT blocks. Never use outside knowledge.
2. If the answer is not in the context, reply exactly:
   "Not found in the indexed sources." and cite the most relevant source URL.
3. Factual answers: at most 3 sentences, plain text, no headings, no bullet lists.
4. End every factual answer with the source link on its own final line, exactly as
   written in the context, and nothing else on that line.
5. Never give investment advice, recommendations, opinions, suitability, rankings,
   or comparisons of "better"/"best". If asked, reply with the facts-only refusal
   text and an educational link.
6. Never state, compute, compare, or estimate returns, CAGR, NAV performance, or
   rankings. If asked, say performance data is in the official factsheet and cite it.
7. Do not ask for or record PAN, Aadhaar, account numbers, OTPs, email, or phone.
8. Do not invent numbers. If a figure is missing, say so."""

REFUSAL_ADVICE = (
    "I can share facts about HDFC scheme terms, but I cannot give investment advice, "
    "recommendations, or tell you whether to buy or sell a fund. Please read the scheme's "
    "SID/KIM and factsheet, and consider a SEBI-registered investment advisor for guidance "
    "on your own situation."
)

REFUSAL_PII = (
    "I cannot accept or look up personal information. Please do not share your PAN, Aadhaar "
    "number, account number, OTP, email address, or phone number with any assistant. I can "
    "only answer general questions about publicly published HDFC scheme pages."
)

REFUSAL_NO_CONTEXT = (
    "Not found in the indexed sources. I can only answer from the HDFC scheme pages listed "
    "in the source list, so please rephrase the question using terms from those pages."
)

EDUCATIONAL_LINKS: dict[str, str] = {
    "advice": "https://www.investor.gov.in/",
    "performance": "https://www.amfiindia.com/investor-education",
    "how_to": "https://www.hdfcassetmanagement.com/about-us/investor-education",
}

DISCLAIMER = (
    "Facts only. No investment advice. Answers are generated from HDFC public pages listed "
    "in the source list and may be out of date. Mutual fund investments are subject to "
    "market risks; read the scheme's SID/KIM and factsheet before investing. Please do not "
    "share PAN, Aadhaar, account numbers, OTPs, or any personal information."
)

USER_AGENT = "rag-mf-faq-demo/0.1 (class project; contact: student@example.edu)"
HTTP_TIMEOUT_S = 20
HTTP_RETRIES = 1
POLITENESS_DELAY_S = 1.0

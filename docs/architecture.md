# Architecture — Mutual Fund FAQ Assistant (Facts-Only RAG Chatbot)

**Derived from:** `PRD.md` v1.0
**Scope of this document:** how the PRD's functional requirements are realized in code and
data flow — data ingestion stages (1–5) and data retrieval stages (6–13).
**Status:** Draft v1.0

---

## 1. Design Principles

| # | Principle | Consequence in the design |
|---|-----------|---------------------------|
| P1 | **Stages are explicit and separable** | One module per RAG stage; each has a pure-ish input/output contract so a stage can be swapped or ablated without touching the rest |
| P2 | **Grounding over fluency** | The LLM is a *renderer* of retrieved text, not a knowledge source; every claim must trace to a chunk |
| P3 | **Deterministic and reproducible** | `temperature=0`, fixed chunk params, cached raw HTML, content-hash corpus versioning |
| P4 | **Local retrieval, hosted Groq** | Local embeddings and Chroma; answers are rendered by Groq |
| P5 | **Constraints enforced in code, not just in prompts** | Post-generation validators (citation, length, advice/return language, PII) can reject a draft answer |
| P6 | **Fail soft, never fail the demo** | Every stage has a documented fallback (extractive answer, "not found in sources", cached HTML) |
| P7 | **Scope is enforced by allowlist** | Only the 5 corpus URLs may ever be cited; enforced at ingest, retrieval, and output |

---

## 2. System Context

```
            ┌───────────────────────────────┐
  (offline) │  Public web (once, at ingest)│   5 allowlisted pages
            └───────────────┬───────────────┘   (Groww / HDFC / SEBI / AMFI)
                            │ HTTPS GET
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  LOCAL RAG SERVICE  (no network at query time)                       │
│                                                                      │
│   INGESTION (offline)  ──▶  PERSISTED STATE  ◀──  RETRIEVAL (online) │
│   load→clean→chunk      data/raw, data/processed,   query→search→     │
│   →embed→store         data/chroma, manifest        ground→generate   │
│                                                    →guardrail→answer  │
└───────────────┬──────────────────────────────────────┬───────────────┘
                │                                      │
                ▼                                      ▼
        data/corpus_manifest.json            ┌──────────────────┐
        corpus_manifest + Chroma collection  │ UI (Streamlit)  │
        data/chroma/mf_faq_hdfc_v1            │ CLI alternative │
                                             └──────────────────┘
                                                    │
                                          ┌─────────▼──────────┐
                                          │ Groq (hosted)     │
                                          │ Chat Completions   │
                                          └────────────────────┘
```

**Trust boundaries:** ingest fetches only allowlisted hosts. At query time embeddings
stay local; generation POSTs the retrieved context to Groq. Logs redact PII.

---

## 3. Component Architecture

```
┌───────────────────────────── UI LAYER ──────────────────────────────┐
│ app.py  (Streamlit)                                                │
│  • welcome line (scope: HDFC · 5 schemes)                           │
│  • 3 example-question chips                                        │
│  • persistent disclaimer banner                                     │
│  • answer card: answer text · 1 source link · last-updated date     │
│  • state: session-only (no chat-history DB)                         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  answer_question(question) -> AnswerResult
┌───────────────────────────────▼─────────────────────────────────────┐
│ ORCHESTRATION  src/answer.py :: answer_question()                   │
│  order: PII check → intent route → retrieve → build_context         │
│         → build_prompt → generate → validate → repair/refuse        │
└───┬───────────────┬────────────────┬──────────────────┬─────────────┘
    │               │                │                  │
    ▼               ▼                ▼                  ▼
guardrails.py   retrieve.py       vectorstore.py      embed.py
  · detect_pii     · normalize       · PersistentClient · all-MiniLM
  · is_advice      · route_intent    · get_or_create    · embed_texts
  · refusal        · retrieve        · upsert_chunks    · embed_query
  · validate_      · diversify/MMR   · search           (cached model)
    answer         · assert_         · stats
  · trim_          allowlisted                            ▲
    sentences                                        ┌───┴────┐
                                                     │  STORE │
┌───────────────────────────────┐                     └────────┘
│ OFFLINE PIPELINE              │
│ ingest.py    ─┐               │                    ┌──────────────┐
│ chunking.py  ─┼─ run.sh ─▶ vectorstore.upsert_chunks ─▶ ChromaDB   │
│ embed.py     ─┘                                     └──────────────┘
│      │ writes
│      ▼
│ data/raw/*.html · data/processed/*.md · corpus_manifest.json
└──────────────────────────────────────────────────────────────────────┘
```

### Module responsibilities (PRD §5, Appendix C)

| Module | Stage(s) | Responsibility | Depends on |
|--------|----------|----------------|------------|
| `src/config.py` | — | Corpus allowlist, chunk params, thresholds, prompt constants, paths. Single source of truth | — |
| `src/ingest.py` | 1–2 | HTTP fetch with cache, main-content extraction, HTML→Markdown, manifest writing | config, fs |
| `src/chunking.py` | 3 | Heading-aware recursive split, atomic-fact protection, provenance headers, fact-integrity validation | config |
| `src/embed.py` | 4 | `all-MiniLM-L6-v2` singleton, batch document embedding, query embedding, 384-dim assertion | sentence-transformers |
| `src/vectorstore.py` | 5 | Chroma persistent client, collection create/reset/upsert/search/stats | chromadb, embed |
| `src/retrieve.py` | 6–9 | Query normalization, intent routing, similarity search + threshold, allowlist enforcement, MMR dedupe, context assembly | config, embed, vectorstore |
| `src/guardrails.py` | 12 | PII patterns, advice patterns, refusal builder, answer validators (citation/length/claims), sentence trimmer | config |
| `src/answer.py` | 10–11, 13 | Prompt construction, LLM call, citation extraction, last-updated injection, extractive fallback, orchestration | all above |
| `src/logger.py` | — | JSONL retrieval/answer log | — |
| `app.py` | — | UI: scope line, 3 examples, disclaimer, answer + link rendering | answer |

---

## 4. Ingestion Pipeline (Stages 1–5)

### 4.1 Stage flow

```
Stage 1  LOAD
  sources.csv / config.CORPUS (5 SourceSpec)
      │  HTTP GET, User-Agent set, 20s timeout, 1 retry, 1s politeness delay
      │  cache hit? → data/raw/<doc_id>.html   (skip network: NFR-06)
      ▼
  data/raw/<doc_id>.html  +  fetched_at (ISO date)  +  sha256(html)

Stage 2  CLEAN
  trafilatura / bs4 main-content extraction
      │  drop: nav, header, footer, cookie banner, login prompt, "related funds" block
      ▼
  <page_title> + Markdown body (headings, lists, tables preserved)

Stage 3  CHUNK  (see §5)
      ▼
  Chunk[doc_id, ordinal, text(+provenance header), token_count, metadata]

Stage 4  EMBED
  all-MiniLM-L6-v2, batch size 32, normalize_embeddings=True
      ▼
  float32 matrix [n_chunks, 384]

Stage 5  STORE
  chromadb.PersistentClient(data/chroma)
  get_or_create_collection("mf_faq_hdfc_v1",
      metadata={"hnsw:space": "cosine", "corpus_version": <manifest sha>})
  upsert(ids=chunk_id, embeddings=..., documents=..., metadatas=...)
      ▼
  data/chroma/  +  corpus_manifest.json
```

### 4.2 Corpus manifest (`corpus_manifest.json`)

Single artifact that makes the corpus auditable and versioned; also the source of the
`Last updated from sources:` value.

```json
{
  "corpus_version": "mf_faq_hdfc_v1",
  "generated_at": "2026-09-27T14:02:11Z",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "chunking": { "strategy": "recursive_heading_aware", "chunk_tokens": 600,
                "overlap_tokens": 90, "preserve_tables": true },
  "documents": [
    {
      "doc_id": "a1b2c3d4e5f6",
      "scheme_name": "HDFC Large Cap Fund",
      "category": "Large Cap",
      "plan": "Direct Growth",
      "url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
      "page_title": "HDFC Large Cap Fund Direct Growth ...",
      "fetched_at": "2026-09-27",
      "content_sha256": "9f2c…",
      "raw_path": "data/raw/a1b2c3d4e5f6.html",
      "processed_path": "data/processed/a1b2c3d4e5f6.md",
      "chunk_count": 41
    }
  ],
  "totals": { "documents": 5, "chunks": 212 }
}
```

**ID conventions**

| ID | Derivation | Purpose |
|----|-----------|---------|
| `doc_id` | `sha1(url)[:12]` | Stable page identity across re-ingests |
| `chunk_id` | `f"{doc_id}:{ordinal:03d}"` | Stable chunk identity → idempotent upsert (FR-05) |
| `corpus_version` | `mf_faq_hdfc_v1` + manifest hash | Chroma collection name; a changed corpus means a new collection, not silent overwrite |

**Idempotency (FR-05):** `upsert_chunks` upserts by `chunk_id`; stale chunks (ordinals beyond
the new count) are pruned at the end of ingest. Two consecutive runs on unchanged HTML yield
identical chunk counts and IDs.

---

## 5. Chunking Strategy — Design Detail

**Decision (ADR-1, PRD FR-06…FR-10):** heading-aware **recursive** character splitting.

**Why not semantic chunking:** the corpus is structured scheme pages — definitions, fee
tables, exit-load slabs, minimum-SIP lines. Semantic chunking is nondeterministic across runs
(breaks P3), costs an extra embedding pass, and offers no benefit when natural section
boundaries already exist in the markup.

**Algorithm (`chunking.split_text`)**

```
1. Parse Markdown into sections:  [(H1 title, body), (H2 "Fees and expenses", body), …]
2. For each section:
     a. Protect atomic facts — treat each Markdown table row and each "label: value"
        line as an indivisible unit (never split mid-row).
     b. Recursively split the body with separators:
         ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]
        using RecursiveCharacterTextSplitter(chunk_size=600, overlap=90)
     c. Re-attach the section heading + breadcrumb to every emitted chunk.
3. Prepend provenance header (FR-09):
     "Scheme: HDFC Small Cap Fund | Plan: Direct Growth | Category: Small Cap
      | Section: Exit load | Source: https://groww.in/…"
4. Drop chunks with < 25 tokens and no digits/keywords (boilerplate guard).
5. Verify fact integrity (FR-08): each must-have fact string
   (e.g. "1% exit load", "₹500", "3 years") is fully contained in ≥ 1 chunk;
   failures are logged to logs/chunking_report.json and surfaced in the README.
```

**Parameter rationale**

| Param | Value | Reason |
|-------|-------|--------|
| `chunk_size` | 600 tokens | A Groww fee/exit-load block is 150–500 tokens; 600 keeps a whole fact block with its heading while staying well inside the 4096 `num_ctx` budget for 4–5 chunks |
| `chunk_overlap` | 90 tokens | ~15% — enough to preserve a table header row when a section straddles a boundary, small enough that duplicates are rare (FR-17 still enforces per-section caps) |
| Separators | headings → paragraphs → sentences → words | Preserves meaning hierarchy; headings are the highest-value boundary |
| Embeddings | normalized | Enables inner-product = cosine, and makes the Chroma threshold directly interpretable |

---

## 6. Data Contracts

### 6.1 Chunk record persisted in Chroma

| Field | Type | Source | Notes |
|-------|------|--------|-------|
| `id` | str | `chunk_id` | Primary key |
| `document` | str | chunk text **with** provenance header | What the embedder sees |
| `embedding` | float32[384] | all-MiniLM-L6-v2 | L2-normalized |
| `metadata.source_url` | str | manifest | Used for the citation and allowlist check (FR-16) |
| `metadata.scheme_name` | str | manifest | Answer prefix / scheme disambiguation |
| `metadata.category` | str | manifest | Large Cap, Flexi Cap, ELSS, Small Cap, Balanced Advantage |
| `metadata.plan` | str | manifest | Always `Direct Growth` in this corpus |
| `metadata.section` | str | chunker | Nearest preceding heading |
| `metadata.page_title` | str | manifest | Human-readable provenance |
| `metadata.fetched_at` | str (ISO date) | manifest | Feeds `Last updated from sources:` |
| `metadata.chunk_ordinal` | int | chunker | Provenance/debug |

### 6.2 Runtime result types (`src/answer.py`, `src/retrieve.py`)

```python
class Intent(StrEnum):
    FACT = "fact"
    HOW_TO = "how_to"
    ADVICE = "advice"
    PII = "pii"

@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float          # cosine similarity in [0,1]
    metadata: dict

@dataclass(frozen=True)
class RetrievalResult:
    question: str
    chunks: list[RetrievedChunk]   # post-threshold, post-dedupe, post-MMR
    dropped_low_score: int
    dropped_out_of_scope: int
    latency_ms: int

@dataclass(frozen=True)
class AnswerResult:
    answer: str                    # <= 3 sentences, or refusal text
    source_url: str | None         # exactly one citation, or None on refusal
    last_updated: str | None       # ISO date
    refused: bool
    refusal_reason: str | None     # "advice" | "pii" | "out_of_scope" | "no_context"
    guardrail_flags: list[str]     # e.g. ["missing_citation","too_long","perf_claim"]
    used_fallback: bool            # extractive path
    top_chunks: list[RetrievedChunk]
    latency_ms: int
```

### 6.3 Function contracts (signatures, per module)

```python
# src/ingest.py
fetch_page(spec: SourceSpec, force: bool = False) -> PageDocument
extract_main_content(html: str) -> tuple[str, str]                 # (title, markdown)
build_corpus(sources: list[SourceSpec], force: bool) -> list[PageDocument]
write_manifest(docs: list[PageDocument], chunks_per_doc: dict) -> Path
main(force: bool) -> int                                           # CLI entry

# src/chunking.py
parse_sections(markdown: str) -> list[Section]
split_text(markdown: str, doc: PageDocument) -> list[Chunk]
provenance_header(doc: PageDocument, section: str) -> str
validate_fact_integrity(chunks: list[Chunk]) -> list[str]           # FR-08

# src/embed.py
get_model() -> SentenceTransformer                                  # cached singleton
embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray   # [n,384] normalized
embed_query(text: str) -> list[float]                               # [384] normalized

# src/vectorstore.py
get_client() -> chromadb.PersistentClient
get_or_create_collection(reset: bool = False) -> Collection
upsert_chunks(chunks: list[Chunk], vectors: np.ndarray) -> int
prune_stale(keep_ids: set[str]) -> int
search(query_vec: list[float], k: int, where: dict | None) -> list[RetrievedChunk]
collection_stats() -> dict
reset_collection() -> None

# src/retrieve.py
normalize_query(q: str) -> str
retrieve(question: str, k: int = TOP_K, min_score: float = MIN_SIMILARITY) -> RetrievalResult
diversify(chunks: list[RetrievedChunk], per_section_cap: int) -> list[RetrievedChunk]  # MMR
assert_allowlisted(urls: Iterable[str]) -> None                    # raises OutOfScopeError
build_context(chunks: list[RetrievedChunk]) -> str

# src/guardrails.py
detect_pii(text: str) -> list[str]                                 # pan/aadhaar/otp/…
is_advice_intent(q: str) -> bool
refusal_response(reason: str, link: str | None) -> AnswerResult
MAX_SENTENCES: int = 3
trim_sentences(answer: str, max_sentences: int = MAX_SENTENCES) -> str
validate_answer(answer: str, context_urls: set[str], last_updated: str)
    -> tuple[bool, list[str], str]                                 # (ok, flags, repaired)

# src/answer.py
build_prompt(question: str, context: str, last_updated: str) -> list[dict]
generate(messages: list[dict]) -> str                               # groq, temp 0
extract_citation(answer: str, context_urls: set[str]) -> str | None
ensure_last_updated(answer: str, last_updated: str) -> str
extractive_fallback(question: str, chunks: list[RetrievedChunk]) -> AnswerResult
answer_question(question: str) -> AnswerResult                      # orchestrator
```

---

## 7. Retrieval Pipeline (Stages 6–13)

### 7.1 Orchestration order in `answer.answer_question`

```
 0. PII pre-filter      guardrails.detect_pii(q)          → PII ? refuse : continue   (FR-29)
 1. Intent route        guardrails.is_advice_intent(q)    → ADVICE ? refuse : continue(FR-26,27)
 2. Query normalize     lowercase, strip punctuation, collapse whitespace
 3. Query embed         embed.embed_query(q)                                   (FR-11)
 4. Similarity search   vectorstore.search(v, k=5, cosine)                    (FR-14)
 5. Threshold filter    keep s >= 0.35  (s = 1 - cosine_distance)             (FR-15)
 6. Allowlist filter    assert_allowlisted(urls)  → violation ⇒ drop chunk     (FR-16)
 7. Dedupe + MMR        max 2 chunks per section, λ=0.7                        (FR-17)
 8. Empty ⇒ no_context  "Not found in the indexed sources" + nearest page link(FR-23)
 9. Build context       numbered blocks + provenance lines
10. Build prompt        system rules + context + question                      (FR-20,21,22)
11. Generate            Groq via GroqChat (OpenAI-compatible), temp 0, max_tokens 220  (FR-19)
12. Validate/repair     citation present & in-context? ≤3 sentences? no perf/advice
                        language? last-updated present?                         (FR-20–22,28)
      ├ pass → answer
      └ fail → trim/repair once → if still invalid: extractive_fallback       (FR-25)
13. Log                 logs/retrieval.jsonl + logs/answers.jsonl             (FR-18)
```

### 7.2 Retrieval parameters

| Param | Value | Notes |
|-------|-------|-------|
| `k` (pre-filter) | 5 | Small corpus; 5 gives redundancy for MMR without diluting the prompt |
| `min_score` | 0.35 | Tuned on the 10-query sample set; a "nonsense" query must fall below it |
| `MMR λ` | 0.7 | Favour relevance; reduce to 0.5 if answers cite too many schemes for one question |
| per-section cap | 2 | Prevents a single long section (e.g. "Fees") from filling the whole context |
| context budget | ≤ 2400 tokens | 4 chunks × 600; leaves room in `num_ctx=4096` for system prompt + 220-token output |
| max hops | 1 (single retrieval) | v1 is single-hop; follow-up questions are re-phrased client-side by the UI ("context" is not carried across turns) |

### 7.3 Grounding prompt (`build_prompt`)

**System message (constant, in `config.PROMPT_SYSTEM`)**

```
You are a mutual fund FAQ assistant for HDFC Asset Management schemes
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
8. Do not invent numbers. If a figure is missing, say so.
```

**User message**

```
CONTEXT:
[1] Scheme: HDFC Small Cap Fund | Section: Exit load | Source: <url>
<chunk text>
[2] Scheme: HDFC Small Cap Fund | Section: Minimum investment | Source: <url>
<chunk text>

Last updated from sources: 2026-09-27
Question: <user question>
```

**Post-processing order (deterministic):** strip preambles ("Sure, here is…") →
`trim_sentences(3)` → `ensure_last_updated()` → `extract_citation()` (must be in
`context_urls`, else flag `missing_citation`) → advice/perf pattern scan → numeric-fact
spot check for the 5 must-have fact types.

### 7.4 Guardrail failure handling

| Failure | Detection | Action |
|---------|-----------|--------|
| Advice/portfolio question | `is_advice_intent` (regex + keyword weights) or `ADVICE` intent from router | `refusal_response("advice")` with SEBI/AMFI educational link; no LLM call (cheap, deterministic) |
| PII in input | PAN/Aadhaar/account/OTP/email/phone patterns | `refusal_response("pii")`; input is **not** logged verbatim (only a hashed form) |
| Performance question | `returns|cagr|best performing|which grew|profit` | Redirect to official factsheet link; no numbers |
| Out-of-corpus AMC | "HDFC" absent from question and no chunk over threshold | "Not found in the indexed sources" + link to source list |
| Missing citation after generation | URL not in `context_urls` | One repair pass, then `extractive_fallback` |
| > 3 sentences | sentence count | `trim_sentences(3)` (keeps the leading factual sentences) |
| LLM unavailable (Groq down / missing key) | connection error / timeout / HTTP error | `extractive_fallback`: top chunk's first 3 sentences + link (FR-25) |
| Zero chunks over threshold | `RetrievalResult.chunks == []` | `no_context` refusal |

---

## 8. Layering & Dependency Rules

```
app.py  ──▶  answer.py  ──▶  retrieve.py  ──▶  vectorstore.py
                │               │                    │
                │               └──────▶ embed.py ───┘
                ├──▶ guardrails.py
                ├──▶ logger.py
                └──▶ config.py
ingest.py ──▶ chunking.py ──▶ embed.py ──▶ vectorstore.py ──▶ config.py
all modules ──▶ config.py (no module imports another module's internals)
```

Rules:
1. `config.py` imports nothing from the project.
2. `retrieve.py` never calls the LLM; `answer.py` never touches Chroma directly.
3. `guardrails.py` has no I/O except optional logging; it is trivially unit-testable.
4. No module holds cross-request global state except the cached embedding model and the
   Chroma client handle.
5. `app.py` contains zero business logic — it renders `AnswerResult`.

---

## 9. Configuration (`src/config.py`)

| Key | Value | PRD ref |
|-----|-------|---------|
| `CORPUS` | 5 `SourceSpec(url, scheme_name, category, plan)` | §4 |
| `ALLOWED_URLS` | derived frozenset from `CORPUS` (+ official factsheet URLs if added) | §4 |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-l6-v2` | FR-11 |
| `EMBEDDING_DIM` | 384 | FR-11 |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | 600 / 90 | FR-07 |
| `MIN_CHUNK_TOKENS` | 25 | §5 |
| `COLLECTION_NAME` | `mf_faq_hdfc_v1` | FR-13 |
| `CHROMA_DIR` | `data/chroma` | FR-12 |
| `HNSW_SPACE` | `cosine` | FR-12 |
| `TOP_K` | 5 | FR-14 |
| `MIN_SIMILARITY` | 0.35 | FR-15 |
| `MMR_LAMBDA` / `MAX_CHUNKS_PER_SECTION` | 0.7 / 2 | FR-17 |
| `LLM_MODEL` / `LLM_TEMPERATURE` / `LLM_NUM_PREDICT` / `LLM_NUM_CTX` | `openai/gpt-oss-120b` / 0.0 / 220 / 4096 | FR-19, NFR-02 |
| `MAX_SENTENCES` | 3 | FR-20 |
| `DISCLAIMER`, `REFUSAL_ADVICE`, `REFUSAL_PII`, `EDUCATIONAL_LINKS` | constants | FR-24,27,29, App. A/B |
| `LOG_DIR` | `logs/` | FR-18 |

Env overrides: `HF_HOME`, `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_BASE_URL`, `RAG_DATA_DIR`.
`GROQ_API_KEY` lives in gitignored `.env` only.

---

## 10. Observability

**`logs/retrieval.jsonl`** (one line per query, FR-18) — used for the demo write-up and
threshold tuning:

```json
{"ts":"2026-09-27T14:05:02Z","question":"Exit load of HDFC Large Cap?",
 "intent":"fact","pii":false,"k":5,"min_score":0.35,
 "hits":[{"chunk_id":"a1b2…:007","scheme":"HDFC Large Cap Fund","section":"Exit load","score":0.71}],
 "dropped_low_score":2,"dropped_out_of_scope":0,"context_tokens":1180,
 "retrieval_ms":210,"generation_ms":4200,"total_ms":4460,
 "guardrail_flags":[],"source_url":"https://groww.in/…","refused":false}
```

**`logs/answers.jsonl`** — question, final answer, citation, flags (append-only; no PII ever
written, per FR-29).

**Health/self-check (`python -m src.answer --selftest`)**: collection exists, chunk count >
100, embedding dim = 384, model reachable, one canned query returns a cited answer. Run before
recording the demo video.

---

## 11. Runbook

```bash
# 0. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# set GROQ_API_KEY in .env  (required; chat calls Groq, not Ollama)

# 1. ingestion (stages 1-5)  → data/raw, data/processed, data/chroma, manifest
python -m src.ingest            # add --force to bypass the HTML cache

# 2. inspect / verify chunking decisions
python -m src.chunking --report          # logs/chunking_report.json + fact-integrity result
python -m src.vectorstore --stats        # documents, chunks, collection metadata

# 3. terminal chat
python -m src.answer --chat

# 4. UI
streamlit run app.py

# 5. checks
pytest -q                                 # unit tests per stage
python -m src.answer --selftest
```

**Latency budget (NFR-02)**

| Step | Budget |
|------|--------|
| Query embed (MiniLM, CPU) | ~40 ms |
| Chroma search (n≈200) | ~5 ms |
| Guardrails | ~1 ms |
| Groq, 220 output tokens | typically 1–4 s |
| **Total** | **< 8 s** (meets PRD) |

---

## 12. Requirement Traceability

| PRD ref | Realized by |
|---------|-------------|
| FR-01…05 | `ingest.py` (§4), manifest + `doc_id`/`chunk_id` scheme, `prune_stale` |
| FR-06…10 | `chunking.py` (§5) + `validate_fact_integrity` + README ablation note |
| FR-11…13 | `embed.py`, `vectorstore.py`, collection name/metadata |
| FR-14…18 | `retrieve.py` (§7.2), `assert_allowlisted`, JSONL logging |
| FR-19…25 | `answer.build_prompt/generate`, post-processing order (§7.3), `extractive_fallback` |
| FR-26…29 | `guardrails.py` (§7.4), `Intent`, refusal constants, hashed-PII logging |
| FR-30…34 | `app.py` (welcome line, 3 chips, disclaimer, answer card, session state) |
| NFR-01…08 | local stack, temperature 0, cached HTML, modular layering, 16 GB footprint |

---

## 13. Architecture Decisions (ADR summary)

| ADR | Decision | Rationale | Revisit when |
|-----|----------|-----------|--------------|
| ADR-1 | Recursive heading-aware chunking (600/90) | Structured fee/table pages have natural section boundaries; deterministic | Corpus shifts to long prose (KIM/SID PDFs) → test semantic chunking |
| ADR-2 | Groq via GroqChat (OpenAI-compatible) | Hosted instruction-following for grounded short answers; no local LLM runtime | Key missing or Groq down → extractive fallback (FR-25) |
| ADR-3 | ChromaDB persistent, cosine | Required by PRD; zero-ops, restart-safe | Corpus > 50k chunks or multi-user → pgvector/Qdrant |
| ADR-4 | No agentic LangChain pipeline; direct stage calls | The demo's value is *visible* stage separation; frameworks hide it | Never for the class demo; fine for production |
| ADR-5 | No cross-encoder reranker in v1 | Corpus is tiny; MMR + threshold is sufficient at 5 chunks | Answer quality plateaus → add `bge-reranker-base` as stage 9.5 |
| ADR-6 | No multi-turn memory | Q&A must be independently verifiable; simpler guardrail surface | Conversational UX becomes a requirement |

---

## 14. Extension Points

- **Add a scheme / AMC:** append a `SourceSpec` to `config.CORPUS`, bump `corpus_version`,
  re-run ingest. Guardrails and allowlist adapt automatically.
- **Add official factsheets (PDF):** new ingestor writes Markdown + same manifest shape;
  provenance line marks `source_type=factsheet`, surfaced as the "performance data lives here"
  redirect.
- **Better retrieval:** insert a reranker between stages 9 and 10; `build_context` is the
  single seam.
- **Multi-turn:** wrap `answer_question` with a question-rewriting step that uses only the
  last user turn; guardrails stay unchanged.
- **Different corpus / domain:** this is a *mutual fund* facts-only assistant; the pattern
  transfers, but the prompt rules (no advice, no performance) are domain-specific.

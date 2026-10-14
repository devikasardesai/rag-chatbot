# PRD — Mutual Fund FAQ Assistant (Facts-Only RAG Chatbot)

**Project type:** Academic class demo — Retrieval-Augmented Generation (RAG) pipeline
**Status:** Draft v1.0
**Owner:** RAG Chatbot Team
**Last updated:** 2026-09-27

---

## 1. Problem Statement

Retail investors and support/content teams repeatedly ask the same factual questions about
mutual fund schemes — expense ratio, exit load, minimum SIP amount, ELSS lock-in period,
riskometer, benchmark, and how to download statements. Today these answers are scattered
across AMC pages, factsheets, KIM/SID documents, and SEBI/AMFI notes, which is slow for the
user and expensive for support teams.

We will build a **facts-only RAG chatbot** that answers mutual fund scheme questions using
**only official public pages** from a single AMC. Every answer must cite **one source link**.
The assistant must **refuse opinionated / portfolio questions** ("Should I buy or sell?")
with a polite facts-only message plus an educational link, and must **never give investment
advice, compute returns, or compare performance**.

**Deliverable of record:** a working prototype + demo video.

---

## 2. Goals

| # | Goal | Success signal |
|---|------|----------------|
| G1 | Answer factual MF scheme questions grounded in retrieved source text | ≥ 8 of 10 sample queries answered correctly in manual review |
| G2 | Every answer carries exactly one verifiable source link | 100% of answers in sample Q&A have a link from the approved source list |
| G3 | Zero investment advice / performance claims | 0 advice or return-computation cases in adversarial test set |
| G4 | Demonstrates the complete RAG lifecycle | Ingestion (load → clean → chunk → embed → store) and retrieval (embed → search → filter → ground → generate) stages both implemented and visible in README |
| G5 | Runs locally and cheaply | No paid API keys required; runs on a standard laptop |

### Non-Goals (explicitly out of scope)

- No NAV / live price data, no return calculations, no fund comparison or rankings.
- No portfolio advice, no buy/sell/hold guidance, no goal-based recommendations.
- No user accounts, login, chat history persistence beyond session, or PII of any kind.
- No multi-AMC corpus (single AMC only), no live web crawling at query time.
- No Hindi/regional language support, no voice input.

---

## 3. Target Users

| User | Need | Success measure |
|------|------|-----------------|
| Retail investor comparing HDFC schemes | Quick factual comparison basics (fees, lock-in, benchmark) | Gets a cited fact in one question, without opening 5 tabs |
| Support / content team | Answer repetitive MF questions consistently | Reduces repeated lookups; every reply is traceable to a source |

---

## 4. Scope — Corpus Definition

**AMC in scope:** HDFC Asset Management (one AMC only, per problem statement).

**Schemes in scope (5):** all **Direct – Growth** plans.

| # | Category | Scheme | Source URL |
|---|----------|--------|------------|
| S1 | Large Cap | HDFC Large Cap Fund – Direct Growth | `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth` |
| S2 | Flexi Cap | HDFC Equity Fund (Flexi Cap) – Direct Growth | `https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth` |
| S3 | ELSS (Tax Saver) | HDFC ELSS Tax Saver Fund – Direct Plan – Growth | `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth` |
| S4 | Small Cap | HDFC Small Cap Fund – Direct Growth | `https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth` |
| S5 | Balanced Advantage (Hybrid) | HDFC Balanced Advantage Fund – Direct Growth | `https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth` |

> **Correction noted:** the problem statement contains a typo for S5
> (`httroww.in/...`). The canonical URL above (`groww.in`) is the approved source.

**Source policy**

- Allowed: AMC pages (hdfc.com), Groww scheme pages, SEBI, AMFI, and official factsheets /
  KIM / SID documents linked from those pages.
- Not allowed: third-party blogs, news portals, forums, social media, aggregator content.
- Factsheet / KIM / SID are the authoritative fallback for anything not on the scheme page
  (e.g., exact TER band, statutory riskometer wording).
- The 5 Groww URLs above are the **primary** corpus. Any additional official document added
  must be recorded in `sources.csv` (see §12) before it is ingested.

---

## 5. System Architecture

The build must explicitly cover **all** RAG stages: data ingestion **and** data retrieval.

```
┌──────────────────────────── DATA INGESTION (offline, run once) ────────────────────────────┐
│                                                                                          │
│  [1] LOAD       Fetch 5 public pages (HTTP) → cache raw HTML to /data/raw/*.html          │
│        ↓                                                                                 │
│  [2] CLEAN      Strip nav/ads/boilerplate → main content → Markdown (headings + tables)   │
│        ↓                                                                                 │
│  [3] CHUNK      Recursive, heading-aware split; keep table rows & fact blocks intact      │
│        ↓                                                                                 │
│  [4] EMBED      sentence-transformers/all-MiniLM-L6-v2  (384-dim, normalized)             │
│        ↓                                                                                 │
│  [5] STORE      ChromaDB persistent client → /data/chroma (collection: mf_faq_hdfc)        │
│                                                                                          │
│  Side outputs: corpus_manifest.json (url, title, sha256, fetched_at, last_updated)         │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌──────────────────────────── DATA RETRIEVAL (per user query) ─────────────────────────────┐
│  [6] QUERY PREP   Normalize query; detect intent (fact vs advice vs meta/how-to)          │
│        ↓                                                                                 │
│  [7] EMBED QUERY  all-MiniLM-L6-v2 (same model/space as documents)                        │
│        ↓                                                                                 │
│  [8] SEARCH      Chroma similarity search, top-k = 5, cosine distance threshold             │
│        ↓                                                                                 │
│  [9] FILTER      Enforce corpus scope (allowlisted URLs only) + diversity (MMR)            │
│        ↓                                                                                 │
│ [10] GROUND      Build grounded prompt: context chunks + facts-only rules + citation rule  │
│        ↓                                                                                 │
│ [11] GENERATE    Groq → ≤ 3 sentences + 1 source link + last-updated                   │
│        ↓                                                                                 │
│ [12] GUARDRAIL   Output validator: citation present? advice/return language? length?       │
│        ↓                                                                                 │
│  [13] RESPOND    CLI / tiny web UI reply, or safe refusal + educational link              │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

**Component map**

| Stage | Module | Key technology |
|-------|--------|----------------|
| 1–2 Ingest/clean | `src/ingest.py` | `httpx`/`requests` + `beautifulsoup4`, `markdownify`/`trafilatura` |
| 3 Chunk | `src/chunking.py` | `langchain-text-splitters` (RecursiveCharacterTextSplitter) |
| 4 Embed | `src/embed.py` | `sentence-transformers` → `all-MiniLM-L6-v2` |
| 5 Store | `src/vectorstore.py` | `chromadb` (persistent client) |
| 6–9 Retrieve | `src/retrieve.py` | `chromadb` query + intent router + MMR |
| 10–13 Generate/guard | `src/answer.py`, `src/guardrails.py` | Groq via GroqChat (OpenAI-compatible) |
| UI | `app.py` | Streamlit (or Gradio) — welcome line, 3 examples, disclaimer |

---

## 6. Functional Requirements

### 6.1 Ingestion pipeline

| ID | Requirement | Acceptance |
|----|-------------|-----------|
| FR-01 | Ingest loads all 5 allowlisted URLs and caches raw HTML + fetch timestamp | `data/raw/` contains 5 files; manifest lists 5 URLs with `fetched_at` |
| FR-02 | Cleaning keeps only main scheme content; drops nav, footer, cookie banners, login prompts | No nav/footer text in any chunk; ≥ 80% of chunks are fact-bearing |
| FR-03 | Content is converted to Markdown preserving heading hierarchy and table structure | Chunks for "Fees and expenses" / "Exit load" retain their section heading |
| FR-04 | Every chunk carries metadata: `source_url`, `scheme_name`, `category`, `plan`, `section`, `page_title`, `fetched_at` | Metadata present on 100% of Chroma records |
| FR-05 | Ingestion is idempotent — re-running rebuilds the collection cleanly | 2 consecutive runs yield identical corpus (same chunk count, no duplicates) |

### 6.2 Chunking strategy

**Decision: Recursive, heading-aware character splitting.** Rationale: the corpus is
structured web pages (headings, definition lists, fee tables, not prose essays), so heading
boundaries are the semantically correct cut points; semantic chunking adds cost and
non-determinism for little gain on tabular fee data.

| ID | Requirement | Acceptance |
|----|-------------|-----------|
| FR-06 | Split recursively on markdown heading boundaries first, then paragraph/line, then sentence | No chunk merges two unrelated sections |
| FR-07 | `chunk_size ≈ 500–700` tokens, `chunk_overlap ≈ 80–100` tokens | Configurable via constants; documented in README |
| FR-08 | Atomic facts (expense ratio rows, exit-load slabs, minimum SIP, lock-in, benchmark, riskometer) are never split across chunks | Spot-check: each of these facts is fully contained in ≥ 1 chunk |
| FR-09 | Prepend a provenance header to each chunk before embedding | Chunk text begins with `Scheme · Section · Source URL` |
| FR-10 | A short ablation note (semantic vs recursive) is recorded in the README | README contains a "Chunking strategy decision" section |

### 6.3 Embedding + vector store

| ID | Requirement | Acceptance |
|----|-------------|-----------|
| FR-11 | Use `sentence-transformers/all-MiniLM-L6-v2` (384-dim) for documents and queries | Same model instance/space for both; dimensionality check passes |
| FR-12 | Store in ChromaDB persistent collection with cosine space | `data/chroma/` persists; collection re-loads without re-embedding |
| FR-13 | One collection per corpus version, named in the manifest | `mf_faq_hdfc_v1` present |

### 6.4 Retrieval

| ID | Requirement | Acceptance |
|----|-------------|-----------|
| FR-14 | Query embedding uses the identical model; retrieval is top-k cosine | k default = 5, configurable |
| FR-15 | Apply a minimum similarity threshold; below-threshold matches are dropped | `min_score ≈ 0.30–0.40`, tuned on the sample Q&A set |
| FR-16 | Returned chunks must belong to the allowlisted corpus | Guard in code rejects non-allowlisted `source_url` |
| FR-17 | Final context is deduplicated and de-duplicated per section (max 2 chunks/section) | No repeated section blocks in the prompt |
| FR-18 | Logged per query: top chunks, scores, chosen source URL, latency (for the demo write-up) | `logs/retrieval.log` line per query |

### 6.5 Generation (grounded answer)

| ID | Requirement | Acceptance |
|----|-------------|-----------|
| FR-19 | Model: **Groq** (`openai/gpt-oss-120b` via GroqChat) | `GROQ_API_KEY` set; temperature 0, `max_tokens` documented |
| FR-20 | Answer is **≤ 3 sentences** | Post-generation validator truncates/rejects over-length answers |
| FR-21 | Answer contains **exactly one** citation link from the retrieved context | Citation validator checks URL is in the context chunks |
| FR-22 | Answer ends with `Last updated from sources: <YYYY-MM-DD>` using source `fetched_at` / factsheet date | Present in 100% of answers |
| FR-23 | Model may only use provided context; instructed to say so if the answer is absent | Out-of-context query returns "not found in sources" + link, not a guess |
| FR-24 | Fixed disclaimer line is shown in every response or persistently in the UI | `Facts only. Not investment advice.` |
| FR-25 | If the LLM is unavailable, fall back to an extractive answer (top chunk sentence + link) | Demo never hard-fails |

### 6.6 Refusal / guardrails

| ID | Requirement | Acceptance |
|----|-------------|-----------|
| FR-26 | Intent router classifies queries as `FACT`, `HOW_TO`, or `ADVICE` | ≥ 90% correct on the 10-query test set |
| FR-27 | `ADVICE` (buy/sell/hold, "which is better", portfolio allocation, "should I invest in X?") → polite refusal + facts-only message + 1 educational link (AMC/SEBI investor-education page) | No advice words in refusal path |
| FR-28 | Output validator blocks: return/performance claims, comparative ranking, PII solicitation | Adversarial suite of ≥ 10 queries all blocked or redirected |
| FR-29 | Never request or store PAN, Aadhaar, account number, OTP, email, or phone | Input filter drops PII-looking input and returns a safe message; logs store no PII |

### 6.7 UI

| ID | Requirement | Acceptance |
|----|-------------|-----------|
| FR-30 | Welcome line naming the scope (HDFC, 5 schemes) | Visible on load |
| FR-31 | Exactly 3 example questions shown as clickable suggestions | 3 chips rendered |
| FR-32 | Persistent note: **"Facts-only. No investment advice."** | Visible without scrolling |
| FR-33 | Each answer shows source link (clickable) + `Last updated from sources:` | Verified in demo video |
| FR-34 | Simple single-turn input box, no chat-history DB | Session-only |

### 6.8 Sample question set the pipeline must handle

| # | Query | Type |
|---|-------|------|
| Q1 | What is the exit load of HDFC Large Cap Fund? | FACT |
| Q2 | What is the minimum SIP amount for HDFC Small Cap Fund? | FACT |
| Q3 | What is the lock-in period for HDFC ELSS Tax Saver Fund? | FACT |
| Q4 | What is the expense ratio / TER of HDFC Equity Fund? | FACT |
| Q5 | Which benchmark does HDFC Balanced Advantage Fund follow, and what is its riskometer category? | FACT |
| Q6 | How do I download the capital-gains statement? | HOW_TO |
| Q7 | Should I buy HDFC Small Cap Fund? | ADVICE → refuse |
| Q8 | Which of these 5 funds has given the best returns? | ADVICE/performance → refuse + factsheet link |
| Q9 | Is HDFC ELSS better than a large cap fund for my portfolio? | ADVICE → refuse |
| Q10 | My PAN is 1234567890, please fetch my folio | PII → safe message |

---

## 7. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-01 | Runs on a macOS/Linux/Windows laptop, Python 3.10+; needs one Groq API key, no other paid API keys |
| NFR-02 | First response ≤ 10 s (embed + retrieve + Groq). Streaming/token output permitted to keep perceived latency low |
| NFR-03 | Deterministic: `temperature = 0`, fixed chunk size, seeded where applicable |
| NFR-04 | Re-ingest is idempotent and completes in < 3 min for 5 pages |
| NFR-05 | No PII persisted anywhere; no analytics/telemetry leaving the machine |
| NFR-06 | Corpus is fully reproducible offline after ingest (cached HTML + local Chroma + local MiniLM embeddings); answering reuses it and needs no re-fetch |
| NFR-07 | Code is modular and readable: separate ingest / retrieve / answer / guardrails modules, typed, no secrets in repo |
| NFR-08 | Total demo footprint fits on a standard 16 GB RAM machine |

---

## 8. Tech Stack

| Layer | Choice | Rationale (per problem statement) |
|-------|--------|----------------------------------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Required; fast, 384-dim, CPU-friendly |
| Chunking | `RecursiveCharacterTextSplitter` (heading-aware) | Best fit for structured web pages |
| Vector DB | `chromadb` (persistent) | Required; zero-ops local store |
| LLM | **Groq** (`openai/gpt-oss-120b`, GroqChat) via `GROQ_API_KEY` | Required for generation; extractive fallback if unavailable |
| Ingestion | `httpx` + `beautifulsoup4` + `markdownify` (or `trafilatura`) | Lightweight HTML → Markdown |
| Orchestration | Direct library calls (LangChain optional) | Show the stages explicitly; framework is not the point of the demo |
| UI | Streamlit (or Gradio) | Fastest path to the "tiny UI" requirement |

---

## 9. Data Handling & Content Rules (hard constraints)

1. **Public sources only.** No screenshots of any app back-end in deliverables. No third-party
   blogs as sources.
2. **No PII.** The system must not ask for, accept, or store PAN, Aadhaar, account numbers,
   OTPs, emails, or phone numbers. Detect and refuse.
3. **No performance claims.** Never compute, compare, rank, or state returns/CAGR. If asked,
   reply that performance data is in the official factsheet and link it.
4. **Clarity & transparency.** Answers ≤ 3 sentences, one citation, plus
   `Last updated from sources: <date>`.
5. **Facts only, no advice.** No "should", "recommend", "suitable for you", "better than".
6. **Citations must resolve.** Every cited URL must be one of the allowlisted corpus URLs.
7. **Disclaimers always visible.** In the UI and in every response.

---

## 10. Acceptance Criteria (demo sign-off)

- [ ] `python -m src.ingest` builds the Chroma collection from the 5 URLs; manifest written.
- [ ] Chunking decision documented with rationale and parameters in README.
- [ ] ≥ 8/10 sample queries answered correctly with one valid citation each.
- [ ] All 3 ADVICE/performance queries refused with a polite message + educational link.
- [ ] PII query refused safely; no PII in logs.
- [ ] `Last updated from sources:` present on every factual answer.
- [ ] No answer exceeds 3 sentences.
- [ ] UI shows welcome line, 3 example questions, and the facts-only disclaimer.
- [ ] README contains setup steps, scope (AMC + 5 schemes), architecture, and known limits.
- [ ] `sources.csv` lists the 5 URLs with fetch dates.
- [ ] `sample_qa.md` contains 5–10 queries with answers + links.
- [ ] Demo video recorded showing the UI flow (no back-end screenshots).

---

## 11. Milestones

| Milestone | Deliverable | Est. effort |
|-----------|-------------|-------------|
| M1 — Ingestion | Fetch + clean + chunk + embed + store; `corpus_manifest.json` | 1 day |
| M2 — Retrieval | Query embed, top-k search, threshold, metadata filter, retrieval log | 1 day |
| M3 — Generation | Grounded Groq prompt, ≤3 sentences, citation, last-updated line | 1 day |
| M4 — Guardrails + UI | Intent router, refusal path, PII filter, Streamlit UI with 3 examples | 1 day |
| M5 — Hardening | Threshold tuning against sample Q&A, ablation note, latency check | 0.5 day |
| M6 — Docs + demo | README, sources.csv, sample_qa.md, disclaimer snippet, demo video | 0.5 day |

---

## 12. Deliverables

| # | Deliverable | File / location |
|---|-------------|-----------------|
| D1 | This PRD | `PRD.md` |
| D2 | Working demo video (UI only, no back-end screenshots) | `demo.mp4` (or linked drive video if hosting is not possible) |
| D3 | Source list of the 5 URLs used | `sources.csv` (url, scheme, category, plan, fetched_at, sha256) |
| D4 | README: setup steps, scope, architecture, chunking decision, known limits | `README.md` |
| D5 | Sample Q&A (5–10 queries, answers + links) | `sample_qa.md` |
| D6 | Disclaimer snippet used in the UI | `DISCLAIMER.md` |
| D7 | Source code (ingest / retrieve / answer / guardrails / app) | `src/`, `app.py` |
| D8 | Requirements + run scripts | `requirements.txt`, `run.sh` |

---

## 13. Known Limits (to state in the README)

1. **Freshness** — corpus is a point-in-time snapshot of 5 web pages; "last updated" reflects
   the fetch date, not a live NAV or AUM feed.
2. **Single AMC** — HDFC only; questions about other AMCs will correctly return
   "not found in the indexed sources".
3. **Page-dependent coverage** — any fact not present on the 5 pages (e.g., full statutory
   riskometer text, slab-wise TER table) is out of scope unless an official factsheet is added
   to the corpus.
4. **Hosted-model quality** — Groq may occasionally be verbose or drop a citation; the
   output validator catches these, but answer quality is bounded by the model.
5. **English only, text only.**
6. **No multi-turn memory** — each question is answered independently.
7. **Not financial advice** — output is a retrieval summary of public pages, and public pages
   are not a substitute for a scheme's SID/KIM or an advisor.

---

## 14. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Groww page HTML changes / bot blocking | Ingestion breaks | Cache raw HTML; use `trafilatura`; fall back to AMC factsheet URL; cache makes demo reproducible offline |
| Small model hallucinates a fact | Incorrect answer | Strict grounding prompt + citation validator + `not in sources` fallback + refusal for unsourced claims |
| Advice leaks through paraphrased questions | Constraint violation | Intent router + keyword/pattern guardrails on both input and output + adversarial test set |
| Numeric facts split across chunks | Wrong numbers | Heading-aware splitting, atomic-fact integrity check (FR-08) |
| Missing `GROQ_API_KEY` or Groq unreachable | Demo generation fails | Document `.env` (`GROQ_API_KEY`); provide extractive fallback (FR-25) |
| Latency on CPU | Poor demo feel | Cache embeddings, `num_predict` cap, streaming output |

---

## 15. Appendix A — Disclaimer Snippet (used in UI and responses)

> **Facts only. No investment advice.** Answers are generated from HDFC public pages listed in
> the source list and may be out of date. Mutual fund investments are subject to market risks;
> read the scheme's SID/KIM and factsheet before investing. Please do not share PAN, Aadhaar,
> account numbers, OTPs, or any personal information.

## 16. Appendix B — Educational Link for Refusals

- HDFC MF investor education / knowledge centre: `https://www.hdfcassetmanagement.com/about-us/investor-education`
- SEBI investor education: `https://www.investor.gov.in/`
- AMFI investor education: `https://www.amfiindia.com/investor-education`

## 17. Appendix C — Project Layout

```
rag-chatbot/
├── PRD.md
├── README.md
├── DISCLAIMER.md
├── sources.csv
├── sample_qa.md
├── requirements.txt
├── run.sh
├── app.py                     # Streamlit UI
├── src/
│   ├── config.py              # URLs, chunk params, thresholds, allowlist
│   ├── ingest.py              # STAGE 1-2  load + clean
│   ├── chunking.py            # STAGE 3    chunk
│   ├── embed.py               # STAGE 4    embed
│   ├── vectorstore.py         # STAGE 5    store
│   ├── retrieve.py            # STAGE 6-9  query prep + search + filter
│   ├── answer.py              # STAGE 10-11 ground + generate
│   ├── guardrails.py          # STAGE 12   intent, refusal, PII, validators
│   └── logger.py              # retrieval logging
└── data/
    ├── raw/                   # cached HTML
    ├── processed/             # markdown per page
    └── chroma/                # persistent vector store
```

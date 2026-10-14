# Mutual Fund FAQ Assistant — Facts-Only RAG Chatbot

A retrieval-augmented question-answering assistant for five **HDFC Asset Management** mutual
fund schemes. It answers factual questions (expense ratio, exit load, minimum SIP, ELSS
lock-in, benchmark/riskometer, how-to steps) using **only** the five allowlisted public
scheme pages, cites **exactly one** source link per answer, and refuses investment-advice,
performance-comparison, and PII questions.

Every factual answer is ≤ 3 sentences, ends with `Last updated from sources: <date>`, and
carries the footer `Facts only. Not investment advice.`

This build implements the complete RAG lifecycle — data ingestion (stages 1–5) and data
retrieval (stages 6–13) — as separable modules.

- Specification: [`PRD.md`](docs/PRD.md)
- Design & data contracts: [`docs/architecture.md`](docs/architecture.md)
- Phase-by-phase build plan: [`docs/implementation.md`](docs/implementation.md)
- Tuning evidence: [`tuning_notes.md`](tuning_notes.md)
- Sample transcripts: [`sample_qa.md`](sample_qa.md)

## Scope

**AMC in scope:** HDFC Asset Management (single AMC). **Plans:** Direct – Growth only.

| # | Category | Scheme | Source URL |
|---|----------|--------|------------|
| S1 | Large Cap | HDFC Large Cap Fund | `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth` |
| S2 | Flexi Cap | HDFC Equity Fund (Flexi Cap) | `https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth` |
| S3 | ELSS (Tax Saver) | HDFC ELSS Tax Saver Fund | `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth` |
| S4 | Small Cap | HDFC Small Cap Fund | `https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth` |
| S5 | Balanced Advantage (Hybrid) | HDFC Balanced Advantage Fund | `https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth` |

The 5-URL source list with fetch dates and content hashes is generated in
[`sources.csv`](sources.csv) (from `corpus_manifest.json`).

Out of scope: NAV/live price, returns, rankings, fund comparisons, portfolio advice, user
accounts, multi-AMC, languages other than English, and any question naming another AMC.

## Architecture

```
DATA INGESTION (offline, run once)                     DATA RETRIEVAL (per query)
1  LOAD      fetch 5 pages → cache raw HTML            6  QUERY PREP  normalize + intent
2  CLEAN     strip nav/boilerplate → Markdown          7  EMBED QUERY all-MiniLM-L6-v2
3  CHUNK     heading-aware recursive split             8  SEARCH      top-k cosine
4  EMBED     all-MiniLM-L6-v2 (384-d, normalized)      9  FILTER      allowlist + MMR
5  STORE     ChromaDB persistent (mf_faq_hdfc_v1)      10 GROUND      context + facts-only rules
                                                        11 GENERATE    Groq → ≤3 sentences
   → corpus_manifest.json                               12 GUARDRAIL   citation/length/claims
                                                        13 RESPOND     CLI / Streamlit UI
```

| Stage | Module | Key technology |
|-------|--------|----------------|
| 1–2 Load/clean | `src/ingest.py` | `httpx`, `trafilatura`, `beautifulsoup4`, `markdownify` |
| 3 Chunk | `src/chunking.py` | `langchain-text-splitters` (`RecursiveCharacterTextSplitter`) |
| 4 Embed | `src/embed.py` | `sentence-transformers` → `all-MiniLM-L6-v2` |
| 5 Store | `src/vectorstore.py` | `chromadb` persistent client (cosine) |
| 6–9 Retrieve | `src/retrieve.py` | Chroma query + threshold + hybrid re-rank + MMR |
| 10–13 Generate/guard | `src/answer.py`, `src/guardrails.py` | Groq (OpenAI-compatible), validators |
| Logging | `src/logger.py` | JSONL retrieval/answer logs (PII redacted) |
| UI | `app.py` | Streamlit — welcome line, 3 examples, disclaimer, answer card |

The orchestrator `src/answer.answer_question` enforces this order: PII pre-filter → advice/
performance intent (deterministic refusal, no LLM) → retrieval → `no_context` refusal →
grounded prompt → generation → validator/repair → extractive fallback. No path can emit
uncited text.

## Chunking strategy decision

**Decision:** heading-aware **recursive** character splitting
(`RecursiveCharacterTextSplitter`, `chunk_size = 2400` chars ≈ 600 tokens,
`chunk_overlap = 360` chars ≈ 90 tokens), with atomic-fact protection.

**Rationale.** The corpus is structured scheme pages — definitions, fee tables, exit-load
slabs, minimum-SIP lines — not prose essays. Heading boundaries are therefore the
semantically correct cut points. Markdown table rows and `**Label:** value` pairs are treated
as indivisible so a numeric fact is never split across chunks, and every chunk is re-attached
to its section heading and prefixed with a provenance header
(`Scheme: … | Plan: … | Category: … | Section: … | Source: …`) before embedding.

**Ablation note (recursive vs semantic).** Semantic chunking was not used because it is
nondeterministic across runs (it would break the temperature-0 reproducibility goal), costs an
extra embedding pass, and offers no benefit when the natural section boundaries already exist
in the markup. A recursive split on those boundaries is deterministic, cheaper, and keeps each
fact block intact with its heading. The decision would be revisited only if the corpus shifted
to long prose documents (e.g. KIM/SID PDFs).

Measured result (from `logs/chunking_report.json`): **459 chunks** across 5 documents, median
**213 tokens** (max 254, min 72, mean 199), median character↔token drift 0.221, and an empty
`fact_integrity_failures` list. The MiniLM tokenizer caps at 256 tokens and **0 chunks exceed
it**, so no fact is truncated at embed time. Median size is below the 600-token budget because
most page sections are simply smaller than the budget; `chunk_size` is an upper bound, not a
fill target.

## Setup

Requires **Python 3.10+** on macOS/Linux/Windows and a Groq API key (generation only;
embeddings and retrieval stay local).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # then set GROQ_API_KEY=<your key>
```

`GROQ_API_KEY` is read from the gitignored `.env`. Without it the system still runs and
answers via the grounded extractive fallback (FR-25) — it never hard-fails.

## Run

```bash
./run.sh ingest      # stages 1-5: fetch, clean, chunk, embed, store + manifest
./run.sh report      # chunking report + fact-integrity result
./run.sh stats       # vector store statistics
./run.sh chat        # terminal chat (stages 6-13)
./run.sh ui          # Streamlit UI
./run.sh test        # offline test suite
./run.sh selftest    # pipeline self-test (collection, dim, Groq, canned query)
```

Equivalent direct commands:

```bash
python -m src.ingest            # add --force to bypass the HTML cache
python -m src.answer --question "What is the exit load of HDFC Large Cap Fund?"
python eval/run_eval.py         # deterministic offline evaluation
python eval/run_eval.py --live  # evaluation against Groq
pytest -q                       # unit tests per stage
```

## Project layout

```
rag-chatbot/
├── PRD.md / architecture.md / implementation.md   (in docs/)
├── README.md
├── DISCLAIMER.md
├── sources.csv                  # generated from corpus_manifest.json
├── sample_qa.md
├── tuning_notes.md
├── requirements.txt
├── run.sh
├── pytest.ini
├── app.py                       # Streamlit UI
├── corpus_manifest.json         # generated by ingest
├── eval/
│   ├── golden_queries.json      # 10 PRD queries + 3 nonsense
│   ├── run_eval.py
│   └── report.json / report.md
├── src/
│   ├── config.py                # 1 source of truth for tunables
│   ├── ingest.py                # stages 1-2 (+ orchestrates 3-5)
│   ├── chunking.py              # stage 3
│   ├── embed.py                 # stage 4
│   ├── vectorstore.py           # stage 5
│   ├── retrieve.py              # stages 6-9
│   ├── answer.py                # stages 10-11, 13 + orchestration
│   ├── guardrails.py            # stage 12
│   └── logger.py
└── data/{raw,processed,chroma}/  # gitignored except .gitkeep
```

## Tuning results

Retrieval constants are set in `src/config.py` and justified with a sweep in
[`tuning_notes.md`](tuning_notes.md):

| Constant | Value | Why |
|----------|-------|-----|
| `MIN_SIMILARITY` | **0.45** | Cosine-similarity threshold. Nonsense queries top out at 0.203; FACT queries bottom out at 0.70. The safe plateau is 0.25–0.55; 0.45 is its midpoint, giving a balanced margin on both sides. |
| `MMR_LAMBDA` | **0.7** | Correctness is identical for 0.5 and 0.7 across the sweep; 0.7 selected more distinct schemes (9 vs 8) for the same top-1 relevance. |
| `TOP_K` | 5 | Small corpus; enough redundancy for MMR without diluting the prompt. |
| `MAX_CHUNKS_PER_SECTION` | 2 | Stops one long section from filling the context. |

Chunk statistics: **459 chunks / 5 documents**, median 213 tokens, max 254, median
char↔token drift 0.221. Corpus version `mf_faq_hdfc_v1`, embedding
`sentence-transformers/all-MiniLM-L6-v2` (384-d).

Evaluation (`eval/run_eval.py`, deterministic offline path):

| Metric | Result |
|--------|--------|
| Overall golden set | 12/13 |
| PRD §6.8 queries | 9/10 answered with exactly one in-corpus link |
| Advice / performance queries | 3/3 refused with an educational link |
| PII query | refused; no raw PII in logs |
| Nonsense queries | 3/3 return `no_context` |
| Offline test suite | `pytest` 461 passed |
| Latency (offline) | ≈7.3 s wall for 13 queries |

Known `must_mention` limitation under the deterministic path: for Q4 the numeric value `0.77`
lives in a chunk ranked below the expense-ratio *definition*, and the extractive fallback is
contract-limited to the top chunk. The value is retrieved and the live model answers it
correctly; see `tuning_notes.md` §7.

## Sample Q&A

Ten real transcripts (questions, answers, links) are in [`sample_qa.md`](sample_qa.md).

## Known limits

1. **Freshness** — the corpus is a point-in-time snapshot of 5 web pages (fetched
   2026-09-27); "last updated" reflects the fetch date, not a live NAV or AUM feed.
2. **Single AMC** — HDFC only; questions about other AMCs correctly return "not found in the
   indexed sources".
3. **Page-dependent coverage** — any fact not present on the 5 pages (e.g. full statutory
   riskometer text, slab-wise TER table) is out of scope unless an official factsheet is
   added to the corpus.
4. **Hosted-model quality** — Groq may occasionally be verbose or drop a citation; the
   output validator catches these, but answer quality is bounded by the model. When Groq is
   unavailable the deterministic extractive fallback answers from the top retrieved chunk.
5. **English only, text only.**
6. **No multi-turn memory** — each question is answered independently.
7. **Not financial advice** — the output is a retrieval summary of public pages, and public
   pages are not a substitute for a scheme's SID/KIM or an advisor.

## Disclaimer

> **Facts only. No investment advice.** Answers are generated from HDFC public pages listed
> in the source list and may be out of date. Mutual fund investments are subject to market
> risks; read the scheme's SID/KIM and factsheet before investing. Please do not share PAN,
> Aadhaar, account numbers, OTPs, or any personal information.

See [`DISCLAIMER.md`](DISCLAIMER.md). No analytics or telemetry leave the machine; no PII is
persisted anywhere.

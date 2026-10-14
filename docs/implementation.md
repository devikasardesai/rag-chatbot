# Implementation Guide — Mutual Fund FAQ Assistant (Facts-Only RAG Chatbot)

**Derived from:** `architecture.md` v1.0 and `PRD.md` v1.0
**Purpose:** phase-by-phase build instructions. Each phase is a self-contained unit of work with
a clear goal, exact file paths, function contracts, implementation gotchas, verification
commands, and a copy-pasteable prompt for a coding agent (Cursor / opencode / Claude Code).

**Status:** Draft v1.0

---

## 0. How To Use This Document

1. **Work in phase order.** Phases 0 → 4 build the offline ingestion pipeline; phases 5 → 8
   build the online retrieval pipeline; phases 9 → 11 are UI, evaluation, and deliverables.
2. **One phase = one agent task.** Do not start phase *N+1* until phase *N*'s verification
   commands pass and its "Definition of Done" is fully checked.
3. **Commit after every phase** with the suggested message. This gives you a working demo
   checkpoint after Phase 4 (data ready) and another after Phase 8 (end-to-end CLI).
4. **Never let the agent skip verification.** Every phase ends with runnable commands; the
   agent must paste the actual output.
5. **Hard constraints apply in every phase** (see §0.3). Any change that violates them is a
   regression, not an enhancement.

### 0.1 Phase overview

| Phase | Name | Stages (PRD/arch) | Est. | Depends on |
|-------|------|-------------------|------|------------|
| 0 | Scaffold, config, logger | — | 0.5 d | — |
| 1 | Ingestion: load + clean | 1–2 | 1 d | 0 |
| 2 | Chunking strategy | 3 | 1 d | 1 |
| 3 | Embedding | 4 | 0.25 d | 0 |
| 4 | Vector store + end-to-end ingest CLI | 5 | 0.5 d | 1,2,3 |
| 5 | Guardrails | 12 (upstream) | 1 d | 0 |
| 6 | Retrieval | 6–9 | 1 d | 3,4,5 |
| 7 | Generation (prompt + LLM) | 10–11 | 1 d | 6 |
| 8 | Orchestration, logging, CLI, selftest | 12–13 | 1 d | 7 |
| 9 | Streamlit UI | FR-30…34 | 0.5 d | 8 |
| 10 | Tests, tuning, evaluation harness | NFR-03, M5 | 1 d | 8 |
| 11 | Docs + demo deliverables | §12 | 0.5 d | 9,10 |

**Critical path:** 0 → 1 → 2 → 4 → 6 → 7 → 8 → 10. Phases 3 and 5 are short and can be
handed to a second person/agent in parallel with 1–2 and 6 respectively.

### 0.2 Global conventions (apply to every phase)

- **Python 3.10+**, type hints on all public functions, `dataclass` (frozen) for result types.
- **No `StrEnum`** (Python 3.11+): use `class Intent(str, Enum)`.
- Docstrings on public functions; **no inline commentary** in the code.
- No secrets, no API keys, no telemetry. Only outbound network access is Phase 1 fetching.
- Every module has a `main()` + `if __name__ == "__main__":` CLI entry.
- All tunable values live in `src/config.py`; no magic numbers in stage modules.
- Errors: raise domain exceptions (`IngestError`, `OutOfScopeError`, `LLMUnavailable`); the
  CLI prints a short message and returns a non-zero exit code.
- No LangChain agents, no LangChain retrieval chains, no paid APIs, no caching frameworks.
  LangChain's `RecursiveCharacterTextSplitter` text splitter is the only permitted
  `langchain-*` dependency.
- Chrome/UI work uses `st.session_state` only; never persist chat history to disk.

### 0.3 Non-negotiable constraints (from PRD §9)

1. Only the 5 allowlisted corpus URLs may be fetched or cited.
2. Never compute, compare, rank, or state returns/CAGR/performance.
3. Never output investment advice, recommendations, or "better/best" comparisons.
4. Answers ≤ 3 sentences + exactly one citation + `Last updated from sources: <date>`.
5. Never request, accept, or store PAN, Aadhaar, account numbers, OTPs, emails, phones.
6. Deliverables/demo show UI only — no back-end screenshots.

### 0.4 Repository layout to be created in Phase 0

```
rag-chatbot/
├── PRD.md                  # exists
├── architecture.md         # exists
├── implementation.md       # this file
├── README.md               # Phase 11
├── DISCLAIMER.md           # Phase 11
├── sources.csv             # Phase 11 (generated from manifest)
├── sample_qa.md            # Phase 10/11
├── requirements.txt
├── run.sh
├── pytest.ini
├── .gitignore
├── app.py                  # Phase 9
├── src/
│   ├── __init__.py
│   ├── config.py           # Phase 0
│   ├── logger.py           # Phase 0
│   ├── ingest.py           # Phase 1 + 4
│   ├── chunking.py         # Phase 2
│   ├── embed.py            # Phase 3
│   ├── vectorstore.py      # Phase 4
│   ├── guardrails.py       # Phase 5
│   ├── retrieve.py         # Phase 6
│   └── answer.py           # Phase 7 + 8
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_ingest.py
│   ├── test_chunking.py
│   ├── test_embed.py
│   ├── test_vectorstore.py
│   ├── test_guardrails.py
│   ├── test_retrieve.py
│   └── test_answer.py
├── data/{raw,processed,chroma}/   # gitignored except .gitkeep
└── logs/                              # gitignored
```

---

## PHASE 0 — Scaffold, Config, Logger

**Goal:** a runnable, importable skeleton with all tunable values in one place.
**Stages:** none (foundation).
**Produces:** `requirements.txt`, `.gitignore`, `pytest.ini`, `run.sh`, `src/config.py`,
`src/logger.py`, `src/__init__.py`, `tests/conftest.py`.

### Tasks

0.1 Create the directory tree above and `src/__init__.py` (empty).
0.2 Write `requirements.txt` with pinned major versions:
    `sentence-transformers`, `chromadb`, `langchain-text-splitters`, `beautifulsoup4`,
    `lxml`, `trafilatura`, `markdownify`, `httpx`, `numpy`, `streamlit`, `pytest`.
    Add a comment block explaining each package maps to an architecture stage.
0.3 Write `src/config.py` exposing **exactly** these names (values from architecture §9):
    - `PROJECT_ROOT`, `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `CHROMA_DIR`, `LOG_DIR`
    - `SourceSpec` dataclass: `url: str`, `scheme_name: str`, `category: str`, `plan: str`
    - `CORPUS: list[SourceSpec]` — the 5 rows from PRD §4 (S1–S5), plan = `"Direct Growth"`,
      categories: `Large Cap`, `Flexi Cap`, `ELSS`, `Small Cap`, `Balanced Advantage`.
      Include the corrected S5 URL `https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth`.
    - `ALLOWED_URLS: frozenset[str]` derived from `CORPUS`
    - `EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"`, `EMBEDDING_DIM = 384`,
      `EMBED_BATCH_SIZE = 32`
    - `CHUNK_SIZE_TOKENS = 600`, `CHUNK_OVERLAP_TOKENS = 90`, `MIN_CHUNK_TOKENS = 25`,
      `CHUNK_CHARS = CHUNK_SIZE_TOKENS * 4`, `CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * 4`
    - `MUST_HAVE_FACTS: dict[str, list[str]]` — per-scheme probe strings used by the
      fact-integrity check (e.g. HDFC ELSS → `["3 years"]`; Small Cap → `["exit load"]`,
      `["SIP"]`; every scheme → `["expense ratio" or "expense ratio (%)"]`,
      `["benchmark" or "Nifty" or "BSE"]`). Best-effort, case-insensitive match.
    - `COLLECTION_NAME = "mf_faq_hdfc_v1"`, `HNSW_SPACE = "cosine"`, `MANIFEST_PATH`,
      `CHUNKING_REPORT_PATH`
    - `TOP_K = 5`, `MIN_SIMILARITY = 0.35`, `MMR_LAMBDA = 0.7`, `MAX_CHUNKS_PER_SECTION = 2`,
      `MAX_CONTEXT_TOKENS = 2400`
    - `LLM_MODEL = "openai/gpt-oss-120b"` (overridable with `GROQ_MODEL`), `LLM_BASE_URL = "https://api.groq.com/openai/v1"`,
      `LLM_TEMPERATURE = 0.0`, `LLM_NUM_PREDICT = 220`, `LLM_NUM_CTX = 4096`,
      `LLM_TIMEOUT_S = 90`
    - `MAX_SENTENCES = 3`, `LAST_UPDATED_PREFIX = "Last updated from sources:"`
    - `PROMPT_SYSTEM: str` — the full system prompt from architecture §7.3 (verbatim; this is
      the grounding contract, do not paraphrase)
    - `REFUSAL_ADVICE`, `REFUSAL_PII`, `REFUSAL_NO_CONTEXT` constants (copy from PRD §16 / §6.6)
    - `EDUCATIONAL_LINKS: dict[str, str]` — `advice`, `performance`, `how_to` keys
    - `DISCLAIMER` — PRD Appendix A text
    - `USER_AGENT`, `HTTP_TIMEOUT_S = 20`, `HTTP_RETRIES = 1`, `POLITENESS_DELAY_S = 1.0`
    - `Env overrides: `RAG_DATA_DIR`, `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_BASE_URL`, `HF_HOME`.
0.4 Write `src/logger.py`:
    - `configure_logging()` → creates `logs/`, one StreamHandler (INFO) + rotating file handler.
    - `log_retrieval(record: dict) -> None` → appends one JSON line to `logs/retrieval.jsonl`.
    - `log_answer(record: dict) -> None` → appends to `logs/answers.jsonl`.
    - `redact_pii(text: str) -> str` → replaces any PII-looking span with `sha256[:8]` so raw
      PII is never written to disk (PRD FR-29).
0.5 Write `tests/conftest.py` with fixtures: `corpus` (from config), `tmp_data_dir`
    (monkeypatches `config.DATA_DIR`/`CHROMA_DIR`/`LOG_DIR` to `tmp_path`), `sample_markdown`
    (a small realistic scheme-page Markdown fixture with headings, a fee table, and an
    exit-load table), `fake_specs` (2 fake `SourceSpec`s with `example.com` URLs).
0.6 Write `.gitignore` (`.venv/`, `__pycache__/`, `data/raw/*`, `data/processed/*`,
    `data/chroma/*`, `logs/*`, `.DS_Store`, `*.mp4`), `pytest.ini`
    (`testpaths = tests`, `addopts = -q`), `run.sh` (chmod +x) with subcommands
    `setup | ingest | report | stats | chat | ui | test | selftest`.

### Implementation notes / gotchas

- `CHUNK_CHARS` exists because `RecursiveCharacterTextSplitter` is **character**-based.
  4 chars/token is a fine approximation; Phase 2 validates real token counts with the MiniLM
  tokenizer and reports the drift.
- The system prompt must be a single triple-quoted string; the agent must not "improve" its
  wording, because Phase 10 tuning depends on the exact rules.
- Keep `CORPUS` as data, not code paths — Phase 11 generates `sources.csv` from it.

### Verification

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "from src import config; print(len(config.CORPUS), sorted(config.ALLOWED_URLS)[0][:60])"
python -m src.logger            # no-op import check
pytest -q tests/test_config.py
```
Expected: `5 https://groww.in/...`; tests pass.

### Definition of Done

- [ ] `pip install -r requirements.txt` succeeds on macOS with Python ≥ 3.10
- [ ] `config.CORPUS` has 5 entries with the exact URLs from PRD §4 (incl. corrected S5)
- [ ] `PROMPT_SYSTEM` matches architecture §7.3
- [ ] `logger.redact_pii` demonstrably removes a PAN-looking string
- [ ] `run.sh` runs every subcommand without `ImportError` (stub targets may exit non-zero)

### Commit

`chore: scaffold project, config constants, logging utility`

### Agent prompt (Phase 0)

```
Implement Phase 0 of implementation.md in this repo (read implementation.md §PHASE 0 and
architecture.md §9 first). Create only: requirements.txt, .gitignore, pytest.ini, run.sh,
src/__init__.py, src/config.py, src/logger.py, tests/conftest.py, tests/test_config.py.
Requirements: Python 3.10+ compatible (use `class X(str, Enum)`, never StrEnum); frozen
dataclasses for SourceSpec; ALL tunable values in config.py; the PROMPT_SYSTEM string must be
copied verbatim from architecture.md §7.3 including rule numbering; logger must write JSONL
and must never persist raw PII (hash it). No new dependencies beyond requirements.txt. No
inline comments in code, docstrings on public functions only. Finish by running the Phase 0
verification commands and pasting the real output. Do not start any later phase.
```

---

## PHASE 1 — Ingestion: Load + Clean (Stages 1–2)

**Goal:** fetch the 5 pages once into a cache and convert each to clean Markdown.
**Depends on:** Phase 0.
**Produces:** `src/ingest.py`, `data/raw/*.html`, `data/processed/*.md`,
`tests/test_ingest.py`.

### Tasks

1.1 Define in `src/ingest.py`:
    - `@dataclass(frozen=True) class PageDocument`: `doc_id, url, scheme_name, category, plan,
      page_title, markdown, fetched_at (ISO date str), content_sha256, raw_path, processed_path`
    - `doc_id_for(url: str) -> str` → `sha1(url).hexdigest()[:12]`
    - `fetch_page(spec: SourceSpec, force: bool = False) -> PageDocument`
1.2 `fetch_page` behaviour:
    - If `RAW_DIR/<doc_id>.html` exists and `force` is False → read cache, set `fetched_at`
      from file mtime (UTC date), skip network.
    - Else `httpx.get` with `USER_AGENT`, `follow_redirects=True`, `HTTP_TIMEOUT_S`, one
      retry with backoff, `POLITENESS_DELAY_S` sleep between URLs, raise `IngestError` with
      the URL on failure.
    - Write raw HTML to `RAW_DIR/<doc_id>.html`; compute `content_sha256` of the HTML bytes.
1.3 `extract_main_content(html: str) -> tuple[str, str]`:
    - Primary: `trafilatura.extract(html, output_format="markdown", include_tables=True,
      include_links=False)`.
    - Fallback: BeautifulSoup heuristic — drop `nav, header, footer, script, style, noscript,
      aside, [role=navigation]`, then pick the element with the highest density of `<p>` text
      and convert with `markdownify(..., heading_style="ATX")`.
    - Normalize: collapse 3+ blank lines, strip trailing whitespace, drop lines matching
      cookie/login/related-funds noise patterns.
    - Return `(page_title, markdown)`. Raise `IngestError` if extracted Markdown < 500 chars.
1.4 `build_corpus(sources, force) -> list[PageDocument]`: iterate `config.CORPUS`, call
    `fetch_page`, write `PROCESSED_DIR/<doc_id>.md`, and **refuse any URL not in
    `ALLOWED_URLS`** (`OutOfScopeError`).
1.5 `main()` CLI: `--force`, `--url <one url>` (debug a single page), prints a per-URL table
    (`scheme | chars | chunks pending | sha256[:8] | cached?`).
1.6 `tests/test_ingest.py`: mock `httpx.get` with a saved fixture HTML; assert cache reuse
    (second call makes no request), `doc_id` stability, allowlist rejection, and
    `IngestError` on too-short extraction. **No network in tests.**

### Implementation notes / gotchas

- Groww scheme pages are server-rendered enough for this to work, but boilerplate is heavy.
  If extraction yields < 500 chars, print the URL and stop — do **not** silently write an empty
  document; the corpus must be complete (5/5) before Phase 2.
- Store `fetched_at` from the cache file's mtime so a cached run reports the *original* fetch
  date. This value is what surfaces as `Last updated from sources:`.
- If a page changes between runs, `content_sha256` changes → that is the signal to re-chunk.

### Verification

```bash
python -m src.ingest
ls -l data/raw data/processed
python - <<'PY'
from src import ingest, config
docs = ingest.build_corpus(config.CORPUS)
assert len(docs) == 5, "need 5/5 pages"
for d in docs:
    print(d.scheme_name, "|", len(d.markdown), "chars |", d.page_title[:50])
PY
pytest -q tests/test_ingest.py
```

### Definition of Done

- [ ] 5/5 pages cached as HTML and converted to Markdown (no network on 2nd run)
- [ ] No navigation/footer/cookie text survives in any `.md` file (manual spot check of 2 files)
- [ ] `ALLOWED_URLS` check enforced; `OutOfScopeError` raised for any other URL
- [ ] `content_sha256` and `fetched_at` recorded per page
- [ ] Tests pass offline

### Commit

`feat(ingest): stage 1-2 load and clean 5 allowlisted pages`

### Agent prompt (Phase 1)

```
Implement Phase 1 of implementation.md (read implementation.md §PHASE 1 and architecture.md
§4.1/§4.2). Create only src/ingest.py and tests/test_ingest.py. Follow the frozen PageDocument
dataclass and the four functions in architecture.md §6.3 exactly. trafilatura primary with a
BeautifulSoup density fallback; strip nav/header/footer/script/style/aside; normalise blank
lines; raise IngestError if extracted markdown < 500 chars; refuse any URL not in
config.ALLOWED_URLS. Cache raw HTML per doc_id (sha1(url)[:12]) and derive fetched_at from the
cache file mtime. Tests must not touch the network (monkeypatch httpx.get). Run the Phase 1
verification commands, paste real output, and confirm 5/5 pages. Do not start Phase 2.
```

---

## PHASE 2 — Chunking Strategy (Stage 3)

**Goal:** heading-aware recursive splitting with atomic-fact protection and a fact-integrity
report. This is the phase the demo is judged on for "did you choose a strategy, and why".
**Depends on:** Phase 1.
**Produces:** `src/chunking.py`, `tests/test_chunking.py`, `logs/chunking_report.json`.

### Tasks

2.1 `Section` dataclass: `title: str`, `level: int`, `body: str`.
2.2 `parse_sections(markdown: str) -> list[Section]`: split on ATX headings (`#`…`####`);
    the text before the first heading becomes a section titled with the page title.
2.3 `Chunk` dataclass: `chunk_id, doc_id, ordinal, text, token_count, metadata: dict`
    (metadata keys per architecture §6.1, all Chroma-safe types: `str`/`int`).
2.4 `provenance_header(doc, section) -> str`:
    `"Scheme: {scheme} | Plan: {plan} | Category: {category} | Section: {section} | Source: {url}"`
    — prepended to `text` **before** embedding (FR-09).
2.5 `split_text(markdown: str, doc: PageDocument) -> list[Chunk]` implementing architecture §5:
    - protect table rows and `label: value` lines (never split mid-row),
    - `RecursiveCharacterTextSplitter.from_tiktoken_encoder` **or** char-based with
      `chunk_size=CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP_CHARS`; prefer the tiktoken-free
      char path to avoid an extra dependency, then measure real tokens with the MiniLM
      tokenizer for reporting,
    - re-attach the section heading to every chunk,
    - drop chunks below `MIN_CHUNK_TOKENS` unless they contain digits,
    - assign `ordinal` per document (0-based) and `chunk_id = f"{doc_id}:{ordinal:03d}"`.
2.6 `validate_fact_integrity(chunks, probes=config.MUST_HAVE_FACTS) -> list[str]`: for each
    scheme, every probe string must appear (case-insensitive, whitespace-normalised) in ≥ 1
    chunk; return a list of human-readable failures like
    `"HDFC ELSS Tax Saver Fund: probe '3 years' not found in any chunk"`.
2.7 CLI: `python -m src.chunking --report` writes `logs/chunking_report.json` with per-document
    chunk count, min/median/max token count, the character↔token drift, section histogram, and
    the fact-integrity result. Print a readable summary table.
2.8 `tests/test_chunking.py`: no chunk mixes two sections; table rows never split
    (assert every chunk's table lines are complete); overlap respected (assert consecutive
    chunks share a suffix/prefix of > 0 chars where the text allows); `chunk_id` format;
    `MUST_HAVE_FACTS` probe logic with a synthetic failing case.

### Implementation notes / gotchas

- **Table integrity is the main risk.** Groww fee/exit-load data is Markdown tables; a naive
  recursive split can cut a row in half and produce a wrong number in an answer. Protect every
  line starting with `|` and every `**Label:** value` pair.
- Overlap applies **within** a section only. Across a heading boundary there is intentionally
  no overlap — the heading is re-attached instead.
- `metadata` must never contain `None` (Chroma rejects it) or lists/dicts. Use `""` for absent
  strings and `0` for absent ints.
- Report the char↔token drift; if median token count deviates > 25% from 600, adjust
  `CHUNK_CHARS` in config and note the final value in the README (Phase 11).

### Verification

```bash
python -m src.chunking --report
python - <<'PY'
import json; r = json.load(open("logs/chunking_report.json"))
print(r["totals"]); print("failures:", r["fact_integrity_failures"])
PY
pytest -q tests/test_chunking.py
```

### Definition of Done

- [ ] Every chunk belongs to exactly one section and carries the provenance header
- [ ] No Markdown table row is split across chunks (test-enforced)
- [ ] `chunk_id`s are stable across two runs on the same input
- [ ] `fact_integrity_failures` is empty, **or** each failure is understood, documented in
      `logs/chunking_report.json`, and handled in the README "known limits" section
- [ ] Report shows median chunk size within 25% of 600 tokens

### Commit

`feat(chunking): stage 3 heading-aware recursive chunking + fact-integrity report`

### Agent prompt (Phase 2)

```
Implement Phase 2 of implementation.md (read implementation.md §PHASE 2 and architecture.md
§5). Create only src/chunking.py and tests/test_chunking.py. Implement Section/Chunk
dataclasses, parse_sections, provenance_header, split_text, validate_fact_integrity and a
`--report` CLI writing logs/chunking_report.json. Hard rules: use a char-based
RecursiveCharacterTextSplitter (chunk_size=CHUNK_CHARS, overlap=CHUNK_OVERLAP_CHARS from
config); never split a Markdown table row or a `**Label:** value` pair; re-attach the section
heading to every chunk; chunk metadata must contain only str/int values, never None; measure
real token counts with the MiniLM tokenizer for the report. Tests must assert single-section
membership, table-row integrity, overlap, and stable chunk_ids. Run the Phase 2 verification,
paste the report summary, and state the median token count. Do not start Phase 3.
```

---

## PHASE 3 — Embedding (Stage 4)

**Goal:** a cached `all-MiniLM-L6-v2` embedder with a 384-dim assertion.
**Depends on:** Phase 0 (can be built in parallel with Phases 1–2).
**Produces:** `src/embed.py`, `tests/test_embed.py`.

### Tasks

3.1 `get_model() -> SentenceTransformer` — module-level `@lru_cache(maxsize=1)` singleton
    (`cache_folder` from `HF_HOME` if set). First call downloads ~90 MB.
3.2 `embed_texts(texts, batch_size=config.EMBED_BATCH_SIZE) -> np.ndarray`:
    `model.encode(texts, batch_size=..., normalize_embeddings=True, convert_to_numpy=True,
    show_progress_bar=False)` → assert `shape[1] == config.EMBEDDING_DIM` and dtype float32.
3.3 `embed_query(text: str) -> list[float]` → `embed_texts([text])[0].tolist()`.
3.4 Empty-input guard: `embed_texts([])` returns a `(0, 384)` float32 array.
3.5 `tests/test_embed.py`: dimension 384, row norms ≈ 1.0, identical text → identical vector,
    empty input shape. Mark with `@pytest.mark.slow` if the model must download.

### Implementation notes / gotchas

- `normalize_embeddings=True` is mandatory — it is what makes Chroma's cosine distance
  directly interpretable as `1 - similarity`.
- Keep the model in one process-local singleton. Re-loading per call costs ~1 s and will blow
  the latency budget.
- Do not add a GPU/`torch` device argument; CPU is the demo target.

### Verification

```bash
python -c "from src import embed; v=embed.embed_query('exit load'); print(len(v), round(sum(x*x for x in v)**.5,4))"
pytest -q tests/test_embed.py
```

### Definition of Done

- [ ] 384-dim, unit-norm vectors
- [ ] Singleton (second call is measurably faster, no re-download)
- [ ] Tests pass

### Commit

`feat(embed): stage 4 all-MiniLM-L6-v2 embedder with dim assertion`

### Agent prompt (Phase 3)

```
Implement Phase 3 of implementation.md (read implementation.md §PHASE 3 and architecture.md
§6.3). Create only src/embed.py and tests/test_embed.py. Use an lru_cache'd SentenceTransformer
singleton for config.EMBEDDING_MODEL, encode with normalize_embeddings=True, assert the 384
dimension, handle empty input as (0,384) float32, and expose embed_query returning a list of
floats. No GPU/device config, no other model. Run the Phase 3 verification and paste output.
Do not start any later phase.
```

---

## PHASE 4 — Vector Store + End-to-End Ingest CLI (Stage 5)

**Goal:** persist chunks in ChromaDB and make `python -m src.ingest` run stages 1→5 in one
command. After this phase you have a complete, inspectable corpus.
**Depends on:** Phases 1, 2, 3.
**Produces:** `src/vectorstore.py`, `src/ingest.py` (extended `main`), `data/chroma/`,
`corpus_manifest.json`, `tests/test_vectorstore.py`.

### Tasks

4.1 `src/vectorstore.py`:
    - `get_client() -> chromadb.PersistentClient` (`persist_directory=config.CHROMA_DIR`,
      lru_cached)
    - `get_or_create_collection(reset=False) -> Collection` with
      `metadata={"hnsw:space": config.HNSW_SPACE, "corpus_version": config.COLLECTION_NAME,
      "embedding_model": config.EMBEDDING_MODEL}`
    - `upsert_chunks(chunks, vectors) -> int` — `ids=[c.chunk_id]`, `documents=[c.text]`,
      `metadatas=[c.metadata]`
    - `prune_stale(keep_ids) -> int` — delete stored ids not in `keep_ids` (idempotency, FR-05)
    - `search(query_vec, k, where=None) -> list[RetrievedChunk]` — returns
      `score = 1.0 - distance` sorted descending
    - `collection_stats() -> dict`, `reset_collection() -> None`
4.2 Extend `src/ingest.py: main()` to the full pipeline:
    `build_corpus` → per doc `chunking.split_text` → `embed.embed_texts(all_chunk_texts)` →
    `vectorstore.upsert_chunks` → `prune_stale` → `write_manifest(docs, chunks_per_doc)` →
    print a summary (documents, chunks, median tokens, collection name, manifest path).
4.3 `write_manifest` emits exactly the JSON shape in architecture §4.2 (corpus_version,
    generated_at, embedding_model, chunking params, documents[], totals).
4.4 `--stats` subcommand prints collection name, count, and the collection metadata.
4.5 `tests/test_vectorstore.py`: use `tmp_path` as persist dir; upsert 3 fake chunks, assert
    count 3; search returns descending scores; `prune_stale` removes the extra id; a metadata
    dict containing `None` raises (add a test that our metadata builder never produces `None`).

### Implementation notes / gotchas

- Chroma metadata values must be `str | int | float | bool`. This is the #1 source of runtime
  errors — sanitise in `upsert_chunks` with a helper that coerces `None → ""`.
- `get_or_create_collection` must not silently change `hnsw:space` on an existing collection;
  if the space differs, log a warning telling the user to re-ingest with `--reset`.
- Idempotency check: two consecutive `python -m src.ingest` runs → identical chunk count and
  identical `chunk_id` set. Verify explicitly.

### Verification

```bash
python -m src.ingest            # run twice
python -m src.ingest --stats
python - <<'PY'
import json; m=json.load(open("corpus_manifest.json"))
print(m["totals"], [d["chunk_count"] for d in m["documents"]])
PY
pytest -q tests/test_vectorstore.py
```

### Definition of Done

- [ ] `data/chroma/` persists the collection; a fresh process re-opens it without re-embedding
- [ ] Two runs → identical chunk count and chunk-id set (FR-05)
- [ ] `corpus_manifest.json` matches architecture §4.2 and lists 5 documents
- [ ] `pytest -q tests/test_vectorstore.py` passes

### Commit

`feat(ingest): stage 5 chroma persistence, manifest, end-to-end ingest CLI`

### Agent prompt (Phase 4)

```
Implement Phase 4 of implementation.md (read implementation.md §PHASE 4 and architecture.md
§4.2/§6.3). Create src/vectorstore.py; extend src/ingest.py main() to run stages 1→5
(corpus → chunk → embed → upsert → prune_stale → write_manifest) and add `--stats`. Follow the
architecture function contracts exactly. Sanitise all Chroma metadata to str/int/float/bool
(never None). Assert that two consecutive ingests produce the same chunk_id set, and report the
manifest totals. Run the Phase 4 verification commands, paste real output, then stop. Do not
start Phase 5.
```

---

## PHASE 5 — Guardrails (Intent, PII, Refusals, Answer Validators)

**Goal:** all constraint logic in one I/O-free module — this is what makes the demo *safe* and
is the most heavily unit-tested component.
**Depends on:** Phase 0. Can be built in parallel with Phases 1–4.
**Produces:** `src/guardrails.py`, `tests/test_guardrails.py`.

### Tasks

5.1 `class Intent(str, Enum)`: `FACT`, `HOW_TO`, `ADVICE`, `PII`.
5.2 PII detection (`detect_pii(text) -> list[str]`), compiled regexes returning labels:
    PAN (`[A-Z]{5}[0-9]{4}[A-Z]`), Aadhaar (`\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b`), account number
    (6–18 consecutive digits), OTP (4–6 digit standalone), email, phone
    (`\+?\d[\d \-]{8,14}\d`). Whitelist false positives: SIP amounts, years, dates, percentages
    (e.g. "500" alone, "2026-09-27", "1.5%") must **not** trigger.
5.3 `is_advice_intent(q) -> bool` — weighted keyword/phrase matcher:
    `should i, should we, recommend, suggest, best fund, better than, worth buying, good time to,
    allocate, portfolio, suita\w+ for me, tell me if i should, which one should`.
    Also `is_performance_intent(q) -> bool` — `returns?, cagr, best performing, which grew,
    profit, has grown, x% return, since inception return`.
5.4 `refusal_response(reason, link=None) -> AnswerResult` — returns the canned
    `AnswerResult` for reasons `advice | pii | performance | no_context`; `refused=True`,
    `source_url=link` (advice/performance → `EDUCATIONAL_LINKS[...]`; pii/no_context → the
    relevant scheme page or the source list), and the disclaimer appended.
5.5 `trim_sentences(answer, max_sentences=3) -> str` — split with a regex that does not break
    on decimals or abbreviations:
    `re.split(r'(?<=[.!?])\s+(?=[A-Z0-9“"(])', text)`.
5.6 `validate_answer(answer, context_urls, last_updated) -> tuple[bool, list[str], str]`:
    - flags: `too_long`, `missing_citation`, `citation_not_in_context`, `missing_last_updated`,
      `advice_language`, `performance_language`, `preamble`
    - repairs in order: strip preamble, `trim_sentences`, `ensure_last_updated`, keep only the
      first in-context URL as the single citation and delete any other URL lines.
7 Not done here: pattern-based flags are heuristics — treat them as a safety net, and log every
    triggered flag to `logs/answers.jsonl` for tuning in Phase 10.
5.7 `tests/test_guardrails.py` — table-driven; must include all 10 PRD §6.8 queries with their
    expected `Intent`, plus PII positive/negative pairs and a `validate_answer` repair case.

### Implementation notes / gotchas

- Order of checks in `answer_question` matters: **PII before intent before retrieval** — never
  embed or log a message that contains PII.
- `refusal_response` must not call the LLM. Refusals are deterministic strings, which makes them
  fast and impossible to leak advice through generation.
- The advice/performance refusal must include an educational link (PRD FR-27) and the
  disclaimer line.

### Verification

```bash
pytest -q tests/test_guardrails.py
python - <<'PY'
from src import guardrails as g
for q in ["Should I buy HDFC Small Cap Fund?",
          "Which of these funds has the best returns?",
          "Exit load of HDFC Large Cap?",
          "My PAN is ABCDE1234F, fetch my folio"]:
    print(g.detect_pii(q), g.is_advice_intent(q), g.is_performance_intent(q))
PY
```

### Definition of Done

- [ ] All 10 PRD §6.8 queries classified as specified
- [ ] PII detector has no false positives on SIP amounts, dates, percentages
- [ ] `validate_answer` repairs an over-long, multi-URL, missing-citation answer in one call
- [ ] Tests pass with no network

### Commit

`feat(guardrails): intent routing, PII filter, refusals, answer validators`

### Agent prompt (Phase 5)

```
Implement Phase 5 of implementation.md (read implementation.md §PHASE 5 and architecture.md
§7.4/§6.3). Create only src/guardrails.py and tests/test_guardrails.py. No network, no
embedding, no LLM calls in this module. Implement Intent(str, Enum), detect_pii (PAN, Aadhaar,
account number, OTP, email, phone) with no false positives on SIP amounts/years/percentages,
is_advice_intent, is_performance_intent, refusal_response(reason) returning a frozen
AnswerResult with refused=True and an educational link, trim_sentences using a regex that does
not split on decimals, and validate_answer(answer, context_urls, last_updated) returning
(ok, flags, repaired) with the repairs in the documented order. Cover all 10 PRD §6.8 queries in
table-driven tests. Run the Phase 5 verification and paste output. Do not start Phase 6.
```

---

## PHASE 6 — Retrieval (Stages 6–9)

**Goal:** turn a question into a small, diverse, in-scope context set.
**Depends on:** Phases 3, 4, 5.
**Produces:** `src/retrieve.py`, `tests/test_retrieve.py`, first entries in
`logs/retrieval.jsonl`.

### Tasks

6.1 `normalize_query(q) -> str`: lowercase, strip punctuation except `?` and `%`, collapse
    whitespace, drop leading "what is the"/"tell me about" filler.
6.2 `retrieve(question, k, min_score) -> RetrievalResult`:
    1. `vec = embed.embed_query(normalize_query(question))`
    2. `hits = vectorstore.search(vec, k=k)`
    3. keep `score >= min_score` → count `dropped_low_score`
    4. `assert_allowlisted(h["metadata"]["source_url"])`; drop out-of-scope (count
       `dropped_out_of_scope`) and raise `OutOfScopeError` only if the *corpus* is misconfigured
    5. `diversify(...)`
    6. measure `latency_ms`, return `RetrievalResult`
6.3 `diversify(chunks, per_section_cap=config.MAX_CHUNKS_PER_SECTION)`:
    - dedupe identical `chunk_id`s and near-identical text (first 120 chars normalised),
    - cap `MAX_CHUNKS_PER_SECTION` per `(scheme, section)` pair,
    - greedy MMR with `MMR_LAMBDA`: `score = λ·sim − (1−λ)·max sim to already-selected`
      (similarity from chunk metadata score for the first pick, then re-scored with
      `embed.embed_texts` on selected vs candidate),
    - return at most `k` chunks.
6.4 `build_context(chunks) -> str` — numbered blocks:
    `[i] Scheme: … | Section: … | Source: <url>\n<text>`, enforcing `MAX_CONTEXT_TOKENS`
    (drop lowest-scoring blocks, never below 2 blocks).
6.5 `--debug` CLI: `python -m src.retrieve --question "…" --k 5 --min-score 0.35` printing a
    table of hits, scores, sections, and the assembled context. This is the tool used for
    threshold tuning in Phase 10.
6.6 `tests/test_retrieve.py`: use the Phase-4 test collection + a monkeypatched `embed_query`;
    assert low-score drops, allowlist rejection, per-section cap, MMR ordering, and empty-result
    behaviour.

### Implementation notes / gotchas

- `min_score` is cosine **similarity**, not Chroma distance. Do not forget the `1 - d` flip.
  With `hnsw:space=cosine`, distance is already cosine distance; similarity `= 1 - distance`.
- Tune `min_score` empirically in Phase 10 with the 10 PRD queries plus 3 nonsense queries; a
  nonsense query must score below the threshold. Do not leave the PRD placeholder value
  unverified.
- MMR re-scoring costs an extra embed call; with ~200 chunks and `k=5` it is negligible, but do
  not re-score the entire collection.

### Verification

```bash
python -m src.retrieve --question "What is the exit load of HDFC Large Cap Fund?" --debug
python -m src.retrieve --question "elephant banana quantum" --debug   # expect 0 chunks
pytest -q tests/test_retrieve.py
tail -n 3 logs/retrieval.jsonl
```

### Definition of Done

- [ ] Exit-load question returns ≥ 1 chunk from the correct scheme/section with a score
- [ ] Nonsense question returns 0 chunks after thresholding
- [ ] Per-section cap and MMR diversity enforced (test-enforced)
- [ ] Out-of-scope URL can never enter the context (test-enforced)
- [ ] `retrieval.jsonl` gains one line per query

### Commit

`feat(retrieve): stages 6-9 query prep, threshold search, allowlist filter, MMR diversity`

### Agent prompt (Phase 6)

```
Implement Phase 6 of implementation.md (read implementation.md §PHASE 6 and architecture.md
§7.1/§7.2). Create only src/retrieve.py and tests/test_retrieve.py. Implement normalize_query,
retrieve (embed → top-k cosine search → min_score filter on similarity=1-distance → allowlist
filter → diversify), diversify (dedupe + per-section cap + greedy MMR with lambda from config),
build_context (numbered blocks, MAX_CONTEXT_TOKENS budget), and a `--debug` CLI that prints hits,
scores, sections and the context. Never import the LLM. Log one JSON line per query to
logs/retrieval.jsonl. Run the Phase 6 verification commands including the nonsense-query check,
paste real output, and stop. Do not start Phase 7.
```

---

## PHASE 7 — Generation: Prompt + LLM Call (Stages 10–11)

**Goal:** ground Groq in the retrieved context and produce ≤ 3 sentences + 1 link +
last-updated, with an extractive fallback when the model is unavailable.
**Depends on:** Phase 6.
**Produces:** `src/answer.py` (prompt + generation parts only), `tests/test_answer.py`.

### Tasks

7.1 `build_prompt(question, context, last_updated) -> list[dict]`: the `[{"role":"system",...},
    {"role":"user",...}]` pair described in architecture §7.3, using `config.PROMPT_SYSTEM`
    verbatim and the CONTEXT block format.
7.2 `generate(messages) -> str`: POST to `f"{LLM_BASE_URL}/chat/completions"` with
    `{"model": LLM_MODEL, "messages": messages, "stream": False,
    "temperature": LLM_TEMPERATURE, "max_tokens": LLM_NUM_PREDICT}` and
    `Authorization: Bearer GROQ_API_KEY`; `LLM_TIMEOUT_S`; on missing key /
    `ConnectError`/`ReadTimeout`/non-200 raise `LLMUnavailable`. Return
    `response["choices"][0]["message"]["content"].strip()`.
7.3 `extract_citation(answer, context_urls) -> str | None`: return the single URL that appears
    in the answer **and** in `context_urls`; `None` if none. If the answer contains multiple
    in-context URLs, keep the first and drop the rest (via `validate_answer`'s repair).
7.4 `ensure_last_updated(answer, last_updated) -> str`: guarantee a final line
    `Last updated from sources: YYYY-MM-DD` (append if missing, replace if the date differs).
7.5 `extractive_fallback(question, chunks) -> AnswerResult`: top chunk's first
    `MAX_SENTENCES` sentences, verbatim, + its URL + last-updated; `used_fallback=True`.
7.6 `tests/test_answer.py`: mock the HTTP call with `respx`/`monkeypatch`; assert
    request body contains `temperature == 0` and `max_tokens == 220`; assert prompt includes
    the CONTEXT and the question; assert citation extraction picks an in-context URL, ignores an
    invented URL, and handles multiple URLs; assert `ensure_last_updated` idempotence; assert
    `LLMUnavailable` → extractive fallback content is grounded and cited.

### Implementation notes / gotchas

- **`GROQ_API_KEY` must be set in `.env`** before Phase 8 verification; document it in
  `run.sh setup` and the README.
- Do not let the model see the question twice, and do not put the citation requirement only in
  the prompt — Phase 8's validator is the real guarantee.
- Keep `num_predict` at 220: it caps runaway verbosity and protects the latency budget.

### Verification

```bash
# GROQ_API_KEY must already be in .env
python -m pytest -q tests/test_answer.py
python - <<'PY'
from src import answer
from src import retrieve, config
r = retrieve.retrieve("What is the minimum SIP amount for HDFC Small Cap Fund?")
print(answer.generate(answer.build_prompt(r.chunks and "What is the minimum SIP amount for HDFC Small Cap Fund?" or "", answer.build_context(r.chunks), "2026-09-27")))
PY
```

### Definition of Done

- [ ] Request body carries `temperature=0`, `max_tokens=220`
- [ ] Generated answer is grounded in the supplied context in a manual read
- [ ] `LLMUnavailable` path returns a cited extractive answer
- [ ] Tests pass without a live model

### Commit

`feat(answer): stages 10-11 grounded prompt, groq call, citation + fallback helpers`

### Agent prompt (Phase 7)

```
Implement Phase 7 of implementation.md (read implementation.md §PHASE 7 and architecture.md
§7.3/§6.3). Create src/answer.py containing ONLY build_prompt, generate, extract_citation,
ensure_last_updated and extractive_fallback (no orchestration yet — that is Phase 8). Use
config.PROMPT_SYSTEM verbatim as the system message. Call Groq at
LLM_BASE_URL/chat/completions with stream=False, temperature, and max_tokens from config
and Authorization Bearer GROQ_API_KEY; raise LLMUnavailable on missing key, connection
error, timeout, or non-200. extract_citation must return only URLs present in the
supplied context_urls set. Tests must mock HTTP (no live model required) and assert the request
options, prompt contents, citation behaviour, and the fallback path. Run the Phase 7
verification and paste output. Do not start Phase 8.
```

---

## PHASE 8 — Orchestration, Guardrail Enforcement, Logging, CLI (Stages 12–13)

**Goal:** the end-to-end pipeline behind one function, with guardrails applied in the right
order and a self-test.
**Depends on:** Phase 7.
**Produces:** completed `src/answer.py`, `--chat` CLI, `--selftest`.

### Tasks

8.1 `answer_question(question: str) -> AnswerResult` implementing architecture §7.1 in order:
    PII pre-filter → advice/performance intent route (refuse, no LLM) → `retrieve` → empty
    context → `REFUSAL_NO_CONTEXT` → `build_context` → `build_prompt` → `generate` (on
    `LLMUnavailable` → `extractive_fallback`) → `guardrails.validate_answer` → repair pass →
    if still invalid → `extractive_fallback` → assemble `AnswerResult` → `log_retrieval` +
    `log_answer` (redacted).
8.2 `last_updated` value = the `fetched_at` of the **highest-scoring retrieved chunk's
    document** (fall back to the manifest's `generated_at`); never invent a date.
8.3 CLI in `src/answer.py`:
    - `--chat` — REPL: prints the answer, the single source link, the last-updated line, and
      the disclaimer; `exit`/`quit` to stop
    - `--question "..."` — one-shot
    - `--selftest` — asserts: collection exists, chunk count > 100, embedding dim 384, Groq
      reachable, one canned query returns a cited answer; prints a PASS/FAIL table
    - `--json` — machine-readable `AnswerResult` for the eval harness (Phase 10)
8.4 Add `logs/answers.jsonl` writes: question (redacted), answer, citation, flags, latency,
    `refused`, `used_fallback`.
8.5 Edge cases to implement and test: empty/whitespace question; question in a non-English
    script (answer "not found in sources" rather than a guess); question naming an AMC outside
    the corpus (e.g. "Axis Small Cap") → `no_context` refusal; very long question (> 2000
    chars → truncate at 500 chars before embedding).

### Implementation notes / gotchas

- This is the phase where the "≤ 3 sentences + 1 link + last-updated" guarantee becomes real.
  Every factual path must pass `validate_answer`; there is no path that returns raw model text.
- Keep the LLM call in a single place (`generate`). Orchestration only decides what to pass it.
- `answer_question` must be side-effect-free apart from logging — that is what makes the Phase 10
  eval harness possible.

### Verification

```bash
python -m src.answer --selftest
python -m src.answer --question "What is the exit load of HDFC Large Cap Fund?"
python -m src.answer --question "Should I buy HDFC Small Cap Fund?"     # expect refusal
python -m src.answer --question "Which fund has the best returns?"      # expect performance redirect
python -m src.answer --question "My PAN is ABCDE1234F"                  # expect PII refusal
tail -n 5 logs/answers.jsonl
```

### Definition of Done

- [ ] All four commands above behave as specified
- [ ] Every factual answer ≤ 3 sentences, one in-context link, last-updated line
- [ ] No answer path can emit uncited text (validator or fallback always applies)
- [ ] `--selftest` prints all PASS
- [ ] Total latency < 10 s for a factual question on CPU (NFR-02)

### Commit

`feat(answer): stages 12-13 orchestration, guardrail enforcement, logging, CLI`

### Agent prompt (Phase 8)

```
Implement Phase 8 of implementation.md (read implementation.md §PHASE 8 and architecture.md
§7.1 and §7.4). Complete src/answer.py with answer_question() following the exact check order:
PII -> advice/performance intent (deterministic refusal, no LLM) -> retrieve -> no_context
refusal -> build_context -> build_prompt -> generate (LLMUnavailable -> extractive_fallback) ->
guardrails.validate_answer -> single repair pass -> extractive_fallback if still invalid ->
assemble AnswerResult -> log_retrieval + log_answer with PII redaction. last_updated must come
from the top chunk's metadata fetched_at. Add the --chat, --question, --json and --selftest CLI
flags, and handle empty, non-Latin, out-of-corpus-AMC, and over-long inputs. Run the four
Phase 8 verification commands, paste the real outputs including the two refusals, and confirm
latency. Do not start Phase 9.
```

---

## PHASE 9 — Streamlit UI (FR-30…34)

**Goal:** the demo surface — welcome line, 3 example questions, persistent disclaimer, answer
card with one source link and last-updated date.
**Depends on:** Phase 8.
**Produces:** `app.py`.

### Tasks

9.1 Page config: title "HDFC Mutual Fund FAQ Assistant", wide layout.
9.2 Header: welcome line naming the AMC and the 5 schemes/categories, one line each.
9.3 Three example-question chips (from PRD §6.8 Q1–Q5; pick the three most distinct: exit
    load, ELSS lock-in, capital-gains statement). Clicking a chip sets the input and asks.
9.4 Persistent disclaimer banner: `config.DISCLAIMER` (PRD Appendix A).
9.5 Answer rendering: answer text, the source URL as a clickable markdown link labelled with
    the scheme name, the `Last updated from sources:` line, and a "Facts only. Not investment
    advice." footer on every answer.
9.6 Refusals render with the same card shape and a distinct accent; never hide the link.
9.7 Text input (`st.text_input`, Enter to submit) + optional `st.spinner` while answering;
    store only the current turn in `st.session_state` (PRD FR-34) — no persistence.
9.8 Sidebar: scope (5 schemes + links), architecture summary (5 ingestion stages, 8 retrieval
    stages), and the sample questions. No back-end internals, no logs, no chunk dumps.
9.9 `run.sh ui` launches `streamlit run app.py`.

### Implementation notes / gotchas

- Keep `app.py` free of business logic — it renders `AnswerResult` only (architecture §8).
- No source-code display, no chunk previews, no back-end screenshots in the UI or the demo.
- Ensure the disclaimer is visible without scrolling on a 1280×800 window.

### Verification

```bash
run.sh ui          # then manually: load page, click all 3 chips, ask 1 refusal, ask 1 PII question
```

### Definition of Done

- [ ] Welcome line names HDFC + 5 schemes
- [ ] Exactly 3 example questions, clickable
- [ ] Disclaimer visible without scrolling
- [ ] Every answer shows one clickable source link + last-updated date
- [ ] Refusal path renders correctly and shows its educational link
- [ ] Nothing persisted to disk

### Commit

`feat(ui): streamlit interface with example questions and disclaimer`

### Agent prompt (Phase 9)

```
Implement Phase 9 of implementation.md (read implementation.md §PHASE 9 and architecture.md §3
UI layer). Create only app.py. It must contain zero business logic: it calls
answer.answer_question() and renders the returned AnswerResult. Required UI: page title, a
welcome line naming HDFC and the 5 schemes with their categories, exactly 3 clickable example
questions, a persistent disclaimer banner using config.DISCLAIMER, a text input, and an answer
card showing the answer, exactly one clickable source link, and the "Last updated from sources:"
line. Refusals must render in the same card shape with their educational link. Keep state in
st.session_state only; do not persist anything; do not show source code, logs, or chunk
contents. Add a sidebar with scope, links, and the RAG stage summary. Verify by running
`run.sh ui` and describing what you clicked and what rendered. Do not start Phase 10.
```

---

## PHASE 10 — Tests, Threshold Tuning, Evaluation Harness

**Goal:** prove the PRD's acceptance criteria with a repeatable, offline evaluation.
**Depends on:** Phase 8 (and 9 for the final sample answers).
**Produces:** `tests/` completed, `eval/golden_queries.json`, `eval/run_eval.py`,
`tuning_notes.md` (input to README), `sample_qa.md`.

### Tasks

10.1 `eval/golden_queries.json`: the 10 PRD §6.8 queries with `id`, `question`, `expected_type`
    (`fact` / `how_to` / `advice` / `pii`), `expected_scheme` (or `null`), `must_mention`
    (keywords, e.g. `["1%"]` for exit load) — plus 3 nonsense queries with
    `expected_type: "no_context"`.
10.2 `eval/run_eval.py`: loops the golden set calling `answer_question` (real LLM optional via
    `--live`; default uses the deterministic extractive path so CI is offline), scores:
    - type correctness (refused vs answered, advice/pii refused with a link)
    - citation present and in the allowlist
    - ≤ 3 sentences
    - `Last updated from sources:` present
    - `must_mention` keywords present for FACT rows
    - latency
    Writes `eval/report.json` + a Markdown table.
10.3 Threshold tuning: sweep `min_score ∈ {0.25, 0.30, 0.35, 0.40}` and `MMR_LAMBDA ∈
    {0.5, 0.7}` over the golden set using `python -m src.retrieve --debug`; confirm nonsense
    queries score below the chosen threshold and that all FACT queries retrieve an
    above-threshold chunk from the right scheme. Record the chosen values and the evidence in
    `tuning_notes.md`, then update `config.py` (with a comment-free change — the numbers in
    `config.py` are the source of truth referenced by the README).
10.4 Unit tests to finish: `test_config.py` (5 URLs, prompt non-empty), `test_embed.py`,
    `test_vectorstore.py` (tmp dir), `test_retrieve.py`, `test_answer.py`, `test_guardrails.py`
    (all 10 PRD queries), `test_ingest.py`, `test_chunking.py`. Whole suite must run offline
    with no live model (`pytest -q` green in < 60 s).
10.5 Numeric spot check: manually confirm the 5 must-have fact types (expense ratio, exit load,
    minimum SIP, ELSS lock-in, benchmark/riskometer) appear in the retrieved chunks and that
    the generated answers match the source text.
10.6 `sample_qa.md`: 5–10 real query/answer/link transcripts captured from the running system.

### Implementation notes / gotchas

- The eval harness must not silently depend on a live LLM; default to the extractive path and
  gate the live path behind `--live` so the suite is reproducible.
- Tuning is where the PRD placeholder `0.35` gets replaced with a defended number. Record the
  sweep table in `tuning_notes.md`; the README quotes it.

### Verification

```bash
pytest -q
python eval/run_eval.py                 # offline, deterministic
python eval/run_eval.py --live          # with GROQ_API_KEY set (calls Groq)
```

### Definition of Done

- [ ] `pytest -q` green offline
- [ ] ≥ 8/10 PRD queries answered correctly with exactly one valid in-corpus link
- [ ] All 3 advice/performance queries refused with an educational link
- [ ] PII query refused; `logs/answers.jsonl` contains no raw PII
- [ ] 3 nonsense queries return `no_context`
- [ ] `tuning_notes.md` documents the final `min_score` and `MMR_LAMBDA` with the sweep evidence
- [ ] `sample_qa.md` has 5–10 real transcripts

### Commit

`test: golden-query eval harness, threshold tuning, full offline suite`

### Agent prompt (Phase 10)

```
Implement Phase 10 of implementation.md (read implementation.md §PHASE 10 and architecture.md
§10/§12). Create eval/golden_queries.json (the 10 PRD §6.8 queries + 3 nonsense), eval/run_eval.py
(must run fully offline by default via the extractive path, with --live for the LLM), and
complete the tests/ suite so `pytest -q` is green offline in under 60s. Score type correctness,
citation presence and allowlist membership, sentence count <= 3, presence of the last-updated
line, must_mention keywords, and latency; write eval/report.json. Then sweep min_score in
{0.25,0.30,0.35,0.40} and MMR_LAMBDA in {0.5,0.7} using src.retrieve --debug, choose the
defensible values, update config.py, and record the sweep table plus justification in
tuning_notes.md. Finally produce sample_qa.md with 5-10 real transcripts. Paste the eval table
and the pytest output. Do not start Phase 11.
```

---

## PHASE 11 — Deliverables: Docs, Source List, Demo

**Goal:** produce the submission artefacts from PRD §12.
**Depends on:** Phases 9, 10.
**Produces:** `README.md`, `sources.csv`, `DISCLAIMER.md`, `sample_qa.md` (final), demo video,
`demo_script.md`.

### Tasks

11.1 `sources.csv` — **generate** from `corpus_manifest.json`, never hand-type:
    columns `url,scheme_name,category,plan,page_title,fetched_at,content_sha256,chunk_count,source_type`
    (5 data rows, `source_type=scheme_page`).
11.2 `README.md` sections: overview · scope (HDFC + 5 schemes table) · **architecture** with
    the ingestion (1–5) and retrieval (6–13) stage list · **chunking strategy decision** with
    the parameters, the rationale from architecture §5, and the ablation note (recursive vs
    semantic — one paragraph: why recursive won) · setup (`python -m venv`, `pip install`,
    `GROQ_API_KEY` in `.env`) · run (`run.sh ingest|chat|ui|test`) · project layout ·
    **tuning results** (final `min_score`, `MMR_LAMBDA`, chunk stats from
    `logs/chunking_report.json`) · sample Q&A link · **known limits** (copy PRD §13) ·
    disclaimer.
11.3 `DISCLAIMER.md` — PRD Appendix A verbatim, marked as the exact text used in the UI.
11.4 `demo_script.md` — a 3–4 minute shot list: intro + scope, 3 example questions via chips,
    one refusal, one PII refusal, the "Last updated from sources:" line, architecture recap,
    disclaimer. **UI only — no back-end screens, no terminals showing logs, no third-party
    blog pages.**
11.5 Record the demo video, or host the project and link it if hosting is possible.
11.6 Final gate: run the full checklist below and paste the result.

### Verification (final gate)

```bash
pytest -q
python -m src.ingest --stats
python -m src.answer --selftest
python eval/run_eval.py --live
python -c "import csv;rows=list(csv.DictReader(open('sources.csv')));assert len(rows)==5;print('sources.csv ok')"
```

### Definition of Done — submission checklist

- [ ] `PRD.md`, `architecture.md`, `implementation.md` present
- [ ] `README.md` with setup, scope, architecture, chunking decision, tuning, known limits
- [ ] `sources.csv` with exactly 5 URLs, generated from the manifest
- [ ] `sample_qa.md` with 5–10 answers, each with a working link
- [ ] `DISCLAIMER.md` matching the UI text exactly
- [ ] Demo video recorded (UI only) or project hosted with a link
- [ ] All Phase 10 acceptance criteria still pass on a clean clone run
- [ ] No back-end screenshots, no third-party blog sources, no PII anywhere in the repo or logs

### Commit

`docs: README, sources.csv, disclaimer, sample Q&A, demo script`

### Agent prompt (Phase 11)

```
Implement Phase 11 of implementation.md (read implementation.md §PHASE 11 and architecture.md
§14). Generate sources.csv from corpus_manifest.json (5 rows, columns url,scheme_name,category,
plan,page_title,fetched_at,content_sha256,chunk_count,source_type). Write DISCLAIMER.md with the
PRD Appendix A text verbatim. Write README.md with all required sections, including the chunking
strategy decision (parameters + recursive-vs-semantic rationale) and the tuned min_score /
MMR_LAMBDA values quoted from tuning_notes.md, and the known limits from PRD §13. Write
demo_script.md as a 3-4 minute UI-only shot list. Then run the final gate commands and paste
the outputs. Do not invent numbers anywhere: every figure must come from a generated artifact
(chunking_report.json, manifest, eval/report.json).
```

---

## 12. Cross-Phase Verification Matrix

| Requirement | Verified in phase | How |
|-------------|------------------|-----|
| FR-01…05 ingestion | 1, 4 | `build_corpus` (5/5), double-ingest idempotency, manifest shape |
| FR-06…10 chunking | 2 | `test_chunking.py` + `chunking_report.json` fact integrity |
| FR-11…13 embed/store | 3, 4 | 384-dim/unit-norm test, collection persistence test |
| FR-14…18 retrieval | 6, 10 | nonsense-query threshold check, allowlist test, eval report |
| FR-19…25 generation | 7, 8, 10 | request-options test, sentence count, citation in eval |
| FR-26…29 guardrails | 5, 8, 10 | 10 PRD queries classified; refusal + PII outputs in eval |
| FR-30…34 UI | 9 | manual click-through checklist |
| NFR-01…08 | 4, 8, 10 | offline suite, latency measurement, memory note in README |
| PRD §12 deliverables | 11 | final gate checklist |

---

## 13. Risk Register for the Build

| Risk | Phase where it appears | Mitigation baked into this plan |
|------|------------------------|----------------------------------|
| Page extraction returns boilerplate or empty | 1 | 500-char floor, BS4 density fallback, stop on incomplete corpus |
| Numbers split across chunks | 2 | table-row protection, fact-integrity probes |
| Missing `GROQ_API_KEY` or Groq unreachable | 7, 8, 11 | `run.sh setup` checks `.env`; extractive fallback (FR-25) |
| Threshold too low → wrong-scheme answers | 10 | sweep + nonsense-query check; values recorded in `tuning_notes.md` |
| Threshold too high → "not found" on valid questions | 10 | all 10 golden FACT/HOW_TO queries must retrieve an above-threshold chunk |
| Model ignores the citation rule | 8 | validator + repair + extractive fallback; no raw-text path exists |
| CPU latency > 10 s | 8 | `num_predict=220`, cached model, streaming avoided for determinism |
| Scope creep into a "financial advisor" | any | guardrails module is mandatory and tested before generation exists |

---

## 14. Definition of Done — Project Level

The project is demo-ready when all of the following are true:

1. `python -m src.ingest` builds the corpus from the 5 allowlisted URLs, idempotently.
2. Chunking strategy is documented with parameters and a written rationale.
3. `python -m src.answer --question "<factual question>"` returns ≤ 3 sentences, exactly one
   in-corpus source link, and `Last updated from sources: <date>`.
4. Advice, performance, and PII queries are refused deterministically with a link.
5. `streamlit run app.py` shows the welcome line, 3 examples, and the disclaimer.
6. `pytest -q` is green offline; `eval/run_eval.py` shows ≥ 8/10 correct with valid citations.
7. README, sources.csv, sample_qa.md, DISCLAIMER.md, and the demo video exist.
8. Zero violations of the PRD §9 constraints.

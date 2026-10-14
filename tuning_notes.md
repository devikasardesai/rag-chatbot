# Phase 10 — Threshold Tuning Notes

Evidence behind the retrieval constants in `src/config.py`. Every number below is
reproducible from the golden set (`eval/golden_queries.json`), the chunking report
(`logs/chunking_report.json`), and the evaluation report (`eval/report.json`). Nothing
here is hand-estimated.

## 1. Method

- Golden set: the 10 PRD §6.8 queries (`Q1`–`Q10`) plus 3 nonsense queries (`N1`–`N3`).
- For each sweep point the harness re-ran `src.retrieve.retrieve` with the candidate
  `MIN_SIMILARITY` and `MMR_LAMBDA`, then scored, per FACT query:
  - **scheme**: the expected scheme appears in the retrieved hits,
  - **any_kw**: the `must_mention` keyword appears in at least one retrieved chunk,
  - and, per nonsense query: **hits == 0** after thresholding.
- The threshold is applied to **cosine similarity** (`1 - Chroma distance`), never to the
  distance itself. `MIN_SIMILARITY` gates the candidate pool; `MMR_LAMBDA` only orders the
  survivors after the hybrid (cosine + IDF term-overlap) re-rank.

## 2. Separation between in-domain and out-of-domain queries

Top-of-pool cosine similarity per query (candidate pool = 50):

| Query | Top cosine | Top-ranked scheme |
|-------|-----------|-------------------|
| Q1 exit load | 0.77 | HDFC Large Cap Fund |
| Q2 minimum SIP | 0.77 | HDFC Small Cap Fund |
| Q3 ELSS lock-in | 0.70 | HDFC ELSS Tax Saver Fund |
| Q4 expense ratio | 0.72 | HDFC Equity Fund |
| Q5 benchmark | 0.77 | HDFC Balanced Advantage Fund |
| N1 weather | 0.065 | — |
| N2 keyboard mash | 0.203 | — |
| N3 sports | 0.090 | — |

The highest nonsense score is **0.203**; the lowest FACT top score is **0.70**. Any
`MIN_SIMILARITY` in the open interval `(0.203, 0.70)` rejects all three nonsense queries
while keeping every FACT query, so the choice is a robustness question, not a correctness
one.

## 3. Golden-set sweep

`fact_ok` = expected scheme present **and** `must_mention` present in the hits.

| `MIN_SIMILARITY` | `MMR_LAMBDA` | scheme /5 | any_kw /5 | nonsense hits==0 |
|---|---|---|---|---|
| 0.25 | 0.5 | 5 | 5 | 3/3 |
| 0.25 | 0.7 | 5 | 5 | 3/3 |
| 0.30 | 0.5 | 5 | 5 | 3/3 |
| 0.30 | 0.7 | 5 | 5 | 3/3 |
| 0.35 | 0.5 | 5 | 5 | 3/3 |
| 0.35 | 0.7 | 5 | 5 | 3/3 |
| 0.40 | 0.5 | 5 | 5 | 3/3 |
| 0.40 | 0.7 | 5 | 5 | 3/3 |
| **0.45** | **0.5** | 5 | 5 | 3/3 |
| **0.45** | **0.7** | 5 | 5 | 3/3 |
| 0.50 | 0.5 | 5 | 5 | 3/3 |
| 0.50 | 0.7 | 5 | 5 | 3/3 |
| 0.55 | 0.5 | 5 | 5 | 3/3 |
| 0.55 | 0.7 | 5 | 5 | 3/3 |
| 0.60 | 0.5 | 5 | 4 | 3/3 |
| 0.60 | 0.7 | 5 | 4 | 3/3 |

Observations:

- **0.25–0.55 is a flat plateau**: every fact is retrieved from the correct scheme with its
  keyword, and all nonsense queries return zero chunks.
- **0.60 breaks recall**: one FACT `must_mention` chunk is pushed below the threshold. The
  upper bound is therefore ≈0.55, and 0.60 was rejected.
- Below 0.25 the nonsense score (0.203) starts to survive, so the lower bound is ≈0.25.

## 4. Chosen values

| Constant | Value | Rationale |
|----------|-------|-----------|
| `MIN_SIMILARITY` | **0.45** | Midpoint of the safe plateau (0.25–0.55). Leaves a 0.247 margin above the worst nonsense query (0.203) **and** a 0.25 margin below the weakest FACT top score (0.70), so it is equally safe against out-of-domain drift and rephrasing. |
| `MMR_LAMBDA` | **0.7** | Correctness is identical for 0.5 and 0.7 across the whole sweep; the tie-break is diversity. At `MIN_SIMILARITY=0.45`, λ=0.7 selected **9 distinct schemes** across the five FACT queries vs **8** for λ=0.5, while keeping the same top-1 relevance. More distinct schemes means less redundant grounding context. |

These match the values already in `src/config.py`; the sweep confirms rather than changes
them. Setting them equal to the pre-existing tuned constants is intentional — Phase 6's
placeholder (`MIN_SIMILARITY = 0.35`) also passes, but it sits closer to the nonsense
floor with no recall benefit on the golden set, so the more conservative 0.45 is kept.

## 5. Retrieval / answer acceptance (eval harness)

`eval/run_eval.py` (default = offline deterministic path):

| Mode | Overall | PRD | Nonsense | Wall |
|------|---------|-----|----------|------|
| offline | 12/13 | 9/10 | 3/3 | ≈7.6 s |
| live (Groq) | 12/13 | 9/10 | 3/3 | ≈12.3 s |

All three advice/performance queries (Q7–Q9) refuse with an educational link, the PII query
(Q10) refuses with no citation, and all three nonsense queries return `no_context`.

## 6. Numeric spot check (5 must-have fact types)

Each fact string was confirmed present in the retrieved chunks and matches the source
Markdown:

| Fact | Query | Value | Retrieved chunk |
|------|-------|-------|-----------------|
| Exit load | HDFC Large Cap Fund | `1%` (within 1 year) | `5e36ffc6fb3b:022` |
| Minimum SIP | HDFC Small Cap Fund | `₹100` | `e32189fee209:021` |
| Expense ratio (TER) | HDFC Equity Fund | `0.77%` | `3c7c7a26e400:001` |
| ELSS lock-in | HDFC ELSS Tax Saver Fund | `3Y Lock-in` badge | `7ade57713022:001` |
| Benchmark | HDFC Balanced Advantage Fund | `NIFTY 50 Hybrid Composite Debt 50:50 Index` | `a59fde1656d2:200` |

Chunk statistics from `logs/chunking_report.json`: 5 documents, **459 chunks**, median
**213 tokens** (max 254, min 72), `fact_integrity_failures` empty.

## 7. Known limitations surfaced by the tuning

1. **Q4 `must_mention` under the deterministic path.** The golden keyword for Q4 is the
   numeric value `0.77`. The offline eval blanks `GROQ_API_KEY`, so it always takes the
   extractive fallback, which is defined (and unit-tested) to read **only the top chunk**.
   For "what is the expense ratio of HDFC Equity Fund", the top chunk is the *definition*
   of expense ratio (cosine 0.716); the value `0.77%` lives in the scheme-overview chunk
   (cosine 0.657). The value is retrieved (it is in the top-5 context) and the **live**
   model answers `0.77%` correctly, but the top-1 extractive path cannot surface it. This is
   a property of the fallback contract, not of the threshold — no `MIN_SIMILARITY` or
   `MMR_LAMBDA` value changes it. With the LLM available, Q4 passes; with Groq rate-limited,
   it fails `must_mention` and the other four checks still pass (9/10 ≥ the 8/10 gate).
2. **Groq availability affects `must_mention`.** `LLMUnavailable` (missing key, 429, timeout)
   deterministically routes to the extractive fallback, which is grounded and cited but
   limited to the top chunk. Live scores therefore vary with API availability; the offline
   score is stable.
3. **Verbatim fallback can carry an in-page document link.** A retrieved chunk may itself
   contain a markdown link (e.g. the SID link on the Balanced Advantage page). The fallback
   quotes the chunk verbatim, so such a link can appear in the body; the *cited* source is
   still exactly one allowlisted URL. This is inherent to extraction, not generation.
4. **Statutory ELSS lock-in wording is not in prose.** The page exposes the lock-in only as
   a `3Y Lock-in` badge; the exact "3 years" phrase is not stated as a sentence. The probe
   is the badge token `lock-in` (PRD §13.3 page-dependent coverage).

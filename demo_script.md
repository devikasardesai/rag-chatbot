# Demo Script — 3–4 minutes (UI only)

Recording rules: **show the UI only.** No terminal, no back-end screens, no logs, no chunk
dumps, no third-party blog pages. Target resolution 1280×800 so the disclaimer is visible
without scrolling.

Pre-flight (do not record):
- `./run.sh ingest` completed (or the Chroma collection already exists).
- `.env` has `GROQ_API_KEY`; `./run.sh selftest` reports all PASS.
- `./run.sh ui` is running at `http://localhost:8501`; browser at 1280×800, zoom 100%.
- One warm-up question asked off-camera so the embedding model is loaded (avoids a slow
  first render).

---

## 0:00–0:30 — Intro and scope

- Open the app. Let the camera rest on the title and welcome line.
- Narrate: "This is a facts-only FAQ assistant for five HDFC Asset Management schemes —
  Large Cap, Flexi Cap, ELSS Tax Saver, Small Cap, and Balanced Advantage, all Direct Growth.
  It answers only from those five public pages and never gives advice."
- Point at the disclaimer banner (visible without scrolling) and read the key phrase:
  "Facts only. No investment advice."

## 0:30–1:20 — Three example questions via the chips

Click each chip in turn and let each answer render:

1. **"What is the exit load of HDFC Large Cap Fund?"** → 1% within 1 year, one source link.
2. **"What is the lock-in period for HDFC ELSS Tax Saver Fund?"** → three-year lock-in.
3. **"How do I download the capital-gains statement?"** → "Not found in the indexed
   sources" with the investor-education reference (this page is not in the corpus).

For each answer, pause on the answer card and highlight:
- the answer text (≤ 3 sentences),
- exactly one clickable source link labelled with the scheme name,
- the `Last updated from sources: 2026-09-27` line,
- the footer "Facts only. Not investment advice."

## 1:20–2:00 — Refusal: advice question

Type into the box and submit: **"Should I buy HDFC Small Cap Fund?"**

Narrate: "Advice questions are refused deterministically, before any model call. Notice the
distinct refusal card, the facts-only explanation, and the educational link — and still no
numbers or recommendations."

## 2:00–2:35 — Refusal: PII question

Type and submit: **"My PAN is ABCDE1234F, please fetch my folio"**

Narrate: "The assistant never accepts or stores personal information. This is refused before
retrieval, no source link is shown, and nothing typed here is written to disk."

## 2:35–3:05 — Architecture recap (still on the UI)

Scroll the sidebar (or point to it) and walk the two stage lists:

- Ingestion (runs once, offline): Load → Clean → Chunk → Embed → Store.
- Question answering: normalize/embed → search → threshold + allowlist → dedupe/MMR →
  ground → generate → validate → log.

Narrate that the question is answered only from the retrieved context, and that a validator
enforces the single citation and the sentence limit.

## 3:05–3:30 — Close

- Show the persistent disclaimer once more.
- Optionally click the Large Cap chip again to end on a clean factual answer card.
- Narrate: "Facts only, one citation, last-updated date on every answer — grounded in five
  public HDFC scheme pages."

Stop recording.

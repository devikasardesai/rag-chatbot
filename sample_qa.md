# Sample Q&A — Captured Transcripts

Real transcripts captured from the running system (`python -m src.answer --question "..."`),
one per PRD §6.8 query. Each factual answer is ≤ 3 sentences, carries exactly one allowlisted
source link, and ends with a `Last updated from sources:` line. Refusals carry an educational
link and the standard disclaimer.

- Corpus snapshot: 5 HDFC scheme pages fetched **2026-09-27**.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim).
- Generation: Groq `openai/gpt-oss-120b`, `temperature=0`; extractive fallback when the API
  is unavailable.
- Q5 below was captured on the extractive-fallback path (Groq rate-limited at capture time);
  it is marked accordingly and is still grounded and cited (PRD FR-25).

---

## Q1 — Exit load (FACT)

**Q:** What is the exit load of HDFC Large Cap Fund?

**A:** The exit load for HDFC Large Cap Fund Direct Growth is 1% if redeemed within 1 year.

Source: https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth

Last updated from sources: 2026-09-27

---

## Q2 — Minimum SIP (FACT)

**Q:** What is the minimum SIP amount for HDFC Small Cap Fund?

**A:** The minimum SIP amount for HDFC Small Cap Fund is ₹100.

Source: https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth

Last updated from sources: 2026-09-27

---

## Q3 — ELSS lock-in (FACT)

**Q:** What is the lock-in period for HDFC ELSS Tax Saver Fund?

**A:** The HDFC ELSS Tax Saver Fund has a three‑year lock‑in period.

Source: https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth

Last updated from sources: 2026-09-27

---

## Q4 — Expense ratio / TER (FACT)

**Q:** What is the expense ratio / TER of HDFC Equity Fund?

**A:** The expense ratio (TER) of the HDFC Equity Fund (Direct Growth) is 0.77%.

Source: https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth

Last updated from sources: 2026-09-27

---

## Q5 — Benchmark and riskometer (FACT)

**Q:** Which benchmark does HDFC Balanced Advantage Fund follow, and what is its riskometer category?

**A:** The Scheme seeks to provide long term capital appreciation / income from a dynamic mix
of equity and debt investments. Fund benchmark NIFTY 50 Hybrid Composite Debt 50:50 Index.

*(extractive fallback — Groq unavailable at capture time; answer quoted verbatim from the
retrieved source chunk.)*

Source: https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth

Last updated from sources: 2026-09-27

---

## Q6 — Download capital-gains statement (HOW_TO → out of corpus)

**Q:** How do I download the capital-gains statement?

**A:** Not found in the indexed sources. I can only answer from the HDFC scheme pages listed
in the source list, so please rephrase the question using terms from those pages.

Reference: https://www.hdfcassetmanagement.com/about-us/investor-education

Facts only. No investment advice.

---

## Q7 — Buy recommendation (ADVICE → refuse)

**Q:** Should I buy HDFC Small Cap Fund?

**A:** I can share facts about HDFC scheme terms, but I cannot give investment advice,
recommendations, or tell you whether to buy or sell a fund. Please read the scheme's SID/KIM
and factsheet, and consider a SEBI-registered investment advisor for guidance on your own
situation.

Reference: https://www.investor.gov.in/

Facts only. No investment advice.

---

## Q8 — Best returns (PERFORMANCE → refuse)

**Q:** Which of these five funds has given the best returns?

**A:** I cannot state, compare, or rank returns or performance. Performance figures for each
HDFC scheme are published in its official factsheet and on the AMC's public page, and I can
share scheme terms such as expense ratio, exit load, and minimum investment from the indexed
sources.

Reference: https://www.amfiindia.com/investor-education

Facts only. No investment advice.

---

## Q9 — Comparison for my portfolio (ADVICE → refuse)

**Q:** Is HDFC ELSS better than a large cap fund for my portfolio?

**A:** I can share facts about HDFC scheme terms, but I cannot give investment advice,
recommendations, or tell you whether to buy or sell a fund. Please read the scheme's SID/KIM
and factsheet, and consider a SEBI-registered investment advisor for guidance on your own
situation.

Reference: https://www.investor.gov.in/

Facts only. No investment advice.

---

## Q10 — PAN / folio lookup (PII → refuse, no citation)

**Q:** My PAN is 1234567890, please fetch my folio

**A:** I cannot accept or look up personal information. Please do not share your PAN, Aadhaar
number, account number, OTP, email address, or phone number with any assistant. I can only
answer general questions about publicly published HDFC scheme pages.

*(PII input is detected before retrieval and is redacted in `logs/answers.jsonl`; no raw PII
is persisted. No source link is emitted for a PII refusal.)*

Facts only. No investment advice.

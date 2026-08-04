# Human Evaluation Log

Manual review of VibeScope's output, recorded in a parseable format so the results can be
read without running a demo.

**Reviewer:** Youssef Shaaban (project author)
**Mode:** offline (`python -m src.main --offline -q "<input>"`) — deterministic, so every
row below is reproducible by re-running the command in the *Test Input* column.
**Method:** for each input I read the parsed profile, the retrieved documents, the ranked
list, and the citations, then judged the row against the stated criterion. This is
single-reviewer review by the system's author, which is the weakest form of human
evaluation — see [Known bias in this review](#known-bias-in-this-review).

---

## 1. Recommendation quality

| # | Test Input | Evaluation Criteria | Result | Notes |
|---|---|---|---|---|
| 1 | `"I need something for a long late-night study session, instrumental and calm"` | Top 3 are low-energy instrumentals; no vocal-forward tracks | **Pass** | Focus Flow, Midnight Coding, Library Rain. All lofi, all under 0.45 energy |
| 2 | `"high energy music for the gym, nothing acoustic"` | Top 3 all above 0.85 energy and below 0.15 acousticness | **Pass** | Pulse Machine (0.95/0.03), Storm Runner (0.91/0.10), Gym Hero (0.93/0.05) |
| 3 | `"something moody and cinematic for driving at night"` | #1 is the synthwave track | **Pass** | Night Drive Loop at #1, blended 7.18 |
| 4 | `"I just went through a breakup and want something sad and quiet"` | #1 is low-valence and low-energy | **Pass** | Winter Elegy (valence 0.28, energy 0.30) |
| 5 | `"upbeat party music people can dance to"` | Top 3 have danceability > 0.75 and valence > 0.75 | **Pass** | Sunrise City (0.79/0.84), Rooftop Lights (0.82/0.81), Pulse Machine (0.92/0.80) |
| 6 | `"upbeat party music people can dance to"` | The catalog's most danceable track reaches the top 3 | **Partial** | Concrete Kingdom (0.90, highest in catalog) ranks #5, behind Gym Hero (0.88). Danceability is not a scored feature — a real gap, not a tuning issue |
| 7 | `"quiet acoustic folk for a rainy afternoon"` | #1 is acoustic (>0.8) and folk-adjacent | **Pass** | Paper Boats (folk, acousticness 0.90) |
| 8 | `"background music for reading in a coffee shop"` | #1 is the jazz track | **Pass** | Coffee Shop Stories at #1 |

## 2. Explanation quality and grounding

| # | Test Input | Evaluation Criteria | Result | Notes |
|---|---|---|---|---|
| 9 | `"high energy music for the gym, nothing acoustic"` | Every cited `doc_id` exists and genuinely supports the pick | **Pass** | 3/3 citations resolve; each note explicitly mentions gym/workout use |
| 10 | `"quiet acoustic folk for a rainy afternoon"` | Rationale reflects the retrieved note, not just the numbers | **Pass** | Cites `rev-paper-boats`, whose note names rainy days directly |
| 11 | `"polka accordion music for a moon landing"` | Does **not** invent a citation when nothing matches | **Pass** | Reports "nothing in the knowledge base supported this pick"; zero citations emitted |
| 12 | Same as #1 | Rule score, evidence score and blended score are all shown and add up | **Pass** | Focus Flow: rules 5.58 + evidence contribution 2.50 = blended 8.08 |

## 3. Failure handling

| # | Test Input | Evaluation Criteria | Result | Notes |
|---|---|---|---|---|
| 13 | `"polka accordion music for a moon landing"` | Handles gracefully — no crash, no empty list | **Pass** | Returns 5 mid-energy songs, flags all 5 low-confidence, states the caveat |
| 14 | `"   "` (whitespace only) | Handles gracefully | **Pass** | Prints `Failed on query "   ": query must be a non-empty string` and exits 1 — no traceback |
| 15 | `"I want something really intense but also melancholy and slow"` (contradictory) | Resolves the conflict, and signals that it is a conflict | **Pass** | Returns a ranking with compressed score margins; mean confidence 0.42 vs 0.65 on a clear request |
| 16 | No `ANTHROPIC_API_KEY` set | Runs end to end and says which mode it used | **Pass** | Prints an offline-mode notice; every result shows `parse=offline, rerank=offline` |
| 17 | Model returns a hallucinated `song_id` (injected via fake LLM in `tests/test_pipeline.py`) | Invalid song dropped, list still full | **Pass** | `4242` dropped and logged; list topped up from the blended order |
| 18 | Model returns a hallucinated `doc_id` (injected) | Citation stripped, pick marked unsupported | **Pass** | Groundedness falls below 1.0 and the removal is logged as a warning |
| 19 | Model returns `confidence: 42` (injected) | Treated as malformed, not as very sure | **Pass** | Schema guardrail rejects the whole ranking; falls back to the offline re-ranker |

## 4. Confidence calibration

| # | Test Input | Evaluation Criteria | Result | Notes |
|---|---|---|---|---|
| 20 | `"calm instrumental music for studying"` | Mean confidence is higher than on an unanswerable request | **Pass** | 0.65 |
| 21 | `"polka accordion music for a moon landing"` | Mean confidence is low and low-confidence picks are flagged | **Pass** | 0.21; all 5 picks flagged `<-- LOW`, all with `evidence 0.00` |
| 22 | Across all 9 golden cases | `confidence_discrimination` (answerable − unanswerable) is positive | **Pass** | +0.271, gated at ≥ 0.10 in `src/config.py` |

---

## Summary

**21 Pass, 1 Partial, 0 Fail** across 22 checks. Rows 17–19 use faults injected through a
fake LLM rather than naturally occurring model output.

The one partial result is real and is not something I can fix by tuning: **danceability
exists in the catalog but is not a scored feature**, so party requests rank on energy and
valence alone and miss the single most relevant signal for that use case. Recorded in
`model_card.md` § 8 as future work rather than quietly dropped.

Rows 13–19 check the behaviour the automated suite asserts, but from the user's side of the
interface rather than from inside the code.

### Corrections made during this review

Two rows in my first draft of this table were wrong, and re-running the commands caught
them. I had recorded Velvet Hours appearing at #4 for the party query — it does not appear
in the top 5 at all — and I had carried a confidence figure of 0.56 from a *differently
worded* eval case into row 20, where the actual value is 0.65. Both are now taken from the
verbatim command output. Worth stating plainly: a hand-written evaluation table is itself an
artefact that can contain errors, which is exactly why the gated metrics in
`python -m src.evaluation` matter more than this file.

## Known bias in this review

I wrote the system, the knowledge base and these criteria, and I am the only reviewer. That
makes rows 1–7 in particular an author grading their own homework: "top 3 are low-energy
instrumentals" is a criterion I chose *because* I knew the system could meet it. The
failure-handling and grounding rows (8–20) are more trustworthy, because they check
falsifiable properties — does this `doc_id` exist, did it crash, is confidence lower here
than there — rather than taste. A second reviewer, and criteria written before seeing any
output, would be the next real improvement to this evaluation.

## Reproducing

```bash
python -m src.main --offline -q "<Test Input>"   # rows 1-14, 18-19
python -m src.evaluation                         # row 20
pytest tests/test_pipeline.py                    # rows 16-17
```

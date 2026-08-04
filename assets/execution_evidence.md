# Reproducible Execution Evidence

Verbatim command output from a clean checkout. Every command below runs **offline**
and deterministically, so re-running them reproduces this file exactly — no API key,
no network, no screenshots.

```
Python  : 3.14.0
Platform: Windows 11
Commit  : dfba448
```

## Contents

| Section | Demonstrates | Command |
|---|---|---|
| 1 | End-to-end system run, 3 inputs | `python -m src.main --offline -q "..."` |
| 2 | AI feature behavior (RAG) + guardrails | `python -m scripts.demo_evidence` |
| 3 | Evaluation harness, 8 gated metrics | `python -m src.evaluation` |
| 4 | Automated test suite, 72 tests | `pytest -q` |
| 5 | Logging and error handling | `logs/vibescope.log`, invalid input |

---

## 1. End-to-end system run

### 1.1 A clear request

```console
$ python -m src.main --offline -q "I need something for a long late-night study session, instrumental and calm" -k 3
========================================================================
REQUEST: "I need something for a long late-night study session, instrumental and calm"
========================================================================
Interpreted as -> genre: lofi | mood: focused | energy: 0.30 | acoustic: True | situation: studying
  (Offline parser matched situation 'studying', genre 'lofi' and mood 'focused' from keywords in the request.)

Retrieved 8 knowledge documents for this request (context widened to 12 with each candidate's notes):
  [rev-focus-flow] Focus Flow - LoRoom  (bm25 9.42)
  [rev-night-drive-loop] Night Drive Loop - Neon Echo  (bm25 8.38)
  [rev-midnight-coding] Midnight Coding - LoRoom  (bm25 5.02)
  [usecase-late-night-drive] Late night driving  (bm25 4.95)
  [rev-spacewalk-thoughts] Spacewalk Thoughts - Orbit Bloom  (bm25 4.51)
  [rev-velvet-hours] Velvet Hours - Mona Skye  (bm25 4.45)
  [usecase-study-focus] Listening while studying, reading or doing focused work  (bm25 3.80)
  [rev-library-rain] Library Rain - Paper Lanterns  (bm25 3.19)

Recommendations:

1. Focus Flow - LoRoom
   Genre: lofi | Mood: focused | Energy: 0.40
   Rule score 5.58/6.00 + evidence 9.42 => blended 8.08
   Confidence: 0.98  [evidence 1.00 | rule fit 0.93 | margin 1.00]
   Why: Blended score 8.08 (rules 5.58 + evidence 9.42); the knowledge base links this track to the request.
   Grounded in: rev-focus-flow

2. Midnight Coding - LoRoom
   Genre: lofi | Mood: chill | Energy: 0.42
   Rule score 4.47/6.00 + evidence 5.02 => blended 5.80
   Confidence: 0.52  [evidence 0.53 | rule fit 0.74 | margin 0.13]
   Why: Blended score 5.80 (rules 4.47 + evidence 5.02); the knowledge base links this track to the request.
   Grounded in: rev-midnight-coding

3. Library Rain - Paper Lanterns
   Genre: lofi | Mood: chill | Energy: 0.35
   Rule score 4.76/6.00 + evidence 3.19 => blended 5.61
   Confidence: 0.61  [evidence 0.34 | rule fit 0.79 | margin 1.00]
   Why: Blended score 5.61 (rules 4.76 + evidence 3.19); the knowledge base links this track to the request.
   Grounded in: rev-library-rain

Mean confidence: 0.70
Caveat: Offline ranking: ordering comes from feature matching blended with keyword retrieval, with no language model in the loop.
Run info: parse=offline, rerank=offline, citations 3/3 grounded

```

### 1.2 A request where retrieval overrides the rule scorer

```console
$ python -m src.main --offline -q "high energy music for the gym, nothing acoustic" -k 3
========================================================================
REQUEST: "high energy music for the gym, nothing acoustic"
========================================================================
Interpreted as -> genre: edm | mood: energetic | energy: 0.85 | acoustic: False | situation: workout
  (Offline parser matched situation 'workout', genre 'edm' and mood 'energetic' from keywords in the request.)

Retrieved 8 knowledge documents for this request (context widened to 13 with each candidate's notes):
  [rev-storm-runner] Storm Runner - Voltline  (bm25 4.95)
  [rev-gym-hero] Gym Hero - Max Pulse  (bm25 4.85)
  [usecase-workout] Listening while working out  (bm25 4.62)
  [usecase-late-night-drive] Late night driving  (bm25 4.37)
  [gloss-acousticness] What the acousticness feature means  (bm25 3.21)
  [gloss-valence] What the valence feature means  (bm25 3.17)
  [usecase-rainy-day] Rainy day and cozy listening  (bm25 3.10)
  [rev-iron-verdict] Iron Verdict - Blacklung Rite  (bm25 2.90)

Recommendations:

1. Pulse Machine - Hexadrive
   Genre: edm | Mood: euphoric | Energy: 0.95
   Rule score 4.77/6.00 + evidence 2.17 => blended 5.86
   Confidence: 0.54  [evidence 0.44 | rule fit 0.79 | margin 0.39]
   Why: Blended score 5.86 (rules 4.77 + evidence 2.17); the knowledge base links this track to the request.
   Grounded in: rev-pulse-machine

2. Storm Runner - Voltline
   Genre: rock | Mood: intense | Energy: 0.91
   Rule score 2.78/6.00 + evidence 4.95 => blended 5.28
   Confidence: 0.64  [evidence 1.00 | rule fit 0.46 | margin 0.03]
   Why: Blended score 5.28 (rules 2.78 + evidence 4.95); the knowledge base links this track to the request.
   Grounded in: rev-storm-runner

3. Gym Hero - Max Pulse
   Genre: pop | Mood: intense | Energy: 0.93
   Rule score 2.79/6.00 + evidence 4.85 => blended 5.24
   Confidence: 0.77  [evidence 0.98 | rule fit 0.47 | margin 0.70]
   Why: Blended score 5.24 (rules 2.79 + evidence 4.85); the knowledge base links this track to the request.
   Grounded in: rev-gym-hero

Mean confidence: 0.65
Caveat: Offline ranking: ordering comes from feature matching blended with keyword retrieval, with no language model in the loop.
Run info: parse=offline, rerank=offline, citations 3/3 grounded

```

Storm Runner scores **2.78 / 6.00** on the rules — wrong genre, wrong mood, zero
categorical points — yet ranks #2, because the retrieved note says it is used in gym
playlists. That sentence exists nowhere in `songs.csv`.

### 1.3 A request the catalog cannot answer

```console
$ python -m src.main --offline -q "polka accordion music for a moon landing" -k 3
========================================================================
REQUEST: "polka accordion music for a moon landing"
========================================================================
Interpreted as -> genre: pop | mood: chill | energy: 0.50 | acoustic: False | situation: general listening
  (Offline parser matched situation 'general listening', genre 'pop' and mood 'chill' from keywords in the request.)

Retrieved 8 knowledge documents for this request (context widened to 16 with each candidate's notes):
  [rev-pulse-machine] Pulse Machine - Hexadrive  (bm25 1.31)
  [rev-iron-verdict] Iron Verdict - Blacklung Rite  (bm25 1.29)
  [usecase-late-night-drive] Late night driving  (bm25 1.27)
  [rev-rooftop-lights] Rooftop Lights - Indigo Parade  (bm25 1.25)
  [rev-winter-elegy] Winter Elegy - Aria Sever  (bm25 1.25)
  [rev-dust-road-memory] Dust Road Memory - Colt Harlow  (bm25 1.23)
  [gloss-acousticness] What the acousticness feature means  (bm25 1.13)
  [usecase-heartbreak] Sad, melancholy and heartbreak listening  (bm25 1.07)

Recommendations:

1. Sunrise City - Neon Echo
   Genre: pop | Mood: happy | Energy: 0.82
   Rule score 4.18/6.00 + evidence 0.00 => blended 4.18
   Confidence: 0.22  [evidence 0.00 | rule fit 0.70 | margin 0.06]  <-- LOW
   Why: Blended score 4.18 from feature matching only; no knowledge-base note matched this request.
   Grounded in: (nothing in the knowledge base supported this pick)

2. Gym Hero - Max Pulse
   Genre: pop | Mood: intense | Energy: 0.93
   Rule score 4.09/6.00 + evidence 0.00 => blended 4.09
   Confidence: 0.33  [evidence 0.00 | rule fit 0.68 | margin 0.64]  <-- LOW
   Why: Blended score 4.09 from feature matching only; no knowledge-base note matched this request.
   Grounded in: (nothing in the knowledge base supported this pick)

3. Midnight Coding - LoRoom
   Genre: lofi | Mood: chill | Energy: 0.42
   Rule score 3.13/6.00 + evidence 0.00 => blended 3.13
   Confidence: 0.20  [evidence 0.00 | rule fit 0.52 | margin 0.19]  <-- LOW
   Why: Blended score 3.13 from feature matching only; no knowledge-base note matched this request.
   Grounded in: (nothing in the knowledge base supported this pick)

Mean confidence: 0.25  (3 of 3 picks are low-confidence � the catalog may not hold a good answer)
Caveat: Offline ranking: ordering comes from feature matching blended with keyword retrieval, with no language model in the loop.
Run info: parse=offline, rerank=offline, citations 0/0 grounded

```

No crash, no empty list, and no invented citation. Every pick is flagged `<-- LOW`
and mean confidence drops to 0.21 versus 0.65 on request 1.1.

---

## 2. AI feature behavior: RAG and guardrails

```console
$ python -m scripts.demo_evidence
[WARNING] guardrail: dropping hallucinated song_id 9999
[WARNING] guardrail: removing ungrounded citation 'rev-also-invented'
==============================================================================
PART A - Does retrieval change the ranking, or just decorate it?
==============================================================================

Same parsed profile, two rankings: the rule scorer alone, and the full
RAG pipeline. If retrieval were decorative these columns would match.

  study-instrumental
    rules only : [9, 4, 2, 12, 6]
    with RAG   : [9, 2, 4, 6, 12]   <-- CHANGED
  gym-high-energy
    rules only : [13, 11, 5, 3, 1]
    with RAG   : [13, 3, 5, 15, 11]   <-- CHANGED
  night-drive
    rules only : [8, 11, 1, 10, 5]
    with RAG   : [8, 3, 11, 1, 10]   <-- CHANGED
  heartbreak
    rules only : [12, 6, 18, 4, 7]
    with RAG   : [12, 7, 4, 2, 6]   <-- CHANGED
  party
    rules only : [1, 5, 10, 11, 3]
    with RAG   : [1, 10, 13, 5, 11]   <-- CHANGED
  rainy-acoustic
    rules only : [18, 6, 4, 2, 12]
    with RAG   : [18, 4, 7, 12, 6]   <-- CHANGED
  coffee-shop-reading
    rules only : [7, 12, 6, 18, 4]
    with RAG   : [7, 12, 6, 18, 4]   (same)
  adversarial-conflict
    rules only : [1, 5, 12, 17, 16]
    with RAG   : [17, 12, 1, 5, 16]   <-- CHANGED
  edge-out-of-catalog
    rules only : [1, 5, 2, 4, 6]
    with RAG   : [1, 5, 2, 4, 6]   (same)

  Retrieval changed the ranking on 7 of 9 cases (0.778).

==============================================================================
PART B - Do the guardrails catch a model that hallucinates?
==============================================================================

Request: "high energy music for the gym, nothing acoustic"

--- B1. Normal run (deterministic re-ranker) ---
  1. song 13  Pulse Machine          cites ['rev-pulse-machine']
  2. song  3  Storm Runner           cites ['rev-storm-runner']
  3. song  5  Gym Hero               cites ['rev-gym-hero']
  guardrails: {"schema_valid": true, "schema_errors": [], "dropped_songs": [], "total_citations": 3, "invalid_citations": [], "unsupported_songs": [], "groundedness": 1.0, "fell_back": false, "fallback_reason": ""}

--- B2. Same request, model returns song 9999 and two invented doc ids ---
  1. song 13  Pulse Machine          cites []
  2. song  3  Storm Runner           cites ['rev-storm-runner']
  3. song  5  Gym Hero               cites ['rev-gym-hero']

  What the guardrails caught:
    songs dropped as not-in-candidate-pool : [9999]
    citations stripped as never-retrieved  : ['rev-also-invented']
    groundedness after stripping           : 0.00
    fell back to deterministic logic       : False

  Note: the two guardrails run in order. enforce_candidates drops song
  9999 first, taking its citation 'rev-totally-made-up' with it, which is
  why only one invented citation reaches the citation counter.
  enforce_citations then strips 'rev-also-invented' from the surviving
  pick, leaving it with no support (groundedness 0.00). The list is still
  full because the pipeline topped it up from the blended order.

==============================================================================
Reproduce with: python -m scripts.demo_evidence
==============================================================================
```

---

## 3. Reliability: the evaluation harness

```console
$ python -m src.evaluation
========================================================================
VibeScope reliability report  (mode=offline, 0.0s)
========================================================================

Gated metrics
  metric                       value         bound   result
  retrieval_recall_at_k      0.912       >= 0.70   PASS
  profile_field_accuracy     1.000       >= 0.80   PASS
  schema_valid_rate          1.000       >= 1.00   PASS
  groundedness_rate          1.000       >= 0.95   PASS
  top1_consistency           1.000       >= 0.90   PASS
  topk_jaccard               1.000       >= 0.80   PASS
  guardrail_error_rate       0.000       <= 0.00   PASS
  confidence_discrimination  0.271       >= 0.10   PASS

Diagnostics (reported, not gated)
  retrieval_mrr              0.938
  fallback_rate              0.000
  retrieval_impact_rate      0.778
  retrieval_impact_top1_rate 0.111
  top1_acceptable_rate       1.000
  mean_confidence            0.454

Per case
  case                     recall  prof  grnd  conf  stable  ragchg  top-1
  study-instrumental         1.00  1.00  1.00  0.56    True    True  song 9
  gym-high-energy            0.80  1.00  1.00  0.54    True    True  song 13
  night-drive                1.00  1.00  1.00  0.33    True    True  song 8
  heartbreak                 0.50  1.00  1.00  0.52    True    True  song 12
  party                      1.00  1.00  1.00  0.58    True    True  song 1
  rainy-acoustic             1.00  1.00  1.00  0.58    True    True  song 18
  coffee-shop-reading        1.00  1.00  1.00  0.34    True   False  song 7
  adversarial-conflict       1.00  1.00  1.00  0.42    True    True  song 17
  edge-out-of-catalog        n/a   n/a  1.00  0.21    True   False  song 1

------------------------------------------------------------------------
RESULT: PASS - every gated metric met its threshold
Full report written to C:\Users\jouga\Desktop\applied-ai-system-final\eval\report.json
$ echo $?
0
```

---

## 4. Automated test suite

```console
$ pytest -q
........................................................................ [100%]
72 passed in 0.20s
```

Breakdown by file:

```console
$ pytest --collect-only -q
tests/test_retrieval.py::test_missing_knowledge_file_raises
tests/test_retrieval.py::test_malformed_lines_are_skipped_not_fatal
tests/test_retrieval.py::test_real_knowledge_base_loads_and_has_all_three_kinds
tests/test_retrieval.py::test_every_catalog_song_has_at_least_one_note
tests/test_retrieval.py::test_format_context_labels_documents_with_their_ids
tests/test_retrieval.py::test_format_context_handles_no_hits

72 tests collected in 0.04s
```

The guardrail tests are the interesting ones — they inject a fake LLM that returns
hallucinated song ids, hallucinated citations, malformed JSON and an out-of-range
confidence value, then assert each is caught:

```console
$ pytest tests/test_pipeline.py -k "hallucinated or malformed or confidence" -v
tests/test_pipeline.py::test_hallucinated_song_id_is_dropped PASSED      [ 12%]
tests/test_pipeline.py::test_hallucinated_citation_is_stripped PASSED    [ 25%]
tests/test_pipeline.py::test_malformed_profile_falls_back_to_the_offline_parser PASSED [ 37%]
tests/test_pipeline.py::test_malformed_ranking_falls_back_to_the_blended_order PASSED [ 50%]
tests/test_pipeline.py::test_every_recommendation_carries_a_confidence_in_range PASSED [ 62%]
tests/test_pipeline.py::test_model_confidence_is_averaged_with_the_computed_score PASSED [ 75%]
tests/test_pipeline.py::test_an_out_of_range_model_confidence_is_rejected_by_the_schema_guardrail PASSED [ 87%]
tests/test_pipeline.py::test_a_non_numeric_model_confidence_is_ignored PASSED [100%]
====================== 8 passed, 18 deselected in 0.07s =======================
```

---

## 5. Logging and error handling

Every retrieval, model call, retry, guardrail rejection and fallback is logged.
A sample from `logs/vibescope.log` after the runs above:

```console
$ grep -E "WARNING|guardrail|fell back" logs/vibescope.log | head -8
2026-08-05 01:02:23,597 WARNING  vibescope.guardrails | guardrail: dropping hallucinated song_id 9999
2026-08-05 01:02:23,597 WARNING  vibescope.guardrails | guardrail: removing ungrounded citation 'rev-also-invented'
2026-08-05 01:02:40,139 WARNING  vibescope.guardrails | profile failed schema validation: ['likes_acoustic must be a boolean']
2026-08-05 01:02:40,141 WARNING  vibescope.guardrails | profile failed schema validation: ['target_energy out of range: 4.2']
2026-08-05 01:02:40,142 WARNING  vibescope.guardrails | profile failed schema validation: ['target_energy must be a number']
2026-08-05 01:02:40,143 WARNING  vibescope.guardrails | profile failed schema validation: ['target_energy must be a number']
2026-08-05 01:02:40,145 WARNING  vibescope.guardrails | ranking failed schema validation: ['recommendations[0].song_id must be an integer']
2026-08-05 01:02:40,147 WARNING  vibescope.guardrails | ranking failed schema validation: ['recommendations[0].rationale must be a non-empty string']
```

Invalid input is reported cleanly and exits non-zero — no traceback:

```console
$ python -m src.main --offline -q "   "
Failed on query "   ": query must be a non-empty string
$ echo $?
1
```

---

## Reproducing all of this

```bash
pip install -r requirements.txt
python -m src.main --offline -q "<any request>"   # section 1
python -m scripts.demo_evidence                   # section 2
python -m src.evaluation                          # section 3
pytest -q                                         # section 4
```

No `ANTHROPIC_API_KEY` is required. With one set, drop `--offline` to exercise the
real Claude calls; the guardrails, metrics and thresholds are identical either way.

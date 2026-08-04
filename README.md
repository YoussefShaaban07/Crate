# 🎵 VibeScope 2.0

**A retrieval-augmented music recommender that explains itself — and a test harness that proves it works.**

Ask for music in plain English. VibeScope searches a knowledge base before answering, ranks
songs using what it finds, and cites the specific documents behind every recommendation.
An eight-metric evaluation harness gates every change, and the whole system runs with or
without an API key.

```bash
pip install -r requirements.txt
python -m src.main -q "I need something calm and instrumental for a long study session"
```

| | |
|---|---|
| **Language** | Python 3.10+ |
| **AI** | Claude (`claude-opus-5`) via the Anthropic Messages API, with structured outputs |
| **Advanced features** | Retrieval-Augmented Generation · Reliability & testing system |
| **Dependencies** | `anthropic`, `pytest` — that's all |
| **Tests** | 72 passing, no API key required |
| **Eval** | 8 gated metrics, all passing (`python -m src.evaluation`) |

---

## The original project (Modules 1–3)

**Music Recommender Simulation** — my Modules 1–3 project, a rule-based content recommender
over an 18-song CSV. Its goal was to represent songs and a listener's taste as data, design
a transparent scoring rule that turns those numbers into a ranking, and evaluate what the
rule got right and wrong.

Version 1.0 could do exactly one thing: take a hard-coded profile of four fields
(`favorite_genre`, `favorite_mood`, `target_energy`, `likes_acoustic`), award each song up
to 6 points across four weighted rules, sort, and print the top 5 with a breakdown of the
points earned. There was no AI in it — no natural language, no retrieval, no model, no
guardrails, and no automated evaluation. That scorer still ships here as the candidate
generation stage, and `python -m src.main --profiles` runs it standalone as a baseline.

### What version 2.0 adds

| | v1.0 | v2.0 |
|---|---|---|
| Input | Hard-coded 4-field profile | Any plain-English request |
| Knowledge | 18 rows × 9 numeric columns | + 31-document knowledge base, retrieved at query time |
| Ranking | 4 scoring rules | 4 rules **+ a retrieval evidence term that reorders results** |
| Explanation | Points earned | Points earned **+ cited source documents** |
| Failure handling | None | Schema validation, 4 guardrails, retries, deterministic fallback |
| Uncertainty | None | Per-pick confidence score, low-confidence flagging |
| Evaluation | Reading the output | 9 golden cases × 3 repeats, 8 gated metrics, 72 tests, 22 human-reviewed inputs |

---

## Quick start

### 1. Install

```bash
git clone <this-repo>
cd applied-ai-system-final

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

### 2. (Optional) Enable the Claude calls

```bash
setx ANTHROPIC_API_KEY "sk-ant-..."      # Windows, then reopen the terminal
export ANTHROPIC_API_KEY="sk-ant-..."    # macOS / Linux
```

**No key? Everything still runs.** The two Claude calls fall back to deterministic
stand-in logic; retrieval, scoring, guardrails, logging, the tests and the evaluation
harness are unaffected. The CLI tells you which mode it's in, and so does every result.

### 3. Run

```bash
python -m src.main                                # demo: five example requests
python -m src.main -q "gym music, nothing acoustic"
python -m src.main -q "sad quiet music" --json    # machine-readable output
python -m src.main --offline -v                   # force offline mode, verbose logging
python -m src.main --profiles                     # v1.0 rule-only baseline
python -m src.main --help
```

### 4. Verify

```bash
pytest                          # 72 tests
python -m src.evaluation        # 8 gated metrics; exit 0 = pass, 1 = regression
python -m scripts.demo_evidence # proves RAG changes the ranking and guardrails fire
```

Expected: `72 passed` and `RESULT: PASS - every gated metric met its threshold`.

---

## Architecture overview

Full diagram: **[`diagrams/architecture.mmd`](diagrams/architecture.mmd)** — Mermaid source;
paste it into [mermaid.live](https://mermaid.live) to render.

The diagram is organised as `INPUT → PROCESS → OUTPUT`, with a fourth layer for
verification. Colour encodes role: **purple** for the two Claude calls, **green** for the
retrieval components, **orange** for automated checks on AI output, **blue** for testing,
**yellow** for the points where a human is in the loop. Solid arrows are data; dashed arrows
are checks, logs and feedback.

Six stages, orchestrated by `src/pipeline.py`:

```
free-text request
   │
   ├─[1] RETRIEVE   BM25 over 31 knowledge documents → top 8
   │
   ├─[2] PARSE      Claude: request + retrieved docs → structured taste profile
   │                └─ schema check → falls back to a keyword parser
   │
   ├─[3] SCORE      rule scoring over all 18 songs → candidate pool of 8
   │
   ├─[4] BLEND      blended = rule_score + 2.5 × normalised evidence score
   │                ← retrieval reorders the shortlist here
   │
   ├─[5] RE-RANK    Claude: candidates + retrieved docs → final order,
   │                rationale, and the doc_ids that justify each pick
   │                └─ schema check → falls back to the blended order
   │
   └─[6] GUARDRAILS drop invented songs · strip invented citations · top up to k
                    → grounded recommendations
```

**Retrieval enters the answer at two points, which is what keeps it from being decorative.**
At stage 2 the retrieved situation and glossary documents are in the prompt when Claude
turns "calm" into a specific `target_energy`. At stage 4 each candidate earns an evidence
score — BM25 of the request against that song's own notes, normalised and multiplied by
2.5 — which is larger than a genre match is worth, so it routinely changes who wins.

**Verification happens at two cadences.** Per request, the guardrails check every model
output automatically. Per change, `pytest` and the evaluation harness must pass before
anything ships, and a human reviews any failed gate.

### The knowledge base

[`data/knowledge/notes.jsonl`](data/knowledge/notes.jsonl) — 31 documents, three kinds:

| Kind | Count | Contents | Why the recommender needs it |
|---|---|---|---|
| `review` | 18 | One critic-style note per song | Captures what the CSV can't: whether a track has vocals, what people actually use it for |
| `usecase` | 7 | Studying, gym, night driving, sleep, party, heartbreak, rainy day | Turns a *situation* into feature targets |
| `glossary` | 6 | What energy / valence / acousticness / danceability / tempo mean | Grounds the model's numeric choices in definitions rather than guesses |

### Scoring

| Rule | Points | Formula |
|---|---|---|
| Genre match | +2.0 | `+2.0 if song.genre == favorite_genre` |
| Mood match | +1.0 | `+1.0 if song.mood == favorite_mood` |
| Energy closeness | 0 → +2.0 | `2.0 × (1 − abs(song.energy − target_energy))` |
| Acoustic preference | 0 → +1.0 | `acousticness`, or `1 − acousticness` if the listener wants electronic |
| **Evidence (RAG)** | **0 → +2.5** | **`2.5 × (bm25(request, song's notes) / peak)`** |

All weights live in [`src/config.py`](src/config.py), so an experiment is a one-line change.

---

## Sample interactions

### 1. A clear request — the system agrees with itself

**Input:** `"I need something for a long late-night study session, instrumental and calm"`

```
Interpreted as -> genre: lofi | mood: focused | energy: 0.30 | acoustic: True | situation: studying

Retrieved 8 knowledge documents (context widened to 11 with each candidate's notes):
  [rev-focus-flow]     Focus Flow - LoRoom                (bm25 9.33)
  [rev-midnight-coding] Midnight Coding - LoRoom          (bm25 4.97)
  [usecase-study-focus] Listening while studying, reading or doing focused work
  ...

1. Focus Flow - LoRoom       rule 5.58 + evidence 9.33 => blended 8.08   [rev-focus-flow]
2. Midnight Coding - LoRoom  rule 4.47 + evidence 4.97 => blended 5.80   [rev-midnight-coding]
3. Library Rain              rule 4.76 + evidence 3.16 => blended 5.61   [rev-library-rain]

citations 4/4 grounded
```

Rules and retrieval agree on the winner. Retrieval's contribution here is ordering
positions 2–5 — *Library Rain* has a higher rule score than *Midnight Coding* but drops
below it, because the knowledge base ties "coding" and "study" more strongly to the latter.

### 2. Retrieval overriding the rules — the clearest proof RAG is load-bearing

**Input:** `"high energy music for the gym, nothing acoustic"`

```
1. Pulse Machine - Hexadrive   rule 4.77 + evidence 2.17 => blended 5.86   [rev-pulse-machine]
2. Storm Runner - Voltline     rule 2.78 + evidence 4.95 => blended 5.28   [rev-storm-runner]
3. Gym Hero - Max Pulse        rule 2.79 + evidence 4.85 => blended 5.24   [rev-gym-hero]

citations 3/3 grounded
```

**Storm Runner scores 2.78 out of 6.00 on the rules.** It's a rock song and the parsed
profile asked for edm, so it earns *zero* genre points and *zero* mood points — the rule
scorer alone ranks it below several songs that merely happen to be loud. It reaches #2 only
because the knowledge base says it is *"frequently used in gym, workout and running
playlists."* That sentence exists nowhere in the CSV. It was retrieved at query time, and it
changed the output.

The harness measures this across the whole golden set: **`retrieval_impact_rate = 0.778`** —
retrieval changes the returned ranking on 7 of 9 cases. If that number were 0, the RAG layer
would be cosmetic, and the report would say so.

### 3. An unanswerable request — graceful degradation

**Input:** `"polka accordion music for a moon landing"`

```
Interpreted as -> genre: pop | mood: chill | energy: 0.50 | acoustic: False | situation: general listening

Retrieved 8 documents, top bm25 score 1.31 (all weak matches)

1. Sunrise City - Neon Echo   rule 4.18 + evidence 0.00 => blended 4.18
   Why: Blended score 4.18 from feature matching only; no knowledge-base note matched
        this request.
   Grounded in: (nothing in the knowledge base supported this pick)
```

No polka, no accordion, no moon in the catalog or the knowledge base. The system doesn't
crash, doesn't return an empty list, and — importantly — **doesn't invent a citation**. It
falls back to numeric feature matching and says plainly that nothing supported the pick.

### 4. Guardrails catching a model that hallucinates

Reproducible by command — `scripts/demo_evidence.py` runs the same request against a fake
model that returns a song id that doesn't exist and two citations that were never retrieved:

```console
$ python -m scripts.demo_evidence

--- B1. Normal run (deterministic re-ranker) ---
  1. song 13  Pulse Machine          cites ['rev-pulse-machine']
  2. song  3  Storm Runner           cites ['rev-storm-runner']
  3. song  5  Gym Hero               cites ['rev-gym-hero']
  guardrails: {"schema_valid": true, "dropped_songs": [], "invalid_citations": [],
               "groundedness": 1.0, "fell_back": false}

--- B2. Same request, model returns song 9999 and two invented doc ids ---
  1. song 13  Pulse Machine          cites []
  2. song  3  Storm Runner           cites ['rev-storm-runner']
  3. song  5  Gym Hero               cites ['rev-gym-hero']

  What the guardrails caught:
    songs dropped as not-in-candidate-pool : [9999]
    citations stripped as never-retrieved  : ['rev-also-invented']
    groundedness after stripping           : 0.00
    fell back to deterministic logic       : False
```

Song 9999 never reaches the user. The two guardrails run in order, so
`enforce_candidates` drops 9999 along with its fake citation, then
`enforce_citations` strips `rev-also-invented` from the surviving pick — leaving it
honestly marked as unsupported rather than falsely cited. The list is still full because
the pipeline topped it up from the blended order.

> **Full verbatim output** for every command in this README — including the complete
> evaluation report, test run and log samples — is in
> **[`assets/execution_evidence.md`](assets/execution_evidence.md)**.

---

## Design decisions and trade-offs

**BM25 instead of vector embeddings.** *Trade-off: semantic recall for reproducibility.*
Embeddings would match "music that sounds like rain on a window" to the right notes; BM25
only matches words. But BM25 is ~120 lines of dependency-free Python that produces identical
scores on every machine and every run — which is the precondition for measuring retrieval
recall and run-to-run consistency at all. I chose the option I could put a number on. The
cost is real and shows up in the metrics: the heartbreak case scores 0.50 recall because
"breakup" doesn't appear in every relevant note.

**A rule-based first pass instead of asking the model to rank all 18 songs.** *Trade-off:
some ranking quality for auditability and cost.* Shortlisting with transparent arithmetic
means every recommendation decomposes into a rule score and an evidence score you can
inspect, and the model reasons over 8 candidates instead of 18. It also makes the ablation
possible — the harness can compare "rules alone" against "rules + retrieval" precisely
because the rule stage still exists independently.

**A deterministic offline fallback for both model calls.** *Trade-off: a second code path to
maintain, for a system that always runs.* This was the single most useful decision. It means
a reviewer with no API key gets a working system and a passing test suite; it means the
evaluation harness can isolate the retrieval and guardrail layers from model variance; and
it means an API outage degrades the product instead of breaking it. The `LLMClient` picks a
strategy and the pipeline never knows which one ran.

**Guardrails that degrade rather than raise.** No model output reaches the user unvalidated,
but no model failure produces a stack trace either. Malformed JSON, invented songs, invented
citations, rate limits, timeouts and refusals all route to deterministic behaviour, and the
`Run info:` line on every result reports what actually happened.

**Evidence weight of 2.5.** Chosen deliberately so retrieval can outweigh a genre match
(+2.0). Any lower and RAG would only ever break ties. The `retrieval_impact_rate` diagnostic
exists specifically so this number can be justified with evidence instead of taste.

**Hand-written schema validators instead of `jsonschema`.** The shapes are small and it keeps
installation to two packages and one command. Not the right call on a larger schema surface.

---

## Testing summary

> **In short:** 72 automated tests pass and all 8 gated metrics clear their thresholds
> (retrieval recall 0.912, groundedness 1.000, top-1 consistency 1.000). Manual review of 22
> inputs gave 21 Pass / 1 Partial — the partial is a genuine gap, since danceability is in
> the data but is not a scored feature. Recall started at 0.685 and improved to 0.912 after
> the harness exposed a missing stemmer; citation groundedness went from 0.75 to 1.000 after
> a guardrail was found to be deleting valid citations. The system is measurably least
> confident exactly where it should be — 0.21 on an out-of-catalog request versus 0.65 on a
> clear one.

Four independent reliability mechanisms, at three cadences.

| Mechanism | Cadence | Where |
|---|---|---|
| **Automated tests** — 72, no API key needed | Per change | `tests/` |
| **Confidence scoring** — every pick rates itself 0–1 from three observable signals | Per request | `src/pipeline.py` |
| **Logging and error handling** — every call, retry, rejection and fallback recorded | Per request | `src/logging_setup.py` → `logs/vibescope.log` |
| **Human evaluation** — 22 inputs reviewed against stated criteria | Per release | [`eval/human_review.md`](eval/human_review.md) |

### Confidence scoring

Each recommendation carries a 0–1 confidence built from three things the pipeline can
actually check, rather than the model's opinion of itself:

```
confidence = 0.50 × evidence   (how strongly the knowledge base backs this song)
           + 0.30 × rule_fit   (how well the numeric features match the profile)
           + 0.20 × margin     (how far clear of the next candidate it is)
```

When Claude supplies its own `confidence`, the two are **averaged** — so a model that flags
its own uncertainty is heard, but a model that is confidently wrong gets pulled back toward
the observable evidence. Picks below 0.35 are labelled `<-- LOW` in the output.

This is gated, not decorative. `confidence_discrimination` measures mean confidence on
answerable cases minus the unanswerable one; it must stay above 0.10, which stops confidence
from being a constant that "works" by never varying. It currently sits at **+0.271**:

```
"calm instrumental music for studying"      mean confidence 0.65
"polka accordion music for a moon landing"  mean confidence 0.21, all 5 picks flagged LOW
```

### Test layers

**Unit and integration tests — 72, no API key needed.**

| File | Covers |
|---|---|
| `tests/test_recommender.py` | Scoring maths, tie-breaking determinism, catalog loading, malformed CSV rows |
| `tests/test_retrieval.py` | Stemming, ranking, determinism, per-song evidence, corrupt knowledge base |
| `tests/test_guardrails.py` | Every schema rule, invented songs, invented citations, groundedness maths |
| `tests/test_pipeline.py` | End to end, plus confidence calibration, using a **fake LLM that deliberately hallucinates** so the guardrails can be proven to fire |

**Evaluation harness — 9 golden cases × 3 repeats.** Seven ordinary requests, one
deliberately contradictory ("intense but also melancholy and slow"), one asking for a genre
and mood the catalog doesn't contain.

```
Gated metrics
  metric                       value         bound   result
  retrieval_recall_at_k       0.912       >= 0.70   PASS
  profile_field_accuracy      1.000       >= 0.80   PASS
  schema_valid_rate           1.000       >= 1.00   PASS
  groundedness_rate           1.000       >= 0.95   PASS
  top1_consistency            1.000       >= 0.90   PASS
  topk_jaccard                1.000       >= 0.80   PASS
  guardrail_error_rate        0.000       <= 0.00   PASS
  confidence_discrimination   0.271       >= 0.10   PASS

Diagnostics (reported, not gated)
  retrieval_mrr               0.938
  retrieval_impact_rate       0.778
  retrieval_impact_top1_rate  0.111
  top1_acceptable_rate        1.000
  mean_confidence             0.454

RESULT: PASS - every gated metric met its threshold
```

### What worked

Guardrails and fallbacks held under every failure I could inject: hallucinated song IDs,
hallucinated citations, malformed JSON, empty rankings, short rankings, and an out-of-range
confidence value. Determinism held too — `top1_consistency` and `topk_jaccard` are both
1.000, which is what makes the other metrics trustworthy. Confidence behaves the way it is
supposed to: lowest on the request the catalog cannot answer.

### What didn't

**Two bugs the harness caught that reading the output never would have.**

1. **BM25 had no stemming.** The first eval run scored `retrieval_recall_at_k = 0.685`,
   below the 0.70 gate. `"study"` and `"studying"` were different index terms, so a query
   about studying never matched the document titled *"Listening while studying…"*. The
   output before the fix looked entirely reasonable. Adding a suffix stemmer took recall to
   **0.912**.
2. **A guardrail was over-firing.** Citation validation checked against the query-level
   top-k, but a candidate song's own note is frequently outside that set — so *correct*
   citations were being deleted as hallucinations, dropping groundedness to 0.75. Fixed by
   widening the model's context to the query hits plus every candidate's notes. A guardrail
   that over-fires is as much a bug as one that under-fires, and only the metric revealed it.

**One golden label was wrong, not the system.** `rev-paper-boats` was listed as a correct
retrieval for the heartbreak query, but Paper Boats is dreamy folk, not heartbreak music. I
removed the label and left the remaining 0.50 recall on that case visible in the report,
rather than editing the corpus until the number went green.

### What I learned

Every output I inspected by hand looked correct, and two of them were produced by broken
code. The useful question turned out not to be "does this look right" but "what number would
go down if this were wrong" — and if no such number exists, the feature isn't finished. That
also changed how I read a failing gate: when recall came in at 0.685, the fast fix was to
lower the threshold to 0.65. Fixing the stemmer took longer and was the only honest option.

I also learned to report the metric that doesn't flatter the system.
`retrieval_impact_top1_rate` is **0.111** — RAG changes the *winner* on only one case in
nine. That looks weak next to the 0.778 headline, but it's the correct result: when a request
is unambiguous, the rules and the knowledge base agree, and retrieval reorders positions 2–5.
Retrieval overturns the #1 exactly where the rules are weakest. Both numbers are in the
report.

---

## Project structure

```
├── src/
│   ├── main.py            CLI entry point
│   ├── pipeline.py        6-stage orchestration — the main application logic
│   ├── retrieval.py       BM25 retriever + stemmer (the "R" in RAG)
│   ├── llm.py             Claude calls + deterministic offline stand-ins (the "G")
│   ├── recommender.py     Rule-based scorer — candidate generation
│   ├── guardrails.py      Schema validation + citation/candidate enforcement
│   ├── evaluation.py      Reliability harness — 8 gated metrics
│   ├── config.py          Every tunable weight, threshold and model setting
│   └── logging_setup.py   Structured logging to logs/vibescope.log
├── data/
│   ├── songs.csv          18-song catalog
│   └── knowledge/notes.jsonl   31-document knowledge base
├── eval/
│   ├── cases.json         9 golden cases with gold documents and profiles
│   └── human_review.md    22 manually reviewed inputs, in a parseable table
├── scripts/
│   └── demo_evidence.py   Reproducible RAG-ablation and guardrail demonstration
├── assets/
│   └── execution_evidence.md   Verbatim output of every command in this README
├── tests/                 72 tests
├── diagrams/architecture.mmd   System diagram (Mermaid)
├── model_card.md          Intended use, limitations, bias, evaluation, reflection
└── ai_interactions.md     How AI tooling was used, and what I corrected
```

---

## Limitations

The short version: 18 songs; keyword retrieval that misses meaning; a knowledge base I wrote
myself, which means a `Grounded in: [doc-id]` tag can make my own judgement look like sourced
evidence; and a genre weighting that can still outrank a severe energy mismatch.

The full treatment — including who the system serves worse and why explainability doesn't
imply fairness — is in **[`model_card.md`](model_card.md) § 6**.

---

## Further reading

- **[`model_card.md`](model_card.md)** — intended use, data, strengths, limitations and bias,
  evaluation results, future work, and the responsible-AI reflection.
- **[`ai_interactions.md`](ai_interactions.md)** — how AI tooling was used to build this,
  which suggestions I took, and which I had to correct.
- **[`diagrams/architecture.mmd`](diagrams/architecture.mmd)** — the full system diagram.
- **[`assets/execution_evidence.md`](assets/execution_evidence.md)** — verbatim, reproducible
  output for every command referenced here.
- **[`eval/human_review.md`](eval/human_review.md)** — 22 manually reviewed inputs with
  criteria and results.

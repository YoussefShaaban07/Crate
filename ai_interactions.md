# AI Interactions Log

Documenting how AI tooling was used to build Crate 2.0, and what I had to verify or
correct myself.

---

## Agentic Workflow (SF8)

> Using an AI agent (Claude Code) to make multi-step changes autonomously.

**What task did I give the agent?**

Take Crate 1.0 — a rule-based scorer over an 18-song CSV with no AI in it at all — and
turn it into a system meeting the advanced-feature requirement. I chose **RAG + a
reliability/testing system**, with the constraint that the whole thing must still run and
pass tests for someone with no Anthropic API key.

**Prompts used (the load-bearing ones):**

- *"The feature must be fully integrated into the main application logic — if I add RAG, the
  system must actively use the retrieved data to formulate its response, not print it
  alongside a standard answer."* This shaped the design more than anything else: it ruled out
  bolting a retrieval printout onto the existing scorer, and led to retrieval entering at two
  points (the parse prompt, and a blended evidence term in the ranking score).
- *"Use the Claude API, but add a deterministic offline fallback so the project runs
  reproducibly without a key."*
- *"Build the evaluation harness so it fails the build, not so it prints a nice number."*
- *"The harness is failing recall — fix the root cause, don't lower the threshold."*

**What did the agent generate or change?**

New: `src/retrieval.py` (BM25 + stemmer), `src/llm.py` (Claude calls + offline stand-ins),
`src/pipeline.py` (the six-stage orchestration), `src/guardrails.py`, `src/evaluation.py`,
`src/config.py`, `src/logging_setup.py`, `data/knowledge/notes.jsonl` (31 documents),
`eval/cases.json` (9 golden cases), `tests/test_retrieval.py`, `tests/test_guardrails.py`,
`tests/test_pipeline.py`, `diagrams/architecture.mmd`.

Rewritten: `src/recommender.py` (implemented the `Recommender` stubs, moved weights into
config, added catalog error handling), `src/main.py` (natural-language CLI),
`tests/test_recommender.py`, `README.md`, `model_card.md`, `requirements.txt` (dropped the
unused `pandas` and `streamlit`).

Commands run: `python -m src.main`, `python -m src.evaluation`, `pytest` — iteratively,
with the eval report driving the next change.

**What did I verify or fix manually?**

The interesting part. Three things the agent got wrong or left to me:

1. **The first eval run failed** — `retrieval_recall_at_k = 0.685` against a 0.70 gate. The
   easy fix was lowering the threshold. The real cause was that BM25 had no stemming, so
   `"study"` and `"studying"` were unrelated index terms and a query about studying never
   matched the document titled *"Listening while studying..."*. Fixed the stemmer; recall
   went to 0.912.
2. **A guardrail was over-firing.** Citation validation checked against the query-level
   top-k, but a candidate song's own note is often outside that set, so *valid* citations
   were being stripped as hallucinations. I only noticed because `groundedness` was 0.75 on
   a case where the output looked perfect. Fixed by widening the model's context to the query
   hits plus every candidate's notes.
3. **I corrected one golden label rather than tuning around it.** `rev-paper-boats` was
   listed as a correct retrieval for the heartbreak query, but Paper Boats is dreamy folk,
   not heartbreak music — a bad label, not a retrieval failure. I removed it and left the
   remaining 0.50 recall on that case visible in the report instead of editing the corpus
   until the number went green.

I also made the judgement calls the agent shouldn't: the scoring weights (genre over mood —
which the model card now argues against), BM25 over embeddings (reproducibility over
semantic recall), and `EVIDENCE_WEIGHT = 2.5`, chosen so evidence can outweigh a genre match.

---

## Design Pattern (SF10)

**Which design pattern did I use?**

Two, working together: **Pipeline** (a fixed sequence of stages, each with one job) and
**Strategy** (each model call has two interchangeable implementations — Claude or a
deterministic stand-in).

**How did AI help me brainstorm or implement it?**

I described the offline-fallback requirement and the agent proposed the split I ended up
with: `LLMClient` owns the transport and picks a strategy, while the pipeline never knows
which one ran. My initial sketch had `if api_key:` branches scattered through the pipeline;
the agent pointed out that this would make the evaluation harness impossible to write,
because there'd be no single place to force offline mode. That argument convinced me — the
`force_offline` flag exists precisely so the harness can measure retrieval and guardrails in
isolation.

**How does the pattern appear in the final code?**

- **Pipeline**: `RecommendationPipeline.run()` in `src/pipeline.py` calls `retrieve` →
  `build_profile` → `build_candidates` → `expand_context` → `finalise` in order. Each stage
  is independently testable, and `tests/test_pipeline.py` exercises them separately.
- **Strategy**: `LLMClient.parse_profile` / `.rerank` return an `LLMResult` carrying a `mode`
  field. Online they call Claude; offline (or after a failure, or when the schema check
  rejects the output) they call `offline_parse_profile` / `offline_rerank`. The caller
  handles one return type either way, and `result.modes` reports which strategy actually ran.
- The fake LLM in `tests/test_pipeline.py` is a third strategy, used to prove the guardrails
  catch hallucinated songs and citations without needing a model that actually hallucinates.

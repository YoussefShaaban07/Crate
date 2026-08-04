"""
Central configuration for VibeScope.

Every tunable lives here so experiments are a one-line change and so the
evaluation harness can report exactly which settings produced a report.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --- Paths -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SONGS_CSV = PROJECT_ROOT / "data" / "songs.csv"
KNOWLEDGE_JSONL = PROJECT_ROOT / "data" / "knowledge" / "notes.jsonl"
EVAL_CASES = PROJECT_ROOT / "eval" / "cases.json"
EVAL_REPORT = PROJECT_ROOT / "eval" / "report.json"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "vibescope.log"

# --- Scoring weights (the rule-based first-pass "Algorithm Recipe") --------
# score_song() in recommender.py reads these directly.

GENRE_WEIGHT = 2.0     # points for an exact genre match
MOOD_WEIGHT = 1.0      # points for an exact mood match
ENERGY_WEIGHT = 2.0    # max points for a perfect energy match (scaled by closeness)
ACOUSTIC_WEIGHT = 1.0  # max points for the acoustic preference

MAX_RULE_SCORE = GENRE_WEIGHT + MOOD_WEIGHT + ENERGY_WEIGHT + ACOUSTIC_WEIGHT

# --- Retrieval -------------------------------------------------------------

# Tuned with `python -m src.evaluation`: at k=6 recall@k was 0.69, below the
# 0.70 gate, because gold sets run up to 5 documents. k=8 clears it comfortably.
RETRIEVAL_TOP_K = 8        # documents pulled into the LLM context per query
BM25_K1 = 1.5
BM25_B = 0.75

# How much retrieved evidence is allowed to move a song in the final ranking.
# This is what makes retrieval load-bearing rather than decorative: a song with
# strong supporting evidence can overtake a song the rule scorer preferred.
EVIDENCE_WEIGHT = 2.5

# Candidates handed from the rule scorer to the evidence-aware re-ranker.
CANDIDATE_POOL = 8

# --- Confidence scoring ----------------------------------------------------
# Each recommendation carries a 0-1 confidence built from three observable
# signals, so "the system is unsure" is a computed fact rather than a vibe.
# Weights sum to 1.0.

CONF_EVIDENCE_WEIGHT = 0.50   # how strongly the knowledge base backs this song
CONF_RULE_FIT_WEIGHT = 0.30   # how well the numeric features match the profile
CONF_MARGIN_WEIGHT = 0.20     # how far clear of the next candidate it is

# A margin of this many blended points counts as full separation.
CONF_MARGIN_SATURATION = 1.5

# Below this, the CLI labels the pick low-confidence and says why.
LOW_CONFIDENCE_THRESHOLD = 0.35

# --- LLM -------------------------------------------------------------------

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
PARSE_MAX_TOKENS = 4000
RERANK_MAX_TOKENS = 8000
PARSE_EFFORT = "low"       # profile extraction is a simple structured task
RERANK_EFFORT = "medium"   # re-ranking with evidence benefits from more thought
LLM_MAX_RETRIES = 2        # on top of the SDK's own retries
LLM_TIMEOUT_SECONDS = 60.0


def api_key_present() -> bool:
    """True when an Anthropic credential looks available in the environment."""
    return bool(
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )


# --- Evaluation thresholds -------------------------------------------------
# The reliability harness fails (non-zero exit) if a metric drops below these.

@dataclass(frozen=True)
class Thresholds:
    retrieval_recall_at_k: float = 0.70
    profile_field_accuracy: float = 0.80
    schema_valid_rate: float = 1.00
    groundedness_rate: float = 0.95
    top1_consistency: float = 0.90
    topk_jaccard: float = 0.80
    guardrail_error_rate: float = 0.00  # unhandled failures are never acceptable
    # Confidence must be lower on the out-of-catalog case than on answerable
    # ones. Without this gate, confidence could be a constant and still "work".
    confidence_discrimination: float = 0.10


THRESHOLDS = Thresholds()

CONSISTENCY_REPEATS = 3  # pipeline runs per case when measuring stability


@dataclass(frozen=True)
class RunSettings:
    """Snapshot of the settings used for a run, embedded in eval reports."""

    model: str = MODEL
    genre_weight: float = GENRE_WEIGHT
    mood_weight: float = MOOD_WEIGHT
    energy_weight: float = ENERGY_WEIGHT
    acoustic_weight: float = ACOUSTIC_WEIGHT
    evidence_weight: float = EVIDENCE_WEIGHT
    retrieval_top_k: int = RETRIEVAL_TOP_K
    candidate_pool: int = CANDIDATE_POOL
    thresholds: dict = field(default_factory=lambda: asdict(THRESHOLDS))

    def to_dict(self) -> dict:
        return asdict(self)

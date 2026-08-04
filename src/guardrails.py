"""
Guardrails applied to every model output before it reaches the user.

Three things can go wrong with a generated recommendation, and each has a check:

  1. **Malformed output** - the JSON does not match the schema we asked for.
     `validate_profile` / `validate_ranking` reject it and the caller falls back.
  2. **Hallucinated songs** - the model ranks a track that was not a candidate.
     `enforce_candidates` drops it.
  3. **Hallucinated citations** - the model cites a doc_id that was never
     retrieved. `enforce_citations` strips it and records a violation.

Nothing here raises on bad model output. A guardrail failure degrades the
result and is counted; only programmer errors raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from src.logging_setup import get_logger

log = get_logger("guardrails")

VALID_MOODS = {
    "happy", "chill", "intense", "relaxed", "moody", "focused", "energetic",
    "melancholy", "euphoric", "nostalgic", "aggressive", "uplifting",
    "romantic", "dreamy",
}


@dataclass
class GuardrailReport:
    """Everything the guardrails noticed during one pipeline run."""

    schema_valid: bool = True
    schema_errors: List[str] = field(default_factory=list)
    dropped_songs: List[Any] = field(default_factory=list)
    total_citations: int = 0
    invalid_citations: List[str] = field(default_factory=list)
    unsupported_songs: List[str] = field(default_factory=list)
    fell_back: bool = False
    fallback_reason: str = ""

    @property
    def groundedness(self) -> float:
        """Fraction of citations that resolved to an actually-retrieved document."""
        if self.total_citations == 0:
            return 1.0
        good = self.total_citations - len(self.invalid_citations)
        return good / self.total_citations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_valid": self.schema_valid,
            "schema_errors": self.schema_errors,
            "dropped_songs": self.dropped_songs,
            "total_citations": self.total_citations,
            "invalid_citations": self.invalid_citations,
            "unsupported_songs": self.unsupported_songs,
            "groundedness": round(self.groundedness, 4),
            "fell_back": self.fell_back,
            "fallback_reason": self.fallback_reason,
        }


# --- schema validation -----------------------------------------------------
# Hand-written rather than jsonschema-based: the shapes are small, and keeping
# the dependency list to `anthropic` + `pytest` makes setup one command.


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_profile(payload: Any) -> Tuple[bool, List[str]]:
    """Check a parsed taste profile against the schema we requested."""
    errors: List[str] = []

    if not isinstance(payload, dict):
        return False, ["profile is not a JSON object"]

    genre = payload.get("favorite_genre")
    if not isinstance(genre, str) or not genre.strip():
        errors.append("favorite_genre must be a non-empty string")

    mood = payload.get("favorite_mood")
    if not isinstance(mood, str) or not mood.strip():
        errors.append("favorite_mood must be a non-empty string")

    energy = payload.get("target_energy")
    if not _is_number(energy):
        errors.append("target_energy must be a number")
    elif not 0.0 <= float(energy) <= 1.0:
        errors.append(f"target_energy out of range: {energy}")

    if not isinstance(payload.get("likes_acoustic"), bool):
        errors.append("likes_acoustic must be a boolean")

    situation = payload.get("situation")
    if situation is not None and not isinstance(situation, str):
        errors.append("situation must be a string when present")

    if errors:
        log.warning("profile failed schema validation: %s", errors)
    return not errors, errors


def validate_ranking(payload: Any) -> Tuple[bool, List[str]]:
    """Check a re-ranking response against the schema we requested."""
    errors: List[str] = []

    if not isinstance(payload, dict):
        return False, ["ranking is not a JSON object"]

    items = payload.get("recommendations")
    if not isinstance(items, list) or not items:
        return False, ["recommendations must be a non-empty list"]

    for i, item in enumerate(items):
        where = f"recommendations[{i}]"
        if not isinstance(item, dict):
            errors.append(f"{where} is not an object")
            continue
        if not isinstance(item.get("song_id"), int) or isinstance(
            item.get("song_id"), bool
        ):
            errors.append(f"{where}.song_id must be an integer")
        rationale = item.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append(f"{where}.rationale must be a non-empty string")
        evidence = item.get("evidence")
        if not isinstance(evidence, list):
            errors.append(f"{where}.evidence must be a list of doc ids")
        elif not all(isinstance(e, str) for e in evidence):
            errors.append(f"{where}.evidence must contain only strings")

        # Optional: the offline re-ranker omits it and the pipeline computes
        # confidence itself. When the model does supply one it must be usable.
        if "confidence" in item and item["confidence"] is not None:
            if not _is_number(item["confidence"]):
                errors.append(f"{where}.confidence must be a number")
            elif not 0.0 <= float(item["confidence"]) <= 1.0:
                errors.append(f"{where}.confidence out of range: {item['confidence']}")

    if errors:
        log.warning("ranking failed schema validation: %s", errors[:5])
    return not errors, errors


# --- content guardrails ----------------------------------------------------


def enforce_candidates(
    items: Sequence[Dict[str, Any]],
    allowed_ids: Iterable[int],
    report: GuardrailReport,
) -> List[Dict[str, Any]]:
    """Drop any ranked song that was not in the candidate pool we supplied."""
    allowed = set(allowed_ids)
    kept: List[Dict[str, Any]] = []
    seen: set[int] = set()

    for item in items:
        song_id = item.get("song_id")
        if song_id not in allowed:
            report.dropped_songs.append(song_id)
            log.warning("guardrail: dropping hallucinated song_id %r", song_id)
            continue
        if song_id in seen:
            report.dropped_songs.append(song_id)
            log.warning("guardrail: dropping duplicate song_id %r", song_id)
            continue
        seen.add(song_id)
        kept.append(item)

    return kept


def enforce_citations(
    items: Sequence[Dict[str, Any]],
    retrieved_ids: Iterable[str],
    report: GuardrailReport,
) -> List[Dict[str, Any]]:
    """
    Strip citations that point at documents we never retrieved.

    A recommendation whose every citation was invented is flagged as
    unsupported so the CLI can label it honestly instead of implying the
    knowledge base backed it up.
    """
    available = set(retrieved_ids)
    cleaned: List[Dict[str, Any]] = []

    for item in items:
        raw = list(item.get("evidence") or [])
        report.total_citations += len(raw)

        good = [doc_id for doc_id in raw if doc_id in available]
        bad = [doc_id for doc_id in raw if doc_id not in available]

        for doc_id in bad:
            report.invalid_citations.append(doc_id)
            log.warning("guardrail: removing ungrounded citation %r", doc_id)

        updated = dict(item)
        updated["evidence"] = good
        updated["supported"] = bool(good)

        if raw and not good:
            report.unsupported_songs.append(str(item.get("song_id")))

        cleaned.append(updated)

    return cleaned

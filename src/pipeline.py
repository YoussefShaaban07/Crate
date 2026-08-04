"""
The main application logic: a retrieval-augmented recommendation pipeline.

    free-text request
        |
        v
  [1] retrieve      BM25 over the knowledge base -> top-k documents
        |
        v
  [2] parse         Claude turns the request + retrieved docs into a
        |           structured taste profile (schema-validated)
        v
  [3] score         rule-based feature scoring over the whole catalog,
        |           shortlisting a candidate pool
        v
  [4] blend         each candidate's retrieval evidence score is folded into
        |           its ranking score - retrieval can and does reorder here
        v
  [5] re-rank       Claude re-orders the shortlist using the retrieved
        |           documents and must cite the doc_ids it relied on
        v
  [6] guardrails    schema check, candidate check, citation check
        |
        v
   grounded recommendations

Retrieval is load-bearing at two separate points (steps 2 and 4-5), so the
system's output genuinely depends on what the knowledge base says — it is not
printed alongside an answer that would have been the same without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.config import (
    CANDIDATE_POOL,
    CONF_EVIDENCE_WEIGHT,
    CONF_MARGIN_SATURATION,
    CONF_MARGIN_WEIGHT,
    CONF_RULE_FIT_WEIGHT,
    EVIDENCE_WEIGHT,
    LOW_CONFIDENCE_THRESHOLD,
    MAX_RULE_SCORE,
    RETRIEVAL_TOP_K,
)
from src.guardrails import (
    GuardrailReport,
    enforce_candidates,
    enforce_citations,
    validate_profile,
    validate_ranking,
)
from src.llm import LLMClient, offline_parse_profile, offline_rerank
from src.logging_setup import get_logger
from src.recommender import (
    catalog_vocabulary,
    load_songs,
    recommend_songs,
    score_song,
)
from src.retrieval import BM25Retriever, Retrieved, build_default_retriever, format_context

log = get_logger("pipeline")


@dataclass
class Recommendation:
    """One final, guardrail-checked recommendation."""

    rank: int
    song: Dict[str, Any]
    rule_score: float
    evidence_score: float
    blended_score: float
    rationale: str
    evidence: List[str]
    supported: bool
    confidence: float = 0.0
    computed_confidence: float = 0.0
    model_confidence: Optional[float] = None
    confidence_parts: Dict[str, float] = field(default_factory=dict)
    rule_reasons: List[str] = field(default_factory=list)

    @property
    def low_confidence(self) -> bool:
        return self.confidence < LOW_CONFIDENCE_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "song_id": self.song["id"],
            "title": self.song["title"],
            "artist": self.song["artist"],
            "genre": self.song["genre"],
            "mood": self.song["mood"],
            "rule_score": round(self.rule_score, 3),
            "evidence_score": round(self.evidence_score, 3),
            "blended_score": round(self.blended_score, 3),
            "rationale": self.rationale,
            "evidence": self.evidence,
            "supported": self.supported,
            "confidence": round(self.confidence, 3),
            "computed_confidence": round(self.computed_confidence, 3),
            "model_confidence": self.model_confidence,
            "confidence_parts": {k: round(v, 3) for k, v in self.confidence_parts.items()},
            "low_confidence": self.low_confidence,
            "rule_reasons": self.rule_reasons,
        }


@dataclass
class PipelineResult:
    """Everything one run produced, including the audit trail."""

    query: str
    profile: Dict[str, Any]
    retrieved: List[Retrieved]       # query-level search results, ranked
    context: List[Retrieved]         # retrieved + per-candidate notes
    candidates: List[Dict[str, Any]]
    recommendations: List[Recommendation]
    caveat: str
    modes: Dict[str, str]
    guardrails: GuardrailReport

    @property
    def retrieved_ids(self) -> List[str]:
        """Query-level hits, in rank order. This is what retrieval is scored on."""
        return [hit.doc_id for hit in self.retrieved]

    @property
    def context_ids(self) -> List[str]:
        """Every document the model was allowed to cite."""
        return [hit.doc_id for hit in self.context]

    @property
    def top_song_ids(self) -> List[int]:
        return [rec.song["id"] for rec in self.recommendations]

    @property
    def mean_confidence(self) -> float:
        """Average confidence across the returned list, 0.0 if it is empty."""
        if not self.recommendations:
            return 0.0
        return sum(r.confidence for r in self.recommendations) / len(self.recommendations)

    @property
    def low_confidence_count(self) -> int:
        return sum(1 for r in self.recommendations if r.low_confidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "profile": self.profile,
            "retrieved": [
                {"doc_id": h.doc_id, "score": round(h.score, 3), "title": h.document.title}
                for h in self.retrieved
            ],
            "context_ids": self.context_ids,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "mean_confidence": round(self.mean_confidence, 3),
            "low_confidence_count": self.low_confidence_count,
            "caveat": self.caveat,
            "modes": self.modes,
            "guardrails": self.guardrails.to_dict(),
        }


class RecommendationPipeline:
    """Wires retrieval, the rule scorer, the model and the guardrails together."""

    def __init__(
        self,
        songs: Optional[List[Dict[str, Any]]] = None,
        retriever: Optional[BM25Retriever] = None,
        llm: Optional[LLMClient] = None,
        force_offline: bool = False,
    ) -> None:
        self.songs = songs if songs is not None else load_songs()
        self.retriever = retriever or build_default_retriever()
        self.llm = llm or LLMClient(force_offline=force_offline)
        self.genres, self.moods = catalog_vocabulary(self.songs)
        log.info(
            "pipeline ready: %d songs, %d knowledge docs, llm mode=%s",
            len(self.songs),
            len(self.retriever.documents),
            self.llm.mode,
        )

    # -- steps --------------------------------------------------------------

    def retrieve(self, query: str, k: int = RETRIEVAL_TOP_K) -> List[Retrieved]:
        """[1] Pull the documents that will ground both model calls."""
        return self.retriever.search(query, k=k)

    def build_profile(
        self,
        query: str,
        hits: Sequence[Retrieved],
        report: GuardrailReport,
    ) -> tuple[Dict[str, Any], str]:
        """[2] Free-text request -> validated structured profile."""
        result = self.llm.parse_profile(
            query, format_context(hits), self.genres, self.moods
        )
        payload, mode = result.payload, result.mode

        ok, errors = validate_profile(payload)
        if not ok:
            report.schema_valid = False
            report.schema_errors.extend(errors)
            report.fell_back = True
            report.fallback_reason = f"profile schema invalid: {errors[0]}"
            log.warning("profile from %s failed validation, using offline parser", mode)
            payload = offline_parse_profile(query, self.genres, self.moods)
            mode = "offline"

        # Clamp rather than reject: an out-of-vocabulary genre simply scores
        # zero on the genre rule, which is the correct graceful degradation.
        payload["target_energy"] = min(1.0, max(0.0, float(payload["target_energy"])))
        if payload["favorite_genre"] not in self.genres:
            log.info(
                "profile genre %r is not in the catalog - no genre points will fire",
                payload["favorite_genre"],
            )
        return payload, mode

    def build_candidates(
        self,
        query: str,
        profile: Dict[str, Any],
        pool: int = CANDIDATE_POOL,
    ) -> List[Dict[str, Any]]:
        """[3]+[4] Rule shortlist, then blend in retrieval evidence."""
        ranked = recommend_songs(profile, self.songs, k=pool)

        raw: List[Dict[str, Any]] = []
        for song, rule_score, _ in ranked:
            evidence_score, evidence_docs = self.retriever.evidence_score(query, song["id"])
            _, reasons = score_song(profile, song)
            raw.append(
                {
                    "song": song,
                    "rule_score": rule_score,
                    "rule_reasons": reasons,
                    "evidence_score": evidence_score,
                    "evidence_docs": evidence_docs,
                }
            )

        # Normalise evidence to 0..1 across the pool so EVIDENCE_WEIGHT means
        # "at most this many points of movement" regardless of BM25's scale.
        peak = max((c["evidence_score"] for c in raw), default=0.0)
        for cand in raw:
            normalised = (cand["evidence_score"] / peak) if peak > 0 else 0.0
            cand["evidence_normalised"] = normalised
            cand["blended_score"] = cand["rule_score"] + EVIDENCE_WEIGHT * normalised

        raw.sort(key=lambda c: (-c["blended_score"], c["song"]["id"]))

        moved = [
            c["song"]["title"]
            for c in raw
            if c["evidence_normalised"] > 0
        ]
        log.debug("evidence supported %d/%d candidates: %s", len(moved), len(raw), moved)
        return raw

    @staticmethod
    def compute_confidence(
        candidate: Dict[str, Any],
        next_best_score: Optional[float],
    ) -> tuple[float, Dict[str, float]]:
        """
        How sure the system is about one pick, from three observable signals.

        This is deliberately *not* the model's opinion of itself — a model asked
        "how confident are you" tends to say "very". These three numbers are
        things the pipeline can check:

          evidence  - how strongly the knowledge base backs this song (0-1)
          rule_fit  - how well the numeric features match the profile (0-1)
          margin    - how far clear it is of the next candidate (0-1)

        When the model *does* supply its own confidence, the caller averages the
        two so a model that flags its own uncertainty is listened to, but a model
        that is confidently wrong is pulled back toward the observable evidence.
        """
        evidence = float(candidate.get("evidence_normalised", 0.0))
        rule_fit = max(0.0, min(1.0, candidate["rule_score"] / MAX_RULE_SCORE))

        if next_best_score is None:
            margin = 1.0  # nothing below it to be confused with
        else:
            gap = max(0.0, candidate["blended_score"] - next_best_score)
            margin = min(1.0, gap / CONF_MARGIN_SATURATION)

        parts = {"evidence": evidence, "rule_fit": rule_fit, "margin": margin}
        score = (
            CONF_EVIDENCE_WEIGHT * evidence
            + CONF_RULE_FIT_WEIGHT * rule_fit
            + CONF_MARGIN_WEIGHT * margin
        )
        return max(0.0, min(1.0, score)), parts

    def expand_context(
        self,
        hits: Sequence[Retrieved],
        candidates: Sequence[Dict[str, Any]],
    ) -> List[Retrieved]:
        """
        Widen the retrieved set to cover every shortlisted song.

        The query-level search answers "what does the knowledge base say about
        this request"; it does not guarantee a note for each candidate. Adding
        each candidate's own notes means the re-ranker can judge every option
        on evidence, and means a citation to a candidate's note is legitimate
        rather than something the citation guardrail has to strip.
        """
        context: List[Retrieved] = list(hits)
        present = {hit.doc_id for hit in context}

        for cand in candidates:
            for doc in self.retriever.notes_for_song(cand["song"]["id"]):
                if doc.doc_id in present:
                    continue
                present.add(doc.doc_id)
                context.append(Retrieved(document=doc, score=0.0))

        log.debug(
            "context expanded from %d to %d documents", len(hits), len(context)
        )
        return context

    def finalise(
        self,
        query: str,
        profile: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        hits: Sequence[Retrieved],
        k: int,
        report: GuardrailReport,
    ) -> tuple[List[Recommendation], str, str]:
        """[5]+[6] Model re-rank, then guardrails, then assemble the output."""
        result = self.llm.rerank(query, profile, candidates, format_context(hits), k)
        payload, mode = result.payload, result.mode

        ok, errors = validate_ranking(payload)
        if not ok:
            report.schema_valid = False
            report.schema_errors.extend(errors)
            report.fell_back = True
            report.fallback_reason = f"ranking schema invalid: {errors[0]}"
            log.warning("ranking from %s failed validation, using offline re-ranker", mode)
            payload = offline_rerank(candidates, k)
            mode = "offline"

        allowed_ids = [c["song"]["id"] for c in candidates]
        items = enforce_candidates(payload["recommendations"], allowed_ids, report)
        items = enforce_citations(items, [h.doc_id for h in hits], report)

        if not items:
            report.fell_back = True
            report.fallback_reason = "every ranked song was rejected by the guardrails"
            log.warning("all ranked songs rejected - falling back to blended order")
            payload = offline_rerank(candidates, k)
            items = enforce_citations(payload["recommendations"], [h.doc_id for h in hits], report)
            mode = "offline"

        by_id = {c["song"]["id"]: c for c in candidates}

        # Top up from the blended order if the model returned fewer than k.
        if len(items) < k:
            chosen = {item["song_id"] for item in items}
            for cand in candidates:
                if len(items) >= k:
                    break
                sid = cand["song"]["id"]
                if sid in chosen:
                    continue
                items.append(
                    {
                        "song_id": sid,
                        "rationale": "Added from the blended ranking to fill the list.",
                        "evidence": list(cand["evidence_docs"]),
                        "supported": bool(cand["evidence_docs"]),
                    }
                )
                chosen.add(sid)

        # Candidates in blended order, so each pick's margin is measured against
        # the best thing it actually beat.
        ordered = sorted(candidates, key=lambda c: -c["blended_score"])
        position = {c["song"]["id"]: i for i, c in enumerate(ordered)}

        recommendations: List[Recommendation] = []
        for rank, item in enumerate(items[:k], start=1):
            cand = by_id[item["song_id"]]

            idx = position[item["song_id"]]
            next_best = (
                ordered[idx + 1]["blended_score"] if idx + 1 < len(ordered) else None
            )
            computed, parts = self.compute_confidence(cand, next_best)

            raw_model_conf = item.get("confidence")
            model_conf = (
                max(0.0, min(1.0, float(raw_model_conf)))
                if isinstance(raw_model_conf, (int, float))
                and not isinstance(raw_model_conf, bool)
                else None
            )
            # Average the two when the model gave one, so self-reported certainty
            # is heard but cannot override what the evidence actually shows.
            confidence = computed if model_conf is None else (computed + model_conf) / 2

            recommendations.append(
                Recommendation(
                    rank=rank,
                    song=cand["song"],
                    rule_score=cand["rule_score"],
                    evidence_score=cand["evidence_score"],
                    blended_score=cand["blended_score"],
                    rationale=item["rationale"].strip(),
                    evidence=item.get("evidence", []),
                    supported=bool(item.get("supported", bool(item.get("evidence")))),
                    confidence=confidence,
                    computed_confidence=computed,
                    model_confidence=model_conf,
                    confidence_parts=parts,
                    rule_reasons=cand["rule_reasons"],
                )
            )

        low = [r.song["title"] for r in recommendations if r.low_confidence]
        if low:
            log.info("%d/%d picks below the confidence floor: %s",
                     len(low), len(recommendations), low)

        caveat = str(payload.get("caveat", "")).strip() or (
            "This catalog holds only a handful of songs per genre, so the tail of "
            "any top-5 list is padding rather than a real match."
        )
        return recommendations, caveat, mode

    # -- entry point --------------------------------------------------------

    def run(self, query: str, k: int = 5) -> PipelineResult:
        """Run the full pipeline for one free-text listening request."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        log.info("=== query: %r (k=%d) ===", query, k)
        report = GuardrailReport()

        hits = self.retrieve(query)
        if not hits:
            log.warning("no documents retrieved - recommendations will be feature-only")

        profile, parse_mode = self.build_profile(query, hits, report)
        log.info("profile: %s", {k_: v for k_, v in profile.items() if k_ != "reasoning"})

        candidates = self.build_candidates(query, profile)
        context = self.expand_context(hits, candidates)
        recommendations, caveat, rerank_mode = self.finalise(
            query, profile, candidates, context, k, report
        )

        log.info(
            "returned %d recommendations (parse=%s, rerank=%s, groundedness=%.2f)",
            len(recommendations),
            parse_mode,
            rerank_mode,
            report.groundedness,
        )

        return PipelineResult(
            query=query,
            profile=profile,
            retrieved=list(hits),
            context=context,
            candidates=candidates,
            recommendations=recommendations,
            caveat=caveat,
            modes={"parse": parse_mode, "rerank": rerank_mode},
            guardrails=report,
        )


# --- presentation ----------------------------------------------------------


def format_result(result: PipelineResult, show_retrieval: bool = True) -> str:
    """Render a PipelineResult as the block the CLI prints."""
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append(f'REQUEST: "{result.query}"')
    lines.append("=" * 72)

    profile = result.profile
    lines.append(
        f"Interpreted as -> genre: {profile['favorite_genre']} | "
        f"mood: {profile['favorite_mood']} | "
        f"energy: {profile['target_energy']:.2f} | "
        f"acoustic: {profile['likes_acoustic']} | "
        f"situation: {profile.get('situation', 'n/a')}"
    )
    if profile.get("reasoning"):
        lines.append(f"  ({profile['reasoning']})")

    if show_retrieval:
        lines.append("")
        lines.append(
            f"Retrieved {len(result.retrieved)} knowledge documents for this request "
            f"(context widened to {len(result.context)} with each candidate's notes):"
        )
        for hit in result.retrieved:
            lines.append(f"  [{hit.doc_id}] {hit.document.title}  (bm25 {hit.score:.2f})")

    lines.append("")
    lines.append("Recommendations:")
    for rec in result.recommendations:
        lines.append("")
        lines.append(f"{rec.rank}. {rec.song['title']} - {rec.song['artist']}")
        lines.append(
            f"   Genre: {rec.song['genre']} | Mood: {rec.song['mood']} | "
            f"Energy: {rec.song['energy']:.2f}"
        )
        lines.append(
            f"   Rule score {rec.rule_score:.2f}/{MAX_RULE_SCORE:.2f} "
            f"+ evidence {rec.evidence_score:.2f} "
            f"=> blended {rec.blended_score:.2f}"
        )
        parts = rec.confidence_parts
        stated = (
            "" if rec.model_confidence is None
            else f", model said {rec.model_confidence:.2f}"
        )
        lines.append(
            f"   Confidence: {rec.confidence:.2f}"
            f"  [evidence {parts.get('evidence', 0):.2f} |"
            f" rule fit {parts.get('rule_fit', 0):.2f} |"
            f" margin {parts.get('margin', 0):.2f}{stated}]"
            + ("  <-- LOW" if rec.low_confidence else "")
        )
        lines.append(f"   Why: {rec.rationale}")
        if rec.evidence:
            lines.append(f"   Grounded in: {', '.join(rec.evidence)}")
        else:
            lines.append("   Grounded in: (nothing in the knowledge base supported this pick)")

    lines.append("")
    lines.append(
        f"Mean confidence: {result.mean_confidence:.2f}"
        + (
            f"  ({result.low_confidence_count} of {len(result.recommendations)} "
            f"picks are low-confidence — the catalog may not hold a good answer)"
            if result.low_confidence_count
            else ""
        )
    )
    lines.append(f"Caveat: {result.caveat}")

    guard = result.guardrails
    lines.append(
        f"Run info: parse={result.modes['parse']}, rerank={result.modes['rerank']}, "
        f"citations {guard.total_citations - len(guard.invalid_citations)}/"
        f"{guard.total_citations} grounded"
    )
    if guard.invalid_citations:
        lines.append(f"  guardrail removed invented citations: {guard.invalid_citations}")
    if guard.dropped_songs:
        lines.append(f"  guardrail removed invalid song ids: {guard.dropped_songs}")
    if guard.fell_back:
        lines.append(f"  fell back to deterministic logic: {guard.fallback_reason}")

    return "\n".join(lines)


def explain_song(result: PipelineResult, song_id: int) -> str:
    """Full rule breakdown for one recommended song, for debugging a ranking."""
    for rec in result.recommendations:
        if rec.song["id"] == song_id:
            return f"{rec.song['title']}: " + "; ".join(rec.rule_reasons)
    return f"song {song_id} was not in the returned recommendations"

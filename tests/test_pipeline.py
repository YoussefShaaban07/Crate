"""
End-to-end tests for the retrieval-augmented pipeline.

These use a fake LLM so the whole system — retrieval, blending, guardrails,
fallback — is exercised deterministically and without an API key.
"""

import pytest

from src.llm import LLMResult, offline_parse_profile, offline_rerank
from src.pipeline import RecommendationPipeline, format_result


class FakeLLM:
    """A scriptable stand-in for LLMClient."""

    def __init__(self, profile=None, ranking=None):
        self._profile = profile
        self._ranking = ranking
        self.mode = "claude"
        self.online = True
        self.parse_calls = 0
        self.rerank_calls = 0

    def parse_profile(self, query, context, genres, moods):
        self.parse_calls += 1
        if self._profile is None:
            return LLMResult(offline_parse_profile(query, genres, moods), "offline")
        return LLMResult(self._profile, "claude")

    def rerank(self, query, profile, candidates, context, k):
        self.rerank_calls += 1
        if self._ranking is None:
            return LLMResult(offline_rerank(candidates, k), "offline")
        return LLMResult(self._ranking, "claude")


@pytest.fixture(scope="module")
def offline_pipeline():
    return RecommendationPipeline(force_offline=True)


def pipeline_with(llm):
    return RecommendationPipeline(llm=llm)


# -- happy path -------------------------------------------------------------


def test_run_returns_the_requested_number_of_recommendations(offline_pipeline):
    result = offline_pipeline.run("calm instrumental music for studying", k=5)
    assert len(result.recommendations) == 5
    assert [r.rank for r in result.recommendations] == [1, 2, 3, 4, 5]


def test_study_request_is_parsed_into_a_low_energy_profile(offline_pipeline):
    result = offline_pipeline.run("calm instrumental music for studying")
    assert result.profile["target_energy"] < 0.5
    assert result.profile["situation"] == "studying"


def test_gym_request_is_parsed_into_a_high_energy_non_acoustic_profile(offline_pipeline):
    result = offline_pipeline.run("high energy music for the gym, nothing acoustic")
    assert result.profile["target_energy"] > 0.7
    assert result.profile["likes_acoustic"] is False


def test_retrieval_actually_runs_and_feeds_the_context(offline_pipeline):
    result = offline_pipeline.run("music for the gym")
    assert result.retrieved, "nothing was retrieved"
    # Context must cover the query hits plus every candidate's own notes.
    assert set(result.retrieved_ids) <= set(result.context_ids)
    assert len(result.context) >= len(result.retrieved)


def test_every_citation_points_at_a_document_in_context(offline_pipeline):
    result = offline_pipeline.run("quiet acoustic folk for a rainy afternoon")
    available = set(result.context_ids)
    for rec in result.recommendations:
        assert set(rec.evidence) <= available
    assert result.guardrails.groundedness == 1.0


def test_results_are_reproducible(offline_pipeline):
    a = offline_pipeline.run("something moody for driving at night")
    b = offline_pipeline.run("something moody for driving at night")
    assert a.top_song_ids == b.top_song_ids


# -- retrieval is load-bearing ---------------------------------------------


def test_retrieval_changes_the_ranking_versus_rules_alone(offline_pipeline):
    """
    The rubric's bar: retrieved data must change the output, not decorate it.

    Same profile, two rankings — one from the rule scorer alone, one from the
    full pipeline. They must differ, and the difference must come from the
    evidence term.
    """
    from src.recommender import recommend_songs

    query = "calm instrumental music for a long study session"
    result = offline_pipeline.run(query, k=5)
    rule_only = [
        row[0]["id"] for row in recommend_songs(result.profile, offline_pipeline.songs, k=5)
    ]

    assert result.top_song_ids != rule_only
    assert any(c["evidence_score"] > 0 for c in result.candidates)


def test_blended_score_exceeds_rule_score_when_evidence_supports_a_song(offline_pipeline):
    result = offline_pipeline.run("music for the gym")
    supported = [c for c in result.candidates if c["evidence_normalised"] > 0]
    assert supported
    for cand in supported:
        assert cand["blended_score"] > cand["rule_score"]


# -- guardrails, end to end -------------------------------------------------


def test_hallucinated_song_id_is_dropped():
    ranking = {
        "recommendations": [
            {"song_id": 4, "rationale": "real song", "evidence": []},
            {"song_id": 4242, "rationale": "invented song", "evidence": []},
        ],
        "caveat": "test",
    }
    pipeline = pipeline_with(FakeLLM(ranking=ranking))
    result = pipeline.run("calm music for studying", k=3)

    assert 4242 not in result.top_song_ids
    assert 4242 in result.guardrails.dropped_songs


def test_hallucinated_citation_is_stripped():
    ranking = {
        "recommendations": [
            {
                "song_id": 4,
                "rationale": "cites something that does not exist",
                "evidence": ["rev-does-not-exist"],
            }
        ],
        "caveat": "test",
    }
    pipeline = pipeline_with(FakeLLM(ranking=ranking))
    result = pipeline.run("calm music for studying", k=3)

    assert "rev-does-not-exist" in result.guardrails.invalid_citations
    for rec in result.recommendations:
        assert "rev-does-not-exist" not in rec.evidence
    assert result.guardrails.groundedness < 1.0


def test_malformed_profile_falls_back_to_the_offline_parser():
    pipeline = pipeline_with(FakeLLM(profile={"favorite_genre": 12, "nope": True}))
    result = pipeline.run("high energy music for the gym", k=3)

    assert result.guardrails.schema_valid is False
    assert result.guardrails.fell_back is True
    assert result.modes["parse"] == "offline"
    # It still produced a usable answer.
    assert len(result.recommendations) == 3
    assert result.profile["target_energy"] > 0.7


def test_malformed_ranking_falls_back_to_the_blended_order():
    pipeline = pipeline_with(FakeLLM(ranking={"recommendations": "not a list"}))
    result = pipeline.run("calm music for studying", k=3)

    assert result.guardrails.schema_valid is False
    assert result.modes["rerank"] == "offline"
    assert len(result.recommendations) == 3


def test_a_ranking_of_only_invalid_songs_falls_back_rather_than_returning_nothing():
    ranking = {
        "recommendations": [{"song_id": 9999, "rationale": "nope", "evidence": []}],
        "caveat": "test",
    }
    pipeline = pipeline_with(FakeLLM(ranking=ranking))
    result = pipeline.run("calm music for studying", k=3)

    assert len(result.recommendations) == 3
    assert result.guardrails.fell_back is True


def test_short_ranking_is_topped_up_to_k():
    ranking = {
        "recommendations": [{"song_id": 4, "rationale": "only one", "evidence": []}],
        "caveat": "test",
    }
    pipeline = pipeline_with(FakeLLM(ranking=ranking))
    result = pipeline.run("calm music for studying", k=5)
    assert len(result.recommendations) == 5
    assert len({r.song["id"] for r in result.recommendations}) == 5


# -- confidence scoring -----------------------------------------------------


def test_every_recommendation_carries_a_confidence_in_range(offline_pipeline):
    result = offline_pipeline.run("calm music for studying", k=5)
    for rec in result.recommendations:
        assert 0.0 <= rec.confidence <= 1.0
        assert set(rec.confidence_parts) == {"evidence", "rule_fit", "margin"}


def test_the_system_is_less_confident_about_an_unanswerable_request(offline_pipeline):
    """The point of confidence: it must drop when the catalog has no answer."""
    answerable = offline_pipeline.run("calm instrumental music for studying", k=5)
    unanswerable = offline_pipeline.run("polka accordion music for a moon landing", k=5)

    assert unanswerable.mean_confidence < answerable.mean_confidence
    assert unanswerable.low_confidence_count > 0


def test_a_pick_with_no_evidence_scores_lower_than_one_with_evidence(offline_pipeline):
    result = offline_pipeline.run("music for the gym", k=8)
    with_ev = [r for r in result.recommendations if r.confidence_parts["evidence"] > 0]
    without = [r for r in result.recommendations if r.confidence_parts["evidence"] == 0]
    if with_ev and without:
        assert max(r.confidence for r in without) < max(r.confidence for r in with_ev)


def test_model_confidence_is_averaged_with_the_computed_score():
    ranking = {
        "recommendations": [
            {"song_id": 4, "rationale": "sure", "evidence": [], "confidence": 1.0}
        ],
        "caveat": "test",
    }
    pipeline = pipeline_with(FakeLLM(ranking=ranking))
    rec = pipeline.run("calm music for studying", k=1).recommendations[0]

    assert rec.model_confidence == 1.0
    # A model claiming total certainty cannot drag the score to 1.0 on its own.
    assert rec.confidence == pytest.approx((rec.computed_confidence + 1.0) / 2)
    assert rec.confidence < 1.0


def test_an_out_of_range_model_confidence_is_rejected_by_the_schema_guardrail():
    """A confidence of 42 is a malformed response, not a very sure one."""
    ranking = {
        "recommendations": [
            {"song_id": 4, "rationale": "x", "evidence": [], "confidence": 42}
        ],
        "caveat": "test",
    }
    pipeline = pipeline_with(FakeLLM(ranking=ranking))
    result = pipeline.run("calm music for studying", k=1)

    assert result.guardrails.schema_valid is False
    assert result.modes["rerank"] == "offline"
    # The fallback still returns a usable answer scored on observable signals.
    rec = result.recommendations[0]
    assert rec.model_confidence is None
    assert 0.0 <= rec.confidence <= 1.0


def test_a_non_numeric_model_confidence_is_ignored():
    ranking = {
        "recommendations": [
            {"song_id": 4, "rationale": "x", "evidence": [], "confidence": "very"}
        ],
        "caveat": "test",
    }
    pipeline = pipeline_with(FakeLLM(ranking=ranking))
    rec = pipeline.run("calm music for studying", k=1).recommendations[0]
    assert rec.model_confidence is None
    assert rec.confidence == pytest.approx(rec.computed_confidence)


# -- edge cases -------------------------------------------------------------


def test_out_of_catalog_request_degrades_gracefully(offline_pipeline):
    result = offline_pipeline.run("polka accordion music for a moon landing", k=5)
    assert len(result.recommendations) == 5
    assert result.caveat


def test_conflicting_request_still_produces_a_ranking(offline_pipeline):
    result = offline_pipeline.run("really intense but also melancholy and slow", k=5)
    assert len(result.recommendations) == 5


def test_empty_query_is_rejected(offline_pipeline):
    with pytest.raises(ValueError):
        offline_pipeline.run("   ")


def test_k_larger_than_the_candidate_pool_is_capped(offline_pipeline):
    result = offline_pipeline.run("music for the gym", k=50)
    assert 0 < len(result.recommendations) <= len(result.candidates)


# -- presentation and serialisation ----------------------------------------


def test_format_result_mentions_grounding(offline_pipeline):
    text = format_result(offline_pipeline.run("calm music for studying", k=2))
    assert "Grounded in:" in text
    assert "Retrieved" in text


def test_result_serialises_to_json_safe_dict(offline_pipeline):
    import json

    payload = offline_pipeline.run("music for the gym", k=3).to_dict()
    json.dumps(payload)  # must not raise
    assert payload["guardrails"]["groundedness"] == 1.0
    assert len(payload["recommendations"]) == 3

"""Tests for the guardrails that sit between the model and the user."""

from src.guardrails import (
    GuardrailReport,
    enforce_candidates,
    enforce_citations,
    validate_profile,
    validate_ranking,
)


def good_profile():
    return {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.35,
        "likes_acoustic": True,
        "situation": "studying",
        "reasoning": "because",
    }


def good_ranking():
    return {
        "recommendations": [
            {"song_id": 4, "rationale": "calm and acoustic", "evidence": ["rev-library-rain"]}
        ],
        "caveat": "small catalog",
    }


# -- profile schema ---------------------------------------------------------


def test_valid_profile_passes():
    ok, errors = validate_profile(good_profile())
    assert ok and errors == []


def test_profile_missing_a_field_fails():
    payload = good_profile()
    del payload["likes_acoustic"]
    ok, errors = validate_profile(payload)
    assert not ok and any("likes_acoustic" in e for e in errors)


def test_profile_with_out_of_range_energy_fails():
    payload = good_profile()
    payload["target_energy"] = 4.2
    ok, errors = validate_profile(payload)
    assert not ok and any("out of range" in e for e in errors)


def test_profile_with_string_energy_fails():
    payload = good_profile()
    payload["target_energy"] = "low"
    assert validate_profile(payload)[0] is False


def test_boolean_is_not_accepted_as_a_number():
    payload = good_profile()
    payload["target_energy"] = True
    assert validate_profile(payload)[0] is False


def test_non_object_profile_fails():
    assert validate_profile(["not", "an", "object"])[0] is False


# -- ranking schema ---------------------------------------------------------


def test_valid_ranking_passes():
    ok, errors = validate_ranking(good_ranking())
    assert ok and errors == []


def test_empty_recommendations_fails():
    assert validate_ranking({"recommendations": [], "caveat": "x"})[0] is False


def test_ranking_with_a_non_integer_song_id_fails():
    payload = good_ranking()
    payload["recommendations"][0]["song_id"] = "four"
    assert validate_ranking(payload)[0] is False


def test_ranking_with_a_blank_rationale_fails():
    payload = good_ranking()
    payload["recommendations"][0]["rationale"] = "   "
    assert validate_ranking(payload)[0] is False


# -- hallucinated songs -----------------------------------------------------


def test_songs_outside_the_candidate_pool_are_dropped():
    report = GuardrailReport()
    items = [{"song_id": 4}, {"song_id": 999}]
    kept = enforce_candidates(items, allowed_ids=[4, 9], report=report)
    assert [i["song_id"] for i in kept] == [4]
    assert report.dropped_songs == [999]


def test_duplicate_songs_are_dropped():
    report = GuardrailReport()
    kept = enforce_candidates(
        [{"song_id": 4}, {"song_id": 4}], allowed_ids=[4], report=report
    )
    assert len(kept) == 1
    assert report.dropped_songs == [4]


# -- hallucinated citations -------------------------------------------------


def test_invented_citations_are_stripped_and_counted():
    report = GuardrailReport()
    items = [{"song_id": 4, "evidence": ["real-doc", "made-up-doc"]}]
    cleaned = enforce_citations(items, retrieved_ids=["real-doc"], report=report)

    assert cleaned[0]["evidence"] == ["real-doc"]
    assert report.invalid_citations == ["made-up-doc"]
    assert report.total_citations == 2
    assert report.groundedness == 0.5


def test_a_pick_with_only_invented_citations_is_marked_unsupported():
    report = GuardrailReport()
    items = [{"song_id": 4, "evidence": ["fake"]}]
    cleaned = enforce_citations(items, retrieved_ids=["real"], report=report)

    assert cleaned[0]["supported"] is False
    assert report.unsupported_songs == ["4"]


def test_groundedness_is_one_when_nothing_was_cited():
    assert GuardrailReport().groundedness == 1.0


def test_report_serialises_for_the_eval_json():
    report = GuardrailReport()
    enforce_citations([{"song_id": 1, "evidence": ["x"]}], ["x"], report)
    payload = report.to_dict()
    assert payload["groundedness"] == 1.0
    assert payload["schema_valid"] is True

"""Tests for the rule-based scorer that produces the candidate shortlist."""

import pytest

from src.config import MAX_RULE_SCORE
from src.recommender import (
    CatalogError,
    Recommender,
    Song,
    UserProfile,
    catalog_vocabulary,
    load_songs,
    recommend_songs,
    score_song,
)


def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


POP_USER = UserProfile(
    favorite_genre="pop",
    favorite_mood="happy",
    target_energy=0.8,
    likes_acoustic=False,
)


# -- starter tests ----------------------------------------------------------


def test_recommend_returns_songs_sorted_by_score():
    rec = make_small_recommender()
    results = rec.recommend(POP_USER, k=2)

    assert len(results) == 2
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    rec = make_small_recommender()
    explanation = rec.explain_recommendation(POP_USER, rec.songs[0])

    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# -- scoring ----------------------------------------------------------------


def test_a_perfect_match_scores_the_maximum():
    song = {
        "genre": "pop", "mood": "happy", "energy": 0.8, "acousticness": 0.0,
    }
    prefs = {
        "favorite_genre": "pop", "favorite_mood": "happy",
        "target_energy": 0.8, "likes_acoustic": False,
    }
    score, _ = score_song(prefs, song)
    assert score == pytest.approx(MAX_RULE_SCORE)


def test_genre_match_is_worth_more_than_mood_match():
    prefs = {
        "favorite_genre": "pop", "favorite_mood": "happy",
        "target_energy": 0.5, "likes_acoustic": False,
    }
    genre_only = {"genre": "pop", "mood": "chill", "energy": 0.5, "acousticness": 0.5}
    mood_only = {"genre": "rock", "mood": "happy", "energy": 0.5, "acousticness": 0.5}

    assert score_song(prefs, genre_only)[0] > score_song(prefs, mood_only)[0]


def test_energy_scoring_rewards_closeness_not_loudness():
    prefs = {
        "favorite_genre": "x", "favorite_mood": "y",
        "target_energy": 0.3, "likes_acoustic": False,
    }
    close = {"genre": "a", "mood": "b", "energy": 0.32, "acousticness": 0.5}
    far = {"genre": "a", "mood": "b", "energy": 0.95, "acousticness": 0.5}

    assert score_song(prefs, close)[0] > score_song(prefs, far)[0]


def test_acoustic_preference_flips_direction():
    acoustic_song = {"genre": "a", "mood": "b", "energy": 0.5, "acousticness": 0.9}
    likes = {"favorite_genre": "x", "favorite_mood": "y",
             "target_energy": 0.5, "likes_acoustic": True}
    dislikes = {**likes, "likes_acoustic": False}

    assert score_song(likes, acoustic_song)[0] > score_song(dislikes, acoustic_song)[0]


def test_reasons_are_returned_for_every_rule_that_fired():
    prefs = {"favorite_genre": "pop", "favorite_mood": "happy",
             "target_energy": 0.8, "likes_acoustic": False}
    song = {"genre": "pop", "mood": "happy", "energy": 0.8, "acousticness": 0.1}
    _, reasons = score_song(prefs, song)

    assert any("genre match" in r for r in reasons)
    assert any("mood match" in r for r in reasons)
    assert any("energy" in r for r in reasons)


# -- ranking ----------------------------------------------------------------


def test_recommend_songs_is_deterministic_on_ties():
    songs = [
        {"id": 2, "title": "B", "artist": "x", "genre": "a", "mood": "b",
         "energy": 0.5, "acousticness": 0.5},
        {"id": 1, "title": "A", "artist": "x", "genre": "a", "mood": "b",
         "energy": 0.5, "acousticness": 0.5},
    ]
    prefs = {"favorite_genre": "z", "favorite_mood": "z",
             "target_energy": 0.5, "likes_acoustic": False}
    ranked = recommend_songs(prefs, songs, k=2)
    assert [row[0]["id"] for row in ranked] == [1, 2]  # lower id wins the tie


def test_unknown_genre_and_mood_degrade_gracefully():
    songs = load_songs()
    prefs = {"favorite_genre": "polka", "favorite_mood": "sad",
             "target_energy": 0.5, "likes_acoustic": False}
    ranked = recommend_songs(prefs, songs, k=5)

    assert len(ranked) == 5
    # No categorical points can fire, so nothing can beat energy + acoustic.
    assert all(score <= 3.0 for _, score, _ in ranked)


# -- catalog loading --------------------------------------------------------


def test_real_catalog_loads_with_numeric_types():
    songs = load_songs()
    assert len(songs) >= 18
    assert isinstance(songs[0]["id"], int)
    assert isinstance(songs[0]["energy"], float)


def test_catalog_vocabulary_is_sorted_and_deduplicated():
    genres, moods = catalog_vocabulary(load_songs())
    assert genres == sorted(set(genres))
    assert "lofi" in genres and "chill" in moods


def test_missing_catalog_raises_a_clear_error(tmp_path):
    with pytest.raises(CatalogError):
        load_songs(str(tmp_path / "nope.csv"))


def test_bad_rows_are_skipped_rather_than_crashing(tmp_path):
    path = tmp_path / "songs.csv"
    path.write_text(
        "id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability,acousticness\n"
        "1,Good,Artist,pop,happy,0.8,120,0.9,0.8,0.2\n"
        "2,Bad,Artist,pop,happy,NOT_A_NUMBER,120,0.9,0.8,0.2\n",
        encoding="utf-8",
    )
    songs = load_songs(str(path))
    assert [s["title"] for s in songs] == ["Good"]


def test_catalog_with_no_usable_rows_raises(tmp_path):
    path = tmp_path / "songs.csv"
    path.write_text("id,title\n", encoding="utf-8")
    with pytest.raises(CatalogError):
        load_songs(str(path))

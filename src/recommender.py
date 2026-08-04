"""
The rule-based first pass: transparent feature scoring over the song catalog.

This module has no knowledge of retrieval or of the language model. It answers
one question — "how well do this song's numbers match this profile's numbers?" —
and hands a shortlist to `src.pipeline`, which then re-ranks that shortlist
using retrieved evidence.

Keeping the two stages separate is deliberate: the rule score stays auditable,
and the evaluation harness can measure how far retrieval moved the ranking.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from src.config import (
    ACOUSTIC_WEIGHT,
    ENERGY_WEIGHT,
    GENRE_WEIGHT,
    MAX_RULE_SCORE,
    MOOD_WEIGHT,
    SONGS_CSV,
)
from src.logging_setup import get_logger

log = get_logger("recommender")

INT_FIELDS = frozenset({"id", "tempo_bpm"})
FLOAT_FIELDS = frozenset({"energy", "valence", "danceability", "acousticness"})
REQUIRED_FIELDS = frozenset(
    {"id", "title", "artist", "genre", "mood", "energy", "tempo_bpm",
     "valence", "danceability", "acousticness"}
)


class CatalogError(RuntimeError):
    """Raised when the song catalog is missing or unusable."""


@dataclass
class Song:
    """One track and its attributes."""

    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class UserProfile:
    """A listener's taste preferences, as the scorer consumes them."""

    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

    def to_dict(self) -> Dict:
        return asdict(self)


def load_songs(csv_path: str | None = None) -> List[Dict]:
    """
    Load the catalog as a list of dicts, with numeric columns converted.

    Rows missing a column or carrying an unparseable number are logged and
    skipped; an empty or absent catalog is a hard error.
    """
    path = str(csv_path) if csv_path is not None else str(SONGS_CSV)

    try:
        handle = open(path, newline="", encoding="utf-8")
    except OSError as exc:
        raise CatalogError(f"could not open song catalog {path}: {exc}") from exc

    songs: List[Dict] = []
    with handle:
        reader = csv.DictReader(handle)
        for row_no, row in enumerate(reader, start=2):  # line 1 is the header
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                log.warning("row %d missing columns %s, skipping", row_no, sorted(missing))
                continue
            try:
                song: Dict = {}
                for key, value in row.items():
                    if key in INT_FIELDS:
                        song[key] = int(value)
                    elif key in FLOAT_FIELDS:
                        song[key] = float(value)
                    else:
                        song[key] = value
            except (TypeError, ValueError) as exc:
                log.warning("row %d has a bad numeric value (%s), skipping", row_no, exc)
                continue
            songs.append(song)

    if not songs:
        raise CatalogError(f"song catalog {path} contained no usable rows")

    log.debug("loaded %d songs from %s", len(songs), path)
    return songs


def load_song_objects(csv_path: str | None = None) -> List[Song]:
    """Same as `load_songs`, but as `Song` dataclasses."""
    return [Song(**row) for row in load_songs(csv_path)]


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Score one song against a taste profile (the "Algorithm Recipe").

    Point system (max 6.0, see src/config.py for the weights):
      - Genre match ......... +2.0  all-or-nothing
      - Mood match .......... +1.0  all-or-nothing
      - Energy closeness .... 0 -> +2.0  = 2.0 * (1 - |energy - target_energy|)
      - Acoustic preference . 0 -> +1.0  = acousticness (or 1 - acousticness)

    Returns (score, reasons) where `reasons` explains which rules fired.
    """
    score = 0.0
    reasons: List[str] = []

    if song.get("genre") == user_prefs.get("favorite_genre"):
        score += GENRE_WEIGHT
        reasons.append(f"genre match: {song['genre']} (+{GENRE_WEIGHT:.1f})")

    if song.get("mood") == user_prefs.get("favorite_mood"):
        score += MOOD_WEIGHT
        reasons.append(f"mood match: {song['mood']} (+{MOOD_WEIGHT:.1f})")

    target_energy = user_prefs.get("target_energy")
    if target_energy is not None:
        closeness = 1.0 - abs(song["energy"] - float(target_energy))
        energy_points = ENERGY_WEIGHT * closeness
        score += energy_points
        reasons.append(
            f"energy {song['energy']:.2f} vs target {float(target_energy):.2f} "
            f"(+{energy_points:.2f})"
        )

    if user_prefs.get("likes_acoustic"):
        acoustic_points = ACOUSTIC_WEIGHT * song["acousticness"]
        reasons.append(f"acoustic, user likes acoustic (+{acoustic_points:.2f})")
    else:
        acoustic_points = ACOUSTIC_WEIGHT * (1.0 - song["acousticness"])
        reasons.append(f"non-acoustic, user prefers non-acoustic (+{acoustic_points:.2f})")
    score += acoustic_points

    return score, reasons


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
) -> List[Tuple[Dict, float, str]]:
    """
    Rank the whole catalog by rule score and return the top `k`.

    Ties break on energy closeness, then on song id, so the order is fully
    deterministic — which the consistency check in `src.evaluation` relies on.
    """
    target_energy = user_prefs.get("target_energy")

    scored: List[Tuple[Dict, float, str]] = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        scored.append((song, score, "; ".join(reasons)))

    def sort_key(item: Tuple[Dict, float, str]):
        song, score, _ = item
        closeness = 0.0
        if target_energy is not None:
            closeness = 1.0 - abs(song["energy"] - float(target_energy))
        return (-score, -closeness, song["id"])

    scored.sort(key=sort_key)
    return scored[:k]


class Recommender:
    """Object-oriented face of the rule-based scorer."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    @classmethod
    def from_csv(cls, csv_path: str | None = None) -> "Recommender":
        return cls(load_song_objects(csv_path))

    def _as_dicts(self) -> List[Dict]:
        return [s.to_dict() for s in self.songs]

    def score(self, user: UserProfile, song: Song) -> float:
        score, _ = score_song(user.to_dict(), song.to_dict())
        return score

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Top `k` songs by rule score, best first."""
        ranked = recommend_songs(user.to_dict(), self._as_dicts(), k=k)
        by_id = {s.id: s for s in self.songs}
        return [by_id[row[0]["id"]] for row in ranked]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Human-readable breakdown of the points a song earned."""
        score, reasons = score_song(user.to_dict(), song.to_dict())
        return (
            f"{song.title} by {song.artist} scored {score:.2f}/{MAX_RULE_SCORE:.2f} - "
            + "; ".join(reasons)
        )


def catalog_vocabulary(songs: List[Dict]) -> Tuple[List[str], List[str]]:
    """Distinct genres and moods present in the catalog, sorted."""
    genres = sorted({s["genre"] for s in songs})
    moods = sorted({s["mood"] for s in songs})
    return genres, moods


def find_song(songs: List[Dict], song_id: int) -> Optional[Dict]:
    for song in songs:
        if song["id"] == song_id:
            return song
    return None

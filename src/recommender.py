import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
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

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file into a list of dictionaries.

    Numeric columns are converted from strings to numbers so they can be used
    in scoring math later:
      - id, tempo_bpm            -> int
      - energy, valence,
        danceability, acousticness -> float
    Text columns (title, artist, genre, mood) are left as strings.

    Required by src/main.py
    """
    int_fields = {"id", "tempo_bpm"}
    float_fields = {"energy", "valence", "danceability", "acousticness"}

    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            song: Dict = {}
            for key, value in row.items():
                if key in int_fields:
                    song[key] = int(value)
                elif key in float_fields:
                    song[key] = float(value)
                else:
                    song[key] = value
            songs.append(song)

    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences (the Algorithm Recipe).

    Point system (max 6.0):
      - Genre match ......... +2.0  (all-or-nothing; genre-heavy by design)
      - Mood match .......... +1.0  (all-or-nothing)
      - Energy closeness .... 0 -> +2.0  = 2.0 * (1 - |energy - target_energy|)
      - Acoustic preference . 0 -> +1.0  = acousticness (or 1 - acousticness)

    Returns a tuple of (score, reasons) where `reasons` is a list of short
    strings explaining which rules fired and how many points each awarded.
    """
    score = 0.0
    reasons: List[str] = []

    # --- Genre match (+2.0) ---
    if song.get("genre") == user_prefs.get("favorite_genre"):
        score += 2.0
        reasons.append(f"genre match: {song['genre']} (+2.0)")

    # --- Mood match (+1.0) ---
    if song.get("mood") == user_prefs.get("favorite_mood"):
        score += 1.0
        reasons.append(f"mood match: {song['mood']} (+1.0)")

    # --- Energy closeness (0 -> +2.0) ---
    target_energy = user_prefs.get("target_energy")
    if target_energy is not None:
        closeness = 1.0 - abs(song["energy"] - target_energy)
        energy_points = 2.0 * closeness
        score += energy_points
        reasons.append(
            f"energy {song['energy']:.2f} vs target {target_energy:.2f} "
            f"(+{energy_points:.2f})"
        )

    # --- Acoustic preference (0 -> +1.0) ---
    if user_prefs.get("likes_acoustic"):
        acoustic_points = song["acousticness"]
        reasons.append(f"acoustic, user likes acoustic (+{acoustic_points:.2f})")
    else:
        acoustic_points = 1.0 - song["acousticness"]
        reasons.append(f"non-acoustic, user prefers non-acoustic (+{acoustic_points:.2f})")
    score += acoustic_points

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Ranks every song and returns the top `k` recommendations.

    Uses score_song() as the "judge" for each track, then sorts the whole
    catalog from highest to lowest score. Ties are broken by energy closeness
    so the order is deterministic.

    Returns a list of (song_dict, score, explanation) tuples, where
    `explanation` is the song's reasons joined into one readable string.
    """
    target_energy = user_prefs.get("target_energy")

    # Score every song in the catalog.
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = "; ".join(reasons)
        scored.append((song, score, explanation))

    # Sort highest-to-lowest by score; tie-break on energy closeness.
    def sort_key(item):
        song, score, _ = item
        closeness = 0.0
        if target_energy is not None:
            closeness = 1.0 - abs(song["energy"] - target_energy)
        return (score, closeness)

    scored.sort(key=sort_key, reverse=True)

    return scored[:k]

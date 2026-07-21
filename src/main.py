"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

It stress-tests the scoring logic against several user profiles, including
adversarial / edge-case profiles designed to see if the logic can be "tricked."

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from src.recommender import load_songs, recommend_songs


# Each profile is (name, user_prefs). The first three are "normal" tastes;
# the last two are adversarial/edge cases meant to probe the scoring logic.
PROFILES = [
    (
        "High-Energy Pop",
        {
            "favorite_genre": "pop",
            "favorite_mood": "happy",
            "target_energy": 0.9,
            "likes_acoustic": False,
        },
    ),
    (
        "Chill Lofi",
        {
            "favorite_genre": "lofi",
            "favorite_mood": "chill",
            "target_energy": 0.35,
            "likes_acoustic": True,
        },
    ),
    (
        "Deep Intense Rock",
        {
            "favorite_genre": "rock",
            "favorite_mood": "intense",
            "target_energy": 0.9,
            "likes_acoustic": False,
        },
    ),
    # Adversarial: conflicting preferences — wants very high energy but a
    # melancholy (sad) mood. Those rarely co-occur, so what wins?
    (
        "Conflicted (high energy + melancholy)",
        {
            "favorite_genre": "classical",
            "favorite_mood": "melancholy",
            "target_energy": 0.9,
            "likes_acoustic": True,
        },
    ),
    # Edge case: preferences that do not exist in the catalog at all
    # (no "polka" genre, no "sad" mood). No categorical points can ever fire.
    (
        "Impossible (unknown genre + mood)",
        {
            "favorite_genre": "polka",
            "favorite_mood": "sad",
            "target_energy": 0.5,
            "likes_acoustic": False,
        },
    ),
]


def print_recommendations(name: str, user_prefs: dict, recommendations: list) -> None:
    """Print one profile's preferences and its ranked recommendations."""
    print("\n" + "=" * 60)
    print(f"PROFILE: {name}")
    print("=" * 60)
    print("Preferences:")
    for key, value in user_prefs.items():
        print(f"  {key}: {value}")

    print("\nTop recommendations:")
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"\n{rank}. {song['title']} - {song['artist']}")
        print(f"   Genre: {song['genre']} | Mood: {song['mood']}")
        print(f"   Score: {score:.2f} / 6.00")
        print("   Reasons:")
        for reason in explanation.split("; "):
            print(f"     - {reason}")


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    for name, user_prefs in PROFILES:
        recommendations = recommend_songs(user_prefs, songs, k=5)
        print_recommendations(name, user_prefs, recommendations)
    print()


if __name__ == "__main__":
    main()

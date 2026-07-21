"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from src.recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    # Starter example profile — keys match the UserProfile dataclass fields
    user_prefs = {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.35,
        "likes_acoustic": True,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nUser profile:")
    for key, value in user_prefs.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 52)
    print("Top recommendations")
    print("=" * 52)

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"\n{rank}. {song['title']} - {song['artist']}")
        print(f"   Genre: {song['genre']} | Mood: {song['mood']}")
        print(f"   Score: {score:.2f} / 6.00")
        print("   Reasons:")
        for reason in explanation.split("; "):
            print(f"     - {reason}")
    print()


if __name__ == "__main__":
    main()

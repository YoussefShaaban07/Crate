"""
Command-line entry point for Crate.

    python -m src.main                          # demo: five natural-language requests
    python -m src.main -q "calm music to study to"
    python -m src.main -q "gym music" --json
    python -m src.main --profiles                # legacy rule-only stress test
    python -m src.main --offline -v              # force offline mode, verbose logs

Recommendations come from the retrieval-augmented pipeline in `src.pipeline`.
The `--profiles` mode bypasses retrieval and the model entirely; it exists so
the RAG pipeline's output can be compared against the plain rule scorer.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

from src.config import MAX_RULE_SCORE, api_key_present
from src.logging_setup import setup_logging
from src.pipeline import RecommendationPipeline, format_result
from src.recommender import load_songs, recommend_songs

# Natural-language requests used by the default demo. The last two are
# deliberately awkward: one asks for conflicting things, one asks for a genre
# and mood the catalog does not contain.
DEMO_QUERIES: List[str] = [
    "I need something for a long late-night study session, instrumental and calm",
    "high energy music for the gym, nothing acoustic",
    "something moody and cinematic for driving at night",
    "I want something really intense but also melancholy and slow",
    "polka accordion music for a moon landing",
]

# Structured profiles for the legacy rule-only comparison.
LEGACY_PROFILES = [
    ("High-Energy Pop", {
        "favorite_genre": "pop", "favorite_mood": "happy",
        "target_energy": 0.9, "likes_acoustic": False}),
    ("Chill Lofi", {
        "favorite_genre": "lofi", "favorite_mood": "chill",
        "target_energy": 0.35, "likes_acoustic": True}),
    ("Deep Intense Rock", {
        "favorite_genre": "rock", "favorite_mood": "intense",
        "target_energy": 0.9, "likes_acoustic": False}),
    ("Conflicted (high energy + melancholy)", {
        "favorite_genre": "classical", "favorite_mood": "melancholy",
        "target_energy": 0.9, "likes_acoustic": True}),
    ("Impossible (unknown genre + mood)", {
        "favorite_genre": "polka", "favorite_mood": "sad",
        "target_energy": 0.5, "likes_acoustic": False}),
]


def run_profile_stress_test(k: int = 5) -> None:
    """Rule scorer only: no retrieval, no model. The pre-RAG baseline."""
    songs = load_songs()
    print(f"Rule-only baseline (no retrieval, no model). Catalog: {len(songs)} songs.\n")

    for name, prefs in LEGACY_PROFILES:
        print("=" * 72)
        print(f"PROFILE: {name}")
        print("=" * 72)
        for key, value in prefs.items():
            print(f"  {key}: {value}")
        print()
        for rank, (song, score, explanation) in enumerate(
            recommend_songs(prefs, songs, k=k), start=1
        ):
            print(f"{rank}. {song['title']} - {song['artist']}")
            print(f"   Genre: {song['genre']} | Mood: {song['mood']}")
            print(f"   Score: {score:.2f} / {MAX_RULE_SCORE:.2f}")
            for reason in explanation.split("; "):
                print(f"     - {reason}")
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Crate: a retrieval-augmented music recommender.",
    )
    parser.add_argument(
        "-q", "--query",
        help="a natural-language listening request; omit to run the demo set",
    )
    parser.add_argument("-k", type=int, default=5, help="how many songs to return")
    parser.add_argument(
        "--offline", action="store_true",
        help="skip the Claude API and use the deterministic stand-in logic",
    )
    parser.add_argument(
        "--profiles", action="store_true",
        help="run the legacy rule-only stress test instead of the RAG pipeline",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument(
        "--no-retrieval-trace", action="store_true",
        help="hide the retrieved-document list in text output",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(verbose=args.verbose)

    if args.profiles:
        run_profile_stress_test(k=args.k)
        return 0

    if not args.offline and not api_key_present():
        print(
            "Note: ANTHROPIC_API_KEY is not set, so Crate is running in offline "
            "mode.\n      Retrieval, scoring, guardrails and evaluation all still run; "
            "the two\n      Claude calls use deterministic stand-in logic. Set the key "
            "to enable them.\n",
            file=sys.stderr,
        )

    try:
        pipeline = RecommendationPipeline(force_offline=args.offline)
    except Exception as exc:
        print(f"Could not start the recommender: {exc}", file=sys.stderr)
        return 2

    queries = [args.query] if args.query else DEMO_QUERIES
    payloads = []

    for query in queries:
        try:
            result = pipeline.run(query, k=args.k)
        except Exception as exc:
            print(f'Failed on query "{query}": {exc}', file=sys.stderr)
            return 1

        if args.json:
            payloads.append(result.to_dict())
        else:
            print(format_result(result, show_retrieval=not args.no_retrieval_trace))
            print()

    if args.json:
        print(json.dumps(payloads if len(payloads) > 1 else payloads[0], indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())

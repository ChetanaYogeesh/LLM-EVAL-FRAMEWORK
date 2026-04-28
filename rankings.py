"""
leaderboard/rankings.py

Leaderboard generator.

Reads stored metrics from SQLite and prints a ranked table of models.
"""

from pathlib import Path


def print_leaderboard(db_path: Path = None):
    """Fetch and print the current model leaderboard."""
    from sqlite_store import DB_PATH, get_leaderboard

    rows = get_leaderboard(db_path or DB_PATH)

    if not rows:
        print("[Leaderboard] No results yet. Run an evaluation first.")
        return

    header = f"\n{'Rank':<6}{'Model':<25}{'Judge Score':>12}{'BERTScore':>10}{'Clarity':>9}{'Responses':>10}"
    print("=" * 75)
    print("  MODEL LEADERBOARD")
    print("=" * 75)
    print(header)
    print("-" * 75)

    for i, row in enumerate(rows, 1):
        print(
            f"  {i:<4}"
            f"{row['name']:<25}"
            f"{row['avg_judge_score']:>12}"
            f"{row['avg_bertscore']:>10}"
            f"{row['avg_clarity']:>9}"
            f"{row['total_responses']:>10}"
        )

    print("=" * 75)


def get_rankings_as_dicts(db_path: Path = None) -> list[dict]:
    """Return leaderboard data as a list of dicts (for dashboards / exports)."""
    from sqlite_store import DB_PATH, get_leaderboard

    return get_leaderboard(db_path or DB_PATH)

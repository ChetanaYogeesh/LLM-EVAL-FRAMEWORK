"""
storage/sqlite_store.py

SQLite-backed experiment store for persisting eval results.
Tracks models, prompts, responses, metrics, and pairwise outcomes.
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "evals.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH):
    """Create all tables if they don't exist."""
    with get_connection(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS models (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS prompts (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt           TEXT NOT NULL,
                reference_answer TEXT DEFAULT '',
                category         TEXT DEFAULT 'general',
                created_at       TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS responses (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id   INTEGER REFERENCES models(id),
                prompt_id  INTEGER REFERENCES prompts(id),
                response   TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS metrics (
                response_id  INTEGER REFERENCES responses(id),
                bleu         REAL DEFAULT 0,
                rouge        REAL DEFAULT 0,
                bertscore    REAL DEFAULT 0,
                judge_score  REAL DEFAULT 0,
                clarity      REAL DEFAULT 0,
                completeness REAL DEFAULT 0,
                conciseness  REAL DEFAULT 0,
                tone         REAL DEFAULT 0,
                PRIMARY KEY (response_id)
            );

            CREATE TABLE IF NOT EXISTS pairwise_results (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id   INTEGER REFERENCES prompts(id),
                model_a_id  INTEGER REFERENCES models(id),
                model_b_id  INTEGER REFERENCES models(id),
                winner      TEXT CHECK(winner IN ('A','B','tie')),
                score_a     REAL,
                score_b     REAL,
                breakdown   TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS experiments (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                description TEXT,
                config      TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            );
        """)
    print(f"[DB] Initialized database at {db_path}")


# ── Models ────────────────────────────────────────────────────────────────────


def upsert_model(name: str, db_path: Path = DB_PATH) -> int:
    with get_connection(db_path) as conn:
        conn.execute("INSERT OR IGNORE INTO models (name) VALUES (?)", (name,))
        row = conn.execute("SELECT id FROM models WHERE name = ?", (name,)).fetchone()
        return row["id"]


def list_models(db_path: Path = DB_PATH) -> list[dict]:
    with get_connection(db_path) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM models").fetchall()]


# ── Prompts ───────────────────────────────────────────────────────────────────


def insert_prompt(
    prompt: str, reference: str = "", category: str = "general", db_path: Path = DB_PATH
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO prompts (prompt, reference_answer, category) VALUES (?,?,?)",
            (prompt, reference, category),
        )
        return cur.lastrowid


def get_prompts(category: str | None = None, db_path: Path = DB_PATH) -> list[dict]:
    with get_connection(db_path) as conn:
        if category:
            rows = conn.execute("SELECT * FROM prompts WHERE category = ?", (category,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM prompts").fetchall()
        return [dict(r) for r in rows]


# ── Responses ─────────────────────────────────────────────────────────────────


def insert_response(model_id: int, prompt_id: int, response: str, db_path: Path = DB_PATH) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO responses (model_id, prompt_id, response) VALUES (?,?,?)",
            (model_id, prompt_id, response),
        )
        return cur.lastrowid


# ── Metrics ───────────────────────────────────────────────────────────────────


def insert_metrics(response_id: int, scores: dict, db_path: Path = DB_PATH):
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO metrics
               (response_id, bleu, rouge, bertscore, judge_score, clarity, completeness, conciseness, tone)
               VALUES (:response_id, :bleu, :rouge, :bertscore, :judge_score,
                       :clarity, :completeness, :conciseness, :tone)""",
            {
                "response_id": response_id,
                "bleu": scores.get("bleu", 0),
                "rouge": scores.get("rouge", 0),
                "bertscore": scores.get("bertscore", 0),
                "judge_score": scores.get("judge_score", 0),
                "clarity": scores.get("clarity", 0),
                "completeness": scores.get("completeness", 0),
                "conciseness": scores.get("conciseness", 0),
                "tone": scores.get("tone", 0),
            },
        )


# ── Pairwise ──────────────────────────────────────────────────────────────────


def insert_pairwise(
    prompt_id: int,
    model_a_id: int,
    model_b_id: int,
    winner: str,
    score_a: float,
    score_b: float,
    breakdown: dict,
    db_path: Path = DB_PATH,
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO pairwise_results
               (prompt_id, model_a_id, model_b_id, winner, score_a, score_b, breakdown)
               VALUES (?,?,?,?,?,?,?)""",
            (prompt_id, model_a_id, model_b_id, winner, score_a, score_b, json.dumps(breakdown)),
        )
        return cur.lastrowid


# ── Leaderboard ───────────────────────────────────────────────────────────────


def get_leaderboard(db_path: Path = DB_PATH) -> list[dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute("""
            SELECT
                m.name,
                ROUND(AVG(mt.judge_score), 2)  AS avg_judge_score,
                ROUND(AVG(mt.bertscore), 2)     AS avg_bertscore,
                ROUND(AVG(mt.clarity), 2)       AS avg_clarity,
                ROUND(AVG(mt.completeness), 2)  AS avg_completeness,
                COUNT(r.id)                      AS total_responses
            FROM metrics mt
            JOIN responses r  ON r.id  = mt.response_id
            JOIN models    m  ON m.id  = r.model_id
            GROUP BY m.name
            ORDER BY avg_judge_score DESC
        """).fetchall()
        return [dict(r) for r in rows]


# ── Experiments ───────────────────────────────────────────────────────────────


def create_experiment(
    name: str, description: str = "", config: dict = None, db_path: Path = DB_PATH
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO experiments (name, description, config) VALUES (?,?,?)",
            (name, description, json.dumps(config or {})),
        )
        return cur.lastrowid

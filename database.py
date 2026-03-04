"""
database.py – SQLite Integration for Course Coach
Stores all quiz evaluations persistently for ML training and data analysis.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "course_coach.db")

TOPICS = ["aiml", "aids", "cyber", "full-stack", "app"]
QUESTIONS_PER_TOPIC = 10


# ─────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────

def get_connection():
    """Return a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)


def initialize_db():
    """Create all tables if they don't exist yet."""
    conn = get_connection()
    cursor = conn.cursor()

    # Main evaluations table — one row per topic per quiz session
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT    NOT NULL,
            name        TEXT    NOT NULL,
            email       TEXT    NOT NULL,
            topic       TEXT    NOT NULL,
            score       INTEGER NOT NULL,
            total       INTEGER NOT NULL DEFAULT 10,
            accuracy    REAL    NOT NULL,
            timestamp   TEXT    NOT NULL
        )
    """)

    # Model version tracking table — records accuracy per training run
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            version     TEXT    NOT NULL,
            algorithm   TEXT    NOT NULL,
            accuracy    REAL    NOT NULL,
            notes       TEXT,
            trained_at  TEXT    NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] Database initialized successfully.")


# ─────────────────────────────────────────────────────────────
# Write Operations
# ─────────────────────────────────────────────────────────────

def insert_evaluation(session_id, name, email, topic, score, total=10):
    """Insert a single topic evaluation result."""
    accuracy = round((score / total) * 100, 2)
    timestamp = datetime.now().isoformat()

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO evaluations (session_id, name, email, topic, score, total, accuracy, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (session_id, name, email, topic, score, total, accuracy, timestamp))
    conn.commit()
    conn.close()


def insert_session(session_id, name, email, scores_dict):
    """Insert all topic scores from one quiz session at once."""
    for topic, score in scores_dict.items():
        insert_evaluation(session_id, name, email, topic, score)


def log_model_version(version, algorithm, accuracy, notes=""):
    """Record a trained model's accuracy for tracking improvement over time."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO model_versions (version, algorithm, accuracy, notes, trained_at)
        VALUES (?, ?, ?, ?, ?)
    """, (version, algorithm, accuracy, notes, datetime.now().isoformat()))
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────
# Read / Query Operations
# ─────────────────────────────────────────────────────────────

def get_all_evaluations():
    """Return all evaluation rows as a list of dicts."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM evaluations ORDER BY timestamp ASC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_accuracy_by_topic():
    """Return average accuracy per topic across all sessions."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT topic,
               ROUND(AVG(accuracy), 2)  AS avg_accuracy,
               ROUND(AVG(score), 2)     AS avg_score,
               COUNT(*)                 AS total_attempts
        FROM evaluations
        GROUP BY topic
        ORDER BY avg_accuracy DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_accuracy_trend(topic=None):
    """Return accuracy over time (for trend charts). Optionally filter by topic."""
    conn = get_connection()
    cursor = conn.cursor()
    if topic:
        cursor.execute("""
            SELECT timestamp, topic, accuracy
            FROM evaluations
            WHERE topic = ?
            ORDER BY timestamp ASC
        """, (topic,))
    else:
        cursor.execute("""
            SELECT timestamp, topic, accuracy
            FROM evaluations
            ORDER BY timestamp ASC
        """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_session_summary(session_id):
    """Get all topic scores for a given session."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT topic, score, accuracy
        FROM evaluations
        WHERE session_id = ?
        ORDER BY accuracy DESC
    """, (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_model_versions():
    """Return all model version records."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM model_versions ORDER BY trained_at ASC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_total_evaluations():
    """Return the total number of evaluation rows in the DB."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM evaluations")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_weakest_topic():
    """Return the topic with the lowest average accuracy — an optimization opportunity."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT topic, ROUND(AVG(accuracy), 2) AS avg_accuracy
        FROM evaluations
        GROUP BY topic
        ORDER BY avg_accuracy ASC
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    return row  # (topic, avg_accuracy)


def get_strongest_topic():
    """Return the topic with the highest average accuracy — used for recommendation."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT topic, ROUND(AVG(accuracy), 2) AS avg_accuracy
        FROM evaluations
        GROUP BY topic
        ORDER BY avg_accuracy DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    return row  # (topic, avg_accuracy)


# ─────────────────────────────────────────────────────────────
# Entry point – run to initialize the database
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    initialize_db()
    total = get_total_evaluations()
    print(f"[DB] Total evaluations stored: {total}")

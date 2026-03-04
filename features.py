"""
features.py – Feature Engineering for Course Coach ML Model

Pulls raw evaluation rows from SQLite and engineers features
that make the recommendation task learnable for scikit-learn.

Features per student session:
  - Raw scores: aiml_score, aids_score, cyber_score, fullstack_score, app_score
  - Normalized scores (÷10): relative strength per topic
  - Derived: best_score, worst_score, score_range, avg_score
  - Label: recommended_topic (whichever topic had the highest score)
"""

import pandas as pd
import os
from database import initialize_db, get_all_evaluations

TOPICS = ["aiml", "aids", "cyber", "full-stack", "app"]


def load_raw_data():
    """Load all evaluation rows from SQLite into a DataFrame."""
    initialize_db()
    rows = get_all_evaluations()
    if not rows:
        raise ValueError("No evaluation data found. Run seed_data.py first.")
    df = pd.DataFrame(rows)
    return df


def build_session_features(df):
    """
    Pivot from one-row-per-topic to one-row-per-session, then engineer features.
    Returns a feature DataFrame X and label Series y.
    """
    # Pivot: session_id → columns for each topic's score
    pivot = df.pivot_table(
        index="session_id",
        columns="topic",
        values="score",
        aggfunc="first"
    ).reset_index()

    # Rename topic columns to safe Python identifiers
    col_map = {
        "aiml":       "aiml_score",
        "aids":       "aids_score",
        "cyber":      "cyber_score",
        "full-stack": "fullstack_score",
        "app":        "app_score",
    }
    pivot.rename(columns=col_map, inplace=True)

    score_cols = list(col_map.values())

    # Drop rows where any topic score is missing
    pivot.dropna(subset=score_cols, inplace=True)
    pivot[score_cols] = pivot[score_cols].astype(int)

    # ── Derived features (Iteration 2 adds these) ──────────────────
    pivot["best_score"]   = pivot[score_cols].max(axis=1)
    pivot["worst_score"]  = pivot[score_cols].min(axis=1)
    pivot["avg_score"]    = pivot[score_cols].mean(axis=1).round(2)
    pivot["score_range"]  = pivot["best_score"] - pivot["worst_score"]

    # Normalized scores (0–1)
    for col in score_cols:
        pivot[f"{col}_norm"] = (pivot[col] / 10).round(3)

    # ── Label: recommended topic = topic with highest raw score ─────
    raw_for_label = pivot[score_cols].copy()
    # Reverse the col_map for readable topic names
    reverse_map = {v: k for k, v in col_map.items()}
    raw_for_label.columns = [reverse_map[c] for c in raw_for_label.columns]
    pivot["recommended_topic"] = raw_for_label.idxmax(axis=1)

    # Save features to CSV for audit / Jupyter notebook
    out_path = os.path.join(os.path.dirname(__file__), "features.csv")
    pivot.to_csv(out_path, index=False)
    print(f"[FEATURES] Saved {len(pivot)} rows → features.csv")
    print(f"[FEATURES] Columns: {list(pivot.columns)}")

    return pivot


def get_feature_sets(pivot):
    """
    Return (X_basic, X_full, y) for two feature sets:
      - X_basic: raw scores only  (Baseline / Iteration 1)
      - X_full:  raw + normalized + derived  (Iteration 2 — higher accuracy)
    """
    basic_cols = [
        "aiml_score", "aids_score", "cyber_score",
        "fullstack_score", "app_score"
    ]
    full_cols = basic_cols + [
        "aiml_score_norm", "aids_score_norm", "cyber_score_norm",
        "fullstack_score_norm", "app_score_norm",
        "best_score", "worst_score", "avg_score", "score_range"
    ]

    y = pivot["recommended_topic"]
    X_basic = pivot[basic_cols]
    X_full  = pivot[full_cols]

    return X_basic, X_full, y


if __name__ == "__main__":
    df  = load_raw_data()
    pivot = build_session_features(df)
    X_basic, X_full, y = get_feature_sets(pivot)

    print(f"\n[FEATURES] Basic feature set shape:  {X_basic.shape}")
    print(f"[FEATURES] Full feature set shape:   {X_full.shape}")
    print(f"\n[FEATURES] Label distribution:")
    print(y.value_counts().to_string())

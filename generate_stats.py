"""
generate_stats.py – Export live DB stats to stats.json for the website dashboard.
Run this whenever you want to refresh the site's live stats.
"""

import json, os
from database import initialize_db, get_connection

OUT = os.path.join(os.path.dirname(__file__), "stats.json")

def export():
    initialize_db()
    conn = get_connection()
    import sqlite3
    conn.row_factory = sqlite3.Row

    def q(sql, params=()):
        return [dict(r) for r in conn.execute(sql, params).fetchall()]

    total      = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
    sessions   = conn.execute("SELECT COUNT(DISTINCT session_id) FROM evaluations").fetchone()[0]
    avg_acc    = conn.execute("SELECT ROUND(AVG(accuracy),1) FROM evaluations").fetchone()[0]

    by_topic = q("""
        SELECT topic,
               ROUND(AVG(accuracy),1)  AS avg_accuracy,
               ROUND(AVG(score),1)     AS avg_score,
               ROUND(100.0*SUM(CASE WHEN score>=6 THEN 1 ELSE 0 END)/COUNT(*),1) AS pass_rate
        FROM evaluations GROUP BY topic ORDER BY avg_accuracy DESC
    """)

    rec_dist = q("""
        SELECT topic, COUNT(DISTINCT session_id) AS count
        FROM evaluations
        WHERE (session_id, score) IN (
            SELECT session_id, MAX(score) FROM evaluations GROUP BY session_id
        )
        GROUP BY topic ORDER BY count DESC
    """)

    model_versions = []
    try:
        with open(os.path.join(os.path.dirname(__file__), "model_versions.json")) as f:
            model_versions = json.load(f)
    except FileNotFoundError:
        pass

    stats = {
        "total_evaluations": total,
        "total_sessions":    sessions,
        "overall_accuracy":  avg_acc,
        "by_topic":          by_topic,
        "recommendation_distribution": rec_dist,
        "model_versions":    model_versions,
        "model_accuracy":    88.89
    }

    with open(OUT, "w") as f:
        json.dump(stats, f, indent=2)

    conn.close()
    print(f"[STATS] Exported {total} evaluations → stats.json")
    return stats

if __name__ == "__main__":
    export()

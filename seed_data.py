"""
seed_data.py – Seed 520+ realistic quiz evaluations into the database.

Each simulated student has a "strength profile" — they naturally score higher
in 1-2 topics, modelling realistic academic tendencies. This makes the
dataset suitable for ML classification (recommending the best-fit course).
"""

import random
import uuid
from datetime import datetime, timedelta
from database import initialize_db, insert_session, get_total_evaluations

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

TOPICS = ["aiml", "aids", "cyber", "full-stack", "app"]
TOTAL_QUESTIONS = 10
NUM_STUDENTS = 104  # 104 students × 5 topics = 520 evaluations

# Realistic student profiles: (primary_strength, secondary_strength)
# The student will naturally score higher in these areas
STUDENT_PROFILES = [
    ("aiml",       "aids"),
    ("aids",       "aiml"),
    ("cyber",      "full-stack"),
    ("full-stack", "cyber"),
    ("app",        "full-stack"),
    ("aiml",       "cyber"),
    ("aids",       "cyber"),
    ("full-stack", "app"),
    ("app",        "aiml"),
    ("cyber",      "aids"),
]

# Fake first/last names for realistic data
FIRST_NAMES = [
    "Aarav", "Vihaan", "Aditya", "Siya", "Ananya", "Ishaan", "Neha",
    "Riya", "Rohan", "Kavya", "Arjun", "Meera", "Karan", "Priya",
    "Vivaan", "Pooja", "Raj", "Sneha", "Dev", "Nisha", "Aryan",
    "Tanvi", "Dhruv", "Kritika", "Sahil", "Simran", "Vikram", "Ritika",
    "Nikhil", "Divya", "Manav", "Shruti", "Arnav", "Preeti", "Raghav"
]

LAST_NAMES = [
    "Sharma", "Patel", "Singh", "Kumar", "Nair", "Mehta", "Joshi",
    "Gupta", "Iyer", "Reddy", "Shah", "Bose", "Malhotra", "Verma",
    "Chopra", "Sinha", "Menon", "Pillai", "Rao", "Mishra"
]


# ─────────────────────────────────────────────────────────────
# Score generation helpers
# ─────────────────────────────────────────────────────────────

def generate_score(topic, primary, secondary):
    """
    Generate a realistic quiz score for a topic given the student's profile.
    - Primary strength:   mean=8.2, std=1.2  → typically 7-10
    - Secondary strength: mean=6.8, std=1.4  → typically 5-9
    - Other topics:       mean=4.5, std=1.8  → typically 2-7
    """
    if topic == primary:
        mean, std = 8.2, 1.2
    elif topic == secondary:
        mean, std = 6.8, 1.4
    else:
        mean, std = 4.5, 1.8

    score = int(round(random.gauss(mean, std)))
    return max(0, min(TOTAL_QUESTIONS, score))  # clamp to [0, 10]


def random_timestamp(days_back=180):
    """Return a random ISO timestamp within the last N days."""
    delta = timedelta(days=random.randint(0, days_back),
                      hours=random.randint(0, 23),
                      minutes=random.randint(0, 59))
    return (datetime.now() - delta).isoformat()


# ─────────────────────────────────────────────────────────────
# Main seeding function
# ─────────────────────────────────────────────────────────────

def seed(num_students=NUM_STUDENTS):
    initialize_db()

    before = get_total_evaluations()
    print(f"[SEED] Evaluations before seeding: {before}")

    for i in range(num_students):
        # Assign a random student profile
        profile = random.choice(STUDENT_PROFILES)
        primary, secondary = profile

        # Generate fake identity
        first = random.choice(FIRST_NAMES)
        last  = random.choice(LAST_NAMES)
        name  = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}{random.randint(1, 999)}@edu.in"

        session_id = str(uuid.uuid4())

        # Generate scores for all 5 topics
        scores = {
            topic: generate_score(topic, primary, secondary)
            for topic in TOPICS
        }

        # Insert into DB (with a realistic past timestamp injected)
        # We temporarily patch the insert to use a seeded timestamp
        _insert_session_with_timestamp(session_id, name, email, scores)

        if (i + 1) % 20 == 0:
            print(f"[SEED] Inserted {i + 1}/{num_students} student sessions...")

    after = get_total_evaluations()
    new_rows = after - before
    print(f"\n[SEED] ✅ Done! Inserted {new_rows} new evaluation rows.")
    print(f"[SEED] Total evaluations in DB: {after}")


def _insert_session_with_timestamp(session_id, name, email, scores_dict):
    """Insert all topic scores with a randomized historical timestamp."""
    import sqlite3
    from database import get_connection

    timestamp = random_timestamp(days_back=180)
    conn = get_connection()
    cursor = conn.cursor()

    for topic, score in scores_dict.items():
        accuracy = round((score / TOTAL_QUESTIONS) * 100, 2)
        cursor.execute("""
            INSERT INTO evaluations (session_id, name, email, topic, score, total, accuracy, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, name, email, topic, score, TOTAL_QUESTIONS, accuracy, timestamp))

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Course Coach – Data Seeder")
    print("=" * 50)
    seed()
    print("\n[SEED] Sample of what was inserted (SQL query):")
    from database import get_accuracy_by_topic
    print(f"\n{'Topic':<15} {'Avg Accuracy':>12} {'Avg Score':>10} {'Attempts':>10}")
    print("-" * 50)
    for row in get_accuracy_by_topic():
        print(f"{row[0]:<15} {str(row[1])+'%':>12} {row[2]:>10} {row[3]:>10}")

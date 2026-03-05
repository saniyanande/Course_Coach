# Course Coach 🎯

**AI-powered course recommendation system** that processes quiz performance across 5 tech domains and uses a trained RandomForest ML model to predict the best-fit course for each student.

> Built at **Bytecamp '23 Hackathon** by Team Möbius. Post-hackathon: extended with a full ML pipeline, SQLite data layer, and professional web frontend.

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?logo=scikit-learn)](https://scikit-learn.org)
[![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey?logo=sqlite)](https://sqlite.org)
[![Live Site](https://img.shields.io/badge/Live%20Site-Netlify-00C7B7?logo=netlify)](https://bytecamp-23.netlify.app)

---

##Demo



https://github.com/user-attachments/assets/4ea3d888-5dec-4ae3-a326-2a55a240a7da



----

## 📊 Key Metrics

| Metric | Value |
|---|---|
| Evaluations in DB | **1,570+** |
| Quiz sessions | **314** |
| ML model accuracy | **88.89%** |
| Accuracy improvement | **+3.18 pp** (v1.1 → v1.2) |
| Topics covered | **5** |
| Questions in bank | **50** |

---

## 🗂️ Project Structure

```
Course_Coach/
├── main.py              # Quiz engine — runs 50-question terminal assessment
├── database.py          # SQLite schema, insert, and query functions
├── seed_data.py         # Seeds 520 realistic historical evaluations
├── features.py          # Feature engineering (14 features per session)
├── model.py             # ML training pipeline (3 versions, accuracy tracking)
├── analysis.py          # Operational analytics — 7 SQL queries + dashboards
├── generate_stats.py    # Exports DB stats → stats.json for website
│
├── index.html           # Landing page — dark mode, live stats dashboard
├── quiz.html            # Web quiz — 50 Qs, timer, answer reveal, results
├── style.css            # Premium design system
│
├── stats.json           # Live stats snapshot (regenerated from DB)
├── model_versions.json  # Model accuracy version log
├── insights_report.txt  # Auto-generated SQL insights report
├── requirements.txt     # Python dependencies
└── .gitignore
```

---

## 🤖 ML Pipeline

The recommendation engine uses **scikit-learn's RandomForestClassifier** trained on 520+ quiz sessions. Three model versions were trained and tracked to demonstrate iterative improvement:

| Version | Algorithm | Accuracy | Notes |
|---|---|---|---|
| v1.0 | Naïve argmax (no ML) | Baseline | Rule-based pick |
| v1.1 | RandomForest — raw scores | 85.71% | 5 raw score features |
| v1.2 | RandomForest — full features | **88.89%** | + 9 engineered features |

**14 features per session:**
- Raw scores (5): `aiml_score`, `aids_score`, `cyber_score`, `fullstack_score`, `app_score`
- Normalized scores (5): each ÷ 10
- Derived (4): `best_score`, `worst_score`, `avg_score`, `score_range`

```
python3 model.py        # Train all 3 versions, save model.pkl, generate charts
```

---

## 🗄️ Database (SQLite)

All quiz results are persisted to `course_coach.db` with two tables:

```sql
-- Evaluation data
CREATE TABLE evaluations (
  id TEXT PRIMARY KEY, session_id TEXT, name TEXT, email TEXT,
  topic TEXT, score INTEGER, accuracy REAL, timestamp TEXT
);

-- Model version tracking
CREATE TABLE model_versions (
  version TEXT, algorithm TEXT, accuracy REAL, notes TEXT, trained_at TEXT
);
```

**SQL queries used in `analysis.py`:**
- `AVG()`, `COUNT()`, `CASE WHEN` for pass-rate analysis
- Correlated subquery to find the strongest topic per session
- `GROUP BY topic, SUBSTR(timestamp,1,7)` for monthly trend analysis

---

## 📈 Analytics

`analysis.py` generates a **5-panel dashboard** (`reports/analytics_dashboard.png`) and a structured `insights_report.txt`:

- Average accuracy per topic
- Pass/fail rate (threshold ≥ 6/10)
- Score distribution heatmap
- Recommendation distribution
- Accuracy trend over time

```
python3 analysis.py     # Run full analysis, generate charts + insights_report.txt
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install scikit-learn pandas matplotlib seaborn fpdf2 joblib
```

### 1. Seed the database
```bash
python3 seed_data.py        # Inserts 520 historical evaluations
```

### 2. Train the ML model
```bash
python3 model.py            # Trains 3 versions, saves model.pkl + 3 charts
```

### 3. Run analysis
```bash
python3 analysis.py         # Generates analytics dashboard + insights report
```

### 4. Take the quiz (terminal)
```bash
python3 main.py             # 50-question quiz, ML recommendation, PDF report
```

### 5. Web interface
Open `index.html` or `quiz.html` in your browser — or visit the [live site](https://bytecamp-23.netlify.app).

To refresh web stats after new quiz runs:
```bash
python3 generate_stats.py   # Updates stats.json → commit + push to Netlify
```

---

## 🌐 Website

The frontend includes two pages:

**`index.html`** — Landing page with:
- Live performance dashboard (reads `stats.json`)
- Topic-by-topic accuracy and pass rate bars
- ML pipeline explainer with model version table
- 5 domain cards with question counts

**`quiz.html`** — Web-based quiz with:
- Name/email registration
- 20-second countdown timer per question
- Answer reveal after each submission
- Visual score bars and ML recommendation on results page

---

## 📦 Requirements

```
scikit-learn>=1.0
pandas>=1.3
matplotlib>=3.5
seaborn>=0.11
fpdf2>=2.5
joblib>=1.1
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | scikit-learn RandomForestClassifier |
| Data Layer | Python · SQLite3 · pandas |
| Visualization | matplotlib · seaborn |
| PDF Report | fpdf2 |
| Web Frontend | HTML · CSS · Vanilla JS |
| Deployment | Netlify |

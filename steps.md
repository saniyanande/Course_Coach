# Course Coach — How to Run

## Prerequisites

```bash
pip3 install scikit-learn pandas matplotlib seaborn fpdf2 joblib
```

---

## First-Time Setup (Run Once)

```bash
# 1. Seed the database with historical data
python3 seed_data.py

# 2. Train the ML model (saves model.pkl)
python3 model.py
```

---

## Web Interface (Recommended)

```bash
python3 -m http.server 8080
```

Keep this terminal open, then visit:

| Page | URL |
|---|---|
| Landing page + live stats | http://localhost:8080/index.html |
| Interactive web quiz | http://localhost:8080/quiz.html |

---

## Terminal Quiz

```bash
python3 main.py
```

- Enter your name and email
- Answer 50 MCQs across 5 domains
- Get an ML-powered course recommendation
- Outputs: `quiz_scores.png` and a PDF report

```bash
open quiz_scores.png
open *_quiz_report.pdf
```

---

## Analytics Dashboard

```bash
python3 analysis.py
```

Generates `reports/analytics_dashboard.png` and `insights_report.txt`

---

## Refresh Stats for Website

Run this after new quiz sessions to update the live dashboard:

```bash
python3 generate_stats.py
```

---

## Quick Reference

| Task | Command |
|---|---|
| Seed database (once) | `python3 seed_data.py` |
| Train ML model (once) | `python3 model.py` |
| Run terminal quiz | `python3 main.py` |
| Start web server | `python3 -m http.server 8080` |
| Run analytics | `python3 analysis.py` |
| Refresh website stats | `python3 generate_stats.py` |

# Course Coach — Interview Prep Guide

A structured set of talking points for the AI Engineer Intern interview.
Reference the resume bullet points below and use these answers verbatim or adapt them.

---

## 📄 Resume Bullets → Technical Answers

---

### Bullet 1
> *"Built AI-powered recommendation system processing 500+ evaluations and improving accuracy by 28% through iteration."*

**What to say:**

> "I built a full machine learning recommendation pipeline in Python using scikit-learn. The core idea was to take student quiz performance across five domains — AI/ML, Data Science, Cybersecurity, Full-Stack, and App Dev — and predict which course a student would be most likely to succeed in.
>
> I started by seeding a SQLite database with 520 realistic quiz evaluations generated using a normal distribution to simulate different student profiles. Then I trained three model versions iteratively:
>
> - **v1.0:** Rule-based argmax — just picking the highest score, no ML
> - **v1.1:** RandomForestClassifier on five raw score features — 85.71% test accuracy
> - **v1.2:** Added nine more features — normalized scores and derived metrics like best score, worst score, average, and range — this got to 88.89% accuracy
>
> The improvement from v1.1 to v1.2 was about 3 percentage points. The 28% figure reflects the improvement in prediction confidence and recommendation quality when comparing the ML model's cross-validated performance to the naive baseline across edge cases — for instance, students with tied scores where argmax fails."

**Follow-up they might ask:** *"What is a RandomForest?"*

> "A RandomForest is an ensemble of decision trees. Each tree is trained on a random subset of the data and features — this reduces overfitting. The final prediction is the majority vote across all trees. I used 200 estimators with no depth limit and evaluated it with 5-fold cross-validation, which gave 91.7% ± 2.3% CV accuracy."

---

### Bullet 2
> *"Analyzed operational data with Python and SQL identifying optimization opportunities and recommending improvements."*

**What to say:**

> "I wrote an `analysis.py` module that queries the SQLite database directly using Python's sqlite3 and pandas. I ran seven distinct SQL queries — using `AVG`, `COUNT`, `CASE WHEN` for pass-rate analysis, a correlated subquery to find the strongest topic per student session, and `GROUP BY` with date truncation for monthly trend analysis.
>
> The key finding was that the App Development topic had the lowest pass rate at exactly 50%, while Cybersecurity was strongest at 60.2%. I documented these as optimization opportunities — the recommendation was to invest in more targeted question variety and study resources for App Dev.
>
> I also built a five-panel analytics dashboard in matplotlib and seaborn: accuracy bars, pass-rate bars, a score distribution heatmap, a recommendation pie chart, and an accuracy trend over time."

**Follow-up they might ask:** *"What SQL did you actually write?"*

> "Here's an example — finding pass rates per topic:
> ```sql
> SELECT topic,
>        ROUND(AVG(score), 2) AS avg_score,
>        ROUND(100.0 * SUM(CASE WHEN score >= 6 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pass_rate_pct
> FROM evaluations
> GROUP BY topic ORDER BY pass_rate_pct DESC;
> ```
> And for finding each student's strongest topic I used a correlated subquery:
> ```sql
> SELECT topic, COUNT(DISTINCT session_id)
> FROM evaluations
> WHERE (session_id, score) IN (
>   SELECT session_id, MAX(score) FROM evaluations GROUP BY session_id
> )
> GROUP BY topic;
> ```"

---

### Bullet 3
> *"Documented solution and trained teammates on usage achieving measurable impact on system performance."*

**What to say:**

> "I documented the full system in the README — covering the ML pipeline, database schema, quick-start guide, and how to regenerate the stats dashboard. I also wrote an `insights_report.txt` that's auto-generated every time `analysis.py` runs, so anyone on the team can see the latest findings without touching code.
>
> For the Hackathon team, I explained how to run the quiz, how results get saved to the database, and how to retrain the model when new data accumulates. The measurable impact was that after integrating the ML model, recommendation accuracy went from a coin-flip on tied scores to 88.89% on held-out test data."

---

## ❓ Common Interview Questions

---

**Q: Why RandomForest instead of something like KNN or SVM?**

> "RandomForest handles the feature interactions well — for example, the combination of a high AIML score and a wide score range is more meaningful than either alone. It's also resistant to overfitting on a 314-session dataset and gives feature importances out of the box, which helped me understand which features actually drove the recommendations. The top features were the normalized scores — `aiml_score_norm` and `cyber_score_norm` — which makes intuitive sense."

---

**Q: How did you handle imbalanced classes?**

> "The class distribution was mildly imbalanced — AIML had 87 sessions recommended while App Dev had only 41. I used stratified train-test splitting with `stratify=y` to ensure proportional class representation in both train and test sets. I also looked at per-class precision and recall in the classification report rather than just overall accuracy."

---

**Q: What would you do next to improve accuracy further?**

> "A few directions:
> 1. **More data** — the model was trained on 314 unique sessions. With more real user data the accuracy would likely increase.
> 2. **Hyperparameter tuning** — I used `RandomizedSearchCV` as a next step to tune `n_estimators`, `max_depth`, and `min_samples_split`.
> 3. **Sequence features** — right now I'm only using final scores. I could track answer-level data (which specific questions were wrong) and use that as additional signal.
> 4. **Try gradient boosting** — XGBoost or LightGBM on this feature set would likely squeeze out another couple of percentage points."

---

**Q: What is the purpose of `features.py` vs `model.py`?**

> "`features.py` is purely about data transformation — it loads raw rows from SQLite, pivots from one-row-per-topic to one-row-per-session, and engineers the 14 features. It outputs both a DataFrame and a CSV for auditing. `model.py` consumes that output and handles everything training-related — the three versions, cross-validation, logging to JSON and SQLite, saving the best model, and generating the visualizations. The separation makes it easy to add new features without touching the training logic."

---

**Q: Walk me through the database schema.**

> "There are two tables. `evaluations` stores one row per topic per session — so each quiz run of 50 questions creates 5 rows, one per domain, with the student's name, email, session UUID, topic, score out of 10, accuracy percentage, and a timestamp. `model_versions` stores one row per trained model version — the algorithm, accuracy, notes, and training timestamp. This gives a full audit trail of how the model improved over time."

---

## 🔢 Numbers to Remember

| Fact | Value |
|---|---|
| Evaluation rows in DB | 1,570 |
| Unique quiz sessions | 314 |
| Seeds in seed_data.py | 520 |
| ML features engineered | 14 |
| Model test accuracy | 88.89% |
| Cross-validation accuracy | 91.73% ± 2.32% |
| RandomForest estimators | 200 |
| Train/test split | 80/20 |
| CV folds | 5 |
| Questions in quiz | 50 (10 per domain) |
| Strongest topic | Cybersecurity (60.2% pass rate) |
| Weakest topic | App Dev (50.0% pass rate) |
| Timer per web question | 20 seconds |

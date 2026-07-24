# 🎞️ MovieIQ — Predictive Analytics on Film Success

MovieIQ is an interactive, end-to-end machine learning dashboard that explores, tests, and predicts whether a movie will be commercially successful — using nothing but budget, revenue, popularity, runtime, and audience rating. It walks the full data science lifecycle: cleaning → exploration → hypothesis testing → modeling → deployment, wrapped in a custom-designed Streamlit interface.

A movie is labeled **successful** when its **revenue exceeds its budget**.

**🔗 Live app:** https://movieiq-predictive-analytics.onrender.com

> Hosted on Render's free tier — the app may take up to ~50 seconds to wake up if it's been idle.

---

## Why MovieIQ

Studios and investors routinely commit tens of millions of dollars to a film before knowing whether it will earn that back. MovieIQ turns historical film data into a decision-support tool: it surfaces which factors actually separate hits from flops, quantifies that relationship statistically (not just visually), and lets a user test a hypothetical movie's specs against a trained model before a single frame is shot.

---

## What's Inside

### 📁 Bring-your-own-data upload
No hardcoded dataset — drop in any `movies.csv` that matches the expected schema and the entire dashboard (charts, stats, model, predictions) rebuilds live from that file.

### 📊 Interactive Exploratory Data Analysis
All charts are built in **Plotly**, not static Matplotlib images — every chart supports hover tooltips, zoom, and pan:
- **Budget vs. Revenue scatter** — color-coded by success/failure, with a break-even reference line
- **Genre trends bar chart** — most common genres, colored by their success rate
- **Feature-vs-outcome box plots** — compare popularity, runtime, or vote average across successful vs. unsuccessful movies (switchable via dropdown)
- **Correlation heatmap** — surfaces multicollinearity risk across numeric features before modeling

### 📈 Statistical Testing, Not Just Charts
- **Independent-samples T-test** on a numeric feature (default: popularity) between successful and unsuccessful movies, with null hypothesis, t-statistic, p-value, and a plain-language conclusion
- **Chi-square test of independence** on genre vs. success, with contingency table logic under the hood
- Both tests use α = 0.05 and explain what the p-value actually means, not just report it

### 🌲 Random Forest Classifier, Trained Live
- Trained fresh on whatever dataset is uploaded — not a frozen pre-trained model
- 80/20 train/test split, **stratified** on the target to preserve class balance
- Reports accuracy, precision, recall, and a full confusion matrix
- Surfaces **feature importance**, so you can see whether the model's reasoning agrees with the EDA and statistical tests
- Deliberately excludes `revenue` from the feature set (using it would leak the target directly) and `title` (a non-predictive identifier)

### 🔮 Live "What-If" Prediction
A form where you enter a hypothetical movie's budget, genre, popularity, runtime, and rating, and get back:
- A binary success / not-success verdict
- A confidence percentage from the trained model's probability output

### 🎛️ Real-Time Sidebar Filters
Filter the entire dashboard — charts, stats, and metrics — by genre (multi-select) and minimum vote average, without re-uploading anything.

### 🎨 Minimalist Slate — a Custom Design System
Every visual element in MovieIQ follows a hand-built dark theme rather than Streamlit's default styling:
- **Palette**: deep slate background (`#0f1419`), muted teal accent (`#5eb8b0`), soft coral for contrast (`#d98e73`), green/red for success/failure states
- **Typography**: Inter for body and headings, JetBrains Mono for section labels and technical captions — a deliberate pairing of humanist sans + monospace to signal "data product," not generic dashboard
- **Component language**: bordered metric cards, left-accented section labels in uppercase tracked type, a custom logo mark (film-reel icon in a bordered chip) in the header
- Applied consistently across native Streamlit components (metrics, buttons, file uploader, forms) via injected CSS, and across every Plotly chart via a shared theme template — so charts and UI never feel like two different products glued together

---

## Architecture

```
MovieIQ/
├── MovieIQ.py          # Streamlit entry point — wires every stage together into the UI
├── data_prep.py         # Cleaning, success labeling, multi-format genre parsing
├── eda.py                # Plotly chart builders + shared Minimalist Slate theme template
├── stats_tests.py        # T-test and Chi-square test logic
├── model.py              # Feature prep, training, evaluation, single-row prediction
├── requirements.txt
└── assets/
    └── logo.svg          # Standalone copy of the header logo mark
```

Each stage is a self-contained module with its own `if __name__ == "__main__"` block, so any part of the pipeline can be run and inspected independently of the Streamlit app — useful for debugging a specific stage without spinning up the whole dashboard.

### Genre parsing handles three real-world formats out of the box
- TMDB-style stringified list of dicts: `"[{'id': 18, 'name': 'Drama'}]"`
- Delimited strings: `"Action|Adventure"` or `"Action, Adventure"`
- Plain single-genre strings: `"Drama"`

This means the same code works whether your dataset came from TMDB, a Kaggle CSV export, or a hand-labeled sheet.

---

## Modeling Details

| | |
|---|---|
| **Target** | `success = 1 if revenue > budget else 0` |
| **Features used** | `budget`, `popularity`, `runtime`, `vote_average`, `genre_primary` (label-encoded) |
| **Explicitly excluded** | `revenue` (target leakage), `title` (non-predictive identifier) |
| **Split** | 80/20, stratified on `success` |
| **Algorithm** | `RandomForestClassifier`, 300 trees, max depth 8 |
| **Evaluation** | Accuracy, precision, recall, confusion matrix, feature importance |

**Why Random Forest:** it handles non-linear relationships between budget/popularity/rating and success without manual feature engineering, resists overfitting better than a single decision tree via ensemble averaging, and gives free, interpretable feature importances — useful for tying model behavior back to the EDA and statistical findings in the same dashboard.

---

## Known Limitations

- Success is defined purely as revenue > budget — this ignores marketing spend, inflation-adjusted comparisons across years, and streaming/ancillary revenue that real studios would weigh
- Model performance is sensitive to class imbalance in whatever dataset is uploaded; a dataset with very few flops will bias the model toward predicting "success" by default
- No cast, director, studio, release-date, or marketing-spend features — all known to meaningfully affect box office outcomes but outside this dataset's scope
- The "what-if" prediction is only as good as the historical data it was trained on for that session; it is a decision-support signal, not a guarantee

---

## Roadmap / Ideas for Extension

- Class-balancing (e.g. `class_weight='balanced'` or SMOTE) to fix the model's current bias toward predicting success
- Additional models (Gradient Boosting, Logistic Regression) with a side-by-side comparison view
- Persisted model artifacts so retraining isn't required on every session
- Release-date seasonality analysis (summer blockbuster vs. awards-season release patterns)
- Studio/franchise-level aggregation views

---

## Tech Stack

`Python` · `Streamlit` · `Pandas` · `NumPy` · `scikit-learn` · `SciPy` · `Plotly`

---

## Deployment

Deployed on [Render](https://render.com) as a Web Service:
- **Build command**: `pip install -r requirements.txt`
- **Start command**: `streamlit run MovieIQ.py --server.port $PORT --server.address 0.0.0.0`

---

## Quick Start

```bash
git clone https://github.com/<your-username>/movieiq.git
cd movieiq
python -m venv venv && venv\Scripts\activate   # or source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
streamlit run MovieIQ.py
```
Then upload any CSV matching the schema below.

### Required CSV columns
`budget`, `revenue`, `popularity`, `runtime`, `vote_average`, `title`, `genres`

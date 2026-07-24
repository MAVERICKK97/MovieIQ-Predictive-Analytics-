# 🎞️ MovieIQ — Predictive Analytics on Film Success

MovieIQ is an interactive dashboard that analyses and predicts whether a movie will be commercially successful, based on its budget, revenue, popularity, runtime, and audience rating. A movie is labeled **successful** when its **revenue exceeds its budget**.

**🔗 Live app:** https://movieiq-predictive-analytics.onrender.com

> Note: hosted on Render's free tier — the app may take up to ~50 seconds to wake up if it's been idle.

---

## Features

- 📁 **CSV upload** — bring your own `movies.csv`, no setup required
- 📊 **Interactive EDA** — Plotly charts (budget vs. revenue, genre trends, feature comparisons, correlation heatmap) with hover tooltips
- 📈 **Statistical testing** — T-test and Chi-square test with hypotheses, p-values, and conclusions
- 🌲 **Random Forest model** — trained live on your uploaded data, with accuracy/precision/recall, confusion matrix, and feature importance
- 🔮 **Live prediction** — enter a movie's details and get a success / not-success prediction with confidence score
- 🎛️ **Sidebar filters** — filter the whole dashboard by genre and minimum vote average
- 🎨 **Minimalist Slate design** — custom dark theme, built from scratch

---

## Project Structure

```
MovieIQ/
├── MovieIQ.py          # Streamlit app (entry point)
├── data_prep.py         # Stage 1 — data cleaning & success labeling
├── eda.py                # Stage 2 — Plotly chart builders
├── stats_tests.py        # Stage 3 — T-test & Chi-square test
├── model.py              # Stage 4 — Random Forest training & prediction
├── requirements.txt
└── assets/
    └── logo.svg
```

---

## Dataset Requirements

Your CSV must contain these columns:

| Column | Description |
|---|---|
| `budget` | Production budget (numeric) |
| `revenue` | Box office revenue (numeric) |
| `popularity` | Popularity score (numeric) |
| `runtime` | Runtime in minutes (numeric) |
| `vote_average` | Average audience rating, 0–10 (numeric) |
| `title` | Movie title (text) |
| `genres` | Genre(s) — supports pipe/comma-separated strings (`Action\|Drama`) or TMDB-style stringified lists (`[{'id': 18, 'name': 'Drama'}]`) |

Rows with a missing or zero-value `budget`/`revenue` are dropped during cleaning, since a 0 typically means the value was never reported.

---

## Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/<your-username>/movieiq.git
cd movieiq
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run MovieIQ.py
```
Open `http://localhost:8501` and upload your `movies.csv` to get started.

---

## Running the Pipeline Stages Individually

Each stage can also be run standalone from the command line for exploration or debugging:

```bash
python data_prep.py      # prints shape, summary stats, class balance
python eda.py             # saves interactive charts to assets/
python stats_tests.py     # prints T-test & Chi-square results
python model.py           # trains the model, prints evaluation metrics
```

---

## Modeling Notes

- **Target**: `success = 1 if revenue > budget else 0`
- **Features used**: `budget`, `popularity`, `runtime`, `vote_average`, `genre_primary` (encoded)
- **Excluded**: `revenue` (would leak the target directly), `title` (not predictive)
- **Split**: 80/20 train/test, stratified on `success`
- **Model**: `RandomForestClassifier` (300 trees, max depth 8)

---

## Limitations

- Success is defined purely as revenue > budget — it ignores marketing spend, inflation, and ancillary/streaming revenue
- Performance depends heavily on class balance in the uploaded dataset; highly imbalanced data can bias the model toward the majority class
- Genre, cast, director, and release timing effects beyond what's in the dataset are not modeled

---

## Tech Stack

`Python` · `Streamlit` · `Pandas` · `NumPy` · `scikit-learn` · `SciPy` · `Plotly`

---

## Deployment

Deployed on [Render](https://render.com) as a Web Service:
- **Build command**: `pip install -r requirements.txt`
- **Start command**: `streamlit run MovieIQ.py --server.port $PORT --server.address 0.0.0.0`

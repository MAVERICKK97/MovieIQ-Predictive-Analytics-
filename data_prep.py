"""
Stage 1 — Data Preparation
Loads movies.csv, cleans it, and creates the success target column.
"""

import pandas as pd
import numpy as np
import ast


REQUIRED_COLS = ["budget", "revenue", "popularity", "runtime", "vote_average", "title", "genres"]


def load_raw_data(source="movies.csv") -> pd.DataFrame:
    """
    Load the raw dataset and report basic shape/summary info.
    `source` can be a file path (str) OR a file-like object
    (e.g. a Streamlit UploadedFile from st.file_uploader).
    """
    df = pd.read_csv(source)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(df[["budget", "revenue", "popularity", "runtime", "vote_average"]].describe())
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and zeros in budget/revenue.

    A budget or revenue of 0 almost always means the value was never
    reported (not that the movie truly cost/earned nothing). Keeping
    these rows would corrupt the success label (revenue > budget),
    so we drop rows where either field is missing or exactly 0.
    """
    df = df.copy()

    # Drop rows with missing critical fields
    df = df.dropna(subset=["budget", "revenue", "popularity", "runtime", "vote_average"])

    # Drop rows where budget or revenue is zero (unreported, not real)
    before = len(df)
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)]
    removed = before - len(df)
    print(f"Removed {removed} rows with zero/missing budget or revenue")

    return df


def add_success_label(df: pd.DataFrame) -> pd.DataFrame:
    """Create the binary target: success = 1 if revenue > budget else 0."""
    df = df.copy()
    df["success"] = (df["revenue"] > df["budget"]).astype(int)

    balance = df["success"].value_counts(normalize=True)
    print("Class balance:\n", balance)
    return df


def process_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the genres column into a clean list of genre-name strings per row.

    Supports three common formats:
      1. TMDB-style JSON list of dicts: "[{'id': 10749, 'name': 'Romance'}, ...]"
      2. Delimited strings: "Action|Adventure", "Action, Adventure"
      3. Plain single genre: "Action"

    Also produces a genre_primary column (first genre) for simple filtering.
    """
    import ast
    import re

    df = df.copy()

    def split_genres(val):
        if pd.isna(val):
            return []
        val = str(val).strip()
        if not val:
            return []

        # Case 1: TMDB-style JSON/dict-list string, e.g. "[{'id': 10749, 'name': 'Romance'}]"
        if val.startswith("[") and "name" in val:
            try:
                parsed = ast.literal_eval(val)
                return [d["name"] for d in parsed if isinstance(d, dict) and "name" in d]
            except (ValueError, SyntaxError):
                return re.findall(r"'name':\s*'([^']+)'", val)

        # Case 2: delimited strings
        for sep in ["|", ",", ";"]:
            if sep in val:
                return [g.strip() for g in val.split(sep) if g.strip()]

        # Case 3: single plain genre
        return [val]

    df["genres_list"] = df["genres"].apply(split_genres)
    df["genre_primary"] = df["genres_list"].apply(lambda g: g[0] if g else "Unknown")
    return df


def full_pipeline(source="movies.csv") -> pd.DataFrame:
    """Run the full Stage 1 pipeline end to end. `source` = path or uploaded file object."""
    df = load_raw_data(source)
    df = clean_data(df)
    df = add_success_label(df)
    df = process_genres(df)
    return df


if __name__ == "__main__":
    data = full_pipeline("movies.csv")
    print(data.head())
    data.to_csv("movies_clean.csv", index=False)
    print("Saved cleaned dataset to movies_clean.csv")

"""
Stage 4 — Predictive Modeling (Random Forest)
Trains a classifier to predict movie success from budget-independent features.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Note: revenue is excluded because success is *derived* from revenue —
# including it would leak the target directly into the features.
# title is excluded because it's a free-text identifier, not a predictive signal.
FEATURES = ["budget", "popularity", "runtime", "vote_average", "genre_primary"]
TARGET = "success"


def prepare_features(df: pd.DataFrame):
    """Encode categorical genre_primary and split into X, y."""
    df = df.copy()
    le = LabelEncoder()
    df["genre_encoded"] = le.fit_transform(df["genre_primary"])

    X = df[["budget", "popularity", "runtime", "vote_average", "genre_encoded"]]
    y = df[TARGET]
    return X, y, le


def train_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split 80/20 (train/test). A held-out test set is essential because it
    lets us measure how the model performs on data it has never seen,
    which is the only honest estimate of real-world accuracy.
    """
    X, y, label_encoder = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # A random forest builds many decision trees on random subsets of the
    # data/features, and each tree "votes" on the outcome; the majority
    # vote becomes the prediction. This reduces overfitting versus a
    # single decision tree.
    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

    return clf, label_encoder, metrics, importances, (X_test, y_test, y_pred)


def predict_single(clf, label_encoder, budget, popularity, runtime, vote_average, genre):
    """Predict success for a single user-entered movie."""
    try:
        genre_encoded = label_encoder.transform([genre])[0]
    except ValueError:
        # unseen genre label -> fall back to most common encoded class
        genre_encoded = 0

    X_new = pd.DataFrame([{
        "budget": budget,
        "popularity": popularity,
        "runtime": runtime,
        "vote_average": vote_average,
        "genre_encoded": genre_encoded,
    }])
    pred = clf.predict(X_new)[0]
    proba = clf.predict_proba(X_new)[0][1]
    return pred, proba


if __name__ == "__main__":
    from data_prep import full_pipeline

    df = full_pipeline("movies.csv")
    clf, le, metrics, importances, _ = train_model(df)

    print("Accuracy: ", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:   ", metrics["recall"])
    print("Confusion matrix:\n", metrics["confusion_matrix"])
    print("\nFeature importances:\n", importances)

    joblib.dump(clf, "rf_model.joblib")
    joblib.dump(le, "genre_encoder.joblib")
    print("\nSaved model to rf_model.joblib")

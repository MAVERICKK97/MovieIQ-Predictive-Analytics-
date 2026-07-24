"""
Stage 2 — Exploratory Data Analysis
All charts are built with Plotly so they are interactive (hover tooltips,
zoom, pan) both when saved as HTML and when embedded in the Streamlit app.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# "Minimalist Slate" palette — used consistently across every chart
SLATE = {
    "bg": "#0f1419",
    "panel": "#1a2028",
    "grid": "#2a323c",
    "text": "#e7ebef",
    "muted": "#8a94a3",
    "accent": "#5eb8b0",     # muted teal
    "accent2": "#d98e73",    # muted coral
    "success": "#6fbf8c",
    "fail": "#c96b6b",
}

TEMPLATE = go.layout.Template()
TEMPLATE.layout = go.Layout(
    paper_bgcolor=SLATE["bg"],
    plot_bgcolor=SLATE["bg"],
    font=dict(color=SLATE["text"], family="Inter, -apple-system, sans-serif", size=13),
    xaxis=dict(gridcolor=SLATE["grid"], zerolinecolor=SLATE["grid"]),
    yaxis=dict(gridcolor=SLATE["grid"], zerolinecolor=SLATE["grid"]),
    colorway=[SLATE["accent"], SLATE["accent2"], SLATE["success"], SLATE["fail"], "#7a8fc9", "#c9a86a"],
    margin=dict(l=40, r=20, t=50, b=40),
)


def budget_vs_revenue(df: pd.DataFrame):
    """Scatter plot: Budget vs Revenue, colored by success, with hover tooltips."""
    fig = px.scatter(
        df, x="budget", y="revenue", color=df["success"].map({1: "Success", 0: "Not Successful"}),
        hover_data=["title", "vote_average", "popularity"],
        color_discrete_map={"Success": SLATE["success"], "Not Successful": SLATE["fail"]},
        title="Budget vs. Revenue",
        labels={"budget": "Budget ($)", "revenue": "Revenue ($)", "color": "Outcome"},
        template=TEMPLATE,
        opacity=0.7,
    )
    # reference line where revenue == budget (break-even)
    max_val = max(df["budget"].max(), df["revenue"].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val], mode="lines",
        line=dict(color=SLATE["muted"], dash="dash"), name="Break-even line"
    ))
    return fig


def genre_trends(df: pd.DataFrame):
    """Bar chart: most common genres and their success rate."""
    exploded = df.explode("genres_list")
    genre_stats = (
        exploded.groupby("genres_list")["success"]
        .agg(count="count", success_rate="mean")
        .sort_values("count", ascending=False)
        .head(15)
        .reset_index()
    )
    fig = px.bar(
        genre_stats, x="genres_list", y="count", color="success_rate",
        color_continuous_scale=[SLATE["fail"], SLATE["muted"], SLATE["success"]],
        title="Most Common Genres & Their Success Rate",
        labels={"genres_list": "Genre", "count": "Number of Movies", "success_rate": "Success Rate"},
        hover_data={"success_rate": ":.1%"},
        template=TEMPLATE,
    )
    return fig


def feature_vs_success(df: pd.DataFrame, feature: str):
    """Box plot comparing a numeric feature across success vs failure."""
    plot_df = df.copy()
    plot_df["Outcome"] = plot_df["success"].map({1: "Success", 0: "Not Successful"})
    fig = px.box(
        plot_df, x="Outcome", y=feature, color="Outcome",
        color_discrete_map={"Success": SLATE["success"], "Not Successful": SLATE["fail"]},
        points="outliers",
        title=f"{feature.replace('_', ' ').title()} by Outcome",
        template=TEMPLATE,
    )
    return fig


def correlation_heatmap(df: pd.DataFrame):
    """Correlation heatmap of numeric features."""
    numeric_cols = ["budget", "revenue", "popularity", "runtime", "vote_average"]
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale=["#c96b6b", "#1a2028", "#5eb8b0"],
        title="Correlation Heatmap of Numeric Features",
        template=TEMPLATE, aspect="auto",
    )
    return fig


if __name__ == "__main__":
    from data_prep import full_pipeline

    df = full_pipeline("movies.csv")
    budget_vs_revenue(df).write_html("assets/budget_vs_revenue.html")
    genre_trends(df).write_html("assets/genre_trends.html")
    feature_vs_success(df, "popularity").write_html("assets/popularity_vs_success.html")
    correlation_heatmap(df).write_html("assets/correlation_heatmap.html")
    print("Saved interactive charts to assets/")

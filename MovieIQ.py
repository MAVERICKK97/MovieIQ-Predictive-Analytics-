"""
MovieIQ — Predictive Analytics on Film Success
Streamlit dashboard: "Minimalist Slate" design, interactive Plotly charts.

Run locally with:  streamlit run MovieIQ.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data_prep import full_pipeline
from eda import budget_vs_revenue, genre_trends, feature_vs_success, correlation_heatmap, SLATE, TEMPLATE
from stats_tests import run_ttest, run_chi_square
from model import train_model, predict_single

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# MINIMALIST SLATE THEME — custom CSS injected into the Streamlit app
# ----------------------------------------------------------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, sans-serif;
    }}

    .stApp {{
        background-color: {SLATE['bg']};
        color: {SLATE['text']};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {SLATE['panel']};
        border-right: 1px solid {SLATE['grid']};
    }}

    section[data-testid="stSidebar"] * {{
        color: {SLATE['text']} !important;
    }}

    /* Headline block */
    .mq-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        border-bottom: 1px solid {SLATE['grid']};
        padding-bottom: 18px;
        margin-bottom: 28px;
        flex-wrap: wrap;
    }}
    .mq-header .mq-subtitle {{
        margin-left: 4px;
    }}
    .mq-title {{
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: {SLATE['text']};
        margin: 0;
    }}
    .mq-subtitle {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: {SLATE['accent']};
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin: 0;
    }}

    /* Section labels */
    .mq-section {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {SLATE['muted']};
        border-left: 3px solid {SLATE['accent']};
        padding-left: 10px;
        margin: 34px 0 14px 0;
    }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background-color: {SLATE['panel']};
        border: 1px solid {SLATE['grid']};
        border-radius: 10px;
        padding: 16px 18px;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {SLATE['muted']} !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {SLATE['text']} !important;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {SLATE['accent']};
        color: #0f1419;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        transition: opacity 0.15s ease;
    }}
    .stButton > button:hover {{
        opacity: 0.85;
        color: #0f1419;
    }}

    /* Result banners */
    .mq-result-success {{
        background: rgba(111, 191, 140, 0.12);
        border: 1px solid {SLATE['success']};
        border-radius: 10px;
        padding: 18px 20px;
        color: {SLATE['success']};
        font-weight: 600;
        font-size: 1.05rem;
    }}
    .mq-result-fail {{
        background: rgba(201, 107, 107, 0.12);
        border: 1px solid {SLATE['fail']};
        border-radius: 10px;
        padding: 18px 20px;
        color: {SLATE['fail']};
        font-weight: 600;
        font-size: 1.05rem;
    }}

    /* Dataframe */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {SLATE['grid']};
        border-radius: 8px;
    }}

    hr {{
        border-color: {SLATE['grid']};
    }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# DATA LOADING (cached) — via CSV upload
# ----------------------------------------------------------------------------
@st.cache_data
def get_data(file_bytes):
    import io
    return full_pipeline(io.BytesIO(file_bytes))


@st.cache_resource
def get_model(df):
    clf, le, metrics, importances, _ = train_model(df)
    return clf, le, metrics, importances


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="mq-header">
    <div style="width:44px;height:44px;background:{SLATE['panel']};border:1px solid {SLATE['grid']};border-radius:10px;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
        <svg width="26" height="26" viewBox="0 0 40 40">
            <circle cx="20" cy="20" r="15" fill="none" stroke="{SLATE['accent']}" stroke-width="3"/>
            <circle cx="20" cy="5.5" r="3.6" fill="{SLATE['accent']}"/>
            <circle cx="34.5" cy="20" r="3.6" fill="{SLATE['accent']}"/>
            <circle cx="20" cy="34.5" r="3.6" fill="{SLATE['accent']}"/>
            <circle cx="5.5" cy="20" r="3.6" fill="{SLATE['accent']}"/>
        </svg>
    </div>
    <p class="mq-title">Movie<span style="color:{SLATE['accent']};">IQ</span></p>
    <p class="mq-subtitle">Predictive Analytics on Film Success</p>
</div>
""", unsafe_allow_html=True)

st.markdown("**Upload your movies CSV file**")
uploaded_file = st.file_uploader(
    "Upload your movies CSV file",
    type=["csv"],
    label_visibility="collapsed",
    help="Must contain: budget, revenue, popularity, runtime, vote_average, title, genres",
)

if uploaded_file is None:
    st.info("Upload a `movies.csv` file above to load the MovieIQ dashboard.")
    st.stop()

df = get_data(uploaded_file.getvalue())
clf, label_encoder, metrics, importances = get_model(df)

# ----------------------------------------------------------------------------
# SIDEBAR — FILTERS
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About this app")
    st.caption(
        "MovieIQ uses machine learning to predict whether a movie will be "
        "**successful** based on: Budget, Popularity, Runtime, Average Votes. "
        "Includes EDA, Stats, and Prediction."
    )
    st.markdown("---")
    st.markdown("### Filter Options")
    all_genres = sorted(set(g for genres in df["genres_list"] for g in genres))
    selected_genres = st.multiselect("Select Genre(s)", options=all_genres, default=[])
    min_vote = st.slider("Minimum vote average", 0.0, 10.0, 0.0, 0.1)

filtered_df = df.copy()
if selected_genres:
    filtered_df = filtered_df[filtered_df["genres_list"].apply(lambda g: any(x in g for x in selected_genres))]
filtered_df = filtered_df[filtered_df["vote_average"] >= min_vote]

# ----------------------------------------------------------------------------
# TOP METRICS
# ----------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Total Movies", f"{len(filtered_df):,}")
c2.metric("Success %", f"{filtered_df['success'].mean()*100:.1f}%")
c3.metric("Unique Genres", f"{len(all_genres)}")

# ----------------------------------------------------------------------------
# DATASET OVERVIEW TABLE
# ----------------------------------------------------------------------------
st.markdown('<p class="mq-section">Dataset Overview</p>', unsafe_allow_html=True)
st.dataframe(filtered_df.head(20), use_container_width=True)

# ----------------------------------------------------------------------------
# EDA SECTION
# ----------------------------------------------------------------------------
st.markdown('<p class="mq-section">Exploratory Data Analysis</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(budget_vs_revenue(filtered_df), use_container_width=True)
with col2:
    st.plotly_chart(genre_trends(filtered_df), use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    feature_choice = st.selectbox("Compare feature vs. outcome", ["popularity", "runtime", "vote_average"])
    st.plotly_chart(feature_vs_success(filtered_df, feature_choice), use_container_width=True)
with col4:
    st.plotly_chart(correlation_heatmap(filtered_df), use_container_width=True)

# ----------------------------------------------------------------------------
# STATISTICAL TESTS SECTION
# ----------------------------------------------------------------------------
st.markdown('<p class="mq-section">Statistical Testing</p>', unsafe_allow_html=True)

t_result = run_ttest(filtered_df, "popularity")
chi_result = run_chi_square(filtered_df, "genre_primary")

tcol1, tcol2 = st.columns(2)
with tcol1:
    st.markdown("**T-Test — popularity vs. success**")
    st.write(f"H0: mean popularity is equal for successful and unsuccessful movies.")
    st.write(f"p-value: `{t_result['p_value']:.4f}`")
    st.write(t_result["conclusion"])
with tcol2:
    st.markdown("**Chi-Square — genre vs. success**")
    st.write("H0: genre and success are independent.")
    st.write(f"p-value: `{chi_result['p_value']:.4f}`")
    st.write(chi_result["conclusion"])

# ----------------------------------------------------------------------------
# MODEL PERFORMANCE SECTION
# ----------------------------------------------------------------------------
st.markdown('<p class="mq-section">Model Performance — Random Forest</p>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
m2.metric("Precision", f"{metrics['precision']*100:.1f}%")
m3.metric("Recall", f"{metrics['recall']*100:.1f}%")

cm = metrics["confusion_matrix"]
cm_fig = go.Figure(data=go.Heatmap(
    z=cm, x=["Predicted: Fail", "Predicted: Success"], y=["Actual: Fail", "Actual: Success"],
    colorscale=[[0, SLATE["panel"]], [1, SLATE["accent"]]], text=cm, texttemplate="%{text}",
))
cm_fig.update_layout(template=TEMPLATE, title="Confusion Matrix")
imp_fig = go.Figure(go.Bar(
    x=importances.values, y=importances.index, orientation="h", marker_color=SLATE["accent"],
))
imp_fig.update_layout(template=TEMPLATE, title="Feature Importance")

mc1, mc2 = st.columns(2)
with mc1:
    st.plotly_chart(cm_fig, use_container_width=True)
with mc2:
    st.plotly_chart(imp_fig, use_container_width=True)

# ----------------------------------------------------------------------------
# PREDICTION SECTION
# ----------------------------------------------------------------------------
st.markdown('<p class="mq-section">Predict a New Movie</p>', unsafe_allow_html=True)

with st.form("prediction_form"):
    p1, p2, p3 = st.columns(3)
    with p1:
        in_budget = st.number_input("Budget ($)", min_value=1000, value=20_000_000, step=1_000_000)
        in_genre = st.selectbox("Primary genre", options=sorted(df["genre_primary"].unique()))
    with p2:
        in_popularity = st.number_input("Popularity score", min_value=0.0, value=15.0, step=0.5)
        in_runtime = st.number_input("Runtime (minutes)", min_value=1, value=110, step=1)
    with p3:
        in_vote = st.slider("Average vote (0-10)", 0.0, 10.0, 6.5, 0.1)

    submitted = st.form_submit_button("Predict success")

if submitted:
    pred, proba = predict_single(clf, label_encoder, in_budget, in_popularity, in_runtime, in_vote, in_genre)
    if pred == 1:
        st.markdown(
            f'<div class="mq-result-success">Predicted: SUCCESS &nbsp;·&nbsp; confidence {proba*100:.1f}%</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="mq-result-fail">Predicted: NOT SUCCESSFUL &nbsp;·&nbsp; confidence {(1-proba)*100:.1f}%</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")
st.caption("MovieIQ · Built with Streamlit, Plotly & scikit-learn · Minimalist Slate theme")

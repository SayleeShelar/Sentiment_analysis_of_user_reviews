"""
app.py  –  Sentiment Analysis Web Interface
Run: streamlit run app.py
"""

import os, re, joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 2rem;
        color: white; text-align: center;
    }
    .main-header h1 { font-size: 2.6rem; font-weight: 700; margin: 0; }
    .main-header p  { font-size: 1.1rem; opacity: 0.9; margin: 0.4rem 0 0; }

    .metric-card {
        background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08); text-align: center;
        border-left: 4px solid #667eea;
    }
    .metric-card .val { font-size: 2rem; font-weight: 700; color: #667eea; }
    .metric-card .lbl { font-size: 0.85rem; color: #888; margin-top: 0.2rem; }

    .result-positive {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745; border-radius: 14px; padding: 1.5rem;
        text-align: center;
    }
    .result-negative {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545; border-radius: 14px; padding: 1.5rem;
        text-align: center;
    }
    .result-positive h2, .result-negative h2 { margin: 0; font-size: 1.8rem; }

    .step-badge {
        display: inline-block; background: #667eea; color: white;
        border-radius: 50%; width: 28px; height: 28px; line-height: 28px;
        text-align: center; font-weight: 700; font-size: 0.85rem;
        margin-right: 8px;
    }
    .section-title { font-size: 1.3rem; font-weight: 600; margin: 1.5rem 0 0.8rem; }

    div[data-testid="stTabs"] button { font-size: 1rem; font-weight: 500; }
    .stTextArea textarea { font-size: 1rem; border-radius: 10px; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 10px;
        padding: 0.6rem 2rem; font-size: 1rem; font-weight: 600;
        width: 100%; transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ── paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
MODEL  = os.path.join(BASE, "model")
ASSETS = os.path.join(BASE, "assets")
DATA   = os.path.join(BASE, "data")

# ── helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    vec  = joblib.load(os.path.join(MODEL, "tfidf_vectorizer.joblib"))
    clf  = joblib.load(os.path.join(MODEL, "logistic_model.joblib"))
    meta = joblib.load(os.path.join(MODEL, "meta.joblib"))
    return vec, clf, meta

@st.cache_resource
def get_nlp():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet",   quiet=True)
    nltk.download("omw-1.4",   quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()

def clean_text(text: str, stop, lem) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(lem.lemmatize(w) for w in text.split()
                    if w not in stop and len(w) > 2)

def asset(name):
    p = os.path.join(ASSETS, name)
    return Image.open(p) if os.path.exists(p) else None

def model_ready():
    return all(os.path.exists(os.path.join(MODEL, f))
               for f in ["tfidf_vectorizer.joblib", "logistic_model.joblib", "meta.joblib"])

# ── header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🎭 Sentiment Analysis of User Reviews</h1>
  <p>End-to-end NLP pipeline · IMDb 50 000 reviews · TF-IDF + Logistic Regression</p>
</div>
""", unsafe_allow_html=True)

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/sentiment-analysis.png", width=80)
    st.markdown("## 🗂️ Navigation")
    st.markdown("""
    Use the **tabs** on the right to explore:
    - 📊 Data Overview
    - 🧹 Data Cleaning
    - 📈 Visualisations
    - 🤖 Model & Metrics
    - 🔮 Live Prediction
    """)
    st.divider()
    if model_ready():
        _, _, meta = load_model()
        st.success("✅ Model loaded")
        st.metric("Accuracy",  f"{meta['accuracy']:.2%}")
        st.metric("ROC-AUC",   f"{meta['roc_auc']:.4f}")
        st.metric("Train size", f"{meta['train_size']:,}")
    else:
        st.error("⚠️ Model not found.\nRun `python train_pipeline.py` first.")

# ── tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs(["📊 Data Overview", "🧹 Data Cleaning",
                "📈 Visualisations", "🤖 Model & Metrics", "🔮 Live Prediction"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<p class="section-title"><span class="step-badge">1</span>Dataset Overview</p>',
                unsafe_allow_html=True)

    st.markdown("""
    **Dataset:** [Stanford IMDb Large Movie Review Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)

    | Property | Value |
    |---|---|
    | Total reviews | 50 000 |
    | Positive reviews | 25 000 |
    | Negative reviews | 25 000 |
    | Source | IMDb movie reviews |
    | Task | Binary sentiment classification |
    | Labels | `positive` / `negative` |
    """)

    clean_csv = os.path.join(DATA, "reviews_clean.csv")
    if os.path.exists(clean_csv):
        df = pd.read_csv(clean_csv)
        if "word_count" not in df.columns:
            df["word_count"] = df["clean_text"].apply(lambda x: len(str(x).split()))
        st.markdown("#### Sample rows from cleaned dataset")
        st.dataframe(
            df[["text", "sentiment", "clean_text", "word_count"]].sample(8, random_state=7),
            use_container_width=True, height=280
        )

        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in [
            (c1, f"{len(df):,}",                        "Total Reviews"),
            (c2, f"{df['word_count'].mean():.0f}",      "Avg Word Count"),
            (c3, f"{df['word_count'].max():,}",         "Max Word Count"),
            (c4, f"{df['sentiment'].value_counts()['positive']:,}", "Positive Reviews"),
        ]:
            col.markdown(f'<div class="metric-card"><div class="val">{val}</div>'
                         f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
    else:
        st.info("Run `python train_pipeline.py` to generate the dataset.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p class="section-title"><span class="step-badge">2</span>Data Cleaning Pipeline</p>',
                unsafe_allow_html=True)

    steps = [
        ("Lowercasing",          "Convert all text to lowercase to ensure uniformity."),
        ("HTML Tag Removal",     "Strip `<br />`, `<b>`, etc. using regex `<[^>]+>`."),
        ("Special Char Removal", "Remove punctuation, numbers, symbols — keep only letters."),
        ("Whitespace Normalisation", "Collapse multiple spaces into one."),
        ("Stopword Removal",     "Remove common English stopwords (NLTK list, 179 words)."),
        ("Lemmatisation",        "Reduce words to base form: *running → run*, *better → good*."),
        ("Short Token Filter",   "Drop tokens with length ≤ 2 (noise removal)."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        with st.expander(f"Step {i} — {title}", expanded=(i <= 3)):
            st.markdown(desc)

    st.markdown("#### Before vs After Cleaning")
    clean_csv = os.path.join(DATA, "reviews_clean.csv")
    if os.path.exists(clean_csv):
        df = pd.read_csv(clean_csv)
        sample = df.sample(3, random_state=99)[["text", "clean_text", "sentiment"]]
        for _, row in sample.iterrows():
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Raw text:**")
                st.text_area("", row["text"][:400], height=100, key=f"raw_{_}",
                             label_visibility="collapsed")
            with col2:
                st.markdown("**Cleaned text:**")
                st.text_area("", row["clean_text"][:400], height=100, key=f"clean_{_}",
                             label_visibility="collapsed")
            st.caption(f"Sentiment: **{row['sentiment']}**")
            st.divider()
    else:
        st.info("Run `python train_pipeline.py` to see examples.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="section-title"><span class="step-badge">3</span>Exploratory Data Analysis</p>',
                unsafe_allow_html=True)

    viz_items = [
        ("class_dist.png",        "Class Distribution",              "Balanced dataset — 25k positive, 25k negative."),
        ("length_dist.png",       "Review Length Distribution",      "Positive reviews tend to be slightly longer."),
        ("wordcloud_positive.png","Word Cloud — Positive Reviews",   "Dominant words: *great, love, best, wonderful*."),
        ("wordcloud_negative.png","Word Cloud — Negative Reviews",   "Dominant words: *bad, worst, waste, boring*."),
        ("top_words.png",         "Top 20 Words per Class",          "Clear lexical separation between classes."),
    ]

    for fname, title, caption in viz_items:
        img = asset(fname)
        if img:
            st.markdown(f"#### {title}")
            st.image(img, use_container_width=True)
            st.caption(caption)
            st.divider()
        else:
            st.warning(f"Image `{fname}` not found — run the pipeline first.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – MODEL & METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p class="section-title"><span class="step-badge">4</span>Model Training & Evaluation</p>',
                unsafe_allow_html=True)

    st.markdown("""
    #### Model Architecture
    | Component | Details |
    |---|---|
    | Feature Extraction | TF-IDF (unigrams + bigrams, 50 000 features, sublinear TF) |
    | Classifier | Logistic Regression (C=5, lbfgs solver, max_iter=1000) |
    | Train/Test Split | 80% / 20% stratified |
    | Training samples | ~40 000 |
    | Test samples | ~10 000 |
    """)

    if model_ready():
        _, _, meta = load_model()

        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in [
            (c1, f"{meta['accuracy']:.2%}",   "Accuracy"),
            (c2, f"{meta['f1_positive']:.4f}", "F1 — Positive"),
            (c3, f"{meta['f1_negative']:.4f}", "F1 — Negative"),
            (c4, f"{meta['roc_auc']:.4f}",     "ROC-AUC"),
        ]:
            col.markdown(f'<div class="metric-card"><div class="val">{val}</div>'
                         f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            img = asset("confusion_matrix.png")
            if img:
                st.markdown("#### Confusion Matrix")
                st.image(img, use_container_width=True)
        with col2:
            img = asset("roc_curve.png")
            if img:
                st.markdown("#### ROC Curve")
                st.image(img, use_container_width=True)

        # classification report table
        report_csv = os.path.join(ASSETS, "classification_report.csv")
        if os.path.exists(report_csv):
            st.markdown("#### Classification Report")
            rdf = pd.read_csv(report_csv, index_col=0)
            st.dataframe(rdf.style.format("{:.4f}").background_gradient(cmap="Blues"),
                         use_container_width=True)
    else:
        st.info("Run `python train_pipeline.py` to train and evaluate the model.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<p class="section-title"><span class="step-badge">5</span>Live Sentiment Prediction</p>',
                unsafe_allow_html=True)

    if not model_ready():
        st.error("Model not found. Run `python train_pipeline.py` first.")
        st.stop()

    vec, clf, meta = load_model()
    stop, lem = get_nlp()

    # ── single review ──────────────────────────────────────────────────────────
    st.markdown("#### 🔍 Analyse a Single Review")
    user_input = st.text_area(
        "Paste your review here:",
        placeholder="e.g. This movie was absolutely fantastic! The acting was superb and the story kept me hooked throughout.",
        height=130,
    )

    if st.button("Analyse Sentiment"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            cleaned = clean_text(user_input, stop, lem)
            vec_input = vec.transform([cleaned])
            pred  = clf.predict(vec_input)[0]
            proba = clf.predict_proba(vec_input)[0]
            pos_p, neg_p = proba[list(clf.classes_).index("positive")], \
                           proba[list(clf.classes_).index("negative")]

            emoji = "😊" if pred == "positive" else "😞"
            css   = "result-positive" if pred == "positive" else "result-negative"
            color = "#28a745" if pred == "positive" else "#dc3545"

            st.markdown(f"""
            <div class="{css}">
              <h2>{emoji} {pred.upper()}</h2>
              <p style="margin:0.4rem 0 0; font-size:1rem;">
                Confidence: <strong>{max(pos_p, neg_p):.1%}</strong>
              </p>
            </div>
            """, unsafe_allow_html=True)

            # gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pos_p * 100,
                title={"text": "Positive Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": color},
                    "steps": [
                        {"range": [0,  40], "color": "#fde8e8"},
                        {"range": [40, 60], "color": "#fff3cd"},
                        {"range": [60, 100],"color": "#d4edda"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
                },
                number={"suffix": "%", "font": {"size": 28}},
            ))
            fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("🔬 Cleaned text used for prediction"):
                st.code(cleaned)

    st.divider()

    # ── batch prediction ───────────────────────────────────────────────────────
    st.markdown("#### 📋 Batch Prediction")
    st.markdown("Enter multiple reviews (one per line):")

    batch_input = st.text_area(
        "Batch reviews:",
        placeholder="The film was a masterpiece.\nTerrible acting, waste of time.\nDecent movie, nothing special.",
        height=160,
        label_visibility="collapsed",
    )

    if st.button("Analyse Batch"):
        lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
        if not lines:
            st.warning("Please enter at least one review.")
        else:
            cleaned_lines = [clean_text(l, stop, lem) for l in lines]
            vecs   = vec.transform(cleaned_lines)
            preds  = clf.predict(vecs)
            probas = clf.predict_proba(vecs)
            pos_idx = list(clf.classes_).index("positive")

            results = pd.DataFrame({
                "Review":     lines,
                "Sentiment":  preds,
                "Confidence": [f"{max(p):.1%}" for p in probas],
                "Pos Prob":   [f"{p[pos_idx]:.1%}" for p in probas],
            })

            def highlight(row):
                color = "#d4edda" if row["Sentiment"] == "positive" else "#f8d7da"
                return [f"background-color: {color}"] * len(row)

            st.dataframe(results.style.apply(highlight, axis=1),
                         use_container_width=True)

            # mini bar chart
            counts = pd.Series(preds).value_counts()
            fig = px.bar(
                x=counts.index, y=counts.values,
                color=counts.index,
                color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c"},
                labels={"x": "Sentiment", "y": "Count"},
                title="Batch Results Summary",
            )
            fig.update_layout(showlegend=False, height=300,
                              margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

# ── footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.85rem;'>"
    "Sentiment Analysis · IMDb Dataset · TF-IDF + Logistic Regression · Built with Streamlit"
    "</p>",
    unsafe_allow_html=True,
)

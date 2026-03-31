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
    - 📂 CSV Analyser
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
                "📈 Visualisations", "🤖 Model & Metrics", "🔮 Live Prediction", "📂 CSV Analyser"])

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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 – CSV ANALYSER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<p class="section-title"><span class="step-badge">6</span>CSV Bulk Analyser</p>',
                unsafe_allow_html=True)

    st.markdown("""
    Upload any CSV file that contains a column of reviews or feedback text.
    The model will automatically analyse every row and give you a full sentiment dashboard.
    """)

    if not model_ready():
        st.error("Model not found. Run `python train_pipeline.py` first.")
    else:
        vec, clf, meta = load_model()
        stop, lem = get_nlp()
        pos_idx = list(clf.classes_).index("positive")

        uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded:
            try:
                try:
                    udf = pd.read_csv(uploaded)
                except UnicodeDecodeError:
                    uploaded.seek(0)
                    udf = pd.read_csv(uploaded, encoding="latin-1")
                st.success(f"File loaded: {len(udf):,} rows, {len(udf.columns)} columns")

                # let user pick the text column
                text_col = st.selectbox(
                    "Select the column that contains the review/feedback text:",
                    options=udf.columns.tolist()
                )

                if st.button("Analyse CSV"):
                    reviews = udf[text_col].astype(str).tolist()

                    with st.spinner(f"Analysing {len(reviews):,} reviews..."):
                        cleaned  = [clean_text(r, stop, lem) for r in reviews]
                        vecs     = vec.transform(cleaned)
                        preds    = clf.predict(vecs)
                        probas   = clf.predict_proba(vecs)

                    udf["Sentiment"]  = preds
                    udf["Confidence"] = [f"{max(p):.1%}" for p in probas]
                    udf["Pos Prob"]   = [f"{probas[i][pos_idx]:.1%}" for i in range(len(probas))]

                    # ── summary metrics ────────────────────────────────────────
                    total    = len(preds)
                    n_pos    = (preds == "positive").sum()
                    n_neg    = (preds == "negative").sum()
                    avg_conf = float(pd.Series([max(p) for p in probas]).mean())

                    st.markdown("### Dashboard")
                    c1, c2, c3, c4 = st.columns(4)
                    for col, val, lbl in [
                        (c1, f"{total:,}",        "Total Reviews"),
                        (c2, f"{n_pos:,}",         "Positive"),
                        (c3, f"{n_neg:,}",         "Negative"),
                        (c4, f"{avg_conf:.1%}",    "Avg Confidence"),
                    ]:
                        col.markdown(f'<div class="metric-card"><div class="val">{val}</div>'
                                     f'<div class="lbl">{lbl}</div></div>',
                                     unsafe_allow_html=True)

                    st.markdown("")

                    # ── pie + bar side by side ─────────────────────────────────
                    col1, col2 = st.columns(2)

                    with col1:
                        fig_pie = px.pie(
                            values=[n_pos, n_neg],
                            names=["Positive", "Negative"],
                            color_discrete_sequence=["#2ecc71", "#e74c3c"],
                            title="Sentiment Split"
                        )
                        fig_pie.update_layout(height=320, margin=dict(t=40,b=10,l=10,r=10))
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col2:
                        conf_df = pd.DataFrame({
                            "Sentiment": preds,
                            "Confidence": [max(p) for p in probas]
                        })
                        fig_box = px.box(
                            conf_df, x="Sentiment", y="Confidence",
                            color="Sentiment",
                            color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c"},
                            title="Confidence Distribution"
                        )
                        fig_box.update_layout(height=320, showlegend=False,
                                              margin=dict(t=40,b=10,l=10,r=10))
                        st.plotly_chart(fig_box, use_container_width=True)

                    # ── top negative words ─────────────────────────────────────
                    neg_reviews = [cleaned[i] for i in range(len(preds)) if preds[i] == "negative"]
                    top_neg_words = []
                    if neg_reviews:
                        st.markdown("#### Top Words in Negative Reviews")
                        from sklearn.feature_extraction.text import CountVectorizer as CV
                        GENERIC = {
                            "movie", "film", "like", "good", "really", "just", "make",
                            "people", "thing", "way", "time", "even", "one", "get",
                            "see", "watch", "would", "could", "much", "also", "well",
                            "scene", "story", "character", "plot", "show", "made",
                            "think", "know", "say", "going", "come", "look", "want"
                        }
                        cv = CV(max_features=50, stop_words="english")
                        cv.fit(neg_reviews)
                        freq = dict(zip(
                            cv.get_feature_names_out(),
                            cv.transform(neg_reviews).toarray().sum(axis=0)
                        ))
                        freq = {k: v for k, v in freq.items() if k not in GENERIC}
                        freq_s = pd.Series(freq).nlargest(15).sort_values(ascending=True)
                        top_neg_words = freq_s.index.tolist()
                        fig_words = px.bar(
                            x=freq_s.values, y=freq_s.index,
                            orientation="h",
                            color_discrete_sequence=["#e74c3c"],
                            labels={"x": "Frequency", "y": "Word"},
                            title="Most Frequent Words in Negative Reviews"
                        )
                        fig_words.update_layout(height=380, margin=dict(t=40,b=10,l=10,r=10))
                        st.plotly_chart(fig_words, use_container_width=True)

                    # ── recommendations ─────────────────────────────────────
                    st.markdown("#### 💡 Recommendations")

                    RULES = [
                        (
                            ["slow", "delay", "late", "wait", "waiting", "delayed", "long", "time"],
                            "Speed / Delivery",
                            "Customers are complaining about **slow speed or delays**. Consider improving response time, delivery speed, or processing time."
                        ),
                        (
                            ["rude", "staff", "behaviour", "behavior", "unprofessional", "impolite", "arrogant", "attitude"],
                            "Customer Service",
                            "Complaints about **staff behaviour or rudeness** detected. Staff training and customer service improvement is recommended."
                        ),
                        (
                            ["quality", "bad", "broken", "cheap", "defective", "poor", "damage", "damaged", "worst", "terrible"],
                            "Product / Service Quality",
                            "Customers are unhappy with **quality**. Review your product or service standards and consider quality checks."
                        ),
                        (
                            ["price", "expensive", "costly", "overpriced", "worth", "money", "cheap", "cost"],
                            "Pricing",
                            "Pricing concerns are appearing in negative reviews. Consider reviewing your **pricing strategy** or offering better value."
                        ),
                        (
                            ["dirty", "smell", "clean", "hygiene", "unhygienic", "filthy", "stink"],
                            "Hygiene / Cleanliness",
                            "Customers are mentioning **hygiene or cleanliness** issues. Immediate attention to cleanliness standards is advised."
                        ),
                        (
                            ["wrong", "incorrect", "mistake", "error", "missing", "incomplete", "inaccurate"],
                            "Accuracy / Order Issues",
                            "Customers are reporting **wrong or missing items/information**. Review your order fulfilment or accuracy processes."
                        ),
                        (
                            ["return", "refund", "exchange", "replace", "replacement", "warranty"],
                            "Returns & Refunds",
                            "Multiple mentions of **returns or refunds** in negative reviews. Simplify your return/refund policy to improve trust."
                        ),
                        (
                            ["fake", "fraud", "scam", "cheat", "lie", "mislead", "false"],
                            "Trust & Authenticity",
                            "Serious complaints about **fraud or misleading information** detected. Urgently review authenticity and transparency."
                        ),
                    ]

                    RULES = [
                        (
                            ["slow", "delay", "late", "wait", "waiting", "delayed", "long", "time"],
                            "Speed / Delivery",
                            "Customers are complaining about **slow speed or delays**. Consider improving response time, delivery speed, or processing time."
                        ),
                        (
                            ["rude", "staff", "behaviour", "behavior", "unprofessional", "impolite", "arrogant", "attitude"],
                            "Customer Service",
                            "Complaints about **staff behaviour or rudeness** detected. Staff training and customer service improvement is recommended."
                        ),
                        (
                            ["quality", "bad", "broken", "cheap", "defective", "poor", "damage", "damaged", "worst", "terrible"],
                            "Product / Service Quality",
                            "Customers are unhappy with **quality**. Review your product or service standards and consider quality checks."
                        ),
                        (
                            ["price", "expensive", "costly", "overpriced", "worth", "money", "cheap", "cost"],
                            "Pricing",
                            "Pricing concerns are appearing in negative reviews. Consider reviewing your **pricing strategy** or offering better value."
                        ),
                        (
                            ["dirty", "smell", "clean", "hygiene", "unhygienic", "filthy", "stink"],
                            "Hygiene / Cleanliness",
                            "Customers are mentioning **hygiene or cleanliness** issues. Immediate attention to cleanliness standards is advised."
                        ),
                        (
                            ["wrong", "incorrect", "mistake", "error", "missing", "incomplete", "inaccurate"],
                            "Accuracy / Order Issues",
                            "Customers are reporting **wrong or missing items/information**. Review your order fulfilment or accuracy processes."
                        ),
                        (
                            ["return", "refund", "exchange", "replace", "replacement", "warranty"],
                            "Returns & Refunds",
                            "Multiple mentions of **returns or refunds** in negative reviews. Simplify your return/refund policy to improve trust."
                        ),
                        (
                            ["fake", "fraud", "scam", "cheat", "lie", "mislead", "false"],
                            "Trust & Authenticity",
                            "Serious complaints about **fraud or misleading information** detected. Urgently review authenticity and transparency."
                        ),
                    ]

                    neg_pct = (n_neg / total) * 100

                    # detect domain from top words
                    all_words = " ".join([cleaned[i] for i in range(len(preds))])
                    word_list = all_words.split()

                    movie_keywords    = ["film", "movie", "actor", "actress", "director",
                                         "cinema", "acting", "screenplay", "sequel", "horror",
                                         "comedy", "thriller", "imdb", "dvd", "scene"]
                    product_keywords  = ["product", "delivery", "order", "seller", "price",
                                         "quality", "package", "shipping", "refund", "return",
                                         "amazon", "flipkart", "bought", "purchase", "item",
                                         "brand", "battery", "size", "color", "material"]

                    movie_score   = sum(word_list.count(w) for w in movie_keywords)
                    product_score = sum(word_list.count(w) for w in product_keywords)
                    is_movie_domain = movie_score > product_score

                    MOVIE_RULES = [
                        (
                            ["boring", "slow", "drag", "dull", "long", "tedious", "pace"],
                            "Pacing / Story Flow",
                            "Viewers find the **pacing slow or the story boring**. Films with tighter editing and better story flow tend to get higher ratings."
                        ),
                        (
                            ["acting", "actor", "actress", "performance", "cast", "character"],
                            "Acting & Characters",
                            "Negative feedback about **acting or characters**. Weak performances are a top reason viewers rate films poorly."
                        ),
                        (
                            ["plot", "story", "script", "writing", "screenplay", "sense", "logic"],
                            "Plot & Writing",
                            "Viewers are criticising the **plot or screenplay**. A weak or predictable story is a common complaint in negative reviews."
                        ),
                        (
                            ["ending", "end", "finish", "conclusion", "finale"],
                            "Ending",
                            "The **ending** is receiving negative feedback. Unsatisfying endings significantly impact overall viewer satisfaction."
                        ),
                        (
                            ["waste", "money", "ticket", "expensive", "worth", "overrated"],
                            "Value for Money",
                            "Viewers feel the film was **not worth watching**. Managing audience expectations through better marketing can help."
                        ),
                        (
                            ["effect", "cgi", "visual", "graphic", "special"],
                            "Visual Effects",
                            "Complaints about **visual effects or CGI quality**. Poor visuals are increasingly noticed by modern audiences."
                        ),
                    ]

                    # overall health score
                    health = int((n_pos / total) * 100)
                    if health >= 70:
                        health_color = "#2ecc71"
                        health_label = "Good"
                        health_emoji = "🟢"
                    elif health >= 50:
                        health_color = "#f39c12"
                        health_label = "Needs Attention"
                        health_emoji = "🟡"
                    else:
                        health_color = "#e74c3c"
                        health_label = "Critical"
                        health_emoji = "🔴"

                    st.markdown(
                        f"<div style='background:{health_color}22; border-left:5px solid {health_color}; "
                        f"padding:1rem 1.5rem; border-radius:10px; margin-bottom:1rem;'>"
                        f"<b style='font-size:1.2rem;'>{health_emoji} Customer Satisfaction Score: {health}/100 — {health_label}</b><br>"
                        f"<span style='color:#555;'>{n_pos:,} positive out of {total:,} total reviews</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    # keyword-based recommendations
                    active_rules = MOVIE_RULES if is_movie_domain else RULES
                    domain_label = "Movie / Entertainment Reviews" if is_movie_domain else "Product / Service Reviews"
                    st.caption(f"Detected domain: **{domain_label}**")

                    matched = []
                    for keywords, category, advice in active_rules:
                        if any(w in top_neg_words for w in keywords):
                            matched.append((category, advice))

                    if matched:
                        for category, advice in matched:
                            st.markdown(
                                f"<div style='background:#fff8e1; border-left:4px solid #f39c12; "
                                f"padding:0.8rem 1.2rem; border-radius:8px; margin-bottom:0.6rem;'>"
                                f"<b>⚠️ {category}</b><br>{advice}"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                    elif neg_pct > 30:
                        st.markdown(
                            "<div style='background:#fff8e1; border-left:4px solid #f39c12; "
                            "padding:0.8rem 1.2rem; border-radius:8px;'>"
                            "<b>⚠️ General</b><br>A significant portion of reviews are negative. "
                            "Review customer feedback carefully and identify areas of improvement."
                            "</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.success("✅ No major issues detected. Keep maintaining your current standards!")
                    # ── full results table ─────────────────────────────────────
                    st.markdown("#### Full Results")

                    def highlight_csv(row):
                        color = "#d4edda" if row["Sentiment"] == "positive" else "#f8d7da"
                        return [f"background-color: {color}"] * len(row)

                    display_df = udf.head(1000)
                    if len(udf) > 1000:
                        st.caption(f"Showing first 1,000 of {len(udf):,} rows. Download CSV below for all results.")
                    st.dataframe(
                        display_df.style.apply(highlight_csv, axis=1),
                        use_container_width=True, height=350
                    )

                    # ── download button ────────────────────────────────────────
                    csv_out = udf.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_out,
                        file_name="sentiment_results.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Error reading file: {e}")

# ── footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.85rem;'>"
    "Sentiment Analysis · IMDb Dataset · TF-IDF + Logistic Regression · Built with Streamlit"
    "</p>",
    unsafe_allow_html=True,
)

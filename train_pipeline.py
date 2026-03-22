"""
train_pipeline.py
Run this ONCE to: download data → clean → visualize → train → save model
"""

import os, re, warnings, joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc)
from datasets import load_dataset

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data")
MODEL  = os.path.join(BASE, "model")
ASSETS = os.path.join(BASE, "assets")
for p in [DATA, MODEL, ASSETS]:
    os.makedirs(p, exist_ok=True)

# ── 1. DOWNLOAD DATA ───────────────────────────────────────────────────────────
print("\n[1/6] Downloading dataset …")
raw_csv = os.path.join(DATA, "reviews_raw.csv")

if not os.path.exists(raw_csv):
    ds = load_dataset("stanfordnlp/imdb", split="train+test")
    df = pd.DataFrame({"text": ds["text"], "label": ds["label"]})
    # label: 0=negative, 1=positive  →  map to string
    df["sentiment"] = df["label"].map({0: "negative", 1: "positive"})
    df.drop(columns=["label"], inplace=True)
    df.to_csv(raw_csv, index=False)
    print(f"   Saved {len(df):,} rows -> {raw_csv}")
else:
    df = pd.read_csv(raw_csv)
    print(f"   Loaded {len(df):,} rows from cache")

# ── 2. DATA CLEANING ───────────────────────────────────────────────────────────
print("\n[2/6] Cleaning data …")

nltk.download("stopwords",    quiet=True)
nltk.download("wordnet",      quiet=True)
nltk.download("omw-1.4",      quiet=True)

STOP  = set(stopwords.words("english"))
lemma = WordNetLemmatizer()

def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML
    text = re.sub(r"[^a-z\s]", " ", text)          # keep letters only
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [lemma.lemmatize(w) for w in text.split() if w not in STOP and len(w) > 2]
    return " ".join(tokens)

tqdm.pandas(desc="   cleaning")
df["clean_text"] = df["text"].progress_apply(clean)
df.dropna(subset=["clean_text"], inplace=True)
df = df[df["clean_text"].str.strip() != ""]

clean_csv = os.path.join(DATA, "reviews_clean.csv")
df.to_csv(clean_csv, index=False)
print(f"   Clean dataset: {len(df):,} rows -> {clean_csv}")

# ── 3. EDA / VISUALISATIONS ────────────────────────────────────────────────────
print("\n[3/6] Generating visualisations …")

sns.set_theme(style="whitegrid", palette="muted")

# 3a. Class distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["sentiment"].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=["#e74c3c", "#2ecc71"], edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f"{val:,}", ha="center", va="bottom", fontweight="bold")
ax.set_title("Sentiment Class Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Sentiment"); ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS, "class_dist.png"), dpi=150)
plt.close()

# 3b. Review length distribution
df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
fig, ax = plt.subplots(figsize=(8, 4))
for sent, color in [("positive", "#2ecc71"), ("negative", "#e74c3c")]:
    subset = df[df["sentiment"] == sent]["word_count"]
    ax.hist(subset, bins=60, alpha=0.6, color=color, label=sent, edgecolor="none")
ax.set_title("Review Length Distribution (word count)", fontsize=14, fontweight="bold")
ax.set_xlabel("Word Count"); ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(ASSETS, "length_dist.png"), dpi=150)
plt.close()

# 3c. Word clouds
for sent, color in [("positive", "Greens"), ("negative", "Reds")]:
    text_blob = " ".join(df[df["sentiment"] == sent]["clean_text"].sample(5000, random_state=42))
    wc = WordCloud(width=800, height=400, background_color="white",
                   colormap=color, max_words=150).generate(text_blob)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Most Frequent Words — {sent.capitalize()} Reviews",
              fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, f"wordcloud_{sent}.png"), dpi=150)
    plt.close()

# 3d. Top-20 words per class (bar chart)
from sklearn.feature_extraction.text import CountVectorizer
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, sent, color in zip(axes, ["positive", "negative"], ["#2ecc71", "#e74c3c"]):
    cv = CountVectorizer(max_features=20)
    cv.fit_transform(df[df["sentiment"] == sent]["clean_text"])
    freq = dict(zip(cv.get_feature_names_out(),
                    cv.transform(df[df["sentiment"] == sent]["clean_text"]).toarray().sum(axis=0)))
    freq_s = pd.Series(freq).sort_values(ascending=True)
    freq_s.plot(kind="barh", ax=ax, color=color, edgecolor="white")
    ax.set_title(f"Top 20 Words — {sent.capitalize()}", fontweight="bold")
    ax.set_xlabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS, "top_words.png"), dpi=150)
plt.close()

print("   Saved 5 visualisation images to assets/")

# ── 4. TRAIN / TEST SPLIT ──────────────────────────────────────────────────────
print("\n[4/6] Splitting data …")
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["sentiment"],
    test_size=0.2, random_state=42, stratify=df["sentiment"]
)
print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 5. TRAIN MODEL ─────────────────────────────────────────────────────────────
print("\n[5/6] Training TF-IDF + Logistic Regression …")
vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=1000, C=5, solver="lbfgs", n_jobs=-1)
clf.fit(X_train_vec, y_train)

y_pred  = clf.predict(X_test_vec)
y_proba = clf.predict_proba(X_test_vec)[:, 1]

report = classification_report(y_test, y_pred, output_dict=True)
print(f"\n   Accuracy : {report['accuracy']:.4f}")
print(f"   F1 (pos) : {report['positive']['f1-score']:.4f}")
print(f"   F1 (neg) : {report['negative']['f1-score']:.4f}")

# save report
pd.DataFrame(report).transpose().to_csv(os.path.join(ASSETS, "classification_report.csv"))

# 5a. Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["positive", "negative"])
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(cm, display_labels=["Positive", "Negative"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS, "confusion_matrix.png"), dpi=150)
plt.close()

# 5b. ROC curve
fpr, tpr, _ = roc_curve((y_test == "positive").astype(int), y_proba)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"AUC = {roc_auc:.4f}")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS, "roc_curve.png"), dpi=150)
plt.close()

# ── 6. SAVE MODEL ──────────────────────────────────────────────────────────────
print("\n[6/6] Saving model …")
joblib.dump(vectorizer, os.path.join(MODEL, "tfidf_vectorizer.joblib"))
joblib.dump(clf,        os.path.join(MODEL, "logistic_model.joblib"))

meta = {
    "accuracy":    report["accuracy"],
    "f1_positive": report["positive"]["f1-score"],
    "f1_negative": report["negative"]["f1-score"],
    "roc_auc":     roc_auc,
    "train_size":  len(X_train),
    "test_size":   len(X_test),
}
joblib.dump(meta, os.path.join(MODEL, "meta.joblib"))
print("   Model saved to model/")
print("\nPipeline complete! Run: streamlit run app.py")

# 🎭 Sentiment Analysis — COMPLETE DETAILED GUIDE

**Everything Explained: Concepts + Code + Files + Folders**

## 📋 **Table of Contents**
1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Core Concepts](#core-concepts)
4. [`train_pipeline.py` — Full Breakdown](#train_pipelinepy)
5. [`sentiment_analysis.ipynb` — Cell-by-Cell](#sentiment_analysisipynb)
6. [`app.py` — Web App](#apppy)
7. [Model Files Explained](#model-files)
8. [Assets/Charts](#assetscharts)
9. [Run Instructions](#run-instructions)
10. [Why This Approach?](#why-this-approach)

---

## ## 📊 **Project Overview**

**Goal**: Predict movie review sentiment (**positive** vs **negative**).
```
Input: "This movie was terrible waste time"
Output: NEGATIVE (92% confidence)
```

**Pipeline**:
```
Raw IMDb reviews (HF) 
→ Clean (NLTK) 
→ TF-IDF features (sklearn) 
→ Logistic Regression (sklearn) 
→ Web predictions (Streamlit)
```

**Performance**: **90.6% accuracy** (10K test reviews).

---

## ## 📁 **Folder Structure** (What You'll Find Where)

```
sentiment analysis of user review/          # Root
├── app.py                          # Web app (run: streamlit run app.py)
├── train_pipeline.py               # Train script (run ONCE)
├── requirements.txt                # pip install -r requirements.txt
├── run.bat                         # Windows: double-click
├── README.md                       # Original (somewhat inaccurate)
├── README_SIMPLE.md                # Simplified version
├── README_DETAILED.md              # ← THIS FILE
│
├── data/                           # Raw + cleaned data
│   ├── reviews_raw.csv             # 50K raw IMDb reviews (text + sentiment)
│   └── reviews_clean.csv           # Cleaned version (+ word_count)
│
├── model/                          # Trained model files (joblib)
│   ├── tfidf_vectorizer.joblib     # Text → 50K numbers
│   ├── logistic_model.joblib       # Numbers → positive/negative
│   └── meta.joblib                 # Accuracy/F1/AUC scores
│
├── assets/                         # Charts (auto-generated)
│   ├── class_dist.png              # Positive/Negative bar chart
│   ├── length_dist.png             # Word count histogram
│   ├── wordcloud_*.png             # Top words (visual)
│   ├── top_words.png               # Bar chart top 20 words/class
│   ├── confusion_matrix.png        # Model errors (2x2 grid)
│   ├── roc_curve.png               # Model quality (AUC=0.965)
│   └── classification_report.csv   # Precision/Recall/F1 table
│
└── notebooks/
    └── sentiment_analysis.ipynb    # Interactive Jupyter tutorial
```

---

## ## 🧠 **Core Concepts** (Simple → Technical)

### **1. Sentiment Analysis**
**Simple**: "Good movie 👍" = positive, "Waste of time 👎" = negative.
**Technical**: Binary text classification (positive=1, negative=0).

### **2. Classical ML vs Deep Learning**
```
Classical (THIS PROJECT)      | Deep Learning (BERT)
──────────────────────────────┼───────────────────────
TF-IDF → Logistic Regression  | Transformer embeddings
5 min training                | Hours/days + GPU
Interpretable (see words)     | Black box
Deploy anywhere               | Needs GPU server
90% accuracy                  | 93% accuracy
```

### **3. TF-IDF (Term Frequency-Inverse Document Frequency)**
**Simple**: Score words by "how common in THIS review, how RARE across ALL reviews".
```
"not good" in 1 review, rare everywhere → HIGH score (negative signal)
"the" everywhere → LOW score (no signal)
```
**Code**:
```python
TfidfVectorizer(max_features=50000, ngram_range=(1,2))
# 50K features: unigrams + bigrams ("not good")
```
**Result**: Text → sparse vector `[0.0, 0.42, 0.61, 0, ...]`.

### **4. Logistic Regression**
**Simple**: Linear model with sigmoid → probability (0-1).
**Math**: `P(positive) = 1/(1 + e^(-linear combo of features))`
**Why?** State-of-art for TF-IDF text (1988-2020).

### **5. Cleaning Pipeline** (7 Steps)
```python
def clean(text):
    text = text.lower()                           # 1
    text = re.sub(r'<[^>]+>', ' ', text)         # 2: HTML
    text = re.sub(r'[^a-z\s]', ' ', text)        # 3: symbols
    text = re.sub(r'\s+', ' ', text).strip()     # 4: spaces
    tokens = [lemma.lemmatize(w) for w in text.split() 
              if w not in STOPWORDS and len(w) > 2]  # 5,6,7
    return ' '.join(tokens)
```
**Example**:
```
Raw: "I LOVED <br /> this AMAZING!!! film. 10/10"
Clean: "loved amazing film"
```

---

## ## ⚙️ **`train_pipeline.py`** — Complete Code Breakdown

**Purpose**: Download → Clean → EDA → Train → Save (run **ONCE**, 5 mins).

```python
# Imports (concepts explained above)
import pandas, sklearn, nltk, datasets, matplotlib, etc.

# [1/6] Download (Hugging Face)
ds = load_dataset("stanfordnlp/imdb", split="train+test")  # 50K reviews
df = pd.DataFrame({"text": ds["text"], "label": ds["label"]})
df["sentiment"] = df["label"].map({0:"negative", 1:"positive"})
df.to_csv("data/reviews_raw.csv")  # Cache forever

# [2/6] Clean (NLTK pipeline above)
df["clean_text"] = df["text"].progress_apply(clean)
df.to_csv("data/reviews_clean.csv")

# [3/6] EDA (5 charts saved to assets/)
- class_dist.png: Bar chart (25K/25K)
- length_dist.png: Histogram (50-300 words)
- wordcloud_positive.png: "love great amazing"
- wordcloud_negative.png: "bad waste boring"  
- top_words.png: Top 20/class bar chart

# [4/6] Split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["sentiment"], test_size=0.2, stratify=True
)  # 40K train, 10K test

# [5/6] Train
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)  # Text → numbers

clf = LogisticRegression(C=5, max_iter=1000)
clf.fit(X_train_vec, y_train)

# Evaluate + charts (confusion_matrix.png, roc_curve.png)

# [6/6] Save
joblib.dump(vectorizer, "model/tfidf_vectorizer.joblib")
joblib.dump(clf, "model/logistic_model.joblib")
joblib.dump({"accuracy": 0.906, ...}, "model/meta.joblib")
```

---

## ## 📓 **`sentiment_analysis.ipynb`** — Cell-by-Cell Breakdown

**Purpose**: Interactive tutorial (same pipeline as script).

| Cell # | Title | Code | Output |
|--------|-------|------|--------|
| 0 | Import Libraries | `pandas, sklearn, nltk, matplotlib` | "All libraries imported!" |
| 1 | Load Dataset | `pd.read_csv("data/reviews_raw.csv")` | Shape: (50000, 2), head() |
| 2 | Dataset Info | `value_counts(), nulls, duplicates` | Perfect 50/50 balance |
| 3 | Samples | Raw positive/negative examples | Truncated reviews |
| 4 | NLTK Downloads | `nltk.download('stopwords')` | Downloads |
| 5 | Cleaning Demo | Manual `clean()` on sample | 7-step before/after |
| 6 | Clean All | `df['clean_text'] = df['text'].progress_apply(clean)` | Progress bar |
| 7 | Class Distribution | `sns.barplot()` | PNG-ready chart |
| 8 | Length Dist | `hist()` overlay pos/neg | Stats printed |
| 9 | Word Clouds | `WordCloud()` x2 | PNGs saved |
| 10 | Top 20 Words | `CountVectorizer(max_features=20)` | Side-by-side bars |
| 11 | Summary Stats | `df.describe()` | Word count stats |
| 12 | Train/Test Split | `train_test_split(stratify=True)` | Sizes printed |
| 13 | TF-IDF | `TfidfVectorizer()` + sample | Top terms example |
| 14 | Train Model | `LogisticRegression().fit()` | "Model trained!" |
| 15 | Top Learned Words | `clf.coef_` → bar charts | Model insights |
| 16 | Evaluate | `classification_report()` | Metrics table |
| 17 | Confusion Matrix | `ConfusionMatrixDisplay()` | PNG saved |
| 18 | ROC Curve | `roc_curve(), auc()` | PNG + AUC printed |
| 19 | Live Predict | `def predict(text):` | 3 test examples |
| 20 | Save Model | `joblib.dump()` x3 | "Model saved!" |

**Note**: Notebook **skips** HF download (uses existing CSV).

---

## ## 🌐 **`app.py`** — Streamlit Web App

**5 Tabs** (interactive predictions):

1. **Data Overview**: Cards + DataFrame samples
2. **Data Cleaning**: 7 accordions (steps) + before/after
3. **Visualizations**: 5 charts (Plotly interactive)
4. **Model & Metrics**: Cards + tables + static charts
5. **Live Prediction**:
   ```python
   # Single
   cleaned = clean(st.text_area("Your review"))
   vec = vectorizer.transform([cleaned])
   pred = clf.predict(vec)[0]
   conf = clf.predict_proba(vec).max()
   
   # Batch (multi-line input → DataFrame)
   ```

---

## ## 💾 **Model Files** (`model/` folder)

| File | Size | Purpose | Load Code |
|------|------|---------|-----------|
| `tfidf_vectorizer.joblib` | ~20MB | Text → 50K-dim vector | `joblib.load()` |
| `logistic_model.joblib` | ~4MB | Vector → prediction | `joblib.load()` |
| `meta.joblib` | ~1KB | Scores (accuracy=0.906) | `joblib.load()` |

**Predict Function**:
```python
def predict(text):
    cleaned = clean(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    return pred, prob
```

---

## ## 📈 **Assets/Charts** (`assets/`)

| Chart | What It Shows | Why Important |
|-------|---------------|---------------|
| `class_dist.png` | 25K pos / 25K neg | Perfect balance ✓ |
| `length_dist.png` | 50-300 words avg | Data quality |
| `wordcloud_*.png` | Top words visual | Pos/neg vocabulary |
| `top_words.png` | CountVectorizer top 20 | Frequent words |
| `confusion_matrix.png` | TP/TN/FP/FN | Model errors |
| `roc_curve.png` | AUC=0.965 | Separation quality |
| `classification_report.csv` | Precision/Recall/F1 | Detailed metrics |

---

## ## 🚀 **Run Instructions** (Copy-Paste Ready)

```bash
# 1. Install (1st time)
pip install -r requirements.txt

# 2. Train (1st time, 5 mins)
python train_pipeline.py

# 3. Launch web app
streamlit run app.py
```
**URL**: `http://localhost:8501`

**Windows**: `run.bat` does all 3 steps.

---

## ## 🤔 **Why This Approach? (vs Modern LLMs)**

```
✅ TEACHES ML fundamentals (cleaning → features → model)
✅ PRODUCTION ready (no GPU, loads instantly)
✅ INTERPRETABLE ("hate" pushes negative by 1.2 coef)
✅ CHEAP (runs on $5 server)
✅ FAST (5s predict vs 30s GPT)
❌ Slightly lower accuracy (90% vs 95%)
```

**Live Demo**: Paste ANY review → instant result + confidence.

**Questions?** File-specific details above!

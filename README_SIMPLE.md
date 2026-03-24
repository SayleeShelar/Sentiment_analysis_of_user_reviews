# 🎭 Sentiment Analysis of User Reviews — SIMPLE EXPLANATION

## 🚀 **What This Project Does**
Classifies movie reviews as **POSITIVE** or **NEGATIVE** using **classical machine learning**.
- Dataset: **50K real IMDb reviews** (25K positive + 25K negative).
- Model: **TF-IDF + Logistic Regression** (trained from scratch).
- **Accuracy: ~90.6%** on test set.
- **Web App**: Predict sentiment of **your own reviews** via Streamlit.

**No deep learning, no Hugging Face models — pure sklearn + NLTK.**

---

## 📁 **Project Structure** (Simple View)
```
sentiment analysis of user review/
├── app.py                 ← Web interface (5 tabs: data, cleaning, charts, metrics, predict)
├── train_pipeline.py      ← ONE-TIME: download → clean → train → save model
├── notebooks/
│   └── sentiment_analysis.ipynb  ← Interactive Jupyter demo (ALL STEPS)
├── data/
│   ├── reviews_raw.csv    ← 50K raw IMDb reviews (from HF)
│   └── reviews_clean.csv  ← Cleaned version
├── model/                 ← Trained files (.joblib)
│   ├── tfidf_vectorizer.joblib
│   ├── logistic_model.joblib
│   └── meta.joblib        ← Accuracy/F1/AUC scores
└── assets/                ← Charts (PNG): class dist, word clouds, confusion matrix, ROC
```

---

## 🎯 **How It Works** (8 Simple Steps)

### **Step 1: Get Data** (`train_pipeline.py`)
```python
from datasets import load_dataset  # Hugging Face (dataset ONLY)
ds = load_dataset("stanfordnlp/imdb")  # 50K reviews
df.to_csv("data/reviews_raw.csv")
```
- **Source**: Stanford IMDb dataset via Hugging Face Datasets Hub.
- **Format**: `text` (review) + `sentiment` (positive/negative).

### **Step 2: Clean Text** (NLTK)
Raw reviews = messy → Clean = machine-readable.
```
1. Lowercase: "Great Film!" → "great film!"
2. Remove HTML: "<br />" → " "
3. Remove symbols: "!!!" → " "
4. Normalize spaces
5. Remove stopwords: "the, is, and" → gone
6. Lemmatize: "running" → "run", "movies" → "movie"
7. Filter short words: "ok, it" → gone
```
**Result**: `clean_text` column.

### **Step 3: Explore Data** (EDA Charts)
```
- Class balance: 50/50 positive/negative ✓
- Word clouds: "great/love" (pos) vs "bad/waste" (neg)
- Review lengths: 50-300 words avg
- Top words per class ✓
```

### **Step 4: Features (TF-IDF)**
Convert text → numbers (50K features):
```
"not bad movie" → [0.0, 0.42, 0.61, 0, ...]  # Sparse vector
```
- **Why TF-IDF?** Rare words that appear often in one review = high score.
- **Bigrams**: "not bad", "highly recommend".

### **Step 5: Train Model**
```
80% data → train Logistic Regression
20% data → test accuracy
```
**Hyperparams**: `C=5, max_iter=1000` → **90.6% accuracy**.

### **Step 6: Evaluate**
```
Accuracy: 90.6%
F1 Positive: 90.7%
F1 Negative: 90.5%
ROC-AUC: 96.5%
```
**Charts**: Confusion matrix, ROC curve.

### **Step 7: Save Model** (joblib)
```
tfidf_vectorizer.joblib     ← Text → numbers
logistic_model.joblib       ← Predict positive/negative
meta.joblib                 ← Scores for app
```

### **Step 8: Predict New Reviews** (`app.py`)
```
Input: "This movie was terrible!"
↓ Clean + TF-IDF
↓ LogisticRegression.predict()
Output: NEGATIVE (92% confidence)
```

---

## 🛠 **Tech Stack** (Simple)
```
Data:       pandas, HuggingFace datasets (download only)
Cleaning:   NLTK (stopwords, lemmatization)
Features:   sklearn TfidfVectorizer
Model:      sklearn LogisticRegression
Web:        Streamlit (interactive app)
Charts:     matplotlib, seaborn, wordcloud, plotly
Save/Load:  joblib
```

---

## 🚀 **QUICK START** (2 Commands)

### **Windows (One-Click)**
```
double-click run.bat  # Installs + trains + launches app
```

### **Manual**
```bash
pip install -r requirements.txt
python train_pipeline.py  # Run ONCE (5 mins)
streamlit run app.py      # Launch web app
```
**Open**: `http://localhost:8501`

### **Jupyter Demo**
```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

---

## 🔮 **Web App** (5 Tabs)
1. **Data Overview**: Dataset stats + samples
2. **Data Cleaning**: Before/after + 7 cleaning steps
3. **Visualizations**: 5 EDA charts
4. **Model & Metrics**: Confusion matrix + ROC + report
5. **🔮 Live Prediction**:
   - **Single**: Paste review → instant result
   - **Batch**: Multiple reviews → table + summary chart

---

## 💡 **Why Classical ML? (Not BERT/Deep Learning)**
```
✅ FAST training (5 mins vs hours)
✅ INTERPRETABLE (see top words pushing pos/neg)
✅ DEPLOYABLE (loads in <1s, no GPU)
✅ TEACHES fundamentals (TF-IDF, Logistic Regression)
❌ Less accurate (~90% vs BERT 93%)
```

**Production**: Works on Raspberry Pi. BERT needs GPU server.

---

## 📈 **Model Performance**
```
Test Set (10K reviews):
Accuracy:     90.6%
F1-Positive:  90.7%
F1-Negative:  90.5%
ROC-AUC:      96.5%
```

**Top Positive Words**: love, great, amazing, perfect, excellent
**Top Negative Words**: bad, worst, waste, boring, terrible

---

## 🎯 **Key Files** (What Does What?)
| File | Purpose | Run When? |
|------|---------|-----------|
| `train_pipeline.py` | Download + train + save | **ONCE** |
| `sentiment_analysis.ipynb` | Interactive tutorial | Learning |
| `app.py` | Web predictions | **Every time** |

**Questions?** Ask about any step!

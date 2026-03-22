# 🎭 Sentiment Analysis of User Reviews

End-to-end NLP project: data collection → cleaning → EDA → model training → web interface.

---

## 📁 Project Structure

```
sentiment analysis of user review/
├── app.py                  ← Streamlit web interface
├── train_pipeline.py       ← Full pipeline (run once)
├── requirements.txt
├── run.bat                 ← One-click setup & launch (Windows)
├── data/
│   ├── reviews_raw.csv     ← Downloaded IMDb dataset (50k reviews)
│   └── reviews_clean.csv   ← After cleaning + feature engineering
├── model/
│   ├── tfidf_vectorizer.joblib
│   ├── logistic_model.joblib
│   └── meta.joblib         ← Accuracy, F1, AUC scores
└── assets/                 ← All generated charts & images
    ├── class_dist.png
    ├── length_dist.png
    ├── wordcloud_positive.png
    ├── wordcloud_negative.png
    ├── top_words.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── classification_report.csv
```

---

## 🚀 Quick Start

### Option A — One click (Windows)
```
Double-click run.bat
```

### Option B — Manual
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model (run ONCE — ~5-10 min, downloads 50k reviews)
python train_pipeline.py

# 3. Launch app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🔬 Pipeline Steps

| Step | Description |
|------|-------------|
| 1. Data Collection | IMDb 50k movie reviews via HuggingFace `datasets` |
| 2. Data Cleaning | Lowercase → HTML strip → regex → stopwords → lemmatise |
| 3. EDA | Class dist, length dist, word clouds, top-word bar charts |
| 4. Feature Engineering | TF-IDF (unigrams + bigrams, 50k features, sublinear TF) |
| 5. Model Training | Logistic Regression (C=5, lbfgs) on 40k samples |
| 6. Evaluation | Accuracy, F1, ROC-AUC, confusion matrix |
| 7. Persistence | Model saved with `joblib` — no retraining needed |

---

## 🌐 App Features

- **Data Overview** — dataset stats and sample rows
- **Data Cleaning** — step-by-step explanation + before/after examples
- **Visualisations** — all EDA charts rendered inline
- **Model & Metrics** — confusion matrix, ROC curve, classification report
- **Live Prediction** — single review with gauge chart + batch CSV-style results

---

## 📊 Expected Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~89–91% |
| F1 (positive) | ~0.89–0.91 |
| F1 (negative) | ~0.89–0.91 |
| ROC-AUC | ~0.96–0.97 |

# 🎭 Sentiment Analysis of User Reviews

A complete end-to-end Natural Language Processing (NLP) project that classifies movie reviews as **positive** or **negative**. Built with Python, trained on 50,000 real IMDb reviews, and served through an interactive Streamlit web interface.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Project Structure](#-project-structure)
3. [Tech Stack](#-tech-stack)
4. [Quick Start](#-quick-start)
5. [Step 1 — Data Collection](#-step-1--data-collection)
6. [Step 2 — Data Cleaning](#-step-2--data-cleaning)
7. [Step 3 — Exploratory Data Analysis (EDA)](#-step-3--exploratory-data-analysis-eda)
8. [Step 4 — Feature Engineering](#-step-4--feature-engineering)
9. [Step 5 — Model Training](#-step-5--model-training)
10. [Step 6 — Model Evaluation](#-step-6--model-evaluation)
11. [Step 7 — Model Persistence](#-step-7--model-persistence)
12. [Step 8 — Web Interface](#-step-8--web-interface)
13. [Model Performance](#-model-performance)
14. [How Prediction Works](#-how-prediction-works)
15. [Troubleshooting](#-troubleshooting)

---

## 📖 Project Overview

Sentiment analysis is the task of automatically identifying whether a piece of text expresses a **positive** or **negative** opinion. This project demonstrates a full production-style NLP pipeline:

```
Raw Text Reviews
      ↓
  Data Cleaning  (remove noise, normalize text)
      ↓
  EDA            (understand the data visually)
      ↓
  Feature Eng.   (convert text to numbers via TF-IDF)
      ↓
  Model Training (Logistic Regression classifier)
      ↓
  Evaluation     (accuracy, F1, ROC-AUC)
      ↓
  Save Model     (joblib — no retraining needed)
      ↓
  Web App        (Streamlit interface for live predictions)
```

The trained model achieves **~90.6% accuracy** on 10,000 held-out test reviews.

---

## 📁 Project Structure

```
sentiment analysis of user review/
│
├── app.py                        ← Streamlit web interface (5 tabs)
├── train_pipeline.py             ← Full pipeline script (run once)
├── requirements.txt              ← Python dependencies
├── run.bat                       ← One-click Windows launcher
├── README.md                     ← This file
│
├── notebooks/
│   └── sentiment_analysis.ipynb  ← Interactive Jupyter notebook (all steps)
│
├── data/
│   ├── reviews_raw.csv           ← Raw IMDb dataset (50,000 reviews)
│   └── reviews_clean.csv         ← Cleaned + processed dataset
│
├── model/
│   ├── tfidf_vectorizer.joblib   ← Saved TF-IDF vectorizer
│   ├── logistic_model.joblib     ← Saved trained classifier
│   └── meta.joblib               ← Saved accuracy/F1/AUC scores
│
└── assets/
    ├── class_dist.png            ← Bar chart: class balance
    ├── length_dist.png           ← Histogram: review lengths
    ├── wordcloud_positive.png    ← Word cloud: positive reviews
    ├── wordcloud_negative.png    ← Word cloud: negative reviews
    ├── top_words.png             ← Top 20 words per class
    ├── confusion_matrix.png      ← Model confusion matrix
    ├── roc_curve.png             ← ROC curve with AUC score
    └── classification_report.csv ← Full precision/recall/F1 table
```

---

## 🛠 Tech Stack

| Category | Library | Purpose |
|---|---|---|
| Data | `pandas`, `numpy` | Data loading, manipulation |
| NLP | `nltk` | Stopwords, lemmatization |
| ML | `scikit-learn` | TF-IDF, Logistic Regression, metrics |
| Visualization | `matplotlib`, `seaborn` | EDA charts |
| Word Clouds | `wordcloud` | Visual word frequency |
| Interactive Charts | `plotly` | Gauge chart, bar charts in app |
| Dataset | `datasets` (HuggingFace) | Download IMDb dataset |
| Model Saving | `joblib` | Serialize/deserialize model |
| Web Interface | `streamlit` | Interactive browser app |

---

## 🚀 Quick Start

### Option A — One Click (Windows)
```
Double-click  run.bat
```
This will automatically install dependencies, train the model, and launch the app.

### Option B — Manual (Step by Step)

**Step 1: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2: Train the model** *(run only once — takes ~5 minutes)*
```bash
python train_pipeline.py
```

**Step 3: Launch the web app**
```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

> **Note:** After running `train_pipeline.py` once, the model is saved to the `model/` folder. You never need to retrain — just run `streamlit run app.py` directly next time.

### Option C — Jupyter Notebook (Interactive exploration)

```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

The notebook walks through every step interactively with inline charts and outputs.

> **Important:** Run all cells **except the last one (Step 8 — Save Model)** if the model is already trained. The last cell overwrites the saved model files — safe to skip.

---

## 📥 Step 1 — Data Collection

**File:** `train_pipeline.py` → lines `[1/6]`

**What happens:**
- The dataset is downloaded automatically from HuggingFace using the `datasets` library
- Source: [Stanford IMDb Large Movie Review Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
- Both the `train` and `test` splits are combined to give 50,000 reviews total
- Labels are mapped: `0 → "negative"`, `1 → "positive"`
- The raw data is saved to `data/reviews_raw.csv` so it is never downloaded again

**Dataset properties:**

| Property | Value |
|---|---|
| Total reviews | 50,000 |
| Positive reviews | 25,000 (50%) |
| Negative reviews | 25,000 (50%) |
| Source | IMDb movie reviews |
| Language | English |
| Label type | Binary (positive / negative) |

**Why 50,000 reviews?**
A larger dataset means the model sees more vocabulary, more writing styles, and more edge cases — resulting in a robust model that generalizes well to new, unseen reviews.

**Code snippet:**
```python
ds = load_dataset("stanfordnlp/imdb", split="train+test")
df = pd.DataFrame({"text": ds["text"], "label": ds["label"]})
df["sentiment"] = df["label"].map({0: "negative", 1: "positive"})
```

---

## 🧹 Step 2 — Data Cleaning

**File:** `train_pipeline.py` → lines `[2/6]`

Raw text from the internet is messy. Before feeding it to a machine learning model, it must be cleaned and normalized. The pipeline applies 7 cleaning steps in sequence:

### Cleaning Steps

**Step 1 — Lowercasing**
```
"The Movie Was GREAT!" → "the movie was great!"
```
Ensures that "Great", "GREAT", and "great" are treated as the same word.

**Step 2 — HTML Tag Removal**
```
"Great film!<br /><b>Loved it</b>" → "Great film! Loved it"
```
IMDb reviews often contain HTML markup. These are stripped using the regex `<[^>]+>`.

**Step 3 — Special Character Removal**
```
"film... 10/10 would watch again!!!" → "film     would watch again   "
```
Punctuation, numbers, and symbols are removed. Only letters and spaces are kept using `[^a-z\s]`.

**Step 4 — Whitespace Normalisation**
```
"film     would  watch  again" → "film would watch again"
```
Multiple consecutive spaces are collapsed into a single space.

**Step 5 — Stopword Removal**
```
"this is a really great film" → "really great film"
```
Common English words like "the", "is", "a", "this", "and" carry no sentiment signal. NLTK's list of 179 English stopwords is used to remove them.

**Step 6 — Lemmatisation**
```
"running" → "run"
"better"  → "good"
"movies"  → "movie"
"loved"   → "love"
```
Words are reduced to their base (dictionary) form using NLTK's `WordNetLemmatizer`. This ensures "love", "loved", and "loving" are all counted as the same word.

**Step 7 — Short Token Filter**
```
Removes any token with length ≤ 2 (e.g. "ok", "it", "is")
```
Very short tokens are usually noise or stopwords that slipped through.

### Before vs After Example

| | Text |
|---|---|
| **Raw** | `"I saw this movie last night and it was absolutely AMAZING!! The acting was top-notch. 10/10 <br/> Would recommend!"` |
| **Cleaned** | `"saw movie last night absolutely amazing acting top notch would recommend"` |

**Output:** `data/reviews_clean.csv` with a new `clean_text` column added.

---

## 📊 Step 3 — Exploratory Data Analysis (EDA)

**File:** `train_pipeline.py` → lines `[3/6]`

Before training, we visually explore the data to understand its structure and characteristics. Five charts are generated and saved to the `assets/` folder.

### Chart 1 — Class Distribution (`class_dist.png`)
A bar chart showing the count of positive vs negative reviews.

- **Finding:** The dataset is perfectly balanced — exactly 25,000 of each class.
- **Why it matters:** A balanced dataset means the model won't be biased toward predicting one class more than the other.

### Chart 2 — Review Length Distribution (`length_dist.png`)
A histogram showing the distribution of word counts (after cleaning) for both classes, overlaid on the same chart.

- **Finding:** Most reviews are between 50–300 words. Positive reviews tend to be slightly longer on average.
- **Why it matters:** Helps identify if review length itself could be a useful signal, and whether very short/long reviews need special handling.

### Chart 3 — Word Cloud: Positive Reviews (`wordcloud_positive.png`)
A visual representation where the size of each word reflects how frequently it appears in positive reviews.

- **Dominant words:** `great`, `love`, `best`, `wonderful`, `excellent`, `perfect`, `beautiful`
- **Why it matters:** Confirms that positive reviews have a distinct vocabulary that the model can learn from.

### Chart 4 — Word Cloud: Negative Reviews (`wordcloud_negative.png`)
Same as above but for negative reviews.

- **Dominant words:** `bad`, `worst`, `waste`, `boring`, `terrible`, `awful`, `stupid`
- **Why it matters:** Shows clear lexical separation between the two classes — a good sign for model performance.

### Chart 5 — Top 20 Words per Class (`top_words.png`)
Side-by-side horizontal bar charts showing the 20 most frequent words in positive and negative reviews respectively.

- **Finding:** The top words are almost entirely different between the two classes, confirming strong signal for classification.
- **Why it matters:** Validates that TF-IDF will be an effective feature extraction method.

---

## ⚙️ Step 4 — Feature Engineering

**File:** `train_pipeline.py` → lines `[5/6]` (TF-IDF fitting)

Machine learning models cannot work with raw text — text must be converted into numbers. We use **TF-IDF (Term Frequency–Inverse Document Frequency)**.

### What is TF-IDF?

TF-IDF assigns a numerical score to each word in a document based on two factors:

- **TF (Term Frequency):** How often does this word appear in *this* review?
- **IDF (Inverse Document Frequency):** How rare is this word across *all* reviews?

Words that appear frequently in one review but rarely across all reviews get a **high score** — they are distinctive and informative.

Words that appear in almost every review (like "movie", "film") get a **low score** — they don't help distinguish positive from negative.

### Configuration Used

```python
TfidfVectorizer(
    max_features=50_000,   # Keep only the top 50,000 most informative terms
    ngram_range=(1, 2),    # Use single words AND two-word phrases
    sublinear_tf=True      # Apply log(1 + tf) to dampen very high frequencies
)
```

### Why Bigrams (ngram_range=(1,2))?

Bigrams capture two-word phrases that carry meaning together:
- `"not good"` → negative (unigrams alone would see "not" and "good" separately)
- `"highly recommend"` → positive
- `"waste time"` → negative

This significantly improves accuracy compared to using only single words.

### Output

Each review is converted into a sparse vector of 50,000 dimensions. The training set produces a matrix of shape `(40,000 × 50,000)`.

---

## 🤖 Step 5 — Model Training

**File:** `train_pipeline.py` → lines `[5/6]`

### Train/Test Split

The cleaned dataset is split into:
- **Training set:** 40,000 reviews (80%) — used to train the model
- **Test set:** 10,000 reviews (20%) — used only for final evaluation

The split is **stratified**, meaning both sets have exactly 50% positive and 50% negative reviews.

```python
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["sentiment"],
    test_size=0.2, random_state=42, stratify=df["sentiment"]
)
```

### Why Logistic Regression?

Logistic Regression is chosen over more complex models (like neural networks) for several reasons:

| Reason | Explanation |
|---|---|
| Speed | Trains in seconds on 40,000 samples |
| Interpretability | Weights directly show which words push toward positive/negative |
| Performance | Achieves ~90%+ accuracy on text classification with TF-IDF |
| No retraining needed | Saves and loads instantly with `joblib` |
| Proven baseline | Industry-standard approach for binary text classification |

### Hyperparameters

```python
LogisticRegression(
    C=5,            # Regularization strength (higher = less regularization)
    solver="lbfgs", # Optimization algorithm (efficient for multiclass)
    max_iter=1000   # Maximum iterations to ensure convergence
)
```

- `C=5` was chosen to allow the model enough flexibility to fit the training data without overfitting.
- `lbfgs` is a memory-efficient quasi-Newton optimization method well-suited for this problem size.

---

## 📈 Step 6 — Model Evaluation

**File:** `train_pipeline.py` → lines `[5/6]` (after training)

After training, the model is evaluated on the **held-out test set** (10,000 reviews it has never seen).

### Metrics Explained

**Accuracy**
The percentage of reviews correctly classified.
```
Accuracy = (Correct Predictions) / (Total Predictions)
```

**Precision**
Of all reviews predicted as positive, what fraction actually were positive?
```
Precision = True Positives / (True Positives + False Positives)
```

**Recall**
Of all actual positive reviews, what fraction did the model correctly identify?
```
Recall = True Positives / (True Positives + False Negatives)
```

**F1 Score**
The harmonic mean of Precision and Recall. A single balanced metric.
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**ROC-AUC**
Area Under the ROC Curve. Measures how well the model separates the two classes across all probability thresholds. A score of 1.0 is perfect; 0.5 is random guessing.

### Evaluation Artifacts Generated

| File | Description |
|---|---|
| `assets/confusion_matrix.png` | 2×2 grid showing TP, TN, FP, FN counts |
| `assets/roc_curve.png` | ROC curve with AUC score annotated |
| `assets/classification_report.csv` | Full table of precision, recall, F1 per class |
| `model/meta.joblib` | Dictionary of all scores for the app to display |

### Confusion Matrix Explained

```
                  Predicted
                Positive  Negative
Actual Positive  [ TP ]   [ FN ]
       Negative  [ FP ]   [ TN ]
```

- **TP (True Positive):** Correctly predicted positive
- **TN (True Negative):** Correctly predicted negative
- **FP (False Positive):** Predicted positive but actually negative
- **FN (False Negative):** Predicted negative but actually positive

---

## 💾 Step 7 — Model Persistence

**File:** `train_pipeline.py` → lines `[6/6]`

After training and evaluation, three files are saved using `joblib`:

```python
joblib.dump(vectorizer, "model/tfidf_vectorizer.joblib")  # TF-IDF transformer
joblib.dump(clf,        "model/logistic_model.joblib")    # Trained classifier
joblib.dump(meta,       "model/meta.joblib")              # Performance scores
```

**Why joblib?**
- Faster than Python's built-in `pickle` for large NumPy arrays
- The TF-IDF vectorizer contains a large vocabulary matrix — joblib handles this efficiently
- Files load in under 1 second when the app starts

**What is saved in meta.joblib?**
```python
{
    "accuracy":    0.9059,
    "f1_positive": 0.9069,
    "f1_negative": 0.9049,
    "roc_auc":     0.9650,
    "train_size":  40000,
    "test_size":   10000
}
```

When the app loads, it reads these scores directly — no need to re-evaluate the model.

---

## 🌐 Step 8 — Web Interface

**File:** `app.py`

The Streamlit app provides a clean, interactive interface with **5 tabs** and a **sidebar**.

### Sidebar
Always visible on the left. Shows:
- Model status (loaded / not found)
- Live accuracy, ROC-AUC, and training size metrics

### Tab 1 — 📊 Data Overview
- Dataset description and properties table
- Sample rows from the cleaned dataset (random 8 rows)
- 4 metric cards: total reviews, avg word count, max word count, positive count

### Tab 2 — 🧹 Data Cleaning
- 7 expandable sections, one per cleaning step, each with a description
- "Before vs After" side-by-side comparison using 3 random real reviews from the dataset

### Tab 3 — 📈 Visualisations
- All 5 EDA charts displayed inline with captions:
  - Class distribution bar chart
  - Review length histogram
  - Positive word cloud
  - Negative word cloud
  - Top 20 words per class

### Tab 4 — 🤖 Model & Metrics
- Model architecture table (TF-IDF config, classifier config, split sizes)
- 4 metric cards: Accuracy, F1 Positive, F1 Negative, ROC-AUC
- Confusion matrix image
- ROC curve image
- Full classification report table with color gradient

### Tab 5 — 🔮 Live Prediction

**Single Review Mode:**
1. User pastes any review text into the text box
2. Clicks "Analyse Sentiment"
3. The app applies the same 7-step cleaning pipeline to the input
4. The cleaned text is vectorized using the saved TF-IDF vectorizer
5. The saved Logistic Regression model predicts the class and probability
6. Results shown as:
   - Color-coded result card (green = positive, red = negative)
   - Confidence percentage
   - Interactive gauge chart showing positive probability (0–100%)
   - Expandable section showing the cleaned text used for prediction

**Batch Mode:**
1. User enters multiple reviews, one per line
2. Clicks "Analyse Batch"
3. All reviews are processed simultaneously
4. Results shown as a color-highlighted table (green rows = positive, red rows = negative)
5. A summary bar chart shows the count of positive vs negative predictions

---

## 📊 Model Performance

Results on the 10,000-review test set:

| Metric | Score |
|---|---|
| Accuracy | ~90.6% |
| Precision (Positive) | ~0.906 |
| Recall (Positive) | ~0.908 |
| F1 Score (Positive) | ~0.907 |
| Precision (Negative) | ~0.906 |
| Recall (Negative) | ~0.904 |
| F1 Score (Negative) | ~0.905 |
| ROC-AUC | ~0.965 |

These results are consistent with published benchmarks for TF-IDF + Logistic Regression on the IMDb dataset.

---

## 🔍 How Prediction Works

When you type a review and click "Analyse Sentiment", here is exactly what happens internally:

```
User Input: "This movie was absolutely fantastic! The acting was superb."
     ↓
Step 1 — Lowercase:       "this movie was absolutely fantastic! the acting was superb."
     ↓
Step 2 — Remove HTML:     "this movie was absolutely fantastic! the acting was superb."
     ↓
Step 3 — Remove symbols:  "this movie was absolutely fantastic  the acting was superb "
     ↓
Step 4 — Normalize spaces:"this movie was absolutely fantastic the acting was superb"
     ↓
Step 5 — Remove stopwords:"movie absolutely fantastic acting superb"
     ↓
Step 6 — Lemmatize:       "movie absolutely fantastic acting superb"
     ↓
Step 7 — Filter short:    "movie absolutely fantastic acting superb"
     ↓
TF-IDF Vectorize:         [0, 0, 0.42, 0, 0.61, 0, 0.38, ...] (50,000 dimensions)
     ↓
Logistic Regression:      P(positive) = 0.97, P(negative) = 0.03
     ↓
Output:                   POSITIVE (97% confidence)
```

---

## ❓ Troubleshooting

**"Model not found" error in the app**
```bash
# Run the training pipeline first
python train_pipeline.py
```

**UnicodeEncodeError on Windows**
This can happen if your terminal uses cp1252 encoding. The pipeline has been patched to use ASCII-safe characters. If you still see this, run:
```bash
set PYTHONIOENCODING=utf-8
python train_pipeline.py
```

**Slow first run**
The first run downloads ~84MB of IMDb data from HuggingFace. Subsequent runs use the cached `data/reviews_raw.csv` and skip the download entirely.

**Port already in use**
```bash
streamlit run app.py --server.port 8502
```

**Package installation fails**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**IProgress / ipywidgets error in Jupyter notebook**
If you see `ImportError: IProgress not found` while running the notebook, it means the notebook was using `tqdm.notebook`. This has been fixed to use plain `tqdm`. If you still see it, restart the kernel and re-run all cells.

**Notebook asks "file changed on disk — revert or overwrite?"**
Always click **Revert** to load the latest version from disk.

---

## 📄 License

This project uses the [Stanford IMDb Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) which is freely available for research and educational use.

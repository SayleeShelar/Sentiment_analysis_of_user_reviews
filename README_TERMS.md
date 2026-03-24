# 📚 **ALL TECHNICAL TERMS EXPLAINED** (Beginner → Advanced)

**Every concept from the project, simple explanations + code + math**

## ## 🔤 **Alphabetical Glossary**

### **Accuracy** 
**Simple**: % correct predictions.
**Formula**: `(Correct) / (Total)`
**Example**: 9060/10000 = **90.6%**
**Code**: `classification_report()['accuracy']`

### **AUC (Area Under Curve)**
**Simple**: Model quality (1.0=perfect, 0.5=random).
**This project**: **0.965** = excellent.
**Why?** Works at ANY threshold (not just 50%).

### **Bigrams**
**Simple**: 2-word phrases.
**Example**: "not good" (negative), "highly recommend" (positive).
**Code**: `TfidfVectorizer(ngram_range=(1,2))`

### **Classification Report**
**Simple**: Table of 3 metrics per class.
```
              precision  recall  f1-score
positive      0.91      0.90    0.907
negative      0.90      0.91    0.905
```

### **Clean Text Pipeline** (7 Steps)
```
1. lower(): "Great!" → "great!"
2. HTML: "<br />" → " "
3. Symbols: "!!!" → " "
4. Spaces: "  a  b  " → "a b"
5. Stopwords: "the, is, and" → removed (NLTK list)
6. Lemmatize: "running"→"run", "movies"→"movie" (NLTK)
7. Short words: "ok", "it" → removed (len≤2)
```
**Code**:
```python
tokens = [lemma.lemmatize(w) for w in text.split() 
          if w not in STOPWORDS and len(w)>2]
```

### **Confusion Matrix** (2×2 Grid)
```
                Predicted
              Pos     Neg
Actual Pos   4525    475  ← False Negatives
     Neg     425   4575  ← False Positives
```
- **TP**: True Positive (correct pos)
- **TN**: True Negative (correct neg)  
- **FP**: False Positive (wrong pos)
- **FN**: False Negative (wrong neg)

### **F1-Score**
**Simple**: Balance of Precision + Recall.
**Formula**: `2 × (precision × recall) / (precision + recall)`
**Why?** Handles imbalanced classes.

### **joblib**
**Simple**: Save/load Python objects (faster than pickle).
**Code**:
```python
joblib.dump(model, "model.joblib")  # Save
model = joblib.load("model.joblib") # Load
```

### **Lemma/Lemmatization** (NLTK)
**Simple**: Base word form.
```
running → run
better → good
movies → movie
```
**Code**: `WordNetLemmatizer().lemmatize(w)`

### **Logistic Regression**
**Simple**: Probability from linear combo.
**Math**: `P(y=1) = 1/(1 + e^(-(w1*x1 + w2*x2 + ... + b)))`
**Code**: `LogisticRegression(C=5)` 
- **C=5**: Less regularization (fit data harder).
**Output**: Weights show word impact ("hate"=-1.2 → negative).

### **N-gram**
**Simple**: Word chunks.
- **Unigram** (n=1): "good", "bad"
- **Bigram** (n=2): "not good", "very bad"

### **NLTK** (Natural Language Toolkit)
**Simple**: Python library for text processing.
```
stopwords: ["the", "is", "and"]
WordNetLemmatizer: words → base form
```

### **Precision**
**Simple**: Of predicted positive, what % ARE positive?
**Formula**: `TP / (TP + FP)`
**Example**: Predict 5000 pos, 4550 correct = **91%**.

### **Recall** 
**Simple**: Of ACTUAL positive, what % found?
**Formula**: `TP / (TP + FN)`
**Example**: 5000 actual pos, 4525 found = **90.5%**.

### **ROC Curve**
**Simple**: True Positive Rate vs False Positive Rate.
```
Y-axis: Recall (TPR)
X-axis: 1-Specificity (FPR)
Diagonal: Random guess (AUC=0.5)
Our curve: AUC=0.965 (excellent)
```
**Code**: `roc_curve(y_test, y_proba), auc(fpr,tpr)`

### **sklearn** (Scikit-learn)
**Simple**: ML library.
```
TfidfVectorizer: Text → numbers
LogisticRegression: Train/predict
train_test_split: 80/20 split
classification_report: Metrics table
```

### **Specificity**
**Simple**: Of actual negative, % correctly negative.
**Formula**: `TN / (TN + FP)`

### **Stopwords** (NLTK)
**Simple**: Useless common words (179 English).
```
"the", "is", "in", "at", "which", "on"
```
**No signal** for sentiment.

### **Stratify** (train_test_split)
**Simple**: Keep same % pos/neg in train+test.
**Code**: `train_test_split(..., stratify=y)` → 50/50 both sets.

### **Streamlit** 
**Simple**: Python → web app (no HTML/JS).
**Code**: 
```python
st.title("Sentiment Analyzer")
review = st.text_area("Your review")
if st.button("Predict"):
    st.write(pred, confidence)
```

### **Sublinear TF**
**Simple**: `log(1+tf)` → dampen super-common words.
**Code**: `TfidfVectorizer(sublinear_tf=True)`

### **TF-IDF** (Term Frequency-Inverse Document Frequency)
**Simple**: Word importance score.
**Formula**:
```
TF: count in THIS review
IDF: log(total reviews / reviews with word)
TF-IDF = TF × IDF
```
**Example**:
```
Word "hate" in 1 review (rare): HIGH score
Word "the" in all reviews: LOW score
```
**Code**:
```python
TfidfVectorizer(
    max_features=50000,    # Top 50K words only
    ngram_range=(1,2),     # Words + 2-word phrases
    sublinear_tf=True      # log scaling
)
```
**Output**: Text → `[0.12, 0.0, 0.45, 0.0, ...]` (50K numbers).

### **Tqdm**
**Simple**: Progress bars.
**Code**: `df['clean_text'].progress_apply(clean)`

### **Train/Test Split**
**Simple**: 80% train model, 20% test accuracy.
**Code**: `train_test_split(test_size=0.2, random_state=42)`
- **random_state=42**: Same split every run.

### **WordCloud**
**Simple**: Words sized by frequency.
**Code**: `WordCloud().generate(text_blob)`

---

## ## 📊 **Metrics Table** (From Project)
```
              precision recall f1-score support
positive      0.906    0.908  0.907   5000
negative      0.905    0.903  0.904   5000
accuracy                          0.906  10000
ROC-AUC                            0.965
```

**All terms explained — no jargon left!**

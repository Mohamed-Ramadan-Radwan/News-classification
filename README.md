# 📰 BBC News Classifier

**Multinomial Naive Bayes — Built from Scratch**

Classifies BBC News articles into 5 categories:
`Business · Entertainment · Politics · Sport · Tech`

No `sklearn`, no `nltk`, no `spacy` — only `numpy` and `pandas`.

---

## Setup

```bash
pip install -r requirements.txt
```

## Download Dataset

Download from Kaggle:  
https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification

Place the file as: `data/bbc-text.csv`

## Train the Model

```bash
python train.py
```

Saves artifacts to `models/`.

## Run the App

```bash
streamlit run app/streamlit_app.py
```

## Run Tests

```bash
python tests/test_preprocessor.py
python tests/test_naive_bayes.py
```

---

## Project Structure

```
bbc-news-classifier/
├── data/
│   └── bbc-text.csv          ← download from Kaggle
├── src/
│   ├── preprocessor.py       ← text cleaning from scratch
│   ├── tfidf.py              ← TF-IDF from scratch
│   ├── naive_bayes.py        ← Naive Bayes from scratch
│   └── evaluator.py          ← metrics from scratch
├── models/                   ← saved after training
├── app/
│   ├── streamlit_app.py      ← main page
│   └── pages/
│       ├── 01_classifier.py  ← live classification
│       ├── 02_analysis.py    ← confusion matrix + metrics
│       └── 03_about_model.py ← model explanation
├── tests/
├── train.py
├── requirements.txt
└── README.md
```

## How It Works

1. **Preprocessing**: lowercase → remove punctuation/numbers → remove stop words → tokenize
2. **TF-IDF**: compute TF(w,d) × IDF(w) for each word in each document  
3. **Naive Bayes**: compute log P(c) + Σ log P(w|c) for each class, pick the max  
4. **Laplace Smoothing**: add α=1 to all word counts to avoid zero probabilities

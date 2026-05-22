"""
train.py — Train the full pipeline and save model artifacts
Run once: python train.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(__file__))
from src.preprocessor import TextPreprocessor
from src.tfidf import TFIDFVectorizer
from src.naive_bayes import MultinomialNaiveBayes
from src.evaluator import ModelEvaluator


DATA_PATH   = "D:\\Term2\\bbc-news-classifier_full\\bbc-news-classifier\\data\\bbc-text.csv"
MODELS_DIR  = "D:\\Term2\\bbc-news-classifier_full\\bbc-news-classifier\\models"
RANDOM_SEED = 42
TEST_SIZE   = 0.2

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    texts  = df['text'].tolist()
    labels = df['category'].tolist()
    classes = sorted(list(set(labels)))
    print(f"  {len(texts)} articles | {len(classes)} classes: {classes}")

    # ── 2. Preprocess ──────────────────────────────────────────
    print("Preprocessing...")
    preprocessor = TextPreprocessor(remove_stopwords=True)
    tokenized = preprocessor.process_batch(texts)

    # ── 3. Split ───────────────────────────────────────────────
    tok_train, tok_test, y_train, y_test = train_test_split(
        tokenized, labels, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    print(f"  Train: {len(tok_train)} | Test: {len(tok_test)}")

    # ── 4. TF-IDF ──────────────────────────────────────────────
    print("Building TF-IDF...")
    vectorizer = TFIDFVectorizer(max_features=15000, min_df=2)
    X_train = vectorizer.fit_transform(tok_train)
    X_test  = vectorizer.transform(tok_test)
    print(f"  Vocabulary size: {vectorizer.vocab_size}")
    print(f"  X_train shape: {X_train.shape}")

    # ── 5. Train Naive Bayes ───────────────────────────────────
    print("Training Naive Bayes (alpha=0.01)...")
    model = MultinomialNaiveBayes(alpha=0.01)
    model.fit(X_train, np.array(y_train))

    # ── 6. Evaluate ────────────────────────────────────────────
    print("Evaluating...")
    y_pred = model.predict(X_test)
    evaluator = ModelEvaluator(classes)
    report = evaluator.full_report(y_test, y_pred)
    evaluator.print_report(report)

    # ── 7. Save artifacts ──────────────────────────────────────
    print("Saving model artifacts...")

    model.save(f"{MODELS_DIR}/model_params.pkl")
    vectorizer.save(f"{MODELS_DIR}/tfidf_params.pkl")

    with open(f"{MODELS_DIR}/meta.pkl", 'wb') as f:
        pickle.dump({
            'classes': classes,
            'report': report,
            'vocab_size': vectorizer.vocab_size,
            'n_train': len(tok_train),
            'n_test': len(tok_test)
        }, f)

    # Save idx->word mapping for top words analysis
    idx_to_word = {v: k for k, v in vectorizer.vocabulary.items()}
    with open(f"{MODELS_DIR}/idx_to_word.pkl", 'wb') as f:
        pickle.dump(idx_to_word, f)

    print(f"\n Artifacts saved to {MODELS_DIR}/")
    print(f"Final accuracy: {report['accuracy']*100:.2f}%")
    print("\nRun the app with:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()

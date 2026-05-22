"""
tests/test_naive_bayes.py
"""

import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.naive_bayes import MultinomialNaiveBayes


def make_dummy_data():
    np.random.seed(42)
    X_sport    = np.random.dirichlet(np.ones(20), 30) * 10
    X_politics = np.random.dirichlet(np.ones(20), 30) * 10
    X = np.vstack([X_sport, X_politics])
    y = np.array(['sport'] * 30 + ['politics'] * 30)
    return X, y


def test_fit_sets_classes():
    X, y = make_dummy_data()
    nb = MultinomialNaiveBayes()
    nb.fit(X, y)
    assert set(nb.classes) == {'sport', 'politics'}

def test_predict_returns_correct_length():
    X, y = make_dummy_data()
    nb = MultinomialNaiveBayes()
    nb.fit(X, y)
    preds = nb.predict(X)
    assert len(preds) == len(y)

def test_predict_valid_classes():
    X, y = make_dummy_data()
    nb = MultinomialNaiveBayes()
    nb.fit(X, y)
    preds = nb.predict(X)
    assert all(p in ['sport', 'politics'] for p in preds)

def test_probabilities_sum_to_one():
    X, y = make_dummy_data()
    nb = MultinomialNaiveBayes()
    nb.fit(X, y)
    probas = nb.predict_proba(X[:5])
    for p_dict in probas:
        total = sum(p_dict.values())
        assert abs(total - 1.0) < 1e-5

def test_laplace_no_zero_probabilities():
    X, y = make_dummy_data()
    nb = MultinomialNaiveBayes(alpha=1.0)
    nb.fit(X, y)
    for cls in nb.classes:
        assert np.all(nb.class_log_likelihoods[cls] > -np.inf)

def test_accuracy_above_chance():
    X, y = make_dummy_data()
    nb = MultinomialNaiveBayes()
    nb.fit(X, y)
    preds = nb.predict(X)
    acc = sum(1 for t, p in zip(y, preds) if t == p) / len(y)
    assert acc > 0.5, f"Accuracy {acc:.2f} is below chance"


if __name__ == "__main__":
    tests = [test_fit_sets_classes, test_predict_returns_correct_length,
             test_predict_valid_classes, test_probabilities_sum_to_one,
             test_laplace_no_zero_probabilities, test_accuracy_above_chance]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")

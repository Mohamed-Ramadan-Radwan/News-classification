
import math
import pickle
import numpy as np
from collections import Counter


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):

        self.alpha = alpha
        self.classes = []
        self.class_priors = {}       # log P(c)
        self.class_word_counts = {}  # count of each word per class
        self.class_totals = {}       # total word count per class
        self.vocab_size = 0
        self.class_log_likelihoods = {}  # log P(w|c) for all words

    def fit(self, X: np.ndarray, y: np.ndarray):

        n_docs, self.vocab_size = X.shape
        self.classes = list(np.unique(y))

        for cls in self.classes:

            cls_mask = (y == cls)
            cls_docs = X[cls_mask]
            n_cls = cls_mask.sum()

            # Prior: log P(c) = log(n_docs_in_c / total_docs)
            self.class_priors[cls] = math.log(n_cls / n_docs)

            # Word counts per class (sum across all docs in class)
            word_counts = cls_docs.sum(axis=0)  # shape: (vocab_size,)
            self.class_word_counts[cls] = word_counts
            self.class_totals[cls] = word_counts.sum()

        # Precompute log P(w|c) for all classes and words
        self._compute_log_likelihoods()
        return self

    def _compute_log_likelihoods(self):

        for cls in self.classes:
            counts = self.class_word_counts[cls]
            total = self.class_totals[cls]
            # Add alpha (Laplace) to every word count
            smoothed = counts + self.alpha
            denominator = total + self.alpha * self.vocab_size
            self.class_log_likelihoods[cls] = np.log(smoothed / denominator)

    def _compute_posterior(self, x: np.ndarray) -> dict:
        """
        Compute log posterior for one document.
        log P(c|x) ∝ log P(c) + Σ x_i * log P(w_i|c)
        We use dot product since x_i weights already encode frequency.
        """
        posteriors = {}
        for cls in self.classes:
            log_prior = self.class_priors[cls]
            log_likelihood_sum = np.dot(x, self.class_log_likelihoods[cls])
            posteriors[cls] = log_prior + log_likelihood_sum
        return posteriors

    def predict(self, X: np.ndarray) -> list:
        predictions = []
        for i in range(X.shape[0]):
            posteriors = self._compute_posterior(X[i])
            predicted = max(posteriors, key=posteriors.get)
            predictions.append(predicted)
        return predictions

    def predict_proba(self, X: np.ndarray) -> list:
        all_probas = []
        for i in range(X.shape[0]):
            log_posts = self._compute_posterior(X[i])
            # Softmax over log posteriors → probabilities
            log_vals = np.array(list(log_posts.values()))
            # Subtract max for numerical stability
            log_vals -= log_vals.max()
            exp_vals = np.exp(log_vals)
            probs = exp_vals / exp_vals.sum()
            all_probas.append(dict(zip(log_posts.keys(), probs)))
        return all_probas

    def predict_single(self, x_vector: np.ndarray) -> tuple:
        posteriors = self._compute_posterior(x_vector)
        predicted_class = max(posteriors, key=posteriors.get)

        # Convert to probabilities
        log_vals = np.array(list(posteriors.values()))
        log_vals -= log_vals.max()
        exp_vals = np.exp(log_vals)
        probs = exp_vals / exp_vals.sum()
        prob_dict = dict(zip(posteriors.keys(), probs))

        return predicted_class, prob_dict

    def get_top_words_per_class(self, idx_to_word: dict, top_n=10) -> dict:
        result = {}
        for cls in self.classes:
            log_lk = self.class_log_likelihoods[cls]
            top_indices = np.argsort(log_lk)[-top_n:][::-1]
            result[cls] = [(idx_to_word.get(i, '?'), float(log_lk[i]))
                           for i in top_indices]
        return result

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'alpha': self.alpha,
                'classes': self.classes,
                'class_priors': self.class_priors,
                'class_word_counts': self.class_word_counts,
                'class_totals': self.class_totals,
                'vocab_size': self.vocab_size,
                'class_log_likelihoods': self.class_log_likelihoods
            }, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.alpha = data['alpha']
        self.classes = data['classes']
        self.class_priors = data['class_priors']
        self.class_word_counts = data['class_word_counts']
        self.class_totals = data['class_totals']
        self.vocab_size = data['vocab_size']
        self.class_log_likelihoods = data['class_log_likelihoods']
        return self

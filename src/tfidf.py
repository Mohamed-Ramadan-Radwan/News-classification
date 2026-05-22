"""
TF-IDF Vectorizer - built from scratch (numpy only)
"""

import math
import pickle
import numpy as np
from collections import Counter


class TFIDFVectorizer:
    def __init__(self, max_features=10000, min_df=2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary = {}       # word -> index
        self.idf_values = {}       # word -> idf score
        self.vocab_size = 0

    def _compute_tf(self, tokens: list) -> dict:
        """TF = count(word in doc) / total words in doc"""
        if not tokens:
            return {}
        count = Counter(tokens)
        total = len(tokens)
        return {word: freq / total for word, freq in count.items()}

    def _compute_idf(self, tokenized_docs: list):
        """IDF = log(N / df(word)) — computed over all training docs"""
        N = len(tokenized_docs)
        doc_freq = Counter()

        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for word in unique_tokens:
                doc_freq[word] += 1

        # Filter by min_df and build vocabulary
        filtered = {w: df for w, df in doc_freq.items() if df >= self.min_df}

        # Sort by frequency descending, take top max_features
        sorted_words = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        sorted_words = sorted_words[:self.max_features]

        self.vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_words)}
        self.vocab_size = len(self.vocabulary)

        # Compute IDF for each word in vocabulary
        for word in self.vocabulary:
            df = doc_freq[word]
            self.idf_values[word] = math.log((N + 1) / (df + 1)) + 1  # smooth IDF

    def fit(self, tokenized_docs: list):
        """Build vocabulary and compute IDF from training data"""
        self._compute_idf(tokenized_docs)
        return self

    def transform(self, tokenized_docs: list) -> np.ndarray:
        """Convert tokenized docs to TF-IDF matrix"""
        matrix = np.zeros((len(tokenized_docs), self.vocab_size), dtype=np.float32)

        for doc_idx, tokens in enumerate(tokenized_docs):
            tf = self._compute_tf(tokens)
            for word, tf_val in tf.items():
                if word in self.vocabulary:
                    word_idx = self.vocabulary[word]
                    idf_val = self.idf_values[word]
                    matrix[doc_idx, word_idx] = tf_val * idf_val

        return matrix

    def fit_transform(self, tokenized_docs: list) -> np.ndarray:
        """Fit then transform in one step"""
        self.fit(tokenized_docs)
        return self.transform(tokenized_docs)

    def get_feature_names(self) -> list:
        """Return words sorted by their index"""
        return [w for w, _ in sorted(self.vocabulary.items(), key=lambda x: x[1])]

    def get_top_words_per_class(self, class_word_scores: dict, top_n=15) -> dict:
        """Return top N words for each class based on scores"""
        result = {}
        idx_to_word = {v: k for k, v in self.vocabulary.items()}
        for cls, scores in class_word_scores.items():
            top_indices = np.argsort(scores)[-top_n:][::-1]
            result[cls] = [(idx_to_word[i], scores[i]) for i in top_indices if i in idx_to_word]
        return result

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'vocabulary': self.vocabulary,
                         'idf_values': self.idf_values,
                         'vocab_size': self.vocab_size,
                         'max_features': self.max_features,
                         'min_df': self.min_df}, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vocabulary = data['vocabulary']
        self.idf_values = data['idf_values']
        self.vocab_size = data['vocab_size']
        self.max_features = data['max_features']
        self.min_df = data['min_df']
        return self

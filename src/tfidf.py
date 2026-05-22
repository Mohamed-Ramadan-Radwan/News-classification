
import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFVectorizer:
    def __init__(self, max_features=10000, min_df=2):

        self.max_features = max_features
        self.min_df = min_df

        # sklearn vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df
        )

        self.vocabulary = {}
        self.idf_values = {}
        self.vocab_size = 0

    def fit(self, tokenized_docs: list):
        """
        Build vocabulary and compute IDF
        """

        # convert token lists -> strings
        docs = [' '.join(tokens) for tokens in tokenized_docs]

        # train vectorizer
        self.vectorizer.fit(docs)

        # save useful attributes
        self.vocabulary = self.vectorizer.vocabulary_

        self.idf_values = {
            word: idf
            for word, idf in zip(
                self.vectorizer.get_feature_names_out(),
                self.vectorizer.idf_
            )
        }

        self.vocab_size = len(self.vocabulary)

        return self

    def transform(self, tokenized_docs: list) -> np.ndarray:
        """
        Convert docs to TF-IDF matrix
        """

        docs = [' '.join(tokens) for tokens in tokenized_docs]

        matrix = self.vectorizer.transform(docs)

        return matrix.toarray().astype(np.float32)

    def fit_transform(self, tokenized_docs: list) -> np.ndarray:
        """
        Fit and transform in one step
        """

        docs = [' '.join(tokens) for tokens in tokenized_docs]

        matrix = self.vectorizer.fit_transform(docs)

        self.vocabulary = self.vectorizer.vocabulary_

        self.idf_values = {
            word: idf
            for word, idf in zip(
                self.vectorizer.get_feature_names_out(),
                self.vectorizer.idf_
            )
        }

        self.vocab_size = len(self.vocabulary)

        return matrix.toarray().astype(np.float32)

    def get_feature_names(self) -> list:
        """
        Return vocabulary words
        """
        return list(self.vectorizer.get_feature_names_out())

    def get_top_words_per_class(
        self,
        class_word_scores: dict,
        top_n=15
    ) -> dict:
        """
        Return top N words per class
        """

        result = {}

        feature_names = self.get_feature_names()

        for cls, scores in class_word_scores.items():

            top_indices = np.argsort(scores)[-top_n:][::-1]

            result[cls] = [
                (feature_names[i], scores[i])
                for i in top_indices
            ]

        return result

    def save(self, path: str):
        """
        Save vectorizer
        """

        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: str):
        """
        Load vectorizer
        """

        with open(path, 'rb') as f:
            self.vectorizer = pickle.load(f)

        self.vocabulary = self.vectorizer.vocabulary_

        self.idf_values = {
            word: idf
            for word, idf in zip(
                self.vectorizer.get_feature_names_out(),
                self.vectorizer.idf_
            )
        }

        self.vocab_size = len(self.vocabulary)

        return self

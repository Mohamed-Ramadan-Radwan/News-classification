
import re
import string


STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "it", "its", "this",
    "that", "these", "those", "i", "me", "my", "we", "our", "you", "your",
    "he", "she", "they", "them", "his", "her", "their", "what", "which",
    "who", "when", "where", "how", "all", "each", "any", "both", "more",
    "also", "not", "no", "so", "as", "up", "out", "if", "about", "into",
    "than", "then", "there", "after", "over", "new", "just", "said", "one",
    "two", "three", "like", "get", "go", "make", "see", "know", "time",
    "year", "way", "say", "come", "take", "us", "him", "been", "while",
    "before", "had", "has", "have", "do", "did", "does", "such", "through"
}


class TextPreprocessor:
    def __init__(self, remove_stopwords=True, min_word_length=2):
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length

    def clean(self, text: str) -> str:
        """Full cleaning pipeline"""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> list:
        """Split text into tokens"""
        return text.split()

    def remove_stops(self, tokens: list) -> list:
        """Remove stop words and short tokens"""
        return [
            t for t in tokens
            if t not in STOP_WORDS and len(t) >= self.min_word_length
        ]

    def process(self, text: str) -> list:
        """Full pipeline: clean → tokenize → remove stops"""
        cleaned = self.clean(text)
        tokens = self.tokenize(cleaned)
        if self.remove_stopwords:
            tokens = self.remove_stops(tokens)
        return tokens

    def process_batch(self, texts: list) -> list:
        """Process a list of texts"""
        return [self.process(t) for t in texts]

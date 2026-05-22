"""
tests/test_preprocessor.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessor import TextPreprocessor


def test_lowercase():
    p = TextPreprocessor()
    assert p.clean("Hello WORLD") == "hello world"

def test_removes_punctuation():
    p = TextPreprocessor()
    result = p.clean("Hello, world!")
    assert ',' not in result and '!' not in result

def test_removes_numbers():
    p = TextPreprocessor()
    result = p.clean("3 goals in 90 minutes")
    assert '3' not in result and '90' not in result

def test_removes_stopwords():
    p = TextPreprocessor(remove_stopwords=True)
    tokens = p.process("the cat sat on the mat")
    assert "the" not in tokens
    assert "on" not in tokens

def test_min_word_length():
    p = TextPreprocessor(min_word_length=3)
    tokens = p.process("I am a good programmer")
    # "i", "am", "a" should be filtered
    assert all(len(t) >= 3 for t in tokens)

def test_empty_text():
    p = TextPreprocessor()
    result = p.process("")
    assert result == []


if __name__ == "__main__":
    tests = [test_lowercase, test_removes_punctuation, test_removes_numbers,
             test_removes_stopwords, test_min_word_length, test_empty_text]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")

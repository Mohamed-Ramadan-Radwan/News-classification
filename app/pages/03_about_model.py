"""
pages/03_about_model.py — Explain the model internals with real examples
"""

import os
import sys
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.naive_bayes import MultinomialNaiveBayes
from src.tfidf import TFIDFVectorizer

st.set_page_config(page_title="Model Explained", page_icon="🧠", layout="wide")

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
COLORS = {"business": "#1f77b4", "entertainment": "#e377c2",
          "politics": "#ff7f0e", "sport": "#2ca02c", "tech": "#9467bd"}


@st.cache_resource
def load_artifacts():
    model = MultinomialNaiveBayes()
    model.load(os.path.join(MODELS_DIR, 'model_params.pkl'))
    vectorizer = TFIDFVectorizer()
    vectorizer.load(os.path.join(MODELS_DIR, 'tfidf_params.pkl'))
    with open(os.path.join(MODELS_DIR, 'idx_to_word.pkl'), 'rb') as f:
        idx_to_word = pickle.load(f)
    return model, vectorizer, idx_to_word


def plot_top_words(top_words_dict, classes):
    fig, axes = plt.subplots(1, len(classes), figsize=(14, 4))
    for ax, cls in zip(axes, classes):
        words_scores = top_words_dict[cls][:10]
        words  = [w for w, _ in words_scores][::-1]
        scores = [s for _, s in words_scores][::-1]
        color  = COLORS.get(cls, '#888')
        bars = ax.barh(words, scores, color=color, alpha=0.8)
        ax.set_title(cls.capitalize(), fontsize=11, fontweight='bold', color=color)
        ax.set_xlabel("log P(w|c)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis='x', linestyle='--', alpha=0.4)
    fig.suptitle("Top 10 Most Indicative Words per Category", fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


# ── UI ─────────────────────────────────────────────────────────

st.title("🧠 How the Model Works")

if not os.path.exists(os.path.join(MODELS_DIR, 'model_params.pkl')):
    st.error("⚠️ Model not trained yet. Run `python train.py` first.")
    st.stop()

model, vectorizer, idx_to_word = load_artifacts()

# ── Naive Bayes Explanation
st.markdown("## 1. Naive Bayes — The Math")
st.markdown("""
Naive Bayes classifies a document by finding the class with the **highest posterior probability**:

$$\\hat{c} = \\arg\\max_c \\; P(c) \\cdot \\prod_{w \\in d} P(w \\mid c)$$

We take **log** of both sides to avoid numerical underflow:

$$\\hat{c} = \\arg\\max_c \\; \\log P(c) + \\sum_{w \\in d} \\log P(w \\mid c)$$

Where:
- $P(c)$ = **Prior** — how common is class $c$ in training data?
- $P(w|c)$ = **Likelihood** — how often does word $w$ appear in class $c$?
""")

# ── Priors
st.markdown("### Class Priors — log P(c)")
prior_data = {cls: np.exp(lp) for cls, lp in model.class_priors.items()}
cols = st.columns(len(prior_data))
for col, (cls, prob) in zip(cols, sorted(prior_data.items())):
    col.metric(cls.capitalize(), f"{prob*100:.1f}%",
               help=f"log P({cls}) = {model.class_priors[cls]:.4f}")

st.divider()

# ── Laplace Smoothing
st.markdown("## 2. Laplace Smoothing")
st.markdown("""
Without smoothing, if a word never appeared in class $c$ during training,  
$P(w|c) = 0$ and the **entire posterior becomes 0** — catastrophic!

**Laplace smoothing** adds a small count $\\alpha=1$ to every word:

$$P(w \\mid c) = \\frac{\\text{count}(w, c) + \\alpha}{\\text{total words in } c + \\alpha \\times |V|}$$

This ensures **no word ever gets zero probability**.
""")

col1, col2 = st.columns(2)
with col1:
    st.info(f"**Vocabulary size** |V| = {vectorizer.vocab_size:,} words")
with col2:
    st.info(f"**Alpha (α)** = {model.alpha} (standard Laplace)")

st.divider()

# ── TF-IDF
st.markdown("## 3. TF-IDF — Feature Representation")
st.markdown("""
Instead of raw word counts, we use **TF-IDF** to weight each word:

| Formula | Meaning |
|---|---|
| $TF(w,d) = \\frac{\\text{count}(w,d)}{\\text{len}(d)}$ | How often does word $w$ appear in document $d$? |
| $IDF(w) = \\log\\frac{N+1}{df(w)+1} + 1$ | How rare is word $w$ across all documents? |
| $TF\\text{-}IDF = TF \\times IDF$ | High if word is frequent in doc but rare overall |

Common words like "the", "is" → **low IDF** (appear everywhere)  
Specific words like "parliament", "bitcoin" → **high IDF** (appear rarely)
""")

st.divider()

# ── Top words per class
st.markdown("## 4. Most Indicative Words per Category")
st.markdown("These are the words with the **highest log-likelihood** per class — the words that most strongly push the model toward each category.")

top_words = model.get_top_words_per_class(idx_to_word, top_n=10)
fig = plot_top_words(top_words, model.classes)
st.pyplot(fig, use_container_width=True)

st.divider()

# ── Why "Naive"?
st.markdown("## 5. Why is it called 'Naive'?")
st.info("""
The model assumes all words are **conditionally independent** given the class:

$$P(w_1, w_2, ..., w_n \\mid c) = P(w_1 \\mid c) \\cdot P(w_2 \\mid c) \\cdots P(w_n \\mid c)$$

This is **naive** because in reality words are not independent.  
"Prime Minister" being two words that often appear together is ignored.  

Despite this strong assumption, Naive Bayes works **surprisingly well** for text classification
because the independence assumption helps avoid overfitting on small datasets.
""")

"""
pages/01_classifier.py — Live classification page
"""

import os
import sys
import pickle
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.preprocessor import TextPreprocessor
from src.tfidf import TFIDFVectorizer
from src.naive_bayes import MultinomialNaiveBayes

st.set_page_config(page_title="Classifier", page_icon="🔍", layout="wide")

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

CATEGORY_COLORS = {
    "business":      "#1f77b4",
    "entertainment": "#e377c2",
    "politics":      "#ff7f0e",
    "sport":         "#2ca02c",
    "tech":          "#9467bd"
}

CATEGORY_ICONS = {
    "business":      "💼",
    "entertainment": "🎬",
    "politics":      "🏛️",
    "sport":         "⚽",
    "tech":          "💻"
}

SAMPLE_TEXTS = {
    "📈 Business": "The Bank of England has raised interest rates to 5.25%, the highest level in 15 years, as it continues its battle against inflation. Economists warn the move could slow economic growth.",
    "🎬 Entertainment": "The Oscars ceremony drew record viewers as the best picture award went to an independent film. Hollywood stars celebrated at the after-party while critics praised the evening's surprises.",
    "🏛️ Politics": "The prime minister announced sweeping reforms to the healthcare system, promising billions in new funding. Opposition parties called the proposals insufficient ahead of the upcoming general election.",
    "⚽ Sport": "Manchester United secured a dramatic last-minute victory with a stunning header in the final moments. The manager praised the team's resilience after a difficult first half at Old Trafford.",
    "💻 Tech": "Apple unveiled its latest MacBook Pro featuring the new M3 chip, promising significantly faster performance and battery life. The device will go on sale next month at a starting price of £1,799."
}


@st.cache_resource
def load_model():
    model = MultinomialNaiveBayes()
    model.load(os.path.join(MODELS_DIR, 'model_params.pkl'))
    vectorizer = TFIDFVectorizer()
    vectorizer.load(os.path.join(MODELS_DIR, 'tfidf_params.pkl'))
    preprocessor = TextPreprocessor()
    return model, vectorizer, preprocessor


def classify_text(text, model, vectorizer, preprocessor):
    tokens = preprocessor.process(text)
    if not tokens:
        return None, None
    # Build TF-IDF vector for single document
    x_vec = vectorizer.transform([tokens])[0]
    pred_class, prob_dict = model.predict_single(x_vec)
    return pred_class, prob_dict


# ── UI ─────────────────────────────────────────────────────────

st.title("🔍 News Article Classifier")
st.markdown("Type or paste any news article and the model will classify it instantly.")

# Check if model is trained
if not os.path.exists(os.path.join(MODELS_DIR, 'model_params.pkl')):
    st.error("⚠️ Model not trained yet. Run `python train.py` first.")
    st.stop()

model, vectorizer, preprocessor = load_model()

# Sample buttons
st.markdown("**Try a sample:**")
cols = st.columns(len(SAMPLE_TEXTS))
for i, (label, sample) in enumerate(SAMPLE_TEXTS.items()):
    with cols[i]:
        if st.button(label, use_container_width=True):
            st.session_state['input_text'] = sample

# Text input
text_input = st.text_area(
    "Enter news article text:",
    value=st.session_state.get('input_text', ''),
    height=180,
    placeholder="Paste your news article here..."
)

classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)

if classify_btn or st.session_state.get('input_text', ''):
    if text_input.strip():
        pred_class, prob_dict = classify_text(text_input, model, vectorizer, preprocessor)

        if pred_class is None:
            st.warning("Text too short or contains only stop words. Please add more content.")
        else:
            st.divider()
            icon = CATEGORY_ICONS.get(pred_class, "📰")
            color = CATEGORY_COLORS.get(pred_class, "#888")

            # Main result
            st.markdown(f"""
            <div style="background:{color}18; border-left: 4px solid {color};
                        padding: 16px 20px; border-radius: 8px; margin-bottom: 16px;">
                <div style="font-size: 14px; color: #888; margin-bottom: 4px;">Predicted Category</div>
                <div style="font-size: 28px; font-weight: 600; color: {color};">
                    {icon} {pred_class.upper()}
                </div>
                <div style="font-size: 13px; color: #888; margin-top: 4px;">
                    Confidence: {prob_dict[pred_class]*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown("**Probability distribution across all categories:**")
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            for cls, prob in sorted_probs:
                c = CATEGORY_COLORS.get(cls, "#888")
                icon_c = CATEGORY_ICONS.get(cls, "📰")
                col_label, col_bar, col_pct = st.columns([2, 6, 1])
                with col_label:
                    st.markdown(f"{icon_c} **{cls}**")
                with col_bar:
                    st.progress(float(prob))
                with col_pct:
                    st.markdown(f"`{prob*100:.1f}%`")
    else:
        st.warning("Please enter some text to classify.")

"""
streamlit_app.py — Main entry point for BBC News Classifier
Run: streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="BBC News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📰 BBC News Classifier")
st.markdown("#### Multinomial Naive Bayes — Built from Scratch")

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**🔍 Classifier**\n\nClassify any news article into one of 5 categories in real-time.")
with col2:
    st.success("**📊 Results & Analysis**\n\nConfusion matrix, accuracy, precision, recall, and F1 per class.")
with col3:
    st.warning("**🧠 Model Explained**\n\nUnderstand how Naive Bayes and TF-IDF work under the hood.")

st.divider()

st.markdown("""
### About This Project

This classifier is built **entirely from scratch** using only `numpy` and `pandas` — no `sklearn`, no `nltk`, no `spacy`.

| Component | Description |
|---|---|
| **Dataset** | BBC News — 2,225 articles across 5 categories |
| **Algorithm** | Multinomial Naive Bayes with Laplace Smoothing |
| **Features** | TF-IDF (custom implementation) |
| **Categories** | Business · Entertainment · Politics · Sport · Tech |

**Use the sidebar** to navigate between pages.
""")

st.markdown("---")
st.caption("Built from scratch · numpy · pandas · streamlit")

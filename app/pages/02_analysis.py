"""
pages/02_analysis.py — Model results and visual analysis
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

st.set_page_config(page_title="Results & Analysis", page_icon="📊", layout="wide")

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

COLORS = ["#1f77b4", "#e377c2", "#ff7f0e", "#2ca02c", "#9467bd"]

@st.cache_data
def load_meta():
    with open(os.path.join(MODELS_DIR, 'meta.pkl'), 'rb') as f:
        return pickle.load(f)


def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight='bold')

    thresh = cm.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    return fig


def plot_per_class_metrics(per_class, classes):
    metrics = ['precision', 'recall', 'f1']
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, metric in enumerate(metrics):
        vals = [per_class[c][metric] for c in classes]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(),
                      color=COLORS[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Precision, Recall & F1 per Category", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# ── UI ─────────────────────────────────────────────────────────

st.title("📊 Results & Analysis")

if not os.path.exists(os.path.join(MODELS_DIR, 'meta.pkl')):
    st.error("⚠️ Model not trained yet. Run `python train.py` first.")
    st.stop()

meta   = load_meta()
report = meta['report']
classes = meta['classes']

# ── Top-line metrics
st.markdown("### Overall Performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy",  f"{report['accuracy']*100:.2f}%")
c2.metric("Macro F1",  f"{report['macro_avg']['f1']*100:.2f}%")
c3.metric("Training samples", meta['n_train'])
c4.metric("Test samples",     meta['n_test'])

st.divider()

# ── Two-column charts
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Confusion Matrix")
    cm = report['confusion_matrix']
    fig_cm = plot_confusion_matrix(cm, classes)
    st.pyplot(fig_cm, use_container_width=True)

with col_right:
    st.markdown("#### Per-class Metrics")
    fig_bar = plot_per_class_metrics(report['per_class'], classes)
    st.pyplot(fig_bar, use_container_width=True)

st.divider()

# ── Per-class table
st.markdown("### Detailed Metrics Table")
rows = []
for cls in classes:
    m = report['per_class'][cls]
    rows.append({
        "Category":  cls.capitalize(),
        "Precision": f"{m['precision']:.4f}",
        "Recall":    f"{m['recall']:.4f}",
        "F1 Score":  f"{m['f1']:.4f}",
        "Support":   m['support']
    })
# Add averages
rows.append({
    "Category":  "Macro avg",
    "Precision": f"{report['macro_avg']['precision']:.4f}",
    "Recall":    f"{report['macro_avg']['recall']:.4f}",
    "F1 Score":  f"{report['macro_avg']['f1']:.4f}",
    "Support":   meta['n_test']
})

df_table = pd.DataFrame(rows)
st.dataframe(df_table, use_container_width=True, hide_index=True)

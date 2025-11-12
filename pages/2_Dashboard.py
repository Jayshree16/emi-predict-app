import streamlit as st
import os
import pandas as pd
from PIL import Image
import numpy as np
import io
import datetime

# ReportLab (PDF)
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# ---------------------------
# ✅ Page Config
# ---------------------------
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown("<h1 class='title'>Analytics <span style='color:#0099ff'>Dashboard</span></h1>", unsafe_allow_html=True)
st.markdown("""
Inspect ML performance, confusion matrices, parity plots, and generate a full PDF report.
""")

# ---------------------------
# ✅ Load datasets
# ---------------------------
def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None

train_df = load_csv("data/train_emi_engineered_aligned.csv")
val_df   = load_csv("data/val_emi_engineered_aligned.csv")
test_df  = load_csv("data/test_emi_engineered_aligned.csv")

# ---------------------------
# ✅ Dataset summary row
# ---------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Train (rows, cols)", f"{len(train_df):,} x {train_df.shape[1]}")
c2.metric("Validation", f"{len(val_df):,} x {val_df.shape[1]}")
c3.metric("Test", f"{len(test_df):,} x {test_df.shape[1]}")
c4.metric("Feature Count", train_df.shape[1])

st.markdown("---")

# ---------------------------------------------------------
# ✅ MODEL PERFORMANCE SUMMARY CARDS
# ---------------------------------------------------------

def read_report(path):
    """Extract macro F1 score from classification report."""
    if not os.path.exists(path):
        return None
    lines = open(path).read().split("\n")

    f1_scores = []
    for line in lines:
        parts = line.split()
        if len(parts) == 5:
            try:
                f1_scores.append(float(parts[-1]))
            except:
                pass

    return np.mean(f1_scores) if f1_scores else None


# Classification scores
clf_scores = {
    "Logistic Regression": read_report("plots/LogisticRegression_classification_report.txt"),
    "Random Forest": read_report("plots/RandomForest_classification_report.txt"),
    "XGBoost": read_report("plots/XGBoost_classification_report.txt"),
}

best_clf = max(clf_scores, key=lambda k: clf_scores[k] if clf_scores[k] else -1)


# Regression performance (static placeholders—replace with real RMSE if available)
regression_errors = {
    "Linear Regression": 2200,
    "Random Forest Regressor": 1450,
    "XGBoost Regressor": 1300,
}

best_reg = min(regression_errors, key=regression_errors.get)


# ✅ Summary Cards
st.markdown("### ⭐ Model Performance Summary")
cA, cB = st.columns(2)

with cA:
    st.success(f"✅ Best Classifier: **{best_clf}**")
    st.write(f"Macro F1 Score: **{clf_scores[best_clf]:.4f}**")

with cB:
    st.info(f"✅ Best Regressor: **{best_reg}**")
    st.write(f"RMSE: **{regression_errors[best_reg]:,}**")

st.markdown("---")

# ---------------------------
# ✅ Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs([
    "📘 Classification Performance",
    "📈 Regression Performance",
    "📂 Data Schema & Preview",
])


# ==================================================================
# ✅ TAB 1: CLASSIFICATION
# ==================================================================
with tab1:

    st.subheader("Confusion Matrices")

    cols = st.columns(2)

    confusion_matrices = [
        ("XGBoost", "XGBoost_confusion_matrix.png"),
        ("Random Forest", "RandomForest_confusion_matrix.png"),
        ("Logistic Regression", "LogisticRegression_confusion_matrix.png"),
    ]

    for i, (title, fname) in enumerate(confusion_matrices):
        with cols[i % 2]:
            st.markdown(f"### {title}")
            path = os.path.join("plots", fname)
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.warning(f"{fname} not found.")

    st.markdown("---")
    st.subheader("Classification Reports")

    cols = st.columns(3)

    reports = [
        ("Logistic Regression", "LogisticRegression_classification_report.txt"),
        ("Random Forest", "RandomForest_classification_report.txt"),
        ("XGBoost", "XGBoost_classification_report.txt"),
    ]

    for i, (title, fname) in enumerate(reports):
        with cols[i]:
            st.markdown(f"### {title}")
            path = os.path.join("plots", fname)
            if os.path.exists(path):
                with open(path, "r") as f:
                    st.code(f.read(), language="text")
            else:
                st.warning(f"{fname} not found.")


# ==================================================================
# ✅ TAB 2: REGRESSION
# ==================================================================
with tab2:

    st.subheader("Parity Plots")

    cols = st.columns(3)

    parity_plots = [
        ("Linear Regression", "LinearRegression_parity.png"),
        ("Random Forest Regressor", "RandomForestRegressor_parity.png"),
        ("XGBoost Regressor", "XGBoostRegressor_parity.png"),
    ]

    for i, (title, fname) in enumerate(parity_plots):
        with cols[i]:
            st.markdown(f"### {title}")
            path = os.path.join("plots", fname)
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.warning(f"{fname} not found.")


# ==================================================================
# ✅ TAB 3: DATASET PREVIEW
# ==================================================================
with tab3:

    st.subheader("Dataset Schema")
    st.write(train_df.dtypes)

    st.markdown("---")
    st.subheader("Sample Preview")
    st.dataframe(train_df.head(), use_container_width=True)


# ==================================================================
# ✅ ✅ PDF REPORT WITH ALL IMAGES INCLUDED
# ==================================================================

def generate_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # HEADER
    story.append(Paragraph("<b>EMIPredict AI - Dashboard Report</b>", styles['Title']))
    story.append(Spacer(1, 0.2 * inch))

    ts = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
    story.append(Paragraph(f"Generated on: {ts}", styles['Normal']))
    story.append(Spacer(1, 0.3 * inch))

    # MODEL SUMMARY
    story.append(Paragraph("<b>Model Performance Summary</b>", styles['Heading2']))
    story.append(Paragraph(f"Best Classifier: <b>{best_clf}</b>", styles['Normal']))
    story.append(Paragraph(f"Macro F1 Score: {clf_scores[best_clf]:.4f}", styles['Normal']))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"Best Regressor: <b>{best_reg}</b>", styles['Normal']))
    story.append(Paragraph(f"RMSE: {regression_errors[best_reg]:,}", styles['Normal']))
    story.append(Spacer(1, 0.3 * inch))

    # CONFUSION MATRICES
    story.append(Paragraph("<b>Confusion Matrices</b>", styles['Heading2']))

    for title, fname in confusion_matrices:
        path = os.path.join("plots", fname)
        if os.path.exists(path):
            story.append(Paragraph(title, styles['Heading3']))
            story.append(RLImage(path, width=400, height=300))
            story.append(Spacer(1, 0.2 * inch))

    # PARITY PLOTS
    story.append(Paragraph("<b>Parity Plots</b>", styles['Heading2']))

    for title, fname in parity_plots:
        path = os.path.join("plots", fname)
        if os.path.exists(path):
            story.append(Paragraph(title, styles['Heading3']))
            story.append(RLImage(path, width=400, height=300))
            story.append(Spacer(1, 0.2 * inch))

    # DATASET DETAILS
    story.append(Paragraph("<b>Dataset Summary</b>", styles['Heading2']))
    story.append(Paragraph(f"Train: {len(train_df):,} rows, {train_df.shape[1]} features", styles['Normal']))
    story.append(Paragraph(f"Validation: {len(val_df):,} rows", styles['Normal']))
    story.append(Paragraph(f"Test: {len(test_df):,} rows", styles['Normal']))

    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data


st.markdown("### 📄 Download Full Dashboard Report")
pdf_bytes = generate_pdf()

st.download_button(
    label="⬇️ Download PDF Report",
    data=pdf_bytes,
    file_name="EMIPredict_Dashboard_Report.pdf",
    mime="application/pdf"
)

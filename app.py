import streamlit as st
from pathlib import Path


from download_models import download_models_if_missing
download_models_if_missing()

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# CSS Loader
# ----------------------------
def load_css():
    css_path = Path("style.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ----------------------------
# Card Equal Height CSS
# ----------------------------
LOCAL_CSS = """
/* Make all cards same height */
.equal-card {
    min-height: 230px;          /* ✅ Final perfectly balanced height */
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

/* Title */
.card-title{
  margin: 0 0 .35rem 0;
}

/* Muted text */
.card-muted{
  color: rgba(2,6,23,.65);
  margin: 0;
}
"""

st.markdown(f"<style>{LOCAL_CSS}</style>", unsafe_allow_html=True)
load_css()

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    logo = Path("assets/logo.png")
    if logo.exists():
        st.image(str(logo), width=90)

    st.markdown("## EMIPredict AI")
    st.markdown("---")

    st.page_link("pages/1_Eligibility_Predictor.py", label="🚀 Eligibility & EMI Predictor")
    st.page_link("pages/2_Dashboard.py", label="📊 Analytics Dashboard")
    st.page_link("pages/3_Admin_Panel.py", label="🛠 Admin Panel")


# ----------------------------
# Hero Section
# ----------------------------
left, right = st.columns([1.4, 1])

with left:
    st.markdown("## **EMIPredict AI**")
    st.markdown("""
An intelligent platform for real-time **Loan Eligibility**, **EMI Affordability**,  
and **Financial Risk** assessment — powered by multiple ML models.
    """)
    st.write("")
    st.write("")
    st.markdown("### 👉 Start exploring the app below")

with right:
    st.info("### Multi-Model Stack\nRF • XGBoost • Logistic/Linear • RF Regressor • XGB Regressor")
    st.info("### Tracking & Ops\nMLflow • Artifacts • Metrics")

st.write("")

# ----------------------------
# Cards Row (ALL Equal Height)
# ----------------------------
c1, c2, c3 = st.columns(3)

# ---------- CARD 1 ----------
with c1:
    with st.container(border=True):
        st.markdown("""
        <div class="equal-card">
            <div>
                <h3 class="card-title">🚀 Eligibility Predictor</h3>
                <p class="card-muted">
                    Classifies applicants as <b>Eligible</b>, <b>High Risk</b>, or <b>Not Eligible</b>.
                </p>
            </div>
            <div>
        """, unsafe_allow_html=True)

        st.page_link("pages/1_Eligibility_Predictor.py", label="✨ Start Prediction")

        st.markdown("</div></div>", unsafe_allow_html=True)

# ---------- CARD 2 ----------
with c2:
    with st.container(border=True):
        st.markdown("""
        <div class="equal-card">
            <div>
                <h3 class="card-title">📘 EMI Recommendation</h3>
                <p class="card-muted">
                    Predicts safe <b>Monthly EMI</b> using regression models.
                </p>
            </div>
            <div>
        """, unsafe_allow_html=True)

        st.page_link("pages/1_Eligibility_Predictor.py", label="📘 Estimate EMI")

        st.markdown("</div></div>", unsafe_allow_html=True)

# ---------- CARD 3 ----------
with c3:
    with st.container(border=True):
        st.markdown("""
        <div class="equal-card">
            <div>
                <h3 class="card-title">📊 Analytics Dashboard</h3>
                <p class="card-muted">
                    Explore ML performance, confusion matrices, and metrics.
                </p>
            </div>
            <div>
        """, unsafe_allow_html=True)

        st.page_link("pages/2_Dashboard.py", label="📊 View Analytics")

        st.markdown("</div></div>", unsafe_allow_html=True)

# ----------------------------
# About
# ----------------------------
st.write("## About EMIPredict AI")
st.markdown("""
EMIPredict AI is a modern fintech solution designed to evaluate credit risk and loan affordability.
It uses ML models, aligned datasets, and dashboards for deep financial insights.
""")
st.markdown("""
- 🔐 Clean & aligned data pipeline  
- 🤖 Loan Eligibility + EMI prediction  
- 📈 MLflow-backed experiment tracking  
""")

# ----------------------------
# Footer
# ----------------------------
st.markdown(
    "<br><div style='text-align:center;opacity:0.6'>© 2025 EMIPredict AI • Built by Jayshree Pawar</div>",
    unsafe_allow_html=True,
)

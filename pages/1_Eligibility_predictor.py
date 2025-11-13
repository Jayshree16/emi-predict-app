import streamlit as st
import joblib, json, math, requests
from pathlib import Path
import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime

# ---------------------------
# Google Drive Model IDs
# ---------------------------
DRIVE_MODELS = {
    "clf": "12OJpdV-0int-VZlP6h4sh7YgsIfiEzMq",          # best_classification_model_new.joblib
    "reg": "1n6TQjMc2GCd1SPmr_aIzONb0uDp6WOzs",         # best_regression_model_new.joblib
    "le":  "1NBRmnhWk_a8le-v7uR1tJ89yEB__ax5e"          # label_encoder.joblib
}

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------
# Download Utility
# ---------------------------
def download_from_drive(file_id, save_path):
    """Download file from Google Drive only if not already downloaded."""
    if save_path.exists():
        return

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)

    with open(save_path, "wb") as f:
        f.write(response.content)


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Eligibility & EMI Predictor", page_icon="🚀", layout="wide")


# ---------------------------
# Load Models (AUTO-DOWNLOAD)
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_models():

    clf_path = MODELS_DIR / "best_classification_model_new.joblib"
    reg_path = MODELS_DIR / "best_regression_model_new.joblib"
    le_path  = MODELS_DIR / "label_encoder.joblib"

    # download if missing
    download_from_drive(DRIVE_MODELS["clf"], clf_path)
    download_from_drive(DRIVE_MODELS["reg"], reg_path)
    download_from_drive(DRIVE_MODELS["le"],  le_path)

    # load
    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    le  = joblib.load(le_path)

    return clf, reg, le, clf_path.name, reg_path.name


# ---------------------------
# Helper functions
# ---------------------------
def inr(x):
    try:
        return f"₹{int(round(float(x))):,}"
    except Exception:
        return "₹0"


def engineered_row(ui):
    """Build the 80-feature row used during training"""

    total_exp = ui['groceries_utilities'] + ui['other_monthly_expenses']
    dti  = (ui['existing_emi'] or 0) / (ui['monthly_salary'] + 1e-6)
    eti  = total_exp / (ui['monthly_salary'] + 1e-6)
    aff  = (ui['monthly_salary'] - total_exp - (ui['existing_emi'] or 0)) / (ui['monthly_salary'] + 1e-6)

    credit_risk = ui['credit_score'] / 850
    emp_stab    = ui['years_of_employment'] / (max(1, ui['years_of_employment']))
    bank_norm   = ui['bank_balance'] / (max(1, ui['bank_balance']))
    fin_idx     = 0.5*credit_risk + 0.3*emp_stab + 0.2*bank_norm

    base = {
        "age": ui['age'],
        "monthly_salary": ui['monthly_salary'],
        "years_of_employment": ui['years_of_employment'],
        "monthly_rent": ui['monthly_rent'],
        "family_size": ui['family_size'],
        "dependents": ui['dependents'],
        "groceries_utilities": ui['groceries_utilities'],
        "other_monthly_expenses": ui['other_monthly_expenses'],
        "current_emi_amount": ui['existing_emi'],
        "credit_score": ui['credit_score'],
        "bank_balance": ui['bank_balance'],
        "emergency_fund": ui['emergency_fund'],
        "requested_amount": ui['requested_amount'],
        "requested_tenure": ui['requested_tenure'],

        # engineered
        "total_expenses": total_exp,
        "debt_to_income": dti,
        "expense_to_income": eti,
        "affordability_ratio": aff,
        "credit_risk_score": credit_risk,
        "employment_stability": emp_stab,
        "bank_balance_norm": bank_norm,
        "financial_stability_index": fin_idx,
        "income_x_credit": ui['monthly_salary'] * credit_risk,
        "income_x_stability": ui['monthly_salary'] * emp_stab,
        "debt_x_expense": dti * eti,
    }

    cats = {
        "existing_loans_No": 1 if ui['existing_emi'] == 0 else 0,
        "existing_loans_Yes": 0 if ui['existing_emi'] == 0 else 1,
    }

    features = {**base, **cats}
    return pd.DataFrame([features])


def align_to_model(df, model):
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        for c in cols:
            if c not in df.columns:
                df[c] = 0
        return df[cols]
    return df.to_numpy()


# ---------------------------
# UI Section
# ---------------------------
st.markdown("## 🚀 Loan Eligibility & EMI Predictor")

# Inputs
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", 18, 75, 25)
    salary = st.number_input("Monthly Salary", 0, 1_000_000, 50000)
    yoe = st.number_input("Years of Employment", 0, 40, 1)
    rent = st.number_input("Monthly Rent", 0, 200000, 8000)
    fam = st.number_input("Family Size", 1, 12, 3)
    dep = st.number_input("Dependents", 0, 10, 1)

with c2:
    score = st.number_input("Credit Score", 300, 850, 700)
    bank = st.number_input("Bank Balance", 0, 5000000, 20000)
    emerg = st.number_input("Emergency Fund", 0, 5000000, 0)
    groc = st.number_input("Groceries & Utilities", 0, 200000, 6000)
    other = st.number_input("Other Monthly Expenses", 0, 200000, 4000)
    exist = st.number_input("Existing EMI", 0, 200000, 0)

st.divider()

c3, c4 = st.columns(2)
with c3:
    req_amt = st.number_input("Requested Amount", 0, 5000000, 300000)
with c4:
    req_ten = st.number_input("Requested Tenure (months)", 3, 120, 24)

ui = dict(
    age=age, monthly_salary=salary, years_of_employment=yoe,
    monthly_rent=rent, family_size=fam, dependents=dep,
    groceries_utilities=groc, other_monthly_expenses=other,
    credit_score=score, bank_balance=bank, emergency_fund=emerg,
    existing_emi=exist, requested_amount=req_amt, requested_tenure=req_ten
)

# ---------------------------
# Prediction
# ---------------------------
go = st.button("✨ Predict Eligibility", use_container_width=True)

clf_model, reg_model, label_encoder, clf_name, reg_name = load_models()

if go:
    with st.spinner("Scoring models..."):
        row = engineered_row(ui)
        Xc = align_to_model(row.copy(), clf_model)
        Xr = align_to_model(row.copy(), reg_model)

        y_raw = clf_model.predict(Xc)
        y_text = label_encoder.inverse_transform(y_raw.astype(int))[0]

        emi = float(reg_model.predict(Xr)[0])

    st.success(f"Eligibility: {y_text}")
    st.info(f"Recommended EMI: {inr(emi)}")

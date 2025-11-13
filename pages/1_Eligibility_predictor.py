import streamlit as st
import joblib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Eligibility & EMI Predictor", page_icon="🚀", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def inr(x): 
    try:
        return f"₹{int(round(float(x))):,}"
    except Exception:
        return "₹0"

def pick_latest_model(folder: Path, contains: str):
    """Return latest .joblib whose name contains the tag; fallback to exact name if exists."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Models folder not found: {folder}")
    cands = sorted([p for p in folder.glob("*.joblib") if contains in p.name], key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        return cands[0]
    # last-resort: try the plain name
    p = folder / contains
    if p.exists():
        return p
    raise FileNotFoundError(f"No model matching '{contains}' in {folder}")

@st.cache_resource(show_spinner=False)
def load_models():
    models_dir = Path("models")
    # You can change the substrings if your saved names differ
    clf_path = pick_latest_model(models_dir, "best_classification_model_new")
    reg_path = pick_latest_model(models_dir, "best_regression_model_new")
    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    # Optional label encoder (only if you saved it)
    le_path = models_dir / "label_encoder.joblib"
    le = joblib.load(le_path) if le_path.exists() else None
    return clf, reg, le, clf_path.name, reg_path.name

def engineered_row(ui):
    """Build the **same 80-feature** vector you used in training.
       Keep this in lockstep with your training pipeline."""
    # base numerics
    total_exp = ui['groceries_utilities'] + ui['other_monthly_expenses']
    dti  = (ui['existing_emi'] or 0) / (ui['monthly_salary'] + 1e-6)
    eti  = total_exp / (ui['monthly_salary'] + 1e-6)
    aff  = (ui['monthly_salary'] - total_exp - (ui['existing_emi'] or 0)) / (ui['monthly_salary'] + 1e-6)
    credit_risk = ui['credit_score'] / 850.0
    emp_stab    = ui['years_of_employment'] /  (max(1, ui['years_of_employment']))  # safe ~1.0 when >0
    bank_norm   = ui['bank_balance'] / (max(1, ui['bank_balance']))
    fin_idx     = 0.5*credit_risk + 0.3*emp_stab + 0.2*bank_norm

    # minimal set that your trained columns certainly contain:
    base = {
        "age": ui['age'],
        "monthly_salary": ui['monthly_salary'],
        "years_of_employment": ui['years_of_employment'],
        "monthly_rent": ui['monthly_rent'],
        "family_size": ui['family_size'],
        "dependents": ui['dependents'],
        "groceries_utilities": ui['groceries_utilities'],
        "other_monthly_expenses": ui['other_monthly_expenses'],
        "current_emi_amount": ui['existing_emi'] or 0,
        "credit_score": ui['credit_score'],
        "bank_balance": ui['bank_balance'],
        "emergency_fund": ui['emergency_fund'] or 0,
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

    # one-hot placeholders (keep names your training used; missing ones default to 0)
    # if your train didn’t include these, they’ll be ignored while aligning.
    cats = {
        # put your chosen fixed categories here; for demo we keep “_0/_1” forms at 0
        "existing_loans_No": 1 if (ui['existing_emi'] or 0) == 0 else 0,
        "existing_loans_Yes": 0 if (ui['existing_emi'] or 0) == 0 else 1,
    }

    features = {**base, **cats}
    return pd.DataFrame([features])

def align_to_model(df, model):
    """Align columns to model.feature_names_in_ (sklearn) or Booster feature names (xgboost)."""
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
    else:
        # xgboost sometimes doesn’t store names; infer from training saved alongside the model if you saved.
        # As a fallback, pass df as numpy in the same order each run.
        return df[sorted(df.columns)].to_numpy()
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    df = df[cols]
    return df

def build_pdf(ui, eligibility_text, emi_pred, clf_name, reg_name):
    """Create a small PDF report with ReportLab and return as bytes."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    from reportlab.lib import colors

    buff = BytesIO()
    c = canvas.Canvas(buff, pagesize=A4)
    W, H = A4

    # header
    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2*cm, H-2.5*cm, "EMIPredict AI – Eligibility Report")
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.black)
    c.drawString(2*cm, H-3.1*cm, f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")

    # summary card
    y = H-4.2*cm
    c.roundRect(2*cm, y-2.6*cm, W-4*cm, 2.4*cm, 10, stroke=1, fill=0)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2.4*cm, y-0.7*cm, f"Eligibility: {eligibility_text}")
    c.drawString(2.4*cm, y-1.5*cm, f"Recommended Safe EMI: {inr(emi_pred)}")

    # inputs
    y2 = y-3.4*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y2, "Input Summary")
    c.setFont("Helvetica", 10)
    rows = [
        ("Age", ui['age']), ("Family Size", ui['family_size']),
        ("Dependents", ui['dependents']), ("Monthly Salary", inr(ui['monthly_salary'])),
        ("Monthly Rent", inr(ui['monthly_rent'])), ("Groceries & Utilities", inr(ui['groceries_utilities'])),
        ("Other Expenses", inr(ui['other_monthly_expenses'])), ("Credit Score", ui['credit_score']),
        ("Bank Balance", inr(ui['bank_balance'])), ("Emergency Fund", inr(ui['emergency_fund'] or 0)),
        ("Existing EMI", inr(ui['existing_emi'] or 0)),
    ]
    y_line = y2-0.7*cm
    for i, (k,v) in enumerate(rows):
        c.drawString(2*cm, y_line, f"{k}: {v}")
        y_line -= 0.55*cm

    # footer
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.HexColor("#64748b"))
    c.drawString(2*cm, 1.8*cm, f"Models: {clf_name}  |  {reg_name}")
    c.showPage()
    c.save()
    buff.seek(0)
    return buff.getvalue()

# ---------------------------
# CSS (UI upgrade + shimmer)
# ---------------------------
st.markdown("""
<style>
:root{
  --bg: #f7fbff;
  --card:#ffffff;
  --accent:#0ea5e9;
}
main .block-container{padding-top:2rem; padding-bottom:3rem;}
.card{background:var(--card); border:1px solid rgba(2,6,23,.06);
      border-radius:16px; padding:18px 16px; box-shadow:0 8px 22px rgba(2,6,23,.06);}
.btn{display:inline-block;padding:.7rem 1.1rem;border-radius:10px;
     background:linear-gradient(135deg,#22d3ee,#3b82f6); color:white; font-weight:700; text-decoration:none;}
.badge{font-weight:700; color:#0f172a;}
/* shimmer */
.shimmer {
  position: relative; overflow: hidden; background: #eef2f7; border-radius: 10px; height: 46px;
}
.shimmer:before {
  content: ""; position: absolute; top:0; left:-150px; height:100%; width:150px;
  background: linear-gradient(90deg, rgba(255,255,255,0), rgba(255,255,255,.6), rgba(255,255,255,0));
  animation: loading 1.2s infinite; 
}
@keyframes loading { 0%{left:-150px} 50%{left:100%} 100%{left:100%} }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🚀 Loan Eligibility & EMI Predictor")

# ---------------------------
# Input UI
# ---------------------------
c1,c2 = st.columns(2)

with c1:
    age   = st.number_input("Age", 18, 75, 25)
    salary= st.number_input("Monthly Salary", 0, 1_000_000, 50_000, step=1000)
    yoe   = st.number_input("Years of Employment", 0, 40, 1)
    rent  = st.number_input("Monthly Rent", 0, 200_000, 8000, step=500)
    fam   = st.number_input("Family Size", 1, 12, 3)
    dep   = st.number_input("Dependents", 0, 10, 1)
with c2:
    score = st.number_input("Credit Score", 300, 850, 700)
    bank  = st.number_input("Bank Balance", 0, 5_000_000, 20_000, step=1000)
    emerg = st.number_input("Emergency Fund (₹)", 0, 5_000_000, 0, step=1000)
    groc  = st.number_input("Groceries & Utilities", 0, 200_000, 6000, step=500)
    other = st.number_input("Other Monthly Expenses", 0, 200_000, 4000, step=500)
    exist = st.number_input("Existing EMI (₹)", 0, 200_000, 0, step=500)

st.divider()
c3,c4 = st.columns(2)
with c3:
    req_amt   = st.number_input("Requested Amount (₹)", 0, 5_000_000, 300_000, step=5000)
with c4:
    req_ten   = st.number_input("Requested Tenure (months)", 3, 120, 24)

ui = dict(
    age=age, monthly_salary=salary, years_of_employment=yoe, monthly_rent=rent,
    family_size=fam, dependents=dep, credit_score=score, bank_balance=bank,
    emergency_fund=emerg, groceries_utilities=groc, other_monthly_expenses=other,
    existing_emi=exist, requested_amount=req_amt, requested_tenure=req_ten
)

# ---------------------------
# Predict button + shimmer
# ---------------------------
go = st.button("✨ Predict Eligibility", use_container_width=True)

clf_model, reg_model, label_encoder, clf_name, reg_name = load_models()

if go:
    # Shimmer placeholders while computing
    ph = st.container()
    with ph:
        st.markdown('<div class="shimmer"></div>', unsafe_allow_html=True)
        st.markdown('<div class="shimmer"></div>', unsafe_allow_html=True)

    with st.spinner("Scoring models…"):
        row = engineered_row(ui)
        X_clf = align_to_model(row.copy(), clf_model)
        X_reg = align_to_model(row.copy(), reg_model)

        # Class prediction (handle encoded labels if you saved them)
        y_raw = clf_model.predict(X_clf)
        if label_encoder is not None:
            try:
                y_text = label_encoder.inverse_transform(y_raw.astype(int))[0]
            except Exception:
                # already text
                y_text = str(y_raw[0])
        else:
            y_text = str(y_raw[0])

        # Regression
        emi_safe = float(reg_model.predict(X_reg)[0])
        emi_safe = max(0.0, emi_safe)

    ph.empty()  # remove shimmer

    # Nice result cards
    st.success(f"**Eligibility Status:** {y_text}")
    st.info(f"**Recommended Safe EMI:** {inr(emi_safe)}")

    # ---------------------------
    # Build & offer PDF report
    # ---------------------------
    pdf_bytes = build_pdf(ui, y_text, emi_safe, clf_name, reg_name)
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name=f"EMIPredict_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

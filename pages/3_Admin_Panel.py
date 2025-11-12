# pages/3_Admin_Panel.py
import os
import io
import shutil
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import streamlit as st

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="Admin Panel", page_icon="🛠", layout="wide")

MODELS_DIR  = "models"
DATA_DIR    = "data"
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
KNOWN_ROLES = [
    "train_emi.csv", "val_emi.csv", "test_emi.csv",
    "train_emi_engineered.csv", "val_emi_engineered.csv", "test_emi_engineered.csv",
    "train_emi_engineered_aligned.csv", "val_emi_engineered_aligned.csv",
    "test_emi_engineered_aligned.csv",
]
ALLOWED_MODEL_EXT = (".joblib",)
ALLOWED_CSV_EXT   = (".csv",)

for d in (MODELS_DIR, DATA_DIR, UPLOADS_DIR):
    os.makedirs(d, exist_ok=True)

# -----------------------------------------------------------
# UTILS
# -----------------------------------------------------------
def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"

def list_files(path, exts):
    return sorted([f for f in os.listdir(path) if f.lower().endswith(exts)])

def safe_read_csv(path, nrows=5):
    try:
        if os.path.getsize(path) == 0:
            return None
        return pd.read_csv(path, nrows=nrows)
    except:
        return None

def df_meta(path: str):
    meta = {"name": os.path.basename(path), "size": human_bytes(os.path.getsize(path))}
    df = safe_read_csv(path, nrows=20)
    if df is not None:
        meta["rows"] = f"{len(df):,}"
        meta["cols"] = df.shape[1]
    else:
        meta["rows"] = "—"
        meta["cols"] = "—"
    return meta

def download_bytes(path):
    with open(path, "rb") as f:
        return f.read()

# -----------------------------------------------------------
# HEADER
# -----------------------------------------------------------
st.markdown("""
<h1 style='margin-bottom:5px;'>🛠 Admin Panel</h1>
<p style='color:gray;'>Manage models & datasets used in the application.</p>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------------------------
# 1) MODEL MANAGEMENT
# -----------------------------------------------------------
st.markdown("## 1️⃣ Model Management")

left, right = st.columns([1.5, 1])

# ------------------ LEFT (Existing Models) ------------------
with left:
    st.subheader("Existing Models")

    model_files = list_files(MODELS_DIR, ALLOWED_MODEL_EXT)

    if not model_files:
        st.info("No .joblib models found in /models")
    else:
        for mf in model_files:
            path = os.path.join(MODELS_DIR, mf)
            size = human_bytes(os.path.getsize(path))

            row = st.container()
            with row:
                c1, c2, c3, c4 = st.columns([0.55, 0.15, 0.15, 0.15])

                c1.markdown(f"### 📦 {mf} \n<small>{size}</small>", unsafe_allow_html=True)

                with c2:
                    st.download_button(
                        "📥 Download",
                        data=download_bytes(path),
                        file_name=mf,
                        key=f"dl-{mf}",
                        use_container_width=True
                    )

                with c3:
                    rename_click = st.button("📝 Rename", key=f"rn-btn-{mf}", use_container_width=True)

                with c4:
                    if st.button("❌ Delete", key=f"del-{mf}", use_container_width=True):
                        os.remove(path)
                        st.success(f"Deleted: {mf}")
                        st.rerun()

                # ✅ Show rename box ONLY when Rename is clicked
                if rename_click:
                    new_name = st.text_input(
                        "Enter new filename (.joblib)",
                        value=mf,
                        key=f"rename-input-{mf}"
                    )

                    if st.button("✅ Save Rename", key=f"rename-save-{mf}"):
                        if not new_name.endswith(".joblib"):
                            st.error("Filename must end with .joblib")
                        else:
                            os.rename(path, os.path.join(MODELS_DIR, new_name))
                            st.success("Model renamed successfully!")
                            st.rerun()

    st.markdown("---")

# ------------------ RIGHT (Upload Model) ------------------
with right:
    st.subheader("Upload New Model (.joblib)")
    up_model = st.file_uploader("Drag & drop or browse", type=["joblib"])

    if up_model:
        save_path = os.path.join(MODELS_DIR, up_model.name)
        with open(save_path, "wb") as f:
            f.write(up_model.getbuffer())
        st.success(f"Uploaded: {up_model.name}")
        st.rerun()

st.markdown("---")

# -----------------------------------------------------------
# 2) DATASET MONITORING
# -----------------------------------------------------------
st.markdown("## 2️⃣ Dataset Monitoring")

data_cols = st.columns([1.4, 1])

# ------------------- LEFT (Current Data) -------------------
with data_cols[0]:
    st.subheader("Current Datasets (data/)")

    data_files = list_files(DATA_DIR, ALLOWED_CSV_EXT)

    if not data_files:
        st.info("No CSV files in data/")
    else:
        for f in data_files:
            full = os.path.join(DATA_DIR, f)
            meta = df_meta(full)

            exp = st.expander(f"📄 {meta['name']} — {meta['size']} | Rows: {meta['rows']} • Cols: {meta['cols']}")

            with exp:
                df = safe_read_csv(full)
                if df is not None:
                    st.dataframe(df.head(), use_container_width=True)

                cA, cB = st.columns([0.3, 0.7])
                cA.download_button("📥 Download", download_bytes(full), file_name=f)
                
                if cB.button("🗑 Delete", key=f"x-data-{f}"):
                    os.remove(full)
                    st.success(f"Deleted {f}")
                    st.rerun()

# ------------------- RIGHT (Upload CSV) -------------------
with data_cols[1]:
    st.subheader("Upload CSV → data/uploads")

    up_csv = st.file_uploader("Upload .csv file", type=["csv"])

    if up_csv:
        save_to = os.path.join(UPLOADS_DIR, up_csv.name)
        with open(save_to, "wb") as f:
            f.write(up_csv.getbuffer())

        st.success(f"Uploaded → {save_to}")

        try:
            st.dataframe(pd.read_csv(io.BytesIO(up_csv.getvalue()), nrows=8))
        except:
            st.warning("Preview failed.")

        st.stop()

st.markdown("---")

# -----------------------------------------------------------
# 3) PROMOTE & VALIDATE CSVs
# -----------------------------------------------------------
st.markdown("## 3️⃣ Promote & Validate Uploaded Files")

uploads = list_files(UPLOADS_DIR, ALLOWED_CSV_EXT)

if not uploads:
    st.info("No CSVs in data/uploads/")
else:
    ref_path = os.path.join(DATA_DIR, "train_emi_engineered_aligned.csv")
    ref_cols = None

    if os.path.exists(ref_path):
        ref_cols = list(pd.read_csv(ref_path, nrows=1).columns)

    for f in uploads:
        full = os.path.join(UPLOADS_DIR, f)
        meta = df_meta(full)

        exp = st.expander(f"📦 {meta['name']} — {meta['size']}")

        with exp:
            df = safe_read_csv(full)
            if df is not None:
                st.dataframe(df.head())

            if ref_cols:
                curr_cols = list(pd.read_csv(full, nrows=1).columns)
                missing = [c for c in ref_cols if c not in curr_cols]
                extra   = [c for c in curr_cols if c not in ref_cols]

                if not missing and not extra:
                    st.success("✅ Schema OK (Matches aligned training schema)")
                else:
                    if missing:
                        st.error(f"Missing: {missing}")
                    if extra:
                        st.warning(f"Extra: {extra}")

            col1, col2, col3 = st.columns([0.3, 0.3, 0.4])

            target = col1.selectbox("Promote as:", ["(select)"] + KNOWN_ROLES, key=f"role-{f}")

            if col2.button("Promote", key=f"promote-{f}"):
                if target == "(select)":
                    st.error("Choose a filename")
                else:
                    shutil.move(full, os.path.join(DATA_DIR, target))
                    st.success(f"Promoted → {target}")
                    st.rerun()

            if col3.button("Delete Upload", key=f"x-u-{f}"):
                os.remove(full)
                st.success("Removed")
                st.rerun()

st.markdown("---")

# -----------------------------------------------------------
# 4) HOUSEKEEPING
# -----------------------------------------------------------
st.markdown("## 4️⃣ Housekeeping")

colA, colB = st.columns([0.4, 0.6])

with colA:
    if st.button("🧹 Remove zero-byte CSVs"):
        removed = []
        for folder in (DATA_DIR, UPLOADS_DIR):
            for f in list_files(folder, ALLOWED_CSV_EXT):
                p = os.path.join(folder, f)
                if os.path.getsize(p) == 0:
                    os.remove(p)
                    removed.append(f)
        if removed:
            st.success("Removed:\n" + "\n".join(removed))
        else:
            st.info("No zero-byte files found.")

with colB:
    if st.button("📄 Create aligned template (.csv)"):
        ref = os.path.join(DATA_DIR, "train_emi_engineered_aligned.csv")
        if not os.path.exists(ref):
            st.error("Reference aligned CSV missing")
        else:
            cols = pd.read_csv(ref, nrows=1).columns
            empty = pd.DataFrame(columns=cols)
            out = os.path.join(DATA_DIR, f"template_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            empty.to_csv(out, index=False)
            st.success(f"Template created → {out}")
            st.download_button("Download Template", download_bytes(out), file_name=os.path.basename(out))

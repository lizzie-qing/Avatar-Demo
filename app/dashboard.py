import os
import json
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Avatar Demo Dashboard", layout="wide")

OUT_DIR = "outputs"
API_URL = st.sidebar.text_input("API URL", value="http://127.0.0.1:8001")

st.title("Avatar Feature → Impression (Demo) Dashboard")

# ---------- Data block ----------
st.header("1) Data")
col1, col2 = st.columns(2)

features_path = os.path.join(OUT_DIR, "features.csv")
cleaned_path = os.path.join(OUT_DIR, "cleaned.csv")

with col1:
    st.subheader("features.csv")
    if os.path.exists(features_path):
        df_feat = pd.read_csv(features_path)
        st.write({"rows": len(df_feat), "cols": list(df_feat.columns)})
        st.dataframe(df_feat.head(10))
    else:
        st.warning("outputs/features.csv not found. Run pipeline first.")

with col2:
    st.subheader("cleaned.csv")
    if os.path.exists(cleaned_path):
        df_clean = pd.read_csv(cleaned_path)
        st.write({"rows": len(df_clean), "cols": list(df_clean.columns)})
        st.dataframe(df_clean.head(10))
    else:
        st.warning("outputs/cleaned.csv not found. Run pipeline first.")

# ---------- Model block ----------
st.header("2) Model")
reg_path = os.path.join(OUT_DIR, "regression_summary.txt")
if os.path.exists(reg_path):
    st.subheader("Regression summary")
    st.code(open(reg_path, "r", encoding="utf-8").read(), language="text")
else:
    st.warning("outputs/regression_summary.txt not found.")

pca_path = os.path.join(OUT_DIR, "pca.png")
if os.path.exists(pca_path):
    st.subheader("PCA plot")
    st.image(pca_path, use_container_width=True)
else:
    st.warning("outputs/pca.png not found.")

# ---------- Predict block ----------
st.header("3) Predict (Upload image → API)")
uploaded = st.file_uploader("Upload a PNG image", type=["png"])

if uploaded is not None:
    st.image(uploaded, caption="Uploaded image", use_container_width=False)

    if st.button("Send to /predict"):
        files = {"file": (uploaded.name, uploaded.getvalue(), "image/png")}
        try:
            resp = requests.post(f"{API_URL}/predict", files=files, timeout=60)
            st.write("Status:", resp.status_code)
            st.json(resp.json())
        except Exception as e:
            st.error(f"Request failed: {e}")

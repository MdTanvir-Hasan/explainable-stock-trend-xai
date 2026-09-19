"""M10 — SP7 Streamlit decision-support prototype.

Run from the code/ directory:
    streamlit run m10_prototype/streamlit_app.py

Scope: research demonstration. Not a trading system.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from m0_setup.config import load_config
from m10_prototype.prototype import explain_stock, interpretation_text

st.set_page_config(page_title="XAI stock trend", layout="centered")
config = load_config(ROOT / "config.yaml")

st.title("Explainable stock-trend prediction")
st.caption("Research demonstration only — not a trading system or financial advice.")

stock = st.selectbox("Stock", config["data"]["stocks"], index=config["data"]["stocks"].index("BHP"))

with st.spinner("Loading model and explanations..."):
    prediction = explain_stock(stock, config)

st.metric(f"{stock} — next trading day", prediction.label, f"{prediction.probability_up:.1%} up")
st.write(interpretation_text(prediction))

left, right = st.columns(2)
with left:
    st.subheader("Local SHAP drivers")
    st.bar_chart(prediction.drivers.set_index("feature")["shap"])
with right:
    st.subheader("Financial theme contribution (%)")
    st.bar_chart(prediction.group_shares)

st.caption(f"As of {prediction.date}. SHAP values are plausibility signals, not causal effects.")

# PB2PD Screening Tool – Clean UI version (H-Y hidden but preserved for model compatibility)

import os, io, json, math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# ---------- optional: PDF export ----------
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    HAVE_PDF = True
except Exception:
    HAVE_PDF = False

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="PB2PD Screening Tool",
    page_icon="🚶🏻‍♀️‍➡️",
    layout="wide"
)

MODEL_PATH   = "models/rf_model.pkl"
SCALER_PATH  = "models/scaler.pkl"
THR_JSON     = "models/thresholds.json"
METRICS_JSON = "results/train_metrics.json"
RULES_JSON   = "artifacts/surrogate_rules_deploy.json"
PI_GUIDE_PATH = "artifacts/PI_GUIDE.md"

RESULTS_DIR  = "results"
LOG_PATH     = os.path.join(RESULTS_DIR, "pilot_log.csv")

# H-Y is intentionally NOT shown in UI,
# but must be included in this list in the correct order
FEATURES = [
    "MSE ML","iHR V","MSE V","MSE AP","Weigth","Age",
    "Sex (M=1, F=2)","H-Y","Gait Speed","Duration (years)"
]

NUM_FEATURES = [
    "MSE ML","iHR V","MSE V","MSE AP","Weigth","Age",
    "Gait Speed","Duration (years)"
]

DEFAULT_THRESHOLDS = {"low": 0.26, "mid": 0.40, "high": 0.50}

FALLBACK_OP = {
    0.26: {"sensitivity": 0.82, "specificity": 0.55},
    0.40: {"sensitivity": 0.68, "specificity": 0.64},
    0.50: {"sensitivity": 0.57, "specificity": 0.72},
    0.60: {"sensitivity": 0.48, "specificity": 0.80},
}

DEFAULT_PI_GUIDE_MD = """
# PB2PD – Principal Investigator Guide

This tool supports probability screening of prodromal-pattern positive subjects.
Non-diagnostic; research-only.
"""

# =========================
# LOADERS
# =========================
@st.cache_resource
def load_model_scaler_rules():
    rf = load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    scaler = load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    rules = None
    if os.path.exists(RULES_JSON):
        try:
            with open(RULES_JSON, "r") as f:
                rules = json.load(f)
        except:
            rules = None

    thr_suggested = DEFAULT_THRESHOLDS.copy()
    if os.path.exists(THR_JSON):
        try:
            t = json.load(open(THR_JSON))
            thr_suggested["low"]  = float(t.get("thr_green",  thr_suggested["low"]))
            thr_suggested["mid"]  = float(t.get("thr_yellow", thr_suggested["mid"]))
            thr_suggested["high"] = float(t.get("thr_orange", thr_suggested["high"]))
        except:
            pass

    op = FALLBACK_OP.copy()
    if os.path.exists(METRICS_JSON):
        try:
            m = json.load(open(METRICS_JSON))
            if "operating_points" in m:
                for k,v in m["operating_points"].items():
                    op[float(k)] = {
                        "sensitivity": float(v["sensitivity"]),
                        "specificity": float(v["specificity"])
                    }
        except:
            pass

    return rf, scaler, rules, thr_suggested, op


@st.cache_resource
def load_pi_guide_md():
    try:
        with open(PI_GUIDE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return DEFAULT_PI_GUIDE_MD


# =========================
# UTILS
# =========================
def ensure_dataframe_order(df):
    df = df.rename(columns={c: c.strip() for c in df.columns})
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[FEATURES].copy()


def build_X_from_series(row: pd.Series):
    vals = {}
    for k in FEATURES:
        if k == "H-Y":
            vals[k] = 2  # neutral hidden value
        else:
            try:
                vals[k] = float(row[k])
            except:
                vals[k] = row[k]
    return pd.DataFrame([vals], columns=FEATURES)


def predict_proba_single(model, scaler, row: pd.Series) -> float:
    X = build_X_from_series(row)
    if scaler is not None:
        try:
            X[NUM_FEATURES] = scaler.transform(X[NUM_FEATURES])
        except:
            pass
    return float(model.predict_proba(X)[:,1][0])


def predict_proba_batch(model, scaler, X_in):
    X = ensure_dataframe_order(X_in)
    if scaler is not None:
        try:
            X[NUM_FEATURES] = scaler.transform(X[NUM_FEATURES])
        except:
            pass
    return model.predict_proba(X)[:, 1]


# =========================
# UI
# =========================

st.title("PB2PD Screening Tool 🚶🏻‍♀️‍➡️")

rf, scaler, rules_json, thr_suggested, OP_TABLE = load_model_scaler_rules()
if rf is None:
    st.error("Model not found.")
    st.stop()

PI_GUIDE_MD = load_pi_guide_md()

with st.sidebar:
    st.header("Settings")
    colA, colB = st.columns(2)
    thr_low  = colA.number_input("Green→Yellow", value=thr_suggested["low"], step=0.01)
    thr_mid  = colB.number_input("Yellow→Orange", value=thr_suggested["mid"], step=0.01)
    thr_high = colA.number_input("Orange→Red", value=thr_suggested["high"], step=0.01)

    st.caption(f"Suggested: {thr_suggested['low']:.2f} / {thr_suggested['mid']:.2f} / {thr_suggested['high']:.2f}")

    st.markdown("---")
    st.subheader("Screening mode")
    screening_on = st.checkbox("Enable screening mode", value=True)
    prevalence = st.slider("Expected prevalence (%)", 1.0, 80.0, 40.0, 0.5)
    screen_thr_choice = st.selectbox(
        "Screening threshold",
        [thr_low, thr_mid, thr_high, 0.60],
        format_func=lambda x: f"{x:.2f}"
    )

    st.markdown("---")
    audit_log = st.checkbox("Log case (anonymous)")

tab1, tab2, tab3 = st.tabs(["👤 Single patient","📥 Batch CSV","📘 PI Guide"])

# -------------
# TAB 1
# -------------
with tab1:

    st.subheader("Patient input")
    c1, c2, c3 = st.columns(3)

    with c1:
        mse_ml = st.number_input("MSE ML", min_value=0.0, value=1.75, step=0.01)
        mse_v  = st.number_input("MSE V", min_value=0.0, value=1.50, step=0.01)
        mse_ap = st.number_input("MSE AP", min_value=0.0, value=1.30, step=0.01)
        ihr_v  = st.number_input("iHR V", min_value=0.0, value=65.0, step=0.5)

    with c2:
        speed  = st.number_input("Gait Speed (m/s)", min_value=0.0, value=1.10, step=0.01)
        weight = st.number_input("Weigth (kg)", min_value=30.0, value=75.0, step=0.5)
        age    = st.number_input("Age (years)", min_value=18.0, value=70.0, step=1.0)
        dur    = st.number_input("Duration (years)", min_value=0.0, value=5.0, step=0.5)

    with c3:
        sex = st.selectbox("Sex (M=1, F=2)", [1,2], index=0)
        st.write("")

    row = pd.Series({
        "MSE ML": mse_ml,
        "iHR V": ihr_v,
        "MSE V": mse_v,
        "MSE AP": mse_ap,
        "Weigth": weight,
        "Age": age,
        "Sex (M=1, F=2)": sex,
        "Gait Speed": speed,
        "Duration (years)": dur
    })

    st.divider()

    if st.button("Compute risk", type="primary"):

        p = predict_proba_single(rf, scaler, row)

        clr = (
            "#2ecc71" if p < thr_low else
            "#f1c40f" if p < thr_mid else
            "#e67e22" if p < thr_high else
            "#e74c3c"
        )

        st.markdown(f"### Probability: **{p:.3f}**")
        st.markdown(
            f"<div style='width:100%;height:16px;background:{clr};border-radius:6px'></div>",
            unsafe_allow_html=True
        )

        band_idx = sum(p >= np.array([thr_low, thr_mid, thr_high]))
        band = ["Green","Yellow","Orange","Red"][band_idx]
        st.caption(f"**Band: {band}**")

        # LOG
        if audit_log:
            log_row = {"timestamp": datetime.now().isoformat(), "p": float(p), "band": band}
            for k in row.index:
                log_row[k] = row[k]
            os.makedirs(RESULTS_DIR, exist_ok=True)
            df = pd.DataFrame([log_row])
            if not os.path.exists(LOG_PATH):
                df.to_csv(LOG_PATH, index=False)
            else:
                df.to_csv(LOG_PATH, index=False, mode="a", header=False)
            st.success("Logged.")

# -------------
# TAB 2
# -------------
with tab2:

    st.subheader("Batch CSV")
    st.caption("Required columns:")

    st.code(", ".join(FEATURES))

    upl = st.file_uploader("Upload CSV", type=["csv"])

    if upl:
        try:
            df = pd.read_csv(upl)
            proba = predict_proba_batch(rf, scaler, df.copy())
            bands = np.select(
                [proba < thr_low, proba < thr_mid, proba < thr_high],
                ["Green","Yellow","Orange"],
                default="Red"
            )
            out = ensure_dataframe_order(df.copy())
            out["proba"] = proba
            out["band"] = bands
            st.dataframe(out.head())
        except Exception as e:
            st.error(f"Error: {e}")

# -------------
# TAB 3
# -------------
with tab3:
    st.markdown(PI_GUIDE_MD)


st.caption("Research-use only. Non-diagnostic.")
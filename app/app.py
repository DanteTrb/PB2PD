# ============================================
# PB2PD Screening Tool – Full Version
# Hidden Stage, SHAP + Rules + PDF
# ============================================

import os, io, json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

# ---- SHAP ----
import shap
shap.initjs()

# ---- PDF ----
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    HAVE_PDF = True
except:
    HAVE_PDF = False


# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="PB2PD Screening Tool",
    page_icon="🚶🏻‍♀️‍➡️",
    layout="wide"
)

MODEL_PATH   = "../models/rf_model.pkl"
SCALER_PATH  = "../models/scaler.pkl"
THR_JSON     = "../models/thresholds.json"
RULES_JSON   = "artifacts/surrogate_rules_deploy.json"
METRICS_JSON = "results/train_metrics.json"
PI_GUIDE_PATH = "artifacts/PI_GUIDE.md"

RESULTS_DIR = "results"
LOG_PATH    = os.path.join(RESULTS_DIR, "pilot_log.csv")

# ===== MODEL FEATURES (order must be exact) =====
FEATURES = [
    "MSE ML","iHR V","MSE V","MSE AP",
    "Weigth","Age","Sex (M=1, F=2)","H-Y",
    "Gait Speed","Duration (years)"
]

NUM_FEATURES = [
    "MSE ML","iHR V","MSE V","MSE AP",
    "Weigth","Age","Gait Speed","Duration (years)"
]

DEFAULT_THRESHOLDS = {"low":0.26,"mid":0.40,"high":0.50}

DEFAULT_PI_GUIDE_MD = """
# PB2PD – Principal Investigator Guide
Research-use only — probability screening of prodromal-like PD signatures.
"""


# =====================================================
# LOADERS
# =====================================================
@st.cache_resource
def load_all():
    model = load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    scaler = load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

    # surrogate rules
    rules = None
    if os.path.exists(RULES_JSON):
        try:
            rules = json.load(open(RULES_JSON))
        except:
            rules = None

    # thresholds
    thr = DEFAULT_THRESHOLDS.copy()
    if os.path.exists(THR_JSON):
        try:
            js = json.load(open(THR_JSON))
            thr["low"]  = float(js.get("thr_green", thr["low"]))
            thr["mid"]  = float(js.get("thr_yellow",thr["mid"]))
            thr["high"] = float(js.get("thr_orange",thr["high"]))
        except:
            pass

    # PI Guide
    try:
        guide_md = open(PI_GUIDE_PATH, "r", encoding="utf-8").read()
    except:
        guide_md = DEFAULT_PI_GUIDE_MD

    return model, scaler, rules, thr, guide_md


# =====================================================
# UTILS
# =====================================================

def build_X_single(row):
    """Build X for a single patient with H-Y hidden (forced = 2)."""
    x = {}
    for f in FEATURES:
        if f == "H-Y":
            x[f] = 2        # hidden fixed value
        else:
            x[f] = row[f]
    return pd.DataFrame([x], columns=FEATURES)


def predict_single(model, scaler, row):
    X = build_X_single(row)
    if scaler is not None:
        try:
            X[NUM_FEATURES] = scaler.transform(X[NUM_FEATURES])
        except:
            pass
    return float(model.predict_proba(X)[:,1][0])


def make_pdf(buffer, p, band, row, shap_vals, rules):
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4

    y = h - 2*cm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, y, "PB2PD Screening Report")
    y -= 1*cm

    c.setFont("Helvetica", 12)
    c.drawString(2*cm, y, f"Probability: {p:.3f}   |   Band: {band}")
    y -= 1*cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Input Features:")
    y -= 0.8*cm
    c.setFont("Helvetica", 11)
    for k,v in row.items():
        c.drawString(2.5*cm, y, f"{k}: {v}")
        y -= 0.6*cm

    # ---- SHAP top feature ----
    if shap_vals is not None:
        y -= 0.6*cm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, y, "Most Influential Feature:")
        y -= 0.7*cm
        idx = np.argmax(np.abs(shap_vals))
        c.setFont("Helvetica", 11)
        c.drawString(2.5*cm, y, f"{FEATURES[idx]} (SHAP: {shap_vals[idx]:.3f})")
        y -= 1*cm

    # ---- Rules surrogate ----
    if rules is not None:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, y, "Surrogate Rules:")
        y -= 0.7*cm
        c.setFont("Helvetica", 11)
        for r in rules.get("rules", []):
            c.drawString(2.5*cm, y, "- " + r)
            y -= 0.6*cm

    c.showPage()
    c.save()


# =====================================================
# LOAD EVERYTHING
# =====================================================
model, scaler, rules, thr_suggested, PI_GUIDE_MD = load_all()

if model is None:
    st.error("Model not found.")
    st.stop()


# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.header("Settings")

    thr_low  = st.number_input("Green → Yellow", value=thr_suggested["low"], step=0.01)
    thr_mid  = st.number_input("Yellow → Orange", value=thr_suggested["mid"], step=0.01)
    thr_high = st.number_input("Orange → Red", value=thr_suggested["high"], step=0.01)

    st.caption(f"Suggested: {thr_suggested['low']:.2f} / {thr_suggested['mid']:.2f} / {thr_suggested['high']:.2f}")

    st.markdown("---")
    log_on = st.checkbox("Log (anonymous)")


# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3 = st.tabs(["👤 Single patient", "📥 Batch CSV", "📘 PI Guide"])


# =====================================================
# TAB 1 — SINGLE PATIENT
# =====================================================
with tab1:

    st.subheader("Patient Input")

    c1, c2, c3 = st.columns(3)

    with c1:
        mse_ml = st.number_input("MSE ML", 0.0, value=1.75, step=0.01)
        mse_v  = st.number_input("MSE V",  0.0, value=1.50, step=0.01)
        mse_ap = st.number_input("MSE AP", 0.0, value=1.30, step=0.01)
        ihr_v  = st.number_input("iHR V",  0.0, value=65.0, step=0.5)

    with c2:
        speed = st.number_input("Gait Speed (m/s)", 0.0, value=1.10, step=0.01)
        weight = st.number_input("Weigth (kg)", 30.0, value=75.0, step=0.5)
        age = st.number_input("Age (years)", 18.0, value=70.0, step=1.0)
        dur = st.number_input("Duration (years)", 0.0, value=5.0, step=0.5)

    with c3:
        sex = st.selectbox("Sex (M=1, F=2)", [1,2])

    row = {
        "MSE ML": mse_ml,
        "iHR V": ihr_v,
        "MSE V": mse_v,
        "MSE AP": mse_ap,
        "Weigth": weight,
        "Age": age,
        "Sex (M=1, F=2)": sex,
        "Gait Speed": speed,
        "Duration (years)": dur
    }

    st.markdown("---")

    if st.button("Compute risk", type="primary"):
        p = predict_single(model, scaler, row)

        # ----- color band -----
        clr = (
            "#2ecc71" if p < thr_low else
            "#f1c40f" if p < thr_mid else
            "#e67e22" if p < thr_high else
            "#e74c3c"
        )
        band_idx = sum(p >= np.array([thr_low, thr_mid, thr_high]))
        band = ["Green","Yellow","Orange","Red"][band_idx]

        st.markdown(f"### Probability: **{p:.3f}**")
        st.markdown(
            f"<div style='width:100%;height:16px;background:{clr};border-radius:6px'></div>",
            unsafe_allow_html=True
        )
        st.caption(f"Band: **{band}**")

        # ---- SHAP local ----
        st.subheader("Local SHAP Explanation")
        Xtmp = build_X_single(row)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(Xtmp)[1][0]

        st.pyplot(shap.force_plot(
            explainer.expected_value[1],
            shap_vals,
            Xtmp.iloc[0,:],
            matplotlib=True,
            figsize=(10,3)
        ))

        # ---- SHAP summary ----
        st.subheader("SHAP Summary")
        try:
            st.pyplot(shap.summary_plot(
                explainer.shap_values(Xtmp)[1],
                Xtmp,
                plot_type="bar",
                show=False
            ))
        except:
            st.info("Summary unavailable for single-example input.")

        # ---- Rules ----
        if rules is not None:
            st.subheader("Surrogate Rules")
            for r in rules.get("rules", []):
                st.markdown(f"- {r}")

        # ---- Logging ----
        if log_on:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            row2 = row.copy()
            row2["prob"] = p
            row2["band"] = band
            row2["timestamp"] = datetime.now().isoformat()
            df = pd.DataFrame([row2])
            if not os.path.exists(LOG_PATH):
                df.to_csv(LOG_PATH, index=False)
            else:
                df.to_csv(LOG_PATH, index=False, mode="a", header=False)

        # ---- PDF Export ----
        if HAVE_PDF:
            buf = io.BytesIO()
            make_pdf(buf, p, band, row, shap_vals, rules)
            st.download_button(
                "Download PDF Report",
                data=buf.getvalue(),
                file_name="PB2PD_report.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("PDF export unavailable (ReportLab missing).")


# =====================================================
# TAB 2 — BATCH CSV
# =====================================================
with tab2:
    st.subheader("Batch CSV Upload")
    st.caption("Required columns: " + ", ".join(FEATURES))

    upl = st.file_uploader("Upload CSV", type=["csv"])
    if upl:
        try:
            df = pd.read_csv(upl)
            # Construct X
            for f in FEATURES:
                if f not in df.columns:
                    st.error(f"Missing column: {f}")
                    st.stop()

            # Fix H-Y to 2
            df["H-Y"] = 2

            # scale
            if scaler is not None:
                try:
                    df[NUM_FEATURES] = scaler.transform(df[NUM_FEATURES])
                except:
                    pass

            prob = model.predict_proba(df[FEATURES])[:,1]
            band = np.select(
                [prob < thr_low, prob < thr_mid, prob < thr_high],
                ["Green","Yellow","Orange"],
                default="Red"
            )

            out = df.copy()
            out["proba"] = prob
            out["band"] = band
            st.dataframe(out.head())

        except Exception as e:
            st.error(f"Error: {e}")


# =====================================================
# TAB 3 — PI GUIDE
# =====================================================
with tab3:
    st.markdown(PI_GUIDE_MD)


st.caption("Research-use only — non diagnostic.")
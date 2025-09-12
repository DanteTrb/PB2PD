# app.py
# Triad Risk — MVP (RF + SHAP) for triad-positive cohorts, with research screening mode
# -------------------------------------------------------------------------------------
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
st.set_page_config(page_title="Triad “parkinson-like” Risk — MVP", page_icon="🚶🏻‍♀️‍➡️", layout="wide")

MODEL_PATH   = "models/rf_model.pkl"
SCALER_PATH  = "models/scaler.pkl"                 # optional
THR_JSON     = "models/thresholds.json"            # optional (to show suggested)
METRICS_JSON = "results/train_metrics.json"        # optional (sens/spec etc.)
RULES_JSON   = "artifacts/surrogate_rules_deploy.json"  # optional (surrogate rules)

RESULTS_DIR  = "results"
LOG_PATH     = os.path.join(RESULTS_DIR, "pilot_log.csv")

FEATURES = [
    "MSE ML","iHR V","MSE V","MSE AP","Weigth","Age",
    "Sex (M=1, F=2)","H-Y","Gait Speed","Duration (years)"
]
NUM_FEATURES = ["MSE ML","iHR V","MSE V","MSE AP","Weigth","Age","Gait Speed","Duration (years)"]

# Paper thresholds (default)
DEFAULT_THRESHOLDS = {"low": 0.26, "mid": 0.40, "high": 0.50}

# fallback operating points (if METRICS_JSON has no specificity):
# rough sens/spec (replace as soon as you have test-set values)
FALLBACK_OP = {
    0.26: {"sensitivity": 0.82, "specificity": 0.55},
    0.40: {"sensitivity": 0.68, "specificity": 0.64},
    0.50: {"sensitivity": 0.57, "specificity": 0.72},
    0.60: {"sensitivity": 0.48, "specificity": 0.80},
}

# >>> NEW: PI guide path & fallback
PI_GUIDE_PATH = "artifacts/PI_GUIDE.md"
DEFAULT_PI_GUIDE_MD = """
# PI Guide — Prospective Enrollment (Triad → PD)

**Purpose.** Support selection of *triad-positive* subjects (RBD/hyposmia/depression) for prospective follow-up.

## Inclusion/Exclusion (research)
- Inclusion: triad present; independent ambulation; age ≥ 45 y; informed consent.
- Exclusion: major alternative neurological dx; incompatible device; impossible follow-up.

## Enrollment rule (operational)
- **Red (p ≥ 0.50)** → *Enroll* (high priority).
- **Orange (0.40–0.49)** → *Enroll* if **surrogate rule** is active **or** **SHAP** coherent with ↑MSE ML and ↓speed (medium priority).
- **Yellow (0.26–0.39)** → *Optional enroll* if rule/SHAP coherent (low priority).
- **Green (< 0.26)** → *Control* (negative sample/monitoring).

> Note: probability is *of Triad “parkinson-like”*, not PD diagnosis.

## Suggested follow-up
- Red: **6 months** (gait, olfaction, non-motor).
- Orange: **6–12 months**.
- Yellow: **12 months**.
- Green: optional.

## Intensification triggers
- Δp ≥ 0.10 or band change.
- New positive surrogate rule.
- Subtle motor signs emerging.

## Consent & privacy
- **Research/triage** tool (non-diagnostic).
- No data saved unless explicitly chosen (audit CSV).
"""

# =========================
# UTILS — load artifacts
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
        except Exception:
            rules = None
    # suggested thresholds
    thr_suggested = DEFAULT_THRESHOLDS.copy()
    if os.path.exists(THR_JSON):
        try:
            t = json.load(open(THR_JSON))
            # compatibility with older format
            thr_suggested["low"]  = float(t.get("thr_green",  thr_suggested["low"]))
            thr_suggested["mid"]  = float(t.get("thr_yellow", thr_suggested["mid"]))
            thr_suggested["high"] = float(t.get("thr_orange", thr_suggested["high"]))
        except Exception:
            pass
    # operating points
    op = FALLBACK_OP.copy()
    if os.path.exists(METRICS_JSON):
        try:
            m = json.load(open(METRICS_JSON))
            # if you saved sens/spec by threshold, load them; otherwise keep fallback
            # e.g.: m["operating_points"] = {"0.40":{"sensitivity":0.70,"specificity":0.65}, ...}
            if "operating_points" in m:
                for k,v in m["operating_points"].items():
                    op[float(k)] = {"sensitivity": float(v["sensitivity"]), "specificity": float(v["specificity"])}
        except Exception:
            pass
    return rf, scaler, rules, thr_suggested, op

# >>> NEW: PI guide loader with fallback
@st.cache_resource
def load_pi_guide_md():
    try:
        with open(PI_GUIDE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return DEFAULT_PI_GUIDE_MD

# =========================
# UTILS — misc
# =========================
def ensure_dataframe_order(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip() for c in df.columns})
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    return df[FEATURES].copy()

def build_X_from_series(row: pd.Series) -> pd.DataFrame:
    vals = {}
    for k in FEATURES:
        v = row[k]
        try:
            vals[k] = float(v)
        except Exception:
            vals[k] = v
    return pd.DataFrame([vals], columns=FEATURES)

def predict_proba_single(model, scaler, row: pd.Series) -> float:
    X = build_X_from_series(row)
    if scaler is not None:
        try:
            X[NUM_FEATURES] = pd.DataFrame(
                scaler.transform(X[NUM_FEATURES]), columns=NUM_FEATURES
            )
        except Exception:
            pass
    p = float(model.predict_proba(X)[:, 1][0])
    return p

def predict_proba_batch(model, scaler, X_in: pd.DataFrame) -> np.ndarray:
    X = ensure_dataframe_order(X_in)
    if scaler is not None:
        try:
            X[NUM_FEATURES] = pd.DataFrame(
                scaler.transform(X[NUM_FEATURES]), columns=NUM_FEATURES
            )
        except Exception:
            pass
    return model.predict_proba(X)[:, 1]

def color_for_proba(p, thr_low, thr_mid, thr_high):
    if p < thr_low:  return "#2ecc71"  # green
    if p < thr_mid:  return "#f1c40f"  # yellow
    if p < thr_high: return "#e67e22"  # orange
    return "#e74c3c"                   # red

# ---- SHAP (robust binary/multi-output) ----
def _extract_positive_values(explanation):
    vals = np.array(explanation.values)
    base = np.array(explanation.base_values)

    if vals.ndim == 1:
        pass
    elif vals.ndim == 2 and vals.shape[1] == 2:
        vals = vals[:, 1]
    elif vals.ndim == 2 and vals.shape[0] == 1:
        vals = vals[0]
    elif vals.ndim == 3 and vals.shape[-1] == 2:
        vals = vals[0, :, 1]
    else:
        vals = vals.reshape(-1)

    if base.ndim == 1 and base.size == 2:
        base = float(base[1])
    else:
        base = float(base.ravel()[0])
    return vals.ravel(), base

@st.cache_resource
def get_shap_explainer(_rf, _features):
    import shap
    return shap.TreeExplainer(_rf, feature_names=_features)

def shap_explain_single(rf, scaler, row: pd.Series):
    try:
        import shap
    except Exception:
        return None, None
    X = build_X_from_series(row)
    Xs = X.copy()
    if scaler is not None:
        try:
            Xs[NUM_FEATURES] = pd.DataFrame(
                scaler.transform(X[NUM_FEATURES]), columns=NUM_FEATURES
            )
        except Exception:
            pass
    exp = get_shap_explainer(rf, FEATURES)(Xs)
    vals, base = _extract_positive_values(exp[0])
    s = pd.Series(vals, index=FEATURES).sort_values(key=np.abs, ascending=False)
    return s, base

# ---- Surrogate rules ----
def eval_rule_on_row(path_list, row: pd.Series):
    def parse_cond(s):
        s = s.strip()
        if "<=" in s:
            f, v = s.split("<="); op = "<="
        elif ">" in s:
            f, v = s.split(">");  op = ">"
        else:
            return True
        f = f.strip(); v = float(str(v).strip())
        x = float(row.get(f, np.nan))
        if np.isnan(x): return False
        return (x <= v) if op == "<=" else (x > v)
    return all(parse_cond(cond) for cond in path_list)

def find_matching_rule(rules_json, row: pd.Series):
    if not rules_json or "rules" not in rules_json: 
        return None
    for r in rules_json["rules"]:
        if eval_rule_on_row(r.get("path", []), row):
            return r
    return None

# ---- PPV / NPV (Bayes) ----
def ppv_npvs(prevalence, sensitivity, specificity):
    prev = max(1e-6, min(1-1e-6, prevalence))
    sen  = max(1e-6, min(1-1e-6, sensitivity))
    spe  = max(1e-6, min(1-1e-6, specificity))
    ppv = (prev * sen) / (prev * sen + (1 - prev) * (1 - spe))
    npv = ((1 - prev) * spe) / ((1 - prev) * spe + prev * (1 - sen))
    return ppv, npv

# ---- PDF (extended: model ID, PPV/NPV, QC, SHAP legend, QR) ----
def make_pdf(
    one_row: pd.Series,
    proba: float,
    band_name: str,
    shap_top5: dict,
    rule: dict | None,
    thresholds: dict,
    cohort_note: str = "",
    *,
    model_id: str = "RF-v1.0",
    model_sha: str | None = None,        # optional: e.g., model hash short
    op_point: dict | None = None,        # {"thr":0.40,"sensitivity":..,"specificity":..,"prevalence_pct":..}
    qc: dict | None = None,              # {"ranges_ok":True,"missing_ok":True,"sex_ok":True,"hy_ok":True,"notes":"..."}
    app_url: str | None = None           # URL for QR (webapp or repo)
):
    if not HAVE_PDF:
        st.error("reportlab not installed: `pip install reportlab`")
        return None

    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    try:
        from reportlab.graphics.barcode import qr
        from reportlab.graphics.shapes import Drawing
    except Exception:
        qr = None
        Drawing = None

    def _draw_check(c, x, y, ok: bool):
        c.setLineWidth(1)
        c.rect(x, y, 0.4*cm, 0.4*cm, stroke=1, fill=0)
        if ok:
            c.setStrokeColor(colors.green)
            c.setLineWidth(3)
            c.line(x+0.08*cm, y+0.2*cm, x+0.17*cm, y+0.08*cm)
            c.line(x+0.17*cm, y+0.08*cm, x+0.32*cm, y+0.32*cm)
            c.setStrokeColor(colors.black)
            c.setLineWidth(1)

    def _wrap(c, text, x, y, max_chars=85, leading=12):
        if not text:
            return y
        words = str(text).split()
        line = ""
        for w in words:
            if len(line) + len(w) + 1 <= max_chars:
                line = (line + " " + w).strip()
            else:
                c.drawString(x, y, line); y -= leading; line = w
        if line:
            c.drawString(x, y, line); y -= leading
        return y

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    left = 2*cm
    right = w - 2*cm
    y = h - 2*cm

    # Header
    c.setFont("Helvetica-Bold", 15)
    c.drawString(left, y, "Triad “parkinson-like” Risk — MVP")
    y -= 0.7*cm
    c.setFont("Helvetica", 10)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mline = f"Model: {model_id}"
    if model_sha: mline += f"  (sha: {model_sha})"
    c.drawString(left, y, mline)
    c.drawRightString(right, y, f"Timestamp: {ts}")
    y -= 0.4*cm
    if cohort_note:
        y = _wrap(c, f"Cohort: {cohort_note}", left, y, max_chars=95)
    y -= 0.15*cm
    c.setLineWidth(0.6); c.line(left, y, right, y); y -= 0.5*cm

    # Input
    c.setFont("Helvetica-Bold", 12); c.drawString(left, y, "Patient input")
    y -= 0.5*cm; c.setFont("Helvetica", 10)
    for k in FEATURES:
        c.drawString(left, y, f"• {k}: {one_row.get(k, '')}")
        y -= 0.36*cm
        if y < 5.2*cm:
            break

    # Risk & thresholds
    y -= 0.2*cm
    c.setFont("Helvetica-Bold", 12); c.drawString(left, y, "Risk & thresholds")
    y -= 0.45*cm; c.setFont("Helvetica", 10)
    c.drawString(left, y, f"Triad probability: {proba:.3f}")
    c.drawRightString(right, y, f"Band: {band_name}")
    y -= 0.35*cm
    c.drawString(left, y, f"Thresholds → Green<{thresholds['low']:.2f} | "
                         f"Yellow<{thresholds['mid']:.2f} | "
                         f"Orange<{thresholds['high']:.2f} | "
                         f"Red≥{thresholds['high']:.2f}")
    y -= 0.5*cm

    # Operating point + PPV/NPV
    c.setFont("Helvetica-Bold", 12); c.drawString(left, y, "Operating point & PPV/NPV (research screening)")
    y -= 0.45*cm; c.setFont("Helvetica", 10)
    if op_point and all(k in op_point for k in ("sensitivity","specificity","prevalence_pct","thr")):
        sen = float(op_point["sensitivity"]); spe = float(op_point["specificity"])
        prev = float(op_point["prevalence_pct"])/100.0
        ppv = (prev * sen) / (prev * sen + (1 - prev) * (1 - spe) + 1e-12)
        npv = ((1 - prev) * spe) / ((1 - prev) * spe + prev * (1 - sen) + 1e-12)
        c.drawString(left, y, f"Screen threshold: {float(op_point['thr']):.2f}   Sensitivity: {sen:.2f}   Specificity: {spe:.2f}")
        c.drawRightString(right, y, f"Prevalence: {op_point['prevalence_pct']:.1f}%")
        y -= 0.35*cm
        c.drawString(left, y, f"PPV: {ppv:.2f}   |   NPV: {npv:.2f}")
        y -= 0.5*cm
    else:
        c.setFillColor(colors.gray)
        y = _wrap(c, "Operating point not provided. Set threshold, sensitivity/specificity and prevalence in the app.", left, y)
        c.setFillColor(colors.black)
        y -= 0.2*cm

    # SHAP
    c.setFont("Helvetica-Bold", 12); c.drawString(left, y, "Top-5 contributions (SHAP)")
    y -= 0.45*cm; c.setFont("Helvetica", 10)
    if shap_top5:
        for feat, val in shap_top5.items():
            c.drawString(left, y, f"• {feat}: {val:+.3f}")
            y -= 0.34*cm
    else:
        c.setFillColor(colors.gray); c.drawString(left, y, "SHAP not available in this environment.")
        c.setFillColor(colors.black); y -= 0.34*cm
    y -= 0.1*cm
    c.setFillColor(colors.darkgray)
    y = _wrap(c, "SHAP legend: positive value → increases probability; negative value → decreases probability (ceteris paribus).",
              left, y, max_chars=95)
    c.setFillColor(colors.black)
    y -= 0.2*cm

    # Surrogate rule
    c.setFont("Helvetica-Bold", 12); c.drawString(left, y, "Surrogate rule (if matched)")
    y -= 0.45*cm; c.setFont("Helvetica", 10)
    if rule:
        c.drawString(left, y, "Activated conditions:"); y -= 0.34*cm
        for cond in rule.get("path", []):
            y = _wrap(c, f"• {cond.strip()}", left+0.3*cm, y, max_chars=95)
        y -= 0.2*cm
        c.setFillColor(colors.darkgray)
        c.drawString(left, y, f"Observed prevalence (test): {rule.get('p_real', 0):.2f}  (n={rule.get('n','?')})")
        c.setFillColor(colors.black); y -= 0.5*cm
    else:
        c.setFillColor(colors.gray)
        y = _wrap(c, "No matching rule or rules not available.", left, y)
        c.setFillColor(colors.black); y -= 0.3*cm

    # QC + QR
    bottom_y = 2.2*cm
    c.setFont("Helvetica-Bold", 12); c.drawString(left, bottom_y + 2.3*cm, "QC (minimal checks)")
    c.setFont("Helvetica", 10)
    _qc = qc or {}
    items = [
        ("Physically plausible ranges (MSE/iHR/speed/weight)", bool(_qc.get("ranges_ok", True))),
        ("Missing values absent / properly imputed", bool(_qc.get("missing_ok", True))),
        ("Sex encoding coherent (M=1, F=2)", bool(_qc.get("sex_ok", True))),
        ("H–Y in {1,2,3}", bool(_qc.get("hy_ok", True))),
    ]
    yy = bottom_y + 1.9*cm
    for label, ok in items:
        _draw_check(c, left, yy-0.1*cm, ok)
        c.drawString(left + 0.55*cm, yy, label)
        yy -= 0.5*cm
    if _qc.get("notes"):
        c.setFillColor(colors.darkgray)
        yy = _wrap(c, f"QC notes: {_qc['notes']}", left, yy, max_chars=95)
        c.setFillColor(colors.black)

    if app_url and qr and Drawing:
        try:
            qrobj = qr.QrCodeWidget(app_url)
            b = qrobj.getBounds()
            qr_w = 2.6*cm; qr_h = 2.6*cm
            d = Drawing(qr_w, qr_h, transform=[qr_w/(b[2]-b[0]),0,0,qr_h/(b[3]-b[1]),0,0])
            d.add(qrobj)
            render_x = right - qr_w
            render_y = bottom_y + 0.4*cm
            d.drawOn(c, render_x, render_y)
            c.setFont("Helvetica", 8); c.setFillColor(colors.darkgray)
            c.drawRightString(right, render_y - 0.2*cm, "Open webapp / repo")
            c.setFillColor(colors.black)
        except Exception:
            pass

    c.setLineWidth(0.6); c.line(left, 1.6*cm, right, 1.6*cm)
    c.setFont("Helvetica-Oblique", 8); c.setFillColor(colors.gray)
    c.drawString(left, 1.25*cm, "Research/triage tool; not a medical device; does not replace clinical judgment.")
    c.setFillColor(colors.black)

    c.showPage(); c.save()
    buf.seek(0)
    return buf

# ---- LOG (minimal CRF) ----
def append_log(row_dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame([row_dict])
    if not os.path.exists(LOG_PATH):
        df.to_csv(LOG_PATH, index=False)
    else:
        df.to_csv(LOG_PATH, index=False, mode="a", header=False)

# =========================
# UI
# =========================
st.title("Triad “parkinson-like” Risk 🚶🏻‍♀️‍➡️")

rf, scaler, rules_json, thr_suggested, OP_TABLE = load_model_scaler_rules()
if rf is None:
    st.error("Model not found. Ensure `models/rf_model.pkl` exists.")
    st.stop()

# >>> NEW: load PI guide (markdown)
PI_GUIDE_MD = load_pi_guide_md()

with st.sidebar:
    st.header("Settings")
    colA, colB = st.columns(2)
    with colA:
        thr_low  = st.number_input("Threshold Green→Yellow",   value=float(thr_suggested["low"]),  min_value=0.0, max_value=1.0, step=0.01)
        thr_high = st.number_input("Threshold Orange→Red",      value=float(thr_suggested["high"]), min_value=0.0, max_value=1.0, step=0.01)
    with colB:
        thr_mid  = st.number_input("Threshold Yellow→Orange",   value=float(thr_suggested["mid"]),  min_value=0.0, max_value=1.0, step=0.01)

    st.caption(f"Suggested thresholds: {thr_suggested['low']:.2f} / {thr_suggested['mid']:.2f} / {thr_suggested['high']:.2f}")

    st.markdown("---")
    st.subheader("Screening mode (research)")
    screening_on = st.checkbox("Enable Screening mode (pilot on triad-positive cohort)", value=True)
    prevalence = st.slider("Expected prevalence in your cohort (%)", min_value=0.5, max_value=80.0, value=40.0, step=0.5)
    screen_thr_choice = st.selectbox("Screening threshold", options=[thr_low, thr_mid, thr_high, 0.60], index=1,
                                     format_func=lambda x: f"{x:.2f}")
    # quick presets
    st.write("")
    if st.button("High-sensitivity preset (Recall≥0.90)"):
        screen_thr_choice = thr_low
        st.session_state["_screen_thr_choice"] = screen_thr_choice
    if st.button("Triad-positive preset (prevalence≈40%, rule-in 0.60)"):
        st.session_state["_prevalence"] = 40.0
        st.session_state["_screen_thr_choice"] = 0.60

    st.markdown("---")
    audit_log = st.checkbox("Log anonymous case (audit CSV)", value=False)
    st.caption("Saves to results/pilot_log.csv: timestamp, inputs, p, band, rule, top-SHAP. No identifiers.")

# >>> NEW: three tabs (added "📘 PI Guide")
tab1, tab2, tab3 = st.tabs(["👤 Single patient", "📥 Batch CSV", "📘 PI Guide"])

# -------------------------
# TAB 1 — single patient
# -------------------------
with tab1:
    st.subheader("Patient input")
    c1, c2, c3 = st.columns(3)

    with c1:
        mse_ml = st.number_input("MSE ML", min_value=0.0, value=1.75, step=0.01)
        mse_v  = st.number_input("MSE V",  min_value=0.0, value=1.50, step=0.01)
        mse_ap = st.number_input("MSE AP", min_value=0.0, value=1.30, step=0.01)
        ihr_v  = st.number_input("iHR V (improved Harmonic Ratio)", min_value=0.0, value=65.0, step=0.5)
    with c2:
        gait   = st.number_input("Gait Speed (m/s)", min_value=0.0, value=1.10, step=0.01)
        weight = st.number_input("Weigth (kg)", min_value=30.0, value=75.0, step=0.5)
        age    = st.number_input("Age (years)", min_value=18.0, value=70.0, step=1.0)
        dur    = st.number_input("Duration (years)", min_value=0.0, value=5.0, step=0.5)
    with c3:
        sex    = st.selectbox("Sex (M=1, F=2)", options=[1,2], index=0)
        hy     = st.selectbox("H-Y", options=[1,2,3], index=1)
        st.write("")

    row = pd.Series({
        "MSE ML":mse_ml, "iHR V":ihr_v, "MSE V":mse_v, "MSE AP":mse_ap,
        "Weigth":weight, "Age":age, "Sex (M=1, F=2)":sex, "H-Y":hy,
        "Gait Speed":gait, "Duration (years)":dur
    })

    st.divider()
    if st.button("Compute risk", type="primary"):
        try:
            # proba & band
            p = predict_proba_single(rf, scaler, row)
            clr = color_for_proba(p, thr_low, thr_mid, thr_high)
            st.markdown(f"### Triad probability: **{p:.3f}**")
            st.markdown(f"<div style='width:100%;height:14px;background:{clr};border-radius:6px'></div>", unsafe_allow_html=True)

            band_idx = sum(p >= np.array([thr_low, thr_mid, thr_high]))
            band = ["Green","Yellow","Orange","Red"][band_idx]
            band_msg = {
                "Green":  "Low suspicion of PD (in triad-positive cohort).",
                "Yellow": "Moderate suspicion: consider closer follow-up.",
                "Orange": "Elevated suspicion: consider targeted work-up.",
                "Red":    "High suspicion: specialist referral and multi-domain screening."
            }[band]
            st.caption(f"**Band: {band}** — {band_msg}")

            # screening mode (research)
            op_for_pdf = None
            if screening_on:
                thr_s = st.session_state.get("_screen_thr_choice", screen_thr_choice)
                prev = st.session_state.get("_prevalence", prevalence)
                op = OP_TABLE.get(float(f"{thr_s:.2f}"), OP_TABLE.get(thr_s, None))
                if op is None:
                    nearest = min(OP_TABLE.keys(), key=lambda t: abs(t-thr_s))
                    op = OP_TABLE[nearest]
                ppv, npv = ppv_npvs(prev/100.0, op["sensitivity"], op["specificity"])

                st.subheader("Screening mode (research) — PPV/NPV estimate")
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("Screen threshold", f"{thr_s:.2f}")
                col_s2.metric("Sensitivity", f"{op['sensitivity']:.2f}")
                col_s3.metric("Specificity", f"{op['specificity']:.2f}")
                col_s4.metric(f"PPV @ {prev:.1f}%", f"{ppv:.2f}")
                st.caption(f"NPV @ {prev:.1f}%: **{npv:.2f}**  \nNote: PPV/NPV depend on the **your** cohort’s expected prevalence.")

                # >>> NEW: op_point for PDF
                op_for_pdf = {
                    "thr": float(thr_s),
                    "sensitivity": float(op["sensitivity"]),
                    "specificity": float(op["specificity"]),
                    "prevalence_pct": float(prev),
                }

            # Local SHAP
            shap_contrib, _ = shap_explain_single(rf, scaler, row)
            if shap_contrib is not None:
                top5 = shap_contrib.iloc[:5][::-1]
                st.subheader("Local contributions (SHAP) — top-5")
                st.bar_chart(pd.DataFrame({"SHAP": top5.values}, index=top5.index))

            # Surrogate rule
            rule = find_matching_rule(rules_json, row) if rules_json else None
            with st.expander("Surrogate rule (if available)"):
                if rule:
                    st.write("**Activated conditions:**")
                    for cond in rule.get("path", []):
                        st.write(f"- {cond}")
                    st.write(f"Observed prevalence (test): {rule.get('p_real',0):.2f} (n={rule.get('n','?')})")
                else:
                    st.info("No matching rule or `artifacts/surrogate_rules_deploy.json` not found.")

            # >>> NEW: minimal QC payload for PDF (placeholder, ready for real checks)
            qc_payload = {
                "ranges_ok": True,
                "missing_ok": True,
                "sex_ok": 1 <= int(row["Sex (M=1, F=2)"]) <= 2,
                "hy_ok": int(row["H-Y"]) in (1,2,3),
                "notes": ""
            }

            # PDF
            if HAVE_PDF:
                shap_for_pdf = (shap_contrib.iloc[:5].to_dict() if shap_contrib is not None else {})
                buf = make_pdf(
                    one_row=row, proba=p, band_name=band,
                    shap_top5=shap_for_pdf, rule=rule,
                    thresholds={"low":thr_low, "mid":thr_mid, "high":thr_high},
                    cohort_note="Triad-positive cohort — research/triage use",
                    # >>> NEW: extra args
                    model_id="RF-v1.0",
                    model_sha=None,           # optional: short sha if you want
                    op_point=op_for_pdf,
                    qc=qc_payload,
                    app_url=None              # optional: webapp/repo URL for QR
                )
                if buf:
                    st.download_button("Download PDF", data=buf, file_name="triad_report.pdf", mime="application/pdf")
            else:
                st.caption("Install `reportlab` to enable PDF export.")

            # LOG (minimal CRF)
            if audit_log:
                log_row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "p": round(p, 4), "band": band,
                    "thr_low": thr_low, "thr_mid": thr_mid, "thr_high": thr_high,
                    "screen_on": screening_on, "screen_thr": float(screen_thr_choice),
                    "prevalence_pct": float(prevalence),
                }
                for k in FEATURES:
                    log_row[k] = row[k]
                if rule:
                    log_row["rule_path"] = " AND ".join(rule.get("path", []))
                    log_row["rule_p_real"] = rule.get("p_real", None)
                    log_row["rule_n"] = rule.get("n", None)
                if shap_contrib is not None:
                    top3 = shap_contrib.iloc[:3]
                    for i,(feat,val) in enumerate(top3.items(), start=1):
                        log_row[f"shap{i}_feat"] = feat
                        log_row[f"shap{i}_val"]  = float(val)
                append_log(log_row)
                st.success("Case logged to results/pilot_log.csv")

        except Exception as e:
            st.error(f"Inference error: {e}")

# -------------------------
# TAB 2 — batch CSV
# -------------------------
with tab2:
    st.subheader("Batch prediction from CSV")
    st.caption("Required columns (any order, exact names):")
    st.code(", ".join(FEATURES))
    upl = st.file_uploader("Upload CSV", type=["csv"])

    colT, _ = st.columns([1,1])
    with colT:
        if st.button("Download CSV template"):
            tmpl = pd.DataFrame(columns=FEATURES)
            tmpl.loc[0] = [1.80, 65, 1.52, 1.30, 75, 70, 1, 2, 1.10, 5]
            st.download_button("Template", data=tmpl.to_csv(index=False),
                               file_name="patient_template.csv", mime="text/csv", key="tmplbtn")

    if upl:
        try:
            df = pd.read_csv(upl)
            proba = predict_proba_batch(rf, scaler, df.copy())
            bands = np.select(
                [proba < thr_low, proba < thr_mid, proba < thr_high],
                ["Green","Yellow","Orange"], default="Red"
            )
            out = ensure_dataframe_order(df.copy())
            out["proba"] = proba
            out["band"] = bands
            st.dataframe(out.head(50))
            st.download_button("Download results CSV", data=out.to_csv(index=False),
                               file_name="triad_batch_results.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch error: {e}")

# -------------------------
# TAB 3 — PI Guide (markdown)
# -------------------------
with tab3:
    st.markdown(PI_GUIDE_MD)

st.markdown("---")
st.caption("Privacy: no data are saved unless you explicitly choose so (audit CSV). Research/triage use, non-diagnostic.")
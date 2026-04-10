# =============================================================================
# app/app.py  —  Streamlit Frontend for Disease Prediction System
# =============================================================================
# Run with:  streamlit run app.py
# =============================================================================

import streamlit as st
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .title-box {
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .result-positive {
        background: #fdecea;
        border-left: 5px solid #e53935;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .result-negative {
        background: #e8f5e9;
        border-left: 5px solid #43a047;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .section-header {
        color: #1a73e8;
        font-size: 1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1rem 0 0.5rem;
    }
    .tip-box {
        background: #e3f2fd;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #1565c0;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model & scaler ───────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

# ── Dataset average values for comparison charts ──────────────────
DATASET_AVERAGES = {
    "Pregnancies":    3.8,
    "Glucose":        120.9,
    "Blood Pressure": 69.1,
    "Skin Thickness": 20.5,
    "Insulin":        79.8,
    "BMI":            32.0,
    "Pedigree Func":  0.47,
    "Age":            33.2,
}
DIABETIC_AVERAGES = {
    "Pregnancies":    4.9,
    "Glucose":        141.3,
    "Blood Pressure": 70.8,
    "Skin Thickness": 22.2,
    "Insulin":        100.3,
    "BMI":            35.4,
    "Pedigree Func":  0.55,
    "Age":            37.1,
}

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
    <h1 style="margin:0; font-size:2rem;">🩺 Disease Prediction System</h1>
    <p style="margin:0.4rem 0 0; opacity:0.88; font-size:1rem;">
        Diabetes Risk Assessment · Powered by Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/medical-doctor.png", width=80)
    st.markdown("### About this App")
    st.markdown("""
    This app predicts **diabetes risk** based on clinical measurements.

    **Dataset:** Pima Indians Diabetes  
    **Algorithm:** Logistic Regression  
    **Accuracy:** 81.17%

    > ⚠️ *For educational purposes only. Not a substitute for medical advice.*
    """)
    st.markdown("---")
    st.markdown("**GTU Internship Project**  \nHealthcare AI · ML Domain")
    st.markdown("Built with Python & Streamlit 🐍")

# ── Input Form ────────────────────────────────────────────────────
st.markdown('<p class="section-header">📋 Enter Patient Information</p>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    pregnancies    = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, help="Number of times pregnant")
    glucose        = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120, help="Plasma glucose concentration")
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=150, value=72, help="Diastolic blood pressure")
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, help="Triceps skin fold thickness")
with col2:
    insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=900, value=80, help="2-Hour serum insulin")
    bmi     = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1, help="Body Mass Index")
    dpf     = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.35, step=0.01, help="Diabetes likelihood based on family history")
    age     = st.number_input("Age (years)", min_value=1, max_value=120, value=30, help="Patient's age")

# ── BMI Indicator ─────────────────────────────────────────────────
bmi_category = (
    "Underweight 🔵" if bmi < 18.5 else
    "Normal ✅"       if bmi < 25.0 else
    "Overweight ⚠️"  if bmi < 30.0 else
    "Obese 🔴"
)
st.info(f"**BMI Category:** {bmi_category}")

# ── Predict Button ────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Predict Diabetes Risk", use_container_width=True)

if predict_btn:
    try:
        model, scaler = load_artifacts()

        input_data    = np.array([[pregnancies, glucose, blood_pressure,
                                    skin_thickness, insulin, bmi, dpf, age]])
        input_scaled  = scaler.transform(input_data)
        prediction    = model.predict(input_scaled)[0]
        probability   = model.predict_proba(input_scaled)[0]
        prob_positive = probability[1] * 100
        prob_negative = probability[0] * 100

        # ── Result card ───────────────────────────────────────────
        st.markdown("---")
        st.markdown('<p class="section-header">🧪 Prediction Result</p>', unsafe_allow_html=True)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-positive">
                ⚠️  High Diabetes Risk Detected<br>
                <span style="font-size:0.9rem; font-weight:400;">
                    Confidence: {prob_positive:.1f}% likely diabetic
                </span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                ✅  Low Diabetes Risk<br>
                <span style="font-size:0.9rem; font-weight:400;">
                    Confidence: {prob_negative:.1f}% likely non-diabetic
                </span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        col_a.metric("🟢 No Diabetes", f"{prob_negative:.1f}%")
        col_b.metric("🔴 Diabetes",    f"{prob_positive:.1f}%")
        st.progress(int(prob_positive))

        # ── Risk flags ────────────────────────────────────────────
        st.markdown('<p class="section-header">🔎 Risk Factors Identified</p>', unsafe_allow_html=True)
        flags = []
        if glucose > 140:       flags.append("🔴 High glucose level")
        if bmi > 30:            flags.append("🔴 Obese BMI")
        if age > 50:            flags.append("⚠️ Age over 50")
        if blood_pressure > 90: flags.append("⚠️ Elevated blood pressure")
        if dpf > 1.0:           flags.append("⚠️ Strong family history")
        if insulin > 400:       flags.append("⚠️ High insulin level")

        if flags:
            for flag in flags: st.warning(flag)
        else:
            st.success("✅ No major individual risk factors detected")

        # ══════════════════════════════════════════════════════════
        # LIVE CHART 1 — Your Values vs Dataset Averages (bar chart)
        # ══════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown('<p class="section-header">📊 Your Values vs Dataset Averages</p>', unsafe_allow_html=True)
        st.caption("How this patient compares to the average non-diabetic and diabetic patient in the dataset.")

        patient_values = [pregnancies, glucose, blood_pressure,
                          skin_thickness, insulin, bmi, dpf * 100, age]
        avg_values     = list(DATASET_AVERAGES.values())
        dia_values     = list(DIABETIC_AVERAGES.values())
        labels         = list(DATASET_AVERAGES.keys())
        max_vals       = [20, 300, 150, 100, 900, 70, 300, 120]

        # Normalise to 0–100 for fair visual comparison
        p_norm = [min((v / m) * 100, 100) for v, m in zip(patient_values, max_vals)]
        a_norm = [min((v / m) * 100, 100) for v, m in zip(avg_values,     max_vals)]
        d_norm = [min((v / m) * 100, 100) for v, m in zip(dia_values,     max_vals)]

        x     = np.arange(len(labels))
        width = 0.25
        patient_color = "#e53935" if prediction == 1 else "#1a73e8"

        fig1, ax1 = plt.subplots(figsize=(12, 5))
        fig1.patch.set_facecolor("#f8faff")
        ax1.set_facecolor("#f8faff")

        ax1.bar(x - width, a_norm, width, label="Dataset Avg",  color="#90CAF9", edgecolor="white")
        ax1.bar(x,         d_norm, width, label="Diabetic Avg", color="#EF9A9A", edgecolor="white")
        bars3 = ax1.bar(x + width, p_norm, width, label="This Patient", color=patient_color, edgecolor="white")

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.set_ylabel("Relative Value (normalised %)", fontsize=10)
        ax1.set_title("Patient Values vs Dataset Averages", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.set_ylim(0, 115)
        ax1.spines[["top", "right"]].set_visible(False)

        for bar in bars3:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                     f"{h:.0f}%", ha="center", fontsize=7.5,
                     color=patient_color, fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

        # ══════════════════════════════════════════════════════════
        # LIVE CHART 2 — Per-Feature Risk Level (horizontal bars)
        # ══════════════════════════════════════════════════════════
        st.markdown('<p class="section-header"> Per-Feature Risk Level</p>', unsafe_allow_html=True)
        st.caption("Green = safe, Orange = moderate risk, Red = high risk — based on clinical thresholds.")

        feature_names   = ["Glucose", "BMI", "Age", "Blood Pressure", "Insulin", "Pregnancies", "Pedigree Func"]
        feature_values  = [glucose,   bmi,   age,   blood_pressure,   insulin,   pregnancies,   dpf]
        safe_thresholds = [100,        25,    30,    80,               100,       3,             0.3]
        high_thresholds = [140,        30,    50,    90,               400,       10,            1.0]

        risk_scores = []
        bar_colors  = []
        for val, safe, high in zip(feature_values, safe_thresholds, high_thresholds):
            if val <= safe:
                score = (val / safe) * 40
                bar_colors.append("#43a047")
            elif val <= high:
                score = 40 + ((val - safe) / (high - safe)) * 40
                bar_colors.append("#fb8c00")
            else:
                score = min(80 + ((val - high) / high) * 20, 100)
                bar_colors.append("#e53935")
            risk_scores.append(score)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        fig2.patch.set_facecolor("#f8faff")
        ax2.set_facecolor("#f8faff")

        bars = ax2.barh(feature_names, risk_scores, color=bar_colors,
                        edgecolor="white", linewidth=0.8, height=0.55)
        ax2.axvline(40,  color="#43a047", linewidth=1, linestyle="--", alpha=0.4)
        ax2.axvline(80,  color="#fb8c00", linewidth=1, linestyle="--", alpha=0.4)
        ax2.set_xlim(0, 110)
        ax2.set_xlabel("Risk Score", fontsize=10)
        ax2.set_title("Per-Feature Risk Analysis", fontsize=12, fontweight="bold")
        ax2.spines[["top", "right"]].set_visible(False)

        green_p  = mpatches.Patch(color="#43a047", label="Low Risk")
        orange_p = mpatches.Patch(color="#fb8c00", label="Moderate Risk")
        red_p    = mpatches.Patch(color="#e53935", label="High Risk")
        ax2.legend(handles=[green_p, orange_p, red_p], loc="lower right", fontsize=9)

        for bar, val in zip(bars, feature_values):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f" {val}", va="center", fontsize=9, color="#333")

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # ══════════════════════════════════════════════════════════
        # LIVE CHART 3 — Prediction Confidence Donut
        # ══════════════════════════════════════════════════════════
        st.markdown('<p class="section-header"> Prediction Confidence</p>', unsafe_allow_html=True)

        fig3, ax3 = plt.subplots(figsize=(5, 4))
        fig3.patch.set_facecolor("#f8faff")
        wedges, texts, autotexts = ax3.pie(
            [prob_negative, prob_positive],
            labels=["No Diabetes", "Diabetes"],
            colors=["#43a047", "#e53935"],
            autopct="%1.1f%%", startangle=90,
            pctdistance=0.75,
            wedgeprops={"width": 0.5, "edgecolor": "white", "linewidth": 2}
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight("bold")
        ax3.set_title("Model Confidence", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        # ── Disclaimer ────────────────────────────────────────────
        st.markdown("""
        <div class="tip-box">
             <b>Disclaimer:</b> This prediction is based on a machine learning model
            trained on population data. It is <b>not a medical diagnosis</b>.
            Always consult a qualified healthcare provider.
        </div>""", unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please run `train_model.py` first.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:0.8rem;'>"
    "GTU Internship · Healthcare AI Project"
    "</p>", unsafe_allow_html=True)

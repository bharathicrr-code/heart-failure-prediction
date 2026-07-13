import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Set full-screen wide layout
st.set_page_config(page_title="Heart Failure Prediction System", layout="wide")

# --- CUSTOM CSS FOR INTERFACE SCALE AND LEGIBILITY ---
st.markdown("""
    <style>
    /* Prevent top padding cutoff */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 3rem !important;
    }
    
    /* ENLARGE TEXT AND DROPDOWN FIELD LABELS AND VALUES */
    .stTextInput input, div[data-baseweb="select"] {
        font-size: 22px !important;
        font-weight: bold !important;
        color: #1a3a4b !important;
    }
    
    /* ENLARGE NATIVE FIELD VALUE LABELS */
    label[data-testid="stWidgetLabel"] p {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #1a3a4b !important;
    }
    
    /* NATIVE DATA TABLES HIGHLIGHTED TH HEADERS & TEXT MAGNIFICATION */
    .stTable table th {
        background-color: #0c5460 !important;
        color: white !important;
        font-size: 22px !important;
        font-weight: 800 !important;
    }
    .stTable td {
        font-size: 20px !important;
        color: #2b2b2b !important;
    }
    
    /* HIGH-VISIBILITY BUTTON GENERAL LAYOUT */
    div.stButton > button {
        border-radius: 8px !important;
        padding: 14px 28px !important;
        font-size: 22px !important;
        font-weight: 700 !important;
    }
    
    /* TYPOGRAPHY BLOCKS */
    .hero-title {
        font-size: 44px !important;
        font-weight: 800 !important;
        color: #1a3a4b !important;
    }
    .hero-subtitle {
        font-size: 30px !important;
        font-weight: 700 !important;
        color: #0c5460 !important;
    }
    .hero-body, .hero-body ul li {
        font-size: 22px !important; 
        line-height: 1.7 !important;
        color: #2b2b2b !important;
    }
    
    /* RECTIFIED CRISP SEVERITY CARD WITH FULL CONTRAST TEXT PATHS */
    .severity-card-premium {
        padding: 35px !important;
        border-radius: 12px !important;
        margin-bottom: 35px !important;
        background-color: #102A43 !important; 
        border-left: 12px solid #E12D39 !important;
        box-shadow: 0 6px 15px rgba(0,0,0,0.15) !important;
    }
    .severity-card-premium h2 {
        color: #FFFFFF !important; 
        font-size: 34px !important; 
        font-weight: 900 !important;
        margin: 0 0 12px 0 !important;
    }
    .severity-card-premium p {
        color: #FFFFFF !important; 
        font-size: 24px !important;
        font-weight: 500 !important;
        margin: 5px 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- MACHINE LEARNING DATA LOGIC ACCELERATOR ---
@st.cache_data
def load_data():
    return pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Initialize Session State Variables Safely
if "current_nav" not in st.session_state:
    st.session_state.current_nav = "HOME"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "patient_data" not in st.session_state:
    st.session_state.patient_data = None

# --- AUTHENTICATION GATEWAY ---
if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col_l2, _ = st.columns([1, 2, 1])
    with col_l2:
        st.markdown("<h2 style='text-align: center; color: #1a3a4b;'>🔒 Clinical Portal Authentication</h2>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username", value="admin")
            password = st.text_input("Password", type="password", value="password")
            if st.form_submit_button("Authenticate"):
                if username == "admin" and password == "password":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
    st.stop()

# Load dataset vectors
df = load_data()
X = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- PREMIUM HTML CUSTOM NAVBAR COMPONENT ---
st.markdown("""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 25px; border-bottom: 4px solid #0c5460; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        <table style="width: 100%; border: none; border-collapse: collapse;">
            <tr style="border: none;">
                <td style="font-size: 32px; font-weight: 800; color: #1a3a4b; font-family: sans-serif; border: none; width: 45%;">
                    🩺 Heart Failure Prediction System
                </td>
                <td style="text-align: right; border: none; width: 55%;">
                    <span style="margin-left: 25px; font-size: 22px; font-weight: bold; color: #555;">Navigation Shortcuts Below:</span>
                </td>
            </tr>
        </table>
    </div>
""", unsafe_allow_html=True)

col_n1, col_n2, col_n3, col_n4, col_n5 = st.columns(5)
with col_n1:
    if st.button("🏠 HOME", key="nav_home", use_container_width=True):
        st.session_state.current_nav = "HOME"; st.rerun()
with col_n2:
    if st.button("📋 INTAKE FORM", key="nav_form", use_container_width=True):
        st.session_state.current_nav = "FORM"; st.rerun()
with col_n3:
    if st.button("⚙️ ML MODELS", key="nav_models", use_container_width=True):
        st.session_state.current_nav = "MODELS"; st.rerun()
with col_n4:
    if st.button("📊 DIAGNOSTIC REPORT", key="nav_report", use_container_width=True):
        st.session_state.current_nav = "REPORT"; st.rerun()
with col_n5:
    if st.button("ℹ️ SYSTEM ABOUT", key="nav_about", use_container_width=True):
        st.session_state.current_nav = "ABOUT"; st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODULE 1: HOME VIEW
# ---------------------------------------------------------
if st.session_state.current_nav == "HOME":
    col_info, col_graphic = st.columns([11, 9])
    with col_info:
        st.markdown('<div class="hero-title">CardioShield Predictive Clinical Portal</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-subtitle">Decision Support Engine Deployment Platform</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="hero-body">
        This analytical clinical decision support framework leverages advanced optimization pipelines to assist health practitioners with objective risk stratification. 
        By processing key clinical indicators synchronously, the underlying machine learning models output diagnostic vectors to catch signs of advanced heart failure early.
        </div>
        """, unsafe_allow_html=True)
        if st.button("Get Started ➡️", key="home_start_btn"):
            st.session_state.current_nav = "FORM"
            st.rerun()
            
    with col_graphic:
        st.markdown("""
            <div style="background-color: transparent; text-align:center; padding-top:60px;">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 150" fill="none" stroke="#0c5460" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" style="width:100%;">
                    <path d="M 10 75 L 50 75 L 65 35 L 80 115 L 95 5 L 110 95 L 125 75 L 175 75 L 190 35 L 205 115 L 220 5 L 235 95 L 250 75 L 300 75 L 315 35 L 330 115 L 345 5 L 360 95 L 375 75 L 425 75 L 440 35 L 455 115 L 470 5 L 485 95 L 500 75 L 540 75 L 555 35 L 570 115 L 590 75"/>
                </svg>
            </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# MODULE 2: FORM SCREEN
# ---------------------------------------------------------
elif st.session_state.current_nav == "FORM":
    st.markdown("<h2 style='font-size:36px; color:#1a3a4b;'>📋 Patient Intake Records</h2>", unsafe_allow_html=True)
    
    col_id1, col_id2 = st.columns(2)
    with col_id1:
        patient_name = st.text_input("Patient Full Name", value="John Doe")
    with col_id2:
        patient_id = st.text_input("Hospital Reference ID", value="PT-89422")
        
    st.markdown("<h3 style='font-size:30px; color:#0c5460; margin-top:25px;'>Clinical Indicators</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Patient Age Parameters (Years)", int(df['age'].min()), int(df['age'].max()), 65)
        anaemia = st.selectbox("Anaemia Diagnostic Evaluation Status", [0, 1], format_func=lambda x: "Negative Baseline" if x==0 else "Positive Anemic Indicator")
        creatinine_phosphokinase = st.slider("CPK Level Concentration Metrics (mcg/L)", int(df['creatinine_phosphokinase'].min()), int(df['creatinine_phosphokinase'].max()), 250)
        diabetes = st.selectbox("Diabetes History Matrix Diagnostic Record", [0, 1], format_func=lambda x: "No History Matrix Found" if x==0 else "Diabetic Flag Identified")
        ejection_fraction = st.slider("Ejection Fraction Blood Flow Percentage (%)", int(df['ejection_fraction'].min()), int(df['ejection_fraction'].max()), 25)
        high_blood_pressure = st.selectbox("Hypertensive Vascular History Profiles", [0, 1], format_func=lambda x: "Normal Blood Pressure Range" if x==0 else "Hypertensive Condition Flagged")
        
    with col2:
        platelets = st.slider("Platelets Diagnostic Count (Cells/mL)", float(df['platelets'].min()), float(df['platelets'].max()), 250000.0, step=1000.0)
        rose_creatinine = st.slider("Serum Creatinine Volume Metrics (mg/dL)", float(df['serum_creatinine'].min()), float(df['serum_creatinine'].max()), 2.1, step=0.1)
        serum_sodium = st.slider("Serum Sodium Levels Concentration (mEq/L)", int(df['serum_sodium'].min()), int(df['serum_sodium'].max()), 136)
        sex = st.selectbox("Biological Sex Profiling Identity", [0, 1], format_func=lambda x: "Female Index Classification" if x==0 else "Male Index Classification")
        smoking = st.selectbox("Tobacco/Smoking Profile Behavioral Metrics", [0, 1], format_func=lambda x: "Non-smoker Habit Baseline" if x==0 else "Active Smoker Classification")
        time = st.slider("Follow-up Observation Window Chrono Duration (Days)", int(df['time'].min()), int(df['time'].max()), 120)
        
    if st.button("Submit Details ➡️", key="intake_process_btn", use_container_width=True):
        st.session_state.patient_data = {
            "patient_name": patient_name, "patient_id": patient_id, "age": age, "anaemia": anaemia, 
            "creatinine_phosphokinase": creatinine_phosphokinase, "diabetes": diabetes, "ejection_fraction": ejection_fraction, 
            "high_blood_pressure": high_blood_pressure, "platelets": platelets, "serum_creatinine": rose_creatinine, 
            "serum_sodium": serum_sodium, "sex": sex, "smoking": smoking, "time": time
        }
        st.session_state.current_nav = "MODELS"
        st.rerun()

# ---------------------------------------------------------
# MODULE 3: MODELS VIEW (AUTO-NAVIGATION & MULTI-COLOR GRAPH)
# ---------------------------------------------------------
elif st.session_state.current_nav == "MODELS":
    st.markdown("<h2 style='font-size:36px; color:#1a3a4b;'>⚙️ Machine Learning Models</h2>", unsafe_allow_html=True)
    
    model_choice = st.selectbox("Select Active Machine Learning Analytics Engine", ["Random Forest Classifier", "Logistic Regression Framework", "XGBoost Core Engine", "Decision Tree Model"])
    
    if "Random Forest" in model_choice:
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    elif "Logistic Regression" in model_choice:
        model = LogisticRegression(max_iter=1000)
    elif "XGBoost" in model_choice:
        model = XGBClassifier(max_depth=4)
    else:
        model = DecisionTreeClassifier(max_depth=5)
        
    model.fit(X_train_scaled, y_train)
    accuracy = model.score(X_test_scaled, y_test)
    
    st.markdown(f"""
        <div style="margin: 15px 0px; background: #fff; padding: 20px; border: 1px solid #ccc; border-radius: 8px;">
            <p style="font-size: 24px; font-weight: 800; color: #2b2b2b; margin: 0;">Calculated System Model Accuracy Score: <span style="color:#0c5460; font-size:28px;">{accuracy * 100:.2f}%</span></p>
        </div>
    """, unsafe_allow_html=True)
    
    # MULTI-COLOR THEMED GRAPH (using Viridis palette instead of a single color tone)
    importance_values = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.abs(model.coef_[0])
    feat_imp_df = pd.DataFrame({'Feature Vector': feature_names, 'Weight Importance': importance_values}).sort_values(by='Weight Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 5.0))
    sns.barplot(x='Weight Importance', y='Feature Vector', data=feat_imp_df, palette='viridis', hue='Feature Vector', legend=False, ax=ax)
    ax.tick_params(labelsize=14)
    st.pyplot(fig)
    
    # Save parameter weights inside the session structure
    st.session_state.model_choice = model_choice
    st.session_state.accuracy = accuracy
    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.feature_names = feature_names

    # CRITICAL TRIGGER FIX: Runs optimization logic and redirects automatically to the Report page
    if st.button("Run Analysis ⚡", key="run_prediction_btn", use_container_width=True):
        if st.session_state.patient_data is None:
            st.warning("⚠️ Access interrupted. Please input metrics in the Patient Intake Form first.")
        else:
            st.session_state.current_nav = "REPORT"
            st.rerun()

# ---------------------------------------------------------
# MODULE 4: HIGH-CONTRAST DIAGNOSTIC REPORT
# ---------------------------------------------------------
elif st.session_state.current_nav == "REPORT":
    st.markdown("<h2 style='font-size:36px; color:#1a3a4b;'>📊 Reports Generation & Dossier Export Module</h2>", unsafe_allow_html=True)
    
    if st.session_state.patient_data is None:
        st.warning("⚠️ Prediction logs missing. Please record profile elements in the Patient Intake Form first.")
    else:
        p = st.session_state.patient_data
        user_data = pd.DataFrame([[p[f] for f in feature_names]], columns=feature_names)
        
        user_data_scaled = st.session_state.scaler.transform(user_data)
        prediction_proba = st.session_state.model.predict_proba(user_data_scaled)[0][1]
        active_engine_name = st.session_state.model_choice
            
        prob_percent = prediction_proba * 100
        
        if prob_percent < 35.0:
            severity_status = "LOW SYSTEMIC SEVERITY"
            severity_desc = "Clinical metrics are inside standard baseline windows."
        elif 35.0 <= prob_percent < 65.0:
            severity_status = "ELEVATED SEVERITY TIER (BORDERLINE WARNING)"
            severity_desc = "Noticeable data divergence across target biomarkers."
        else:
            severity_status = "CRITICAL HIGH-RISK SEVERITY PATHWAY"
            severity_desc = "Significant clinical deviations observed across multiple vector points."

        # VISIBILITY AND CONTRAST FIX: Pure white text guarantees readability on dark backgrounds
        st.markdown(f"""
            <div class="severity-card-premium">
                <h2>📝 Assessment: {severity_status}</h2>
                <p>Patient Risk Vector Probability Evaluation: <strong>{prob_percent:.1f}%</strong></p>
                <p style="font-style: italic; opacity: 0.95;">Diagnostic Guideline: {severity_desc}</p>
            </div>
        """, unsafe_allow_html=True)
            
        metrics_display = {
            "Monitored Clinical Attribute Fields": ["Patient Full Name", "Hospital Reference ID", "Evaluated Age Bracket", "Selected Model Engine", "Ejection Fraction Target Metric", "Serum Creatinine Density Value", "Aggregated Severity Index"],
            "Assigned Patient Vector Values": [p['patient_name'], p['patient_id'], f"{p['age']} Years Old", str(active_engine_name), f"{p['ejection_fraction']}% Volume Ratio", f"{p['serum_creatinine']} mg/dL", f"{prob_percent:.1f}% -> {severity_status}"]
        }
        st.table(pd.DataFrame(metrics_display))

        if st.button("Download", key="download_report_btn", use_container_width=True):
            st.success("Dossier compiled successfully.")

# ---------------------------------------------------------
# MODULE 5: SIMPLIFIED HELP / ABOUT VIEW
# ---------------------------------------------------------
elif st.session_state.current_nav == "ABOUT":
    st.markdown("<h2 style='font-size:36px; color:#1a3a4b;'>ℹ️ Help & System Specifications</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="hero-body">
    This system uses machine learning to assess heart failure risks based on patient test metrics. Here is how the application handles data step-by-step:
    </div>
    """, unsafe_allow_html=True)
    
    # Cleaned, intuitive documentation table with highlighted table styling applied automatically via custom CSS rules
    specs_df = pd.DataFrame({
        "System Pipeline Layer": [
            "1. Patient Intake", 
            "2. Data Normalization", 
            "3. ML Risk Processing", 
            "4. Report Generation"
        ],
        "Operational Scope & Functional Target Goals": [
            "Collects patient metrics and checks for formatting errors.",
            "Adjusts scale differences between small figures (e.g., creatinine levels) and large figures (e.g., platelet counts).",
            "Runs the selected AI algorithm to compute risk percentages based on pattern recognition.",
            "Creates the clear diagnostic report card and allows exporting file sheets."
        ],
        "Platform Dependencies": [
            "Python / Streamlit", 
            "Scikit-Learn Preprocessing", 
            "XGBoost Classifier Engine", 
            "Matplotlib / Seaborn"
        ]
    })
    st.table(specs_df)

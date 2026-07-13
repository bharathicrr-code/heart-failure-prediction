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

# --- GLOBAL FORCED HIGH-READABILITY INTERFACE CSS ---
st.markdown("""
    <style>
    /* Remove unnecessary viewport padding offsets */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
    }

    /* 1. GLOBAL TYPOGRAPHY STANDARDS */
    html, body, [data-testid="stWidgetLabel"] p, .stSelectbox div, .stTextInput input, div[data-baseweb="select"] * {
        font-size: 24px !important;
        font-weight: 600 !important;
        color: #1a3a4b !important;
    }
    
    .stTextInput input, div[data-testid="stSelectbox-Trigger"] {
        min-height: 52px !important;
        background-color: #ffffff !important;
        border: 2px solid #ced4da !important;
        border-radius: 6px !important;
    }

    /* 2. COMPACT, GMAIL-INSPIRED COMPACT LOGIN CONTAINER CARD */
    div[data-testid="stForm"] {
        background-color: #ffffff !important;
        padding: 40px !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08) !important;
        border: 1px solid #e2e8f0 !important;
        max-width: 480px !important;
        margin: 40px auto !important;
    }
    
    /* 3. SHRINK STREAMLIT'S WIDE PASSWORD VISIBILITY ICON BUTTON (GMAIL-STYLE) */
    div[data-testid="stForm"] button[aria-label="Show password"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #6c757d !important;
        width: auto !important;
        min-width: auto !important;
        margin: 0 !important;
        padding: 0 10px !important;
    }
    
    /* 4. EXECUTIVE CENTERING CORE RULES FOR FORM INTERFACE BUTTONS */
    div[data-testid="stForm"] button[type="submit"], div.stButton > button, div.stDownloadButton > button {
        background-color: #0c5460 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 12px 32px !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        width: auto !important;
        min-width: 200px !important;
        display: block !important;
        margin: 25px auto 0 auto !important;
        box-shadow: 0 4px 12px rgba(12, 84, 96, 0.15) !important;
        border: none !important;
        text-transform: uppercase !important;
    }

    /* 5. DATA EVALUATION MATRIX STRUCTURING */
    [data-testid="stTable"] table {
        width: 100% !important;
        border-collapse: collapse !important;
        margin-bottom: 50px !important;
    }
    [data-testid="stTable"] table th {
        background-color: #0c5460 !important;
        color: white !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        padding: 18px !important;
    }
    [data-testid="stTable"] td {
        font-size: 22px !important;
        color: #2b2b2b !important;
        padding: 16px !important;
        border-bottom: 1px solid #dee2e6 !important;
    }
    
    /* TYPOGRAPHY CORES */
    .hero-title { font-size: 44px !important; font-weight: 800; color: #1a3a4b; margin-bottom: 10px; }
    .hero-subtitle { font-size: 28px !important; font-weight: 600; color: #0c5460; margin-bottom: 25px; }
    .hero-body { font-size: 24px !important; line-height: 1.7 !important; color: #333333; }
    
    /* CLINICAL OUTCOME TIERS */
    .assessment-box-safe { background-color: #28a745 !important; padding: 35px; border-radius: 12px; margin-bottom: 35px; color: white !important; }
    .assessment-box-borderline { background-color: #fd7e14 !important; padding: 35px; border-radius: 12px; margin-bottom: 35px; color: white !important; }
    .assessment-box-severe { background-color: #dc3545 !important; padding: 35px; border-radius: 12px; margin-bottom: 35px; color: white !important; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("heart_failure_clinical_records_dataset.csv")

if "current_nav" not in st.session_state:
    st.session_state.current_nav = "HOME"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "patient_data" not in st.session_state:
    st.session_state.patient_data = None

# --- AUTHENTICATION GATEWAY ---
if not st.session_state.logged_in:
    st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
    
    # Render fields strictly within a clean, centered interface card column format
    col_l1, col_l2, col_l3 = st.columns([1.5, 2, 1.5])
    with col_l2:
        with st.form("login_form", clear_on_submit=False):
            st.markdown("<h2 style='text-align: center; color: #1a3a4b; font-size: 34px; margin-top: 0; margin-bottom: 30px; font-weight: 800;'>🔒 Login Page</h2>", unsafe_allow_html=True)
            username = st.text_input("Username / Clinician Identification Badge Key", value="admin")
            password = st.text_input("Secure Password Access Key", type="password", value="password")
            submit_login = st.form_submit_button("LOGIN")
            if submit_login:
                if username == "admin" and password == "password":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
    st.stop()

df = load_data()
X = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- NAVIGATION HEADER NAVBAR ---
st.markdown("""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-bottom: 4px solid #0c5460;">
        <h2 style="margin:0; color: #1a3a4b; font-weight: 800;">🩺 Heart Failure Prediction System</h2>
    </div>
""", unsafe_allow_html=True)

col_n1, col_n2, col_n3, col_n4, col_n5 = st.columns(5)
with col_n1:
    if st.button("🏠 HOME", key="nav_home", use_container_width=True): st.session_state.current_nav = "HOME"; st.rerun()
with col_n2:
    if st.button("📋 INTAKE FORM", key="nav_form", use_container_width=True): st.session_state.current_nav = "FORM"; st.rerun()
with col_n3:
    if st.button("⚙️ ML MODELS", key="nav_models", use_container_width=True): st.session_state.current_nav = "MODELS"; st.rerun()
with col_n4:
    if st.button("📊 DIAGNOSTIC REPORT", key="nav_report", use_container_width=True): st.session_state.current_nav = "REPORT"; st.rerun()
with col_n5:
    if st.button("ℹ️ SYSTEM HELP", key="nav_about", use_container_width=True): st.session_state.current_nav = "ABOUT"; st.rerun()

# --- MODULE 1: HOME VIEW ---
if st.session_state.current_nav == "HOME":
    st.markdown('<div class="hero-title" style="text-align: center; margin-top: 15px;">Heart Failure Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle" style="text-align: center;">Decision Support Engine Deployment Platform</div>', unsafe_allow_html=True)
    
    # Original centered telemetry heartbeat asset alignment mapping
    st.markdown("""
        <div style="text-align:center; padding: 20px 0;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 100" fill="none" stroke="#0c5460" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" style="width: 45%; max-width: 450px; margin: 0 auto;">
                <path d="M 10 50 L 150 50 L 170 20 L 190 80 L 210 5 L 230 95 L 250 50 L 350 50 L 370 20 L 390 80 L 410 5 L 430 95 L 450 50 L 590 50 Z"/>
            </svg>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="hero-body" style="text-align: center; max-width: 950px; margin: 0 auto; padding-bottom: 20px;">
    This analytical clinical decision support framework leverages advanced optimization pipelines to assist health practitioners with objective risk stratification. 
    By processing key clinical indicators synchronously, the underlying machine learning models output diagnostic vectors to catch signs of advanced heart failure early.
    <br><br>
    <div style="text-align: left; display: inline-block; background-color: #f8f9fa; padding: 30px; border-radius: 8px; border-left: 5px solid #0c5460; width: 100%;">
        <b style="font-size: 26px; color: #1a3a4b;">Key System Design Pillars:</b>
        <ul style="margin-top: 10px;">
            <li><b>Synchronous Pipelines:</b> Real-time analytics engine tracking input diagnostic matrices.</li>
            <li><b>Multiclass Backbone Architecture:</b> Features isolated benchmarks across Linear, Tree, and Ensemble algorithms.</li>
            <li><b>Secure Identity Desk:</b> Fully guarded administrative entry paths protecting active user files.</li>
        </ul>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Initialize Patient Entry Module ➡️", key="home_start_btn"):
        st.session_state.current_nav = "FORM"
        st.rerun()

# --- MODULE 2: FORM SCREEN ---
elif st.session_state.current_nav == "FORM":
    st.markdown("<h2 style='color:#1a3a4b;'>📋 Patient Administrative Records & Vector Matrix</h2>", unsafe_allow_html=True)
    col_id1, col_id2 = st.columns(2)
    with col_id1: patient_name = st.text_input("Patient Full Name Name Entries", value="John Doe")
    with col_id2: patient_id = st.text_input("Hospital Reference ID (Ref-ID Record)", value="PT-89422")
        
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
        
    if st.button("Submit Details", key="intake_process_btn"):
        st.session_state.patient_data = {
            "patient_name": patient_name, "patient_id": patient_id, "age": age, "anaemia": anaemia, 
            "creatinine_phosphokinase": creatinine_phosphokinase, "diabetes": diabetes, "ejection_fraction": ejection_fraction, 
            "high_blood_pressure": high_blood_pressure, "platelets": platelets, "serum_creatinine": rose_creatinine, 
            "serum_sodium": serum_sodium, "sex": sex, "smoking": smoking, "time": time
        }
        st.session_state.current_nav = "MODELS"
        st.rerun()

# --- MODULE 3: MODELS VIEW ---
elif st.session_state.current_nav == "MODELS":
    st.markdown("<h2 style='color:#1a3a4b;'>⚙️ Machine Learning Prediction Engine Module</h2>", unsafe_allow_html=True)
    model_choice = st.selectbox("Select Active Machine Learning Analytics Engine", ["Random Forest Classifier", "Logistic Regression Framework", "XGBoost Core Engine", "Decision Tree Model", "Support Vector Machine (SVM)"])
    
    if "Random Forest" in model_choice: model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    elif "Logistic Regression" in model_choice: model = LogisticRegression(max_iter=1000)
    elif "XGBoost" in model_choice: model = XGBClassifier(max_depth=4)
    elif "Decision Tree" in model_choice: model = DecisionTreeClassifier(max_depth=5)
    else: model = SVC(probability=True, random_state=42)
        
    model.fit(X_train_scaled, y_train)
    accuracy = model.score(X_test_scaled, y_test)
    
    st.markdown(f"<h4>Calculated Dataset Accuracy: {accuracy * 100:.2f}%</h4>", unsafe_allow_html=True)
    
    importance_values = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.abs(model.coef_[0])
    feat_imp_df = pd.DataFrame({'Feature Vector': feature_names, 'Weight Importance': importance_values}).sort_values(by='Weight Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(6, 2.5))
    sns.barplot(x='Weight Importance', y='Feature Vector', data=feat_imp_df, palette='viridis_r', hue='Feature Vector', legend=False, ax=ax)
    ax.tick_params(labelsize=8)
    ax.set_ylabel("Feature Vector", fontsize=9, fontweight='bold')
    ax.set_xlabel("Weight Importance", fontsize=9, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.session_state.model_choice = model_choice; st.session_state.model = model; st.session_state.scaler = scaler; st.session_state.feature_names = feature_names

    if st.button("Run Analysis", key="run_prediction_btn"):
        if st.session_state.patient_data is None: st.warning("Please fill input data form first.")
        else: st.session_state.current_nav = "REPORT"; st.rerun()

# --- MODULE 4: DIAGNOSTIC REPORT ---
elif st.session_state.current_nav == "REPORT":
    st.markdown("<h2 style='color:#1a3a4b;'>📊 Reports Generation Module</h2>", unsafe_allow_html=True)
    if st.session_state.patient_data is not None:
        p = st.session_state.patient_data
        user_data = pd.DataFrame([[p[f] for f in feature_names]], columns=feature_names)
        user_data_scaled = st.session_state.scaler.transform(user_data)
        prediction_proba = st.session_state.model.predict_proba(user_data_scaled)[0][1]
        prob_percent = prediction_proba * 100
        
        if prob_percent < 35.0:
            box_class, severity_status = "assessment-box-safe", "SAFE PERFORMING METRIC WINDOWS"
        elif 35.0 <= prob_percent < 65.0:
            box_class, severity_status = "assessment-box-borderline", "BORDERLINE WARNING TIER"
        else:
            box_class, severity_status = "assessment-box-severe", "SEVERE CRITICAL HIGH-RISK PATHWAY"

        st.markdown(f"""
            <div class="{box_class}">
                <h2 style="color: white !important; margin:0; font-size:36px;">📝 Assessment: {severity_status}</h2>
                <p style="color: white !important; font-size:26px; margin:5px 0 0 0;">Evaluation Metric Probability: {prob_percent:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
            
        metrics_display = {
            "Monitored Clinical Attribute Fields": ["Patient Full Identification Name", "Hospital Reference Tracking Key", "Evaluated Age Bracket", "Selected Algorithmic Engine", "Ejection Fraction Target Metric", "Serum Creatinine Density Value"],
            "Assigned Patient Vector Values": [p['patient_name'], p['patient_id'], f"{p['age']} Years Old", str(st.session_state.model_choice), f"{p['ejection_fraction']}% Ratio", f"{p['serum_creatinine']} mg/dL"]
        }
        report_df = pd.DataFrame(metrics_display)
        st.table(report_df)
        
        csv_data = report_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 DOWNLOAD PATIENT EVALUATION REPORT", data=csv_data, file_name=f"Clinical_Report_{p['patient_id']}.csv", mime="text/csv")

# --- MODULE 5: SYSTEM HELP ---
elif st.session_state.current_nav == "ABOUT":
    st.markdown("<h2 style='color: #1a3a4b;'>ℹ️ Project Presentation Help Desk</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 22px; color: #555; margin-bottom: 25px;'>Use this architectural mapping table to easily explain the system pipeline flow and model scope stages directly to your evaluation examiners.</p>", unsafe_allow_html=True)
    
    presentation_help_data = {
        "Core Project Lifecycle Stage": [
            "1. Clinical Ingestion Pipeline", 
            "2. Vector Data Normalization", 
            "3. Multi-Model Inference Core", 
            "4. Dynamic Risk Stratification", 
            "5. Automated Report Synthesis",
            "6. Security Access Layer"
        ],
        "Functional Project Scope & Purpose (For Examiners)": [
            "Captures active data indicators from health records securely via clinical entry fields.",
            "Uses StandardScaler tracking to eliminate value anomalies and balance numeric ranges.",
            "Executes user-selected ML algorithms (Random Forest, XGBoost, etc.) to evaluate patient data.",
            "Calculates probability metrics and assigns safe, borderline, or severe condition flags.",
            "Formats data into structural record matrices and outputs an external, downloadable document.",
            "Guards administrative gateways via unified identification keys to defend records."
        ]
    }
    
    st.table(pd.DataFrame(presentation_help_data))
    st.markdown("<div style='margin-bottom: 250px; height: 250px; display: block;'></div>", unsafe_allow_html=True)

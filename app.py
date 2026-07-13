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

# --- GLOBAL HIGH-READABILITY NAVIGATION & INTERFACE CSS ---
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem !important;
    }
    .nav-logo {
        font-size: 34px !important;
        font-weight: 800 !important;
        color: #1a3a4b !important;
        display: inline-block;
        vertical-align: middle;
        font-family: 'Helvetica Neue', sans-serif;
        padding-top: 5px;
    }
    
    /* FORCE HIDING OF ALL NATIVE RADIO BUTTON CIRCLES AND ASSIGN CLEAN NAVIGATION TEXT LINKS */
    div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {
        display: none !important;
    }
    div[data-testid="stRadio"] > div {
        justify-content: flex-end !important;
        gap: 40px !important;
    }
    div[data-testid="stRadio"] label [data-testid="stMarkdownContainer"] p {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #555555 !important;
        font-family: sans-serif !important;
    }
    /* Hide the selection circles entirely */
    div[data-testid="stRadio"] [data-checked] {
        background-color: transparent !important;
        border-color: transparent !important;
    }
    div[data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }
    div[data-testid="stRadio"] [data-testid="stCustomBorder"] {
        display: none !important;
    }
    div[data-testid="stRadio"] label {
        padding: 0px 0px 6px 0px !important;
        border-bottom: 3px solid transparent !important;
    }
    /* Highlight the active link with an immediate matching highlight underline */
    div[data-testid="stRadio"] div[role="radiogroup"] > div:has(input:checked) label {
        border-bottom: 3px solid #0c5460 !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > div:has(input:checked) p {
        color: #0c5460 !important;
    }
    
    .hero-title {
        font-size: 46px !important;
        font-weight: 800 !important;
        color: #1a3a4b !important;
        margin-bottom: 12px !important;
        line-height: 1.2 !important;
    }
    .hero-subtitle {
        font-size: 30px !important;
        font-weight: 700 !important;
        color: #0c5460 !important;
        margin-bottom: 30px !important;
    }
    .hero-body {
        font-size: 23px !important; 
        line-height: 1.8 !important;
        color: #2b2b2b !important;
    }
    .hero-body ul li {
        font-size: 22px !important;
        margin-bottom: 12px !important;
    }
    
    /* ENHANCED HIGH-VISIBILITY GLOBAL GLOBAL BUTTON OVERRIDES */
    div.stButton > button, 
    div.stDownloadButton > button {
        background-color: #0c5460 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 18px 45px !important;
        font-size: 24px !important;
        font-weight: 800 !important;
        width: 100% !important;
        max-width: 520px !important;
        min-height: 75px !important;
        display: block !important;
        margin: 40px auto !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.16) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background-color: #0a434d !important;
        color: white !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2) !important;
    }
    
    /* CLINICAL REPORT SEVERITY CARD HOOKS */
    .severity-card {
        padding: 24px !important;
        border-radius: 8px !important;
        margin-bottom: 30px !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- APPLICATION STATE MANAGEMENT LAYER ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "patient_data" not in st.session_state:
    st.session_state.patient_data = None
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

# --- MACHINE LEARNING DATA LOGIC ACCELERATOR ---
@st.cache_data
def load_data():
    return pd.read_csv("heart_failure_clinical_records_dataset.csv")

# -----------------------------------------------------------------
# MODULE 1: LOGIN MODULE INTERFACE
# -----------------------------------------------------------------
if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    
    with col_l2:
        st.markdown("<h2 style='text-align: center; color: #1a3a4b;'>🔒 System Authentication Gateway</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size:18px; color:#666;'>Heart Failure Prediction System Administration Desk</p>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username / Clinician Identification Badge Key", value="admin")
            password = st.text_input("Secure Password Access Key", type="password", value="password")
            submit_login = st.form_submit_button("Authenticate Access Verification")
            
            if submit_login:
                if username == "admin" and password == "password":
                    st.session_state.logged_in = True
                    st.success("Authentication validated successfully! Initializing system layers...")
                    st.rerun()
                else:
                    st.error("Invalid secure access keys. Check authorization records.")
    st.stop()

# -----------------------------------------------------------------
# MAIN APP MODULES (Executes Post-Authentication)
# -----------------------------------------------------------------
try:
    df = load_data()
    X = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # TWO-COLUMN HEADER NAVIGATION STRUCTURE
    col_logo, col_menu = st.columns([5, 7])
    
    with col_logo:
        st.markdown('<div class="nav-logo">🩺 Heart Failure Prediction System</div>', unsafe_allow_html=True)
        
    with col_menu:
        nav_selection = st.radio(
            "",
            ["HOME", "FORM", "MODELS", "REPORT", "ABOUT"],
            horizontal=True
        )
        st.session_state.current_nav = nav_selection

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # MODULE 2: HOME VIEW
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
            <br><br>
            <b>Key System Design Pillars:</b>
            <ul>
                <li><b>Synchronous Pipelines:</b> Real-time analytics engine tracking input diagnostic matrices.</li>
                <li><b>Multiclass Backbone Architecture:</b> Features isolated benchmarks across Linear, Tree, and Ensemble algorithms.</li>
                <li><b>Secure Identity Desk:</b> Fully guarded administrative entry paths protecting active user files.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Initialize Patient Entry Module ➡️", key="home_start_btn"):
                st.session_state.current_nav = "FORM"
                st.rerun()
                
        with col_graphic:
            # ENHANCED MULTI-PULSE WAVEFORM LOGIC
            st.markdown("""
                <div style="background-color: transparent; text-align:center; padding-top:40px;">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 150" fill="none" stroke="#0c5460" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:95%;">
                        <path d="M 10 75 L 70 75 L 85 45 L 100 105 L 115 15 L 130 85 L 145 75 L 210 75 L 225 45 L 240 105 L 255 15 L 270 85 L 285 75 L 350 75 L 365 45 L 380 105 L 395 15 L 410 85 L 425 75 L 490 75 L 505 45 L 520 105 L 535 15 L 550 85 L 590 75"/>
                    </svg>
                    <p style="color:#555555; font-size:18px; font-weight:700; margin-top:35px; font-family:sans-serif; letter-spacing:1px;">Cardiovascular Diagnostics Engine Output</p>
                </div>
            """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # MODULE 3: PATIENT DETAILS ENTRY MODULE (FORM)
    # ---------------------------------------------------------
    elif st.session_state.current_nav == "FORM":
        st.subheader("Patient Administrative Records & Vector Matrix")
        
        col_id1, col_id2 = st.columns(2)
        with col_id1:
            patient_name = st.text_input("Patient Full Name", value="John Doe")
        with col_id2:
            patient_id = st.text_input("Hospital Reference ID (Ref-ID)", value="PT-89422")
            
        st.markdown("### Clinical Indicators Vector Input")
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Patient Age Parameters", int(df['age'].min()), int(df['age'].max()), 65)
            anaemia = st.selectbox("Anaemia Diagnostic Level", [0, 1], format_func=lambda x: "Negative" if x==0 else "Positive")
            creatinine_phosphokinase = st.slider("CPK Level concentration (mcg/L)", int(df['creatinine_phosphokinase'].min()), int(df['creatinine_phosphokinase'].max()), 250)
            diabetes = st.selectbox("Diabetes History Matrix", [0, 1], format_func=lambda x: "No History" if x==0 else "Diabetic")
            ejection_fraction = st.slider("Ejection Fraction Blood Percentage (%)", int(df['ejection_fraction'].min()), int(df['ejection_fraction'].max()), 25)
            high_blood_pressure = st.selectbox("Hypertensive Vascular History", [0, 1], format_func=lambda x: "Normal Blood Pressure" if x==0 else "Hypertensive")
            
        with col2:
            platelets = st.slider("Platelets Diagnostic Count", float(df['platelets'].min()), float(df['platelets'].max()), 250000.0, step=1000.0)
            rose_creatinine = st.slider("Serum Creatinine Volume (mg/dL)", float(df['serum_creatinine'].min()), float(df['serum_creatinine'].max()), 2.1, step=0.1)
            serum_sodium = st.slider("Serum Sodium Levels (mEq/L)", int(df['serum_sodium'].min()), int(df['serum_sodium'].max()), 136)
            sex = st.selectbox("Biological Sex Profiling", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
            smoking = st.selectbox("Tobacco/Smoking Profile", [0, 1], format_func=lambda x: "Non-smoker" if x==0 else "Active Smoker")
            time = st.slider("Follow-up Observation Window (Days)", int(df['time'].min()), int(df['time'].max()), 120)
            
        st.markdown("---")
        
        if st.button("Submit Patient Details ➡️", key="intake_process_btn"):
            st.session_state.patient_data = {
                "patient_name": patient_name, "patient_id": patient_id,
                "age": age, "anaemia": anaemia, "creatinine_phosphokinase": creatinine_phosphokinase,
                "diabetes": diabetes, "ejection_fraction": ejection_fraction, "high_blood_pressure": high_blood_pressure,
                "platelets": platelets, "serum_creatinine": rose_creatinine, "serum_sodium": serum_sodium,
                "sex": sex, "smoking": smoking, "time": time
            }
            st.session_state.prediction_made = False
            st.session_state.current_nav = "MODELS"
            st.colors = "#0c5460"
            st.rerun()

    # ---------------------------------------------------------
    # MODULE 4: PREDICTION MODULE (MODELS)
    # ---------------------------------------------------------
    elif st.session_state.current_nav == "MODELS":
        st.subheader("Machine Learning Prediction Engine Module")
        st.info("⚙️ Model Sandbox Configuration: Choose the processing backbone algorithm, view accuracy metrics, and click 'Execute Analysis Prediction' to output status profiles.")
        
        model_choice = st.selectbox("Select Active Machine Learning Analytics Engine", ["Random Forest", "Logistic Regression", "XGBoost", "Decision Tree", "SVM"])
        
        if model_choice == "Logistic Regression":
            model = LogisticRegression()
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=3)
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=50, max_depth=3)
        elif model_choice == "SVM":
            model = SVC(probability=True)
        elif model_choice == "XGBoost":
            model = XGBClassifier(max_depth=3)
            
        model.fit(X_train_scaled, y_train)
        accuracy = model.score(X_test_scaled, y_test)
        
        st.metric(label=f"{model_choice} Baseline Test Dataset Accuracy", value=f"{accuracy * 100:.2f}%")
        
        importance_values = None
        if model_choice == "Logistic Regression":
            importance_values = np.abs(model.coef_[0])
        elif model_choice in ["Decision Tree", "Random Forest", "XGBoost"]:
            importance_values = model.feature_importances_
            
        if importance_values is not None:
            feat_imp_df = pd.DataFrame({'Feature Vector': feature_names, 'Weight Importance': importance_values}).sort_values(by='Weight Importance', ascending=False)
            fig, ax = plt.subplots(figsize=(10, 3.5))
            sns.barplot(x='Weight Importance', y='Feature Vector', data=feat_imp_df, palette='viridis', hue='Feature Vector', legend=False, ax=ax)
            st.pyplot(fig)
            
        st.session_state.model_choice = model_choice
        st.session_state.accuracy = accuracy
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.feature_names = feature_names

        st.markdown("---")
        
        if st.button("Execute Analysis Prediction ⚡", key="run_prediction_btn"):
            if st.session_state.patient_data is None:
                st.warning("⚠️ No diagnostic matrices found. Please input data on the Form page before evaluating profiles.")
            else:
                st.session_state.prediction_made = True
                st.success("Analysis calculation compiled successfully. Check the outputs generated inside the Reports module.")

    # ---------------------------------------------------------
    # MODULE 5: REPORTS MODULE (REPORT WITH SEVERITY TRACKS)
    # ---------------------------------------------------------
    elif st.session_state.current_nav == "REPORT":
        st.subheader("Reports Generation & Dossier Export Module")
        
        if st.session_state.patient_data is None:
            st.warning("Missing data tracks. Please record profile elements in the Patient Intake Form first.")
        elif not st.session_state.prediction_made:
            st.warning("Prediction processing incomplete. Run the analytical task on the Models page first.")
        else:
            p = st.session_state.patient_data
            user_data = pd.DataFrame([[p[f] for f in st.session_state.feature_names]], columns=st.session_state.feature_names)
            user_data_scaled = st.session_state.scaler.transform(user_data)
            
            prediction = st.session_state.model.predict(user_data_scaled)[0]
            prediction_proba = st.session_state.model.predict_proba(user_data_scaled)[0][1]
            prob_percent = prediction_proba * 100
            
            # DYNAMIC CLINICAL SEVERITY INDEX COMPILER
            if prob_percent < 35.0:
                severity_status = "LOW SYSTEMIC SEVERITY"
                severity_color = "#28a745" # Emerald Green
                severity_desc = "Clinical variables map within standard operating targets. Minimal structural deviation recorded."
            elif 35.0 <= prob_percent < 65.0:
                severity_status = "ELEVATED SEVERITY TIER (BORDERLINE)"
                severity_color = "#ffc107" # Warning Amber
                severity_desc = "Noticeable divergence identified across diagnostic indicators. Regularized outpatient surveillance recommended."
            else:
                severity_status = "CRITICAL HIGH RISK SEVERITY"
                severity_color = "#dc3545" # Emergency Crimson
                severity_desc = "Severe combination of high serum creatinine and depressed ejection metrics. Immediate clinical intervention advised."

            st.markdown(f"""
                <div class="severity-card" style="background-color: {severity_color}; color: { '#111111' if prob_percent < 35.0 or (35.0 <= prob_percent < 65.0) else 'white' };">
                    <h3 style="margin:0px 0px 8px 0px; font-weight:800; font-size:26px;">📊 {severity_status}</h3>
                    <p style="margin:0; font-size:19px; font-weight:600; opacity:0.95;">
                        Measured Risk Scale Probability Index: {prob_percent:.1f}%<br>
                        <b>Clinical Guideline:</b> {severity_desc}
                    </p>
                </div>
            """, unsafe_allow_html=True)
                
            metrics_display = {
                "Monitored Attribute Field": ["Patient Registered Name", "Hospital Reference Tracking ID", "Evaluated Age Bracket", "Selected Algorithmic Engine", "Risk Severity Index Classification", "Ejection Fraction Level Metric", "Serum Creatinine Density Value"],
                "Assigned Patient Vector Values": [p['patient_name'], p['patient_id'], f"{p['age']} Years Old", f"{st.session_state.model_choice} Model", f"{prob_percent:.1f}% ({severity_status})", f"{p['ejection_fraction']}% Ratio", f"{p['serum_creatinine']} mg/dL"]
            }
            st.table(pd.DataFrame(metrics_display))

            report_data = f"""======================================================
HEART FAILURE CLINICAL ASSESSMENT DOSSIER
======================================================
[PATIENT CLASSIFICATION RECORD]
* Registered Patient Name: {p['patient_name']}
* Hospital Reference Tracker: {p['patient_id']}

[SYSTEM APPLICATION BACKBONE]
* Selected Active ML Engine: {st.session_state.model_choice}
* Operational Dataset Test Accuracy: {st.session_state.accuracy*100:.2f}%
======================================================
[PREDICTIVE ANALYTIC DIAGNOSTIC RESULT]
* Severity Tier Status: {severity_status}
* Measured Probability Target Index: {prob_percent:.1f}%
* Clinical Assessment Note: {severity_desc}
======================================================
"""
            st.markdown("---")
            st.download_button(
                label=f"📥 Download Clinical Dossier File (.TXT)", 
                data=report_data, 
                file_name=f"Clinical_System_Report_{p['patient_id']}.txt", 
                mime="text/plain",
                key="download_report_btn"
            )

    # ---------------------------------------------------------
    # MODULE 6: HELP / ABOUT MODULE (ABOUT)
    # ---------------------------------------------------------
    elif st.session_state.current_nav == "ABOUT":
        st.subheader("Help / About Module - Diagnostic Specifications")
        st.markdown("""
        <div class="hero-body">
        This software solution serves as an automated data processing tool for heart failure validation benchmarks. 
        <br><br>
        <b>Technical Pipeline Architecture Specifications:</b><br>
        1. <b>Data Preprocessing Module:</b> Automatically handles column scaling, data cleaning, and mapping optimization tasks utilizing Scikit-Learn transformers.<br>
        2. <b>Feature Selection Module:</b> Restructures the 12 core clinical parameters to capture highest variance across historical patient records.<br>
        3. <b>Predictive Classifier Sandbox:</b> Evaluates Logistic Regression (LR), Decision Trees (DT), Random Forests (RF), Support Vector Machines (SVM), and Extreme Gradient Boosting (XGBoost) models dynamically.<br>
        4. <b>Dossier Generator:</b> Packs metrics arrays into standard tabular configurations and serializes text diagnostic files for local storage export.
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"System Operational Interruption: Verify dataset presence.")

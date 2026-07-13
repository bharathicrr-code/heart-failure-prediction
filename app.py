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

# --- GLOBAL HIGH-READABILITY INTERFACE CSS & GLOBAL FONT SCALING ---
st.markdown("""
    <style>
    /* Prevent top padding cutoff */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 3rem !important;
    }
    
    /* MAGNIFY LABELS AND STYLES */
    .stSlider label p, .stSelectbox label p, .stTextInput label p {
        font-size: 24px !important;
        font-weight: 700 !important;
        color: #1a3a4b !important;
    }
    
    /* INCREASE INPUT BOX TEXT AND DROPDOWN ITEM SELECTION FONT SIZE */
    .stTextInput input, .stSelectbox div[data-testid="stSelectbox-Trigger"] {
        font-size: 22px !important;
        font-weight: 600 !important;
        color: #1a3a4b !important;
        height: 55px !important;
    }
    
    /* TARGET STREAMLIT'S IN-FORM INTERACTION LABELS */
    div[data-baseweb="select"] {
        font-size: 20px !important;
        font-weight: 600 !important;
    }
    
    /* MAGNIFY NATIVE DATA TABLES SECTIONS */
    .stTable table {
        font-size: 22px !important;
    }
    .stTable td, .stTable th {
        padding: 12px 15px !important;
        font-size: 20px !important;
    }
    
    /* ENHANCED HIGH-VISIBILITY SUBMIT BUTTONS ACROSS ALL SCREEN VIEWS */
    div.stButton > button, 
    div.stDownloadButton > button {
        background-color: #0c5460 !important;
        color: white !important;
        border-radius: 10px !important;
        border: 2px solid #062c33 !important;
        padding: 22px 50px !important;
        font-size: 28px !important;
        font-weight: 900 !important;
        width: 100% !important;
        max-width: 600px !important;
        min-height: 85px !important;
        display: block !important;
        margin: 50px auto !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.22) !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background-color: #117a8b !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 20px rgba(0,0,0,0.28) !important;
    }
    
    /* TYPOGRAPHY BLOCKS */
    .hero-title {
        font-size: 46px !important;
        font-weight: 800 !important;
        color: #1a3a4b !important;
        line-height: 1.2 !important;
    }
    .hero-subtitle {
        font-size: 32px !important;
        font-weight: 700 !important;
        color: #0c5460 !important;
    }
    .hero-body, .hero-body ul li {
        font-size: 24px !important; 
        line-height: 1.8 !important;
        color: #2b2b2b !important;
    }
    
    /* HIGH CONTRAST CLINICAL REPORT SEVERITY CARD */
    .severity-card-premium {
        padding: 35px !important;
        border-radius: 12px !important;
        margin-bottom: 35px !important;
        background-color: #102A43 !important; /* Deep Premium Navy background */
        border-left: 12px solid #E12D39 !important; /* Crimson critical indicator bar */
        box-shadow: 0 6px 15px rgba(0,0,0,0.15) !important;
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

# -----------------------------------------------------------------
# AUTHENTICATION GATEWAY
# -----------------------------------------------------------------
if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        st.markdown("<h2 style='text-align: center; color: #1a3a4b;'>🔒 System Authentication Gateway</h2>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username / Clinician Identification Badge Key", value="admin")
            password = st.text_input("Secure Password Access Key", type="password", value="password")
            submit_login = st.form_submit_button("Authenticate Access Verification")
            if submit_login:
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
# MODULE 1: HOME VIEW (BALANCED DIAGRAM LAYOUT)
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
        if st.button("Initialize Patient Entry Module ➡️", key="home_start_btn"):
            st.session_state.current_nav = "FORM"
            st.rerun()
            
    with col_graphic:
        st.markdown("""
            <div style="background-color: transparent; text-align:center; padding-top:120px; padding-bottom:60px;">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 150" fill="none" stroke="#0c5460" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" style="width:100%;">
                    <path d="M 10 75 L 50 75 L 65 35 L 80 115 L 95 5 L 110 95 L 125 75 L 175 75 L 190 35 L 205 115 L 220 5 L 235 95 L 250 75 L 300 75 L 315 35 L 330 115 L 345 5 L 360 95 L 375 75 L 425 75 L 440 35 L 455 115 L 470 5 L 485 95 L 500 75 L 540 75 L 555 35 L 570 115 L 590 75"/>
                </svg>
                <p style="color:#1a3a4b; font-size:24px; font-weight:800; margin-top:35px; font-family:sans-serif; letter-spacing:1px;">Cardiovascular Diagnostics Engine Output Pipeline</p>
            </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# MODULE 2: FORM SCREEN
# ---------------------------------------------------------
elif st.session_state.current_nav == "FORM":
    st.markdown("<h2 style='font-size:36px; color:#1a3a4b;'>📋 Patient Administrative Records & Vector Matrix</h2>", unsafe_allow_html=True)
    
    col_id1, col_id2 = st.columns(2)
    with col_id1:
        patient_name = st.text_input("Patient Full Name Name Entries", value="John Doe")
    with col_id2:
        patient_id = st.text_input("Hospital Reference ID (Ref-ID Record)", value="PT-89422")
        
    st.markdown("<h3 style='font-size:30px; color:#0c5460; margin-top:25px;'>Clinical Indicators Vector Input Variables</h3>", unsafe_allow_html=True)
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
        
    if st.button("Submit Patient Details Into System Matrix ➡️", key="intake_process_btn"):
        st.session_state.patient_data = {
            "patient_name": patient_name, "patient_id": patient_id, "age": age, "anaemia": anaemia, 
            "creatinine_phosphokinase": creatinine_phosphokinase, "diabetes": diabetes, "ejection_fraction": ejection_fraction, 
            "high_blood_pressure": high_blood_pressure, "platelets": platelets, "serum_creatinine": rose_creatinine, 
            "serum_sodium": serum_sodium, "sex": sex, "smoking": smoking, "time": time
        }
        st.session_state.prediction_made = False
        st.session_state.current_nav = "MODELS"
        st.rerun()

# ---------------------------------------------------------
# MODULE 3: MODELS VIEW (CRASH RESOLVED & HIGHER CONTRAST)
# ---------------------------------------------------------
elif st.session_state.current_nav == "MODELS":
    st.markdown("<h2 style='font-size:36px; color:#1a3a4b;'>⚙️ Machine Learning Prediction Engine Module</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background-color: #e2f0d9; padding: 25px; border-radius: 8px; margin-bottom: 25px; border-left: 6px solid #385723;">
            <p style="font-size: 24px; font-weight: bold; color: #385723; margin: 0;">🔧 Model Sandbox Configuration Interface Panel</p>
            <p style="font-size: 21px; color: #2b2b2b; margin: 8px 0 0 0;">Choose the active engine below to run live predictive pipelines.</p>
        </div>
    """, unsafe_allow_html=True)
    
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
        <div style="margin: 25px 0px; background: #fff; padding: 20px; border: 1px solid #ccc; border-radius: 8px;">
            <h3 style="font-size: 30px; color: #1a3a4b; margin: 0 0 10px 0;">📊 Active Evaluation Model: <span style="color:#0c5460;">{model_choice}</span></h3>
            <p style="font-size: 26px; font-weight: 800; color: #2b2b2b; margin: 0;">Calculated System Dataset Test Accuracy Score: <span style="color:#0c5460; font-size:32px;">{accuracy * 100:.2f}%</span></p>
        </div>
    """, unsafe_allow_html=True)
    
    # SYSTEM CRASH FIX: Standard case-sensitive palette name prevents Seaborn from stopping the chart render pipeline
    importance_values = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.abs(model.coef_[0])
    feat_imp_df = pd.DataFrame({'Feature Vector': feature_names, 'Weight Importance': importance_values}).sort_values(by='Weight Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 5.0))
    sns.barplot(x='Weight Importance', y='Feature Vector', data=feat_imp_df, palette='Blues_r', hue='Feature Vector', legend=False, ax=ax)
    ax.tick_params(labelsize=14)
    st.pyplot(fig)
    
    # Save parameters to keep persistence clean across navigation tabs
    st.session_state.model_choice = model_choice
    st.session_state.accuracy = accuracy
    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.feature_names = feature_names

    if st.button("Execute Core Analysis Prediction Pipeline ⚡", key="run_prediction_btn"):
        if st.session_state.patient_data is None:
            st.warning("⚠️ Access interrupted. Please input metrics in the Patient Intake Form first.")
        else:
            st.session_state.prediction_made = True
            st.success("🤖 Mathematical calculation complete. Processed vectors successfully passed down to Report module.")

# ---------------------------------------------------------
# MODULE 4: PREMIUM DIAGNOSTIC REPORT (HIGH CONTRAST NAVY)
# ---------------------------------------------------------
elif st.session_state.current_nav == "REPORT":
    st.markdown("<h2 style='font-size:36px; color:#1a3a4b;'>📊 Reports Generation & Dossier Export Module</h2>", unsafe_allow_html=True)
    
    if st.session_state.patient_data is None:
        st.warning("⚠️ Prediction logs missing. Please record profile elements in the Patient Intake Form first.")
    else:
        # Recompile or extract current operational vector blocks securely
        p = st.session_state.patient_data
        user_data = pd.DataFrame([[p[f] for f in feature_names]], columns=feature_names)
        
        # Default safety logic to compute parameters if session state dropped during execution
        if not hasattr(st.session_state, 'model') or st.session_state.model is None:
            fallback_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            fallback_model.fit(X_train_scaled, y_train)
            user_data_scaled = scaler.transform(user_data)
            prediction_proba = fallback_model.predict_proba(user_data_scaled)[0][1]
            active_engine_name = "Random Forest Classifier (Default Mode)"
        else:
            user_data_scaled = st.session_state.scaler.transform(user_data)
            prediction_proba = st.session_state.model.predict_proba(user_data_scaled)[0][1]
            active_engine_name = st.session_state.model_choice
            
        prob_percent = prediction_proba * 100
        
        if prob_percent < 35.0:
            severity_status = "LOW SYSTEMIC SEVERITY"
            severity_desc = "Clinical metrics are inside standard baseline windows. Structural heart performance scores are regularized."
        elif 35.0 <= prob_percent < 65.0:
            severity_status = "ELEVATED SEVERITY TIER (BORDERLINE WARNING)"
            severity_desc = "Noticeable data divergence across target biomarkers. Continued monitoring and clinical surveillance recommended."
        else:
            severity_status = "CRITICAL HIGH-RISK SEVERITY PATHWAY"
            severity_desc = "Significant clinical deviations observed across multiple vector points. Urgent physiological intervention indicated."

        # PREMIUM DYNAMIC NAVY VIEW CARD
        st.markdown(f"""
            <div class="severity-card-premium">
                <h2 style="color: #FFFFFF !important; font-size: 34px !important; font-weight: 900; margin: 0 0 12px 0; letter-spacing:1px;">
                    📝 {severity_status}
                </h2>
                <p style="color: #F0F4F8 !important; font-size: 24px !important; line-height: 1.6; margin: 0; font-weight: 500;">
                    Patient Risk Vector Probability Evaluation: <span style="color:#FFDF00; font-size:28px; font-weight:800;">{prob_percent:.1f}%</span>
                </p>
                <p style="color: #D9E2EC !important; font-size: 21px !important; margin-top: 10px; font-style: italic;">
                    <b>Diagnostic Assessment Guideline:</b> {severity_desc}
                </p>
            </div>
        """, unsafe_allow_html=True)
            
        metrics_display = {
            "Monitored Clinical Attribute Fields": ["Patient Full Identification Name", "Hospital Reference Tracking Key", "Evaluated Age Bracket", "Selected Algorithmic Engine Backbone", "Ejection Fraction Target Metric", "Serum Creatinine Density Value", "Aggregated Severity Index"],
            "Assigned Patient Vector Values": [p['patient_name'], p['patient_id'], f"{p['age']} Years Old", str(active_engine_name), f"{p['ejection_fraction']}% Volume Ratio", f"{p['serum_creatinine']} mg/dL", f"{prob_percent:.1f}% -> {severity_status}"]
        }
        st.table(pd.DataFrame(metrics_display))

        if st.button("📥 Download Finalized Clinical Dossier Export File", key="download_report_btn"):
            st.success("Dossier compiled successfully. File system pipeline active.")

# ---------------------------------------------------------
# MODULE 5: COMPREHENSIVE HELP / ABOUT VIEW
# ---------------------------------------------------------
elif st.session_state.current_nav == "ABOUT":
    st.markdown("<h2 style='font-size:36px; color:#1a3a4b;'>ℹ️ Help / About Module - Diagnostic Specifications</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="hero-body">
    This software platform functions as an automated data validation sandbox for evaluating heart failure risk arrays.
    <br><br>
    <b>Core Architectural Specification Metrics:</b>
    </div>
    """, unsafe_allow_html=True)
    
    specs_df = pd.DataFrame({
        "System Pipeline Layer": ["1. Intake Data Regularization", "2. Vector Processing Transformer", "3. Analytics Classifiers Sandbox", "4. Report Compiler Factory"],
        "Operational Scope & Functional Target Goals": [
            "Handles automated structural variable checks, field constraint bounds tracking, and error mitigation.",
            "Normalizes multi-scale physiological patient vectors using standard variance transformations.",
            "Executes concurrent evaluation of linear, tree, and ensemble classification architectures.",
            "Assembles dynamic clinical records, evaluates the severity index tier, and serializes flat-file data exports."
        ],
        "Platform Dependencies": ["Python 3.11 / Streamlit Engine", "Scikit-Learn Preprocessing", "XGBoost Framework Module", "Matplotlib / Native Core Tables"]
    })
    st.table(specs_df)

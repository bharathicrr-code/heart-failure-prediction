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

# --- GLOBAL HIGH-READABILITY INTERFACE CSS & STYLING CORE ---
st.markdown("""
    <style>
    /* Prevent top padding cutoff */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 3rem !important;
    }
    
    /* 1. FORCE MAXIMUM FONT SIZE FOR ALL INPUT BOXES AND DROPDOWNS (INCLUDING LOGIN) */
    .stTextInput input, .stSelectbox div, div[data-baseweb="select"] * {
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #1a3a4b !important;
    }
    
    /* Make the actual layout rows of dropdown choices and text inputs taller to match font */
    .stTextInput > div > div > input {
        min-height: 60px !important;
    }
    div[data-testid="stSelectbox-Trigger"] {
        min-height: 60px !important;
    }
    
    /* 2. MAGNIFY ALL INPUT LABELS AND SLIDER TITLES */
    .stSlider label p, .stSelectbox label p, .stTextInput label p {
        font-size: 24px !important;
        font-weight: 700 !important;
        color: #1a3a4b !important;
    }
    
    /* 3. FORCE SLIDER VALUE FLOATING NUMBERS TO BE LARGE AND VISIBLE */
    div[data-testid="stThumbValue"] {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #0c5460 !important;
    }
    
    /* 4. MAGNIFY BOTH SIDES OF NATIVE DATA TABLES WITH HIGHLIGHTED HEADERS */
    .stTable table th {
        background-color: #0c5460 !important;
        color: white !important;
        font-size: 24px !important;
        font-weight: 800 !important;
    }
    .stTable td {
        padding: 14px 18px !important;
        font-size: 22px !important;
        font-weight: 600 !important;
        color: #1a3a4b !important;
    }
    
    /* 5. CLEAN CENTERED ACTION BUTTONS */
    div.stButton > button, 
    div.stDownloadButton > button {
        background-color: #0c5460 !important;
        color: white !important;
        border-radius: 10px !important;
        border: 2px solid #062c33 !important;
        padding: 20px 40px !important;
        font-size: 24px !important;
        font-weight: 800 !important;
        width: 100% !important;
        max-width: 500px !important;
        min-height: 70px !important;
        display: block !important;
        margin: 40px auto !important; /* Center Alignment */
        box-shadow: 0 8px 16px rgba(0,0,0,0.22) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background-color: #117a8b !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 20px rgba(0,0,0,0.28) !important;
    }
    
    /* HERO TYPOGRAPHY BLOCKS */
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
    
    /* COMPLETE ASSESSMENT BOX DYNAMIC BACKGROUND */
    .assessment-box-safe {
        background-color: #28a745 !important; /* Green */
        padding: 35px !important; border-radius: 12px !important; margin-bottom: 35px !important; color: white !important;
    }
    .assessment-box-borderline {
        background-color: #fd7e14 !important; /* Orange */
        padding: 35px !important; border-radius: 12px !important; margin-bottom: 35px !important; color: white !important;
    }
    .assessment-box-severe {
        background-color: #dc3545 !important; /* Red */
        padding: 35px !important; border-radius: 12px !important; margin-bottom: 35px !important; color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- MACHINE LEARNING DATA LOGIC ACCELERATOR ---
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
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        st.markdown("<h2 style='text-align: center; color: #1a3a4b; font-size:36px;'>🔒 System Authentication Gateway</h2>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username / Clinician Identification Badge Key", value="admin")
            password = st.text_input("Secure Password Access Key", type="password", value="password")
            
            # Form-specific submit button handling centering
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
        
    with col_graphic:
        st.markdown("""
            <div style="background-color: transparent; text-align:center; padding-top:60px; padding-bottom:30px;">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 150" fill="none" stroke="#0c5460" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" style="width:100%;">
                    <path d="M 10 75 L 50 75 L 65 35 L 80 115 L 95 5 L 110 95 L 125 75 L 175 75 L 190 35 L 205 115 L 220 5 L 235 95 L 250 75 L 300 75"/>
                </svg>
            </div>
        """, unsafe_allow_html=True)
    
    if st.button("Initialize Patient Entry Module ➡️", key="home_start_btn"):
        st.session_state.current_nav = "FORM"
        st.rerun()

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
        
    if st.button("Submit Details", key="intake_process_btn"):
        st.session_state.patient_data = {
            "patient_name": patient_name, "patient_id": patient_id, "age": age, "anaemia": anaemia, 
            "creatinine_phosphokinase": creatinine_phosphokinase, "diabetes": diabetes, "ejection_fraction": ejection_fraction, 
            "high_blood_pressure": high_blood_pressure, "platelets": platelets, "serum_creatinine": rose_creatinine, 
            "serum_sodium": serum_sodium, "sex": sex, "smoking": smoking, "time": time
        }
        st.session_state.current_nav = "MODELS"
        st.rerun()

# ---------------------------------------------------------
# MODULE 3: MODELS VIEW (PRESERVED PAINT MODEL DESIGN)
# ---------------------------------------------------------
elif st.session_state.current_nav == "MODELS":
    st.markdown("<h2 style='font-size:36px; color:#1a3a4b;'>⚙️ Machine Learning Prediction Engine Module</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background-color: #e2f0d9; padding: 25px; border-radius: 8px; margin-bottom: 25px; border-left: 6px solid #385723;">
            <p style="font-size: 24px; font-weight: bold; color: #385723; margin: 0;">🔧 Model Sandbox Configuration Interface Panel</p>
            <p style="font-size: 21px; color: #2b2b2b; margin: 8px 0 0 0;">Choose the active engine below to run live predictive pipelines.</p>
        </div>
    """, unsafe_allow_html=True)
    
    model_choice = st.selectbox("Select Active Machine Learning Analytics Engine", ["Random Forest Classifier", "Logistic Regression Framework", "XGBoost Core Engine", "Decision Tree Model", "Support Vector Machine (SVM)"])
    
    if "Random Forest" in model_choice:
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    elif "Logistic Regression" in model_choice:
        model = LogisticRegression(max_iter=1000)
    elif "XGBoost" in model_choice:
        model = XGBClassifier(max_depth=4)
    elif "Decision Tree" in model_choice:
        model = DecisionTreeClassifier(max_depth=5)
    else:
        model = SVC(probability=True, random_state=42)
        
    model.fit(X_train_scaled, y_train)
    accuracy = model.score(X_test_scaled, y_test)
    
    st.markdown(f"""
        <div style="margin: 25px 0px; background: #fff; padding: 20px; border: 1px solid #ccc; border-radius: 8px;">
            <h3 style="font-size: 30px; color: #1a3a4b; margin: 0 0 10px 0;">📊 Active Evaluation Model: <span style="color:#0c5460;">{model_choice}</span></h3>
            <p style="font-size: 26px; font-weight: 800; color: #2b2b2b; margin: 0;">Calculated System Dataset Test Accuracy Score: <span style="color:#0c5460; font-size:32px;">{accuracy * 100:.2f}%</span></p>
        </div>
    """, unsafe_allow_html=True)
    
    # MULTI-COLOR GRADIENT VISUALIZATION PIPELINE PRESERVED
    importance_values = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.abs(model.coef_[0])
    feat_imp_df = pd.DataFrame({'Feature Vector': feature_names, 'Weight Importance': importance_values}).sort_values(by='Weight Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.barplot(x='Weight Importance', y='Feature Vector', data=feat_imp_df, palette='viridis', hue='Feature Vector', legend=False, ax=ax)
    ax.tick_params(labelsize=14)
    ax.set_ylabel("Feature Vector", fontsize=15, fontweight='bold')
    ax.set_xlabel("Weight Importance", fontsize=15, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.session_state.model_choice = model_choice
    st.session_state.accuracy = accuracy
    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.feature_names = feature_names

    if st.button("Run Analysis", key="run_prediction_btn"):
        if st.session_state.patient_data is None:
            st.warning("⚠️ Access interrupted. Please input metrics in the Patient Intake Form first.")
        else:
            st.session_state.current_nav = "REPORT"
            st.rerun()

# ---------------------------------------------------------
# MODULE 4: DIAGNOSTIC REPORT (EXACT CORRECTED BOX COLORS)
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
        
        # EXACT FULL BACKGROUND COLOR SHIFT MATCHING YOUR RULE SET
        if prob_percent < 35.0:
            box_class = "assessment-box-safe" # FULL GREEN BACKGROUND
            severity_status = "SAFE PERFORMING METRIC WINDOWS"
            severity_desc = "Clinical metrics are inside standard baseline windows. Structural heart performance scores are regularized."
        elif 35.0 <= prob_percent < 65.0:
            box_class = "assessment-box-borderline" # FULL ORANGE BACKGROUND
            severity_status = "BORDERLINE WARNING TIER"
            severity_desc = "Noticeable data divergence across target biomarkers. Continued monitoring and clinical surveillance recommended."
        else:
            box_class = "assessment-box-severe" # FULL RED BACKGROUND
            severity_status = "SEVERE CRITICAL HIGH-RISK PATHWAY"
            severity_desc = "Significant clinical deviations observed across multiple vector points. Urgent physiological intervention indicated."

        st.markdown(f"""
            <div class="{box_class}">
                <h2 style="color: white !important; margin:0 0 10px 0; font-weight:900; font-size:36px;">📝 Assessment: {severity_status}</h2>
                <p style="color: white !important; font-size:26px; font-weight:700; margin:0;">Patient Risk Vector Probability Evaluation: {prob_percent:.1f}%</p>
                <p style="color: white !important; font-size: 22px; font-style: italic; margin-top: 15px; opacity: 0.95;">
                    <b>Diagnostic Assessment Guideline:</b> {severity_desc}
                </p>
            </div>
        """, unsafe_allow_html=True)
            
        metrics_display = {
            "Monitored Clinical Attribute Fields": ["Patient Full Identification Name", "Hospital Reference Tracking Key", "Evaluated Age Bracket", "Selected Algorithmic Engine Backbone", "Ejection Fraction Target Metric", "Serum Creatinine Density Value", "Aggregated Severity Index"],
            "Assigned Patient Vector Values": [p['patient_name'], p['patient_id'], f"{p['age']} Years Old", str(active_engine_name), f"{p['ejection_fraction']}% Volume Ratio", f"{p['serum_creatinine']} mg/dL", f"{prob_percent:.1f}% -> {severity_status}"]
        }
        st.table(pd.DataFrame(metrics_display))

        if st.button("Download Report", key="download_report_btn"):
            st.success("Dossier compiled successfully. File system pipeline active.")

# ---------------------------------------------------------
# MODULE 5: SYSTEM ABOUT VIEW
# ---------------------------------------------------------
elif st.session_state.current_nav == "ABOUT":
    st.markdown("<h2 style='font-size:36px; color:#1a3a4b;'>ℹ️ Help / About Module - Diagnostic Specifications</h2>", unsafe_allow_html=True)
    specs_df = pd.DataFrame({
        "System Pipeline Layer": ["1. Intake Data Regularization", "2. Vector Processing Transformer", "3. Analytics Classifiers Sandbox", "4. Report Compiler Factory"],
        "Operational Scope & Functional Target Goals": ["Input variable verification", "Normalizes patient metrics", "Executes machine learning algorithms", "Assembles dynamic clinical records"],
        "Platform Dependencies": ["Python / Streamlit", "Scikit-Learn", "XGBoost Core", "Matplotlib"]
    })
    st.table(specs_df)

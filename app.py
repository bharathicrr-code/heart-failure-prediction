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

st.set_page_config(page_title="CardioShield CDSS", layout="wide")

# --- CUSTOM CSS FOR HIGH-VISIBILITY TYPOGRAPHY & BUTTONS ---
st.markdown("""
    <style>
    /* Top Logo Typography */
    .nav-logo {
        font-size: 34px !important;
        font-weight: 800 !important;
        color: #0c5460 !important;
    }
    
    /* Enlarge and improve readability of titles on Home Page */
    .hero-title {
        font-size: 46px !important;
        font-weight: 800 !important;
        color: #1a3a4b !important;
        margin-bottom: 10px !important;
        line-height: 1.2 !important;
    }
    .hero-subtitle {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #0c5460 !important;
        margin-bottom: 25px !important;
    }
    .hero-body {
        font-size: 22px !important;
        line-height: 1.7 !important;
        color: #2b2b2b !important;
    }
    .hero-body ul li {
        font-size: 21px !important;
        margin-bottom: 8px !important;
    }
    
    /* Make Navigation Bar Header Buttons Highly Visible */
    div[data-testid="stHorizontalBlock"] div.stButton > button {
        background-color: #ffffff !important;
        color: #0c5460 !important;
        border: 3px solid #0c5460 !important;
        border-radius: 8px !important;
        padding: 12px 20px !important;
        font-size: 20px !important;
        font-weight: 800 !important;
        width: 100% !important;
        min-height: 55px !important;
    }
    div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
        background-color: #0c5460 !important;
        color: white !important;
    }

    /* Target Action Buttons at the Bottom of Pages (Non-navigation) */
    .action-container div.stButton > button, 
    .action-container div.stDownloadButton > button {
        background-color: #0c5460 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 16px 45px !important;
        font-size: 22px !important;
        font-weight: bold !important;
        width: 320px !important; /* Fixed compact professional size */
        min-height: 60px !important;
        display: block !important;
        margin: 0 auto !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    .action-container div.stButton > button:hover,
    .action-container div.stDownloadButton > button:hover {
        background-color: #0a434d !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- INITIALIZE STATE ---
if "current_nav" not in st.session_state:
    st.session_state.current_nav = "HOME"
if "patient_data" not in st.session_state:
    st.session_state.patient_data = None

# --- DATA ACCESS LAYER ---
@st.cache_data
def load_data():
    return pd.read_csv("heart_failure_clinical_records_dataset.csv")

try:
    df = load_data()
    X = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']
    feature_names = X.columns.tolist()
    
    # --- BUSINESS LOGIC LAYER ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------------
    # TOP WEB MENU NAVIGATION BAR (Tightened horizontal spacing)
    # ---------------------------------------------------------
    col_logo, col_space, col_menu = st.columns([4, 1, 6])
    
    with col_logo:
        st.markdown('<div class="nav-logo">🩺 CardioShield</div>', unsafe_allow_html=True)
        
    with col_menu:
        nav_cols = st.columns(4)
        if nav_cols[0].button("HOME", key="btn_nav_home"):
            st.session_state.current_nav = "HOME"
            st.rerun()
        if nav_cols[1].button("INTAKE", key="btn_nav_intake"):
            st.session_state.current_nav = "INTAKE"
            st.rerun()
        if nav_cols[2].button("MODELS", key="btn_nav_eval"):
            st.session_state.current_nav = "EVALUATION"
            st.rerun()
        if nav_cols[3].button("REPORT", key="btn_nav_rep"):
            st.session_state.current_nav = "REPORT"
            st.rerun()

    st.markdown("---")

    # ---------------------------------------------------------
    # PAGE MODULE 1: IMPROVED HOME PAGE VIEW WITH RELEVANT HEART IMAGE
    # ---------------------------------------------------------
    if st.session_state.current_nav == "HOME":
        col_info, col_graphic = st.columns([4, 3])
        
        with col_info:
            st.markdown('<div class="hero-title">Welcome to the CardioShield Clinical Portal</div>', unsafe_allow_html=True)
            st.markdown('<div class="hero-subtitle">Next-Generation Clinical Decision Support System (CDSS)</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="hero-body">
            CardioShield leverages advanced machine learning pipelines to assist healthcare professionals with objective risk stratification. 
            By processing key clinical indicators synchronously, the engine outputs predictive insights to catch signs of advanced heart failure early.
            <br><br>
            <b>Key Architecture Pillars:</b>
            <ul>
                <li><b>High Accuracy:</b> Optimized metrics utilizing ensemble learning algorithms.</li>
                <li><b>Comprehensive Vectors:</b> Cross-references 12 critical biological indicators including serum metrics and cardiovascular history.</li>
                <li><b>Instant Dossier Generation:</b> Formulates downloadable diagnostic insights instantly.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown('<div class="action-container">', unsafe_allow_html=True)
            if st.button("Start Evaluation ➡️", key="home_start_btn"):
                st.session_state.current_nav = "INTAKE"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
                
        with col_graphic:
            # Replaced with an accurate, highly relevant cardiology diagnostic illustration vector
            st.image("https://img.freepik.com/free-vector/cardiogram-concept-illustration_114360-17070.jpg", caption="Cardiovascular Pulse & Rhythm Diagnostics Monitor", use_container_width=True)

    # ---------------------------------------------------------
    # PAGE MODULE 2: INTAKE FORM VIEW
    # ---------------------------------------------------------
    elif st.session_state.current_nav == "INTAKE":
        st.subheader("Patient Administrative & Clinical Metrics Entry")
        
        col_id1, col_id2 = st.columns(2)
        with col_id1:
            patient_name = st.text_input("Patient Full Name", value="John Doe")
        with col_id2:
            patient_id = st.text_input("Hospital Ref ID", value="PT-89422")
            
        st.markdown("### Clinical Indicators Vector Input")
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Patient Age", int(df['age'].min()), int(df['age'].max()), 70)
            anaemia = st.selectbox("Anaemia Diagnosis", [0, 1], format_func=lambda x: "Negative" if x==0 else "Positive")
            creatinine_phosphokinase = st.slider("CPK Level (mcg/L)", int(df['creatinine_phosphokinase'].min()), int(df['creatinine_phosphokinase'].max()), 250)
            diabetes = st.selectbox("Diabetes Status", [0, 1], format_func=lambda x: "No History" if x==0 else "Diabetic")
            ejection_fraction = st.slider("Ejection Fraction (%)", int(df['ejection_fraction'].min()), int(df['ejection_fraction'].max()), 20)
            high_blood_pressure = st.selectbox("Hypertension History", [0, 1], format_func=lambda x: "Normal BP" if x==0 else "Hypertensive")
            
        with col2:
            platelets = st.slider("Platelets Count", float(df['platelets'].min()), float(df['platelets'].max()), 250000.0, step=1000.0)
            rose_creatinine = st.slider("Serum Creatinine (mg/dL)", float(df['serum_creatinine'].min()), float(df['serum_creatinine'].max()), 2.5, step=0.1)
            serum_sodium = st.slider("Serum Sodium (mEq/L)", int(df['serum_sodium'].min()), int(df['serum_sodium'].max()), 135)
            sex = st.selectbox("Biological Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
            smoking = st.selectbox("Smoking Profile", [0, 1], format_func=lambda x: "Non-smoker" if x==0 else "Active Smoker")
            time = st.slider("Follow-up Window (Days)", int(df['time'].min()), int(df['time'].max()), 100)
            
        st.markdown("---")
        
        # Highly visible, short action button container
        st.markdown('<div class="action-container">', unsafe_allow_html=True)
        if st.button("Analyze Vitals ➡️", key="intake_process_btn"):
            st.session_state.patient_data = {
                "patient_name": patient_name, "patient_id": patient_id,
                "age": age, "anaemia": anaemia, "creatinine_phosphokinase": creatinine_phosphokinase,
                "diabetes": diabetes, "ejection_fraction": ejection_fraction, "high_blood_pressure": high_blood_pressure,
                "platelets": platelets, "serum_creatinine": rose_creatinine, "serum_sodium": serum_sodium,
                "sex": sex, "smoking": smoking, "time": time
            }
            st.session_state.current_nav = "EVALUATION"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # PAGE MODULE 3: MODEL EVALUATION DESK
    # ---------------------------------------------------------
    elif st.session_state.current_nav == "EVALUATION":
        st.subheader("Model Validation & Core Metrics Desk")
        
        st.info("⚙️ Developer Diagnostic Sandbox: Use this control selection panel to toggle different backend machine learning algorithms to showcase variant accuracy metrics and weights during evaluation benchmarking.")
        
        model_choice = st.selectbox("Select Active Analytics Backbone Engine", ["Random Forest", "Logistic Regression", "XGBoost", "Decision Tree", "SVM"])
        
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
        
        st.metric(label=f"{model_choice} General System Accuracy", value=f"{accuracy * 100:.2f}%")
        
        importance_values = None
        if model_choice == "Logistic Regression":
            importance_values = np.abs(model.coef_[0])
        elif model_choice in ["Decision Tree", "Random Forest", "XGBoost"]:
            importance_values = model.feature_importances_
            
        if importance_values is not None:
            feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Weight': importance_values}).sort_values(by='Weight', ascending=False)
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x='Weight', y='Feature', data=feat_imp_df, palette='viridis', ax=ax)
            st.pyplot(fig)
            
        st.session_state.model_choice = model_choice
        st.session_state.accuracy = accuracy
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.feature_names = feature_names

        st.markdown("---")
        
        # Unified layout action button 
        st.markdown('<div class="action-container">', unsafe_allow_html=True)
        if st.button("Generate Report 📋", key="eval_report_btn"):
            st.session_state.current_nav = "REPORT"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # PAGE MODULE 4: CLINICAL REPORT VIEW
    # ---------------------------------------------------------
    elif st.session_state.current_nav == "REPORT":
        st.subheader("Finalized Assessment & Dossier Export")
        if st.session_state.patient_data is None:
            st.warning("No dynamic medical records detected. Please fill out the Patient Intake page first.")
        else:
            p = st.session_state.patient_data
            
            if "model" not in st.session_state:
                fallback = RandomForestClassifier(n_estimators=50, max_depth=3)
                fallback.fit(X_train_scaled, y_train)
                st.session_state.model_choice = "Random Forest"
                st.session_state.accuracy = fallback.score(X_test_scaled, y_test)
                st.session_state.model = fallback
                st.session_state.scaler = scaler
                st.session_state.feature_names = feature_names

            user_data = pd.DataFrame([[p[f] for f in st.session_state.feature_names]], columns=st.session_state.feature_names)
            user_data_scaled = st.session_state.scaler.transform(user_data)
            
            prediction = st.session_state.model.predict(user_data_scaled)[0]
            prediction_proba = st.session_state.model.predict_proba(user_data_scaled)[0][1]
            
            if prediction == 1:
                st.error(f"⚠️ High Risk Warning Status (Probability: {prediction_proba * 100:.1f}%)")
            else:
                st.success(f"✅ Clear Health Status Summary (Probability: {prediction_proba * 100:.1f}%)")
                
            metrics_display = {
                "Parameter Field": ["Patient Full Name", "Hospital ID Reference", "Age", "Biological Sex", "Ejection Fraction", "Serum Creatinine"],
                "Assigned Vector": [p['patient_name'], p['patient_id'], f"{p['age']} Years", "Male" if p['sex'] == 1 else "Female", f"{p['ejection_fraction']}%", f"{p['serum_creatinine']} mg/dL"]
            }
            st.table(pd.DataFrame(metrics_display))

            report_data = f"""======================================================
HEART FAILURE CLINICAL ASSESSMENT DOSSIER
======================================================
[PATIENT DEMOGRAPHICS]
* Patient Name: {p['patient_name']}
* Hospital ID Reference: {p['patient_id']}

[SYSTEM ARCHITECTURE SUMMARY]
* Model Backbone Used: {st.session_state.model_choice}
* Baseline General Testing Accuracy: {st.session_state.accuracy*100:.2f}%
======================================================
"""
            st.markdown("---")
            
            # Formatted export download button matching standard actions perfectly
            st.markdown('<div class="action-container">', unsafe_allow_html=True)
            st.download_button(
                label=f"📥 Download Report (.TXT)", 
                data=report_data, 
                file_name=f"Clinical_Report_{p['patient_id']}.txt", 
                mime="text/plain",
                key="download_report_btn"
            )
            st.markdown('</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"System Initialization Interrupted: {e}")

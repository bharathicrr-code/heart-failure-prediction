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

# --- CUSTOM CSS FOR CLASSIC WEB NAVIGATION HEADER ---
st.markdown("""
    <style>
    .nav-logo {
        font-size: 24px;
        font-weight: bold;
        color: #0c5460;
    }
    div.stButton > button:first-child {
        font-weight: bold;
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
    # TOP WEB MENU NAVIGATION BAR
    # ---------------------------------------------------------
    col_logo, col_menu = st.columns([1, 2])
    
    with col_logo:
        st.markdown('<div class="nav-logo">🩺 CardioShield</div>', unsafe_allow_html=True)
        
    with col_menu:
        nav_cols = st.columns(4)
        # Highlight active page style using standard buttons
        if nav_cols[0].button("HOME", use_container_width=True):
            st.session_state.current_nav = "HOME"
            st.rerun()
        if nav_cols[1].button("PATIENT INTAKE", use_container_width=True):
            st.session_state.current_nav = "INTAKE"
            st.rerun()
        if nav_cols[2].button("MODEL EVALUATION", use_container_width=True):
            st.session_state.current_nav = "EVALUATION"
            st.rerun()
        if nav_cols[3].button("CLINICAL REPORT", use_container_width=True):
            st.session_state.current_nav = "REPORT"
            st.rerun()

    st.markdown("---")

    # ---------------------------------------------------------
    # PAGE MODULE 1: HOME PAGE VIEW
    # ---------------------------------------------------------
    if st.session_state.current_nav == "HOME":
        st.subheader("Welcome to the CardioShield Clinical Portal")
        
        col_info, col_graphic = st.columns([3, 2])
        with col_info:
            st.markdown("""
            ### About CardioShield CDSS
            An advanced Clinical Decision Support System built to assist medical practitioners 
            in evaluating heart failure probabilities dynamically using machine learning pipelines.
            
            * **Objective:** Modernizing preventative cardiovascular analytics.
            * **Methodology:** Processing 12 patient features against ensemble models.
            """)
            if st.button("Start New Patient Evaluation ➡️"):
                st.session_state.current_nav = "INTAKE"
                st.rerun()
        with col_graphic:
            st.info("💡 **System Core Active:** Pipeline loaded with baseline clinical records successfully.")

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
            serum_creatinine = st.slider("Serum Creatinine (mg/dL)", float(df['serum_creatinine'].min()), float(df['serum_creatinine'].max()), 2.5, step=0.1)
            serum_sodium = st.slider("Serum Sodium (mEq/L)", int(df['serum_sodium'].min()), int(df['serum_sodium'].max()), 135)
            sex = st.selectbox("Biological Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
            smoking = st.selectbox("Smoking Profile", [0, 1], format_func=lambda x: "Non-smoker" if x==0 else "Active Smoker")
            time = st.slider("Follow-up Window (Days)", int(df['time'].min()), int(df['time'].max()), 100)
            
        st.markdown("---")
        # Catchy Premium Button and Automatic Page Transition
        if st.button("Analyze Patient Vitals ➡️", type="primary", use_container_width=True):
            st.session_state.patient_data = {
                "patient_name": patient_name, "patient_id": patient_id,
                "age": age, "anaemia": anaemia, "creatinine_phosphokinase": creatinine_phosphokinase,
                "diabetes": diabetes, "ejection_fraction": ejection_fraction, "high_blood_pressure": high_blood_pressure,
                "platelets": platelets, "serum_creatinine": serum_creatinine, "serum_sodium": serum_sodium,
                "sex": sex, "smoking": smoking, "time": time
            }
            st.session_state.current_nav = "EVALUATION"
            st.rerun()

    # ---------------------------------------------------------
    # PAGE MODULE 3: MODEL EVALUATION DESK
    # ---------------------------------------------------------
    elif st.session_state.current_nav == "EVALUATION":
        st.subheader("Model Validation & Core Metrics Desk")
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
        # Automatic transition button to skip manual header navigation
        if st.button("Generate Official Report 📋", type="primary", use_container_width=True):
            st.session_state.current_nav = "REPORT"
            st.rerun()

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
            # Dedicated clear download trigger at the bottom of the report page
            st.download_button(
                label=f"📥 Download Report for {p['patient_name']} (.TXT)", 
                data=report_data, 
                file_name=f"Clinical_Report_{p['patient_id']}.txt", 
                mime="text/plain",
                use_container_width=True
            )

except Exception as e:
    st.error(f"System Initialization Interrupted: {e}")

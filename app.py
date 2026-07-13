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

st.set_page_config(page_title="CardioShield CDSS", layout="centered")

# --- PRESENTATION LAYER: MASTER CREDENTIALS FOR EXAMINER LIVE DEMO ---
# Stored in temporary RAM memory. No local disk storage required.
if "page" not in st.session_state:
    st.session_state.page = "auth"
if "patient_data" not in st.session_state:
    st.session_state.patient_data = None
if "registered_users" not in st.session_state:
    st.session_state.registered_users = {
        "admin@hospital.com": "admin123",
        "hod@evaluation.edu": "pass123"  # Master key for your presentation
    }

# --- DATA ACCESS LAYER (Loads Kaggle Reference Dataset into Memory) ---
@st.cache_data
def load_data():
    return pd.read_csv("heart_failure_clinical_records_dataset.csv")

try:
    df = load_data()
    X = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']
    feature_names = X.columns.tolist()
    
    # --- BUSINESS LOGIC LAYER (Model Training & Feature Scaling) ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------------
    # PAGE 1: AUTHENTICATION GATEWAY
    # ---------------------------------------------------------
    if st.session_state.page == "auth":
        st.title("🏥 CardioShield Clinical Portal")
        st.write("### System Architecture Verification Gate")
        st.write("Demonstrating secure system entry. Register a new custom user dynamically, or log in using preset presentation keys.")
        
        tab1, tab2 = st.tabs(["🔑 Portal Secure Login", "📝 New Staff Registration"])
        
        with tab1:
            login_email = st.text_input("Username (Email)", key="login_email")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            if st.button("Authenticate and Enter"):
                if login_email in st.session_state.registered_users and st.session_state.registered_users[login_email] == login_pass:
                    st.success("Access Granted. Entering Presentation Layer...")
                    st.session_state.page = "intake"
                    st.rerun()
                else:
                    st.error("Access Denied. Invalid clinical credentials.")
                    
        with tab2:
            st.info("Dynamic Registration: Creating an account here builds temporary session memory for the examiner without changing local disk files.")
            reg_email = st.text_input("Create User Username", key="reg_email")
            reg_pass = st.text_input("Create Account Password", type="password", key="reg_pass")
            if st.button("Register Custom Account"):
                if reg_email and reg_pass:
                    st.session_state.registered_users[reg_email] = reg_pass
                    st.success("Registration added to Session Memory! You can now toggle to the Login tab.")
                else:
                    st.warning("All verification fields are required.")

    # ---------------------------------------------------------
    # PAGE 2: CLINICAL VITALS INTAKE FORM (Live Patient Demo)
    # ---------------------------------------------------------
    elif st.session_state.page == "intake":
        st.title("🩺 Real-Time Patient Intake Form")
        st.write("Input custom health values live in front of the evaluator to test the predictive pipeline.")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Patient Age", int(df['age'].min()), int(df['age'].max()), 70)
            anaemia = st.selectbox("Anaemia Diagnosis", [0, 1], format_func=lambda x: "Negative" if x==0 else "Positive")
            creatinine_phosphokinase = st.slider("Creatinine Phosphokinase (mcg/L)", int(df['creatinine_phosphokinase'].min()), int(df['creatinine_phosphokinase'].max()), 250)
            diabetes = st.selectbox("Diabetes Status", [0, 1], format_func=lambda x: "No History" if x==0 else "Diabetic")
            ejection_fraction = st.slider("Ejection Fraction (%)", int(df['ejection_fraction'].min()), int(df['ejection_fraction'].max()), 20)
            high_blood_pressure = st.selectbox("Hypertension History", [0, 1], format_func=lambda x: "Normal BP" if x==0 else "Hypertensive")
            
        with col2:
            platelets = st.slider("Platelets (kiloplatelets/mL)", float(df['platelets'].min()), float(df['platelets'].max()), 250000.0, step=1000.0)
            serum_creatinine = st.slider("Serum Creatinine (mg/dL)", float(df['serum_creatinine'].min()), float(df['serum_creatinine'].max()), 2.5, step=0.1)
            serum_sodium = st.slider("Serum Sodium (mEq/L)", int(df['serum_sodium'].min()), int(df['serum_sodium'].max()), 135)
            sex = st.selectbox("Biological Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
            smoking = st.selectbox("Smoking Profile", [0, 1], format_func=lambda x: "Non-smoker" if x==0 else "Active Smoker")
            time = st.slider("Follow-up Window (Days)", int(df['time'].min()), int(df['time'].max()), 100)
            
        if st.button("Push Vector to Business Logic Layer ➡️"):
            st.session_state.patient_data = {
                "age": age, "anaemia": anaemia, "creatinine_phosphokinase": creatinine_phosphokinase,
                "diabetes": diabetes, "ejection_fraction": ejection_fraction, "high_blood_pressure": high_blood_pressure,
                "platelets": platelets, "serum_creatinine": serum_creatinine, "serum_sodium": serum_sodium,
                "sex": sex, "smoking": smoking, "time": time
            }
            st.session_state.page = "evaluation"
            st.rerun()

    # ---------------------------------------------------------
    # PAGE 3: DIAGNOSTIC EVALUATION DESK
    # ---------------------------------------------------------
    elif st.session_state.page == "evaluation":
        st.title("🧠 Core Engine Evaluation Desk")
        st.write("Demonstrating model swapping capability within the Business Logic layer.")
        
        # Default to our verified optimal model
        model_choice = st.selectbox("Choose Target Evaluation Algorithm", ["Random Forest", "Logistic Regression", "XGBoost", "Decision Tree", "SVM"])
        
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
        
        st.metric(label=f"Selected Engine: {model_choice} Verification Accuracy", value=f"{accuracy * 100:.2f}%")
        
        # Dynamic Visual Layer Mapping
        st.subheader("Algorithmic Influence Mapping")
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
        else:
            st.info("Mathematical variance feature mapping is skipped for non-linear SVM kernels.")

        st.session_state.model_choice = model_choice
        st.session_state.accuracy = accuracy
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.feature_names = feature_names
        
        if st.button("Compile Official Health Dossier ➡️"):
            st.session_state.page = "report"
            st.rerun()

    # ---------------------------------------------------------
    # PAGE 4: FINAL MEDICAL REPORT EXPORT (Full Summary View)
    # ---------------------------------------------------------
    elif st.session_state.page == "report":
        st.title("📋 Finalized Assessment & Dossier Output")
        st.write("Summary layer compiling live patient details, mathematical predictions, and system metrics.")
        
        p = st.session_state.patient_data
        user_data = pd.DataFrame([[p[f] for f in st.session_state.feature_names]], columns=st.session_state.feature_names)
        user_data_scaled = st.session_state.scaler.transform(user_data)
        
        prediction = st.session_state.model.predict(user_data_scaled)[0]
        prediction_proba = st.session_state.model.predict_proba(user_data_scaled)[0][1]
        
        st.subheader("1. Diagnostic Risk Outcome")
        result_text = ""
        if prediction == 1:
            result_text = f"High Risk of Heart Failure Event! (Probability: {prediction_proba * 100:.1f}%)"
            st.error(f"⚠️ Alert Status: {result_text}")
        else:
            result_text = f"Low Risk of Heart Failure Event. (Probability: {prediction_proba * 100:.1f}%)"
            st.success(f"✅ Clear Status: {result_text}")
            
        st.subheader("2. Live Evaluated Vitals Summary")
        metrics_display = {
            "Clinical Metric": ["Age", "Biological Sex", "Anaemia Status", "Diabetes Status", "Hypertension Status", "Smoking Profile", "Ejection Fraction Capacity", "Serum Creatinine Level", "Serum Sodium Level", "CPK Level", "Platelets Count", "Observation Windows"],
            "Evaluated Input": [f"{p['age']} Years", "Male" if p['sex'] == 1 else "Female", "Positive" if p['anaemia'] == 1 else "Negative", "Diabetic" if p['diabetes'] == 1 else "Normal", "Hypertensive" if p['high_blood_pressure'] == 1 else "Normal", "Active Smoker" if p['smoking'] == 1 else "Non-smoker", f"{p['ejection_fraction']}%", f"{p['serum_creatinine']} mg/dL", f"{p['serum_sodium']} mEq/L", f"{p['creatinine_phosphokinase']} mcg/L", f"{p['platelets']} kiloplatelets/mL", f"{p['time']} Days"]
        }
        st.table(pd.DataFrame(metrics_display))
        
        st.subheader("3. Pipeline Architecture Attributes")
        st.markdown(f"""
        * **Active Decision Model:** {st.session_state.model_choice}
        * **Verified Core Testing Accuracy:** {st.session_state.accuracy * 100:.2f}%
        """)
        
        st.markdown("---")
        
        report_data = f"""======================================================
HEART FAILURE CLINICAL ASSESSMENT DOSSIER
======================================================
[ARCHITECTURE FRAMEWORK SUMMARY]
* Model Backbone Used: {st.session_state.model_choice}
* Baseline General Testing Accuracy: {st.session_state.accuracy*100:.2f}%
* Diagnostic Evaluation: {result_text}

[LIVE INTAKE VITALS DATA]
- Patient Age: {p['age']}
- Biological Sex: {'Male' if p['sex'] == 1 else 'Female'}
- Anaemia: {'Positive' if p['anaemia'] == 1 else 'Negative'}
- Diabetes Profile: {'Diabetic' if p['diabetes'] == 1 else 'Normal'}
- Hypertension Status: {'Hypertensive' if p['high_blood_pressure'] == 1 else 'Normal'}
- Smoking Profile: {'Active Smoker' if p['smoking'] == 1 else 'Non-smoker'}
- Ejection Fraction Capacity: {p['ejection_fraction']}%
- Serum Creatinine Value: {p['serum_creatinine']} mg/dL
- Serum Sodium Level: {p['serum_sodium']} mEq/L
- Creatinine Phosphokinase Level: {p['creatinine_phosphokinase']} mcg/L
- Total Platelets Volume: {p['platelets']} kiloplatelets/mL
- Observation Monitoring Window: {p['time']} Days
======================================================
Generated via deployed CardioShield CDSS Production Instance.
"""
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.download_button(label="📥 Download Official Clinical Report (.TXT)", data=report_data, file_name="live_patient_report.txt", mime="text/plain")
        with col_btn2:
            if st.button("🔄 Reset Portal for New Session"):
                st.session_state.page = "intake"
                st.session_state.patient_data = None
                st.rerun()

except Exception as e:
    st.error(f"System Pipeline Error: {e}")

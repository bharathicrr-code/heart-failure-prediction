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

# --- INITIALIZE SESSION STATE FOR PAGE NAVIGATION ---
if "page" not in st.session_state:
    st.session_state.page = "auth"
if "patient_data" not in st.session_state:
    st.session_state.patient_data = None
if "registered_users" not in st.session_state:
    st.session_state.registered_users = {"doctor@hospital.com": "password123"}

# --- DATA ACCESS & BUSINESS LOGIC LAYER ---
@st.cache_data
def load_data():
    return pd.read_csv("heart_failure_clinical_records_dataset.csv")

try:
    df = load_data()
    X = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------------
    # PAGE 1: AUTHENTICATION GATEWAY (Login / Register)
    # ---------------------------------------------------------
    if st.session_state.page == "auth":
        st.title("🏥 CardioShield Portal Login")
        st.write("Welcome to the Clinical Decision Support System. Please log in or register to begin diagnostic processing.")
        
        tab1, tab2 = st.tabs(["🔑 Account Login", "📝 New User Registration"])
        
        with tab1:
            login_email = st.text_input("Email Address", key="login_email")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            if st.button("Secure Login"):
                if login_email in st.session_state.registered_users and st.session_state.registered_users[login_email] == login_pass:
                    st.success("Authentication successful!")
                    st.session_state.page = "intake"
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
                    
        with tab2:
            reg_email = st.text_input("Create Email Username", key="reg_email")
            reg_pass = st.text_input("Create Secure Password", type="password", key="reg_pass")
            if st.button("Register Account"):
                if reg_email and reg_pass:
                    st.session_state.registered_users[reg_email] = reg_pass
                    st.success("Registration complete! You can now log in above.")
                else:
                    st.warning("Please fill out both fields.")

    # ---------------------------------------------------------
    # PAGE 2: CLINICAL VITALS INTAKE FORM
    # ---------------------------------------------------------
    elif st.session_state.page == "intake":
        st.title("🩺 Patient Vitals Entry Desk")
        st.write("Input the clinical values for the current patient to calculate their diagnostic risk framework.")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", int(df['age'].min()), int(df['age'].max()), 70)
            anaemia = st.selectbox("Anaemia Diagnosis", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
            creatinine_phosphokinase = st.slider("Creatinine Phosphokinase (mcg/L)", int(df['creatinine_phosphokinase'].min()), int(df['creatinine_phosphokinase'].max()), 250)
            diabetes = st.selectbox("Diabetes History", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
            ejection_fraction = st.slider("Ejection Fraction (%)", int(df['ejection_fraction'].min()), int(df['ejection_fraction'].max()), 20)
            high_blood_pressure = st.selectbox("Hypertension (High BP)", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
            
        with col2:
            platelets = st.slider("Platelets count (kiloplatelets/mL)", float(df['platelets'].min()), float(df['platelets'].max()), 250000.0, step=1000.0)
            serum_creatinine = st.slider("Serum Creatinine (mg/dL)", float(df['serum_creatinine'].min()), float(df['serum_creatinine'].max()), 2.5, step=0.1)
            serum_sodium = st.slider("Serum Sodium (mEq/L)", int(df['serum_sodium'].min()), int(df['serum_sodium'].max()), 135)
            sex = st.selectbox("Biological Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
            smoking = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Non-smoker" if x==0 else "Smoker")
            time = st.slider("Follow-up Observation Windows (Days)", int(df['time'].min()), int(df['time'].max()), 100)
            
        if st.button("Process Analytics Engine ➡️"):
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
        st.title("🧠 Diagnostic Engine Benchmarking")
        st.write("The software executes model evaluation in the background. Choose the core analytical engine below to evaluate features.")
        
        model_choice = st.selectbox("Choose Core ML Decision Engine", ["Random Forest", "Logistic Regression", "XGBoost", "Decision Tree", "SVM"])
        
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
        
        st.metric(label=f"{model_choice} System Accuracy", value=f"{accuracy * 100:.2f}%")
        
        # Plot Feature Importance
        st.subheader("Algorithmic Feature Weights Analysis")
        importance_values = None
        if model_choice == "Logistic Regression":
            importance_values = np.abs(model.coef_[0])
        elif model_choice in ["Decision Tree", "Random Forest", "XGBoost"]:
            importance_values = model.feature_importances_
            
        if importance_values is not None:
            feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_values}).sort_values(by='Importance', ascending=False)
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis', ax=ax)
            st.pyplot(fig)
        else:
            st.info("Feature variance mapping omitted for non-linear Support Vector configurations.")

        # Save model details for the final page report
        st.session_state.model_choice = model_choice
        st.session_state.accuracy = accuracy
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.feature_names = feature_names
        
        if st.button("Generate Diagnostic Report ➡️"):
            st.session_state.page = "report"
            st.rerun()

    # ---------------------------------------------------------
    # PAGE 4: FINAL MEDICAL REPORT EXPORT (Full Summary View)
    # ---------------------------------------------------------
    elif st.session_state.page == "report":
        st.title("📋 Finalized Patient Diagnosis Summary")
        st.write("Review the complete analytical results and clinical records below before downloading the medical dossier.")
        
        p = st.session_state.patient_data
        user_data = pd.DataFrame([[p[f] for f in st.session_state.feature_names]], columns=st.session_state.feature_names)
        user_data_scaled = st.session_state.scaler.transform(user_data)
        
        prediction = st.session_state.model.predict(user_data_scaled)[0]
        prediction_proba = st.session_state.model.predict_proba(user_data_scaled)[0][1]
        
        # 1. Show Prediction Outcome Prominently
        st.subheader("1. Diagnostic Determination")
        result_text = ""
        if prediction == 1:
            result_text = f"High Risk of Heart Failure Event! (Probability: {prediction_proba * 100:.1f}%)"
            st.error(f"⚠️ Warning Status: {result_text}")
        else:
            result_text = f"Low Risk of Heart Failure Event. (Probability: {prediction_proba * 100:.1f}%)"
            st.success(f"✅ Safe Status: {result_text}")
            
        # 2. Show Comprehensive Table of Input Data
        st.subheader("2. Evaluated Patient Health Metrics")
        
        metrics_display = {
            "Clinical Parameter": [
                "Age", "Biological Sex", "Anaemia History", "Diabetes History", 
                "Hypertension (High BP)", "Smoking Status", "Ejection Fraction", 
                "Serum Creatinine", "Serum Sodium", "Creatinine Phosphokinase", 
                "Platelets Count", "Observation Windows"
            ],
            "Value": [
                f"{p['age']} years old",
                "Male" if p['sex'] == 1 else "Female",
                "Yes" if p['anaemia'] == 1 else "No",
                "Yes" if p['diabetes'] == 1 else "No",
                "Yes" if p['high_blood_pressure'] == 1 else "No",
                "Smoker" if p['smoking'] == 1 else "Non-smoker",
                f"{p['ejection_fraction']}%",
                f"{p['serum_creatinine']} mg/dL",
                f"{p['serum_sodium']} mEq/L",
                f"{p['creatinine_phosphokinase']} mcg/L",
                f"{p['platelets']} kiloplatelets/mL",
                f"{p['time']} Days"
            ]
        }
        st.table(pd.DataFrame(metrics_display))
        
        # 3. Show System Architecture Details
        st.subheader("3. Technical Evaluation Summary")
        st.markdown(f"""
        * **Decision Core Engine:** {st.session_state.model_choice}
        * **Configured Engine Testing Accuracy:** {st.session_state.accuracy * 100:.2f}%
        """)
        
        st.markdown("---")
        
        # 4. Construct the Final Clean Export Text block
        report_data = f"""======================================================
HEART FAILURE CLINICAL ASSESSMENT RECORD
======================================================
[DIAGNOSTIC CORE METRICS]
* Decision Engine Engine: {st.session_state.model_choice}
* Core System Baseline Accuracy: {st.session_state.accuracy*100:.2f}%
* Analytical Evaluation: {result_text}

[PATIENT RECORD ENTRY SUMMARY]
- Age: {p['age']}
- Sex: {'Male' if p['sex'] == 1 else 'Female'}
- Anaemia Present: {'Yes' if p['anaemia'] == 1 else 'No'}
- Diabetes Present: {'Yes' if p['diabetes'] == 1 else 'No'}
- High Blood Pressure: {'Yes' if p['high_blood_pressure'] == 1 else 'No'}
- Smoking Profile: {'Smoker' if p['smoking'] == 1 else 'Non-smoker'}
- Ejection Fraction Capacity: {p['ejection_fraction']}%
- Serum Creatinine Value: {p['serum_creatinine']} mg/dL
- Serum Sodium Level: {p['serum_sodium']} mEq/L
- Creatinine Phosphokinase Level: {p['creatinine_phosphokinase']} mcg/L
- Total Platelets Volume: {p['platelets']} kiloplatelets/mL
- Follow-up Active Window: {p['time']} Days
======================================================
Report compiled via deployed CardioShield CDSS Production Instance.
"""
        
        # Action Buttons side by side at the very bottom
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.download_button(
                label="📥 Download Official Clinical Report (.TXT)", 
                data=report_data, 
                file_name="patient_clinical_report.txt", 
                mime="text/plain"
            )
        with col_btn2:
            if st.button("🔄 Restart Process for New Patient"):
                st.session_state.page = "intake"
                st.session_state.patient_data = None
                st.rerun()

except Exception as e:
    st.error(f"System Pipeline Error: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import os

# Set full-screen wide layout
st.set_page_config(page_title="Heart Failure Prediction System", layout="wide")

# --- CUSTOM CSS INTERFACE LAYER ---
st.markdown("""
    <style>
    .block-container { padding-top: 2rem !important; }
    .nav-logo {
        font-size: 34px !important; font-weight: 800 !important;
        color: #1a3a4b !important; display: inline-block; vertical-align: middle;
    }
    .custom-navbar { text-align: right; padding-top: 15px; }
    .custom-nav-link {
        font-size: 22px !important; font-weight: 700 !important;
        color: #555555 !important; text-decoration: none !important;
        margin-left: 45px !important; display: inline-block;
    }
    .custom-nav-link:hover { color: #0c5460 !important; }
    .custom-nav-active {
        color: #0c5460 !important; border-bottom: 3px solid #0c5460 !important; padding-bottom: 2px;
    }
    .hero-title { font-size: 46px !important; font-weight: 800 !important; color: #1a3a4b !important; }
    .hero-subtitle { font-size: 30px !important; font-weight: 700 !important; color: #0c5460 !important; }
    .hero-body { font-size: 23px !important; line-height: 1.8 !important; color: #2b2b2b !important; }
    div.stButton > button, div.stDownloadButton > button {
        background-color: #0c5460 !important; color: white !important;
        border-radius: 6px !important; padding: 16px 40px !important;
        font-size: 24px !important; font-weight: 700 !important;
        width: 380px !important; min-height: 65px !important;
        display: block !important; margin: 40px auto !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- STATE LIFECYCLE MANAGEMENT ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

query_params = st.query_params
if "nav" in query_params:
    st.session_state.current_nav = query_params["nav"]
elif "current_nav" not in st.session_state:
    st.session_state.current_nav = "HOME"

if "patient_data" not in st.session_state:
    st.session_state.patient_data = None
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

# --- SAFE DATASET LOADER ENGINE ---
CSV_FILE = "heart_failure_clinical_records_dataset.csv"

if not os.path.exists(CSV_FILE):
    st.error(f"⚠️ **Deployment File Error:** The dataset file `{CSV_FILE}` was not found in your GitHub repository root.")
    st.info("Please verify that the dataset file is uploaded directly to your repository with this exact name.")
    st.stop()

@st.cache_data
def process_ml_data():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    df = pd.read_csv(CSV_FILE)
    X = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, df

# Safely compute or gracefully output system diagnostics 
try:
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, df = process_ml_data()
except Exception as internal_err:
    st.error(f"💥 Core Data Initialization Failed: {internal_err}")
    st.stop()


# -----------------------------------------------------------------
# MODULE 1: LOGIN MODULE INTERFACE
# -----------------------------------------------------------------
if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col_l2, _ = st.columns([1, 2, 1])
    with col_l2:
        st.markdown("<h2 style='text-align: center; color: #1a3a4b;'>🔒 System Authentication Gateway</h2>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username / Clinician ID Key", value="admin")
            password = st.text_input("Secure Password Key", type="password", value="password")
            if st.form_submit_button("Authenticate Access Verification"):
                if username == "admin" and password == "password":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
    st.stop()

# --- TOP NAVIGATION LINK BAR ---
col_logo, col_menu = st.columns([6, 6])
with col_logo:
    st.markdown('<div class="nav-logo">🩺 Heart Failure Prediction System</div>', unsafe_allow_html=True)
with col_menu:
    h_class = "custom-nav-link custom-nav-active" if st.session_state.current_nav == "HOME" else "custom-nav-link"
    i_class = "custom-nav-link custom-nav-active" if st.session_state.current_nav == "INTAKE" else "custom-nav-link"
    m_class = "custom-nav-link custom-nav-active" if st.session_state.current_nav == "EVALUATION" else "custom-nav-link"
    r_class = "custom-nav-link custom-nav-active" if st.session_state.current_nav == "REPORT" else "custom-nav-link"
    a_class = "custom-nav-link custom-nav-active" if st.session_state.current_nav == "ABOUT" else "custom-nav-link"
    
    st.markdown(f"""
        <div class="custom-navbar">
            <a href="?nav=HOME" target="_self" class="{h_class}">HOME</a>
            <a href="?nav=INTAKE" target="_self" class="{i_class}">FORM</a>
            <a href="?nav=EVALUATION" target="_self" class="{m_class}">MODELS</a>
            <a href="?nav=REPORT" target="_self" class="{r_class}">REPORT</a>
            <a href="?nav=ABOUT" target="_self" class="{a_class}">ABOUT</a>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------
# MODULE 2: HOME VIEW
# ---------------------------------------------------------
if st.session_state.current_nav == "HOME":
    col_info, col_graphic = st.columns([11, 9])
    with col_info:
        st.markdown('<div class="hero-title">CardioShield Predictive Clinical Portal</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-subtitle">Decision Support Engine Deployment Platform</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-body">This analytical clinical decision support framework leverages advanced optimization pipelines to assist health practitioners with objective risk stratification.</div>', unsafe_allow_html=True)
        if st.button("Initialize Patient Entry Module ➡️"):
            st.query_params["nav"] = "INTAKE"
            st.session_state.current_nav = "INTAKE"
            st.rerun()
            
    with col_graphic:
        st.markdown("""
            <div style="text-align:center; padding-top:20px;">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#0c5460" stroke-width="1.2" style="width:75%; max-width:275px;">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                </svg>
                <p style="color:#555555; font-size:18px; font-weight:700; margin-top:25px;">Cardiovascular Diagnostics Engine Output</p>
            </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# MODULE 3: PATIENT DETAILS ENTRY MODULE
# ---------------------------------------------------------
elif st.session_state.current_nav == "INTAKE":
    st.subheader("Patient Administrative Records & Vector Matrix")
    patient_name = st.text_input("Patient Full Name", value="John Doe")
    patient_id = st.text_input("Hospital Reference ID (Ref-ID)", value="PT-89422")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Patient Age Parameters", int(df['age'].min()), int(df['age'].max()), 65)
        anaemia = st.selectbox("Anaemia Diagnostic Level", [0, 1], format_func=lambda x: "Negative" if x==0 else "Positive")
        creatinine_phosphokinase = st.slider("CPK Level (mcg/L)", int(df['creatinine_phosphokinase'].min()), int(df['creatinine_phosphokinase'].max()), 250)
        diabetes = st.selectbox("Diabetes History Matrix", [0, 1], format_func=lambda x: "No History" if x==0 else "Diabetic")
        ejection_fraction = st.slider("Ejection Fraction (%)", int(df['ejection_fraction'].min()), int(df['ejection_fraction'].max()), 25)
        high_blood_pressure = st.selectbox("Hypertensive Vascular History", [0, 1], format_func=lambda x: "Normal" if x==0 else "Hypertensive")
    with col2:
        platelets = st.slider("Platelets Diagnostic Count", float(df['platelets'].min()), float(df['platelets'].max()), 250000.0, step=1000.0)
        serum_creatinine = st.slider("Serum Creatinine Volume (mg/dL)", float(df['serum_creatinine'].min()), float(df['serum_creatinine'].max()), 2.1, step=0.1)
        serum_sodium = st.slider("Serum Sodium Levels (mEq/L)", int(df['serum_sodium'].min()), int(df['serum_sodium'].max()), 136)
        sex = st.selectbox("Biological Sex Profiling", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        smoking = st.selectbox("Tobacco Profile", [0, 1], format_func=lambda x: "Non-smoker" if x==0 else "Active Smoker")
        time = st.slider("Follow-up Observation Window (Days)", int(df['time'].min()), int(df['time'].max()), 120)
        
    if st.button("Submit Patient Details ➡️"):
        st.session_state.patient_data = {
            "patient_name": patient_name, "patient_id": patient_id, "age": age, "anaemia": anaemia,
            "creatinine_phosphokinase": creatinine_phosphokinase, "diabetes": diabetes, "ejection_fraction": ejection_fraction,
            "high_blood_pressure": high_blood_pressure, "platelets": platelets, "serum_creatinine": serum_creatinine,
            "serum_sodium": serum_sodium, "sex": sex, "smoking": smoking, "time": time
        }
        st.session_state.prediction_made = False
        st.query_params["nav"] = "EVALUATION"
        st.session_state.current_nav = "EVALUATION"
        st.rerun()

# ---------------------------------------------------------
# MODULE 4: PREDICTION MODULE (Completely Error Insulated)
# ---------------------------------------------------------
elif st.session_state.current_nav == "EVALUATION":
    st.subheader("Machine Learning Prediction Engine Module")
    model_choice = st.selectbox("Select Active Analytics Engine", ["Random Forest", "Logistic Regression", "XGBoost", "Decision Tree", "SVM"])
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    import matplotlib.pyplot as plt
    import seaborn as sns

    if model_choice == "Logistic Regression": model = LogisticRegression()
    elif model_choice == "Decision Tree": model = DecisionTreeClassifier(max_depth=3)
    elif model_choice == "Random Forest": model = RandomForestClassifier(n_estimators=50, max_depth=3)
    elif model_choice == "SVM": model = SVC(probability=True)
    elif model_choice == "XGBoost": model = XGBClassifier(max_depth=3)
        
    model.fit(X_train_scaled, y_train)
    accuracy = model.score(X_test_scaled, y_test)
    st.metric(label=f"{model_choice} Baseline Test Accuracy", value=f"{accuracy * 100:.2f}%")
    
    importance_values = None
    if model_choice == "Logistic Regression": importance_values = np.abs(model.coef_[0])
    elif model_choice in ["Decision Tree", "Random Forest", "XGBoost"]: importance_values = model.feature_importances_
        
    if importance_values is not None:
        try:
            feat_imp_df = pd.DataFrame({'Feature Vector': feature_names, 'Weight Importance': importance_values}).sort_values(by='Weight Importance', ascending=False)
            fig, ax = plt.subplots(figsize=(10, 3.5))
            sns.barplot(x='Weight Importance', y='Feature Vector', data=feat_imp_df, palette='viridis', hue='Feature Vector', legend=False, ax=ax)
            st.pyplot(fig)
        except Exception as plot_err:
            st.warning(f"Feature chart plotting skipped: {plot_err}")
        
    st.session_state.model_choice = model_choice
    st.session_state.accuracy = accuracy
    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.feature_names = feature_names

    if st.button("Execute Analysis Prediction ⚡"):
        if st.session_state.patient_data is None:
            st.warning("⚠️ Please input data on the Form page before evaluating.")
        else:
            st.session_state.prediction_made = True
            st.success("Analysis calculation compiled successfully.")

# ---------------------------------------------------------
# MODULE 5: REPORTS MODULE
# ---------------------------------------------------------
elif st.session_state.current_nav == "REPORT":
    st.subheader("Reports Generation & Dossier Export Module")
    if st.session_state.patient_data is None or not st.session_state.prediction_made:
        st.warning("Incomplete tracking data. Please complete the Intake Form and execute predictions.")
    else:
        p = st.session_state.patient_data
        user_data = pd.DataFrame([[p[f] for f in st.session_state.feature_names]], columns=st.session_state.feature_names)
        user_data_scaled = st.session_state.scaler.transform(user_data)
        
        prediction = st.session_state.model.predict(user_data_scaled)[0]
        prediction_proba = st.session_state.model.predict_proba(user_data_scaled)[0][1]
        
        if prediction == 1:
            st.error(f"⚠️ HIGH RISK CLINICAL STANDING (Probability: {prediction_proba * 100:.1f}%)")
        else:
            st.success(f"✅ CLEAR REGULAR HEALTH STANDING (Probability: {prediction_proba * 100:.1f}%)")
            
        st.table(pd.DataFrame({
            "Monitored Attribute": ["Patient Name", "Hospital ID", "Age", "Engine Backbone"],
            "Value": [p['patient_name'], p['patient_id'], p['age'], st.session_state.model_choice]
        }))

# ---------------------------------------------------------
# MODULE 6: ABOUT MODULE
# ---------------------------------------------------------
elif st.session_state.current_nav == "ABOUT":
    st.subheader("Help / About Module - Diagnostic Specifications")
    st.markdown("<div class='hero-body'>This software solution acts as an architectural verification pipeline matching presentation specifications. Includes clean data routing checkpoints, feature mapping modules, and diagnostic reporting matrix engines.</div>", unsafe_allow_html=True)

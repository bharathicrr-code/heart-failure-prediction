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

# Set up the title of the web app
st.title("Heart Failure Prediction Dashboard")
st.write("This app uses 5 different Machine Learning models to predict heart failure risk.")

# 1. Load Data
@st.cache_data
def load_data():
    data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    return data

try:
    df = load_data()
    st.success("Dataset successfully loaded!")
    
    # Show a small preview of the data
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # 2. Preprocess Data
    X = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']
    
    # Save feature names for later matching
    feature_names = X.columns.tolist()
    
    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Sidebar Model Selection
    st.sidebar.header("Model Settings")
    model_choice = st.sidebar.selectbox(
        "Choose a Machine Learning Model",
        ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "XGBoost"]
    )
    
    # Initialize the selected model
    if model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=3)
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=50, max_depth=3)
    elif model_choice == "SVM":
        model = SVC(probability=True)  # Enable probability calculation
    elif model_choice == "XGBoost":
        model = XGBClassifier(max_depth=3)
        
    # Train the model instantly
    model.fit(X_train_scaled, y_train)
    accuracy = model.score(X_test_scaled, y_test)
    
    # Display results
    st.subheader(f"Model Performance: {model_choice}")
    st.metric(label="Testing Accuracy", value=f"{accuracy * 100:.2f}%")
    
    # 4. Interactive Patient Risk Prediction Section
    st.markdown("---")
    st.subheader("Patient Risk Predictor")
    st.write("Adjust the values below to evaluate a patient's health metrics:")
    
    # Creating individual inputs for the clinical features
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", int(df['age'].min()), int(df['age'].max()), 60)
        anaemia = st.selectbox("Anaemia (Decrease in red blood cells)", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        creatinine_phosphokinase = st.slider("Creatinine Phosphokinase (mcg/L)", int(df['creatinine_phosphokinase'].min()), int(df['creatinine_phosphokinase'].max()), 250)
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        ejection_fraction = st.slider("Ejection Fraction (Percentage of blood leaving heart)", int(df['ejection_fraction'].min()), int(df['ejection_fraction'].max()), 35)
        high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        
    with col2:
        platelets = st.slider("Platelets (kiloplatelets/mL)", float(df['platelets'].min()), float(df['platelets'].max()), 250000.0, step=1000.0)
        serum_creatinine = st.slider("Serum Creatinine (mg/dL)", float(df['serum_creatinine'].min()), float(df['serum_creatinine'].max()), 1.1, step=0.1)
        serum_sodium = st.slider("Serum Sodium (mEq/L)", int(df['serum_sodium'].min()), int(df['serum_sodium'].max()), 135)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        smoking = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Non-smoker" if x==0 else "Smoker")
        time = st.slider("Follow-up Period (Days)", int(df['time'].min()), int(df['time'].max()), 100)

    # Put all user variables into a DataFrame structured exactly like the original training data
    user_data = pd.DataFrame([[
        age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
        high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time
    ]], columns=feature_names)
    
    # Scale the patient's individual input using the fitted training scaler
    user_data_scaled = scaler.transform(user_data)
    
    # Predict using the active model
    prediction = model.predict(user_data_scaled)[0]
    prediction_proba = model.predict_proba(user_data_scaled)[0][1]
    
    # Show prediction output neatly
    st.markdown("### Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Failure Event! (Probability: {prediction_proba * 100:.1f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Failure Event. (Probability: {prediction_proba * 100:.1f}%)")
        
except Exception as e:
    st.error(f"Error executing dashboard pipeline: {e}")

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
    # Reads the dataset we uploaded to your repository
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
    
    # Initialize the selected model with simple default settings
    if model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=3)
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=50, max_depth=3)
    elif model_choice == "SVM":
        model = SVC()
    elif model_choice == "XGBoost":
        model = XGBClassifier(max_depth=3)
        
    # Train the model instantly
    model.fit(X_train_scaled, y_train)
    accuracy = model.score(X_test_scaled, y_test)
    
    # Display results
    st.subheader(f"Model Performance: {model_choice}")
    st.metric(label="Testing Accuracy", value=f"{accuracy * 100:.2f}%")
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please verify that your dataset file name matches 'heart_failure_clinical_records_dataset.csv' exactly in GitHub.")

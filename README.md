# Heart Failure Prediction using Optimized Machine Learning Algorithms

## Project Overview

This project predicts the likelihood of heart failure using optimized machine learning algorithms based on clinical patient data.

The system compares five supervised machine learning models and identifies the best-performing algorithm for heart failure prediction.

The project was developed as part of the M.Tech (Computer Science and Engineering) dissertation.

---

## Objectives

- Predict heart failure risk using clinical attributes
- Compare multiple machine learning algorithms
- Identify the best-performing model
- Provide an easy-to-use web application using Streamlit

---

## Dataset

- Heart Failure Clinical Records Dataset
- Source: Kaggle
- Number of Records: 299
- Clinical Features: 12
- Target Variable:
  - DEATH_EVENT
    - 0 = No Death Event
    - 1 = Death Event

---

## Machine Learning Algorithms

The following algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

Among these, **Random Forest achieved the best overall performance** and was selected as the final prediction model.

---

## Technologies Used

- Python
- Google Colab
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Streamlit
- GitHub

---

## Project Structure

```
heart-failure-prediction/
│
├── app.py
├── requirements.txt
├── heart_failure_clinical_records_dataset.csv
├── model.pkl (if applicable)
├── notebook.ipynb
└── README.md
```

---

## Performance Summary

| Algorithm | Accuracy |
|-----------|----------|
| Random Forest | 83.33% |
| Logistic Regression | 81.67% |
| XGBoost | 81.67% |
| Support Vector Machine | 76.67% |
| Decision Tree | 73.33% |

---

## Streamlit Web Application

The application allows users to:

- Enter patient clinical information
- Select a machine learning algorithm
- Predict heart failure risk
- View prediction results instantly

---

## Live Demo

https://heart-failure-prediction-uxxv72ymf96awl7hw2m8q3.streamlit.app/

---

## GitHub Repository

https://github.com/bharathicrr-code/heart-failure-prediction

---

## Author

**Nersu Bharathi**

M.Tech (Computer Science and Engineering)

Eluru College of Engineering and Technology

Academic Year: 2025–2026# heart-failure-prediction

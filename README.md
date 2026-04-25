# Loan Approval Prediction System

## Project Description
This project is a Machine Learning-based system that predicts whether a loan application will be approved or rejected based on applicant data such as income, loan amount, and other financial features.

It includes a trained classification model, a full preprocessing pipeline, and an interactive web application built using Streamlit.

---

## Key Features
- Predict loan approval status (Approved / Rejected)
- Display prediction probabilities
- Data preprocessing:
  - Outlier handling (IQR)
  - Categorical encoding
  - Log transformation
  - Feature scaling (StandardScaler)
- Interactive and user-friendly UI

---

## Machine Learning Workflow
1. Data Loading  
2. Data Preprocessing  
3. Feature Engineering  
4. Model Training & Selection  
5. Model Saving (`.pkl`)  
6. Deployment using Streamlit  

---

## Technologies Used
- Python  
- Pandas & NumPy  
- Scikit-learn  
- Streamlit  
- Joblib  

---

## How to Run
```bash
pip install -r requirements.txt
streamlit run ui_classification.py

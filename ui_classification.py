import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="Loan Prediction",
    page_icon="💰",
    layout="centered"
)

# ==============================
# Simple CSS
# ==============================
st.markdown("""
    <style>
        .result-approved {
            background-color: #d4edda;
            border: 2px solid #28a745;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            color: #155724;
            font-size: 1.3rem;
            font-weight: bold;
        }
        .result-rejected {
            background-color: #f8d7da;
            border: 2px solid #dc3545;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            color: #721c24;
            font-size: 1.3rem;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Load Data & Model
# ==============================
csv_path = r'C:\Users\DELL\OneDrive\Desktop\DEPI\My_Projects\final project\final project\classification\archive 22\loan_data.csv'
model_path = 'best_model.pkl'

if not os.path.exists(csv_path):
    st.error(f"❌ CSV file not found: {csv_path}")
    st.stop()

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found: {model_path}")
    st.stop()

raw_data = pd.read_csv(csv_path)
model = joblib.load(model_path)

# ==============================
# Preprocessing (same as training)
# ==============================
data = raw_data.copy()

# Outlier clipping
Q1 = data['loan_amnt'].quantile(0.25)
Q3 = data['loan_amnt'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data['loan_amnt'] = data['loan_amnt'].clip(lower_bound, upper_bound)

# Categorical encoding (.cat.codes — same as training)
categorical_cols = data.select_dtypes(include=[np.object_]).columns.tolist()
cat_mappings = {}
for col in categorical_cols:
    data[col] = data[col].astype('category')
    cat_mappings[col] = dict(enumerate(data[col].cat.categories))
    data[col] = data[col].cat.codes

# Log transformation
numerical_cols = raw_data.select_dtypes(include=[np.int64, np.float64]).columns.tolist()
log_transformed_cols = []
for col in numerical_cols:
    if col not in data.columns or data[col].nunique() <= 2:
        continue
    if abs(data[col].skew()) > 1 and (data[col].dropna() >= 0).all():
        data[col] = np.log1p(data[col])
        log_transformed_cols.append(col)

feature_names = data.drop("loan_status", axis=1).columns.tolist()

# Scaler
scaler = StandardScaler()
scaler.fit(data[feature_names])

# ==============================
# Title
# ==============================
st.title("💰 Loan Approval Prediction")
st.write("Fill in the applicant's details and click **Predict**.")
st.divider()

# ==============================
# Input Fields (simple columns)
# ==============================
encoded_inputs = []
raw_inputs = {}

col1, col2 = st.columns(2)
fields = list(feature_names)

for i, col in enumerate(fields):
    container = col1 if i % 2 == 0 else col2
    original = raw_data[col]

    with container:
        if original.dtype == object:
            options = sorted(original.unique().tolist())
            reverse_map = {v: k for k, v in cat_mappings[col].items()}
            selected = st.selectbox(col, options)
            raw_inputs[col] = selected
            encoded_inputs.append(reverse_map.get(selected, 0))
        else:
            min_val = float(original.min())
            max_val = float(original.max())
            mid_val = float((min_val + max_val) / 2)
            value = st.number_input(col, min_value=min_val, max_value=max_val, value=mid_val)
            raw_inputs[col] = value

            processed = value
            if col == 'loan_amnt':
                processed = np.clip(processed, lower_bound, upper_bound)
            if col in log_transformed_cols:
                processed = np.log1p(processed)
            encoded_inputs.append(processed)

st.divider()

# ==============================
# Predict Button
# ==============================
if st.button("🔍 Predict", use_container_width=True):
    input_df = pd.DataFrame([encoded_inputs], columns=feature_names)
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)

    try:
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.markdown('<div class="result-approved">✅ Loan Approved</div>', unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown('<div class="result-rejected">❌ Loan Rejected</div>', unsafe_allow_html=True)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_scaled)[0]
            st.write("")
            c1, c2 = st.columns(2)
            c1.metric("Approval Probability", f"{proba[1]*100:.1f}%")
            c2.metric("Rejection Probability", f"{proba[0]*100:.1f}%")

    except Exception as e:
        st.error(f"Error: {e}")
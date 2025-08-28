import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("loan_approval_pipeline.joblib")

st.title("Loan Approval Prediction")

# =====================
# User Inputs
# =====================
person_age = st.number_input("Age", min_value=18, max_value=100, step=1)
person_income = st.number_input("Annual Income", min_value=0, step=100)
person_gender = st.selectbox("Gender", ["male", "female"])
person_education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

loan_amnt = st.number_input("Loan Amount", min_value=500, step=500)
loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, step=0.1)
loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, step=0.01)
loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])

previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults on File", ["Y", "N"])

# =====================
# Prepare input DataFrame
# =====================
user_input = pd.DataFrame({
    "person_age": [person_age],
    "person_income": [person_income],
    "person_gender": [person_gender],
    "person_education": [person_education],
    "person_home_ownership": [person_home_ownership],
    "loan_amnt": [loan_amnt],
    "loan_int_rate": [loan_int_rate],
    "loan_percent_income": [loan_percent_income],
    "loan_intent": [loan_intent],
    "previous_loan_defaults_on_file": [previous_loan_defaults_on_file]
})

# =====================
# Add engineered features (must match training pipeline!)
# =====================
user_input["income_to_loan_ratio"] = user_input["person_income"] / user_input["loan_amnt"]
user_input["debt_burden"] = user_input["loan_percent_income"] * user_input["loan_int_rate"]

# =====================
# Prediction
# =====================
if st.button("Predict Loan Approval"):
    try:
        prediction = model.predict(user_input)[0]
        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Denied")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

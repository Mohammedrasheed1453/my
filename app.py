import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load models and features
model1 = joblib.load("model1_nb.pkl")
model2 = joblib.load("model2_gb.pkl")
features1 = joblib.load("features1.pkl")
features2 = joblib.load("features2.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("🫀 Heart Disease Risk Predictor")
st.markdown("### Enter the following details to assess heart disease risk:")

with st.form("heart_form"):
    st.subheader("Medical Data")
    medical_inputs = []
    for feature in features1:
        val = st.number_input(f"{feature}", step=0.1, format="%.2f")
        medical_inputs.append(val)

    st.subheader("Lifestyle & Symptoms Data")
    lifestyle_inputs = []
    for feature in features2:
        val = st.radio(f"{feature}", [0, 1], horizontal=True)
        lifestyle_inputs.append(val)

    submitted = st.form_submit_button("Predict")


if submitted:
    input_values = medical_inputs + lifestyle_inputs

    # Convert to DataFrame
    medical_df = pd.DataFrame([input_values[:len(features1)]], columns=features1)
    lifestyle_df = pd.DataFrame([input_values[len(features1):]], columns=features2)

    prob1 = model1.predict_proba(medical_df)[0][1]
    prob2 = model2.predict_proba(lifestyle_df)[0][1]

    # Rule-based prediction logic
    if prob1 >= 0.95:
        final_prediction = 1
    elif prob1 >= 0.8 and prob2 >= 0.6:
        final_prediction = 1
    elif prob1 >= 0.7 and prob2 >= 0.8:
        final_prediction = 1
    elif prob2 >= 0.95:
        final_prediction = 1
    else:
        final_prediction = 0

    if final_prediction == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")

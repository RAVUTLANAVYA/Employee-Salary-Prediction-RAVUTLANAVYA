import streamlit as st
import numpy as np
import joblib  # Make sure this is installed and imported

st.title("Salary Prediction App")

st.divider()

st.write("With this app, you can get an estimate for the salaries of the company employees.")

# Input from user
years = st.number_input("Years of Experience", value=1, step=1, min_value=0)
jobrate = st.number_input("Job Rate (e.g., performance rating)", value=3.5, step=0.5, min_value=0.0)

# Load the model
try:
    model = joblib.load("linearmodel.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'linearmodel.pkl' is in the correct path.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# Prediction
if st.button("Predict Salary"):
    features = np.array([[years, jobrate]])
    salary = model.predict(features)
    st.success(f"Estimated Salary: ₹{salary[0]:,.2f}")

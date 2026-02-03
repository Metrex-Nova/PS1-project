import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model and feature columns
model = pickle.load(open("interest_rate_model.pkl", "rb"))
columns = pickle.load(open("feature_columns.pkl", "rb"))

st.title("📈 Interest Rate Predictor")

st.write("Select or enter the required features below to get the predicted loan interest rate:")

# Dropdown options for categorical variables
state_options = ['South Dakota', 'Indiana', 'Montana', 'Mississippi', 'Others']
loan_purpose_options = [
    'Educational', 'Home Improvement', 'Moving', 'House', 'Debt Consolidation',
    'Credit Card', 'Major Purchase', 'Small Business', 'Other Reason'
]
home_ownership_options = ['Rent', 'Other']

# Streamlit dropdowns
selected_state = st.selectbox("Select State", state_options)
selected_purpose = st.selectbox("Select Loan Purpose", loan_purpose_options)
selected_home = st.selectbox("Select Home Ownership Type", home_ownership_options)

# Numeric feature input
other_cols = [
    col for col in columns
    if not col.startswith('State_') and
       not col.startswith('Loan_Purpose_') and
       not col.startswith('Home_Ownership_')
]

user_input = {}
for col in other_cols:
    user_input[col] = st.number_input(col, value=0.0, format="%.4f")

# Build feature vector
input_vector = []
for col in columns:
    if col.startswith("State_"):
        state_name = col.replace("State_", "")
        input_vector.append(1 if state_name.upper() == selected_state.upper() else 0)
    elif col.startswith("Loan_Purpose_"):
        purpose_name = col.replace("Loan_Purpose_", "")
        input_vector.append(1 if purpose_name.lower() == selected_purpose.lower() else 0)
    elif col.startswith("Home_Ownership_"):
        home_status = col.replace("Home_Ownership_", "")
        input_vector.append(1 if home_status.upper() == selected_home.upper() else 0)
    else:
        input_vector.append(user_input[col])

# Prediction
if st.button("Predict Interest Rate"):
    input_df = pd.DataFrame([input_vector], columns=columns)
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Interest Rate: {round(prediction, 2)}%")
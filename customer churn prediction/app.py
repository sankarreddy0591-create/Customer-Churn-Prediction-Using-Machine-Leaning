
# Customer Churn Prediction App

# import necessary libraries
import streamlit as st
import pickle
import pandas as pd


# load model

with open("models/customer_churn_model.pkl", "rb") as file:
    model=pickle.load(file)

# load scaler

with open("models/scaler.pkl", "rb") as file:
    scaler=pickle.load(file)

# Application Title

st.title("Customer Churn Prediction")
st.write("Enter Customer Details to Predict Whether The Customer is likely To Churn.")


# user inputs

age=st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

gender=st.selectbox(
    "Gender", 
    ["Male", "Female"]
)

tenure=st.number_input(
    "Tenure",
    min_value=0.0,
    max_value=100.0,
    value=12.0
)

monthly_charges=st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=1000.0,
    value=50.0,
    step=1.0
)

# convert Gender

gender_values=1 if gender=='Female' else 0

# prediction

if st.button("Predict Churn"):
    # create input data
    input_data=pd.DataFrame(
        [[
            age,
            gender_values,
            tenure,
            monthly_charges
        ]],
        columns=[
            "Age",
            "Gender",
            "Tenure",
            "MonthlyCharges"
        ]
    )

    # scaler input
    input_scaled=scaler.transform(input_data)

    # make prediction

    prediction=model.predict(input_scaled)

    # Display result    

    if prediction[0]==1:
        st.error("Customer is likely to Churn.")
    else:
        st.success("Customer is likely to stay")


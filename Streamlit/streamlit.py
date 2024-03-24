import streamlit as st
import pandas as pd
import  boto3
import io
import requests
import os

API_ENDPOINT = os.getenv('API_ENDPOINT')

def make_prediction(input_data):
    try:
        response = requests.post(API_ENDPOINT,json=input_data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Error occurred: {e}")

s3_client = boto3.client('s3')

bucket_name = os.getenv('BUCKET_NAME')
object_key = 'Churn_Modelling.csv'
response = s3_client.get_object(Bucket=bucket_name, Key=object_key)


content = response['Body'].read().decode('utf-8')
df = pd.read_csv(io.StringIO(content))

unique_values = df['Geography'].unique()

st.title('Customer Churn Analysis')
col1, col2 = st.columns(2)

# Column 1
with col1:
    Geography = st.selectbox("Country", unique_values)
    Credit = st.number_input("Credit Score of the customer", value=None, placeholder="Type a number...")
    Gender = st.selectbox("Gender", ['Male', 'Female'])
    Age = st.number_input("Age of the customer", value=None, placeholder="Type a number...")
    Tenure = st.number_input("Tenure of the customer", value=None, placeholder="Type a number...")

# Column 2
with col2:
    Balance = st.number_input("Balance of the customer", value=None, placeholder="Type a number...")
    Products = st.number_input("Products of the customer", value=None, placeholder="Type a number...")
    Credits = st.selectbox("Does this Customer has the Credit Card?", ['Yes', 'No'])
    Active = st.selectbox("Is this customer an Active Member?", ['Yes', 'No'])
    Salary = st.number_input("Estimated Salary of the customer", value=None, placeholder="Type a number...")


card= 1 if Credits == 'Yes' else 0
ac= 1 if Active == 'Yes' else 0
input_data = {"Credit":Credit, "Geography": Geography,"Gender": Gender,"Age": Age,"Tenure": Tenure,"Balance":Balance,"Products":Products,"Credits":card,"Active":ac,"Salary":Salary}
if st.button('Predict Churn'):
    with st.spinner('Predicting...'):
        prediction = make_prediction(input_data)
    st.subheader("Prediction Result")
    churn = float(prediction['data'])
    if churn > 0.5 :
        result=churn*100
        st.success(f"There is {round(result, 2)}% chance that the customer will leave the bank")
    else:
        result=(1-churn)*100
        st.success(f"There is {round(result, 2)}% chance  that customer will stay in the bank")

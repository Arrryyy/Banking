# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and feature names
model_path = "models/svm_model.pkl"
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

svm_model = model_data['model']
feature_names = model_data['features']

st.title("💼 Bank Marketing Term Deposit Prediction")

st.write("Fill in the client details below to predict if they will subscribe to a term deposit:")

# Form Inputs
age = st.slider('Age', 18, 100, 30)
job = st.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid',
                           'management', 'retired', 'self-employed', 'services',
                           'student', 'technician', 'unemployed', 'unknown'])
marital = st.selectbox('Marital Status', ['married', 'single', 'divorced', 'unknown'])
education = st.selectbox('Education', ['basic.4y', 'basic.6y', 'basic.9y',
                                       'high.school', 'illiterate', 'professional.course',
                                       'university.degree', 'unknown'])
default = st.selectbox('Has Credit in Default?', ['no', 'yes', 'unknown'])
housing = st.selectbox('Has Housing Loan?', ['no', 'yes', 'unknown'])
loan = st.selectbox('Has Personal Loan?', ['no', 'yes', 'unknown'])
contact = st.selectbox('Contact Communication Type', ['cellular', 'telephone'])
month = st.selectbox('Last Contact Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                            'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
day_of_week = st.selectbox('Day of Week of Last Contact', ['mon', 'tue', 'wed', 'thu', 'fri'])
campaign = st.slider('Number of contacts during campaign', 1, 50, 1)
pdays_contacted = st.selectbox('Previously contacted?', ['No', 'Yes'])
previous = st.slider('Number of previous contacts', 0, 10, 0)
poutcome = st.selectbox('Outcome of Previous Campaign', ['failure', 'nonexistent', 'success'])
emp_var_rate = st.number_input('Employment Variation Rate', value=0.0)
cons_price_idx = st.number_input('Consumer Price Index', value=93.0)
cons_conf_idx = st.number_input('Consumer Confidence Index', value=-40.0)
euribor3m = st.number_input('Euribor 3 Month Rate', value=4.0)
nr_employed = st.number_input('Number of Employees', value=5000.0)

# Prepare input for model
def preprocess_input():
    input_data = {}

    # Add scaled numeric values (approximate scaling for demo purposes)
    input_data['age'] = (age - 40) / 10
    input_data['campaign'] = (campaign - 1) / 10
    input_data['previous'] = (previous - 0) / 5
    input_data['emp.var.rate'] = (emp_var_rate + 2) / 2
    input_data['cons.price.idx'] = (cons_price_idx - 92) / 2
    input_data['cons.conf.idx'] = (cons_conf_idx + 50) / 10
    input_data['euribor3m'] = (euribor3m - 1) / 3
    input_data['nr.employed'] = (nr_employed - 5000) / 500

    # One-hot encode categorical values manually
    all_possible_categories = [
        # Jobs
        'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management',
        'job_retired', 'job_self-employed', 'job_services', 'job_student',
        'job_technician', 'job_unemployed', 'job_unknown',

        # Marital
        'marital_divorced', 'marital_single', 'marital_unknown',

        # Education
        'education_basic.6y', 'education_basic.9y', 'education_high.school',
        'education_illiterate', 'education_professional.course', 'education_university.degree',
        'education_unknown',

        # Default
        'default_unknown', 'default_yes',

        # Housing
        'housing_unknown', 'housing_yes',

        # Loan
        'loan_unknown', 'loan_yes',

        # Contact
        'contact_telephone',

        # Month
        'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar',
        'month_may', 'month_nov', 'month_oct', 'month_sep',

        # Day of week
        'day_of_week_mon', 'day_of_week_thu', 'day_of_week_tue', 'day_of_week_wed',

        # Poutcome
        'poutcome_nonexistent', 'poutcome_success'
    ]

    # Fill all with 0 first
    for feature in all_possible_categories:
        input_data[feature] = 0

    # Now turn the correct ones ON
    if job != 'admin.':
        input_data[f'job_{job}'] = 1
    if marital != 'married':
        input_data[f'marital_{marital}'] = 1
    if education != 'basic.4y':
        input_data[f'education_{education}'] = 1
    if default != 'no':
        input_data[f'default_{default}'] = 1
    if housing != 'no':
        input_data[f'housing_{housing}'] = 1
    if loan != 'no':
        input_data[f'loan_{loan}'] = 1
    if contact != 'cellular':
        input_data[f'contact_{contact}'] = 1
    if month != 'may':
        input_data[f'month_{month}'] = 1
    if day_of_week != 'fri':
        input_data[f'day_of_week_{day_of_week}'] = 1
    if poutcome != 'failure':
        input_data[f'poutcome_{poutcome}'] = 1

    # Pdays contacted
    input_data['pdays_contacted'] = 1 if pdays_contacted == 'Yes' else 0

    return pd.DataFrame([input_data])

if st.button('Predict'):
    user_input = preprocess_input()

    # Reindex user input to match feature names from training
    user_input = user_input.reindex(columns=feature_names, fill_value=0)

    prediction = svm_model.predict(user_input)

    if prediction[0] == 1:
        st.success("The client is likely to subscribe to a term deposit!")
    else:
        st.error("The client is unlikely to subscribe to a term deposit.")
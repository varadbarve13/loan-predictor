import streamlit as st
import pandas as pd
import joblib 
model=joblib.load("SVMRBF_project.pkl")
scaler=joblib.load("scaler_project.pkl")
expected_columns=joblib.load("columns_project.pkl")
st.title("Loan Repayment Predictor")
st.markdown("Provide the following details to check Loan repayment prediction:")

employment_status=st.selectbox("Employment status",["Employed","Unemployed","Self-employed","Retired","Student"])
debt_to_income_ratio=st.slider("debt to income ratio",0.000,0.700,0.055)
credit_score=st.number_input("credit score",0.000,900.000,450.000)
loan_purpose=st.selectbox("Loan purpose",["Car","Debt consolidation","Business","Other","Home","Medical","Education","Vacation"])
interest_rate=st.slider("Interest rate",0.0,21.5,2.0)
installment=st.number_input("Installment",0.000,1700.000,500.000)
grade_subgrade=st.selectbox("grade subgrade",["C3","C4","C2","C5","C1","D1","D3","D4","D2","D5","B2","B1","B5","B3","B4","E4","E3","F5","E1","E2","F4","E5","F1","F2","F3","A5","A3","A2","A4","A1"])
num_of_open_accounts=st.number_input("Number of open accounts",0,16,2)
total_credit_limit=st.number_input("Total credit limit",0.000,500000.000,25000.000)
delinquency_history=st.number_input("Delinquency history",0,12,1)
public_records=st.number_input("Public records",0,3,0)
no_of_delinquencies=st.number_input("Number of delinquencies",0,12,1)
if st.button("Predict"):

    raw_input = {
        'employment_status_' + employment_status: 1,
        'debt_to_income_ratio': debt_to_income_ratio,
        'credit_score': credit_score,
        'loan_purpose_' + loan_purpose: 1,
        'interest_rate': interest_rate,
        'installment': installment,
        'grade_subgrade_' + grade_subgrade: 1,
        'num_of_open_accounts': num_of_open_accounts,
        'total_credit_limit': total_credit_limit,
        'delinquency_history': delinquency_history,
        'public_records': public_records,
        'num_of_delinquencies': no_of_delinquencies
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.success("Loan is repayable")
    else:
        st.error("Loan cannot be repaid")
     
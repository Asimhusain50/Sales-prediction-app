
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("sales_predictor_xgb.pkl")

st.title("💸 Sales Prediction App")

quantity = st.number_input("Quantity", min_value=1, value=1)
unit_price = st.number_input("Unit Price (£)", min_value=0.1, value=1.0)
invoice_hour = st.slider("Invoice Hour (24hr)", 0, 23, 12)
country = st.selectbox("Country", ['United Kingdom', 'Germany', 'France', 'EIRE', 'Spain', 'Netherlands'])
weekday = st.selectbox("Invoice Weekday", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

if st.button("Predict Total Sales"):
    input_df = pd.DataFrame({
        'Quantity': [quantity],
        'UnitPrice': [unit_price],
        'InvoiceHour': [invoice_hour],
        'Country': [country],
        'InvoiceWeekday': [weekday]
    })
    prediction = model.predict(input_df)[0]
    st.success(f"🧾 Predicted Sales Amount: **£{prediction:.2f}**")

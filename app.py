import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set Streamlit config FIRST
st.set_page_config(page_title="Sales Predictor", layout="centered")

# Set Seaborn style
sns.set(style="whitegrid")

# Load trained model
model = joblib.load("sales_predictor_xgb.pkl")

# --- App Title ---
st.title("💸 Sales Prediction App")

st.markdown("Use this tool to predict **total sales value (£)** based on transaction details and explore insights from uploaded sales data.")

# --- Prediction Form ---
st.header("📥 Enter Transaction Details")

quantity = st.number_input("Quantity", min_value=1, value=1)
unit_price = st.number_input("Unit Price (£)", min_value=0.1, value=1.0)
invoice_hour = st.slider("Invoice Hour (24hr format)", 0, 23, 12)
country = st.selectbox("Country", ['United Kingdom', 'Germany', 'France', 'EIRE', 'Spain', 'Netherlands'])
weekday = st.selectbox("Weekday", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

if st.button("🔮 Predict Sales"):
    input_df = pd.DataFrame({
        'Quantity': [quantity],
        'UnitPrice': [unit_price],
        'InvoiceHour': [invoice_hour],
        'Country': [country],
        'InvoiceWeekday': [weekday]
    })
    prediction = model.predict(input_df)[0]
    st.success(f"🧾 Predicted Sales Amount: **£{prediction:.2f}**")

# --- Data Upload Section ---
st.header("📊 Upload CSV for Data Insights")

data_file = st.file_uploader("Upload your sales CSV file (must include InvoiceDate, Quantity, UnitPrice, Country)", type=['csv'])

if data_file:
    df = pd.read_csv(data_file, parse_dates=["InvoiceDate"])

    # --- Cleaning ---
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceHour'] = df['InvoiceDate'].dt.hour
    df['InvoiceWeekday'] = df['InvoiceDate'].dt.day_name()

    # --- Interactive Filters ---
    st.sidebar.markdown("## 🔍 Filter Data")
    countries = st.sidebar.multiselect("Select Countries", df["Country"].unique(), default=df["Country"].unique())
    df = df[df["Country"].isin(countries)]

    min_date = df["InvoiceDate"].min()
    max_date = df["InvoiceDate"].max()
    date_range = st.sidebar.slider("Select Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date))
    df = df[(df["InvoiceDate"] >= date_range[0]) & (df["InvoiceDate"] <= date_range[1])]

    if df.empty:
        st.warning("⚠️ No data available after filtering. Try changing filter options.")
    else:
        # --- Optional Raw Data Viewer ---
        if st.checkbox("🧾 Show raw data"):
            st.dataframe(df.head(50))

        # --- Chart 1: Hourly Sales ---
        st.subheader("🕒 Sales by Hour of Day")
        hour_sales = df.groupby("InvoiceHour")["TotalPrice"].sum().reset_index()
        fig1, ax1 = plt.subplots()
        sns.barplot(data=hour_sales, x="InvoiceHour", y="TotalPrice", ax=ax1, palette="Blues")
        ax1.set_title("Hourly Sales Distribution")
        st.pyplot(fig1)

        # --- Chart 2: Weekly Sales ---
        st.subheader("📅 Sales by Weekday")
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_sales = df.groupby("InvoiceWeekday")["TotalPrice"].sum().reindex(weekdays).reset_index()
        fig2, ax2 = plt.subplots()
        sns.barplot(data=weekday_sales, x="InvoiceWeekday", y="TotalPrice", ax=ax2, palette="muted")
        ax2.set_title("Weekly Sales Distribution")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30)
        st.pyplot(fig2)

        # --- Chart 3: Sales Trend ---
        st.subheader("📈 Daily Sales Trend")
        daily_sales = df.set_index("InvoiceDate").resample("D")["TotalPrice"].sum().fillna(0)
        fig3, ax3 = plt.subplots()
        daily_sales.plot(ax=ax3, color='orange')
        ax3.set_title("Revenue Over Time")
        ax3.set_ylabel("Total Revenue")
        st.pyplot(fig3)

# --- Feature Importance ---
st.header("🧠 Model Feature Importance")
try:
    ohe = model.named_steps['preprocessor'].named_transformers_['cat']
    cat_features = ohe.get_feature_names_out(['Country', 'InvoiceWeekday'])
    all_features = ['Quantity', 'UnitPrice', 'InvoiceHour'] + list(cat_features)
    importances = model.named_steps['regressor'].feature_importances_

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances, y=all_features, ax=ax4)
    ax4.set_title("Features Influencing Prediction")
    st.pyplot(fig4)
except:
    st.warning("Feature importance unavailable — model structure may not be compatible.")

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by **Asim** | [Streamlit](https://streamlit.io)")

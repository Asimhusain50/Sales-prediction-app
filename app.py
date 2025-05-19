
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
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set(style="whitegrid")
st.set_page_config(page_title="Sales Predictor", layout="centered")

# Load model
model = joblib.load("sales_predictor_xgb.pkl")

# App Title
st.title("💸 Sales Prediction App")

st.markdown("Use this tool to predict **total sales value (£)** based on transaction details.")

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

# --- Charts Section ---
st.header("📊 Data Insights & Visualizations")

# Sidebar Data Filter (Optional sample or uploaded data)
st.sidebar.markdown("### 📂 Upload CSV for Insights")
data_file = st.sidebar.file_uploader("Upload sales CSV with columns like Quantity, UnitPrice, InvoiceDate, etc.", type=['csv'])

if data_file:
    df = pd.read_csv(data_file, parse_dates=["InvoiceDate"])

    # Basic cleaning if needed
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceHour'] = df['InvoiceDate'].dt.hour
    df['InvoiceWeekday'] = df['InvoiceDate'].dt.day_name()

    # --- Filter ---
    selected_country = st.sidebar.selectbox("Filter by Country", df["Country"].unique())
    df = df[df["Country"] == selected_country]

    # --- Chart 1: Sales by Hour ---
    st.subheader("🕒 Sales by Hour of Day")
    hour_sales = df.groupby("InvoiceHour")["TotalPrice"].sum().reset_index()
    fig1, ax1 = plt.subplots()
    sns.barplot(data=hour_sales, x="InvoiceHour", y="TotalPrice", ax=ax1, palette="Blues_d")
    ax1.set_title("Hourly Sales Distribution")
    st.pyplot(fig1)

    # --- Chart 2: Sales by Weekday ---
    st.subheader("📅 Sales by Weekday")
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_sales = df.groupby("InvoiceWeekday")["TotalPrice"].sum().reindex(weekdays).reset_index()
    fig2, ax2 = plt.subplots()
    sns.barplot(data=weekday_sales, x="InvoiceWeekday", y="TotalPrice", ax=ax2)
    ax2.set_title("Weekly Sales Distribution")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30)
    st.pyplot(fig2)

    # --- Chart 3: Sales Trend Over Time ---
    st.subheader("📈 Daily Sales Trend")
    daily_sales = df.set_index("InvoiceDate").resample("D")["TotalPrice"].sum().fillna(0)
    fig3, ax3 = plt.subplots()
    daily_sales.plot(ax=ax3)
    ax3.set_ylabel("Revenue (£)")
    ax3.set_title("Sales Over Time")
    st.pyplot(fig3)

# --- Feature Importance (Static from trained model) ---
st.header("🧠 Feature Importance (Model Explanation)")
ohe = model.named_steps['preprocessor'].named_transformers_['cat']
cat_features = ohe.get_feature_names_out(['Country', 'InvoiceWeekday'])
full_features = ['Quantity', 'UnitPrice', 'InvoiceHour'] + list(cat_features)
importances = model.named_steps['regressor'].feature_importances_

fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.barplot(x=importances, y=full_features, ax=ax4)
ax4.set_title("Features Influencing Sales Prediction")
st.pyplot(fig4)

st.markdown("---")
st.markdown("Made with ❤️ by Asim | [GitHub](https://github.com/) | [Streamlit](https://streamlit.io)")

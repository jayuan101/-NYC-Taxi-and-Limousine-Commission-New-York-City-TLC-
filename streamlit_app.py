import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="NYC Taxi Trip Analysis", layout="wide")
st.title("🚕 NYC Yellow Taxi Trip Data Analysis")

st.write("""
Welcome!  
This app helps you **explore and predict taxi fares** using real NYC trip data.  
Upload a CSV or use the built-in dataset.
""")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("2017_Yellow_Taxi_Trip_Data.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df0 = load_data()

if df0.empty:
    st.error("❌ Dataset could not be loaded. Please check the CSV file.")
    st.stop()
else:
    df = df0.copy()

# -------------------------------
# Dataset Overview
# -------------------------------
st.header("📂 Dataset Overview")

st.write("### Preview of Data")
st.dataframe(df.head(20))  # safer than df.info()

col1, col2 = st.columns(2)
with col1:
    st.write("**Shape:**", df.shape)
with col2:
    st.write("**Column Types:**")
    st.write(df.dtypes)

# Missing values and duplicates
st.subheader("🛠 Data Quality Check")
st.write(f"🔍 Missing values: **{df.isna().sum().sum()}**")
st.write(f"🔁 Duplicate rows: **{df.duplicated().sum()}**")

# Show stats only if requested
if st.checkbox("Show descriptive statistics"):
    st.write(df.describe())

# -------------------------------
# Data Preprocessing & Features
# -------------------------------
required_columns = [
    'tpep_pickup_datetime', 'tpep_dropoff_datetime',
    'fare_amount', 'trip_distance',
    'PULocationID', 'DOLocationID'
]

if all(col in df.columns for col in required_columns):

    # Convert to datetime
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')

    # Trip duration in minutes
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration'])
    df = df[df['duration'] > 0]

    # Handle outliers (cap extreme values)
    for col in ['fare_amount', 'duration']:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        upper = q3 + 6 * iqr
        df.loc[df[col] > upper, col] = upper

    # Feature engineering
    df['pickup_dropoff'] = df['PULocationID'].astype(str) + "_" + df['DOLocationID'].astype(str)
    grouped = df.groupby('pickup_dropoff').mean(numeric_only=True)[['trip_distance']]
    df['mean_distance'] = df['pickup_dropoff'].map(grouped['trip_distance'].to_dict())

    # -------------------------------
    # Modeling
    # -------------------------------
    st.header("🤖 Predicting Taxi Fares")

    X = df[['mean_distance', 'duration']]
    y = df['fare_amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_test = lr.predict(X_test_scaled)

    # -------------------------------
    # Model Evaluation
    # -------------------------------
    st.subheader("📈 Model Performance")
    st.metric("R² Score", round(r2_score(y_test, y_pred_test), 3))
    st.metric("MAE", round(mean_absolute_error(y_test, y_pred_test), 2))
    st.metric("RMSE", round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 2))

    # -------------------------------
    # Visualizations
    # -------------------------------
    st.subheader("📊 Visualizations")

    if st.checkbox("Show Actual vs Predicted Plot"):
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.5, ax=ax)
        ax.set_xlabel("Actual Fare ($)")
        ax.set_ylabel("Predicted Fare ($)")
        ax.set_title("Actual vs Predicted Fares")
        st.pyplot(fig)

    if st.checkbox("Show Residuals Distribution"):
        residuals = y_test - y_pred_test
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(residuals, bins=50, kde=True, ax=ax2)
        ax2.set_title("Residuals Distribution")
        st.pyplot(fig2)

else:
    st.error("⚠️ Dataset missing required columns. Please check the input file.")

# 🚕 NYC Taxi Trip Data Analysis - Streamlit App

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
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

st.markdown("""
Welcome!  
This app lets you **explore, clean, and model New York City yellow taxi trip data**.  

### What you can do:
1. Preview dataset info and statistics  
2. Check for missing values and duplicates  
3. Train a **Linear Regression model** to predict taxi fares  
4. Visualize predictions vs. actual fares and residuals  
""")

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("⚙️ App Settings")
test_size = st.sidebar.slider("Test Size (for train/test split)", 0.1, 0.5, 0.2, step=0.05)
random_state = st.sidebar.number_input("Random State (reproducibility)", value=42, step=1)

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
with st.expander("📂 Dataset Overview", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Shape of dataset:**", df.shape)
    with col2:
        st.write("**Columns:**", list(df.columns))

    # Show dataset info
    buffer = StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    # Missing values and duplicates
    st.write(f"🔍 Total missing values: **{df.isna().sum().sum()}**")
    st.write(f"🔁 Duplicate rows: **{df.duplicated().sum()}**")

    # Descriptive stats
    st.subheader("📊 Descriptive Statistics")
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

    # Handle outliers
    def outlier_imputer(column_list, iqr_factor=6):
        for col in column_list:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            upper = q3 + (iqr_factor * iqr)
            df.loc[df[col] > upper, col] = upper

    outlier_imputer(['fare_amount', 'duration'])

    # Feature engineering
    df['pickup_dropoff'] = df['PULocationID'].astype(str) + "_" + df['DOLocationID'].astype(str)
    grouped = df.groupby('pickup_dropoff').mean(numeric_only=True)[['trip_distance']]
    df['mean_distance'] = df['pickup_dropoff'].map(grouped['trip_distance'].to_dict())

    # -------------------------------
    # Modeling
    # -------------------------------
    st.header("🤖 Taxi Fare Prediction")

    X = df[['mean_distance', 'duration']]
    y = df['fare_amount']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

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
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Score", f"{r2_score(y_test, y_pred_test):.4f}")
    col2.metric("MAE", f"{mean_absolute_error(y_test, y_pred_test):.2f}")
    col3.metric("MSE", f"{mean_squared_error(y_test, y_pred_test):.2f}")
    col4.metric("RMSE", f"{np.sqrt(me

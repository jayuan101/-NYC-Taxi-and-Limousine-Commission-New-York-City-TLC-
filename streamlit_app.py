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
# Streamlit Setup
# -------------------------------
st.set_page_config(page_title="NYC Taxi Fare Predictor", layout="wide")
st.title("🚕 NYC Taxi Fare Predictor")

st.write("""
Explore NYC taxi trip data and predict fares!  
You can type pickup/drop-off locations, see average prices, and visualize trends.
""")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("2017_Yellow_Taxi_Trip_Data.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# -------------------------------
# Preprocess Data
# -------------------------------
required_cols = [
    'tpep_pickup_datetime', 'tpep_dropoff_datetime',
    'fare_amount', 'trip_distance',
    'PULocationID', 'DOLocationID'
]

if all(col in df.columns for col in required_cols):

    # Datetime
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    df = df[(df['duration'] > 0) & (df['fare_amount'] > 0)]

    # Outlier handling
    for col in ['fare_amount', 'duration']:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        upper = q3 + 6 * iqr
        df.loc[df[col] > upper, col] = upper

    # Feature: pickup_dropoff
    df['pickup_dropoff'] = df['PULocationID'].astype(str) + "_" + df['DOLocationID'].astype(str)
    avg_dist = df.groupby('pickup_dropoff')['trip_distance'].mean().to_dict()
    df['mean_distance'] = df['pickup_dropoff'].map(avg_dist)

    # -------------------------------
    # Train Model
    # -------------------------------
    X = df[['mean_distance', 'duration']]
    y = df['fare_amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # -------------------------------
    # Sidebar: User Prediction
    # -------------------------------
    st.sidebar.header("🔮 Predict a Fare")

    pickup = st.sidebar.text_input("Pickup Location ID (e.g. 132)", "132")
    dropoff = st.sidebar.text_input("Dropoff Location ID (e.g. 138)", "138")
    duration = st.sidebar.number_input("Trip Duration (minutes)", min_value=1.0, value=10.0)

    # Build pickup_dropoff key
    key = f"{pickup}_{dropoff}"
    if key in avg_dist:
        mean_dist = avg_dist[key]
    else:
        mean_dist = df['trip_distance'].mean()  # fallback

    # Predict fare
    user_X = scaler.transform([[mean_dist, duration]])
    predicted_fare = model.predict(user_X)[0]

    st.sidebar.success(f"💰 Predicted Fare: ${predicted_fare:.2f}")

    # -------------------------------
    # Model Performance
    # -------------------------------
    st.header("📈 Model Performance")
    st.metric("R² Score", round(r2_score(y_test, model.predict(X_test_scaled)), 3))
    st.metric("MAE", round(mean_absolute_error(y_test, model.predict(X_test_scaled)), 2))
    st.metric("RMSE", round(np.sqrt(mean_squared_error(y_test, model.predict(X_test_scaled))), 2))

    # -------------------------------
    # Visualizations
    # -------------------------------
    st.header("📊 Visual Insights")

    # 1. Distribution of fares
    st.subheader("Fare Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['fare_amount'], bins=50, ax=ax1, kde=True)
    ax1.set_title("Distribution of Fares")
    st.pyplot(fig1)

    # 2. Trip distance vs fare
    st.subheader("Trip Distance vs Fare")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x="trip_distance", y="fare_amount", data=df.sample(5000), alpha=0.4, ax=ax2)
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, 100)
    ax2.set_title("Trip Distance vs Fare")
    st.pyplot(fig2)

    # 3. Average fare by pickup hour
    st.subheader("Average Fare by Hour of Day")
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    hourly_avg = df.groupby('hour')['fare_amount'].mean()
    fig3, ax3 = plt.subplots()
    sns.lineplot(x=hourly_avg.index, y=hourly_avg.values, marker="o", ax=ax3)
    ax3.set_title("Average Fare by Pickup Hour")
    ax3.set_xlabel("Hour of Day")
    ax3.set_ylabel("Average Fare ($)")
    st.pyplot(fig3)

else:
    st.error("⚠️ Dataset missing required columns. Please check the input file.")

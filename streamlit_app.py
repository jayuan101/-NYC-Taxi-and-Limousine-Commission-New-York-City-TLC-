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
This app lets you explore NYC taxi trips and predict fares.  
Select **pickup & dropoff zones**, input **trip duration**, and get a predicted fare.
""")

# -------------------------------
# Load CSVs
# -------------------------------
@st.cache_data
def load_data():
    try:
        taxi_df = pd.read_csv("2017_Yellow_Taxi_Trip_Data.csv")
        zones_df = pd.read_csv("taxi_zones.csv")  # zone ID to name mapping
        return taxi_df, zones_df
    except Exception as e:
        st.error(f"Error loading CSVs: {e}")
        return pd.DataFrame(), pd.DataFrame()

taxi_df, zones_df = load_data()
if taxi_df.empty or zones_df.empty:
    st.stop()

# -------------------------------
# Merge taxi data with zone names
# -------------------------------
taxi_df = taxi_df.merge(zones_df.rename(columns={"LocationID": "PULocationID", "Zone": "Pickup_Zone", "Borough": "Pickup_Borough"}), on="PULocationID", how="left")
taxi_df = taxi_df.merge(zones_df.rename(columns={"LocationID": "DOLocationID", "Zone": "Dropoff_Zone", "Borough": "Dropoff_Borough"}), on="DOLocationID", how="left")

# -------------------------------
# Preprocess
# -------------------------------
taxi_df['tpep_pickup_datetime'] = pd.to_datetime(taxi_df['tpep_pickup_datetime'], errors='coerce')
taxi_df['tpep_dropoff_datetime'] = pd.to_datetime(taxi_df['tpep_dropoff_datetime'], errors='coerce')
taxi_df['duration'] = (taxi_df['tpep_dropoff_datetime'] - taxi_df['tpep_pickup_datetime']).dt.total_seconds() / 60
taxi_df = taxi_df[(taxi_df['duration'] > 0) & (taxi_df['fare_amount'] > 0)]

# Outlier handling
for col in ['fare_amount', 'duration']:
    q1, q3 = taxi_df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 6 * iqr
    taxi_df.loc[taxi_df[col] > upper, col] = upper

# Feature engineering: mean distance by pickup-dropoff
taxi_df['pickup_dropoff'] = taxi_df['PULocationID'].astype(str) + "_" + taxi_df['DOLocationID'].astype(str)
avg_dist = taxi_df.groupby('pickup_dropoff')['trip_distance'].mean().to_dict()
taxi_df['mean_distance'] = taxi_df['pickup_dropoff'].map(avg_dist)

# -------------------------------
# Train Linear Regression
# -------------------------------
X = taxi_df[['mean_distance', 'duration']]
y = taxi_df['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# -------------------------------
# Sidebar: User Input
# -------------------------------
st.sidebar.header("🔮 Predict a Fare")

pickup_name = st.sidebar.selectbox("Pickup Zone", sorted(taxi_df['Pickup_Zone'].dropna().unique()))
dropoff_name = st.sidebar.selectbox("Dropoff Zone", sorted(taxi_df['Dropoff_Zone'].dropna().unique()))
duration = st.sidebar.number_input("Trip Duration (minutes)", min_value=1.0, value=10.0)

# Map zone names to LocationID
pickup_id = taxi_df[taxi_df['Pickup_Zone'] == pickup_name]['PULocationID'].iloc[0]
dropoff_id = taxi_df[taxi_df['Dropoff_Zone'] == dropoff_name]['DOLocationID'].iloc[0]
key = f"{pickup_id}_{dropoff_id}"
mean_dist = avg_dist.get(key, taxi_df['trip_distance'].mean())

# Predict fare
user_X = scaler.transform([[mean_dist, duration]])
predicted_fare = model.predict(user_X)[0]
st.sidebar.success(f"💰 Predicted Fare: ${predicted_fare:.2f}")

# -------------------------------
# Model Metrics
# -------------------------------
st.header("📈 Model Performance")
st.write("**R² Score:**", round(r2_score(y_test, model.predict(X_test_scaled)), 3))
st.write("**MAE:**", round(mean_absolute_error(y_test, model.predict(X_test_scaled)), 2))
st.write("**RMSE:**", round(np.sqrt(mean_squared_error(y_test, model.predict(X_test_scaled))), 2))

# -------------------------------
# Visualizations
# -------------------------------
st.header("📊 Visualizations")

# Fare Distribution
st.subheader("Fare Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(taxi_df['fare_amount'], bins=50, kde=True, ax=ax1)
ax1.set_title("Fare Distribution")
st.pyplot(fig1)

# Distance vs Fare
st.subheader("Trip Distance vs Fare")
fig2, ax2 = plt.subplots()
sns.scatterplot(x="trip_distance", y="fare_amount", data=taxi_df.sample(5000), alpha=0.4, ax=ax2)
ax2.set_xlim(0, 30)
ax2.set_ylim(0, 100)
ax2.set_title("Trip Distance vs Fare")
st.pyplot(fig2)

# Average fare by pickup hour
st.subheader("Average Fare by Pickup Hour")
taxi_df['hour'] = taxi_df['tpep_pickup_datetime'].dt.hour
hourly_avg = taxi_df.groupby('hour')['fare_amount'].mean()
fig3, ax3 = plt.subplots()
sns.lineplot(x=hourly_avg.index, y=hourly_avg.values, marker="o", ax=ax3)
ax3.set_title("Average Fare by Pickup Hour")
ax3.set_xlabel("Hour of Day")
ax3.set_ylabel("Average Fare ($)")
st.pyplot(fig3)

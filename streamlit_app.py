import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from duckdb_engine import connect

# -------------------------------
# Streamlit Setup
# -------------------------------
st.set_page_config(page_title="NYC Taxi Fare Predictor", layout="wide")
st.title("🚕 NYC Taxi Fare Predictor with MotherDuck")

st.write("""
This app lets you explore NYC taxi trips and predict fares.  
Select **pickup & dropoff zones**, input **trip duration**, and get an instant fare prediction.
""")

# -------------------------------
# Connect to MotherDuck
# -------------------------------
# Use your provided DB URL
MOTHERDUCK_URL = "main@9b91bf816122b90e495db16743c62149e6d1580d"
conn = connect(MOTHERDUCK_URL)

# -------------------------------
# Load Data from MotherDuck
# -------------------------------
@st.cache_data
def load_data():
    query = """
    SELECT
        t.*,
        z_pickup.Zone AS Pickup_Zone,
        z_dropoff.Zone AS Dropoff_Zone,
        z_pickup.Borough AS Pickup_Borough,
        z_dropoff.Borough AS Dropoff_Borough
    FROM my_db.main."TAXI NYC" t
    LEFT JOIN my_db.main.zones z_pickup ON t.PULocationID = z_pickup.LocationID
    LEFT JOIN my_db.main.zones z_dropoff ON t.DOLocationID = z_dropoff.LocationID
    LIMIT 100000;
    """
    df = conn.execute(query).fetchdf()
    return df

df = load_data()

if df.empty:
    st.error("❌ No data loaded. Check MotherDuck tables.")
    st.stop()

# -------------------------------
# Data Preprocessing
# -------------------------------
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

# Feature engineering
df['pickup_dropoff'] = df['PULocationID'].astype(str) + "_" + df['DOLocationID'].astype(str)
avg_dist = df.groupby('pickup_dropoff')['trip_distance'].mean().to_dict()
df['mean_distance'] = df['pickup_dropoff'].map(avg_dist)

# -------------------------------
# Train Linear Regression Model
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
# Sidebar: User Input for Prediction
# -------------------------------
st.sidebar.header("🔮 Predict a Fare")

pickup_name = st.sidebar.selectbox("Pickup Zone", sorted(df['Pickup_Zone'].dropna().unique()))
dropoff_name = st.sidebar.selectbox("Dropoff Zone", sorted(df['Dropoff_Zone'].dropna().unique()))
duration = st.sidebar.number_input("Trip Duration (minutes)", min_value=1.0, value=10.0)

# Map zone names back to LocationID
pickup_id = df[df['Pickup_Zone'] == pickup_name]['PULocationID'].iloc[0]
dropoff_id = df[df['Dropoff_Zone'] == dropoff_name]['DOLocationID'].iloc[0]

key = f"{pickup_id}_{dropoff_id}"
mean_dist = avg_dist.get(key, df['trip_distance'].mean())

# Predict fare
user_X = scaler.transform([[mean_dist, duration]])
predicted_fare = model.predict(user_X)[0]
st.sidebar.success(f"💰 Predicted Fare: ${predicted_fare:.2f}")

# -------------------------------
# Model Performance
# -------------------------------
st.header("📈 Model Performance Metrics")
st.write("**R² Score:**", round(r2_score(y_test, model.predict(X_test_scaled)), 3))
st.write("**MAE:**", round(mean_absolute_error(y_test, model.predict(X_test_scaled)), 2))
st.write("**RMSE:**", round(np.sqrt(mean_squared_error(y_test, model.predict(X_test_scaled))), 2))

# -------------------------------
# Visualizations
# -------------------------------
st.header("📊 Data Visualizations")

# Fare Distribution
st.subheader("Fare Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(df['fare_amount'], bins=50, kde=True, ax=ax1)
ax1.set_title("Distribution of Fares")
st.pyplot(fig1)

# Distance vs Fare
st.subheader("Trip Distance vs Fare")
fig2, ax2 = plt.subplots()
sns.scatterplot(x="trip_distance", y="fare_amount", data=df.sample(5000), alpha=0.4, ax=ax2)
ax2.set_xlim(0, 30)
ax2.set_ylim(0, 100)
ax2.set_title("Trip Distance vs Fare")
st.pyplot(fig2)

# Average fare by pickup hour
st.subheader("Average Fare by Pickup Hour")
df['hour'] = df['tpep_pickup_datetime'].dt.hour
hourly_avg = df.groupby('hour')['fare_amount'].mean()
fig3, ax3 = plt.subplots()
sns.lineplot(x=hourly_avg.index, y=hourly_avg.values, marker="o", ax=ax3)
ax3.set_title("Average Fare by Pickup Hour")
ax3.set_xlabel("Hour of Day")
ax3.set_ylabel("Average Fare ($)")
st.pyplot(fig3)

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

# Optional: catch missing packages
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError as e:
    st.error(f"Missing package: {e.name}. Add it to requirements.txt")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
except ModuleNotFoundError as e:
    st.error(f"Missing package: {e.name}. Add it to requirements.txt")


# Set up Streamlit page
st.set_page_config(page_title="Taxi Trip Analysis", layout="wide")
st.title("Taxi Trip Data Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    df0 = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

df = df0.copy()

# Dataset overview
st.header("Dataset Overview")
st.write("Shape:", df.shape)
buffer = StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())

st.subheader("Missing Data and Duplicates")
st.write("Total missing values:", df.isna().sum().sum())
st.write("Duplicate rows:", df.duplicated().sum())

st.subheader("Descriptive Statistics")
st.write(df.describe())

# Required columns
required_columns = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'fare_amount',
                    'trip_distance', 'PULocationID', 'DOLocationID']

if all(col in df.columns for col in required_columns):
    # Preprocessing
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration'])
    df = df[df['duration'] > 0]

    # Outlier capping
    def outlier_imputer(column_list, iqr_factor=6):
        for col in column_list:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            upper = q3 + (iqr_factor * iqr)
            df.loc[df[col] > upper, col] = upper

    outlier_imputer(['fare_amount', 'duration'])

    # Feature engineering
    df['pickup_dropoff'] = df['PULocationID'].astype(str) + ' ' + df['DOLocationID'].astype(str)
    grouped = df.groupby('pickup_dropoff').mean(numeric_only=True)[['trip_distance']]
    df['mean_distance'] = df['pickup_dropoff'].map(grouped['trip_distance'].to_dict())

    # Modeling
    X = df[['mean_distance', 'duration']]
    y = df['fare_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_test = lr.predict(X_test_scaled)

    # Model evaluation
    st.subheader("Model Evaluation")
    st.write("R^2 Score:", r2_score(y_test, y_pred_test))
    st.write("MAE:", mean_absolute_error(y_test, y_pred_test))
    st.write("MSE:", mean_squared_error(y_test, y_pred_test))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))

    # Visualizations
    st.subheader("Visualizations")
    st.write("Actual vs Predicted Fare")
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.5, ax=ax)
    ax.set_xlabel("Actual Fare")
    ax.set_ylabel("Predicted Fare")
    st.pyplot(fig)

    st.write("Residuals Distribution")
    residuals = y_test - y_pred_test
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.histplot(residuals, bins=50, kde=True, ax=ax2)
    st.pyplot(fig2)

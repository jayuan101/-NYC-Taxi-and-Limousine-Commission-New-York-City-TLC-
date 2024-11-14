import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from io import StringIO

# Set up Streamlit page configuration
st.set_page_config(page_title="Taxi Trip Analysis", layout="wide")

# Load the dataset
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
    st.error("Failed to load the dataset.")
else:
    df = df0.copy()

    # Display basic dataset information
    st.title("Taxi Trip Data Analysis")
    st.header("Dataset Overview")

    st.write("Shape of the dataset:", df.shape)
    
    # Capture the output of df.info() as a string for Streamlit
    buffer = StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Check for missing values and duplicates
    st.subheader("Missing Data and Duplicates")
    st.write("Total missing values:", df.isna().sum().sum())
    st.write("Duplicate rows:", df.duplicated().sum())

    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(df.describe())

    # Ensure required columns exist
    required_columns = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'fare_amount', 'trip_distance', 'PULocationID', 'DOLocationID']
    if all(col in df.columns for col in required_columns):
        
        # Data preprocessing
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
        df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60  # convert to minutes

        # Drop rows with NaT in datetime columns
        df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration'])

        # Outlier detection and handling
        def outlier_imputer(column_list, iqr_factor):
            for col in column_list:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                upper_threshold = q3 + (iqr_factor * iqr)
                df.loc[df[col] > upper_threshold, col] = upper_threshold

        outlier_imputer(['fare_amount', 'duration'], 6)

        # Feature engineering
        df['pickup_dropoff'] = df['PULocationID'].astype(str) + ' ' + df['DOLocationID'].astype(str)
        grouped = df.groupby('pickup_dropoff').mean(numeric_only=True)[['trip_distance']]
        grouped_dict = grouped['trip_distance'].to_dict()
        df['mean_distance'] = df['pickup_dropoff'].map(grouped_dict)

        # Modeling
        X = df[['mean_distance', 'duration']]
        y = df['fare_amount']

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Standardize data
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)

        # Predictions
        y_pred_test = lr.predict(X_test_scaled)

        # Model Evaluation
        st.subheader("Model Evaluation")
        st.write("Coefficient of Determination (R^2):", r2_score(y_test, y_pred_test))
        st.write("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred_test))
        st.write("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_test))
        st.write("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred_test)))

        # Visualization
        st.subheader("Visualizations")

        # Scatterplot
        st.write("Scatterplot of Actual vs Predicted Fare Amounts")
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.5, ax=ax)
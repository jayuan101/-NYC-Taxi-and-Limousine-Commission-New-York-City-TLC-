import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Set up Streamlit page configuration
st.set_page_config(page_title="Taxi Trip Analysis", layout="wide")

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv("2017_Yellow_Taxi_Trip_Data.csv")
    return df

df0 = load_data()
df = df0.copy()

# Display basic dataset information
st.title("Taxi Trip Data Analysis")
st.header("Dataset Overview")

st.write("Shape of the dataset:", df.shape)
st.write("Dataset Info:")
st.write(df.info())

# Check for missing values and duplicates
st.subheader("Missing Data and Duplicates")
st.write("Total missing values:", df.isna().sum().sum())
st.write("Duplicate rows:", df.duplicated().sum())

# Descriptive statistics
st.subheader("Descriptive Statistics")
st.write(df.describe())

# Data preprocessing
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p')
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], format='%m/%d/%Y %I:%M:%S %p')
df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime'])/np.timedelta64(1,'m')

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
grouped_dict = grouped.to_dict()['trip_distance']
df['mean_distance'] = df['pickup_dropoff'].map(grouped_dict)

# Modeling
X = df[['mean_distance', 'duration']]
y = df[['fare_amount']]

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
sns.scatterplot(x=y_test['fare_amount'], y=y_pred_test.ravel(), alpha=0.5, ax=ax)
plt.plot([0, 60], [0, 60], c='red', linewidth=2)
plt.title('Actual vs Predicted Fare Amounts')
plt.xlabel('Actual Fare Amount')
plt.ylabel('Predicted Fare Amount')
st.pyplot(fig)

# Residuals Distribution
st.write("Distribution of Residuals")
fig, ax = plt.subplots()
sns.histplot(y_test['fare_amount'] - y_pred_test.ravel(), bins=30, kde=True, ax=ax)
plt.title('Residual Distribution')
st.pyplot(fig)

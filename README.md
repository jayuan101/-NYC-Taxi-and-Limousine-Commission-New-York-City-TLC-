# 🚕 NYC Taxi Fare Predictor

This Streamlit app allows users to **explore New York City yellow taxi trip data** and **predict taxi fares** based on pickup and drop-off locations and trip duration. The app also includes data visualization and model evaluation features.

---

## Features

1. **Dataset Overview**
   - Preview first rows of the dataset
   - Display dataset shape, columns, and types
   - Check for missing values and duplicate rows
   - Optional descriptive statistics

2. **Data Preprocessing**
   - Converts pickup and drop-off timestamps to datetime
   - Calculates trip duration (minutes)
   - Handles outliers in fare and duration
   - Creates features for prediction, including average trip distance per pickup-dropoff pair

3. **Interactive Fare Prediction**
   - User can enter pickup & drop-off location IDs and trip duration
   - The app predicts fare using a **Linear Regression model**
   - Provides fallback average distance if pickup-dropoff pair is not in the dataset

4. **Model Evaluation**
   - Displays metrics: R², MAE, RMSE

5. **Visualizations**
   - Distribution of fares
   - Trip distance vs fare scatter plot
   - Average fare by pickup hour

---

## How to Use

1. Clone this repository:
   ```bash
   git clone <repository_url>
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Use the sidebar to:

Enter pickup & drop-off locations (IDs)

Input trip duration

See the predicted fare instantly

Explore the visualizations and model performance metrics on the main page.

Dataset
The app uses 2017 NYC Yellow Taxi Trip Data.

Ensure the CSV file is located in the same folder as the app or adjust the path in the code.

Required columns:

Copy code
tpep_pickup_datetime, tpep_dropoff_datetime, fare_amount, trip_distance, PULocationID, DOLocationID
Dependencies
See requirements.txt for all required packages. Minimal dependencies:

nginx
Copy code
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
Screenshots
Fare Prediction Sidebar

Visualizations: Fare Distribution, Trip Distance vs Fare, Hourly Average Fare

Notes
The model uses Linear Regression for simplicity; it can be upgraded to more advanced ML models.

Input validation: Only numeric pickup and drop-off IDs are accepted. The app falls back to average trip distance for unknown pairs.

Large datasets may slow down plotting. Use sampling for scatter plots for better performance.

/NYC-Taxi-Fare-Predictor
│
├─ app.py
├─ 2017_Yellow_Taxi_Trip_Data.csv
├─ requirements.txt
└─ README.md

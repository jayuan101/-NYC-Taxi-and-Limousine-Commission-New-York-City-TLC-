# 🚖 NYC Taxi Trip Analysis

Exploratory Data Analysis (EDA) and predictive modeling on NYC Yellow Taxi trip data.  
This project demonstrates data cleaning, feature engineering, regression modeling, and interactive visualization.

---

## 📊 Project Overview
- **Data Source:** [2017 Yellow Taxi Trip Data](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)  
- **Tools:** Python (Pandas, Scikit-learn, Streamlit), Tableau  
- **Goal:** Explore patterns in trip data and build a regression model to predict fare amounts.

---

## 🔎 Repo Structure
- `app/` → Streamlit app (`taxi_app.py`)  
- `data/` → Sample CSV + instructions to download full dataset  
- `notebooks/` → (Optional) Jupyter notebook with step-by-step EDA  
- `tableau/` → Tableau Public dashboard link  
- `requirements.txt` → Python dependencies  

---

## 🚀 How to Run Locally
```bash
git clone https://github.com/yourusername/taxi-trip-analysis.git
cd taxi-trip-analysis
pip install -r requirements.txt
streamlit run app/taxi_app.py

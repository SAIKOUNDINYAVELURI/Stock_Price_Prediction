# dashboard/app.py
import subprocess
import streamlit as st
import pandas as pd
import os
from components.timeseries_chart import plot_timeseries
from components.today_price import show_today_price
from components.prediction_card import show_prediction
from components.generate_signal import generate_signal
from components.display_signal import display_signal

# ----------------------------
# STEP 1: Run data collection & preprocessing scripts from src/
# ----------------------------
try:
    subprocess.run(["python", "../src/data_collection.py"], check=True)
    subprocess.run(
        ["python", "../src/data_preprocessing_and_feature_engineering.py"], check=True
    )
except subprocess.CalledProcessError as e:
    st.error(f"❌ Error running script: {e}")
    st.stop()

# ----------------------------
# STEP 2: Load the latest dataset
# ----------------------------
processed_path = "../data/AAPL_stock_data_clean.csv"

try:
    df = pd.read_csv(processed_path)
except FileNotFoundError:
    st.error(f"❌ Processed dataset not found at {processed_path}.")
    st.stop()

# ----------------------------
# STEP 3: Streamlit dashboard UI
# ----------------------------
st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")

st.title("Stock Price Prediction Dashboard")
st.title("Apple (AAPL)")

# Time Series Chart
plot_timeseries(df)

# Show Today’s Price Section
today_price = show_today_price(df)

# Show Prediction Section
direction, predicted_price, predicted_date, pred_conf = show_prediction(
    df,
    xgb_model_path="../models/xgboost_classifier.pkl",
    lstm_model_path="../models/lstm_model.keras",
    scaler_path="../models/lstm_scaler.pkl",
)

col1, col2 = st.columns(2)
with col1:
    st.metric("Predicted Direction", direction)
with col2:
    st.metric("Predicted Price", f"${predicted_price:.2f}")


signal, reason = generate_signal(df, direction, predicted_price, pred_conf)
display_signal(signal, reason, today_price, predicted_price)

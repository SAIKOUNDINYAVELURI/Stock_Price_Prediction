import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import timedelta


def show_prediction(data, xgb_model_path, lstm_model_path, scaler_path, lookback=60):
    """
    Predict tomorrow's stock price (LSTM) and direction (XGBoost).
    """
    st.subheader(" Tomorrow's Prediction")

    # Handle today's date
    if "Date" in data.columns:  # Use 'Date' column from dataset
        today = pd.to_datetime(data["Date"].iloc[-1])
    else:  # fallback to index
        today = data.index[-1]

    tomorrow = today + timedelta(days=1)

    # Load models
    clf = joblib.load(xgb_model_path)
    lstm_model = load_model(lstm_model_path)
    scaler = joblib.load(scaler_path)

    # XGBoost features
    features = [
        "Open",
        "High",
        "Low",
        "Volume",
        "MA_10",
        "EMA_10",
        "RSI",
        "BB_MA20",
        "BB_Upper",
        "BB_Lower",
        "Lag_Close_1",
        "Lag_Close_2",
        "Lag_Close_3",
        "Lag_Close_5",
        "Lag_Close_7",
        "Diff_Close_Lag1",
    ]

    # Create additional lag features
    latest_data = data.copy()
    latest_data["Lag_Close_3"] = latest_data["Close"].shift(3)
    latest_data["Lag_Close_5"] = latest_data["Close"].shift(5)
    latest_data["Lag_Close_7"] = latest_data["Close"].shift(7)
    latest_data["Diff_Close_Lag1"] = latest_data["Close"] - latest_data["Lag_Close_1"]
    latest_data.dropna(inplace=True)

    # Prepare data for XGBoost
    X_xgb = latest_data[features].iloc[-1:].values
    pred_dir = clf.predict(X_xgb)[0]
    pred_conf = clf.predict_proba(X_xgb).max() * 100
    direction = "UP 📈" if pred_dir == 1 else "DOWN 📉"

    # LSTM price prediction
    close_prices = data["Close"].values.reshape(-1, 1)
    scaled_data = scaler.transform(close_prices)
    X_lstm = scaled_data[-lookback:]
    X_lstm = np.reshape(X_lstm, (1, X_lstm.shape[0], 1))

    pred_price_scaled = lstm_model.predict(X_lstm, verbose=0)[0][0]
    pred_price = scaler.inverse_transform(np.array([[pred_price_scaled]]))[0][0]

    # Display
    st.write(f" **Today:** {today.strftime('%Y-%m-%d')}")
    st.write(f" **Tomorrow (Predicted):** {tomorrow.strftime('%Y-%m-%d')}")
    st.metric(label=" Predicted Direction", value=f"{direction} ({pred_conf:.2f}%)")
    st.metric(label=" Predicted Close Price", value=f"${pred_price:.2f}")

    return direction, pred_price, tomorrow, pred_conf

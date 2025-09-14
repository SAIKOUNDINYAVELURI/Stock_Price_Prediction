import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load models
xgb_model = joblib.load("../models/xgboost_classifier.pkl")
lstm_model = load_model("../models/lstm_model.h5")
scaler = joblib.load("../models/lstm_scaler.pkl")

# Features for XGBoost
xgb_features = [
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


def predict_next_day(data):
    latest_data = data.iloc[-1:]

    # XGBoost prediction
    X_xgb = latest_data[xgb_features]
    xgb_pred = xgb_model.predict(X_xgb)[0]
    xgb_conf = xgb_model.predict_proba(X_xgb)[0][xgb_pred]
    direction = "UP 📈" if xgb_pred == 1 else "DOWN 📉"

    # LSTM prediction
    lookback = 60
    scaled_data = scaler.transform(data[["Close"]])
    X_last = np.array([scaled_data[-lookback:, 0]]).reshape(1, lookback, 1)
    lstm_pred_scaled = lstm_model.predict(X_last)
    lstm_pred = scaler.inverse_transform(
        np.concatenate(
            [lstm_pred_scaled, np.zeros((1, scaled_data.shape[1] - 1))], axis=1
        )
    )[0, 0]

    return lstm_pred, direction, xgb_conf

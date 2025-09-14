# predict.py

import argparse
import pandas as pd
import joblib
import os
import xgboost as xgb


def predict_next_day(
    input_csv="../data/AAPL_stock_data_clean.csv",
    model_path="../models/xgboost_classifier.pkl",
):
    """
    Load trained XGBoost model and predict next day's stock direction (Up/Down).
    """
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"❌ Model not found at {model_path}. Please train first."
        )

    if model_path.endswith(".json"):
        clf = xgb.XGBClassifier()
        clf.load_model(model_path)
    else:
        clf = joblib.load(model_path)

    # Load data
    data = pd.read_csv(input_csv, index_col=0, parse_dates=True)
    data.columns = data.columns.str.strip()

    # Add features (same as training script)
    data["Lag_Close_3"] = data["Close"].shift(3)
    data["Lag_Close_5"] = data["Close"].shift(5)
    data["Lag_Close_7"] = data["Close"].shift(7)
    data["Diff_Close_Lag1"] = data["Close"] - data["Lag_Close_1"]
    data.dropna(inplace=True)

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

    # Take the most recent row as input
    X_latest = data[features].iloc[-1:].values

    # Predict
    pred = clf.predict(X_latest)[0]
    prob = clf.predict_proba(X_latest)[0]

    direction = "UP 📈" if pred == 1 else "DOWN 📉"
    confidence = prob[pred] * 100

    print("🔮 Prediction for Next Day:")
    print(f"Direction: {direction}")
    print(f"Confidence: {confidence:.2f}%")

    return direction, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict next day's stock movement using XGBoost."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../data/AAPL_stock_data_clean.csv",
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="../models/xgboost_classifier.pkl",
        help="Path to trained model.",
    )
    args = parser.parse_args()

    predict_next_day(args.input, args.model)

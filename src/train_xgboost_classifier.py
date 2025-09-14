import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_xgboost_classifier(
    input_csv="../data/AAPL_stock_data_clean.csv",
    model_path="../models/xgboost_classifier.pkl",
):
    """
    Train XGBoost classifier to predict stock price direction (Up/Down).
    - Features: OHLCV + engineered features (MA, EMA, RSI, Bollinger, Lag, Diff)
    - Target: 1 if next day's Close > today, else 0
    """

    # Load data
    data = pd.read_csv(input_csv, index_col=0, parse_dates=True)
    data.columns = data.columns.str.strip()  # remove extra spaces

    # Create Target column
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data = data[:-1]  # drop last row (NaN target)

    # Add more lag features
    data["Lag_Close_3"] = data["Close"].shift(3)
    data["Lag_Close_5"] = data["Close"].shift(5)
    data["Lag_Close_7"] = data["Close"].shift(7)

    # Difference features
    data["Diff_Close_Lag1"] = data["Close"] - data["Lag_Close_1"]

    # Drop rows with NaN from new features
    data.dropna(inplace=True)

    # Define features
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

    # Verify features exist
    missing = [c for c in features if c not in data.columns]
    if missing:
        raise KeyError(f"Missing features: {missing}")

    X = data[features]
    y = data["Target"]

    # Train-test split (time series: no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Compute class weights for imbalance
    sample_weights = class_weight.compute_sample_weight("balanced", y_train)

    # Train classifier
    clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weights)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("📊 XGBoost Classifier Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"✅ XGBoost classifier saved at: {model_path}")

    return clf


# Test
if __name__ == "__main__":
    model = train_xgboost_classifier()

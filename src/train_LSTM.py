import pandas as pd
import numpy as np
import os
import joblib
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def train_lstm(
    input_csv="../data/AAPL_stock_data_clean.csv",
    model_path="../models/lstm_model.keras",
    scaler_path="../models/lstm_scaler.pkl",
    lookback=60,
    epochs=20,
    batch_size=32,
):
    """
    Train an LSTM model to predict next day's stock Close price.

    Parameters:
    - input_csv: Path to cleaned stock CSV
    - model_path: Path to save LSTM model (.keras format)
    - scaler_path: Path to save scaler (joblib)
    - lookback: Number of past days to use for prediction
    - epochs: Number of training epochs
    - batch_size: Batch size for training

    Returns:
    - Trained LSTM model
    """

    # ------------------------------
    # 1️⃣ Load Data
    # ------------------------------
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"CSV file not found: {input_csv}")

    data = pd.read_csv(input_csv, index_col=0, parse_dates=True)
    close_prices = data["Close"].values.reshape(-1, 1)

    # ------------------------------
    # 2️⃣ Scale Data
    # ------------------------------
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # ------------------------------
    # 3️⃣ Create Sequences
    # ------------------------------
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback : i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # [samples, timesteps, features]

    # ------------------------------
    # 4️⃣ Split Train/Test
    # ------------------------------
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ------------------------------
    # 5️⃣ Build LSTM Model
    # ------------------------------
    model = Sequential()
    model.add(Input(shape=(lookback, 1)))  # Proper input layer
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")

    # ------------------------------
    # 6️⃣ Train Model
    # ------------------------------
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    # ------------------------------
    # 7️⃣ Evaluate
    # ------------------------------
    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = math.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
    print(f"📊 LSTM RMSE: {rmse:.4f}")

    # ------------------------------
    # 8️⃣ Save Model + Scaler
    # ------------------------------
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)  # Native Keras format (.keras)
    joblib.dump(scaler, scaler_path)  # Joblib for scaler

    print(f"✅ LSTM model saved at: {model_path}")
    print(f"✅ Scaler saved at: {scaler_path}")

    return model


# ------------------------------
# Test
# ------------------------------
if __name__ == "__main__":
    train_lstm()

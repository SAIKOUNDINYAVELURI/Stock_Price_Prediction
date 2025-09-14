import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

# Load trained LSTM model
model = load_model("../models/lstm_model.h5")

# Load scaler (joblib instead of pickle)
scaler = joblib.load("../models/lstm_scaler.pkl")

# Load data
data = pd.read_csv("../data/AAPL_stock_data_clean.csv", index_col=0, parse_dates=True)
close_prices = data["Close"].values.reshape(-1, 1)

# Scale the data
scaled_data = scaler.transform(close_prices)

# Prepare last 60 timesteps
lookback = 60
last_sequence = scaled_data[-lookback:]
X_input = np.expand_dims(last_sequence, axis=0)

# Predict next day scaled price
pred_scaled = model.predict(X_input)
pred_price = scaler.inverse_transform(pred_scaled)[0][0]

print("🔮 LSTM Prediction for Next Day:")
print(f"Predicted Close Price: {pred_price:.2f} USD")

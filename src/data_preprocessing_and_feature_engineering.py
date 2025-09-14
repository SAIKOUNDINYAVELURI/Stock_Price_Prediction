import pandas as pd
import os


def preprocess_stock_data(
    input_csv="../data/raw/AAPL_stock_data.csv",
    output_csv="../data/AAPL_stock_data_clean.csv",
):
    """
    Preprocess stock CSV data:
    - Rename 'Price' column to 'Date'
    - Convert Date to datetime & set as index
    - Convert numeric columns
    - Handle missing values
    - Feature engineering: MA, EMA, RSI, Bollinger Bands
    - Lag features
    - Save cleaned data
    """

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Load CSV
    data = pd.read_csv(input_csv)

    # If "Price" exists → treat it as Date
    if "Price" in data.columns:
        data.rename(columns={"Price": "Date"}, inplace=True)

    # Convert Date column to datetime and set as index
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    # Convert numeric columns
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Handle missing values
    data.ffill(inplace=True)

    # Moving averages & EMA
    data["MA_10"] = data["Close"].rolling(10).mean()
    data["EMA_10"] = data["Close"].ewm(span=10, adjust=False).mean()

    # RSI
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    data["BB_MA20"] = data["Close"].rolling(20).mean()
    data["BB_Upper"] = data["BB_MA20"] + 2 * data["Close"].rolling(20).std()
    data["BB_Lower"] = data["BB_MA20"] - 2 * data["Close"].rolling(20).std()

    # Lag features
    data["Lag_Close_1"] = data["Close"].shift(1)
    data["Lag_Close_2"] = data["Close"].shift(2)

    # Drop rows with NaN from rolling calculations
    data.dropna(inplace=True)

    # Save cleaned data
    data.to_csv(output_csv)
    print(f"✅ Preprocessed data saved to: {output_csv}")

    return data


if __name__ == "__main__":
    df = preprocess_stock_data()
    print(df.head())
    print(df.info())

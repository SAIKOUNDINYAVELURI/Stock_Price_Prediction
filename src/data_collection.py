import yfinance as yf
import os
from datetime import datetime, timedelta


def download_stock_data(
    ticker="AAPL",
    start_date="2010-01-01",
    save_path="../data/raw/",
):
    """
    Downloads historical stock data from Yahoo Finance and saves as CSV.
    """
    # End date = yesterday
    end_date = (datetime.today()).strftime("%Y-%m-%d")

    os.makedirs(save_path, exist_ok=True)

    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

    if data.empty:
        print(f"⚠️ No data found for {ticker}")
        return None

    file_path = os.path.join(save_path, f"{ticker}_stock_data.csv")
    data.to_csv(file_path)
    print(f"✅ Data for {ticker} saved successfully at {file_path}")

    return data


if __name__ == "__main__":
    df = download_stock_data()
    print(df.tail())

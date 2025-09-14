import streamlit as st
import pandas as pd


def show_today_price(data):
    st.subheader("Today's Stock Price")

    # Use "Date" column for the latest record
    if "Date" in data.columns:
        today = pd.to_datetime(data["Date"].iloc[-1]).date()
    else:
        today = "Last Record"

    today_price = data["Close"].iloc[-1]

    st.metric(label=f"Date: {today}", value=f"${today_price:.2f}")

    return today_price

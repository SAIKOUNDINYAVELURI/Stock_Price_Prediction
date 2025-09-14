import streamlit as st
import plotly.graph_objs as go


def plot_timeseries(data, ticker="AAPL"):
    """
    Interactive stock price chart with rangeslider and time range buttons.
    """
    st.subheader(f" {ticker} Historical Stock Prices")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["Date"],  # use the Date column
            y=data["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="lime", width=2),
        )
    )

    fig.update_layout(
        title=f"{ticker} Closing Price",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=450,
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1D", step="day", stepmode="backward"),
                        dict(count=5, label="5D", step="day", stepmode="backward"),
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(step="all", label="Max"),
                    ]
                )
            ),
            rangeslider=dict(visible=True),
            type="date",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

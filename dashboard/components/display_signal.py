import streamlit as st


def display_signal(signal, reason, today_price, pred_price):
    """
    Display trading signal with Streamlit metric including predicted price change.
    """
    st.subheader("📊 Trading Signal")

    # Calculate % change
    delta = pred_price - today_price
    delta_pct = (delta / today_price) * 100

    # Map signal to icon (colors handled automatically by st.metric)
    if signal == "Strong BUY":
        icon = "✅"
        delta_color = "normal"  # green for positive
    elif signal == "Weak BUY":
        icon = "🟢"
        delta_color = "normal"
    elif signal == "Strong SELL":
        icon = "❌"
        delta_color = "inverse"  # red for negative
    elif signal == "Weak SELL":
        icon = "🟠"
        delta_color = "inverse"
    else:  # HOLD
        icon = "⚪"
        delta_color = "off"  # no color

    # Display metric
    st.metric(
        label=f"{icon} {signal}",
        value=f"${pred_price:.2f}",
        delta=f"{delta_pct:.2f}%",
        delta_color=delta_color,
    )

    # Display reason
    st.write(f"**Reason:** {reason}")

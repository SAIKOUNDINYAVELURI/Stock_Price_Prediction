def generate_signal(
    data, direction, pred_price, pred_conf, rsi_threshold=(30, 70), base_tolerance=0.01
):
    """
    Generate graded Buy/Sell/Hold signals based on RSI + Model Predictions + Volatility.

    Returns:
        signal (str): "Strong BUY", "Weak BUY", "Strong SELL", "Weak SELL", or "HOLD".
        reason (str): Explanation for the signal.
    """
    today_price = data["Close"].iloc[-1]
    today_rsi = data["RSI"].iloc[-1]

    # --- Dynamic tolerance based on volatility ---
    daily_volatility = data["Close"].pct_change().rolling(10).std().iloc[-1]
    tolerance = max(base_tolerance, daily_volatility / 2)

    signal = "HOLD"
    reason = "No strong signal detected."

    # RSI conditions (override if extreme)
    if today_rsi < rsi_threshold[0]:
        signal = "Strong BUY"
        reason = f"RSI = {today_rsi:.2f} (< {rsi_threshold[0]}) → Oversold condition."
    elif today_rsi > rsi_threshold[1]:
        signal = "Strong SELL"
        reason = f"RSI = {today_rsi:.2f} (> {rsi_threshold[1]}) → Overbought condition."

    else:
        # % difference between predicted and current price
        price_diff_pct = (pred_price - today_price) / today_price

        if abs(price_diff_pct) <= tolerance:
            signal = "HOLD"
            reason = f"Predicted price change is small ({price_diff_pct:.2%}), within ±{tolerance * 100:.2f}% tolerance."
        elif "UP" in direction and pred_price > today_price:
            if pred_conf >= 85 and price_diff_pct >= 0.05:  # ≥5% move & ≥85% confidence
                signal = "Strong BUY"
                reason = f"Model predicts UP with {pred_conf:.2f}% confidence. Large move expected (+{price_diff_pct:.2%})."
            elif (
                pred_conf >= 70 and price_diff_pct >= 0.02
            ):  # 2–5% move or moderate confidence
                signal = "Weak BUY"
                reason = f"Model predicts UP with {pred_conf:.2f}% confidence. Moderate move (+{price_diff_pct:.2%})."
            else:
                signal = "HOLD"
                reason = f"Model confidence low ({pred_conf:.2f}%) or move too small (+{price_diff_pct:.2%})."
        elif "DOWN" in direction and pred_price < today_price:
            if (
                pred_conf >= 85 and price_diff_pct <= -0.05
            ):  # ≥5% drop & ≥85% confidence
                signal = "Strong SELL"
                reason = f"Model predicts DOWN with {pred_conf:.2f}% confidence. Large move expected ({price_diff_pct:.2%})."
            elif (
                pred_conf >= 70 and price_diff_pct <= -0.02
            ):  # 2–5% drop or moderate confidence
                signal = "Weak SELL"
                reason = f"Model predicts DOWN with {pred_conf:.2f}% confidence. Moderate move ({price_diff_pct:.2%})."
            else:
                signal = "HOLD"
                reason = f"Model confidence low ({pred_conf:.2f}%) or move too small ({price_diff_pct:.2%})."

    return signal, reason

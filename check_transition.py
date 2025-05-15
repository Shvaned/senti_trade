import pandas as pd
import numpy as np

sentiment = pd.read_csv("fear_greed_index.csv", parse_dates=["date"])
sentiment = (
    sentiment.set_index("date")
    .resample("D").ffill()
    .reset_index()
)

bins = [-1, 24, 44, 54, 74, 100]
labels = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
sentiment["classification"] = pd.cut(
    sentiment["value"], bins=bins, labels=labels, include_lowest=True
)

sentiment["prev_class"] = sentiment["classification"].shift(1)
sentiment["transition"] = sentiment["prev_class"].astype(str) + " → " + sentiment["classification"].astype(str)
sentiment["date"] = pd.to_datetime(sentiment["date"])


trades = pd.read_csv("historical_data.csv", parse_dates=["Timestamp IST"], dayfirst=True)
trades["trade_date"] = trades["Timestamp IST"].dt.floor("D")
trades["Net PnL"] = trades["Closed PnL"].astype(float) - trades["Fee"].astype(float)


trades = trades.merge(sentiment[["date", "transition"]], left_on="trade_date", right_on="date", how="left")


transition_stats = (
    trades.dropna(subset=["transition"])
    .groupby("transition")
    .agg(
        num_trades=("Trade ID", "nunique"),
        avg_pnl=("Net PnL", "mean"),
        win_rate=("Net PnL", lambda x: (x > 0).mean())
    )
    .sort_values("win_rate", ascending=False)
)


transition_stats.to_csv("sentiment_transition_winrate.csv")


print("\nTop Transitions by Win Rate:")
print(transition_stats.head(10))

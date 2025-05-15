import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sentiment = pd.read_csv("fear_greed_index.csv", parse_dates=["date"])
trades = pd.read_csv("historical_data.csv", parse_dates=["Timestamp IST"], dayfirst=True)

sentiment = (
    sentiment
    .set_index("date")
    .resample("D")
    .ffill()
    .reset_index()
)
bins   = [-1, 24, 44, 54, 74, 100]
labels = ["Extreme Fear","Fear","Neutral","Greed","Extreme Greed"]
sentiment["classification"] = pd.cut(sentiment["value"], bins=bins, labels=labels, include_lowest=True)
sentiment["month_day"] = sentiment["date"].dt.strftime("%m-%d")


trades["trade_date"] = trades["Timestamp IST"].dt.date
trades["month_day"]  = trades["Timestamp IST"].dt.strftime("%m-%d")
trades = trades.merge(sentiment[["month_day","value","classification"]], on="month_day", how="left")

trades["Fee"]        = trades["Fee"].astype(float)
trades["Closed PnL"] = trades["Closed PnL"].astype(float)
trades["Size USD"]   = trades["Size USD"].astype(float)
trades["Net PnL"]    = trades["Closed PnL"] - trades["Fee"]
trades["fee_ratio"]  = trades["Fee"] / trades["Size USD"]
trades["classification"] = trades["classification"].astype(str)

coin_sentiment = (
    trades
    .groupby(["Coin", "classification"], observed=True)
    .agg(
        avg_pnl=("Net PnL", "mean"),
        total_pnl=("Net PnL", "sum"),
        win_rate=("Net PnL", lambda x: (x > 0).mean()),
        num_trades=("Trade ID", "nunique")
    )
    .reset_index()
)

pivot = coin_sentiment.pivot(index="Coin", columns="classification", values="avg_pnl")
pivot.to_csv("coin_sentiment_pivot_metrics.csv", index=True)

overall_sentiment = (
    trades
    .groupby("classification")
    .agg(
        avg_pnl=("Net PnL", "mean"),
        win_rate=("Net PnL", lambda x: (x > 0).mean()),
        total_trades=("Trade ID", "nunique")
    )
    .sort_index()
)
overall_sentiment.to_csv("sentiment_summary_overall.csv")

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar_kws={'label': 'Avg Net PnL'})
plt.title("Avg Net PnL by Coin × Sentiment")
plt.tight_layout()
plt.savefig("avg_pnl_by_coin_sentiment_heatmap.png", dpi=150)
plt.close()

plt.figure(figsize=(12, 6))
sns.barplot(data=coin_sentiment, x="classification", y="win_rate", hue="Coin", errorbar=None)
plt.ylabel("Win Rate")
plt.title("Win Rate by Sentiment and Coin")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Coin")
plt.tight_layout()
plt.savefig("win_rate_by_coin_sentiment_barplot.png", dpi=150)
plt.close()

print("\n--- Sentiment-Level Performance ---")
print(overall_sentiment)

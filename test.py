# pip install requests pandas numpy matplotlib seaborn
import requests
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------
# 1) Fetch data (no API key)
# ----------------------------
URL = "https://api.coinpaprika.com/v1/tickers"
r = requests.get(URL, params={"quotes": "USD"}, timeout=20)
r.raise_for_status()
data = r.json()

rows = []
for d in data:
    rank = d.get("rank")
    usd = (d.get("quotes") or {}).get("USD") or {}
    mcap = usd.get("market_cap")
    if isinstance(rank, int) and isinstance(mcap, (int, float)) and mcap and rank > 0:
        rows.append({
            "rank": rank,
            "id": d.get("id"),
            "symbol": d.get("symbol"),
            "name": d.get("name"),
            "market_cap_usd": float(mcap)
        })

df = pd.DataFrame(rows).sort_values("rank").head(500).reset_index(drop=True)
if df.empty or len(df) < 10:
    raise RuntimeError("Not enough data returned to fit Zipf (need ≥10 assets).")

# 2) Prepare rank and market cap for Zipf fit
ranks = np.arange(1, len(df) + 1, dtype=float)
mcaps = df["market_cap_usd"].to_numpy(dtype=float)
log_r = np.log(ranks)
log_m = np.log(mcaps)

# 3) Fit log(mcap) = a + b * log(rank)
b, a = np.polyfit(log_r, log_m, 1)
C = math.exp(a)
s = -b
log_m_pred = a + b * log_r
ss_res = np.sum((log_m - log_m_pred) ** 2)
ss_tot = np.sum((log_m - np.mean(log_m)) ** 2)
r2 = 1 - ss_res / ss_tot

print("=== Zipf Fit (Top 500, USD market cap) ===")
print(f"C (scale): {C:.6g}")
print(f"s (exponent): {s:.6f}")
print(f"R^2 (log–log): {r2:.4f}")

mcap_hat = C * (ranks ** (-s))
df["zipf_mcap_hat"] = mcap_hat
df["zipf_gap_pct"] = (df["market_cap_usd"] - df["zipf_mcap_hat"]) / df["zipf_mcap_hat"] * 100.0

# 4) Show charts instead of saving
sns.set(style="whitegrid")

# A) log–log plot using seaborn
plt.figure(figsize=(7, 5))
sns.scatterplot(x=log_r, y=log_m, s=40)
xline = np.linspace(log_r.min(), log_r.max(), 200)
yline = a + b * xline
plt.plot(xline, yline, color="red", linewidth=2)
plt.title("Zipf Fit: log(Market Cap) vs log(Rank)")
plt.xlabel("log(Rank)")
plt.ylabel("log(Market Cap, USD)")
plt.tight_layout()
plt.show()

# B) rank vs market cap on log scales + fitted curve using seaborn
plt.figure(figsize=(7, 5))
sns.scatterplot(x=ranks, y=mcaps, s=40)
plt.plot(ranks, mcap_hat, color="red", linewidth=2)
plt.xscale("log")
plt.yscale("log")
plt.title("Observed vs Fitted (Zipf) — Top 500 Market Caps")
plt.xlabel("Rank")
plt.ylabel("Market Cap (USD)")
plt.tight_layout()
plt.show()

print("\nTop 10 with Zipf residuals (%):")
print(df.loc[:9, ["rank", "symbol", "market_cap_usd", "zipf_mcap_hat", "zipf_gap_pct"]])
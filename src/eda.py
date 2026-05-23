"""
UPI Fraud Detector – Exploratory Data Analysis
Generates all charts saved to outputs/ folder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_PATH   = BASE_DIR / "data" / "upi_transactions.csv"
OUTPUT_DIR  = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Styling ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor" : "#0f0f1a",
    "axes.facecolor"   : "#1a1a2e",
    "axes.edgecolor"   : "#444466",
    "axes.labelcolor"  : "#ccccee",
    "xtick.color"      : "#aaaacc",
    "ytick.color"      : "#aaaacc",
    "text.color"       : "#eeeeff",
    "grid.color"       : "#2a2a4a",
    "grid.linestyle"   : "--",
    "grid.alpha"       : 0.5,
    "font.family"      : "DejaVu Sans",
})
FRAUD_COLOR  = "#ff4d6d"
NORMAL_COLOR = "#00c9a7"
ACCENT       = "#7b5ea7"

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
print(f"✅ Loaded {len(df)} transactions")

# ════════════════════════════════════════════════════════════
# PLOT 1 – Overview Dashboard
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("UPI Transaction Fraud Analysis – Overview Dashboard",
             fontsize=18, fontweight="bold", color="#ffffff", y=1.01)

ax = axes[0, 0]
counts = df["is_fraud"].value_counts()
ax.pie(counts, labels=["Normal", "Fraud"],
       colors=[NORMAL_COLOR, FRAUD_COLOR],
       autopct="%1.1f%%", startangle=90,
       textprops={"color": "white", "fontsize": 12},
       wedgeprops={"linewidth": 2, "edgecolor": "#0f0f1a"})
ax.set_title("Fraud vs Normal Transactions", fontsize=13, fontweight="bold")

ax = axes[0, 1]
ax.hist(df[df.is_fraud==0]["amount"], bins=60, color=NORMAL_COLOR, alpha=0.7, label="Normal", log=True)
ax.hist(df[df.is_fraud==1]["amount"], bins=60, color=FRAUD_COLOR,  alpha=0.7, label="Fraud",  log=True)
ax.set_title("Transaction Amount Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Amount (₹)"); ax.set_ylabel("Count (log scale)")
ax.legend(facecolor="#1a1a2e")
ax.axvline(15000, color="yellow", linestyle="--", alpha=0.6)

ax = axes[0, 2]
hourly_fraud  = df[df.is_fraud==1].groupby("hour").size()
hourly_normal = df[df.is_fraud==0].groupby("hour").size() / 19
ax.plot(hourly_normal.index, hourly_normal.values, color=NORMAL_COLOR, linewidth=2, label="Normal (scaled)")
ax.plot(hourly_fraud.index,  hourly_fraud.values,  color=FRAUD_COLOR, linewidth=2.5, marker="o", markersize=5, label="Fraud")
ax.set_title("Fraud by Hour of Day", fontsize=13, fontweight="bold")
ax.set_xlabel("Hour"); ax.set_ylabel("Count")
ax.legend(facecolor="#1a1a2e"); ax.set_xticks(range(0, 24, 2))

ax = axes[1, 0]
city_fraud = df.groupby("sender_city")["is_fraud"].mean().sort_values(ascending=False) * 100
bars = ax.barh(city_fraud.index, city_fraud.values,
               color=[FRAUD_COLOR if v > 5 else ACCENT for v in city_fraud.values])
ax.set_title("Fraud Rate by Sender City (%)", fontsize=13, fontweight="bold")
ax.set_xlabel("Fraud Rate (%)")
for bar, val in zip(bars, city_fraud.values):
    ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va="center", fontsize=9)

ax = axes[1, 1]
type_fraud = df.groupby("transaction_type")["is_fraud"].mean().sort_values(ascending=False) * 100
bars = ax.bar(type_fraud.index, type_fraud.values,
              color=[FRAUD_COLOR, FRAUD_COLOR, ACCENT, NORMAL_COLOR, NORMAL_COLOR])
ax.set_title("Fraud Rate by Transaction Type", fontsize=13, fontweight="bold")
ax.set_xlabel("Transaction Type"); ax.set_ylabel("Fraud Rate (%)")
for bar, val in zip(bars, type_fraud.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{val:.1f}%", ha="center", fontsize=9)

ax = axes[1, 2]
nr_data = df.groupby(["is_new_recipient", "is_fraud"]).size().unstack(fill_value=0)
nr_data.plot(kind="bar", ax=ax, color=[NORMAL_COLOR, FRAUD_COLOR], edgecolor="#0f0f1a", linewidth=1.5)
ax.set_title("New Recipient vs Fraud", fontsize=13, fontweight="bold")
ax.set_xlabel("Is New Recipient (0=No, 1=Yes)"); ax.set_ylabel("Count")
ax.set_xticklabels(["Known Recipient", "New Recipient"], rotation=0)
ax.legend(["Normal", "Fraud"], facecolor="#1a1a2e")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_overview_dashboard.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("✅ Plot 1 saved")

# ════════════════════════════════════════════════════════════
# PLOT 2 – Correlation Heatmap
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#0f0f1a"); ax.set_facecolor("#1a1a2e")
num_cols = ["amount","hour","is_new_recipient","failed_attempts","is_odd_hour","cross_city","is_fraud"]
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax,
            linewidths=0.5, linecolor="#0f0f1a", cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", color="white", pad=15)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_correlation_heatmap.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("✅ Plot 2 saved")

# ════════════════════════════════════════════════════════════
# PLOT 3 – Monthly Trend
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor("#0f0f1a"); ax.set_facecolor("#1a1a2e")
monthly = df.groupby("month").agg(total=("is_fraud","count"), fraud=("is_fraud","sum")).reset_index()
monthly["fraud_rate"] = monthly["fraud"] / monthly["total"] * 100
ax2 = ax.twinx()
ax.bar(monthly["month"], monthly["total"], color=NORMAL_COLOR, alpha=0.4, label="Total Txns")
ax.bar(monthly["month"], monthly["fraud"],  color=FRAUD_COLOR,  alpha=0.8, label="Fraud Txns")
ax2.plot(monthly["month"], monthly["fraud_rate"], color="yellow", linewidth=2.5, marker="D", markersize=7, label="Fraud Rate %")
ax.set_title("Monthly Transaction Volume & Fraud Trend (2023)", fontsize=15, fontweight="bold")
ax.set_xlabel("Month"); ax.set_ylabel("Transaction Count")
ax2.set_ylabel("Fraud Rate (%)", color="yellow"); ax2.tick_params(colors="yellow")
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
ax.set_xticks(range(1,13)); ax.set_xticklabels(months)
ax.legend(loc="upper left", facecolor="#1a1a2e")
ax2.legend(loc="upper right", facecolor="#1a1a2e")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_monthly_trend.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("✅ Plot 3 saved")

print("\n📊 EDA complete! All charts saved to outputs/")

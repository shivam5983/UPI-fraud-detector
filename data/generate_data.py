"""
UPI Transaction Dataset Generator
Generates realistic synthetic UPI transaction data with fraud cases
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import random

np.random.seed(42)
random.seed(42)

# ── Paths (works on Windows + Linux + Mac) ─────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─── Config ───────────────────────────────────────────────
N_TRANSACTIONS = 10000
FRAUD_RATIO     = 0.05

CITIES    = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai",
             "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"]
BANKS     = ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB", "BOB", "Yes Bank"]
MERCHANTS = ["Swiggy", "Zomato", "Amazon", "Flipkart", "PhonePe",
             "Paytm", "BigBasket", "Myntra", "BookMyShow", "MakeMyTrip",
             "Individual", "Electricity Board", "LIC", "Rent Payment"]
TX_TYPES  = ["P2P", "P2M", "Bill Payment", "Recharge", "Investment"]

start_date = datetime(2023, 1, 1)
rows = []

# Normal Transactions
for _ in range(int(N_TRANSACTIONS * (1 - FRAUD_RATIO))):
    ts = start_date + timedelta(
             days=random.randint(0, 364),
             hours=random.randint(7, 23),
             minutes=random.randint(0, 59))
    rows.append({
        "transaction_id"   : f"TXN{random.randint(1000000,9999999)}",
        "timestamp"        : ts,
        "amount"           : round(np.random.lognormal(mean=5.5, sigma=1.2), 2),
        "sender_city"      : random.choice(CITIES),
        "receiver_city"    : random.choice(CITIES),
        "sender_bank"      : random.choice(BANKS),
        "receiver_bank"    : random.choice(BANKS),
        "merchant_category": random.choice(MERCHANTS),
        "transaction_type" : random.choice(TX_TYPES),
        "device_type"      : random.choices(["Android", "iOS", "Android", "Android"])[0],
        "is_new_recipient" : random.choices([0, 1], weights=[0.85, 0.15])[0],
        "failed_attempts"  : random.choices([0, 1, 2], weights=[0.92, 0.06, 0.02])[0],
        "is_fraud"         : 0
    })

# Fraudulent Transactions
for _ in range(int(N_TRANSACTIONS * FRAUD_RATIO)):
    ts = start_date + timedelta(
             days=random.randint(0, 364),
             hours=random.choices([0,1,2,3,4,22,23], weights=[2,2,2,2,1,2,2])[0],
             minutes=random.randint(0, 59))
    rows.append({
        "transaction_id"   : f"TXN{random.randint(1000000,9999999)}",
        "timestamp"        : ts,
        "amount"           : round(random.uniform(15000, 200000), 2),
        "sender_city"      : random.choice(CITIES),
        "receiver_city"    : random.choice(CITIES),
        "sender_bank"      : random.choice(BANKS),
        "receiver_bank"    : random.choice(BANKS),
        "merchant_category": random.choice(["Individual", "Individual", "Paytm", "PhonePe"]),
        "transaction_type" : random.choice(["P2P", "P2P", "P2M"]),
        "device_type"      : random.choice(["Android", "iOS"]),
        "is_new_recipient" : random.choices([0, 1], weights=[0.2, 0.8])[0],
        "failed_attempts"  : random.choices([0, 1, 2, 3], weights=[0.3, 0.3, 0.2, 0.2])[0],
        "is_fraud"         : 1
    })

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df["timestamp"]   = pd.to_datetime(df["timestamp"])
df["hour"]        = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_name()
df["month"]       = df["timestamp"].dt.month
df["is_odd_hour"] = df["hour"].apply(lambda h: 1 if h < 6 or h >= 22 else 0)
df["cross_city"]  = (df["sender_city"] != df["receiver_city"]).astype(int)

out_path = DATA_DIR / "upi_transactions.csv"
df.to_csv(out_path, index=False)
print(f"✅ Dataset generated: {len(df)} transactions | Fraud: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
print(f"   Saved to: {out_path}")

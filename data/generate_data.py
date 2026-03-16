"""
generate_data.py
----------------
Generates a synthetic financial transaction dataset for testing and development.

Usage:
    python data/generate_data.py
"""

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta


TRANSACTION_TYPES      = ["purchase", "transfer", "withdrawal", "deposit", "payment"]
MERCHANT_CATEGORIES    = [
    "grocery", "electronics", "restaurant", "gas_station",
    "clothing", "healthcare", "entertainment", "travel", "online",
]
DOMESTIC_LOCATIONS     = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
]
HIGH_RISK_LOCATIONS    = ["Foreign"]


def _base_record(i: int, customer_id: str, amount: float, hour: int,
                 location: str, tx_type: str, merchant: str,
                 balance: float, base_time: datetime, is_anomaly: int) -> dict:
    time_offset = timedelta(
        days=np.random.randint(0, 365),
        hours=hour,
        minutes=np.random.randint(0, 60),
    )
    return {
        "transaction_id":     f"TXN_{i:06d}",
        "customer_id":        customer_id,
        "transaction_amount": round(amount, 2),
        "transaction_time":   (base_time + time_offset).strftime("%Y-%m-%d %H:%M:%S"),
        "transaction_type":   tx_type,
        "merchant_category":  merchant,
        "location":           location,
        "account_balance":    round(balance, 2),
        "is_anomaly":         is_anomaly,
    }


def generate_transactions(n_normal: int = 1000, n_anomalous: int = 50, seed: int = 42) -> pd.DataFrame:
    """Return a shuffled DataFrame of n_normal + n_anomalous synthetic transactions."""
    np.random.seed(seed)
    random.seed(seed)

    base_time = datetime(2024, 1, 1)
    records: list[dict] = []

    # ── Normal transactions ───────────────────────────────────────────────
    for i in range(1, n_normal + 1):
        cid    = f"CUST_{np.random.randint(1, 201):04d}"
        amount = abs(np.random.normal(150, 80))
        amount = min(amount, 800)           # cap at $800 for normal txns
        hour   = np.random.randint(8, 22)   # business hours
        loc    = random.choice(DOMESTIC_LOCATIONS)
        bal    = np.random.uniform(500, 15_000)
        records.append(_base_record(
            i, cid, amount, hour, loc,
            random.choice(TRANSACTION_TYPES[:4]),
            random.choice(MERCHANT_CATEGORIES[:6]),
            bal, base_time, 0,
        ))

    # ── Anomalous transactions ────────────────────────────────────────────
    anomaly_patterns = ["high_amount", "odd_hours", "foreign_location", "rapid_sequence"]
    for j in range(1, n_anomalous + 1):
        cid     = f"CUST_{np.random.randint(1, 201):04d}"
        pattern = random.choice(anomaly_patterns)

        if pattern == "high_amount":
            amount = np.random.uniform(5_000, 50_000)
            hour   = np.random.randint(8, 22)
            loc    = random.choice(DOMESTIC_LOCATIONS)
            bal    = np.random.uniform(0, 500)

        elif pattern == "odd_hours":
            amount = np.random.uniform(200, 3_000)
            hour   = np.random.randint(0, 5)        # midnight–5 am
            loc    = random.choice(DOMESTIC_LOCATIONS)
            bal    = np.random.uniform(100, 2_000)

        elif pattern == "foreign_location":
            amount = np.random.uniform(500, 8_000)
            hour   = np.random.randint(0, 24)
            loc    = "Foreign"
            bal    = np.random.uniform(0, 500)

        else:   # rapid_sequence
            amount = np.random.uniform(200, 1_000)
            hour   = np.random.randint(0, 24)
            loc    = random.choice(DOMESTIC_LOCATIONS + HIGH_RISK_LOCATIONS)
            bal    = np.random.uniform(0, 300)

        records.append(_base_record(
            n_normal + j, cid, amount, hour, loc,
            random.choice(TRANSACTION_TYPES),
            random.choice(MERCHANT_CATEGORIES),
            bal, base_time, 1,
        ))

    df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_transactions(n_normal=1000, n_anomalous=50)
    out_dir = os.path.dirname(__file__)

    # Save labeled version (used for model evaluation)
    labeled_path = os.path.join(out_dir, "transactions_labeled.csv")
    df.to_csv(labeled_path, index=False)
    print(f"Labeled dataset saved  → {labeled_path}")

    # Save unlabeled version (simulates real-world input)
    unlabeled_path = os.path.join(out_dir, "transactions.csv")
    df.drop(columns=["is_anomaly"]).to_csv(unlabeled_path, index=False)
    print(f"Unlabeled dataset saved → {unlabeled_path}")

    print(f"\nTotal records : {len(df)}")
    print(f"Normal        : {(df['is_anomaly'] == 0).sum()}")
    print(f"Anomalous     : {(df['is_anomaly'] == 1).sum()}")
    print(df.head(5).to_string(index=False))

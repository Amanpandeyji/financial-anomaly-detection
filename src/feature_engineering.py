"""
feature_engineering.py
-----------------------
Derives behavioural and statistical features from the preprocessed transaction
DataFrame.  All operations are performed in-place on a copy of the input so
the caller's DataFrame is never mutated.

Features created
────────────────
tx_frequency         - how many transactions a customer has overall
avg_spending         - customer's mean transaction amount
spending_deviation   - z-score of this transaction vs the customer average
max_daily_tx         - customer's busiest day transaction count
amount_to_balance    - transaction amount ÷ account balance (leverage proxy)
time_since_last_tx   - hours since the same customer's prior transaction
is_high_risk_location- 1 if the transaction is tagged as "Foreign"
rolling_avg_amount   - 5-transaction rolling mean per customer
"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Adds behavioural and statistical features to a transaction DataFrame."""

    HIGH_RISK_LOCATIONS = {"Foreign"}

    # ─────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run every feature-creation step and return the augmented DataFrame."""
        print("[FeatureEngineer] Starting feature engineering ...")
        df = df.copy()
        df = self._transaction_frequency(df)
        df = self._average_spending(df)
        df = self._spending_deviation(df)
        df = self._max_daily_transactions(df)
        df = self._amount_to_balance_ratio(df)
        df = self._time_since_last_transaction(df)
        df = self._high_risk_location(df)
        df = self._rolling_avg_amount(df)
        print(f"[FeatureEngineer] Done -- DataFrame now has {len(df.columns)} columns.")
        return df

    def get_feature_columns(self) -> list[str]:
        """
        Return the ordered list of feature column names expected by the model.
        These columns are guaranteed to exist after calling engineer_features()
        on a preprocessed DataFrame.
        """
        return [
            # ── from preprocessing ──────────────────────────────────────
            "transaction_amount_normalized",
            "account_balance_normalized",
            "hour",
            "day_of_week",
            "is_weekend",
            "is_night",
            "transaction_type_encoded",
            "merchant_category_encoded",
            "location_encoded",
            # ── engineered ──────────────────────────────────────────────
            "tx_frequency",
            "avg_spending",
            "spending_deviation",
            "max_daily_tx",
            "amount_to_balance_ratio",
            "time_since_last_tx",
            "is_high_risk_location",
            "rolling_avg_amount",
        ]

    # ─────────────────────────────────────────────────────────────────────
    # Individual feature steps
    # ─────────────────────────────────────────────────────────────────────

    def _transaction_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Number of transactions per customer in the dataset."""
        freq = df.groupby("customer_id")["transaction_id"].transform("count")
        df["tx_frequency"] = freq
        return df

    def _average_spending(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mean transaction amount per customer."""
        df["avg_spending"] = df.groupby("customer_id")["transaction_amount"].transform("mean")
        return df

    def _spending_deviation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Z-score of each transaction relative to the customer's spending history.
        Returns 0 for customers with only one transaction.
        """
        mean = df.groupby("customer_id")["transaction_amount"].transform("mean")
        std  = df.groupby("customer_id")["transaction_amount"].transform("std").fillna(0)
        df["spending_deviation"] = np.where(std > 0, (df["transaction_amount"] - mean) / std, 0.0)
        return df

    def _max_daily_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maximum number of transactions this customer performed on any single day."""
        if "transaction_time" not in df.columns:
            df["max_daily_tx"] = 1
            return df

        temp = df[["customer_id", "transaction_time"]].copy()
        temp["_date"] = pd.to_datetime(temp["transaction_time"]).dt.date
        daily = temp.groupby(["customer_id", "_date"]).size().rename("_daily_cnt")
        max_daily = daily.groupby(level="customer_id").max().rename("max_daily_tx")
        df = df.join(max_daily, on="customer_id")
        return df

    def _amount_to_balance_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ratio of transaction amount to account balance.
        A very high ratio (spending more than you have) is a fraud signal.
        """
        df["amount_to_balance_ratio"] = np.where(
            df["account_balance"] > 0,
            df["transaction_amount"] / df["account_balance"],
            df["transaction_amount"],   # treat zero balance as divisor = 1
        )
        return df

    def _time_since_last_transaction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Hours elapsed since the customer's immediately preceding transaction.
        First transaction per customer gets the dataset-wide median.
        """
        if "transaction_time" not in df.columns:
            df["time_since_last_tx"] = 0.0
            return df

        original_index = df.index
        df_s = df.sort_values(["customer_id", "transaction_time"])
        delta = (
            df_s.groupby("customer_id")["transaction_time"]
            .diff()
            .dt.total_seconds()
            .div(3600)
        )
        median_val = delta.median()
        df_s["time_since_last_tx"] = delta.fillna(median_val if pd.notna(median_val) else 24.0)
        df = df_s.reindex(original_index)
        return df

    def _high_risk_location(self, df: pd.DataFrame) -> pd.DataFrame:
        """Binary flag: 1 if the transaction location is in the high-risk set."""
        df["is_high_risk_location"] = df["location"].isin(self.HIGH_RISK_LOCATIONS).astype(int)
        return df

    def _rolling_avg_amount(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Per-customer rolling mean of the last `window` transaction amounts,
        ordered by time.  A sudden spike above this rolling mean is suspicious.
        """
        if "transaction_time" not in df.columns:
            df["rolling_avg_amount"] = df["transaction_amount"]
            return df

        original_index = df.index
        df_s = df.sort_values(["customer_id", "transaction_time"])
        df_s["rolling_avg_amount"] = (
            df_s.groupby("customer_id")["transaction_amount"]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
        df = df_s.reindex(original_index)
        return df

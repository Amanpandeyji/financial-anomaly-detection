"""
data_preprocessing.py
---------------------
Handles all data-cleaning and feature-scaling steps before model training.

Pipeline:
    load_data -> remove_duplicates -> handle_missing_values
    -> parse_datetime -> encode_categoricals -> normalize_amounts
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    """Full preprocessing pipeline for financial transaction data."""

    CATEGORICAL_COLS = ["transaction_type", "merchant_category", "location"]
    NUMERICAL_COLS   = ["transaction_amount", "account_balance"]

    def __init__(self):
        self.scalers: dict[str, StandardScaler] = {}
        self.label_encoders: dict[str, LabelEncoder] = {}

    # ─────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────

    def preprocess(self, source, fit: bool = True) -> pd.DataFrame:
        """
        Run the full preprocessing pipeline.

        Args:
            source: Path to CSV file  OR  a pandas DataFrame.
            fit:    True when processing training data (fits scalers/encoders).
                    False when processing new/unseen data.

        Returns:
            Preprocessed DataFrame with new derived columns.
        """
        df = self.load_data(source) if isinstance(source, str) else source.copy()
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        df = self.parse_datetime(df)
        df = self.encode_categoricals(df, fit=fit)
        df = self.normalize_amounts(df, fit=fit)
        print(f"[Preprocessor] Done. Shape: {df.shape}")
        return df

    # ─────────────────────────────────────────────────────────────────────
    # Steps
    # ─────────────────────────────────────────────────────────────────────

    def load_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        print(f"[Preprocessor] Loaded {len(df):,} rows × {len(df.columns)} cols from '{filepath}'")
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        if "transaction_id" in df.columns:
            df = df.drop_duplicates(subset=["transaction_id"])
        removed = before - len(df)
        if removed:
            print(f"[Preprocessor] Removed {removed} duplicate transaction_id rows.")
        return df.reset_index(drop=True)

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = df.isnull().sum()
        total_missing = missing.sum()
        if total_missing == 0:
            print("[Preprocessor] No missing values -- skipping imputation.")
            return df

        print(f"[Preprocessor] Missing values detected ({total_missing} total):")
        print(missing[missing > 0].to_string())

        # Numerical -> median imputation
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())

        # Categorical -> mode imputation
        for col in df.select_dtypes(include="object").columns:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown")

        return df

    def parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        if "transaction_time" not in df.columns:
            return df

        df["transaction_time"] = pd.to_datetime(df["transaction_time"], errors="coerce")

        df["hour"]        = df["transaction_time"].dt.hour
        df["day_of_week"] = df["transaction_time"].dt.dayofweek   # Mon=0, Sun=6
        df["month"]       = df["transaction_time"].dt.month
        df["day"]         = df["transaction_time"].dt.day
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
        df["is_night"]    = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

        print("[Preprocessor] DateTime features extracted: hour, day_of_week, month, day, is_weekend, is_night")
        return df

    def encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        for col in self.CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is None:
                    raise RuntimeError(f"Encoder for '{col}' not fitted. Call preprocess(fit=True) first.")
                # Unknown categories get -1
                df[col + "_encoded"] = df[col].astype(str).map(
                    lambda x, _le=le: int(_le.transform([x])[0]) if x in _le.classes_ else -1
                )
        print(f"[Preprocessor] Encoded categoricals: {self.CATEGORICAL_COLS}")
        return df

    def normalize_amounts(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        for col in self.NUMERICAL_COLS:
            if col not in df.columns:
                continue
            if fit:
                scaler = StandardScaler()
                df[col + "_normalized"] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
            else:
                scaler = self.scalers.get(col)
                if scaler is None:
                    raise RuntimeError(f"Scaler for '{col}' not fitted. Call preprocess(fit=True) first.")
                df[col + "_normalized"] = scaler.transform(df[[col]])
        print(f"[Preprocessor] Normalized: {self.NUMERICAL_COLS}")
        return df

    # ─────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"scalers": self.scalers, "label_encoders": self.label_encoders}, f)
        print(f"[Preprocessor] Saved state -> {filepath}")

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        self.scalers        = state["scalers"]
        self.label_encoders = state["label_encoders"]
        print(f"[Preprocessor] Loaded state <- {filepath}")

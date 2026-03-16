"""
anomaly_detection.py
--------------------
Three anomaly detection models wrapped in a unified API:

    • IsolationForest   - tree-based isolation of rare observations
    • OneClassSVM       - kernel-based boundary around normal data
    • LocalOutlierFactor - density-based (novelty=True for predict support)

All models expose the same interface:
    fit(X), predict(X), score_samples(X), get_risk_scores(X), detect(df, cols)
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# ─────────────────────────────────────────────────────────────────────────────
# Core detector class
# ─────────────────────────────────────────────────────────────────────────────


class AnomalyDetector:
    """
    Unified wrapper around three sklearn anomaly detection algorithms.

    Parameters
    ----------
    model_type    : 'isolation_forest' | 'one_class_svm' | 'local_outlier_factor'
    contamination : expected fraction of anomalies in the data  (0.01 - 0.50)
    random_state  : reproducibility seed (ignored for OneClassSVM)
    """

    SUPPORTED_MODELS = ("isolation_forest", "one_class_svm", "local_outlier_factor")

    def __init__(
        self,
        model_type: str = "isolation_forest",
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from {self.SUPPORTED_MODELS}."
            )
        self.model_type    = model_type
        self.contamination = contamination
        self.random_state  = random_state
        self.scaler        = StandardScaler()
        self.model         = self._build_model()
        self._is_fitted    = False

    # ── Construction ─────────────────────────────────────────────────────

    def _build_model(self):
        if self.model_type == "isolation_forest":
            return IsolationForest(
                n_estimators=200,
                contamination=self.contamination,
                random_state=self.random_state,
                max_features=1.0,
                bootstrap=False,
            )
        if self.model_type == "one_class_svm":
            return OneClassSVM(
                kernel="rbf",
                nu=self.contamination,
                gamma="scale",
            )
        # local_outlier_factor with novelty=True supports predict() on new data
        return LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination,
            novelty=True,
        )

    # ── Training ─────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        """
        Scale and fit the model on feature matrix X.

        Args:
            X: 2-D numpy array of shape (n_samples, n_features).

        Returns:
            self  (for chaining)
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._is_fitted = True
        print(
            f"[AnomalyDetector] '{self.model_type}' fitted on "
            f"{X_scaled.shape[0]:,} samples × {X_scaled.shape[1]} features."
        )
        return self

    # ── Inference ────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels.  Follows sklearn convention:
            +1 = normal,  -1 = anomaly
        """
        return self.model.predict(self.scaler.transform(X))

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Raw anomaly scores (more negative -> more anomalous) for all three models.
        IsolationForest  : score_samples()
        OneClassSVM      : score_samples()   (decision_function, same interface)
        LOF (novelty)    : score_samples()
        """
        return self.model.score_samples(self.scaler.transform(X))

    def get_risk_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Normalise raw anomaly scores to a 0-100 risk scale.
        100 = most anomalous,  0 = most normal.
        """
        raw      = self.score_samples(X)
        inverted = -raw                        # flip: higher = more suspicious
        lo, hi   = inverted.min(), inverted.max()
        if hi > lo:
            return ((inverted - lo) / (hi - lo)) * 100
        return np.zeros_like(inverted)

    # ── End-to-end detection ─────────────────────────────────────────────

    def detect(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """
        Run full detection on a DataFrame and add result columns.

        Adds columns:
            prediction         - +1 normal / -1 anomalous
            risk_score         - 0-100 normalised risk
            is_anomaly_detected- 0 or 1  (1 = anomalous)

        Args:
            df          : DataFrame containing all feature columns.
            feature_cols: List of column names to pass to the model.

        Returns:
            Copy of df with three new columns appended.
        """
        X = df[feature_cols].fillna(0).values
        result = df.copy()
        result["prediction"]          = self.predict(X)
        result["risk_score"]          = self.get_risk_scores(X)
        result["is_anomaly_detected"] = (result["prediction"] == -1).astype(int)

        n = result["is_anomaly_detected"].sum()
        print(
            f"[AnomalyDetector] {n:,} anomalies detected "
            f"({n / len(result) * 100:.1f}% of {len(result):,} transactions)."
        )
        return result

    # ── Evaluation ───────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame, true_label_col: str = "is_anomaly") -> None:
        """
        Print a classification report if ground-truth labels exist.

        Args:
            df             : DataFrame with 'is_anomaly_detected' column.
            true_label_col : Column of ground-truth labels (0 = normal, 1 = anomaly).
        """
        if true_label_col not in df.columns:
            print(f"[AnomalyDetector] Column '{true_label_col}' not found -- skipping evaluation.")
            return

        y_true = df[true_label_col].values
        y_pred = df["is_anomaly_detected"].values

        sep = "=" * 52
        print(f"\n{sep}")
        print(f"  EVALUATION  --  {self.model_type.upper()}")
        print(sep)
        print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

        cm = confusion_matrix(y_true, y_pred)
        print(f"Confusion matrix:\n{cm}")

        try:
            auc = roc_auc_score(y_true, df["risk_score"].values / 100)
            print(f"ROC-AUC : {auc:.4f}")
        except Exception:
            pass

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            "model":         self.model,
            "scaler":        self.scaler,
            "model_type":    self.model_type,
            "contamination": self.contamination,
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f)
        print(f"[AnomalyDetector] Model saved -> {filepath}")

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            payload = pickle.load(f)
        self.model         = payload["model"]
        self.scaler        = payload["scaler"]
        self.model_type    = payload["model_type"]
        self.contamination = payload["contamination"]
        self._is_fitted    = True
        print(f"[AnomalyDetector] Model loaded <- {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────


def visualize_anomalies(df: pd.DataFrame, output_dir: str | None = None) -> plt.Figure:
    """
    Produce a 2×3 grid of plots summarising the anomaly detection results.

    Args:
        df         : DataFrame that includes 'is_anomaly_detected' and 'risk_score'.
        output_dir : If provided, saves 'anomaly_analysis.png' to that directory.

    Returns:
        The matplotlib Figure object.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "Financial Transaction Anomaly Detection -- Analysis Dashboard",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    normal_df  = df[df["is_anomaly_detected"] == 0]
    anomaly_df = df[df["is_anomaly_detected"] == 1]
    BLUE, RED  = "#2980b9", "#e74c3c"

    # ── 1. Transaction amount distribution ───────────────────────────────
    ax = axes[0, 0]
    ax.hist(normal_df["transaction_amount"],  bins=50, alpha=0.70, color=BLUE, label="Normal",  edgecolor="white", linewidth=0.4)
    ax.hist(anomaly_df["transaction_amount"], bins=30, alpha=0.85, color=RED,  label="Anomaly", edgecolor="white", linewidth=0.4)
    ax.set_title("Transaction Amount Distribution")
    ax.set_xlabel("Amount ($)")
    ax.set_ylabel("Frequency (log)")
    ax.set_yscale("log")
    ax.legend()

    # ── 2. Risk score histogram ───────────────────────────────────────────
    ax = axes[0, 1]
    ax.hist(df["risk_score"], bins=50, color="#e67e22", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(x=70, color=RED, linestyle="--", linewidth=1.8, label="High-risk threshold (70)")
    ax.set_title("Risk Score Distribution")
    ax.set_xlabel("Risk Score (0 - 100)")
    ax.set_ylabel("Frequency")
    ax.legend()

    # ── 3. Transactions by hour of day ────────────────────────────────────
    ax = axes[0, 2]
    if "hour" in df.columns:
        h_normal  = normal_df.groupby("hour").size()
        h_anomaly = anomaly_df.groupby("hour").size()
        ax.bar(h_normal.index,  h_normal.values,  alpha=0.70, color=BLUE, label="Normal")
        ax.bar(h_anomaly.index, h_anomaly.values, alpha=0.90, color=RED,  label="Anomaly")
        ax.set_title("Transactions by Hour of Day")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Count")
        ax.legend()

    # ── 4. Amount vs Risk Score scatter ───────────────────────────────────
    ax = axes[1, 0]
    ax.scatter(normal_df["transaction_amount"],  normal_df["risk_score"],
               c=BLUE, alpha=0.40, s=12, label="Normal")
    ax.scatter(anomaly_df["transaction_amount"], anomaly_df["risk_score"],
               c=RED,  alpha=0.80, s=30, marker="X", label="Anomaly")
    ax.axhline(y=70, color=RED, linestyle="--", linewidth=1.2)
    ax.set_title("Transaction Amount vs Risk Score")
    ax.set_xlabel("Amount ($)")
    ax.set_ylabel("Risk Score")
    ax.legend()

    # ── 5. Anomalies by transaction type ──────────────────────────────────
    ax = axes[1, 1]
    if "transaction_type" in df.columns:
        counts = (
            df.groupby(["transaction_type", "is_anomaly_detected"])
            .size()
            .unstack(fill_value=0)
            .rename(columns={0: "Normal", 1: "Anomaly"})
        )
        counts.plot(kind="bar", ax=ax, color=[BLUE, RED], alpha=0.85, edgecolor="white")
        ax.set_title("Anomalies by Transaction Type")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        ax.legend()

    # ── 6. Account balance vs Transaction amount ──────────────────────────
    ax = axes[1, 2]
    ax.scatter(normal_df["account_balance"],  normal_df["transaction_amount"],
               c=BLUE, alpha=0.35, s=10, label="Normal")
    ax.scatter(anomaly_df["account_balance"], anomaly_df["transaction_amount"],
               c=RED,  alpha=0.80, s=30, marker="X", label="Anomaly")
    ax.set_title("Account Balance vs Transaction Amount")
    ax.set_xlabel("Account Balance ($)")
    ax.set_ylabel("Transaction Amount ($)")
    ax.legend()

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "anomaly_analysis.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Visualizer] Plot saved -> {path}")

    plt.show()
    return fig


def train_and_evaluate_all_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.05,
) -> tuple[dict, dict]:
    """
    Train all three anomaly detectors, evaluate (if labels present), and return
    a dict of fitted AnomalyDetector objects and result DataFrames.

    Returns:
        (models_dict, results_dict)  keyed by model name.
    """
    models: dict  = {}
    results: dict = {}

    for model_type in AnomalyDetector.SUPPORTED_MODELS:
        print(f"\n{'─' * 52}")
        print(f"  Training -> {model_type.upper()}")
        print("─" * 52)

        detector = AnomalyDetector(model_type=model_type, contamination=contamination)
        detector.fit(df[feature_cols].fillna(0).values)
        df_res = detector.detect(df, feature_cols)

        if "is_anomaly" in df.columns:
            detector.evaluate(df_res)

        models[model_type]  = detector
        results[model_type] = df_res

    return models, results

"""
main.py
-------
Command-line interface for the Financial Transaction Anomaly Detection System.

Usage examples
──────────────
# Generate sample data, then run Isolation Forest:
    python data/generate_data.py
    python main.py

# Use a custom CSV and choose a model:
    python main.py --data path/to/file.csv --model one_class_svm

# Compare all three models at once:
    python main.py --compare

# Tune the contamination rate:
    python main.py --contamination 0.03
"""

import argparse
import os
import sys

import pandas as pd

# ── Make src importable when running from the project root ──────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from anomaly_detection import AnomalyDetector, train_and_evaluate_all_models, visualize_anomalies  # type: ignore
from data_preprocessing import DataPreprocessor  # type: ignore
from feature_engineering import FeatureEngineer  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI-Based Financial Transaction Anomaly Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        default="data/transactions.csv",
        help="Path to the input transaction CSV file.",
    )
    p.add_argument(
        "--model",
        default="isolation_forest",
        choices=["isolation_forest", "one_class_svm", "local_outlier_factor"],
        help="Anomaly detection algorithm to use.",
    )
    p.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Expected fraction of anomalous transactions (0.01 - 0.50).",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Train and evaluate all three models side-by-side.",
    )
    p.add_argument(
        "--output",
        default="results/suspicious_transactions.csv",
        help="Output CSV path for suspicious transactions.",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving the visualisation plot.",
    )
    return p.parse_args()


def _validate_features(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Feature columns not found (will be skipped): {missing}")
    print(f"[INFO] Using {len(available)} features.")
    return available


def main() -> None:
    args = parse_args()

    banner = "=" * 58
    print(f"\n{banner}")
    print(" AI Financial Transaction Anomaly Detection System")
    print(banner)

    # ── Step 1: Preprocessing ────────────────────────────────────────────
    print("\n[STEP 1] Data Preprocessing")
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess(args.data, fit=True)
    os.makedirs("models", exist_ok=True)
    preprocessor.save("models/preprocessor.pkl")

    # ── Step 2: Feature Engineering ──────────────────────────────────────
    print("\n[STEP 2] Feature Engineering")
    engineer     = FeatureEngineer()
    df           = engineer.engineer_features(df)
    feature_cols = _validate_features(df, engineer.get_feature_columns())

    # ── Step 3: Train & Detect ───────────────────────────────────────────
    if args.compare:
        print("\n[STEP 3] Comparing All Models")
        models, results = train_and_evaluate_all_models(
            df, feature_cols, contamination=args.contamination
        )
        # Use Isolation Forest result for downstream reporting by default
        df_result = results["isolation_forest"]
        detector  = models["isolation_forest"]
    else:
        print(f"\n[STEP 3] Training -> {args.model.upper()}")
        detector  = AnomalyDetector(
            model_type=args.model, contamination=args.contamination
        )
        detector.fit(df[feature_cols].fillna(0).values)
        detector.save("models/trained_model.pkl")

        print("\n[STEP 4] Detecting Anomalies")
        df_result = detector.detect(df, feature_cols)

    # ── Step 4: Evaluation (if ground-truth labels are available) ────────
    labeled_path = args.data.replace("transactions.csv", "transactions_labeled.csv")
    if os.path.exists(labeled_path):
        print("\n[STEP 5] Evaluating against ground-truth labels")
        labeled = pd.read_csv(labeled_path)
        if "transaction_id" in df_result.columns and "transaction_id" in labeled.columns:
            label_map = labeled.set_index("transaction_id")["is_anomaly"].to_dict()
            df_result["is_anomaly"] = df_result["transaction_id"].map(label_map).fillna(0).astype(int)
        detector.evaluate(df_result)

    # ── Step 5: Save suspicious transactions ─────────────────────────────
    print(f"\n[STEP 6] Saving results -> {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    suspicious = df_result[df_result["is_anomaly_detected"] == 1].sort_values(
        "risk_score", ascending=False
    )
    keep = [c for c in ["transaction_id", "customer_id", "transaction_amount",
                         "transaction_time", "transaction_type", "merchant_category",
                         "location", "account_balance", "risk_score"]
            if c in suspicious.columns]
    suspicious[keep].to_csv(args.output, index=False)

    # ── Step 6: Visualisation ────────────────────────────────────────────
    if not args.no_plot:
        print("\n[STEP 7] Generating visualisation")
        visualize_anomalies(df_result, output_dir="results")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{banner}")
    print(" ANALYSIS COMPLETE")
    print(f"{banner}")
    print(f"  Total transactions analysed : {len(df_result):,}")
    print(f"  Anomalies detected          : {df_result['is_anomaly_detected'].sum():,}")
    print(f"  Avg risk score (anomalies)  : "
          f"{df_result[df_result['is_anomaly_detected']==1]['risk_score'].mean():.1f}")
    print(f"  Suspicious CSV saved to     : {args.output}")
    print(f"  Plot saved to               : results/anomaly_analysis.png")
    print(banner + "\n")


if __name__ == "__main__":
    main()

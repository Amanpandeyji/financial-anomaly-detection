#AI-Based Financial Transaction Anomaly Detection System
A production-ready machine learning system that detects unusual or suspicious financial transactions using three unsupervised anomaly detection algorithms.

> **Live demo** → deploy to [Streamlit Community Cloud](https://streamlit.io/cloud) in one click (see [Deployment](#-deployment) section below).

---

## Project Structure

```
financial-anomaly-detection/
│
├── data/
│   ├── generate_data.py          ← Synthetic dataset generator
│   ├── transactions.csv          ← Unlabeled input (generated)
│   └── transactions_labeled.csv  ← Ground-truth labels (generated)
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     ← Load, clean, encode, normalize
│   ├── feature_engineering.py    ← Behavioural & statistical features
│   └── anomaly_detection.py      ← Isolation Forest / One-Class SVM / LOF
│
├── dashboard/
│   └── app.py                    ← Streamlit interactive dashboard
│
├── models/
│   ├── trained_model.pkl         ← Saved detector (created at runtime)
│   └── preprocessor.pkl          ← Saved scaler/encoders (created at runtime)
│
├── notebooks/
│   └── analysis.ipynb            ← Step-by-step Jupyter analysis
│
├── results/
│   ├── suspicious_transactions.csv
│   └── anomaly_analysis.png
│
├── main.py                       ← CLI entry point
├── requirements.txt
└── README.md
```

---

##  Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate sample data

```bash
python data/generate_data.py
```

This creates `data/transactions.csv` (1,000 normal + 50 anomalous, unlabeled) and `data/transactions_labeled.csv` (with `is_anomaly` column for evaluation).

### 3a. Run the CLI pipeline

```bash
# Default: Isolation Forest, contamination=5%
python main.py

# Choose a different model
python main.py --model one_class_svm
python main.py --model local_outlier_factor

# Compare all three models
python main.py --compare

# Tune sensitivity (fraction of expected anomalies)
python main.py --contamination 0.03

# Custom input/output paths
python main.py --data path/to/transactions.csv --output results/my_suspicious.csv
```

### 3b. Launch the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3c. Step-by-step Jupyter Notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

##  Models

| Model | Technique | Best For |
|-------|-----------|----------|
| **Isolation Forest** | Randomly partitions data; anomalies need fewer splits | High-dimensional tabular data, fast training |
| **One-Class SVM** | Learns a tight boundary around normal data | When class boundaries are well-defined |
| **Local Outlier Factor** | Compares local density to neighbours | Detecting contextual anomalies in clusters |

---

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `tx_frequency` | How many transactions a customer has overall |
| `avg_spending` | Customer's mean transaction amount |
| `spending_deviation` | Z-score vs customer's own spending history |
| `max_daily_tx` | Customer's busiest single-day transaction count |
| `amount_to_balance_ratio` | Amount ÷ balance — a high value signals over-spending |
| `time_since_last_tx` | Hours since the customer's last transaction |
| `is_high_risk_location` | 1 if transaction is flagged as "Foreign" |
| `rolling_avg_amount` | 5-transaction rolling mean per customer |
| `hour`, `day_of_week`, `is_night`, `is_weekend` | Temporal patterns |
| `*_encoded` | Label-encoded categorical variables |
| `*_normalized` | Standard-scaled numerical variables |

---

##  Dashboard Tabs

| Tab | Content |
|-----|---------|
| **Data Upload** | Upload CSV or use built-in demo data; run detection |
| **Overview** | Transaction summary, metrics, anomaly rate by type/category |
| **Suspicious Transactions** | Filterable, color-coded list with download button |
| **Charts** | Amount distribution, risk scores, hourly patterns, scatter plots |
| **Report** | Executive summary, risk breakdown pie chart, top risky customers |

---

## Outputs

After running the pipeline you will find:

- `results/suspicious_transactions.csv` — list of flagged transactions sorted by risk score
- `results/anomaly_analysis.png` — 6-panel visualisation
- `models/trained_model.pkl` — serialised detector
- `models/preprocessor.pkl` — serialised scaler + encoders

---

## Accuracy Improvement Suggestions

1. **Real labelled dataset** — replace synthetic data with a real fraud dataset (e.g., [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)) for proper supervised evaluation.
2. **Supervised second layer** — once labels exist, train XGBoost/LightGBM on the `risk_score` + features for a hybrid model.
3. **Longer rolling windows** — extend `rolling_avg_amount` to 30-day windows to capture monthly cycles.
4. **Graph-based features** — model customer–merchant as a bipartite graph; isolated edges = higher suspicion.
5. **Autoencoder** — deep learning reconstruction error is a powerful unsupervised signal.
6. **Precision-recall threshold tuning** — instead of fixing `contamination`, sweep the risk-score threshold and pick the point that maximises F2-score (recall-weighted).
7. **Concept drift detection** — monitor feature distributions monthly; retrain when drift is detected (e.g., with `alibi-detect`).
8. **Real-time scoring** — wrap the detector in a FastAPI endpoint for sub-millisecond transaction scoring.

---

## Security Notes

- Never store actual customer PII in the CSV without encryption at rest.
- Treat `models/*.pkl` files as sensitive; validate their integrity before loading in production.
- All CSV uploads in the dashboard are processed in-memory and never written to disk.

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0
plotly>=5.17.0
joblib>=1.3.0
```

---

## Deployment

### Option A — Streamlit Community Cloud (free, recommended)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Set:
   - **Repository**: `<YOUR_USERNAME>/financial-anomaly-detection`
   - **Branch**: `main`
   - **Main file path**: `dashboard/app.py`
4. Click **Deploy**. The app is live in ~2 minutes.

> The app uses built-in demo data by default, so no dataset upload is required for the cloud demo.

### Option B — Local / Docker

```bash
# Build
docker build -t anomaly-detector .

# Run
docker run -p 8501:8501 anomaly-detector
```

### Option C — pip-installable (editable)

```bash
pip install -e .
anomaly-detect            # runs the CLI
streamlit run dashboard/app.py
```

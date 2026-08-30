# Behavioral Churn Prediction Across Ecosystems

A reproducible, research-grade framework for predicting customer churn across
**heterogeneous ecosystem types** using standardized behavioral feature
engineering. Built for academic publication.

## Research Question

**"How does ecosystem structure and behavioral observability influence churn predictability?"**

Not: "Which ML model performs best."

## Supported Ecosystems

| Dataset | Type | Churn Window | Behavior |
|---------|------|-------------|----------|
| Olist | Transactional Marketplace | 180 days | Sparse repurchase, inactivity-dominated churn |
| REES46 | Transactional Marketplace | 180 days | Richer behavioral signals, marketplace structure |
| RetailRocket | Clickstream Commerce | 30 days | Browsing + purchase, strong engagement observability |
| Online Retail II | Habitual Retail | 90 days | Repeat gift purchases, seasonal patterns |
| Instacart | Habitual Retail | 60 days | High weekly/biweekly cadence, grocery habitual |
| IBM Telco | Subscription | Native label | Contractual churn, explicit tenure/Churn column |

## Architecture

```
project_root/
├── src/
│   ├── __init__.py
│   ├── datasets/                  # Dataset adapters (one per dataset)
│   │   ├── __init__.py            # Registry: get_dataset("olist"), list_datasets()
│   │   ├── base.py                # AbstractBaseDataset — contract for all adapters
│   │   ├── olist.py
│   │   ├── rees46.py
│   │   ├── retailrocket.py
│   │   ├── online_retail_ii.py
│   │   ├── instacart.py
│   │   └── telco.py
│   ├── config.py                  # Central config, paths, hyperparameters
│   ├── churn_labeling.py          # Dataset-aware temporal churn labeling
│   ├── feature_engineering.py     # Standardised-schema feature groups (8 groups)
│   ├── modeling.py                # LR, RF, XGBoost — unchanged
│   ├── evaluation.py              # Metrics, ECE, imbalance analysis
│   ├── baselines.py               # Majority-class & random baselines
│   ├── pipeline.py                # Dataset-agnostic orchestrator
│   ├── calibration.py             # Calibration curves with bootstrap CI
│   ├── explainability.py          # SHAP analysis
│   ├── visualization.py           # Publication-quality figures
│   ├── statistical_tests.py       # Mann-Whitney U, Cliff's delta, BH correction
│   ├── segmentation.py            # K-Means with PCA
│   ├── ablation.py                # Feature-group ablation study
│   ├── risk_scoring.py            # Churn probability → 0-100 risk score
│   ├── failure_analysis.py        # FP/FN error analysis
│   ├── data_quality.py            # Automated data quality reports
│   ├── exports.py                 # Model, data, metric, master results export
│   ├── experiment_tracker.py      # Metadata logging
│   └── utils.py                   # Logging, seeding, defensive helpers
├── figures/                       # All generated plots
├── results/                       # Metrics, CSVs, master_results.csv
├── models/                        # Trained model artefacts
├── processed_data/                # Train/test feature matrices
├── requirements.txt
└── README.md
```

## Usage

```bash
pip install -r requirements.txt
```

Place dataset CSV files in `data/<dataset_name>/` or Kaggle input directories.

Run the pipeline for any supported dataset:

```bash
# Single dataset
python -m src.pipeline olist

# Or from Python
python -c "from src.pipeline import run_pipeline; run_pipeline('instacart')"
```

## Standardised Feature Groups

| Group | Required Columns | Features | Availability |
|-------|-----------------|----------|-------------|
| purchase | customer_id, event_time, event_type | total_orders, total_items, repeat_ratio | All transactional datasets |
| monetary | transaction_value | total_spent, avg/max/min order value | Olist, REES46, Online Retail II |
| inactivity | event_time | days_since_last_purchase | All (universal) |
| review | review_score | avg/min/var score, low/positive ratio | Datasets with reviews |
| delivery | delivery_delay | avg/max delay, on-time ratio | Olist only |
| payment | payment_type | avg_value, preferred type dummies | Olist only |
| engagement | event_type, session_id | page_views, cart_adds, sessions | Clickstream datasets |
| cadence | event_time | avg_days_between_orders, lifetime_days | All transactional |

Missing groups are automatically detected and disabled with an explicit log entry.

## Baselines

Every model run includes:
- **Majority-class baseline** — always predict the majority class
- **Random baseline** — predict with probability = training churn rate

All metrics must be compared against these to establish lift.

## Cross-Dataset Results

Automatically builds `results/cross_dataset/master_results.csv`:

```
dataset, ecosystem_type, model, roc_auc, pr_auc, f1, precision, recall,
brier_score, calibration_error, churn_rate, imbalance_ratio,
dominant_feature_group
```

## Churn Definition by Dataset

Each dataset uses a **behaviorally justified** churn window, defined *before*
evaluation and never tuned for better metrics:

- **Olist (180d)**: 3x median inter-purchase interval (~60d)
- **REES46 (180d)**: Matches Olist for cross-dataset comparability
- **RetailRocket (30d)**: Dataset spans only ~4.5 months; clickstream users churn faster
- **Online Retail II (90d)**: 2x median inter-purchase interval (~45d), seasonal gift retail
- **Instacart (60d)**: ~4x median inter-purchase interval (~14d), weekly grocery cadence
- **Telco**: Native contractual churn label — no inactivity window applied

## Reproducibility

- All random seeds fixed (RANDOM_SEED=42)
- Package versions pinned in requirements.txt
- Experiment metadata logged with platform, config, metrics
- Temporal cutoffs are deterministic given the data
- Kaggle-compatible (auto-detects `/kaggle/input/`)

## Models

| Model | Imbalance Handling | Key Parameter |
|-------|-------------------|---------------|
| Logistic Regression | class_weight='balanced' | C=0.1, lbfgs |
| Random Forest | class_weight='balanced_subsample' | n_est=200, depth=10 |
| XGBoost | scale_pos_weight | lr=0.05, early stopping |

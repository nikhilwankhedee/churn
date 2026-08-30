"""
Central configuration for the behavioral churn prediction framework.
Environment-aware: auto-detects Kaggle vs local execution.

Dataset-specific overrides (churn windows, feature groups) are defined
in each dataset adapter under src/datasets/.
"""
import os
import sys
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────
# ENVIRONMENT DETECTION
# ──────────────────────────────────────────────
ON_KAGGLE: bool = os.path.exists('/kaggle/input')
ON_KAGGLE_WORKING: bool = os.path.exists('/kaggle/working')

# ──────────────────────────────────────────────
# BASE PATHS
# ──────────────────────────────────────────────
if ON_KAGGLE:
    PROJECT_ROOT: str = '/kaggle/working'
    DATA_DIR: str = '/kaggle/input/datasets'
else:
    PROJECT_ROOT: str = str(Path(__file__).resolve().parent.parent)
    DATA_DIR: str = os.path.join(PROJECT_ROOT, 'data')

OLIST_DIR: str = "/kaggle/input/datasets/olistbr/brazilian-ecommerce"
RETAILROCKET_EVENTS: str = "/kaggle/input/datasets/retailrocket/ecommerce-dataset/events.csv"
REES46_FILE: str = "/kaggle/input/datasets/fridrichmrtn/e-commerce-churn-dataset-rees46/rees46_customer_model.csv"
INSTACART_DIR: str = "/kaggle/input/datasets/psparks/instacart-market-basket-analysis"
TELCO_FILE: str = "/kaggle/input/datasets/blastchar/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
ONLINE_RETAIL_FILE: str = "/kaggle/input/datasets/nikhilwankhedee/online-retail-ii/online_retail_II.xlsx"
LASTFM_PARQUET: str = "/kaggle/input/datasets/nikhilwankhedee/lastfm/lastfm-dataset-1k.snappy.parquet"
LASTFM_PROFILE: str = "/kaggle/input/datasets/nikhilwankhedee/lastfm/userid-profile.tsv"
CREDIT_CARD_FILE: str = "/kaggle/input/datasets/sakshigoyal7/credit-card-customers/BankChurners.csv"

PROCESSED_DIR: str = os.path.join(PROJECT_ROOT, 'processed_data')
FIGURES_DIR: str = os.path.join(PROJECT_ROOT, 'figures')
RESULTS_DIR: str = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR: str = os.path.join(PROJECT_ROOT, 'models')

# Subdirectory structure for organised outputs
FIGURE_SUBDIRS: list = [
    'dataset_analysis', 'churn_analysis', 'correlation_analysis',
    'segmentation', 'model_evaluation', 'shap_analysis',
    'behavioral_insights', 'calibration',
]
RESULT_SUBDIRS: list = [
    'model_metrics', 'statistical_tests', 'risk_scoring',
    'data_quality', 'experiments', 'failure_analysis',
    'ablation', 'shap_values', 'cross_dataset',
]

# ──────────────────────────────────────────────
# DATASET FILENAMES (Olist — kept for backward compat)
# ──────────────────────────────────────────────
ORDERS_FILE: str = 'olist_orders_dataset.csv'
CUSTOMERS_FILE: str = 'olist_customers_dataset.csv'
REVIEWS_FILE: str = 'olist_order_reviews_dataset.csv'
PAYMENTS_FILE: str = 'olist_order_payments_dataset.csv'
ITEMS_FILE: str = 'olist_order_items_dataset.csv'
PRODUCTS_FILE: str = 'olist_products_dataset.csv'
GEOLOCATION_FILE: str = 'olist_geolocation_dataset.csv'
SELLERS_FILE: str = 'olist_sellers_dataset.csv'
CATEGORY_FILE: str = 'product_category_name_translation.csv'

# ──────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────
RANDOM_SEED: int = 42
np.random.seed(RANDOM_SEED)

# ──────────────────────────────────────────────
# CHURN DEFINITION (default — overridden per dataset)
# ──────────────────────────────────────────────
PREDICTION_WINDOW_DAYS: int = 180
TRAIN_SPLIT_QUANTILE: float = 0.7

# ──────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ──────────────────────────────────────────────
LOGISTIC_REGRESSION_PARAMS: dict = {
    'C': 0.1,
    'max_iter': 1000,
    'class_weight': 'balanced',
    'solver': 'lbfgs',
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
}

RANDOM_FOREST_PARAMS: dict = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_leaf': 20,
    'class_weight': 'balanced_subsample',
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
}

XGBOOST_PARAMS: dict = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': None,
    'random_state': RANDOM_SEED,
    'eval_metric': 'logloss',
    'n_jobs': -1,
}

LIGHTGBM_PARAMS: dict = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'is_unbalance': True,
    'random_state': RANDOM_SEED,
    'verbose': -1,
}

SVM_PARAMS: dict = {
    'class_weight': 'balanced',
    'random_state': RANDOM_SEED,
    'max_iter': 2000,
}

# ──────────────────────────────────────────────
# SEGMENTATION
# ──────────────────────────────────────────────
N_CLUSTERS: int = 5

# ──────────────────────────────────────────────
# STANDARDISED FEATURE GROUP NAMES
# These are the canonical feature groups used across ALL datasets.
# Individual datasets declare which groups they support.
# ──────────────────────────────────────────────
STANDARD_FEATURE_GROUPS: list = [
    'purchase', 'monetary', 'inactivity', 'review',
    'delivery', 'payment', 'engagement', 'cadence',
]

# ──────────────────────────────────────────────
# FEATURE GROUP DEFINITIONS (used for ablation & reporting)
# These map logical group → standardized feature column names.
# ──────────────────────────────────────────────
FEATURE_GROUPS: dict = {
    'purchase': [
        'total_orders', 'total_items_purchased', 'repeat_purchase_ratio',
    ],
    'monetary': [
        'total_spent', 'avg_order_value', 'max_order_value',
        'min_order_value',
    ],
    'inactivity': [
        'days_since_last_purchase',
    ],
    'review': [
        'avg_review_score', 'min_review_score', 'low_review_ratio',
        'positive_review_ratio', 'review_variance',
    ],
    'delivery': [
        'avg_delivery_delay_days', 'delayed_order_ratio',
        'max_delivery_delay', 'on_time_delivery_ratio',
    ],
    'payment': [
        'avg_payment_installments', 'avg_payment_value',
    ],
    'engagement': [
        'total_page_views', 'total_cart_adds', 'total_sessions',
        'avg_actions_per_session', 'total_wishlist_adds',
    ],
    'cadence': [
        'avg_days_between_orders', 'customer_lifetime_days',
        'avg_orders_per_month',
    ],
}

PAYMENT_DUMMY_PREFIX: str = 'pay_type_'

# ──────────────────────────────────────────────
# ABLATION / CV
# ──────────────────────────────────────────────
ABLATION_N_SPLITS: int = 5
ABLATION_RANDOM_STATE: int = RANDOM_SEED

# ──────────────────────────────────────────────
# SHAP
# ──────────────────────────────────────────────
SHAP_SAMPLE_SIZE: int = 200
SHAP_MAX_DISPLAY: int = 20

# ──────────────────────────────────────────────
# CALIBRATION
# ──────────────────────────────────────────────
CALIBRATION_N_BINS: int = 10
CALIBRATION_N_BOOTSTRAP: int = 200

# ──────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────
FIGURE_DPI: int = 150
SAVEFIG_DPI: int = 300
FONT_SIZE: int = 12

# ──────────────────────────────────────────────
# SENSITIVITY ANALYSIS
# ──────────────────────────────────────────────
SENSITIVITY_ENABLED: bool = False
SENSITIVITY_WINDOWS: dict = {
    'olist':         [90, 180, 270],
    'rees46':        [90, 180, 270],
    'retailrocket':  [15, 30, 60],
    'online_retail_ii': [45, 90, 180],
    'instacart':     [30, 60, 90],
    'telco':         [],  # native label — no sensitivity
}
SENSITIVITY_RESULTS_DIR: str = 'sensitivity_analysis'

# ──────────────────────────────────────────────
# VALIDATOR THRESHOLDS
# ──────────────────────────────────────────────
VALIDATOR_THRESHOLDS: dict = {
    'max_churn_rate': 0.99,
    'min_orders_per_customer': 1.05,
    'max_imbalance_ratio': 100,
    'min_repeat_purchase_ratio': 0.01,
    'min_customers': 100,
    'max_feature_sparsity': 0.95,
    'auc_variance_warn': 0.15,
    'min_expected_auc': 0.45,
    'max_calibration_error': 0.25,
}

# ──────────────────────────────────────────────
# VALIDATION / FILTERING
# ──────────────────────────────────────────────
TIMESTAMP_MIN: str = '2016-01-01'
TIMESTAMP_MAX: str = '2019-01-01'
OUTLIER_PRICING_PERCENTILE: float = 0.999
MIN_SAMPLES_PER_GROUP: int = 5

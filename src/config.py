"""
Central configuration for the behavioral churn prediction framework.
Environment-aware: auto-detects Kaggle vs local execution.

Dataset-specific overrides (churn windows, feature groups) are defined
in each dataset adapter under src/datasets/.

YAML configuration support:
    Call load_config(yaml_path) to overlay YAML values onto these defaults.
    When no YAML is loaded, all existing constants are used as-is,
    guaranteeing identical behavior to the baseline implementation.
"""
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ──────────────────────────────────────────────
# FRAMEWORK VERSION
# ──────────────────────────────────────────────
FRAMEWORK_VERSION: str = "2.0.0"


# ──────────────────────────────────────────────
# ENVIRONMENT DETECTION
# ──────────────────────────────────────────────
ON_KAGGLE: bool = os.path.exists('/kaggle/input')
ON_KAGGLE_WORKING: bool = os.path.exists('/kaggle/working')

# ──────────────────────────────────────────────
# BASE PATHS
# ──────────────────────────────────────────────
if ON_KAGGLE:
    _candidate = '/kaggle/working/project_root'
    if os.path.isdir(_candidate):
        PROJECT_ROOT: str = _candidate
    else:
        PROJECT_ROOT: str = '/kaggle/working'
else:
    PROJECT_ROOT: str = str(Path(__file__).resolve().parent.parent)

# DATA_DIR is a fallback default — the centralized resolver (dataset_resolver.py)
# handles path resolution. Do NOT hardcode dataset-specific paths here.
DATA_DIR: str = os.environ.get(
    "CHURN_DATA_DIR",
    os.path.join(PROJECT_ROOT, 'data'),
)

# Builtin demo/toy data shipped with the framework — used by the resolver as a
# final fallback so smoke tests and quick runs work without external downloads.
BUILTIN_DATA_DIR: str = os.environ.get(
    "CHURN_BUILTIN_DATA_DIR",
    os.path.join(DATA_DIR, 'builtin'),
)

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
    'max_depth': 10,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'class_weight': 'balanced',
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'verbose': -1,
}

SVM_PARAMS: dict = {
    'kernel': 'rbf',
    'probability': True,
    'class_weight': 'balanced',
    'random_state': RANDOM_SEED,
    'cache_size': 2000,
}

SVM_SUBSET_SIZE: int = 10000

# ──────────────────────────────────────────────
# SMOTE
# ──────────────────────────────────────────────
SMOTE_K_NEIGHBORS: int = 5
SMOTE_RANDOM_STATE: int = RANDOM_SEED

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
    'static', 'listening', 'kkbox',
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
    # Static per-customer attributes (e.g. Telco tenure/charges/demographics).
    # Adapters emit these with a 'static_' prefix in the standardised schema.
    'static': [
        'static_tenure', 'static_monthly_charges', 'static_total_charges',
        'static_senior_citizen',
    ],
    # Listening behaviour (e.g. Last.fm).
    'listening': [
        'total_listens', 'unique_artists', 'unique_tracks', 'active_days',
        'avg_listens_per_day', 'days_since_last_listen',
        'max_gap_between_sessions', 'listening_frequency',
        'artist_diversity_ratio',
    ],
    # KKBox subscription + listening activity.
    'kkbox': [
        'total_transactions', 'total_msno_paid', 'is_auto_renew',
        'avg_plan_list_price', 'n_unique_plans', 'n_unique_payment_methods',
        'n_unique_payment_plan_days', 'avg_payment_plan_days',
        'total_log_days', 'total_num_25', 'total_num_50', 'total_num_75',
        'total_num_985', 'total_num_100', 'total_num_unq', 'total_seconds',
        'active_log_days', 'avg_num_per_day', 'avg_seconds_per_day',
        'days_since_last_listen',
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
# FINAL EXPERIMENT DATASET MATRIX (Section 1)
# Exactly these 8 datasets form the research matrix.
# kkbox remains registered (adapter + WSDM labeler + validation harness)
# but is excluded from the final matrix unless explicitly enabled.
# ──────────────────────────────────────────────
FINAL_EXPERIMENT_DATASETS: list = [
    'olist', 'retailrocket', 'rees46', 'instacart',
    'telco', 'online_retail_ii', 'lastfm', 'credit_card',
]

# ──────────────────────────────────────────────
# FINAL EXPERIMENT MODEL MATRIX (Section 3)
# All five models are mandatory in every run.
# ──────────────────────────────────────────────
FINAL_EXPERIMENT_MODELS: list = [
    'logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'svm',
]

# ──────────────────────────────────────────────
# RESULT LAYOUT (Section 30)
# ──────────────────────────────────────────────
SMOTE_CONDITIONS: list = ['without_smote', 'with_smote']

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


# ──────────────────────────────────────────────
# KKBox (WSDM Cup 2018) constants
# ──────────────────────────────────────────────
KKBOX_CHURN_WINDOW_DAYS: int = 30
KKBOX_TRAIN_FILE: str = 'train.csv'
KKBOX_TRAIN_V2_FILE: str = 'train_v2.csv'
KKBOX_TRANSACTIONS_FILE: str = 'transactions.csv'
KKBOX_TRANSACTIONS_V2_FILE: str = 'transactions_v2.csv'
KKBOX_LOGS_FILE: str = 'user_logs.csv'
KKBOX_LOGS_V2_FILE: str = 'user_logs_v2.csv'
KKBOX_MEMBERS_FILE: str = 'members.csv'
KKBOX_MEMBERS_V3_FILE: str = 'members_v3.csv'
KKBOX_SAMPLE_SUBMISSION: str = 'sample_submission_zero.csv'
KKBOX_LABELLER_SCRIPT: str = 'WSDMChurnLabeller.scala'
# Members with bd outside [0, 99] or invalid are treated as missing (WSDM spec)
KKBOX_VALID_BD_RANGE: tuple = (0, 99)
# Member static columns produced by the KKBox adapter (prefix 'static_')
KKBOX_STATIC_MEMBER_COLUMNS: list = [
    'static_city', 'static_bd', 'static_gender', 'static_registered_via',
    'static_registration_init_time',
]

# ── WSDM label windows (official competition dates) ──────────────────
KKBOX_HISTORY_START: str = '20170101'
KKBOX_HISTORY_CUTOFF: str = '20170131'
KKBOX_PREDICTION_START: str = '20170201'
KKBOX_PREDICTION_END: str = '20170228'
KKBOX_RENEWAL_WINDOW_DAYS: int = 30
# Raw transaction columns that must be read as strings so the plan
# signature (plan_list_price + payment_plan_days + payment_method_id)
# reproduces the exact string concatenation the Scala reference performs.
KKBOX_RAW_STRING_COLUMNS: list = [
    'payment_method_id', 'payment_plan_days', 'plan_list_price',
    'transaction_date', 'membership_expire_date', 'is_cancel',
]
# Validation thresholds: agreement below this (or coverage below this)
# marks the derived labels as VALIDATED_MISMATCH.
KKBOX_AGREEMENT_THRESHOLD: float = 0.999
KKBOX_MIN_OFFICIAL_COVERAGE: float = 0.95



# ──────────────────────────────────────────────
# YAML CONFIGURATION SUPPORT
# ──────────────────────────────────────────────
# When a YAML config is loaded, it is stored here.
# New framework code should use get_config() to access values.
# Existing code continues to use the module-level constants above.
# ──────────────────────────────────────────────
_YAML_CONFIG: Optional[dict] = None


def load_config(yaml_path: str) -> dict:
    """Load a YAML configuration file and store it as the active config.

    This does NOT modify any module-level constants. It stores the YAML
    content for access via get_config() and get_config_value(). Existing
    code that imports constants directly is completely unaffected.

    Parameters
    ----------
    yaml_path : str
        Path to a YAML configuration file.

    Returns
    -------
    dict of the loaded configuration.

    Raises
    ------
    FileNotFoundError if the file does not exist.
    ImportError if PyYAML is not installed.
    """
    global _YAML_CONFIG
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML configuration. "
            "Install with: pip install PyYAML"
        )
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    with open(path, 'r') as f:
        _YAML_CONFIG = yaml.safe_load(f) or {}
    return _YAML_CONFIG


def get_config() -> Optional[dict]:
    """Return the currently loaded YAML configuration, or None."""
    return _YAML_CONFIG


def get_config_value(dotpath: str, default: Any = None) -> Any:
    """Access a nested YAML config value using dot notation.

    Example:
        get_config_value("churn.default_window_days", 180)
        get_config_value("models.xgboost.learning_rate")

    Parameters
    ----------
    dotpath : str
        Dot-separated path into the YAML config.
    default : Any
        Value to return if the path is not found.

    Returns
    -------
    The value at the path, or default if not found.
    """
    if _YAML_CONFIG is None:
        return default
    keys = dotpath.split(".")
    obj = _YAML_CONFIG
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
        else:
            return default
    return obj


def get_configs_dir() -> Path:
    """Return the configs directory, handling both development and installed modes.

    In development, configs live at project_root/configs/.
    When installed as a package, configs live at src/configs/ inside site-packages.
    """
    # Installed package: src/configs/ relative to this file
    pkg_path = Path(__file__).resolve().parent / "configs"
    if pkg_path.is_dir() and (pkg_path / "default.yaml").exists():
        return pkg_path
    # Development: project_root/configs/
    dev_path = Path(PROJECT_ROOT) / "configs"
    if dev_path.is_dir():
        return dev_path
    # Fallback
    return pkg_path


def load_default_config() -> Optional[dict]:
    """Attempt to load the default configs/default.yaml.

    Returns the loaded config dict, or None if the file is not found.
    Does not raise errors — designed for graceful fallback.
    """
    default_path = get_configs_dir() / "default.yaml"
    if default_path.exists():
        try:
            return load_config(str(default_path))
        except Exception:
            return None
    return None

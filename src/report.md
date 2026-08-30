# 1. Project Overview

## Problem Statement

The project implements a behavioral churn prediction framework for heterogeneous customer datasets. It converts raw transactional, clickstream, habitual retail, and subscription records into a standardized customer-level feature matrix, generates churn labels, trains classifiers, evaluates predictive performance, and exports research artifacts.

The implementation is centered on `pipeline.py`, which orchestrates dataset loading, preprocessing, schema normalization, temporal labeling, feature engineering, model training, evaluation, visualization, validation, explainability, and experiment logging.

## Research Objective

The implemented objective is to compare behavioral churn prediction across multiple dataset ecosystems using a shared pipeline and standardized feature groups. The code supports cross-dataset result aggregation through `append_to_master_results()` in `exports.py`.

## Overall Architecture

The architecture is modular:

- Dataset-specific adapters implement loading, cleaning, schema mapping, metadata, churn-window configuration, and feature-group availability.
- Shared pipeline modules perform churn labeling, feature engineering, modeling, evaluation, validation, visualization, explainability, and exports.
- Configuration is centralized in `config.py`.
- Outputs are written to configured `processed_data`, `figures`, `results`, and `models` directories.

## High-Level Workflow

`run_pipeline()` performs:

1. Resolve dataset adapter from registry.
2. Load raw data.
3. Generate raw data quality report.
4. Preprocess dataset.
5. Standardize schema.
6. Validate schema.
7. Compute temporal train/test cutoffs.
8. Generate train/test churn labels.
9. Engineer train/test customer features.
10. Train baseline and ML models.
11. Evaluate metrics.
12. Generate plots, calibration curves, SHAP outputs, risk scores, failure analysis, ablation results, and experiment logs.

# 2. System Architecture

## Dataset Registry

`datasets/__init__.py` defines `_REGISTRY`, mapping dataset names to adapter classes:

- `olist`
- `rees46`
- `retailrocket`
- `online_retail_ii`
- `instacart`
- `telco`

`get_dataset()` lazy-loads and caches adapter instances. `get_ecosystem_type()` maps datasets to ecosystem categories.

## Dataset Adapters

All adapters inherit from `BaseDatasetAdapter` in `datasets/base.py`. The adapter contract requires:

- `dataset_name`
- `ecosystem_type`
- `load_raw_data()`
- `preprocess()`
- `standardize_schema()`
- `available_feature_groups`
- `metadata`

Adapters may override:

- `churn_window_days`
- `uses_native_churn_label`
- `get_native_churn_labels()`

Implemented adapters:

- `OlistAdapter`
- `REES46Adapter`
- `RetailRocketAdapter`
- `OnlineRetailIIAdapter`
- `InstacartAdapter`
- `TelcoAdapter`

## Preprocessing

Preprocessing is split between dataset adapters and the generic `preprocessing.py`. The active pipeline calls adapter-level `preprocess()`, not the generic preprocessing functions directly.

Adapter preprocessing includes timestamp parsing, invalid-row removal, numeric coercion, outlier clipping, missing-value imputation, cancellation removal, and synthetic timestamp construction where needed.

## Schema Standardization

Each adapter maps native columns to the shared schema:

- `customer_id`
- `event_time`
- `transaction_value`
- `event_type`
- `product_id`
- `review_score`
- `payment_type`
- `delivery_delay`
- `engagement_signal`
- `session_id`

Schema validation is implemented in `validate_schema()` in `validators.py`.

## Feature Engineering

`feature_engineering.py` computes customer-level features from standardized events using `engineer_features()`. Feature groups are modular:

- purchase
- monetary
- inactivity
- review
- delivery
- payment
- engagement
- cadence

Each group checks required columns and returns no features if unavailable.

## Churn Labeling

`churn_labeling.py` implements inactivity-based labels. Customers active before a cutoff are labeled churned if they have no events in the following prediction window.

Telco bypasses this and uses native `Churn` labels through `TelcoAdapter.get_native_churn_labels()`.

## Train/Test Split

The project uses temporal snapshot cutoffs from `get_train_test_cutoffs()` in `churn_labeling.py`. For inactivity-based datasets:

- `test_cutoff = max(event_time) - prediction_window_days`
- `train_cutoff = event_time.quantile(TRAIN_SPLIT_QUANTILE)`
- if `train_cutoff >= test_cutoff`, train cutoff is adjusted earlier.

The pipeline then creates train and test feature snapshots at these separate cutoffs.

## Validation Framework

`validators.py` implements four validation layers:

- schema validation
- behavioral sanity checks
- output validation
- cross-dataset validation

## Statistical Testing

`statistical_tests.py` compares numeric feature distributions between retained and churned customers using Mann-Whitney U tests, Benjamini-Hochberg correction, and Cliff's delta.

## Model Training

`modeling.py` trains:

- Logistic Regression
- Random Forest
- XGBoost

LightGBM and SVM are not implemented.

## Explainability

`explainability.py` integrates SHAP. It uses `TreeExplainer` for tree-like models and `LinearExplainer` for models with `coef_`.

## Visualization

`visualization.py` generates model, churn, feature, segmentation, ablation, and behavioral plots using Matplotlib and Seaborn.

## Exports

`exports.py` writes models, processed data, metrics, SHAP values, risk scores, data quality reports, experiment metadata, and cross-dataset master results.

## Experiment Runner

The implemented runner is `run_pipeline()` in `pipeline.py`. The module also supports command-line execution via `if __name__ == '__main__'` with dataset name and `--sensitivity`.

There is no separate dedicated experiment-runner module beyond `pipeline.py` and `sensitivity.py`.

## Configuration

`config.py` defines:

- environment detection
- paths
- output subdirectories
- dataset filenames
- random seed
- churn defaults
- model hyperparameters
- feature groups
- SHAP, calibration, plotting, sensitivity, and validation settings

## Utilities

`utils.py` provides logging, custom `VALIDATION` log level, seeding, directory creation, timing, NaN assertions, and shape validation.

# 3. Dataset Support

## Olist

Source: Brazilian E-Commerce Public Dataset by Olist, configured in `OlistAdapter.metadata`.

Implementation: `datasets/olist.py`

Preprocessing strategy:

- loads orders as required table
- optionally merges customers, reviews, payments, items, products, sellers
- aggregates one-to-many payment and item tables before merging
- parses order timestamps
- filters timestamps to configured `[TIMESTAMP_MIN, TIMESTAMP_MAX]`
- removes negative `price` and `freight_value`
- clips `price` and `freight_value` at the configured 0.999 percentile
- fills missing review score with median
- fills payment and price-related missing values with zero

Schema normalization:

- `customer_unique_id -> customer_id`
- `order_purchase_timestamp -> event_time`
- `payment_value -> transaction_value`
- `event_type = purchase`
- derives `delivery_delay`

Target generation:

- inactivity-based churn
- 180-day churn window

Dataset-specific challenges:

- one-to-many joins across payments/items are controlled by aggregation
- missing optional files are skipped
- missing customers fall back to `customer_id`

Assumptions:

- all rows represent purchase events after standardization
- missing delivery delay is treated as `0.0`
- missing payment type becomes `unknown`

Limitations:

- geolocation/category translation files are configured but not used by the adapter
- Olist-specific standalone `data_loader.py` duplicates some adapter loading logic but is not used by `run_pipeline()`

## REES46

Source: REES46 Marketplace Dataset, configured in `datasets/rees46.py`.

Preprocessing strategy:

- loads required events file
- optionally merges users and items
- parses Unix-second timestamps
- coerces price to numeric
- removes negative prices
- clips price at 0.999 quantile
- drops missing user IDs

Schema normalization:

- `user_id -> customer_id`
- `timestamp -> event_time`
- `price -> transaction_value`
- `item_id -> product_id`
- default `event_type = purchase` if absent
- default `review_score = 0.0`
- default `payment_type = unknown`
- default `delivery_delay = 0.0`
- default `session_id = unknown`

Target generation:

- inactivity-based churn
- 180-day churn window

Dataset-specific challenges:

- optional users/items enrichment
- session IDs may not exist and are synthesized as constant `unknown`

Assumptions:

- event types are already usable if present
- absent event types imply purchases

Limitations:

- review, delivery, and payment feature groups are not declared available
- constant `session_id = unknown` limits meaningful session-based engagement features

## RetailRocket

Source: RetailRocket E-commerce Dataset, configured in `datasets/retailrocket.py`.

Preprocessing strategy:

- loads required events file
- optionally merges item data
- parses millisecond timestamps
- drops missing timestamps, visitor IDs, and event values

Schema normalization:

- `visitorid -> customer_id`
- `timestamp -> event_time`
- `itemid -> product_id`
- `transactionid -> session_id`
- maps `view -> view`, `addtocart -> cart_add`, `transaction -> purchase`
- sets missing monetary value to `0.0`
- defaults review/payment/delivery fields

Target generation:

- inactivity-based churn
- 30-day churn window

Dataset-specific challenges:

- monetary values are not available in the events table
- `transactionid` is used as `session_id`, which is not equivalent to browsing session identity

Assumptions:

- event names map cleanly through the adapter's `event_type_map`
- unknown event values become `other`

Limitations:

- `VISITS_FILE` and category file constants exist but are not loaded
- monetary features are structurally present but based on zero transaction values unless enriched data provides `transaction_value`

## Online Retail II

Source: Online Retail II UCI/Kaggle dataset, configured in `datasets/online_retail_ii.py`.

Preprocessing strategy:

- loads and concatenates 2009-2010 and 2010-2011 CSVs if present
- parses `InvoiceDate`
- removes cancellation invoices starting with `C`
- keeps positive quantities
- keeps nonnegative prices
- clips price at 0.999 quantile
- drops invalid customer IDs

Schema normalization:

- `Customer ID -> customer_id`
- `InvoiceDate -> event_time`
- `StockCode -> product_id`
- `event_type = purchase`
- `transaction_value = Quantity * Price`
- defaults review/payment/delivery fields

Target generation:

- inactivity-based churn
- 90-day churn window

Dataset-specific challenges:

- cancellation invoices require removal
- customer IDs may be missing or blank

Assumptions:

- remaining non-cancelled rows are purchases
- transaction value is line-item quantity multiplied by price

Limitations:

- review, delivery, payment, and engagement feature groups are not declared available

## Instacart

Source: Instacart Market Basket Analysis dataset, configured in `datasets/instacart.py`.

Preprocessing strategy:

- loads required orders file
- optionally merges order-products, products, aisles, departments
- samples large tables for memory safety
- constructs synthetic `event_time` from `days_since_prior_order`
- defaults purchase value to `0.0`

Schema normalization:

- `user_id -> customer_id`
- `purchase_value -> transaction_value`
- `order_id -> session_id`
- `reordered -> engagement_signal`
- `event_type = purchase`
- defaults review/payment/delivery fields

Target generation:

- inactivity-based churn
- 60-day churn window

Dataset-specific challenges:

- no absolute timestamps are available in raw Instacart order sequence fields
- the adapter creates an approximate timeline anchored at `2017-03-21`
- large tables are sampled deterministically

Assumptions:

- synthetic event dates are acceptable for temporal ordering within users
- purchase values are unavailable and default to zero

Limitations:

- monetary features are generated from zero-valued `purchase_value`
- sampling can reduce full-dataset coverage
- engagement group is not declared available even though `engagement_signal` is standardized

## Telco

Source: IBM Telco Customer Churn dataset, configured in `datasets/telco.py`.

Preprocessing strategy:

- loads required Telco CSV
- cleans `customerID`
- converts `TotalCharges` to numeric
- maps `Churn` Yes/No to 1/0
- coerces `SeniorCitizen`

Schema normalization:

- `customerID -> customer_id`
- `MonthlyCharges -> transaction_value`
- `tenure -> engagement_signal`
- `TotalCharges -> total_charges`
- creates synthetic `event_time` from tenure relative to `2019-01-31`
- `event_type = subscription_event`
- defaults review/payment/delivery fields

Target generation:

- native `Churn` label via `get_native_churn_labels()`
- no inactivity window

Dataset-specific challenges:

- contractual churn differs from behavioral inactivity churn
- synthetic event time is derived from tenure, not observed event history

Assumptions:

- native churn label is authoritative
- tenure-derived timestamp is sufficient for compatibility with the shared pipeline

Limitations:

- many original Telco categorical fields are retained in the DataFrame but are not encoded by the implemented feature engineering pipeline
- native labels are static and are not generated from a temporal future window

# 4. Data Pipeline

Complete data flow in `pipeline.py`:

1. Adapter lookup through `get_dataset()`.
2. Raw data loading through adapter `load_raw_data()`.
3. Raw data quality report via `generate_data_quality_report()`.
4. Dataset-specific preprocessing.
5. Schema standardization.
6. Schema validation.
7. Train/test cutoff calculation.
8. Train label creation.
9. Train feature snapshot creation.
10. Test label creation.
11. Test feature snapshot creation.
12. Train/test feature column alignment.
13. Behavioral validation.
14. Processed data export.
15. Train/validation split inside training snapshot.
16. Model training.
17. Baseline evaluation.
18. Model evaluation.
19. Visualization, calibration, SHAP, segmentation, statistical testing, ablation, risk scoring, failure analysis.
20. Output validation.
21. Cross-dataset result update.
22. Experiment metadata export.
23. Optional sensitivity analysis.

# 5. Feature Engineering

Implemented in `feature_engineering.py`.

## Behavioral Features

Engagement features include:

- `total_page_views`
- `total_cart_adds`
- `total_purchases`
- `total_wishlist_adds`
- `total_events`
- `total_sessions`
- `avg_actions_per_session`
- `avg_engagement_signal`

## Temporal Features

Temporal/cadence features include:

- `days_since_last_purchase`
- `customer_lifetime_days`
- `avg_days_between_orders`
- `avg_orders_per_month`

## Transactional Features

Purchase and monetary features include:

- `total_orders`
- `total_items_purchased`
- `repeat_purchase_ratio`
- `total_spent`
- `avg_order_value`
- `max_order_value`
- `min_order_value`

## Aggregation Strategy

Features are aggregated to one row per customer. The pipeline filters events to `event_time < snapshot_date` before feature creation. Each feature group performs independent `groupby(customer_id)` aggregations, then all groups are outer-joined on customer ID.

## Categorical Encoding

The only implemented categorical encoding is payment type encoding in `_engineer_payment()`: preferred payment type is computed by mode, then one-hot encoded with prefix `pay_type_`.

Other dataset categoricals are not encoded.

## Missing Values

Feature matrices are filled with `0.0` after feature-group joins. `assert_no_nans()` verifies no remaining NaNs.

Dataset adapters also perform earlier missing-value handling, such as review-score median fill and numeric zero fill.

## Feature Selection

No model-facing feature selection step is implemented. Ablation removes predefined feature groups for analysis, but does not select a final feature subset.

# 6. Churn Labeling

Implemented in `churn_labeling.py`.

## Inactivity-Window Approach

For non-native datasets, a customer is labeled churned if:

- the customer has at least one event before the cutoff, and
- the customer has no event after the cutoff and within `prediction_window_days`.

Label value:

- `0`: retained, observed in the future window
- `1`: churned, not observed in the future window

## Snapshot Methodology

Features use history before the snapshot cutoff. Labels use events after the same cutoff. This creates a prediction task at a defined temporal snapshot.

## Temporal Cutoffs

`get_train_test_cutoffs()` computes:

- test cutoff as max event time minus churn window
- train cutoff as event-time quantile
- adjusted train cutoff if it would overlap or exceed test cutoff

## Train/Test Labeling

The pipeline separately generates labels and features for train and test cutoffs. The implementation does not enforce customer-disjoint train/test sets; the same customer can appear in both snapshots if active before both cutoffs.

## Rationale

The code rationale is explicit in comments/docstrings: labels are based on future inactivity and features are based on pre-cutoff behavior.

## Implementation Details

The future label window uses:

```python
(df[event_time_col] > cutoff_date) & (df[event_time_col] <= window_end)
```

Features use:

```python
df[df[STD_EVENT_TIME] < snapshot_date]
```

# 7. Temporal Validation

## Feature Snapshots

Implemented feature snapshots filter to events strictly before the snapshot date in `engineer_features()`.

## Timestamp Filtering

Timestamp parsing and filtering are adapter-specific. Olist additionally filters to configured timestamp bounds.

## Temporal Train/Test Split

The primary train/test split is temporal by snapshot cutoff. A secondary random stratified split is used inside the training snapshot to create validation data for model training.

## Overlap Handling

There is no implemented removal of overlapping customers between train and test snapshots. Train and test examples are separated by snapshot time, not by customer identity.

## Label Generation

For inactivity datasets, labels are generated from post-cutoff windows. For Telco, native static labels are used.

## Preprocessing Order

Preprocessing is performed before temporal split and before feature snapshots.

This matters for leakage assessment: some preprocessing operations learn global statistics from the full dataset, such as Olist review-score median imputation and price clipping in several adapters. These statistics are computed before train/test separation.

## SMOTE Application

SMOTE is not implemented. No `SMOTE` usage exists in the current codebase.

## Leakage Assessment

For behavioral event features, the implementation explicitly prevents direct future-event leakage by filtering features to `event_time < snapshot_date`.

However, the project cannot be described as fully leakage-free because adapter preprocessing is run on the full dataset before temporal splitting and can compute global medians or quantile caps using rows after the training cutoff. This is preprocessing leakage in the strict temporal-validation sense.

Additional caveat: Telco uses static native churn labels and synthetic tenure-derived event times rather than future-window temporal labels, so the same temporal leakage guarantees do not apply to Telco.

# 8. Machine Learning Pipeline

Implemented in `modeling.py`.

## Baseline Models

`baselines.py` implements:

- majority-class baseline
- random baseline using training churn rate

## Logistic Regression

Uses `sklearn.linear_model.LogisticRegression` with parameters from `config.py`:

- `C=0.1`
- `max_iter=1000`
- `class_weight='balanced'`
- `solver='lbfgs'`
- `random_state=42`
- `n_jobs=-1`

## Random Forest

Uses `RandomForestClassifier` with:

- `n_estimators=200`
- `max_depth=10`
- `min_samples_leaf=20`
- `class_weight='balanced_subsample'`
- `random_state=42`
- `n_jobs=-1`

## XGBoost

Uses `XGBClassifier` with:

- `n_estimators=200`
- `max_depth=5`
- `learning_rate=0.05`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `random_state=42`
- `eval_metric='logloss'`
- `n_jobs=-1`

`scale_pos_weight` is computed from the training labels. If validation data is provided, XGBoost uses `early_stopping_rounds=10`.

## LightGBM

Not implemented.

## SVM

Not implemented.

## Training Procedure

The pipeline splits the training snapshot into `X_tr`, `X_val`, `y_tr`, `y_val` using stratified random split with `test_size=0.1`.

## Imbalance Handling

Implemented imbalance handling:

- class weights for Logistic Regression
- balanced subsampling for Random Forest
- `scale_pos_weight` for XGBoost
- imbalance metrics are logged

SMOTE or other resampling is not implemented.

## Serialization

`save_models()` serializes trained models as `.joblib` files in `MODELS_DIR`.

# 9. Evaluation Framework

Implemented in `evaluation.py`.

## Implemented Metrics

Accuracy: fraction of correct predictions.

Precision: fraction of predicted churners that are actually churned. Useful when false-positive interventions are costly.

Recall: fraction of actual churners detected. Useful when missing churners is costly.

F1: harmonic mean of precision and recall. Useful under imbalance when a single thresholded metric is needed.

ROC-AUC: threshold-independent ranking metric over false-positive and true-positive rates. Implemented when probabilities are available.

PR-AUC: implemented as `avg_precision`. Useful for imbalanced churn problems because it emphasizes positive-class retrieval.

Calibration: implemented as expected calibration error through `_expected_calibration_error()`.

Brier score: mean squared error of predicted probabilities. Used for probability quality.

Confusion matrix: exported as counts `tn`, `fp`, `fn`, `tp` and plotted per model.

## Requested But Not Implemented

MCC is not implemented.

Balanced accuracy is not implemented.

# 10. Validation Framework

Implemented in `validators.py`.

## Schema Validation

`validate_schema()` checks:

- required columns `customer_id` and `event_time`
- optional standardized columns
- null percentages
- timestamp validity and time span
- customer ID nulls and unique count
- duplicate rows
- enabled and disabled feature groups

## Behavioral Validation

`validate_behavioral_statistics()` checks:

- customer count
- event count
- churn rate
- imbalance ratio
- orders per customer
- repeat-purchase ratio
- interpurchase intervals
- time span

## Statistical Validation

Statistical validation is implemented separately in `statistical_tests.py`, not as a validator class. It compares feature distributions across churn classes.

## Dataset Validation

Dataset validation is performed through adapter schema validation and metadata/feature-group declarations. The registry validates dataset names by rejecting unknown keys.

## Runtime Validation

Runtime validation is implemented through explicit exceptions and warnings:

- schema errors halt the pipeline
- empty feature matrices raise `RuntimeError`
- optional downstream failures are caught and logged
- output validation checks expected artifacts, metrics, and probability arrays

# 11. Explainability

SHAP integration is in `explainability.py`.

## Supported Models

Implemented support paths:

- tree-based models via `TreeExplainer`
- linear models via `LinearExplainer`

This covers the implemented Random Forest, XGBoost, and Logistic Regression paths.

## Unsupported Models

Models without tree attributes or `coef_` fall through to `TreeExplainer`, which may fail. There is no explicit unsupported-model registry.

## Failure Handling

If SHAP is not installed, explainability is disabled. SHAP failures are caught and logged without stopping the pipeline.

## Exported Artifacts

For each model where SHAP succeeds:

- `{model}_shap_bar.png`
- `{model}_shap_summary.png`
- dependence plots for top 3 SHAP features
- `{model}_shap_values.csv`

# 12. Experiment Management

Implemented in `experiment_tracker.py` and `exports.py`.

## Experiment Directory Structure

Configured directories:

- `processed_data`
- `figures`
- `results`
- `models`

Configured figure subdirectories include:

- `dataset_analysis`
- `churn_analysis`
- `correlation_analysis`
- `segmentation`
- `model_evaluation`
- `shap_analysis`
- `behavioral_insights`
- `calibration`

Configured result subdirectories include:

- `model_metrics`
- `statistical_tests`
- `risk_scoring`
- `data_quality`
- `experiments`
- `failure_analysis`
- `ablation`
- `shap_values`
- `cross_dataset`

## Logging

Logging uses a shared formatter and custom `VALIDATION` level from `utils.py`.

## Outputs

Experiment metadata is appended to:

- `results/experiments/experiment_log.csv`

## Metrics

Per-model metrics are included in both `model_metrics.csv` and experiment metadata.

## Model Persistence

Models are saved as joblib files:

- `models/logistic_regression.joblib`
- `models/random_forest.joblib`
- `models/xgboost.joblib`

## Reproducibility

Implemented reproducibility controls:

- global `RANDOM_SEED = 42`
- NumPy seed
- Python `random` seed
- optional TensorFlow/PyTorch seed if installed
- deterministic train/validation split
- deterministic Instacart sampling
- deterministic KMeans seed

Full bit-for-bit determinism is not guaranteed by the code, especially for external libraries and parallelized estimators.

# 13. Statistical Analysis

Implemented in `statistical_tests.py`.

## Hypothesis Testing

Each numeric feature is compared between retained and churned customers using a two-sided Mann-Whitney U test.

## Effect Sizes

Cliff's delta is computed either through `pingouin.compute_effsize()` if available or a NumPy fallback.

## Significance Testing

The module records:

- raw p-value
- `significant_uncorrected`
- `significant_bh`

Benjamini-Hochberg correction is attempted through SciPy `false_discovery_control`, with a fallback implementation for older SciPy versions.

## Implementation Details

Features with fewer than 5 samples in either churn group are skipped.

Output is written by the pipeline to:

- `results/statistical_tests/feature_tests.csv`

# 14. Visualizations

Generated visualizations in `visualization.py` and `calibration.py`:

- `roc_curves.png`: compares probability ranking across models.
- `pr_curves.png`: compares precision-recall behavior under imbalance.
- `confusion_{model}.png`: shows thresholded classification counts.
- `{model}_importance.png`: tree-model feature importance.
- `correlation_heatmap.png`: feature correlation matrix.
- `segments_scatter.png`: PCA projection of KMeans customer segments.
- `churn_distribution.png`: retained/churned class distribution.
- `{model}_threshold.png`: precision, recall, and F1 across thresholds.
- `ablation_results.png`: ROC-AUC changes after removing feature groups.
- `behavior_boxplots.png`: selected behavioral features by churn class.
- `delivery_delay_dist.png`: delivery delay distribution when available.
- `calibration_curves.png`: predicted probability calibration with bootstrap confidence bands.
- SHAP plots: summary bar, summary dot, and dependence plots.

# 15. Outputs

Implemented artifacts:

## CSV

- `processed_data/train_features.csv`
- `processed_data/test_features.csv`
- `processed_data/train_labels.csv`
- `processed_data/test_labels.csv`
- `results/model_metrics/model_metrics.csv`
- `results/statistical_tests/feature_tests.csv`
- `results/ablation/ablation_results.csv`
- `results/risk_scoring/{model}_risk_scores.csv`
- `results/data_quality/data_quality_summary.csv`
- `results/experiments/experiment_log.csv`
- `results/cross_dataset/master_results.csv`
- `results/failure_analysis/error_groups.csv`
- `results/failure_analysis/fp_fn_comparison.csv`
- `results/shap_values/{model}_shap_values.csv`
- optional `results/sensitivity_analysis/{dataset}_sensitivity.csv`
- optional `results/sensitivity_analysis/all_datasets_sensitivity.csv`

## Plots

All plots listed in section 14.

## Trained Models

- Logistic Regression joblib
- Random Forest joblib
- XGBoost joblib
- segmentation KMeans pickle
- segmentation scaler pickle
- segmentation PCA pickle

## Reports and Summaries

- data quality text report
- data quality CSV summary
- experiment log
- master cross-dataset results
- validation reports captured into experiment metadata

# 16. Design Decisions

## Temporal Split

The code uses temporal cutoffs instead of a simple random train/test split for the primary evaluation. This supports snapshot-based churn prediction.

## Schema Standardization

Adapters normalize heterogeneous datasets into common column names so shared feature engineering and modeling code can run across datasets.

## Train-Only Imbalance Handling

Class imbalance is handled during model fitting through class weights or XGBoost `scale_pos_weight`. No synthetic oversampling is implemented.

## Deterministic Sampling

Instacart large-table sampling uses `random_state=42`. Random baseline, train/validation split, KMeans, and seeds also use the configured random seed.

## Modular Adapters

Each dataset owns its loading, cleaning, schema normalization, metadata, churn-window choice, and feature-group availability.

## Configuration Layout

Centralized configuration avoids scattering model parameters, paths, feature groups, and validation thresholds across modules.

# 17. Known Limitations

Implemented limitations visible in the code:

- LightGBM is not implemented.
- SVM is not implemented.
- SMOTE is not implemented.
- MCC is not implemented.
- Balanced accuracy is not implemented.
- Feature selection is not implemented.
- Customer-disjoint train/test splitting is not implemented.
- Preprocessing occurs before temporal splitting, so global medians/quantile caps can use future rows.
- Telco uses static native labels and synthetic event times, not future-window temporal labels.
- RetailRocket monetary values default to zero unless a `transaction_value` column already exists.
- Instacart monetary values default to zero.
- Several declared constants are unused, including RetailRocket visits/category files and some Olist-compatible config filenames.
- Categorical encoding is limited to preferred payment type dummies.

# 18. Strengths

The strongest implemented technical aspects are:

- clear adapter contract for heterogeneous datasets
- standardized event schema
- modular feature-group construction
- snapshot-based feature generation
- future-window churn labeling for behavioral datasets
- centralized configuration
- broad artifact export coverage
- explicit validation layers
- deterministic seeds across major components
- graceful failure handling for optional analysis steps
- cross-dataset result table support
- integrated SHAP, calibration, ablation, segmentation, risk scoring, and failure analysis

# 19. Codebase Statistics

Approximate statistics from the current working directory:

- Python modules: 23 top-level `.py` files plus 8 files under `datasets/`
- Total Python lines: approximately 5,672
- Supported datasets: 6
- Implemented dataset adapters: 6
- Implemented trained ML models: 3
- Implemented baselines: 2
- Validation components: schema, behavioral, output, cross-dataset, master-results entry validation
- Output types: CSV, TXT, PNG, joblib, pickle
- Main package structure: flat `src` module layout with `datasets/` subpackage

# 20. Overall Assessment

The project is a modular research pipeline for behavioral churn modeling across multiple datasets. Its architecture is maintainable because dataset-specific logic is isolated in adapters and shared modeling/evaluation/reporting logic is centralized.

The strongest architectural choice is the standardized schema plus modular feature groups, which allows heterogeneous raw datasets to flow through common downstream code. The pipeline is also suitable for academic experimentation because it exports metrics, validation metadata, statistical tests, ablation results, figures, models, and cross-dataset summaries.

The main methodological issue is temporal leakage risk from full-dataset preprocessing before temporal splitting. The feature snapshot logic itself is temporally filtered, but the preprocessing order prevents a blanket claim of full leakage prevention. The implementation is extensible, but adding new datasets or models would require careful updates to adapters, feature-group availability, configuration, evaluation expectations, and exports.

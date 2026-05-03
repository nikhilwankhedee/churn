# Changelog

All notable changes to the Customer Churn Research Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-07-25 — v1.0 Architecture Freeze

### Added

#### Interactive Home Screen
- Running `churn` with no arguments now launches an interactive home screen instead of printing help
- Guided menu with 13 options: scan, wizard, download, list, benchmark, doctor, explain, compare, profile, export, experiments, dashboard, docs
- Zero-configuration onboarding for first-time users

#### Dataset Doctor (`churn doctor`)
- Comprehensive data health analyzer with 16 checks:
  - Duplicate customers, duplicate transactions, missing timestamps
  - Future timestamps, invalid dates, negative monetary values
  - Missing IDs, high missingness, class imbalance
  - Data leakage detection, duplicate columns, constant features
  - Extreme outliers (3xIQR), unsupported datatypes, schema compliance
- Weighted health score (0-100%) with severity tiers
- Actionable recommendations for every failed check

#### Auto Explanations (`churn explain`)
- Natural language model interpretation after training
- Feature importance ranking with direction (positive/negative churn impact)
- Risk factors and protection factors identification
- Actionable insights tailored to dataset characteristics
- Human-readable summary: "The model predicts churn primarily using purchase frequency, recency, and average order value."

#### Export Engine (`churn export`)
- Publication-ready outputs in 5 formats: LaTeX, Markdown, CSV, HTML, JSON
- LaTeX tables with proper `\begin{table}` formatting
- Styled HTML reports with metric cards
- JSON export for programmatic consumption

#### Download Manager (`churn datasets` / `churn download`)
- Catalog of 6 supported benchmark datasets with metadata
- Download instructions with URLs, licenses, and setup guides
- Dataset info: ecosystem type, customer count, time range

#### Experiment Management (`churn experiments` / `churn reproduce`)
- Automatic experiment recording on every pipeline run
- Full metadata: dataset, framework version, parameters, seed, hardware, runtime
- `churn experiments` lists history with status and duration
- `churn reproduce <id>` re-runs an experiment with identical parameters
- JSON-based experiment storage in `.experiments/`

#### Plugin Scaffolding (`churn plugin create`)
- Generate plugin templates for custom strategies, adapters, or models
- Pre-structured code with abstract methods stubbed out
- Plugin listing via `churn plugin list`

#### Dashboard (`churn dashboard`)
- Optional local web UI at `http://127.0.0.1:8420`
- Displays registered datasets, experiments, framework version
- Auto-opens in browser (configurable with `--no-browser`)
- Zero-dependency HTTP server

#### Enhanced Benchmark UX
- Rich progress bars with spinner, bar, percentage, and elapsed time
- Live model status display during training
- Clean completion summary

#### Enhanced Compare (`churn compare`)
- Multi-dataset distribution comparison
- Side-by-side metrics: rows, columns, customers, memory, missingness
- Data quality comparison across datasets

#### Enhanced Profile (`churn profile`)
- Automatic insights generation based on profile analysis
- Actionable recommendations for data quality issues
- Retention pattern detection

### Changed
- Package renamed from `churn-research-framework` to `churnlab`
- Entry point `churn` retained for backward compatibility
- Added `churnlab` as additional entry point
- CLI callback now shows interactive home screen when invoked without subcommand

### Architecture
- New modules: `src/doctor/`, `src/explain/`, `src/experiments/`, `src/downloads/`, `src/export/`, `src/dashboard/`, `src/plugins/`
- All new features are manifest-driven and plugin-compatible
- Interactive workflow: `pip install churnlab` → `churn` → guided experience
- First-time user can onboard unknown dataset without reading source code

## [1.1.0] - 2026-07-25

### Changed

#### Manifest-Driven Dataset Architecture (Primary Research Contribution)
- **Single YAML manifest** becomes the sole source of truth for dataset configuration — no more duplicated file names, column mappings, or preprocessing logic inside Python adapters
- **`GenericDatasetAdapter`** — a new manifest-driven adapter that handles load → preprocess → standardize from a YAML file alone, requiring zero Python code for most datasets
- **Unified dataset registry** — `get_dataset("name")` now resolves in order: built-in adapters → manifest files (project root + user registry) → GenericDatasetAdapter fallback
- **`manifest.yaml` format** (schema version 2) — includes `adapter.type`, `adapter.plugin`, `root_directory`, structured `files.required/optional`, `schema` mapping, `computed_columns`, `preprocessing` section
- **`root_directory`** in manifest eliminates hardcoded paths — the resolver reads the manifest to find where datasets live
- **Built-in adapters** (Olist, Telco, REES46, etc.) refactored to read file names and column mappings from manifests instead of module-level constants; keep custom multi-table join logic as plugin code
- **Multi-file merge strategies** in GenericDatasetAdapter: `concat` (year_1 + year_2 → single DataFrame) and `join` (orders + customers + payments via common key)
- **Computed columns** support — expressions like `transaction_value = Quantity * Price` applied during standardization
- **Event type mapping** — maps domain-specific event names (e.g., `view`, `transaction`) to standardized `event_type` column
- **Synthetic timestamp generation** — from `days_since_prior_order` when `event_time` is missing
- **`register_dataset()` API** — persists custom manifests to `.dataset_registry/manifests/` with ecosystem type inference

#### Dependency Strategy (Breaking Installation Fix)
- **Removed all restrictive upper bounds** on dependencies to prevent forced downgrades
- `numpy>=1.24,<2.0` → `numpy>=1.24` (now compatible with NumPy 2.x on Kaggle/Colab)
- `pandas>=2.0,<3.0` → `pandas>=2.0`
- `scikit-learn>=1.3,<2.0` → `scikit-learn>=1.3`
- `xgboost>=2.0,<3.0` → `xgboost>=2.0`
- `matplotlib>=3.7,<4.0` → `matplotlib>=3.7`
- `seaborn>=0.12,<1.0` → `seaborn>=0.12`
- `scipy>=1.11,<2.0` → `scipy>=1.11`
- `PyYAML>=6.0,<7.0` → `PyYAML>=6.0`
- `typer>=0.9,<1.0` → `typer>=0.9`
- `rich>=13.0,<14.0` → `rich>=13.0`
- `joblib>=1.3,<2.0` → `joblib>=1.3`
- Same relaxation applied to all optional dependencies (`shap`, `pingouin`, `statsmodels`, `imbalanced-learn`, `tqdm`, `openpyxl`, `jupyter`, `ipykernel`)
- This eliminates the `ValueError: numpy.dtype size changed` crash on Kaggle (NumPy 2.0.2 + Pandas 2.3.3)

#### Compatibility
- Added Python 3.13 to supported versions and classifiers
- Bumped `setuptools` build requirement to `>=69.0`
- Verified zero deprecated NumPy patterns (`np.int`, `np.float`, `np.bool`, etc.)
- Verified zero deprecated pandas patterns (`DataFrame.append`, etc.)
- No `distutils`, `imp`, or `pkg_resources` usage

#### Packaging
- Package now installs without modifying, downgrading, or breaking existing scientific Python environments
- Designed to coexist with Kaggle, Colab, JupyterLab, conda, virtualenv, and HPC environments

#### Bug Fixes
- Fixed hardcoded user-specific absolute path in `fig_generator.py` — now uses `FIG_ROOT` env var or auto-detected relative path
- Report templates (`experiment_report`, `technical_report`) now reference `FRAMEWORK_VERSION` from config instead of hardcoded string
- Fixed Churn column conversion in GenericDatasetAdapter for pandas 3.x StringDtype compatibility

#### Version
- Bumped to 1.1.0 across `pyproject.toml`, `config.py`, `CITATION.cff`, `README.md`

## [1.0.0] - 2026-07-25

### Added

#### Core Framework
- Universal plugin registry with lazy loading
- Pipeline context dataclass for state management
- YAML configuration system with dot-path accessor
- Data profiler with auto-detection of column roles

#### Dataset Support
- 6 dataset adapters: Olist, REES46, RetailRocket, Online Retail II, Instacart, Telco
- Automatic schema validation and behavioral statistics
- Cross-ecosystem comparison tools

#### Churn Strategies
- Inactivity-based labeling with configurable windows
- Subscription-based labeling for contractual churn
- Cadence-based labeling for habitual purchasers

#### Models
- Logistic Regression with balanced class weights
- Random Forest with balanced subsampling
- XGBoost with scale_pos_weight and early stopping

#### Evaluation
- 8 metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, ECE
- Temporal train/test split respecting time ordering
- Calibration curves with bootstrap confidence intervals
- SHAP-based explainability

#### Pipeline
- Full pipeline orchestrator with graceful degradation
- Batch execution across multiple datasets
- Sensitivity analysis for churn window robustness
- Ablation study for feature group importance

#### Reporting
- 7 report types: executive summary, technical, data quality, model comparison, calibration, explainability, experiment
- Markdown output with HTML/PDF export ready

#### CLI
- 20+ commands via Typer + Rich
- Dataset registration wizard
- Configuration validation
- Dataset readiness checking
- Experiment history explorer

#### Python API
- ChurnFramework class wrapping all pipeline capabilities
- Programmatic access to profiling, validation, comparison, and reporting

#### Testing
- 86 tests covering all components
- CLI integration tests
- Backward compatibility verification

### Fixed
- Initial release

### Changed
- Initial release

## [0.9.0] - 2026-07-01

### Added
- Phase 1: Core infrastructure (registry, context, config, profiler)
- Phase 2: Churn strategies, model wrappers, metrics, SMOTE integration
- Phase 3: Reports, batch execution, experiment explorer, resampler registry, plugins

## [0.8.0] - 2026-06-01

### Added
- Initial research pipeline
- 6 dataset adapters
- Feature engineering pipeline
- Model training and evaluation
- SHAP explainability
- Cross-dataset comparison

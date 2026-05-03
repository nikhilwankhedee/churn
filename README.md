# ChurnLab

**Universal Customer Churn Research Framework**

A production-ready framework for onboarding, benchmarking, explaining, and publishing churn prediction experiments across heterogeneous datasets.

## Installation

```bash
pip install churnlab
```

## First Run

```bash
churn
```

This launches the interactive home screen:

```
         _____ _                 _                    __
    ____/ /__(_)___  __      __(_)___  _____       / /___ _      _______
   / __  / / / __ \_ | /| / / / __ \/ ___/______/ / __ \| |/ /| / ___/
  / /_/ / / / / / / | |/ |/ / / / / (__  )_____/ / /_/ / |/ |/ (__  )
 /\__,_/_/_/_/ /_/  |__/|__/_/_/ /_/ /____/    /_/\____/|__/|__/____/

  v2.0.0 — Universal Customer Churn Research Framework

  6 dataset(s) registered: instacart, olist, online_retail_ii, rees46, retailrocket, telco

  What would you like to do?

  1   Scan current directory for datasets
  2   Register a new dataset (Wizard)
  3   Download benchmark datasets
  4   View registered datasets
  5   Run benchmark
  6   Dataset health check (Doctor)
  7   Explain model predictions
  8   Compare datasets
  9   Profile a dataset
  10  Export results
  11  View experiments
  12  Launch dashboard
  13  Documentation
  0   Exit
```

## Onboarding a New Dataset

```bash
# Point the wizard at any CSV
churn wizard path/to/data.csv

# The wizard:
#   ✓ Scans columns
#   ✓ Detects customer ID, timestamps, monetary values
#   ✓ Infers churn strategy
#   ✓ Generates manifest YAML
#   ✓ Registers dataset
#   ✓ Shows readiness score

# Check dataset health
churn doctor data.csv

# Run the benchmark
churn benchmark data.csv

# Explain the model
churn explain my_dataset

# Export publication-ready results
churn export my_dataset
```

## Complete Workflow

```bash
# 1. Install
pip install churnlab

# 2. Launch interactive mode
churn

# 3. Register a dataset
churn wizard data.csv --name my_company

# 4. Check health
churn doctor data.csv

# 5. Benchmark
churn run my_company

# 6. Explain
churn explain my_company

# 7. Compare with benchmarks
churn compare my_company,olist,telco

# 8. Export for publication
churn export my_company --formats latex,html,markdown

# 9. Reproduce later
churn experiments
churn reproduce experiment_my_company_20260725_120000
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `churn` | Interactive home screen |
| `churn wizard` | Dataset registration wizard |
| `churn doctor` | Comprehensive data health analysis |
| `churn benchmark` | Auto-discover and benchmark datasets |
| `churn explain` | Natural language model explanation |
| `churn compare` | Compare datasets and distributions |
| `churn profile` | Profile a dataset with insights |
| `churn export` | Export results (LaTeX, HTML, CSV, etc.) |
| `churn datasets` | List supported benchmark datasets |
| `churn download` | Get download instructions |
| `churn experiments` | List experiment history |
| `churn reproduce` | Re-run a previous experiment |
| `churn dashboard` | Launch web dashboard |
| `churn plugin create` | Generate plugin template |
| `churn run one` | Run pipeline on single dataset |
| `churn run all` | Run pipeline on all datasets |
| `churn validate` | Validate dataset schema |
| `churn profile` | Comprehensive data profiling |
| `churn health` | Framework health check |

## Supported Datasets

| Dataset | Ecosystem | Customers | Window |
|---------|-----------|-----------|--------|
| Olist | Transactional Marketplace | 99,441 | 180d |
| REES46 | Transactional Marketplace | 700,000 | 180d |
| RetailRocket | Clickstream Commerce | 14,000 | 30d |
| Online Retail II | Habitual Retail | 4,372 | 90d |
| Instacart | Habitual Retail | 206,209 | 60d |
| IBM Telco | Subscription | 7,043 | Native |

## Features

- **Interactive home screen** — guided workflow for first-time users
- **Dataset Wizard** — intelligent CSV inspection and manifest generation
- **Dataset Doctor** — 16 health checks with actionable recommendations
- **Auto Explanations** — natural language model interpretation
- **Benchmark UX** — progress bars, live status, leaderboard
- **Publication Export** — LaTeX, Markdown, CSV, HTML, JSON
- **Experiment Management** — track, list, and reproduce runs
- **Dashboard** — optional local web UI
- **Plugin System** — extend with custom strategies, adapters, models
- **8 behavioral feature groups** — purchase, monetary, inactivity, review, delivery, payment, engagement, cadence
- **3 models** — Logistic Regression, Random Forest, XGBoost
- **8+ metrics** — AUC, F1, precision, recall, Brier, calibration error
- **SHAP explainability** — feature importance analysis
- **4-layer validation** — schema, behavioral, output, cross-dataset

## Python API

```python
from src.api import ChurnFramework

fw = ChurnFramework()
result = fw.run("olist")
profile = fw.profile("olist")
comparison = fw.compare(["olist", "rees46"])
```

## Architecture

```
churnlab/
├── src/
│   ├── cli/            # Interactive CLI (Typer + Rich)
│   ├── core/           # Registry, context, infrastructure
│   ├── datasets/       # Dataset adapters (built-in + generic)
│   ├── churn/          # Churn labeling strategies
│   ├── models/         # Model wrappers
│   ├── metrics/        # Evaluation metrics
│   ├── reports/        # Report generators
│   ├── wizard/         # Dataset registration wizard
│   ├── profiling/      # Data profiling
│   ├── doctor/         # Dataset health analyzer
│   ├── explain/        # Auto-explanation engine
│   ├── experiments/    # Experiment management
│   ├── downloads/      # Benchmark dataset catalog
│   ├── export/         # Publication export engine
│   ├── dashboard/      # Web dashboard
│   ├── plugins/        # Plugin scaffolding
│   └── discovery/      # Dataset auto-discovery
├── configs/            # YAML manifests
├── docs/               # Documentation
└── tests/              # Test suite
```

## Development

```bash
git clone <repo-url>
cd churnlab
pip install -e ".[dev]"
churn health  # verify installation
```

## Citation

```bibtex
@software{wankhede2026churnlab,
  author    = {Nikhil Wankhede},
  title     = {ChurnLab: Universal Customer Churn Research Framework},
  version   = {2.0.0},
  year      = {2026},
  url       = {https://github.com/your-username/churnlab}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

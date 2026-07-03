# Quick Start

Get up and running in 5 minutes.

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd arxiv-retention-paper-research-material/notebooks/code/project_root

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## First Run

```bash
# Check framework health
churn doctor

# List available datasets
churn datasets

# Run the full pipeline on Olist
churn run olist
```

## Register a New Dataset

```bash
# Inspect your CSV and generate a config
churn register path/to/your/data.csv --name my_dataset

# Validate the generated config
churn validate-config configs/datasets/my_dataset.yaml

# Run on your new dataset
churn run my_dataset
```

## Python API

```python
from src.api import ChurnFramework

fw = ChurnFramework()

# Run pipeline
result = fw.run("olist")

# Profile a dataset
profile = fw.profile("olist")

# Compare datasets
comparison = fw.compare(["olist", "rees46"])

# List available components
print(fw.list_datasets())
print(fw.list_models())
print(fw.list_metrics())
```

## What Just Happened

1. **Data Loading**: The framework loaded raw data via the dataset adapter
2. **Preprocessing**: Timestamps parsed, outliers clipped, values imputed
3. **Churn Labeling**: 180-day inactivity window applied
4. **Feature Engineering**: 8 groups of behavioral features computed
5. **Model Training**: Logistic Regression, Random Forest, XGBoost trained
6. **Evaluation**: 8+ metrics computed per model
7. **Explainability**: SHAP values computed for top features
8. **Calibration**: Probability calibration curves generated
9. **Risk Scoring**: Customer risk scores assigned
10. **Segmentation**: Customer segments identified
11. **Reports**: Markdown reports generated

## Next Steps

- [CLI Reference](cli.md) — All available commands
- [API Guide](api.md) — Programmatic access
- [Dataset Registration](registration.md) — Adding new datasets
- [Configuration](configuration.md) — Customizing the framework

---

*Developed by Nikhil Wankhede*

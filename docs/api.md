# Python API Reference

The `ChurnFramework` class provides programmatic access to all framework capabilities.

## Quick Start

```python
from src.api import ChurnFramework

fw = ChurnFramework()
result = fw.run("olist")
```

## Constructor

```python
ChurnFramework(config_path: str = None)
```

- `config_path` — Optional path to a YAML configuration file

## Pipeline Methods

### `run(dataset, sensitivity=False, churn_window=None)`

Run the full prediction pipeline.

```python
result = fw.run("olist")
result = fw.run("rees46", sensitivity=True)
result = fw.run("olist", churn_window=90)
```

**Returns**: Pipeline result dictionary with models, metrics, and outputs.

### `run_batch(datasets=None, sensitivity=False)`

Run the pipeline on multiple datasets.

```python
batch = fw.run_batch(["olist", "rees46"])
print(batch.successful)
print(batch.failed)
```

**Returns**: `BatchResult` with per-dataset results.

## Profiling Methods

### `profile(dataset)`

Profile a registered dataset.

```python
profile = fw.profile("olist")
print(profile["n_rows"], profile["n_customers"])
```

### `profile_csv(csv_path)`

Profile a raw CSV file.

```python
profile = fw.profile_csv("data.csv")
```

## Validation Methods

### `validate(dataset)`

Validate a dataset's schema and behavior.

```python
reports = fw.validate("olist")
print(reports["schema"])
print(reports["behavioral"])
```

### `validate_config(config_path)`

Validate a YAML configuration file.

```python
result = fw.validate_config("config.yaml")
print(result["is_valid"])
```

## Experiment History

### `list_experiments(dataset=None, limit=20)`

List experiment runs.

```python
experiments = fw.list_experiments()
experiments = fw.list_experiments(dataset="olist", limit=5)
```

### `compare(datasets)`

Compare experiments across datasets.

```python
comparison = fw.compare(["olist", "rees46"])
```

### `get_feature_comparison()`

Compare feature groups across datasets.

```python
features = fw.get_feature_comparison()
```

## Report Generation

### `generate_reports(pipeline_result)`

Generate reports from a pipeline result.

```python
reports = fw.generate_reports(result)
for name, markdown in reports.items():
    print(f"## {name}\n{markdown[:200]}")
```

## Registry Introspection

```python
fw.list_datasets()      # ['olist', 'rees46', ...]
fw.list_models()        # ['logistic_regression', 'random_forest', 'xgboost']
fw.list_metrics()       # ['accuracy', 'precision', ...]
fw.list_strategies()    # ['inactivity', 'subscription', 'cadence']
fw.list_resamplers()    # ['smote', 'adasyn']
fw.list_reports()       # ['executive_summary', 'technical_report', ...]
```

## Dataset Registration

```python
config_path = fw.register_dataset(
    csv_path="data.csv",
    name="my_dataset",
    ecosystem="transactional_marketplace",
    customer_id="user_id",
    timestamp="order_date",
    source_url="https://example.com/data",
    citation="Author et al. (2024)",
)
```

## Health Check

```python
checks = fw.doctor()
for component, status in checks.items():
    print(f"{component}: {'OK' if status['ok'] else 'FAIL'}")
```

---

*Developed by Nikhil Wankhede*

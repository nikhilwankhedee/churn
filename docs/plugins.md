# Plugin Development

Extend the framework with custom components.

## Plugin Types

| Type | Base Class | Registration |
|------|-----------|-------------|
| Dataset | `BaseDatasetAdapter` | `src/datasets/` |
| Churn Strategy | `ChurnStrategy` | `src/churn/` |
| Model | `ModelWrapper` | `src/models/` |
| Metric | `EvaluationMetric` | `src/metrics/` |
| Resampler | `Resampler` | `src/resamplers/` |
| Report | `Report` | `src/reports/` |

## Creating a Custom Model

```python
# src/models/my_model.py
from src.models.base import ModelWrapper

class MyModel(ModelWrapper):
    @property
    def model_name(self) -> str:
        return "my_model"

    @property
    def description(self) -> str:
        return "My custom model"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Train your model
        pass

    def predict(self, X):
        # Return binary predictions
        pass

    def predict_proba(self, X):
        # Return [neg_prob, pos_prob]
        pass
```

## Creating a Custom Metric

```python
# src/metrics/my_metric.py
from src.metrics.base import EvaluationMetric, MetricResult

class MyMetric(EvaluationMetric):
    @property
    def metric_name(self) -> str:
        return "my_metric"

    @property
    def description(self) -> str:
        return "My custom metric"

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute(self, y_true, y_pred=None, y_proba=None):
        score = my_computation(y_true, y_proba)
        return MetricResult(name=self.metric_name, value=score)
```

## Creating a Custom Report

```python
# src/reports/my_report.py
from src.reports.base import Report, ReportOutput, ReportSection

class MyReport(Report):
    @property
    def report_name(self) -> str:
        return "my_report"

    @property
    def description(self) -> str:
        return "My custom report"

    def generate(self, pipeline_result):
        return ReportOutput(
            name=self.report_name,
            title="My Report",
            sections=[ReportSection(title="Summary", content="...")],
        )
```

## Plugin Discovery

The framework discovers plugins from:

1. **Local directory**: `plugins/` in project root
2. **Entry points**: `churn_framework.plugins` group

### Local Plugins

Place Python files in `plugins/`:

```
plugins/
  my_model.py
  my_metric.py
```

### Entry Points

In your package's `setup.py` or `pyproject.toml`:

```toml
[project.entry-points."churn_framework.plugins"]
my_model = "my_package.models:MyModel"
```

## Viewing Plugins

```bash
churn plugins
```

Shows all registered plugins across all categories.

---

*Developed by Nikhil Wankhede*

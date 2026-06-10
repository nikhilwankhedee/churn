"""
Pipeline context: a single object that carries all state through a pipeline run.

Replaces the pattern of passing 20+ variables between pipeline steps.
Every step reads from and writes to this context.

The context is serializable for experiment tracking and debugging.

Usage:
    ctx = PipelineContext(dataset="olist")
    ctx.adapter = get_dataset("olist")
    ctx.df = adapter.load_raw_data()
    # ... pass ctx to every step
"""
import datetime
import dataclasses
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclasses.dataclass
class PipelineContext:
    """Carries all pipeline state through a single run."""

    # ── Identity ──────────────────────────────────────────────────
    dataset: str = "olist"
    run_id: str = ""
    ecosystem_type: str = ""
    framework_version: str = ""

    # ── Configuration ─────────────────────────────────────────────
    config: Optional[dict] = None
    churn_window_days: int = 180
    train_split_quantile: float = 0.7
    random_seed: int = 42
    smote_enabled: bool = False

    # ── Adapter ───────────────────────────────────────────────────
    adapter: Any = None  # BaseDatasetAdapter instance

    # ── Data ──────────────────────────────────────────────────────
    df: Optional[pd.DataFrame] = None
    available_feature_groups: Optional[List[str]] = None

    # ── Temporal Split ────────────────────────────────────────────
    train_cutoff: Optional[pd.Timestamp] = None
    test_cutoff: Optional[pd.Timestamp] = None

    # ── Labels ────────────────────────────────────────────────────
    train_labels: Optional[pd.DataFrame] = None
    test_labels: Optional[pd.DataFrame] = None

    # ── Features ──────────────────────────────────────────────────
    train_features: Optional[pd.DataFrame] = None
    test_features: Optional[pd.DataFrame] = None

    # ── Models ────────────────────────────────────────────────────
    models: Optional[Dict[str, Any]] = None
    best_model_name: str = ""
    best_model: Any = None

    # ── Evaluation ────────────────────────────────────────────────
    eval_df: Optional[pd.DataFrame] = None
    prob_dict: Optional[Dict[str, Any]] = None
    pr_data: Optional[Dict[str, tuple]] = None
    cm_dict: Optional[Dict[str, Any]] = None
    imb_train: Optional[Dict[str, float]] = None
    imb_test: Optional[Dict[str, float]] = None

    # ── Analysis ──────────────────────────────────────────────────
    ablation_df: Optional[pd.DataFrame] = None
    stat_results: Optional[pd.DataFrame] = None
    shap_values: Optional[Dict[str, Any]] = None
    risk_df: Optional[pd.DataFrame] = None

    # ── Validation ────────────────────────────────────────────────
    schema_report: Optional[Dict] = None
    behavioral_report: Optional[Dict] = None
    output_report: Optional[Dict] = None
    dq_report: Optional[Dict] = None

    # ── Timing ────────────────────────────────────────────────────
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    step_timings: Optional[Dict[str, float]] = None

    # ── Output ────────────────────────────────────────────────────
    output_dir: str = ""
    figures_dir: str = ""
    results_dir: str = ""
    models_dir: str = ""

    def generate_run_id(self) -> str:
        """Generate a unique run identifier."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{self.dataset}_{ts}"
        return self.run_id

    def start(self) -> None:
        """Mark the pipeline start time."""
        self.start_time = datetime.datetime.utcnow()
        if not self.run_id:
            self.generate_run_id()

    def finish(self) -> None:
        """Mark the pipeline end time."""
        self.end_time = datetime.datetime.utcnow()

    @property
    def duration_seconds(self) -> float:
        """Elapsed time in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the context to a dict for experiment tracking."""
        result = {}
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if val is None:
                continue
            if isinstance(val, pd.DataFrame):
                result[f.name] = f"DataFrame({val.shape})"
            elif isinstance(val, dict):
                result[f.name] = str(val)
            elif hasattr(val, '__class__') and val.__class__.__name__ not in (
                'OlistAdapter', 'REES46Adapter', 'RetailRocketAdapter',
                'OnlineRetailIIAdapter', 'InstacartAdapter', 'TelcoAdapter',
            ):
                try:
                    result[f.name] = val
                except Exception:
                    result[f.name] = str(val)
            else:
                result[f.name] = str(val)
        return result

    def step_timing(self, step_name: str, elapsed: float) -> None:
        """Record timing for a pipeline step."""
        if self.step_timings is None:
            self.step_timings = {}
        self.step_timings[step_name] = elapsed

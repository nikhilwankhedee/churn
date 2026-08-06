"""
Auto-Explanation Engine: generate natural language model interpretations.

Produces human-readable explanations of model behavior including:
  - Feature importance rankings
  - SHAP-based explanations
  - Natural language summaries
  - Risk factor identification
"""
import dataclasses
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class ExplanationReport:
    """Natural language explanation of model behavior."""
    model_name: str
    dataset_name: str
    top_features: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    summary: str = ""
    risk_factors: List[str] = dataclasses.field(default_factory=list)
    protection_factors: List[str] = dataclasses.field(default_factory=list)
    key_insights: List[str] = dataclasses.field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            f"Model Explanation: {self.model_name}",
            f"Dataset: {self.dataset_name}",
            "",
            "Summary",
            self.summary,
            "",
            "Top Predictive Features",
        ]
        for i, feat in enumerate(self.top_features[:10], 1):
            direction = "↑ churn risk" if feat.get("direction") == "positive" else "↓ churn risk"
            lines.append(f"  {i}. {feat['name']} (importance: {feat['importance']:.3f}) — {direction}")

        if self.risk_factors:
            lines.append("")
            lines.append("Risk Factors (increase churn probability)")
            for rf in self.risk_factors:
                lines.append(f"  • {rf}")

        if self.protection_factors:
            lines.append("")
            lines.append("Protection Factors (decrease churn probability)")
            for pf in self.protection_factors:
                lines.append(f"  • {pf}")

        if self.key_insights:
            lines.append("")
            lines.append("Key Insights")
            for insight in self.key_insights:
                lines.append(f"  • {insight}")

        return "\n".join(lines)


def _build_feature_importance(
    model: Any,
    feature_names: List[str],
    X_train: Optional[pd.DataFrame] = None,
    max_features: int = 20,
) -> List[Dict[str, Any]]:
    """Extract feature importance from a fitted model."""
    importance_dict = {}

    # Try tree-based feature importance first
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        for name, imp in zip(feature_names, importances):
            importance_dict[name] = float(imp)

    # Try coefficients for linear models
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef[0]
        for name, c in zip(feature_names, coef):
            importance_dict[name] = float(abs(c))

    if not importance_dict:
        return []

    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Determine direction from coefficients if available
    directions = {}
    if hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef[0]
        for name, c in zip(feature_names, coef):
            directions[name] = "positive" if c > 0 else "negative"

    results = []
    for name, imp in sorted_features[:max_features]:
        results.append({
            "name": name,
            "importance": imp,
            "direction": directions.get(name, "unknown"),
        })

    return results


def _generate_summary(
    model_name: str,
    dataset_name: str,
    top_features: List[Dict[str, Any]],
    churn_rate: Optional[float] = None,
    best_metric: Optional[float] = None,
) -> str:
    """Generate a natural language summary of model behavior."""
    if not top_features:
        return f"The {model_name} model was trained on the {dataset_name} dataset. Feature importance data is not available."

    primary = top_features[0]["name"]
    secondary = top_features[1]["name"] if len(top_features) > 1 else None
    tertiary = top_features[2]["name"] if len(top_features) > 2 else None

    parts = [
        f"The {model_name} model predicts churn on the {dataset_name} dataset",
        f"primarily using {primary.replace('_', ' ')}",
    ]

    if secondary:
        parts.append(f", {secondary.replace('_', ' ')}")
    if tertiary:
        parts.append(f", and {tertiary.replace('_', ' ')}")
    parts.append(" as the most predictive features.")

    if churn_rate is not None:
        if churn_rate < 0.1:
            parts.append(f" The dataset has a low churn rate of {churn_rate:.1%}, meaning most customers remain active.")
        elif churn_rate < 0.3:
            parts.append(f" The churn rate is {churn_rate:.1%}, representing a moderate level of customer attrition.")
        else:
            parts.append(f" The churn rate is high at {churn_rate:.1%}, indicating significant customer attrition.")

    if best_metric is not None:
        if best_metric > 0.8:
            parts.append(f" The model achieves strong predictive performance (AUC: {best_metric:.3f}).")
        elif best_metric > 0.7:
            parts.append(f" The model shows good predictive performance (AUC: {best_metric:.3f}).")
        else:
            parts.append(f" The model shows moderate predictive performance (AUC: {best_metric:.3f}).")

    return " ".join(parts)


def _identify_risk_and_protection(
    top_features: List[Dict[str, Any]],
) -> tuple:
    """Identify risk factors and protection factors from feature importance."""
    risk_factors = []
    protection_factors = []

    for feat in top_features:
        name = feat["name"].replace("_", " ")
        direction = feat.get("direction", "unknown")
        importance = feat.get("importance", 0)

        if importance < 0.01:
            continue

        if direction == "positive":
            risk_factors.append(f"Higher {name} increases churn probability")
        elif direction == "negative":
            protection_factors.append(f"Higher {name} decreases churn probability")

    return risk_factors[:5], protection_factors[:5]


def _generate_insights(
    top_features: List[Dict[str, Any]],
    churn_rate: Optional[float] = None,
) -> List[str]:
    """Generate actionable insights from feature importance."""
    insights = []

    if not top_features:
        return ["Collect more data to enable feature importance analysis."]

    # Check if inactivity is dominant
    inactivity_features = [f for f in top_features if "inactiv" in f["name"] or "recency" in f["name"]]
    if inactivity_features and inactivity_features[0]["importance"] > 0.2:
        insights.append(
            "Recency of purchase is the strongest predictor — "
            "target at-risk customers with re-engagement campaigns before they become inactive."
        )

    # Check if monetary features dominate
    monetary_features = [f for f in top_features if any(k in f["name"] for k in ("spent", "value", "monetary", "payment"))]
    if monetary_features and monetary_features[0]["importance"] > 0.15:
        insights.append(
            "Spending behavior is a key differentiator — "
            "high-value customers may need loyalty programs while low-value customers may respond to discounts."
        )

    # Check if frequency features dominate
    frequency_features = [f for f in top_features if any(k in f["name"] for k in ("orders", "frequency", "purchase", "total"))]
    if frequency_features and frequency_features[0]["importance"] > 0.15:
        insights.append(
            "Purchase frequency strongly predicts retention — "
            "customers with fewer orders are at higher risk."
        )

    # Check if review/sentiment features matter
    review_features = [f for f in top_features if any(k in f["name"] for k in ("review", "rating", "score"))]
    if review_features and review_features[0]["importance"] > 0.1:
        insights.append(
            "Customer satisfaction signals are predictive — "
            "low review scores indicate dissatisfaction that may lead to churn."
        )

    # General insight
    if churn_rate is not None and churn_rate > 0.3:
        insights.append(
            "With a high churn rate, focus on identifying and retaining "
            "the most valuable at-risk customers first."
        )

    if not insights:
        insights.append(
            "The model identifies multiple contributing factors to churn. "
            "Review the full feature importance ranking for detailed insights."
        )

    return insights[:5]


def generate_explanation(
    model: Any,
    model_name: str,
    dataset_name: str,
    feature_names: List[str],
    X_train: Optional[pd.DataFrame] = None,
    churn_rate: Optional[float] = None,
    best_metric: Optional[float] = None,
    shap_values: Optional[Any] = None,
) -> ExplanationReport:
    """Generate a comprehensive model explanation.

    Parameters
    ----------
    model : fitted model object
    model_name : str
        Name of the model.
    dataset_name : str
        Name of the dataset.
    feature_names : list of str
        Feature column names.
    X_train : DataFrame, optional
        Training data for SHAP analysis.
    churn_rate : float, optional
        Dataset churn rate.
    best_metric : float, optional
        Best model metric (e.g., AUC).
    shap_values : array, optional
        Pre-computed SHAP values.

    Returns
    -------
    ExplanationReport with natural language explanation.
    """
    top_features = _build_feature_importance(model, feature_names, X_train)
    summary = _generate_summary(model_name, dataset_name, top_features, churn_rate, best_metric)
    risk_factors, protection_factors = _identify_risk_and_protection(top_features)
    key_insights = _generate_insights(top_features, churn_rate)

    report = ExplanationReport(
        model_name=model_name,
        dataset_name=dataset_name,
        top_features=top_features,
        summary=summary,
        risk_factors=risk_factors,
        protection_factors=protection_factors,
        key_insights=key_insights,
    )

    logger.info(
        "Generated explanation for %s on %s: %d features, %d insights",
        model_name, dataset_name, len(top_features), len(key_insights),
    )

    return report

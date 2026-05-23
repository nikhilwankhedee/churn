"""
Churn risk scoring: map predicted probabilities to 0–100 scores and risk tiers.
"""
import pandas as pd
import numpy as np


def compute_risk_scores(y_proba: np.ndarray, scale: int = 100) -> np.ndarray:
    return np.round(y_proba * scale).astype(int)


def assign_risk_tier(score: int, low: int = 30, high: int = 70) -> str:
    if score < low:
        return 'Low'
    if score < high:
        return 'Medium'
    return 'High'


def generate_risk_table(
    customer_ids: pd.Index,
    y_proba: np.ndarray,
    model_name: str = 'model',
) -> pd.DataFrame:
    scores = compute_risk_scores(y_proba)
    df = pd.DataFrame({
        'customer_unique_id': customer_ids,
        'churn_probability': y_proba,
        'risk_score': scores,
    })
    df['risk_percentile'] = df['risk_score'].rank(pct=True) * 100
    df['risk_tier'] = df['risk_score'].apply(assign_risk_tier)
    df = df.sort_values('risk_score', ascending=False).reset_index(drop=True)
    return df

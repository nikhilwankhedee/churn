"""
Publication-quality visualisation with consistent styling.

All plots are automatically saved to structured subdirectories under FIGURES_DIR.
Plots are closed after saving to prevent matplotlib memory leaks.
Headless-safe (uses 'Agg' backend).
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from sklearn.metrics import (
    RocCurveDisplay, ConfusionMatrixDisplay, PrecisionRecallDisplay,
)
from src.config import SAVEFIG_DPI, FONT_SIZE
from src.run_context import figures_dir
from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)

# ── Global style ────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': SAVEFIG_DPI,
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 2,
    'axes.labelsize': FONT_SIZE + 1,
    'legend.fontsize': FONT_SIZE - 1,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS: sns.palettes._ColorPalette = sns.color_palette("colorblind")
sns.set_palette(COLORS)


def _save_and_close(fig, filepath, tight=True):
    ensure_dir(os.path.dirname(filepath))
    if tight:
        fig.tight_layout()
    fig.savefig(filepath, dpi=SAVEFIG_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.debug("Saved figure: %s", filepath)


# ── ROC curves ──────────────────────────────────────────────────────
def plot_roc_curves(prob_dict: Dict[str, np.ndarray],
                    y_test: pd.Series, suffix: str = '') -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in prob_dict.items():
        RocCurveDisplay.from_predictions(y_test, probs, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_title('ROC Curves')
    _save_and_close(fig, os.path.join(figures_dir('model_evaluation'),
                                      f'roc_curves{suffix}.png'))
    logger.info("ROC curves saved")


# ── PR curves ───────────────────────────────────────────────────────
def plot_pr_curves(pr_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
                   y_test: pd.Series, suffix: str = '') -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (prec, rec, ap) in pr_data.items():
        ax.step(rec, prec, where='post', label=f'{name} (AP={ap:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc='best')
    _save_and_close(fig, os.path.join(figures_dir('model_evaluation'),
                                      f'pr_curves{suffix}.png'))
    logger.info("PR curves saved")


# ── Confusion matrices ──────────────────────────────────────────────
def plot_confusion_matrices(cms: Dict[str, np.ndarray], suffix: str = '') -> None:
    for name, cm in cms.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=['Retained', 'Churned']).plot(
            ax=ax, cmap='Blues', values_format='d')
        ax.set_title(f'Confusion Matrix — {name}')
        _save_and_close(fig, os.path.join(figures_dir('model_evaluation'),
                                          f'confusion_{name}{suffix}.png'))
    logger.info("Confusion matrices saved")


# ── Feature importance (tree-based) ─────────────────────────────────
def plot_feature_importance(imp_df: pd.DataFrame, title: str = '',
                            save_path: str = None, top_n: int = 20) -> None:
    if save_path is None:
        save_path = os.path.join(figures_dir('model_evaluation'),
                                 'feature_importance.png')
    imp = imp_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(6, len(imp) * 0.35)))
    sns.barplot(data=imp, x='importance', y='feature', ax=ax,
                hue='feature', palette='viridis', legend=False)
    ax.set_title(title)
    _save_and_close(fig, save_path)


# ── Correlation heatmap ─────────────────────────────────────────────
def plot_correlation_heatmap(features: pd.DataFrame, suffix: str = '') -> None:
    corr = features.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    _save_and_close(fig, os.path.join(figures_dir('correlation_analysis'),
                                      f'correlation_heatmap{suffix}.png'))
    logger.info("Correlation heatmap saved")


# ── Segmentation scatter ────────────────────────────────────────────
def plot_segmentation(seg_df: pd.DataFrame, suffix: str = '') -> None:
    if 'pca_x' not in seg_df or 'pca_y' not in seg_df:
        logger.warning("PCA columns missing — skipping segmentation plot")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=seg_df, x='pca_x', y='pca_y',
                    hue='segment', palette='tab10', ax=ax, alpha=0.7)
    centroids = seg_df.groupby('segment')[['pca_x', 'pca_y']].mean()
    ax.scatter(centroids['pca_x'], centroids['pca_y'],
               c='black', marker='X', s=120, label='Centroid')
    ax.set_title('Customer Segments (PCA Projection)')
    ax.legend()
    _save_and_close(fig, os.path.join(figures_dir('segmentation'),
                                      f'segments_scatter{suffix}.png'))
    logger.info("Segmentation plot saved")


# ── Churn distribution bar ──────────────────────────────────────────
def plot_churn_distribution(labels: pd.Series, suffix: str = '') -> None:
    counts = labels.value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(['Retained', 'Churned'], counts.values,
                  color=[COLORS[0], COLORS[3]], edgecolor='white')
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{v:.1f}%', ha='center', fontsize=FONT_SIZE)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Churn Distribution')
    _save_and_close(fig, os.path.join(figures_dir('churn_analysis'),
                                      f'churn_distribution{suffix}.png'))
    logger.info("Churn distribution saved")


# ── Threshold analysis ──────────────────────────────────────────────
def plot_threshold_analysis(thresh_df: pd.DataFrame, model_name: str,
                            suffix: str = '') -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresh_df['threshold'], thresh_df['precision'], label='Precision')
    ax.plot(thresh_df['threshold'], thresh_df['recall'], label='Recall')
    ax.plot(thresh_df['threshold'], thresh_df['f1'], label='F1')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title(f'{model_name} — Threshold Analysis')
    ax.legend(loc='best')
    _save_and_close(fig, os.path.join(figures_dir('model_evaluation'),
                                      f'{model_name}_threshold{suffix}.png'))
    logger.info("Threshold analysis saved for %s", model_name)


# ── Ablation results ────────────────────────────────────────────────
def plot_ablation_results(ablation_df: pd.DataFrame,
                          metric: str = 'mean_roc_auc',
                          suffix: str = '') -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in ablation_df['model'].unique():
        sub = ablation_df[ablation_df['model'] == model]
        ax.plot(sub['feature_set'], sub[metric], marker='o', label=model)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title('Ablation Study')
    ax.legend(loc='best')
    fig.tight_layout()
    _save_and_close(fig, os.path.join(figures_dir('model_evaluation'),
                                      f'ablation_results{suffix}.png'))
    logger.info("Ablation plot saved")


# ── Behavioural boxplots ────────────────────────────────────────────
def plot_behavioral_insights(features: pd.DataFrame,
                              churn_labels: pd.Series,
                              suffix: str = '') -> None:
    key_feats = ['days_since_last_purchase', 'total_orders',
                 'total_spent', 'avg_review_score']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, feat in zip(axes.flatten(), key_feats):
        if feat not in features.columns:
            ax.set_visible(False)
            continue
        tmp = pd.DataFrame({'value': features[feat],
                            'Churn': churn_labels.map({0: 'Retained',
                                                       1: 'Churned'})})
        sns.boxplot(data=tmp, x='Churn', y='value', ax=ax, palette='Set2',
                    hue='Churn', legend=False)
        ax.set_title(feat.replace('_', ' ').title())
    fig.suptitle('Behavioural Comparison by Churn Status', fontsize=FONT_SIZE + 2)
    _save_and_close(fig, os.path.join(figures_dir('behavioral_insights'),
                                      f'behavior_boxplots{suffix}.png'))
    logger.info("Behavioural insights saved")


# ── Delivery-time histogram (if data available) ─────────────────────
def plot_delivery_delay_distribution(features: pd.DataFrame, suffix: str = '') -> None:
    if 'avg_delivery_delay_days' not in features:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(features['avg_delivery_delay_days'], bins=50,
                 kde=True, ax=ax)
    ax.set_title('Distribution of Avg Delivery Delay')
    _save_and_close(fig, os.path.join(figures_dir('dataset_analysis'),
                                      f'delivery_delay_dist{suffix}.png'))
    logger.info("Delivery delay histogram saved")

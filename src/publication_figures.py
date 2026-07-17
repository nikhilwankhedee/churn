"""
Publication-ready figure generation.

Generates the full research figure set in publication styling (PNG @ 300 DPI
and PDF) organised into ``results/figures/main/`` (master-level summaries) and
``results/figures/supplementary/`` (per dataset × SMOTE condition):

Supplementary (per condition, every model)
    roc_curves, pr_curves, calibration_curves, confusion_matrices,
    feature_importance (best model, top 20), shap_summary (best model —
    skipped gracefully with the reason recorded when SHAP cannot run).

Main
    model × dataset ROC-AUC heatmap (with / without SMOTE), dataset
    performance, SMOTE effect (negative effects are shown explicitly),
    model ranking, metric distributions.
"""
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation import get_calibration_data, get_pr_data, get_roc_data
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

FINAL_MODELS = ['logistic_regression', 'random_forest', 'xgboost',
                'lightgbm', 'svm']
MODEL_LABELS = {
    'logistic_regression': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM',
    'svm': 'SVM',
    'majority_class': 'Majority Class',
    'random_baseline': 'Random',
}
_PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def _savefig(fig, png_path: str, pdf_path: Optional[str] = None) -> None:
    ensure_dir(os.path.dirname(png_path))
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    if pdf_path:
        ensure_dir(os.path.dirname(pdf_path))
        fig.savefig(pdf_path, bbox_inches='tight')


def _matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'font.size': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'legend.frameon': False,
    })
    return plt


# ═════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY (per dataset × SMOTE condition)
# ═════════════════════════════════════════════════════════════════════

def _condition_dir(experiment_dir: str, dataset: str, cond: str) -> str:
    return os.path.join(experiment_dir, 'results', cond, dataset)


def _best_model_path(experiment_dir: str, dataset: str, cond: str) -> str:
    return os.path.join(experiment_dir, 'results', 'master', 'best_models',
                        dataset, cond, 'best_model.pkl')


def _load_condition(experiment_dir: str, dataset: str, cond: str):
    base = _condition_dir(experiment_dir, dataset, cond)
    preds_path = os.path.join(base, 'predictions.csv')
    feats_path = os.path.join(base, 'test_features.csv')
    if not (os.path.isfile(preds_path) and os.path.isfile(feats_path)):
        return None
    preds = pd.read_csv(preds_path)
    feats = pd.read_csv(feats_path, index_col=0)
    feats = feats.reindex(preds['customer_id'].astype(str))
    return preds, feats


def _feature_importance(model, feature_names: List[str]):
    """Return (names, importances) or (None, reason)."""
    if model is None:
        return None, 'no best model available'
    if hasattr(model, 'feature_importances_'):
        imp = np.asarray(model.feature_importances_, dtype=float).ravel()
        if imp.size == len(feature_names) and np.sum(imp) > 0:
            imp = imp / np.sum(imp)
            return list(feature_names), imp
        return None, 'feature_importances_ shape mismatch'
    if hasattr(model, 'coef_'):
        coef = np.asarray(model.coef_)
        if coef.ndim == 2:
            coef = coef[0]
        if coef.size == len(feature_names):
            return list(feature_names), np.abs(coef).astype(float)
        return None, 'coef_ shape mismatch'
    return None, 'model exposes no feature_importances_ / coef_'


def _plot_roc_pr_calib(preds: pd.DataFrame, out_dir: str,
                       dataset: str, cond: str) -> List[str]:
    plt = _matplotlib()
    generated = []
    y_test = preds['y_test'].values
    models = [c[:-len('_proba')] for c in preds.columns
              if c.endswith('_proba')]
    models = [m for m in FINAL_MODELS if m in models]
    if not models:
        return generated

    # ROC
    fig, ax = plt.subplots(figsize=(6, 6))
    for m, color in zip(models, _PLOT_COLORS):
        fpr, tpr, auc = get_roc_data(pd.Series(y_test),
                                     preds[f'{m}_proba'].values)
        ax.plot(fpr, tpr, color=color, lw=1.6,
                label=f'{MODEL_LABELS.get(m, m)} (AUC {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(f'ROC curves — {dataset} ({cond})')
    ax.legend(fontsize=8, loc='lower right')
    png = os.path.join(out_dir, 'roc_curves.png')
    _savefig(fig, png, png[:-3] + 'pdf')
    plt.close(fig)
    generated.append(png)

    # PR
    fig, ax = plt.subplots(figsize=(6, 6))
    for m, color in zip(models, _PLOT_COLORS):
        prec, rec, ap = get_pr_data(pd.Series(y_test),
                                    preds[f'{m}_proba'].values)
        ax.plot(rec, prec, color=color, lw=1.6,
                label=f'{MODEL_LABELS.get(m, m)} (AP {ap:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-recall curves — {dataset} ({cond})')
    ax.legend(fontsize=8, loc='upper right')
    png = os.path.join(out_dir, 'pr_curves.png')
    _savefig(fig, png, png[:-3] + 'pdf')
    plt.close(fig)
    generated.append(png)

    # Calibration
    fig, ax = plt.subplots(figsize=(6, 6))
    for m, color in zip(models, _PLOT_COLORS):
        p_true, p_pred = get_calibration_data(
            pd.Series(y_test), preds[f'{m}_proba'].values)
        if len(p_true) > 0:
            ax.plot(p_pred, p_true, 'o-', color=color, ms=4, lw=1.4,
                    label=MODEL_LABELS.get(m, m))
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(f'Calibration curves — {dataset} ({cond})')
    ax.legend(fontsize=8, loc='upper left')
    png = os.path.join(out_dir, 'calibration_curves.png')
    _savefig(fig, png, png[:-3] + 'pdf')
    plt.close(fig)
    generated.append(png)
    return generated


def _plot_confusion(preds: pd.DataFrame, out_dir: str,
                    dataset: str, cond: str) -> Optional[str]:
    from sklearn.metrics import confusion_matrix
    plt = _matplotlib()
    y_test = preds['y_test'].values
    models = [m for m in FINAL_MODELS if f'{m}_proba' in preds.columns]
    if not models:
        return None
    n = len(models)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.atleast_2d(axes).reshape(rows, cols)
    for idx, m in enumerate(models):
        ax = axes[idx // cols][idx % cols]
        y_pred = (preds[f'{m}_proba'].values >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title(MODEL_LABELS.get(m, m), fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['neg', 'pos'], fontsize=8)
        ax.set_yticklabels(['neg', 'pos'], fontsize=8)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black',
                        fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for idx in range(len(models), rows * cols):
        axes[idx // cols][idx % cols].axis('off')
    fig.suptitle(f'Confusion matrices (threshold 0.5) — {dataset} ({cond})',
                 fontsize=11)
    fig.tight_layout()
    png = os.path.join(out_dir, 'confusion_matrices.png')
    _savefig(fig, png, png[:-3] + 'pdf')
    plt.close(fig)
    return png


def _plot_feature_importance(experiment_dir: str, dataset: str, cond: str,
                             feats: pd.DataFrame, out_dir: str) -> Optional[str]:
    import joblib
    plt = _matplotlib()
    pkl = _best_model_path(experiment_dir, dataset, cond)
    if not os.path.isfile(pkl):
        _write_skip_reason(out_dir, 'feature_importance',
                           'no persisted best model')
        return None
    try:
        model = joblib.load(pkl)
    except Exception as exc:
        _write_skip_reason(out_dir, 'feature_importance', f'load failed: {exc}')
        return None
    feature_names = list(feats.columns)
    names, imp = _feature_importance(model, feature_names)
    if names is None:
        _write_skip_reason(out_dir, 'feature_importance', imp)
        return None
    order = np.argsort(imp)[-20:]
    names = [names[i] for i in order]
    imp = imp[order]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(names, imp, color='#1f77b4')
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature importance (best model, top 20) — '
                 f'{dataset} ({cond})')
    fig.tight_layout()
    png = os.path.join(out_dir, 'feature_importance.png')
    _savefig(fig, png, png[:-3] + 'pdf')
    plt.close(fig)
    return png


def _plot_shap(experiment_dir: str, dataset: str, cond: str,
               preds: pd.DataFrame, feats: pd.DataFrame,
               out_dir: str) -> Optional[str]:
    pkl = _best_model_path(experiment_dir, dataset, cond)
    if not os.path.isfile(pkl):
        _write_skip_reason(out_dir, 'shap_summary', 'no persisted best model')
        return None
    try:
        import shap
    except ImportError as exc:
        _write_skip_reason(out_dir, 'shap_summary', f'shap not installed: {exc}')
        return None
    try:
        import joblib
        plt = _matplotlib()
        model = joblib.load(pkl)
        sample = feats.sample(n=min(200, len(feats)), random_state=42)
        if hasattr(model, 'get_booster') or hasattr(model, 'estimators_'):
            explainer = shap.TreeExplainer(
                model, feature_perturbation='tree_path_dependent')
        else:
            explainer = shap.Explainer(model, sample)
        sv = explainer(sample)
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.summary_plot(sv, sample, show=False)
        plt.title(f'SHAP summary — {dataset} ({cond})')
        png = os.path.join(out_dir, 'shap_summary.png')
        _savefig(fig, png, png[:-3] + 'pdf')
        plt.close(fig)
        return png
    except Exception as exc:
        _write_skip_reason(out_dir, 'shap_summary', str(exc))
        logger.warning("SHAP skipped for %s/%s: %s", dataset, cond, exc)
        return None


def _write_skip_reason(out_dir: str, kind: str, reason: str) -> None:
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, f'{kind}_skip_reason.txt'), 'w') as f:
        f.write(reason + '\n')


def generate_condition_figures(experiment_dir: str) -> List[str]:
    """Generate supplementary figures for every dataset × SMOTE condition."""
    generated: List[str] = []
    for cond in ['without_smote', 'with_smote']:
        cond_root = os.path.join(experiment_dir, 'results', cond)
        if not os.path.isdir(cond_root):
            continue
        for dataset in sorted(os.listdir(cond_root)):
            preds_feats = _load_condition(experiment_dir, dataset, cond)
            if preds_feats is None:
                continue
            preds, feats = preds_feats
            out_dir = os.path.join(experiment_dir, 'results', 'figures',
                                   'supplementary', dataset, cond)
            ensure_dir(out_dir)
            try:
                generated += _plot_roc_pr_calib(preds, out_dir, dataset, cond)
            except Exception as exc:
                logger.warning("ROC/PR/calib figures failed %s/%s: %s",
                               dataset, cond, exc)
            try:
                cm = _plot_confusion(preds, out_dir, dataset, cond)
                if cm:
                    generated.append(cm)
            except Exception as exc:
                logger.warning("Confusion figures failed %s/%s: %s",
                               dataset, cond, exc)
            try:
                fi = _plot_feature_importance(experiment_dir, dataset, cond,
                                              feats, out_dir)
                if fi:
                    generated.append(fi)
            except Exception as exc:
                logger.warning("Feature-importance figure failed %s/%s: %s",
                               dataset, cond, exc)
            try:
                sh = _plot_shap(experiment_dir, dataset, cond, preds, feats,
                                out_dir)
                if sh:
                    generated.append(sh)
            except Exception as exc:
                logger.warning("SHAP figure failed %s/%s: %s",
                               dataset, cond, exc)
            logger.validation("Supplementary figures | %s/%s", dataset, cond)
    return generated


# ═════════════════════════════════════════════════════════════════════
# MAIN (master-level)
# ═════════════════════════════════════════════════════════════════════

def _non_baseline(all_results: pd.DataFrame) -> pd.DataFrame:
    return all_results[
        ~all_results['model'].astype(str).str.contains('baseline', na=False)
    ]


def _plot_auc_heatmap(all_results: pd.DataFrame, main_dir: str) -> List[str]:
    plt = _matplotlib()
    generated = []
    for cond, title in [('No', 'without SMOTE'), ('Yes', 'with SMOTE')]:
        sub = _non_baseline(all_results[all_results['smote'] == cond])
        pivot = sub.pivot_table(index='dataset', columns='model',
                                values='roc_auc', aggfunc='mean')
        pivot = pivot[FINAL_MODELS] if set(FINAL_MODELS) <= set(pivot.columns) \
            else pivot
        fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(pivot))))
        im = ax.imshow(pivot.values, cmap='viridis', aspect='auto', vmin=0.4,
                       vmax=1.0)
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels([MODEL_LABELS.get(c, c) for c in pivot.columns],
                           rotation=30, ha='right', fontsize=8)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(list(pivot.index), fontsize=9)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                            fontsize=8,
                            color='white' if v < 0.72 else 'black')
        fig.colorbar(im, ax=ax, fraction=0.03)
        ax.set_title(f'Model × dataset ROC-AUC ({title})', fontsize=12)
        png = os.path.join(main_dir, f'auc_heatmap_{cond}.png')
        _savefig(fig, png, png[:-3] + 'pdf')
        plt.close(fig)
        generated.append(png)
    return generated


def _plot_dataset_performance(all_results: pd.DataFrame,
                              main_dir: str) -> str:
    plt = _matplotlib()
    sub = _non_baseline(all_results)
    agg = sub.groupby(['dataset', 'smote'])['roc_auc'].agg(
        ['mean', 'std']).reset_index()
    datasets = sorted(sub['dataset'].unique())
    x = np.arange(len(datasets))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (cond, label, color) in enumerate(
            [('No', 'without SMOTE', '#1f77b4'),
             ('Yes', 'with SMOTE', '#ff7f0e')]):
        d = agg[agg['smote'] == cond]
        vals = [float(d.loc[d['dataset'] == ds, 'mean'].iloc[0])
                if ds in d['dataset'].values else np.nan for ds in datasets]
        errs = [float(d.loc[d['dataset'] == ds, 'std'].iloc[0])
                if ds in d['dataset'].values else np.nan for ds in datasets]
        ax.bar(x + (i - 0.5) * width, vals, width,
               yerr=errs, label=label, color=color,
               capsize=3, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Mean ROC-AUC')
    ax.set_ylim(0.4, 1.0)
    ax.set_title('Dataset performance (non-baseline models)')
    ax.legend(fontsize=9)
    png = os.path.join(main_dir, 'dataset_performance.png')
    _savefig(fig, png, png[:-3] + 'pdf')
    plt.close(fig)
    return png


def _plot_smote_effect(all_results: pd.DataFrame, main_dir: str) -> str:
    """SMOTE effect with negative deltas shown explicitly."""
    plt = _matplotlib()
    without = _non_baseline(all_results[all_results['smote'] == 'No'])
    with_ = _non_baseline(all_results[all_results['smote'] == 'Yes'])
    merged = without[['dataset', 'model', 'roc_auc']].merge(
        with_[['dataset', 'model', 'roc_auc']], on=['dataset', 'model'],
        suffixes=('_no', '_yes'))
    merged['delta'] = merged['roc_auc_yes'] - merged['roc_auc_no']
    merged = merged.dropna()
    if merged.empty:
        logger.warning('No paired SMOTE rows for effect plot')
        return ''
    merged = merged.sort_values('delta')
    colors = ['#d62728' if d < 0 else '#1f77b4' for d in merged['delta']]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(merged))))
    labels = [f"{r['dataset']}\n{MODEL_LABELS.get(r['model'], r['model'])}"
              for _, r in merged.iterrows()]
    ax.barh(range(len(merged)), merged['delta'], color=colors)
    ax.set_yticks(range(len(merged)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('ROC-AUC change when SMOTE applied')
    n_neg = int((merged['delta'] < 0).sum())
    ax.set_title(f'SMOTE effect per dataset × model '
                 f'(negatives = {n_neg} of {len(merged)} shown in red)')
    ax.grid(axis='x')
    png = os.path.join(main_dir, 'smote_effect.png')
    _savefig(fig, png, png[:-3] + 'pdf')
    plt.close(fig)
    return png


def _plot_model_ranking(all_results: pd.DataFrame, main_dir: str) -> str:
    plt = _matplotlib()
    sub = _non_baseline(all_results)
    agg = sub.groupby(['model', 'smote'])['roc_auc'].agg(
        ['mean', 'std']).reset_index()
    models = [m for m in FINAL_MODELS if m in sub['model'].values]
    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (cond, label, color) in enumerate(
            [('No', 'without SMOTE', '#1f77b4'),
             ('Yes', 'with SMOTE', '#ff7f0e')]):
        d = agg[agg['smote'] == cond]
        vals = [float(d.loc[d['model'] == m, 'mean'].iloc[0])
                if m in d['model'].values else np.nan for m in models]
        errs = [float(d.loc[d['model'] == m, 'std'].iloc[0])
                if m in d['model'].values else np.nan for m in models]
        ax.bar(x + (i - 0.5) * width, vals, width,
               yerr=errs, label=label, color=color, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models],
                       rotation=15, ha='right')
    ax.set_ylabel('Mean ROC-AUC')
    ax.set_ylim(0.4, 1.0)
    ax.set_title('Model ranking (mean ROC-AUC across datasets)')
    ax.legend(fontsize=9)
    png = os.path.join(main_dir, 'model_ranking.png')
    _savefig(fig, png, png[:-3] + 'pdf')
    plt.close(fig)
    return png


def _plot_metric_distributions(all_results: pd.DataFrame,
                               main_dir: str) -> str:
    plt = _matplotlib()
    sub = _non_baseline(all_results)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, cond, title in [(axes[0], 'No', 'without SMOTE'),
                            (axes[1], 'Yes', 'with SMOTE')]:
        d = sub[sub['smote'] == cond]
        data = [d.loc[d['model'] == m, 'roc_auc'].dropna().values
                for m in FINAL_MODELS if m in d['model'].values]
        models = [m for m in FINAL_MODELS if m in d['model'].values]
        ax.boxplot(data)
        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
        ax.set_title(f'ROC-AUC distribution ({title})')
        ax.set_ylabel('ROC-AUC')
        ax.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    png = os.path.join(main_dir, 'metric_distributions.png')
    _savefig(fig, png, png[:-3] + 'pdf')
    plt.close(fig)
    return png


def generate_master_figures(experiment_dir: str,
                            all_results: pd.DataFrame) -> List[str]:
    """Generate the master-level figure set into ``results/figures/main/``."""
    if all_results is None or all_results.empty:
        logger.warning('No master results — master figures skipped')
        return []
    main_dir = ensure_dir(os.path.join(experiment_dir, 'results', 'figures',
                                       'main'))
    generated: List[str] = []
    for name, fn in [
        ('heatmaps', _plot_auc_heatmap),
        ('dataset_performance', _plot_dataset_performance),
        ('smote_effect', _plot_smote_effect),
        ('model_ranking', _plot_model_ranking),
        ('metric_distributions', _plot_metric_distributions),
    ]:
        try:
            out = fn(all_results, main_dir)
            generated += out if isinstance(out, list) else ([out] if out else [])
        except Exception as exc:
            logger.warning("Master figure '%s' failed: %s", name, exc)
    logger.validation("Master figures written to %s (%d files)", main_dir,
                      len(generated))
    return generated


def generate_all_figures(experiment_dir: str,
                         all_results: pd.DataFrame) -> Dict[str, List[str]]:
    """Run both supplementary and main figure generation."""
    return {
        'supplementary': generate_condition_figures(experiment_dir),
        'main': generate_master_figures(experiment_dir, all_results),
    }

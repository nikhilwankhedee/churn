"""Tests for the final build phase: KKBox WSDM labeler semantics, validation
harness, statistical comparison, framework quadrant (with independence),
model persistence + reload verification, publication figures, integrity audit,
smoke-test isolation, and the official runner notebook."""
import datetime
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.config import (
    FINAL_EXPERIMENT_DATASETS,
    FINAL_EXPERIMENT_MODELS,
    KKBOX_TRAIN_FILE,
    KKBOX_TRANSACTIONS_FILE,
)
from src.kkbox.labeler import GAP_NO_RENEWAL, WSDMChurnLabeller
from src.kkbox.validation import (
    STATUS_MATCH,
    STATUS_MISMATCH,
    STATUS_MISSING,
    STATUS_UNVALIDATED,
    check_data_availability,
    run_kkbox_validation,
)

FINAL_MODELS = FINAL_EXPERIMENT_MODELS


# ═════════════════════════════════════════════════════════════════════
# KKBox WSDM labeler — reference semantics
# ═════════════════════════════════════════════════════════════════════

def _make_transactions():
    rows = [
        # renewed within window -> 0
        ['A', '20170110', '20170215', '30', '149', '0', 0],
        ['A', '20170218', '20170320', '30', '149', '0', 0],
        # no future activity -> 1
        ['B', '20170112', '20170210', '30', '149', '0', 0],
        ['C', '20170115', '20170220', '30', '199', '0', 0],
        ['C', '20170221', '20170322', '30', '199', '0', 0],
        ['D', '20170118', '20170225', '30', '149', '0', 0],
        ['E', '20170120', '20170228', '30', '149', '0', 0],
        ['F', '20170105', '20170205', '30', '149', '0', 0],
        ['F', '20170226', '20170328', '30', '149', '0', 0],
        ['H', '20170125', '20170215', '30', '149', '0', 0],
        ['J', '20170128', '20170228', '30', '149', '0', 0],
        # expiry outside the prediction month -> no label
        ['G', '20170108', '20170301', '30', '149', '0', 0],
        ['I', '20170114', '20170131', '30', '149', '0', 0],
        ['K', '20170122', '20170128', '30', '149', '0', 0],
        # cancellation moves last_expire earlier, then a renewal fixes the gap
        ['L', '20170110', '20170220', '30', '149', '0', 0],
        ['L', '20170205', '20170210', '30', '149', '0', 1],
        ['L', '20170225', '20170327', '30', '149', '0', 0],
        # cancellation then no renewal -> 1
        ['M', '20170111', '20170220', '30', '149', '0', 0],
        ['M', '20170208', '20170205', '30', '149', '0', 1],
        # same date+signature: cancellation wins (min expire)
        ['P', '20170112', '20170228', '30', '199', '0', 0],
        ['P', '20170112', '20170205', '30', '199', '0', 1],
        # same date+signature: renewals -> max expire wins
        ['Q', '20170112', '20170210', '30', '199', '0', 0],
        ['Q', '20170112', '20170220', '30', '199', '0', 0],
    ]
    return pd.DataFrame(rows, columns=[
        'msno', 'transaction_date', 'membership_expire_date',
        'payment_plan_days', 'plan_list_price', 'payment_method_id',
        'is_cancel'])


class TestWSDMLabelerReference:
    """Faithful reproduction of the WSDM 2018 churn semantics."""

    @pytest.fixture(scope='class')
    def labels(self):
        return WSDMChurnLabeller().compute_churn_labels(_make_transactions())

    def test_reference_suite(self, labels):
        expect = {'A': 0, 'B': 1, 'C': 0, 'D': 1, 'E': 1, 'F': 0,
                  'H': 1, 'J': 1, 'L': 0, 'M': 1, 'P': 1, 'Q': 1}
        got = dict(zip(labels['customer_id'], labels['churn']))
        assert got == expect

    def test_members_outside_prediction_month_unlabeled(self, labels):
        labeled = set(labels['customer_id'])
        assert {'G', 'I', 'K'} & labeled == set()

    def test_columns(self, labels):
        assert list(labels.columns) == ['customer_id', 'churn']

    def test_deterministic(self):
        l1 = WSDMChurnLabeller().compute_churn_labels(_make_transactions())
        l2 = WSDMChurnLabeller().compute_churn_labels(_make_transactions())
        pd.testing.assert_frame_equal(l1.sort_values('customer_id').reset_index(drop=True),
                                      l2.sort_values('customer_id').reset_index(drop=True))

    def test_custom_window(self):
        lab = WSDMChurnLabeller(churn_window_days=0)
        out = lab.compute_churn_labels(_make_transactions())
        got = dict(zip(out['customer_id'], out['churn']))
        assert got['A'] == 1  # window 0 => any gap counts as churn

    def test_renewal_gap_no_renewal_constant(self):
        lab = WSDMChurnLabeller()
        assert lab._renewal_gap(
            pd.DataFrame([{'transaction_date': '20170201',
                           '_date': 20170201, '_sig': 'x', '_cancel': 1,
                           '_expire': 20170205}]),
            20170220) == GAP_NO_RENEWAL


class TestKKBoxValidationHarness:
    """Harness must never fabricate validation results."""

    @pytest.fixture()
    def data_dir(self, tmp_path):
        return tmp_path

    def test_data_missing(self, data_dir):
        r = check_data_availability(str(data_dir))
        assert r['status'] == STATUS_MISSING
        assert run_kkbox_validation(str(data_dir))['status'] == STATUS_MISSING

    def test_unvalidated_without_official(self, data_dir):
        (data_dir / KKBOX_TRANSACTIONS_FILE).write_text(
            'msno,transaction_date,membership_expire_date,payment_plan_days,'
            'plan_list_price,payment_method_id,is_cancel\n'
            'B,20170112,20170210,30,149,0,0\n')
        r = run_kkbox_validation(str(data_dir))
        assert r['status'] == STATUS_UNVALIDATED
        assert r['note'].startswith('No official')

    def test_validated_match(self, data_dir):
        (data_dir / KKBOX_TRANSACTIONS_FILE).write_text(
            'msno,transaction_date,membership_expire_date,payment_plan_days,'
            'plan_list_price,payment_method_id,is_cancel\n'
            'A,20170110,20170215,30,149,0,0\n'
            'A,20170218,20170320,30,149,0,0\n'
            'B,20170112,20170210,30,149,0,0\n'
            'C,20170115,20170220,30,199,0,0\n'
            'C,20170221,20170322,30,199,0,0\n'
            'D,20170118,20170225,30,149,0,0\n'
            'E,20170120,20170228,30,149,0,0\n'
            'F,20170105,20170205,30,149,0,0\n'
            'F,20170226,20170328,30,149,0,0\n'
            'H,20170125,20170215,30,149,0,0\n'
            'J,20170128,20170228,30,149,0,0\n'
            'G,20170108,20170301,30,149,0,0\n'
            'I,20170114,20170131,30,149,0,0\n'
            'K,20170122,20170128,30,149,0,0\n')
        (data_dir / KKBOX_TRAIN_FILE).write_text(
            'msno,is_churn\n'
            'A,0\nB,1\nC,0\nD,1\nE,1\nF,0\nH,1\nJ,1\n')
        r = run_kkbox_validation(str(data_dir))
        assert r['status'] == STATUS_MATCH
        assert r['agreement_rate'] == 1.0
        assert r['coverage'] == 1.0
        assert r['n_matches'] == 8 and r['n_mismatches'] == 0

    def test_validated_mismatch(self, data_dir):
        (data_dir / KKBOX_TRANSACTIONS_FILE).write_text(
            'msno,transaction_date,membership_expire_date,payment_plan_days,'
            'plan_list_price,payment_method_id,is_cancel\n'
            'A,20170110,20170215,30,149,0,0\n'
            'A,20170218,20170320,30,149,0,0\n'
            'B,20170112,20170210,30,149,0,0\n')
        # official flips B -> disagreement
        (data_dir / KKBOX_TRAIN_FILE).write_text(
            'msno,is_churn\nA,0\nB,0\n')
        r = run_kkbox_validation(str(data_dir))
        assert r['status'] == STATUS_MISMATCH
        assert r['agreement_rate'] == 0.5


# ═════════════════════════════════════════════════════════════════════
# Shared synthetic master-results builder
# ═════════════════════════════════════════════════════════════════════

def _synthetic_all_results(datasets=None, n_baseline=2,
                           seed=0):
    """Deterministic all_results frame: datasets × 2 SMOTE × models."""
    datasets = datasets or list(FINAL_EXPERIMENT_DATASETS)
    rng = np.random.RandomState(seed)
    rows = []
    for i, ds in enumerate(datasets):
        for smote in ['No', 'Yes']:
            for model in FINAL_MODELS + (['majority_class', 'random_baseline']
                                         if n_baseline else []):
                base = 0.70 + 0.03 * i + (0.05 if smote == 'Yes' else 0)
                if model == 'random_baseline':
                    auc = 0.5
                elif model == 'majority_class':
                    auc = np.nan
                else:
                    auc = base + 0.02 * FINAL_MODELS.index(model) + \
                        0.02 * rng.randn()
                    auc = min(0.99, max(0.5, auc))
                rows.append({
                    'dataset': ds, 'model': model, 'smote': smote,
                    'accuracy': 0.8, 'precision': 0.7, 'recall': 0.6,
                    'f1': 0.65, 'roc_auc': auc, 'pr_auc': 0.6,
                    'balanced_accuracy': 0.7, 'mcc': 0.4,
                    'brier_score': 0.2, 'calibration_error': 0.1,
                })
    return pd.DataFrame(rows)


def _tiny_trainer(experiment_dir, dataset, cond, n=150, seed=0):
    """Write minimal per-condition artifacts; returns the model dict."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    base = os.path.join(experiment_dir, 'results', cond, dataset)
    os.makedirs(base, exist_ok=True)
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, 6), columns=[f'f{i}' for i in range(6)],
                     index=[f'c{i:05d}' for i in range(n)])
    # nonlinear interaction so tree models (RF/XGB/LGBM) win and expose
    # feature_importances_
    y = (X['f0'] * X['f1'] + 0.4 * X['f2'] + 0.5 * rng.randn(n) > 0).astype(int)

    X.to_csv(os.path.join(base, 'test_features.csv'))
    pd.DataFrame({'churn': y}, index=X.index).to_csv(
        os.path.join(base, 'test_labels.csv'))

    proba_rows = {'customer_id': list(X.index), 'y_test': y}
    metrics_rows = []
    models = {}
    for m in FINAL_MODELS:
        if m == 'random_forest':
            model = RandomForestClassifier(n_estimators=20, random_state=seed)
        elif m == 'svm':
            from sklearn.svm import SVC
            model = SVC(probability=True, random_state=seed)
        elif m == 'xgboost':
            from xgboost import XGBClassifier
            model = XGBClassifier(n_estimators=20, random_state=seed,
                                  verbosity=0)
        elif m == 'lightgbm':
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(n_estimators=20, random_state=seed,
                                   verbose=-1)
        else:
            model = LogisticRegression(max_iter=500, random_state=seed)
        model.fit(X, y)
        proba = model.predict_proba(X)[:, 1]
        models[m] = model
        proba_rows[f'{m}_proba'] = proba
        metrics_rows.append({
            'model': m, 'accuracy': 0.8, 'precision': 0.7, 'recall': 0.6,
            'f1': 0.65, 'roc_auc': float(roc_auc_score(y, proba)),
            'avg_precision': 0.6, 'balanced_accuracy': 0.7, 'mcc': 0.4,
            'brier_score': 0.2, 'calibration_error': 0.1,
            'training_time': 0.1, 'inference_time': 0.01, 'tn': 50, 'fp': 20,
            'fn': 20, 'tp': 60,
        })
        joblib.dump(model, os.path.join(base, f'{m}.joblib'))
    # baselines (documented, with NaN roc_auc for majority)
    metrics_rows.append({'model': 'majority_class', 'roc_auc': np.nan})
    metrics_rows.append({'model': 'random_baseline', 'roc_auc': 0.5})
    pd.DataFrame(proba_rows).to_csv(os.path.join(base, 'predictions.csv'),
                                    index=False)
    pd.DataFrame(metrics_rows).to_csv(os.path.join(base, 'metrics.csv'),
                                      index=False)
    with open(os.path.join(base, 'experiment_metadata.json'), 'w') as f:
        json.dump({'test_ids_hash': 'x', 'test_y_hash': 'y'}, f)
    return models


# ═════════════════════════════════════════════════════════════════════
# Statistical comparison
# ═════════════════════════════════════════════════════════════════════

class TestStatisticalComparison:
    @pytest.fixture()
    def all_results(self):
        return _synthetic_all_results()

    def test_build_matrix(self, all_results):
        from src.statistical_comparison import build_roc_auc_matrix
        m = build_roc_auc_matrix(all_results, smote='No')
        assert list(m.columns) == FINAL_MODELS
        assert len(m) == len(FINAL_EXPERIMENT_DATASETS)
        assert m.notna().all().all()

    def test_friedman(self, all_results):
        from src.statistical_comparison import build_roc_auc_matrix, friedman_test
        m = build_roc_auc_matrix(all_results)
        f = friedman_test(m)
        assert f['n_models'] == 5 and f['n_datasets'] == len(FINAL_EXPERIMENT_DATASETS)
        assert 0.0 <= f['p_value'] <= 1.0

    def test_nemenyi_cd_positive(self, all_results):
        from src.statistical_comparison import build_roc_auc_matrix, nemenyi_test
        m = build_roc_auc_matrix(all_results)
        pvals, cd = nemenyi_test(m)
        assert cd > 0
        assert pvals.shape == (5, 5)

    def test_wilcoxon_smote(self, all_results):
        from src.statistical_comparison import wilcoxon_smote
        w = wilcoxon_smote(all_results)
        assert len(w) == 5
        assert w['raw_p_value'].notna().all()
        assert 'q_value_bh' in w.columns

    def test_outputs_written(self, all_results, tmp_path):
        from src.statistical_comparison import run_statistical_comparison
        out = run_statistical_comparison(str(tmp_path), all_results)
        assert 'pooled' in out['friedman']
        for f in ['statistical_comparison.csv', 'statistical_comparison.xlsx',
                  'statistical_comparison.tex',
                  'critical_difference_diagram.png',
                  'critical_difference_diagram.pdf']:
            assert os.path.isfile(os.path.join(str(tmp_path), 'results',
                                               'master', f)), f

    def test_empty_results_no_crash(self, tmp_path):
        from src.statistical_comparison import run_statistical_comparison
        out = run_statistical_comparison(str(tmp_path), pd.DataFrame())
        assert out == {} or not out.get('friedman')


# ═════════════════════════════════════════════════════════════════════
# Framework quadrant
# ═════════════════════════════════════════════════════════════════════

class TestFrameworkQuadrant:
    def _chars(self):
        return pd.DataFrame({
            'dataset': ['a', 'b', 'c', 'd'],
            'repeat_customer_ratio': [0.1, 0.9, 0.2, 0.8],
            'avg_events_per_customer': [2, 100, 3, 80],
            'time_span_days': [10, 500, 12, 400],
            'n_event_types': [1, 5, 6, 2],
            'n_numerical_features': [2, 20, 15, 3],
            'n_categorical_features': [0, 3, 4, 1],
            'missing_value_pct': [0.4, 0.05, 0.1, 0.3],
        })

    def test_score_ranges(self):
        from src.framework_analysis import score_datasets
        pos = score_datasets(self._chars())
        assert pos['continuity_score'].between(0, 1).all()
        assert pos['observability_score'].between(0, 1).all()
        assert set(pos['quadrant']) <= {
            'Repeated & Observable', 'Repeated & Opaque',
            'Sporadic & Observable', 'Sporadic & Opaque'}

    def test_quadrant_independent_of_performance(self):
        """Adding/perturbing avg_* performance columns must not change scores."""
        from src.framework_analysis import score_datasets
        base = score_datasets(self._chars())
        perf = self._chars().copy()
        # arbitrary, wildly different performance values
        perf['avg_roc_auc'] = [0.5, 0.99, 0.2, 0.7]
        perf['avg_f1'] = [0.1, 0.9, 0.3, 0.6]
        perf['avg_pr_auc'] = [0.05, 0.98, 0.01, 0.4]
        with_perf = score_datasets(perf)
        pd.testing.assert_frame_equal(
            base.sort_values('dataset').reset_index(drop=True),
            with_perf.sort_values('dataset').reset_index(drop=True))

    def test_quadrant_thresholds(self):
        from src.framework_analysis import score_datasets
        df = pd.DataFrame({
            'dataset': ['hh', 'hl', 'lh', 'll'],
            'repeat_customer_ratio': [0.9, 0.9, 0.1, 0.1],
            'avg_events_per_customer': [100, 100, 2, 2],
            'time_span_days': [500, 500, 10, 10],
            'n_event_types': [5, 1, 5, 1],
            'n_numerical_features': [20, 2, 20, 2],
            'n_categorical_features': [3, 0, 3, 0],
            'missing_value_pct': [0.05, 0.4, 0.05, 0.4],
        })
        pos = score_datasets(df)
        q = dict(zip(pos['dataset'], pos['quadrant']))
        assert q['hh'] == 'Repeated & Observable'
        assert q['hl'] == 'Repeated & Opaque'
        assert q['lh'] == 'Sporadic & Observable'
        assert q['ll'] == 'Sporadic & Opaque'

    def test_outputs_written(self, tmp_path):
        from src.framework_analysis import run_framework_analysis
        chars = self._chars()
        os.makedirs(os.path.join(str(tmp_path), 'results', 'master'),
                    exist_ok=True)
        chars.to_csv(os.path.join(str(tmp_path), 'results', 'master',
                                  'dataset_characteristics.csv'), index=False)
        out = run_framework_analysis(str(tmp_path), _synthetic_all_results(
            datasets=['a', 'b', 'c', 'd']))
        assert out
        fw = os.path.join(str(tmp_path), 'results', 'framework')
        for f in ['framework_methodology.json', 'dataset_positions.csv',
                  'quadrant_performance.csv', 'framework_quadrant_plot.png',
                  'framework_quadrant_plot.pdf']:
            assert os.path.isfile(os.path.join(fw, f)), f
        with open(os.path.join(fw, 'framework_methodology.json')) as fh:
            json.load(fh)  # must be JSON-serializable


# ═════════════════════════════════════════════════════════════════════
# Model persistence
# ═════════════════════════════════════════════════════════════════════

class TestModelPersistence:
    @pytest.fixture()
    def exp_dir(self, tmp_path):
        d = str(tmp_path / 'exp')
        _tiny_trainer(d, 'ds1', 'without_smote', seed=1)
        return d

    @pytest.fixture()
    def all_results(self):
        # a single dataset, both conditions, but only ds1/without persisted
        return _synthetic_all_results(datasets=['ds1'])

    def test_persist_and_verify_pass(self, exp_dir, all_results):
        from src.model_persistence import persist_best_models, verify_best_models
        persist_best_models(exp_dir, all_results)
        ver = verify_best_models(exp_dir)
        assert not ver.empty
        assert (ver['status'] == 'PASS').all()
        assert ver['proba_match'].all()
        np.testing.assert_allclose(ver['expected_roc_auc'],
                                   ver['recomputed_roc_auc'], atol=1e-6)

    def test_metadata_written(self, exp_dir, all_results):
        from src.model_persistence import persist_best_models
        persist_best_models(exp_dir, all_results)
        meta = json.load(open(os.path.join(
            exp_dir, 'results', 'master', 'best_models', 'ds1',
            'without_smote', 'best_model_metadata.json')))
        assert meta['dataset'] == 'ds1'
        assert meta['model'] in FINAL_MODELS
        assert meta['selection_criterion'] == 'roc_auc'

    def test_fail_when_predictions_perturbed(self, exp_dir, all_results):

        from src.model_persistence import persist_best_models, verify_best_models
        persist_best_models(exp_dir, all_results)
        preds = pd.read_csv(os.path.join(exp_dir, 'results', 'without_smote',
                                         'ds1', 'predictions.csv'))
        best = json.load(open(os.path.join(
            exp_dir, 'results', 'master', 'best_models', 'ds1',
            'without_smote', 'best_model_metadata.json')))['model']
        col = f'{best}_proba'
        preds.loc[0, col] = preds.loc[0, col] + 0.05
        preds.to_csv(os.path.join(exp_dir, 'results', 'without_smote',
                                  'ds1', 'predictions.csv'), index=False)
        ver = verify_best_models(exp_dir)
        assert (ver['status'] == 'FAIL').any()


# ═════════════════════════════════════════════════════════════════════
# Publication figures
# ═════════════════════════════════════════════════════════════════════

class TestPublicationFigures:
    @pytest.fixture()
    def exp_dir(self, tmp_path):
        d = str(tmp_path / 'exp')
        _tiny_trainer(d, 'ds1', 'without_smote', seed=2)
        _tiny_trainer(d, 'ds1', 'with_smote', seed=3)
        return d

    def test_master_figures(self, exp_dir):
        from src.publication_figures import generate_master_figures
        figs = generate_master_figures(exp_dir, _synthetic_all_results(
            datasets=['ds1']))
        assert any(os.path.basename(f) == 'auc_heatmap_No.png' for f in figs)
        assert any(os.path.basename(f) == 'smote_effect.png' for f in figs)
        assert any(os.path.basename(f) == 'model_ranking.png' for f in figs)

    def test_supplementary_figures(self, exp_dir):
        from src.model_persistence import persist_best_models
        from src.publication_figures import generate_condition_figures
        persist_best_models(exp_dir, _synthetic_all_results(datasets=['ds1']))
        figs = generate_condition_figures(exp_dir)
        out_dir = os.path.join(exp_dir, 'results', 'figures',
                               'supplementary', 'ds1', 'without_smote')
        for f in ['roc_curves.png', 'pr_curves.png', 'calibration_curves.png',
                  'confusion_matrices.png', 'feature_importance.png']:
            assert os.path.isfile(os.path.join(out_dir, f)), f
        assert figs

    def test_shap_skipped_gracefully(self, exp_dir):
        import pandas as pd

        from src.publication_figures import _plot_shap
        preds = pd.read_csv(os.path.join(exp_dir, 'results', 'without_smote',
                                         'ds1', 'predictions.csv'))
        feats = pd.read_csv(os.path.join(exp_dir, 'results', 'without_smote',
                                         'ds1', 'test_features.csv'), index_col=0)
        out_dir = os.path.join(exp_dir, 'results', 'figures', 'supplementary',
                               'ds1', 'without_smote')
        os.makedirs(out_dir, exist_ok=True)
        png = _plot_shap(exp_dir, 'ds1', 'without_smote', preds, feats,
                         out_dir)
        # either a SHAP summary OR an explicit skip reason must exist
        has_png = png is not None and os.path.isfile(png)
        has_reason = os.path.isfile(os.path.join(out_dir,
                                                 'shap_summary_skip_reason.txt'))
        assert has_png or has_reason

    def test_smote_effect_shows_negatives(self):
        import tempfile

        from src.publication_figures import _plot_smote_effect
        rng = np.random.RandomState(0)
        rows = []
        for ds in ['a', 'b']:
            for m in FINAL_MODELS:
                no = 0.6 + 0.3 * rng.rand()
                yes = no + (0.1 if ds == 'a' else -0.1)  # ds b = negative effect
                rows.append({'dataset': ds, 'model': m, 'smote': 'No',
                             'roc_auc': no})
                rows.append({'dataset': ds, 'model': m, 'smote': 'Yes',
                             'roc_auc': yes})
        df = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as td:
            png = _plot_smote_effect(df, td)
            assert png and os.path.isfile(png)


# ═════════════════════════════════════════════════════════════════════
# Integrity audit + completion report
# ═════════════════════════════════════════════════════════════════════

class TestFinalAudit:
    def _exp(self, tmp_path):
        return str(tmp_path)

    def test_expected_count_80(self):
        from src.final_audit import audit_results
        df = _synthetic_all_results()
        a = audit_results('/tmp/x', df, kkbox_status='PENDING — no data')
        row = a[a['check'] == 'expected_experiment_count'].iloc[0]
        assert row['status'] == 'OK'
        assert '80 expected' in row['detail']

    def test_missing_model_cell_fails(self):
        from src.final_audit import audit_results
        df = _synthetic_all_results()
        df = df[~((df['dataset'] == 'olist') & (df['smote'] == 'No') &
                  (df['model'] == 'lightgbm'))]
        a = audit_results('/tmp/x', df, kkbox_status='PENDING')
        row = a[(a['check'] == 'cell_coverage') & (a['dataset'] == 'olist') &
                (a['smote'] == 'No')].iloc[0]
        assert row['status'] == 'FAIL'
        assert 'lightgbm' in str(row['missing_models'])

    def test_nan_in_metrics_fails(self):
        from src.final_audit import audit_results
        df = _synthetic_all_results()
        idx = (df['model'] == 'logistic_regression').idxmax()
        df.loc[idx, 'roc_auc'] = np.nan
        a = audit_results('/tmp/x', df, kkbox_status='PENDING')
        row = a[a['check'] == 'no_nan_inf'].iloc[0]
        assert row['status'] == 'FAIL'

    def test_kkbox_pending_not_success(self):
        from src.final_audit import audit_results
        df = _synthetic_all_results()
        a = audit_results('/tmp/x', df, kkbox_status='PENDING — no data')
        row = a[a['check'] == 'kkbox_status'].iloc[0]
        assert row['status'] == 'OK'
        assert row['kkbox_status'] == 'PENDING — no data'

    def test_identity_fail(self):
        from src.final_audit import audit_results
        df = _synthetic_all_results()
        ids = [{'dataset': 'olist', 'valid': False, 'note': 'FAIL',
                'test_ids_match': False, 'test_y_match': False}]
        a = audit_results('/tmp/x', df, identity_results=ids,
                          kkbox_status='PENDING')
        row = a[a['check'] == 'test_identity'].iloc[0]
        assert row['status'] == 'FAIL'

    def test_completion_report_written(self, tmp_path):
        from src.final_audit import audit_results, write_completion_report
        df = _synthetic_all_results()
        exp = self._exp(tmp_path)
        a = audit_results(exp, df, kkbox_status='PENDING')
        paths = write_completion_report(exp, a, df,
                                        extras={'successful': 16,
                                                'failed': 0})
        assert os.path.isfile(paths['audit'])
        assert os.path.isfile(paths['csv'])
        assert os.path.isfile(paths['json'])
        assert os.path.isfile(paths['txt'])
        rep = json.load(open(paths['json']))
        assert rep['status'] in ('OK', 'WARN', 'FAIL')

    def test_dataset_validity_zero_valid_fails(self):
        """A dataset whose runs all failed must be a hard FAIL, never silently
        omitted from the audit."""
        from src.final_audit import audit_results
        df = _synthetic_all_results()
        df['status'] = 'success'
        df.loc[df['dataset'] == 'olist', 'status'] = 'failed'
        df.loc[df['dataset'] == 'olist', 'roc_auc'] = np.nan
        a = audit_results('/tmp/x', df, kkbox_status='PENDING')
        row = a[(a['check'] == 'dataset_validity') &
                (a['dataset'] == 'olist')].iloc[0]
        assert row['status'] == 'FAIL'
        assert row['valid_models'] == 0
        ok_row = a[(a['check'] == 'dataset_validity') &
                   (a['dataset'] == 'instacart')].iloc[0]
        assert ok_row['status'] == 'OK'

    def test_dataset_validity_check_absent_without_status_col(self):
        """Frames without a status column (legacy callers) skip the new
        per-dataset validity check instead of crashing."""
        from src.final_audit import audit_results
        df = _synthetic_all_results()
        a = audit_results('/tmp/x', df, kkbox_status='PENDING')
        assert not (a['check'] == 'dataset_validity').any()


# ═════════════════════════════════════════════════════════════════════
# PUBLICATION TABLES — all-NaN ROC-AUC must never crash idxmax
# ═════════════════════════════════════════════════════════════════════

class TestPublicationTablesNaN:
    def test_all_nan_roc_auc_no_crash(self, tmp_path):
        """The regression that produced KeyError: nan: every ROC-AUC value
        for a dataset is NaN.  Best-model selection must skip the dataset
        instead of calling .idxmax() on an all-NaN column."""
        import pandas as pd

        from src.experiment_runner import generate_publication_tables
        exp = str(tmp_path / 'exp')
        os.makedirs(os.path.join(exp, 'publication_tables'), exist_ok=True)
        df = _synthetic_all_results(datasets=['ds1'])
        df['roc_auc'] = np.nan
        exported = generate_publication_tables(df, exp)
        assert isinstance(exported, list)
        try:
            best = pd.read_csv(os.path.join(
                exp, 'publication_tables', 'best_model_per_dataset.csv'))
        except pd.errors.EmptyDataError:
            best = pd.DataFrame()
        assert best.empty or 'ds1' not in set(best['Dataset'])

    def test_mixed_valid_and_nan_selects_valid(self, tmp_path):
        """Datasets with a mix of NaN and valid ROC-AUC still pick the best
        valid row and no longer raise on the NaN ones."""
        import pandas as pd

        from src.experiment_runner import generate_publication_tables
        exp = str(tmp_path / 'exp')
        os.makedirs(os.path.join(exp, 'publication_tables'), exist_ok=True)
        df = _synthetic_all_results(datasets=['ds1', 'ds2'])
        df.loc[df['dataset'] == 'ds1', 'roc_auc'] = np.nan
        generate_publication_tables(df, exp)
        best = pd.read_csv(os.path.join(
            exp, 'publication_tables', 'best_model_per_dataset.csv'))
        assert set(best['Dataset']) == {'ds2'}

    def test_dataset_summary_all_nan_roc_auc(self, tmp_path):
        """generate_dataset_summary must also survive an all-NaN roc_auc
        column (previously called .idxmax() directly)."""
        import pandas as pd

        from src.experiment_runner import generate_dataset_summary
        df = _synthetic_all_results(datasets=['ds1'])
        df['roc_auc'] = np.nan
        os.makedirs(os.path.join(str(tmp_path), 'results', 'master'),
                    exist_ok=True)
        summary = generate_dataset_summary(df, str(tmp_path))
        assert summary.empty or 'best_roc_auc' not in summary.columns


# ═════════════════════════════════════════════════════════════════════
# Post-processing wiring + smoke isolation + KKBox status
# ═════════════════════════════════════════════════════════════════════

class TestValidateDatasetsChurn:
    """validate_datasets must reject temporal datasets whose churn
    construction yields (near-)single-class labels."""

    class _FakeAdapter:
        uses_native_churn_label = False
        has_temporal_data = True
        churn_window_days = 60

        def load_raw_data(self):
            return pd.DataFrame({
                'customer_id': [f'c{i}' for i in range(10)],
                'event_time': pd.to_datetime(['2021-01-01'] * 10),
            })

        def preprocess(self, df):
            return df

        def standardize_schema(self, df):
            return df

    def _patch(self, monkeypatch):
        import src.datasets as ds_mod
        fake = self._FakeAdapter()
        monkeypatch.setattr(
            ds_mod, 'get_dataset', lambda name, data_dir=None: fake,
        )
        monkeypatch.setattr(ds_mod, 'list_datasets', lambda: ['fake_ds'])

    def test_degenerate_churn_marked_invalid(self, monkeypatch):
        import src.pipeline as pl_mod

        from src.experiment_runner import validate_datasets
        self._patch(monkeypatch)
        # Force a single-class (all-churned) label construction, which is
        # what a broken synthetic timeline used to produce.
        monkeypatch.setattr(
            pl_mod, 'get_train_test_cutoffs',
            lambda df, q, window: (
                pd.Timestamp('2020-09-01'), pd.Timestamp('2021-01-01'),
            ),
        )
        monkeypatch.setattr(
            pl_mod, 'create_churn_labels',
            lambda df, cutoff, prediction_window_days: pd.DataFrame({
                'customer_id': [f'c{i}' for i in range(10)],
                'churn': [1] * 10,
            }),
        )
        report = validate_datasets(['fake_ds'], {'fake_ds': '/tmp/none'})
        row = report.iloc[0]
        assert bool(row['class_distribution_ok']) is False
        assert bool(row['valid']) is False
        assert any('single-class' in e for e in row['errors'])

    def test_native_non_temporal_still_valid(self, monkeypatch):
        import src.datasets as ds_mod

        from src.experiment_runner import validate_datasets

        class _FakeNative:
            uses_native_churn_label = True
            has_temporal_data = False
            churn_window_days = None

            def load_raw_data(self):
                return pd.DataFrame({
                    'customer_id': [f'c{i}' for i in range(10)],
                    'event_time': pd.to_datetime(['2021-01-01'] * 10),
                })

            def preprocess(self, df):
                return df

            def standardize_schema(self, df):
                return df

            def get_native_churn_labels(self, df, cutoff):
                return pd.DataFrame({
                    'customer_id': df['customer_id'],
                    'churn': [1, 0] * 5,
                })

        monkeypatch.setattr(
            ds_mod, 'get_dataset',
            lambda name, data_dir=None: _FakeNative(),
        )
        monkeypatch.setattr(
            ds_mod, 'list_datasets', lambda: ['fake_native'],
        )
        report = validate_datasets(['fake_native'], {'fake_native': '/tmp/none'})
        row = report.iloc[0]
        assert bool(row['class_distribution_ok']) is True
        assert bool(row['valid']) is True


class TestWiringAndSmoke:
    def test_smoke_output_isolated(self):
        from src.experiment_runner import _resolve_output_dir
        now = datetime.datetime(2026, 1, 1, 0, 0, 0)
        assert _resolve_output_dir('/base', True, now=now) == \
            '/base/smoke_test_20260101_000000'
        assert _resolve_output_dir('/base', False) == '/base'

    def test_smoke_env_flag(self, monkeypatch):
        monkeypatch.setenv('SMOKE_TEST', 'true')
        import os as _os
        assert _os.environ.get('SMOKE_TEST', '').lower() in ('1', 'true', 'yes')

    def test_compute_kkbox_status_pending_no_data(self):
        from src.experiment_runner import compute_kkbox_status
        assert compute_kkbox_status({}) == 'PENDING — KKBox data not present'

    def test_compute_kkbox_status_transactions_only(self, tmp_path):
        from src.experiment_runner import compute_kkbox_status
        (tmp_path / KKBOX_TRANSACTIONS_FILE).write_text(
            'msno,transaction_date,membership_expire_date,payment_plan_days,'
            'plan_list_price,payment_method_id,is_cancel\n'
            'B,20170112,20170210,30,149,0,0\n')
        status = compute_kkbox_status({'kkbox': str(tmp_path)})
        assert status.startswith('PENDING — official labels')

    def test_post_processing_end_to_end(self, tmp_path):
        """Full post-processing over a 2-dataset synthetic tree."""
        from src.experiment_runner import run_post_processing
        exp = str(tmp_path / 'exp')
        for ds in ['olist', 'telco']:
            for cond in ['without_smote', 'with_smote']:
                _tiny_trainer(exp, ds, cond)
        df = _synthetic_all_results(datasets=['olist', 'telco'])
        os.makedirs(os.path.join(exp, 'results', 'master'), exist_ok=True)
        chars = pd.DataFrame({
            'dataset': ['olist', 'telco'],
            'repeat_customer_ratio': [0.7, 0.3],
            'avg_events_per_customer': [20, 5],
            'time_span_days': [300, 60],
            'n_event_types': [4, 2],
            'n_numerical_features': [10, 6],
            'n_categorical_features': [3, 1],
            'missing_value_pct': [0.05, 0.2],
        })
        chars.to_csv(os.path.join(exp, 'results', 'master',
                                  'dataset_characteristics.csv'), index=False)
        ids = [{'dataset': ds, 'valid': True, 'note': 'PASS',
                'test_ids_match': True, 'test_y_match': True}
               for ds in ['olist', 'telco']]
        stages = run_post_processing(exp, df, ids, ['olist', 'telco'], {})
        assert stages['statistical_comparison'].startswith('DONE')
        assert stages['framework_quadrant'].startswith('DONE')
        assert stages['model_persistence'] in ('PASS', 'SKIPPED')
        assert stages['integrity_audit'] == 'OK'
        assert stages['completion_report'] == 'DONE'
        assert stages['kkbox_status'] is None  # no 'kkbox' dir supplied

    def test_post_processing_kkbox_status_included_when_supplied(self, tmp_path):
        """KKBox audit status is only computed when a 'kkbox' dir is given."""
        from src.experiment_runner import run_post_processing
        exp = str(tmp_path / 'exp')
        for ds in ['olist', 'telco']:
            for cond in ['without_smote', 'with_smote']:
                _tiny_trainer(exp, ds, cond)
        df = _synthetic_all_results(datasets=['olist', 'telco'])
        os.makedirs(os.path.join(exp, 'results', 'master'), exist_ok=True)
        ids = [{'dataset': ds, 'valid': True, 'note': 'PASS',
                'test_ids_match': True, 'test_y_match': True}
               for ds in ['olist', 'telco']]
        stages = run_post_processing(exp, df, ids, ['olist', 'telco'],
                                     {'kkbox': str(tmp_path)})
        assert stages['kkbox_status'] == 'PENDING — KKBox transactions not present'


# ═════════════════════════════════════════════════════════════════════
# Official runner notebook
# ═════════════════════════════════════════════════════════════════════

class TestRunnerNotebook:
    NB_PATH = os.path.join(os.path.dirname(__file__), '..', 'notebooks',
                           'behavioral_churn_analysis.ipynb')

    def test_notebook_exists_and_valid(self):
        import nbformat
        with open(self.NB_PATH) as f:
            nb = nbformat.read(f, as_version=4)
        nbformat.validate(nb)
        assert len(nb.cells) > 20

    def test_sixteen_sections(self):
        import nbformat
        with open(self.NB_PATH) as f:
            nb = nbformat.read(f, as_version=4)
        headers = [c.source for c in nb.cells if c.cell_type == 'markdown']
        text = '\n'.join(headers)
        for i in range(1, 17):
            assert f'## {i}.' in text, f'missing section {i}'

    def test_no_kkbox_references(self):
        import nbformat
        with open(self.NB_PATH) as f:
            nb = nbformat.read(f, as_version=4)
        text = '\n'.join(c.source for c in nb.cells).lower()
        assert 'kkbox' not in text, \
            'KKBox must be fully removed from the runner notebook'

    def test_integrity_markers(self):
        import nbformat
        with open(self.NB_PATH) as f:
            nb = nbformat.read(f, as_version=4)
        text = '\n'.join(c.source for c in nb.cells)
        assert 'DATASETS = None' in text
        assert 'SMOKE_TEST = True' in text
        assert 'run_all_experiments' in text
        assert 'DATA_DIRS' in text
        assert 'RETAILROCKET_EVENTS' in text
        assert 'TELCO_FILE' in text
        assert 'CREDIT_CARD_FILE' in text
        assert '/kaggle/input/datasets/' in text

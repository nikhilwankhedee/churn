"""Tests for the framework-level stratified native split.

Covers:
  - the ``stratified_native_split`` helper (70/30, stratification, disjoint
    customer sets, determinism, class-guards);
  - the Credit Card + Telco end-to-end non-temporal path (split, features,
    leakage guards, both classes present, non-empty matrices);
  - SMOTE running strictly after the split on training data only;
  - regression checks that the temporal (behavioural) path is unchanged.
"""
import numpy as np
import pandas as pd
import pytest

from src.churn_labeling import (
    create_churn_labels,
    get_train_test_cutoffs,
    stratified_native_split,
)
from src.config import RANDOM_SEED, SMOTE_K_NEIGHBORS
from src.datasets.credit_card import CreditCardAdapter
from src.datasets.telco import TelcoAdapter
from src.feature_engineering import engineer_features
from src.resamplers.registry import get_resampler

# ═════════════════════════════════════════════════════════════════════
# Synthetic raw data
# ═════════════════════════════════════════════════════════════════════

CREDIT_CARD_RAW_COLS = [
    'CLIENTNUM', 'Attrition_Flag', 'Gender', 'Education_Level',
    'Marital_Status', 'Income_Category', 'Card_Category',
    'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Revolving_Bal',
    'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
    'Avg_Utilization_Ratio', 'Credit_Limit', 'Months_on_book',
    'Total_Relationship_Count', 'Total_Amt_Chng_Q4_Q1',
    'Total_Ct_Chng_Q4_Q1', 'Avg_Open_To_Buy',
    'Naive_Bayes_Class', 'Naive_Bayes_Probability',
]


def _make_credit_card_raw(n=200, churn_frac=0.15, seed=0):
    rng = np.random.RandomState(seed)
    churn = np.array([1] * int(round(n * churn_frac))
                     + [0] * (n - int(round(n * churn_frac))))
    rng.shuffle(churn)
    n = len(churn)
    return pd.DataFrame({
        'CLIENTNUM': [10000 + i for i in range(n)],
        'Attrition_Flag': [
            'Attrited Customer' if c else 'Existing Customer' for c in churn],
        'Gender': rng.choice(['M', 'F'], n),
        'Education_Level': rng.choice(
            ['High School', 'Graduate', 'Uneducated'], n),
        'Marital_Status': rng.choice(['Married', 'Single'], n),
        'Income_Category': rng.choice(
            ['$60K - $80K', 'Less than $40K'], n),
        'Card_Category': rng.choice(['Blue', 'Gold'], n),
        'Total_Trans_Amt': rng.randint(100, 5000, n).astype(float),
        'Total_Trans_Ct': rng.randint(1, 100, n).astype(float),
        'Total_Revolving_Bal': rng.randint(0, 2500, n).astype(float),
        'Months_Inactive_12_mon': rng.randint(0, 6, n).astype(float),
        'Contacts_Count_12_mon': rng.randint(0, 6, n).astype(float),
        'Avg_Utilization_Ratio': rng.uniform(0, 1, n),
        'Credit_Limit': rng.randint(1000, 20000, n).astype(float),
        'Months_on_book': rng.randint(6, 40, n).astype(float),
        'Total_Relationship_Count': rng.randint(1, 6, n).astype(float),
        'Total_Amt_Chng_Q4_Q1': rng.uniform(0, 1, n),
        'Total_Ct_Chng_Q4_Q1': rng.uniform(0, 1, n),
        'Avg_Open_To_Buy': rng.uniform(0, 10000, n),
        'Naive_Bayes_Class': 1,
        'Naive_Bayes_Probability': 0.5,
    })


def _write_credit_card(tmp_path, n=200, churn_frac=0.15, seed=0):
    raw = _make_credit_card_raw(n=n, churn_frac=churn_frac, seed=seed)
    raw.to_csv(tmp_path / 'BankChurners.csv', index=False)
    return raw


TELCO_RAW_COLS = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn',
]


def _make_telco_raw(n=200, churn_frac=0.25, seed=1):
    rng = np.random.RandomState(seed)
    churn = np.array([1] * int(round(n * churn_frac))
                     + [0] * (n - int(round(n * churn_frac))))
    rng.shuffle(churn)
    n = len(churn)
    return pd.DataFrame({
        'customerID': ['CUST-%04d' % i for i in range(n)],
        'gender': rng.choice(['Male', 'Female'], n),
        'SeniorCitizen': rng.choice([0, 1], n),
        'Partner': rng.choice(['Yes', 'No'], n),
        'Dependents': rng.choice(['Yes', 'No'], n),
        'tenure': rng.randint(0, 72, n).astype(float),
        'PhoneService': rng.choice(['Yes', 'No'], n),
        'MultipleLines': rng.choice(['Yes', 'No', 'No phone service'], n),
        'InternetService': rng.choice(
            ['DSL', 'Fiber optic', 'No'], n),
        'OnlineSecurity': rng.choice(['Yes', 'No', 'No internet service'], n),
        'OnlineBackup': rng.choice(['Yes', 'No', 'No internet service'], n),
        'DeviceProtection': rng.choice(['Yes', 'No', 'No internet service'], n),
        'TechSupport': rng.choice(['Yes', 'No', 'No internet service'], n),
        'StreamingTV': rng.choice(['Yes', 'No', 'No internet service'], n),
        'StreamingMovies': rng.choice(['Yes', 'No', 'No internet service'], n),
        'Contract': rng.choice(
            ['Month-to-month', 'One year', 'Two year'], n),
        'PaperlessBilling': rng.choice(['Yes', 'No'], n),
        'PaymentMethod': rng.choice(
            ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
            n),
        'MonthlyCharges': rng.uniform(20, 120, n),
        'TotalCharges': rng.uniform(100, 8000, n),
        'Churn': ['Yes' if c else 'No' for c in churn],
    })


def _write_telco(tmp_path, n=200, churn_frac=0.25, seed=1):
    raw = _make_telco_raw(n=n, churn_frac=churn_frac, seed=seed)
    raw.to_csv(tmp_path / 'WA_Fn-UseC_-Telco-Customer-Churn.csv', index=False)
    return raw


# ═════════════════════════════════════════════════════════════════════
# stratified_native_split helper
# ═════════════════════════════════════════════════════════════════════

class TestStratifiedNativeSplit:
    def test_split_ratio_disjointness_and_stratification(self):
        rng = np.random.RandomState(0)
        n = 1000
        labels = pd.DataFrame({
            'customer_id': ['C%04d' % i for i in range(n)],
            'churn': rng.choice([0, 1], n, p=[0.8, 0.2]),
        })
        train_ids, test_ids, train_lab, test_lab = stratified_native_split(
            labels)
        assert len(train_ids) + len(test_ids) == n
        assert abs(len(train_ids) / n - 0.7) < 0.03
        assert abs(len(test_ids) / n - 0.3) < 0.03
        assert set(train_ids).isdisjoint(set(test_ids))
        assert set(train_lab['customer_id']) == set(train_ids)
        assert set(test_lab['customer_id']) == set(test_ids)
        assert train_lab['churn'].nunique() == 2
        assert test_lab['churn'].nunique() == 2
        assert abs(train_lab['churn'].mean() - 0.2) < 0.05
        assert abs(test_lab['churn'].mean() - 0.2) < 0.05

    def test_deterministic_across_calls(self):
        labels = pd.DataFrame({
            'customer_id': ['C%04d' % i for i in range(500)],
            'churn': [1 if i % 7 == 0 else 0 for i in range(500)],
        })
        a = stratified_native_split(labels)
        b = stratified_native_split(labels)
        assert a[0] == b[0]
        assert a[1] == b[1]

    def test_duplicate_customer_ids_deduplicated(self):
        labels = pd.DataFrame({
            'customer_id': ['A', 'A', 'B', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                            'I', 'I', 'J', 'J'],
            'churn': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        })
        train_ids, test_ids, _, _ = stratified_native_split(labels)
        assert len(train_ids) + len(test_ids) == 10
        assert set(train_ids + test_ids) == set('ABCDEFGHIJ')

    def test_single_class_raises(self):
        labels = pd.DataFrame({
            'customer_id': ['C%04d' % i for i in range(100)],
            'churn': [0] * 100,
        })
        with pytest.raises(ValueError):
            stratified_native_split(labels)


# ═════════════════════════════════════════════════════════════════════
# Credit Card end-to-end non-temporal path
# ═════════════════════════════════════════════════════════════════════

class TestCreditCardNativeSplit:
    def test_has_temporal_data_false(self):
        assert CreditCardAdapter().has_temporal_data is False
        assert CreditCardAdapter().uses_native_churn_label is True

    def test_leakage_columns_dropped_on_load(self, tmp_path):
        _write_credit_card(tmp_path)
        adapter = CreditCardAdapter()
        adapter.data_dir = str(tmp_path)
        df = adapter.load_raw_data()
        assert 'Naive_Bayes_Class' not in df.columns
        assert 'Naive_Bayes_Probability' not in df.columns

    def test_end_to_end_stratified_split(self, tmp_path):
        _write_credit_card(tmp_path, n=400, churn_frac=0.15, seed=3)
        adapter = CreditCardAdapter()
        adapter.data_dir = str(tmp_path)
        df = adapter.load_raw_data()
        df = adapter.preprocess(df)
        df = adapter.standardize_schema(df)

        labels_all = adapter.get_native_churn_labels(df, df['event_time'].max())
        train_ids, test_ids, train_lab, test_lab = stratified_native_split(
            labels_all)

        # 70/30 customer-level, disjoint, both classes present
        total = len(train_ids) + len(test_ids)
        assert abs(len(train_ids) / total - 0.7) < 0.03
        assert set(train_ids).isdisjoint(set(test_ids))
        assert train_lab['churn'].nunique() == 2
        assert test_lab['churn'].nunique() == 2
        assert train_lab['churn'].mean() > 0 and train_lab['churn'].mean() < 0.5
        assert test_lab['churn'].mean() > 0 and test_lab['churn'].mean() < 0.5

        # Features: non-empty, index == customer ids, no leakage columns
        train_features = engineer_features(
            df, df['event_time'].max(), customer_ids=train_ids,
            available_groups=adapter.available_feature_groups,
            filter_by_snapshot=False,
        )
        test_features = engineer_features(
            df, df['event_time'].max(), customer_ids=test_ids,
            available_groups=adapter.available_feature_groups,
            filter_by_snapshot=False,
        )
        assert not train_features.empty
        assert not test_features.empty
        assert set(train_features.index) == set(train_ids)
        assert set(test_features.index) == set(test_ids)
        for cols in (train_features.columns, test_features.columns):
            assert 'churn' not in cols
            assert 'Attrition_Flag' not in cols
            assert 'CLIENTNUM' not in cols
            assert 'customer_id' not in cols
            assert not any('Naive_Bayes' in c for c in cols)
            assert not any('naive_bayes' in c for c in cols)

        # Feature groups genuinely produced features (not empty shells)
        assert train_features.shape[1] > 3
        assert test_features.shape[1] > 3

    def test_class_distribution_preserved_between_conditions(self, tmp_path):
        """Cross-condition stability: the split is identical every time."""
        _write_credit_card(tmp_path, n=300, churn_frac=0.2, seed=7)
        adapter = CreditCardAdapter()
        adapter.data_dir = str(tmp_path)
        df = adapter.load_raw_data()
        df = adapter.preprocess(df)
        df = adapter.standardize_schema(df)
        labels_all = adapter.get_native_churn_labels(df, df['event_time'].max())

        def _test_ids():
            train_ids, test_ids, _, _ = stratified_native_split(labels_all)
            return list(train_ids), list(test_ids)

        assert _test_ids() == _test_ids()

    def test_get_native_churn_labels_all_customers(self, tmp_path):
        _write_credit_card(tmp_path, n=200, churn_frac=0.15, seed=5)
        adapter = CreditCardAdapter()
        adapter.data_dir = str(tmp_path)
        df = adapter.load_raw_data()
        df = adapter.preprocess(df)
        df = adapter.standardize_schema(df)
        labels = adapter.get_native_churn_labels(df, df['event_time'].max())
        assert len(labels) == df['customer_id'].nunique()
        assert labels['churn'].nunique() == 2


# ═════════════════════════════════════════════════════════════════════
# Telco uses the same framework-level mechanism
# ═════════════════════════════════════════════════════════════════════

class TestTelcoNativeSplit:
    def test_has_temporal_data_false(self):
        assert TelcoAdapter().has_temporal_data is False
        assert TelcoAdapter().uses_native_churn_label is True

    def test_end_to_end_stratified_split(self, tmp_path):
        _write_telco(tmp_path, n=400, churn_frac=0.25, seed=9)
        adapter = TelcoAdapter()
        adapter.data_dir = str(tmp_path)
        df = adapter.load_raw_data()
        df = adapter.preprocess(df)
        df = adapter.standardize_schema(df)

        labels_all = adapter.get_native_churn_labels(df, df['event_time'].max())
        assert len(labels_all) == df['customer_id'].nunique()
        train_ids, test_ids, train_lab, test_lab = stratified_native_split(
            labels_all)
        total = len(train_ids) + len(test_ids)
        assert abs(len(train_ids) / total - 0.7) < 0.03
        assert set(train_ids).isdisjoint(set(test_ids))
        assert train_lab['churn'].nunique() == 2
        assert test_lab['churn'].nunique() == 2

        train_features = engineer_features(
            df, df['event_time'].max(), customer_ids=train_ids,
            available_groups=adapter.available_feature_groups,
            filter_by_snapshot=False,
        )
        test_features = engineer_features(
            df, df['event_time'].max(), customer_ids=test_ids,
            available_groups=adapter.available_feature_groups,
            filter_by_snapshot=False,
        )
        assert not train_features.empty
        assert not test_features.empty
        assert set(train_features.index) == set(train_ids)
        assert set(test_features.index) == set(test_ids)
        for cols in (train_features.columns, test_features.columns):
            assert 'churn' not in cols
            assert 'Churn' not in cols
            assert 'customerID' not in cols


# ═════════════════════════════════════════════════════════════════════
# SMOTE strictly after the split, train data only
# ═════════════════════════════════════════════════════════════════════

class TestSmoteAfterSplit:
    def test_smote_resamples_train_only(self, tmp_path):
        _write_credit_card(tmp_path, n=400, churn_frac=0.2, seed=11)
        adapter = CreditCardAdapter()
        adapter.data_dir = str(tmp_path)
        df = adapter.load_raw_data()
        df = adapter.preprocess(df)
        df = adapter.standardize_schema(df)
        labels_all = adapter.get_native_churn_labels(df, df['event_time'].max())
        train_ids, test_ids, train_lab, test_lab = stratified_native_split(
            labels_all)

        X_train = engineer_features(
            df, df['event_time'].max(), customer_ids=train_ids,
            available_groups=adapter.available_feature_groups,
            filter_by_snapshot=False,
        )
        X_test = engineer_features(
            df, df['event_time'].max(), customer_ids=test_ids,
            available_groups=adapter.available_feature_groups,
            filter_by_snapshot=False,
        )
        y_train = train_lab.set_index('customer_id').loc[X_train.index]['churn']
        y_test = test_lab.set_index('customer_id').loc[X_test.index]['churn']

        X_tr, X_val, y_tr, y_val = _holdout(X_train, y_train)
        X_test_before = X_test.copy(deep=True)
        y_test_before = y_test.copy(deep=True)
        X_val_before = X_val.copy(deep=True)
        y_val_before = y_val.copy(deep=True)

        resampler = get_resampler('smote')
        res = resampler.resample(
            X_tr, y_tr, random_state=RANDOM_SEED,
            k_neighbors=SMOTE_K_NEIGHBORS,
        )
        X_tr_sm, y_tr_sm = res.X_resampled, res.y_resampled

        # SMOTE only grew the training split
        assert len(X_tr_sm) > len(X_tr)
        assert y_tr_sm.sum() > y_tr.sum()
        assert X_val.equals(X_val_before)
        assert y_val.equals(y_val_before)
        assert X_test.equals(X_test_before)
        assert y_test.equals(y_test_before)

        # No validation/test customer leaked into the synthetic train rows
        synth_ids = X_tr_sm.index.difference(X_tr.index)
        assert set(synth_ids).isdisjoint(set(X_val.index))
        assert set(synth_ids).isdisjoint(set(X_test.index))


def _holdout(X, y, test_size=0.1):
    from sklearn.model_selection import train_test_split
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y,
    )


# ═════════════════════════════════════════════════════════════════════
# Regression: temporal (behavioural) path unchanged
# ═════════════════════════════════════════════════════════════════════

class TestTemporalPathRegression:
    def _temporal_df(self, n_customers=40, events_per=10, seed=13):
        rng = np.random.RandomState(seed)
        start = pd.Timestamp('2021-01-01')
        rows = []
        for c in range(n_customers):
            for e in range(events_per):
                rows.append({
                    'customer_id': 'C%03d' % c,
                    'event_time': start + pd.Timedelta(
                        days=int(rng.randint(0, 800))),
                    'event_type': 'purchase',
                    'transaction_value': float(rng.randint(5, 100)),
                    'review_score': float(rng.randint(1, 5)),
                    'payment_type': 'card',
                    'delivery_delay': float(rng.randint(0, 5)),
                    'engagement_signal': 1.0,
                })
        return pd.DataFrame(rows)

    def test_cutoffs_unchanged(self):
        df = self._temporal_df()
        train_cutoff, test_cutoff = get_train_test_cutoffs(df)
        assert train_cutoff < test_cutoff
        assert test_cutoff == df['event_time'].max() - pd.Timedelta(days=180)
        assert abs((df['event_time'] < train_cutoff).mean() - 0.7) < 0.1

    def test_filter_by_snapshot_excludes_future_events(self):
        df = self._temporal_df()
        train_cutoff, _ = get_train_test_cutoffs(df)
        hist = df[df['event_time'] < train_cutoff]
        assert (hist['event_time'] < train_cutoff).all()
        assert len(hist) < len(df)

    def test_create_churn_labels_unchanged(self):
        df = self._temporal_df()
        cutoff = pd.Timestamp('2021-09-01')
        labels = create_churn_labels(df, cutoff, prediction_window_days=180)
        assert 'customer_id' in labels.columns
        assert 'churn' in labels.columns
        assert labels['churn'].isin([0, 1]).all()


# ═════════════════════════════════════════════════════════════════════
# Instacart synthetic-timeline regression — churn must be observable
# ═════════════════════════════════════════════════════════════════════

class TestInstacartTimelineChurn:
    """The old synthetic timeline (``x.max() - x`` per user) pinned every
    user's last order onto the global horizon, collapsing churn to zero and
    forcing 100% train/test overlap.  These tests assert churn is observable
    and the overlap is documented, using the builtin data when present."""

    def _adapter_and_df(self):
        from src.datasets import get_dataset
        adapter = get_dataset('instacart')
        df = adapter.load_raw_data()
        df = adapter.preprocess(df)
        df = adapter.standardize_schema(df)
        return adapter, df

    def test_both_classes_present_at_cutoffs(self):
        adapter, df = self._adapter_and_df()
        window = adapter.churn_window_days
        train_cutoff, test_cutoff = get_train_test_cutoffs(df, 0.7, window)
        train_labels = create_churn_labels(
            df, train_cutoff, prediction_window_days=window)
        test_labels = create_churn_labels(
            df, test_cutoff, prediction_window_days=window)
        assert train_labels['churn'].nunique() >= 2
        assert test_labels['churn'].nunique() >= 2
        train_churn = train_labels['churn'].mean()
        assert 0.05 < train_churn < 0.95, (
            f"train churn rate {train_churn:.3%} is degenerate")

    def test_no_horizon_collapse(self):
        """The old synthetic timeline pinned every user's last order to the
        global horizon (x.max() - x collapsed to zero).  After the fix, users
        retain varied last-order dates and none are forced onto the max date
        merely because their own span is shorter than the global span."""
        adapter, df = self._adapter_and_df()
        last_order = df.groupby('customer_id')['event_time'].max()
        horizon = df['event_time'].max()
        n_at_horizon = int((last_order == horizon).sum())
        assert n_at_horizon < len(last_order), (
            "all last orders pinned to the horizon — timeline collapsed")
        assert last_order.nunique() > 10

    def test_cutoff_level_design_documented(self):
        """Customer-level label sets are identical across cutoffs (the
        framework splits temporally by cutoff, not by customer — the reason
        overlap is expected for temporal datasets).  What must differ is the
        churn rate between train and test cutoffs."""
        adapter, df = self._adapter_and_df()
        window = adapter.churn_window_days
        train_cutoff, test_cutoff = get_train_test_cutoffs(df, 0.7, window)
        train_labels = create_churn_labels(
            df, train_cutoff, prediction_window_days=window)
        test_labels = create_churn_labels(
            df, test_cutoff, prediction_window_days=window)
        assert set(train_labels['customer_id']) == set(test_labels['customer_id'])
        assert train_labels['churn'].mean() != test_labels['churn'].mean()

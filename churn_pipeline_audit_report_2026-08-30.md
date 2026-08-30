# Technical Audit — Churn-Prediction Experiment Pipeline

**Date**: 2026-08-30
**Scope**: Ablation pipeline, Credit Card / Telco ablation validity, RetailRocket 0.77↔0.63 discrepancy, RetailRocket leakage, SMOTE placement, result integrity.
**Evidence base**: source code under `og version/project_root/src/`, saved experiment outputs under `og version/exp-v2/` (results/, processed_data/, final-churn-pipeline.log), earlier research artifacts in the parent research-material directory (multi-dataset-churn-analysis-notebook.ipynb, IMP FIGURES/RetailRocket/retailrocket_temporal_results.zip, IMP FIGURES/RetailRocket/v2/retailrocket_metrics.csv).

---

## 1. EXECUTIVE VERDICT

**The current ablation results for Credit Card and Telco are INVALID.**

- Every ablation condition produced byte-identical ROC-AUC **and** byte-identical std to the `all_features` model. This is a deterministic symptom of a real bug: **the feature-removal step removes zero features.**
- Root cause: `run_ablation()` drops columns by matching dataset column names against the global `FEATURE_GROUPS` dictionary (`src/config.py`). Credit Card (34 features) and Telco (42 features) use **custom native feature matrices whose column names are entirely disjoint from `FEATURE_GROUPS`** (verified: 0 of 34/42 columns match any group name). Consequently `remaining = [f for f in all_features if f not in grp_feats]` returns the full feature set for every group, and `_X[remaining]` is the unmodified matrix every time.

**The RetailRocket 0.77 vs 0.63 discrepancy is real and is NOT a simple regression.** The two runs measure **different populations, different feature sets, different cohorts, and (by construction) different prediction tasks.** In addition, the current (~0.63) RetailRocket pipeline contains **user-level train/test overlap leakage** (86% of test users also appear in the training set), so its ~0.63 is *not* a clean, leakage-free estimate. Neither number can be taken as the single "truth"; they are not directly comparable.

**The SMOTE training path itself is structurally correct** (temporal split → train/test separation → features → SMOTE on the 0.9 training fold only → train → evaluate on untouched test), but the RetailRocket temporal split has a *separate* user-overlap leakage problem that affects both original and SMOTE modes equally.

---

## 2. ABLATION PIPELINE STATUS

### 2a. How ablation runs today
`src/pipeline.py:541` calls `run_ablation(X_train, y_train)` with the **original, non-SMOTE** training matrix in both original and SMOTE modes. `run_ablation` then runs a self-contained `StratifiedKFold` (3 folds) `cross_val_score` for 4 models; each fold re-trains a fresh model from `_model_factory`, and computes ROC-AUC on the CV holdout of `X_train`. It never touches `X_test`, never uses SMOTE-resampled data, and reports **only** `mean_roc_auc` / `std_roc_auc` (no PR-AUC, F1, precision, recall, Brier, calibration).

### 2b. Audit checklist (the 12 required checks)

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | Intended feature group actually removed from X_train/X_test | **FAIL** | `FEATURE_GROUPS` names share **0 columns** with CC/Telco feature matrices (verified on saved `train_features.csv`/`test_features.csv`). `removed=[]` for all 8 groups on both datasets. |
| 2 | Modified matrices actually passed to model | **FAIL (no-op)** | `_X[remaining] == _X` when nothing is removed; cross_val_score receives the full matrix. |
| 3 | Model retrained per ablation condition | **PASS** | `_model_factory` creates a fresh estimator per condition inside each CV. |
| 4 | Predictions from ablated X_test | **N/A (design)** | Ablation is internal CV on `X_train`; it never evaluates on `X_test`. No test-ablation is performed at all. |
| 5 | Evaluation receives ablated predictions | **N/A** | Same as #4 — self-contained CV. |
| 6 | Result-writing logs correct experiment | **PASS (given fix)** | Each condition's own `scores` mean is logged. The bug is that the matrix is unchanged, so the "correct" score is the full-feature score. |
| 7 | No cached models/predictions/metrics reused | **PASS** | No `joblib`/cache/short-circuit in `pipeline.py`, `ablation.py`, `modeling.py`, `evaluation.py`. |
| 8 | No feature-selection re-introduces removed features | **PASS** | No downstream feature-selection in the ablation path. |
| 9 | No ColumnTransformer/preprocessing restores columns | **PASS** | `src/preprocessing.py` is **not imported/used** by the pipeline; models train on raw feature matrices. |
| 10 | SMOTE only on training data, after ablation | **PASS (ablation ignores SMOTE)** | Ablation runs on the original `X_train`; SMOTE is applied only to the 0.9 train fold used for the real model training, never to the ablation. (Note: the "smote" ablation is therefore identical to "original" by design.) |
| 11 | Test untouched by SMOTE | **PASS** | `sm.fit_resample(X_tr, y_tr)` only; `X_val` and `X_test` untouched. |
| 12 | Seeds/splits controlled | **PASS** | `StratifiedKFold(shuffle=True, random_state=42)`; `RANDOM_SEED=42`. |

**Verdict: Check #1/`#2` fail for Credit Card and Telco → ablation results for those two datasets are INVALID.**

### 2c. Why the duplicated scores appeared
For CC/Telco, `FEATURE_GROUPS` uses *standardised* column names (`total_orders`, `days_since_last_purchase`, …), but these datasets bypass the standard feature-engineering path via `build_native_modeling_data` and produce *dataset-native* columns (CC: `Total_Trans_Amt`, `Credit_Limit`, `Gender`, `Education_Level_*`; Telco: `engagement_signal`, `contract_*`, `MonthlyCharges→transaction_value`, one-hot `*_No/Yes`). Because none of these names appear in any `FEATURE_GROUPS` list, the "remove group" operation is a no-op for all 8 groups, so every `without_*` condition fits the **full matrix** → identical ROC-AUC and identical std. This is exactly the observed symptom.

### 2d. Sanity: which datasets show the bug
Evaluated on saved `ablation_results.csv` per dataset (number of unique ROC-AUC values among the 9 Rows per model; 9 rows = all_features + 8 groups):

- **Credit Card — INVALID** (1 unique value/model; all 8 groups remove nothing).
- **Telco — INVALID** (1 unique value/model; all 8 groups remove nothing).
- **RetailRocket — structurally OK** (4–5 unique values/model; removal works for the groups that are present; `review`/`delivery`/`payment`/`cadence` groups aren't in the feature set so removing them genuinely removes nothing, which is correct).
- **Online Retail II — structurally OK** (4 unique values/model).
- **Olist — structurally OK** (7 unique values/model).

### 2e. HARD SANITY TEST (required by the brief)
The brief asks for a hard sanity test (full set / remove dominant SHAP group / remove unrelated group / keep-only-dominant group) on LightGBM or XGBoost for CC and Telco. **This cannot be executed in this workspace**: the Credit Card raw file `BankChurners.csv` is not present anywhere in the research material, and Telco raw data is only available as a small local sample (`notebooks/code/project_root/data/builtin/telco_customer_churn.csv`, which is < 1/7 of the real 7,043-row dataset and is a synthetic probe, not the exp-v2 input). The exp-v2 outputs used Kaggle-hosted full data. A faithful hard sanity test must be run on the **same Kaggle/full-data environment** that produced exp-v2.

Given the removal logic demonstrably removes zero features for CC/Telco (proven directly from the saved matrices), the outcome of a sanity test is already foreordained for the buggy code path: every condition would equal the full-feature score. The fix (dataset-aware feature-group mapping) must be implemented before any sanity test can be meaningful.

---

## 3. CREDIT CARD AUDIT

- **Split**: native `Attrition_Flag` label; `build_native_modeling_data` → 70/30 **customer-stratified** split (verified user-disjoint train/test: overlap = 0).
- **Features (34)**: 12 continuous (`Total_Trans_Amt`, `Total_Trans_Ct`, `Total_Revolving_Bal`, `Months_Inactive_12_mon`, `Contacts_Count_12_mon`, `Avg_Utilization_Ratio`, `Credit_Limit`, `Months_on_book`, `Total_Relationship_Count`, `Total_Amt_Chng_Q4_Q1`, `Total_Ct_Chng_Q4_Q1`, `Avg_Open_To_Buy`) + 22 dummies (`Gender`, `Education_Level_*`, `Marital_Status_*`, `Income_Category_*`, `Card_Category_*`).
- **Test** (original): N=3,039, churn rate 16.06%, ROC-AUC LR 0.905 / RF 0.970 / XGB 0.991 / LGBM 0.992 / SVM 0.918.
- **Ablation**: INVALID — every condition equals all_features. **Must rerun.**
- The main CC model metrics themselves come from a clean user-disjoint split and are not affected by the ablation bug; they are believable for this dataset (native label, strong discriminative predictors).

---

## 4. TELCO AUDIT

- **Split**: native `Churn` label; `build_native_modeling_data` → customer-stratified 70/30 (overlap = 0).
- **Features (42)**: `SeniorCitizen`, `engagement_signal` (=tenure), `transaction_value` (=MonthlyCharges), `total_charges`, 5 `*_encoded` binary, and one-hot vectors for `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaymentMethod`, plus constant `review_score`/`delivery_delay`.
- **Test** (original): N=2,113, churn rate 26.55%, ROC-AUC LR 0.844 / RF 0.845 / XGB 0.840 / LGBM 0.821 / SVM 0.843.
- **Ablation**: INVALID — identical across conditions. **Must rerun.**
- Main model metrics come from a clean user-disjoint split and are not affected by the ablation bug.

---

## 5. RETAILROCKET SCORE DISCREPANCY (0.77 vs 0.63)

### 5a. The ~0.77 result
Source artifact: `IMP FIGURES/RetailRocket/retailrocket_temporal_results.zip` → `metrics.csv` reports **roc_auc = 0.7693**.
Reconstructed from `retailrocket_temporal_features.csv`:
- Population: **326,717 users** (empirically counted in the feature file).
- 13 features: `total_events, total_views, total_carts, total_transactions, unique_items, active_days, days_since_last_event_before_cutoff, event_frequency, view_to_cart_ratio, cart_to_purchase_ratio, avg/max/min_interaction_gap`.
- Churn rate: **0.9613** (96.13%).
- Reported: F1 0.828, precision 0.983, recall 0.715, accuracy 0.714, Brier 0.192.
- Caveat: the generating script is **not preserved** in the repository; only its outputs remain, so the exact churn-label rule and split procedure cannot be re-derived from source.

### 5b. The current ~0.63 result
Source artifact: `og version/exp-v2/.../retailrocket/original/model_metrics.csv` and pipeline log.
- Population: train 978,922 + test 1,133,422 = **~2.11M users** (no minimum-activity filter).
- 15 features (standard groups): `total_orders, total_items_purchased, repeat_purchase_ratio, total_spent, avg_order_value, max_order_value, min_order_value, days_since_last_purchase, total_page_views, total_cart_adds, total_purchases, total_wishlist_adds, total_events, total_sessions, avg_actions_per_session`.
- Cutoffs: train 2015-08-02, test 2015-08-19; window 30 days; quantile 0.7.
- Churn rate: 97.68% train / 97.86% test (label = no event in 30 d after cutoff).
- ROC-AUC: LR 0.637 / RF 0.637 / XGB 0.637 / LGBM 0.631 / SVM 0.624.

### 5c. Are they the same prediction task? **NO.**
| Dimension | ~0.77 temporal | ~0.63 current |
|---|---|---|
| Users modeled | 326,717 (filtered/sparse-removed) | ~2.11M (unfiltered, includes single-session visitors) |
| Churn rate | 96.13% | 97.7–97.9% |
| Feature count | 13 (funnel/ratio/cadence) | 15 (standard groups) |
| Feature names | disjoint (different FE pipeline) | disjoint |
| Cohort filter | present (active users only) | none |
| Churn rule | not recoverable (code unpreserved) | no-event-in-30-days |

The number of modelled users differs by ~**6.5×**, the feature sets are entirely different, and the population filters differ. The two numbers are not measurements of the same task, so they **cannot be reconciled into a single RetailRocket figure**.

### 5d. Conclusion for the discrepancy
The instructions allow these conclusions; the honest one is a combination:
1. **The current pipeline is a different, more automated/standardized configuration** (no minimum-activity filter → includes ~2.11M sparse single-event users, which depresses AUC toward majority-class baseline) — it is not a like-for-like regression.
2. **The current pipeline contains user-level train/test overlap leakage** (86% of test users are also in train), so its ~0.63 is **not** a clean leakage-free estimate. This undermines treating 0.63 as the corrected "truth."
3. The ~0.77 result's precise label rule is unrecoverable from preserved artifacts, so I cannot certify it as leakage-free either; I only certify the numbers and population I measured.
4. **We should NOT present a single RetailRocket ROC-AUC as the headline** until a user-disjoint, minimum-activity-controlled, documented churn rule is run. Both historical numbers, and the current one, are incomparable and/or contaminated.

**Preserve the contradiction** — do not force 0.63 to match 0.77. Fix the pipeline (user-disjoint split + explicit cohort/churn rule) and report the new, audited number.

---

## 6. RETAILROCKET LEAKAGE AUDIT (Part 3)

Checked against the **current** `og version/project_root` pipeline for the standard temporal path (`retailrocket` uses the default non-user-relative temporal split).

| Item | Status | Notes |
|---|---|---|
| Features built only from events < cutoff | **PASS** | `engineer_features(...)` filters `hist = df[df.event_time < snapshot]`. |
| Churn labels depend only on post-cutoff | **PASS** | `create_churn_labels` uses events in `(cutoff, cutoff+window]`. |
| Recency feature independent of future window | **PASS** | `days_since_last_purchase = snapshot − last_precutoff_event`; no future use. |
| No dataset-endpoint-derived variables | **PASS** | Endpoint/max-date not used as a feature. |
| No future activity counts | **PASS** | All count features are pre-cutoff aggregates. |
| No final-event-timestamp-vs-endpoint feature | **PASS** | Not present in the 15-feature set. |
| No future events in user aggregation | **PASS** | GroupBy restricted to the pre-cutoff slice. |
| Preprocessing fit only on train | **PASS** | No scaler/column-transformer in main path (`preprocessing.py` unused). |
| **User-level train/test disjointness** | **FAIL / UNCERTAIN** | **Overlap = 978,922 users — 86.37% of test users are also in the training set.** Train and test label windows differ for the same user (train: no event Aug 2–Sep 1; test: no event Aug 19–Sep 18), and test features include events in the period (Aug 2–Aug 19) that overlap the train window. This is *user-level leakage / non-independence* and inflates optimism. This is a different mechanism than the historical ~1.00 temporal-leak, but it is genuine. |

**Verdict: The current RetailRocket result is NOT temporally leakage-free because of the user-overlap issue.** This is not the historical "future-features" leak (that appears fixed — the ~0.63 numbers are far from 1.00), but it is a separate, reportable flaw.

Note: Olist shows the same overlap (40% of test users in train), i.e. the standard temporal path is systemic. Credit Card and Telco (native-stratified) and Instacart (user-relative) are user-disjoint.

---

## 7. SMOTE PIPELINE AUDIT (Part 4)

For every dataset, `run_pipeline(use_smote=…)`:
```
temporal/native split (train/test) 
→ engineer features (pre-cutoff) 
→ ALIGN columns 
→ train/test separation 
→ extra 90/10 train/val split 
→ IF smote: fit_resample ONLY on the 0.9 train fold 
→ train models 
→ evaluate on unmodified X_test
```

- SMOTE **before** temporal split? **No.**
- SMOTE on **test**? **No** (`X_test` untouched).
- SMOTE before train/test separation? **No** (applied after `tts`).
- SMOTE using future info? **No** (operates on features only).
- Original vs SMOTE share identical split/features/churn/model/test population? **PASS** — only the training-fold resampling differs. (Verified: same cutoffs, same feature matrices, same test set; SMOTE-only changes are visible in `model_metrics_smote.csv` differing from `model_metrics.csv`.)

**SMOTE placement is structurally correct.** Caveat: the *ablation* is run on the original `X_train` in both modes, so the "SMOTE ablation" does not reflect SMOTE — but that is a design note, not a correctness failure of the SMOTE training path itself.

---

## 8. EXPERIMENTS THAT ARE SAFE TO KEEP

- **Current main model metrics (hold-out ROC/PR/F1/Brier/ECE) for Credit Card and Telco** — user-disjoint stratified splits; not affected by the ablation bug. (These are dataset-level results; the values are high but plausible given native labels.)
- **SMOTE vs original model-metric comparison** for CC/Telco — valid placement, user-disjoint splits.
- **RetailRocket / Olist ablation results** — structurally correct removal (scores vary) — *but* keep only if you also accept the user-overlap caveat below.
- **Instacart (user-relative split), Online Retail II ablation** — structurally correct.

## 9. EXPERIMENTS THAT MUST BE RERUN

- **Credit Card ablation (original AND smote)** — invalid (all conditions identical). Must re-run with a dataset-aware feature-group mapping.
- **Telco ablation (original AND smote)** — invalid. Must re-run.
- Any ablation across **all 8 datasets** if a global fix is applied (to be consistent and reproducible under the corrected mapping).

## 10. EXPERIMENTS THAT MUST NOT BE REPORTED (as-is)

- **Credit Card / Telco ablation** rows (`without_*`) — they are outputs of a no-op removal; reporting them implies a false "robustness" finding. **Do not report.**
- **Current RetailRocket / Olist** ROC-AUC as a clean leakage-free generalisation estimate — **do not report as-is** because of the **user-level train/test overlap**.
- The **~0.77 RetailRocket** figure — **do not report as-is**; its generation script is unpreserved, its label rule unverifiable, and it is incomparable with the current pipeline.
- The historical **~1.00 RetailRocket** (temporal leakage) — must not be reported (leak).

## 11. EXACT RERUN COMMANDS / SCRIPTS

The exp-v2 run used **Kaggle** full-hosted data (paths `/kaggle/input/datasets/...`). No local credit-card raw data exists in this workspace. Reruns must happen in the same Kaggle/full-data environment.

1. **Fix the ablation** so group membership is defined in terms of the *actual* feature columns of each dataset. Because CC/Telco bypass the standard FE pipeline, the `FEATURE_GROUPS` name map is insufficient. Add a per-dataset "ablation groups → actual column names" mapping (e.g. CC: `purchase→{Total_Trans_Ct, Total_Trans_Amt}`, `inactivity→{Months_Inactive_12_mon}`, `engagement→{Contacts_Count_12_mon, Avg_Utilization_Ratio}`, etc.; Telco: `contract→{Contract_*}`, `payment→{PaymentMethod_*}`, `tenure→{engagement_signal}`, etc.), and make `run_ablation` use that mapping.
2. **Fix the temporal split** to be user-disjoint for RetailRocket/Olist (either adopt user-relative splits like Instacart, or partition users by identity with a time-ordered feature/label construction).

Then, in an environment with the full data:

```bash
# Credit Card (original + smote), after the mapping fix
python -m src.pipeline credit_card
python -m src.pipeline credit_card --use-smote   # (flag is use_smote via run_pipeline API)
python -c "from src.pipeline import run_pipeline; run_pipeline('credit_card', use_smote=False)"
python -c "from src.pipeline import run_pipeline; run_pipeline('credit_card', use_smote=True)"

# Telco
python -m src.pipeline telco
python -c "from src.pipeline import run_pipeline; run_pipeline('telco', use_smote=True)"

# RetailRocket under a user-disjoint split (after Part 6 fix)
python -c "from src.pipeline import run_pipeline; run_pipeline('retailrocket', use_smote=False)"
python -c "from src.pipeline import run_pipeline; run_pipeline('retailrocket', use_smote=True)"

# Full audited sweep (only after both fixes)
python -m src.pipeline --smote-comparison
```
(The `--use-smote` CLI flag does not exist in `src/pipeline.py.__main__`; use the API form above or extend the CLI.)

**Hard sanity test** for CC/Telco, after the mapping fix, on LightGBM:
1. full set
2. remove dominant SHAP group
3. remove unrelated/minor group
4. keep only dominant group

via a short script that loads `X_train`/`X_test` from `processed_data/<ds>/original/`, drops the mapped group columns, and fits/evaluates a fresh `LGBMClassifier` per condition — **do not interpret results; only record them.**

## 12. EXPECTED OUTPUT FILES

After fixing ablation, each CC/Telco rerun should produce:
- `results/<dataset>/{original,smote}/ablation/ablation_results{,_smote}.csv` — now with **distinct ROC-AUC** per removed group (no longer all equal to `all_features`).
- `processed_data/<dataset>/{original,smote}/train_features{,_smote}.csv` / `test_features{,_smote}.csv`
- `results/<dataset>/{original,smote}/model_metrics/model_metrics{,_smote}.csv`
- `results/cross_dataset/master_results_{original,smote}.csv`
- For the sanity test: a new CSV (e.g. `results/<ds>/ablation/hard_sanity_lightgbm.csv`) recording full / minus-dominant / minus-minor / only-dominant ROC-AUC, PR-AUC, F1, precision, recall, Brier, ECE.

After fixing the temporal split: new RetailRocket/Olist processed_data with **no user overlap** between train and test (verify `set(train.customer_id) & set(test.customer_id)` is empty).

## 13. RECOMMENDED FINAL EXPERIMENT MATRIX

Reportable, audited cells (each = dataset × model × ROC-AUC / PR-AUC / F1 / Brier / ECE) — **total 8 datasets × 5 models × 2 SMOTE states = 80 cells** as originally intended, but **only after**:
1. ablation fixed (CC/Telco groups mapped to real columns),
2. RetailRocket/Olist switched to user-disjoint temporal splits,
3. RetailRocket cohort/churn rule made explicit and logged,
4. ablation results re-checked to contain distinct values,
5. leakage checklist re-run and all rows PASS.

**Do NOT** present RetailRocket (or Olist) from the current run, and do **not** report the CC/Telco ablation rows from the current run.

---

## Appendix — Requested per-condition per-feature print-outs (CC/Telco)

Full and ablated matrices were computed from the saved `processed_data`; every ablation `without_*` on CC/Telco yields `removed=[]`, i.e. the ablated matrix is byte-identical to the full matrix. Feature inventories (from `train_features.csv`):

**Credit Card full X_train / X_test (34 feats):**
`Total_Trans_Amt, Total_Trans_Ct, Total_Revolving_Bal, Months_Inactive_12_mon, Contacts_Count_12_mon, Avg_Utilization_Ratio, Credit_Limit, Months_on_book, Total_Relationship_Count, Total_Amt_Chng_Q4_Q1, Total_Ct_Chng_Q4_Q1, Avg_Open_To_Buy, Gender, Education_Level_{College,Doctorate,Graduate,High School,Post-Graduate,Uneducated,Unknown}, Marital_Status_{Divorced,Married,Single,Unknown}, Income_Category_{$120K +,$40K-$60K,$60K-$80K,$80K-$120K,Less than $40K,Unknown}, Card_Category_{Blue,Gold,Platinum,Silver}`

**Telco full X_train / X_test (42 feats):**
`SeniorCitizen, engagement_signal, transaction_value, total_charges, PhoneService_encoded, PaperlessBilling_encoded, gender_encoded, Partner_encoded, Dependents_encoded, MultipleLines_{No,No phone service,Yes}, InternetService_{DSL,Fiber optic,No}, OnlineSecurity_{No,No internet service,Yes}, OnlineBackup_{No,No internet service,Yes}, DeviceProtection_{No,No internet service,Yes}, TechSupport_{No,No internet service,Yes}, StreamingTV_{No,No internet service,Yes}, StreamingMovies_{No,No internet service,Yes}, Contract_{Month-to-month,One year,Two year}, PaymentMethod_{Bank transfer (automatic),Credit card (automatic),Electronic check,Mailed check}, review_score, delivery_delay`

**Ablation removal result for every group on CC/Telco:** `without_{purchase,monetary,inactivity,review,delivery,payment,engagement,cadence}` → `removed=[], X_train_after_nfeats = 34 (CC) / 42 (Telco)` → identical to full → identical ROC-AUC and std. **Confirms INVALID.**

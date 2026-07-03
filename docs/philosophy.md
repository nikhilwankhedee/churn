# Design Philosophy

## Research-First

This framework is built for **researchers**, not AutoML practitioners.

- **No silent model selection** — You choose which model to trust
- **No automatic feature engineering** — Features are explicit and documented
- **No hidden preprocessing** — Every step is logged and reproducible
- **No magic defaults** — Every parameter has a documented rationale

## Reproducibility

- **Fixed random seeds** — `RANDOM_SEED=42` everywhere
- **YAML configurations** — Every experiment is fully specified
- **Experiment tracking** — Results logged to CSV
- **Published baseline** — The original pipeline is never modified

## Modularity

- **Registry-based** — Every component is pluggable
- **Graceful degradation** — Failures are logged, not fatal
- **Composable** — Mix and match strategies, models, metrics
- **Backward compatible** — New features wrap existing code

## Honest Evaluation

- **Multiple metrics** — Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier, ECE
- **Temporal splits** — Train/test split respects time ordering
- **No data leakage** — Strict temporal boundaries enforced
- **Calibration** — Probability reliability assessed
- **Statistical tests** — Cross-dataset comparisons validated

## Cross-Ecosystem Generalization

The framework tests whether behavioral churn patterns generalize across:

- **Transactional marketplaces** (Olist, Instacart)
- **Clickstream commerce** (Rees46, RetailRocket)
- **Subscription services** (Telco)
- **Habitual retail** (Online Retail II)

This is the core research question: do behavioral signatures of churn transcend ecosystem boundaries?

---

*Developed by Nikhil Wankhede*

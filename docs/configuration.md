# Configuration Reference

Customize the framework via YAML configuration files.

## Structure

```yaml
dataset:
  name: olist
  ecosystem_type: transactional_marketplace
  citation: "Author (2024)"
  source_url: https://example.com

churn:
  strategy: inactivity          # inactivity | subscription | cadence
  prediction_window_days: 180
  justification: "180-day window based on..."

features:
  available_groups:
    - purchase
    - monetary
    - inactivity
    - review
    - delivery
    - payment
    - engagement
    - cadence

schema:
  column_mapping:
    customer_unique_id: customer_id
    order_purchase_timestamp: event_time
    payment_value: transaction_value

files:
  orders: olist_orders_dataset.csv
  customers: olist_customers_dataset.csv

preprocessing:
  timestamp_columns:
    - order_purchase_timestamp
  drop_null_timestamp: order_purchase_timestamp
  non_negative_columns: [price, freight_value]
  outlier_cap_percentile: 0.999
  median_fill: [review_score]
  zero_fill: [payment_installments]

resampling:
  enabled: false
  method: smote
  random_state: 42
```

## Loading Configs

```bash
# Via CLI
churn run olist --config my_config.yaml

# Via Python
fw = ChurnFramework(config_path="my_config.yaml")
```

## Config Values

Access config values programmatically:

```python
from src.config import get_config_value

strategy = get_config_value("churn.strategy", "inactivity")
window = get_config_value("churn.prediction_window_days", 180)
```

## Validation

```bash
churn validate-config my_config.yaml
```

Checks: required sections, valid enums, reasonable values, file existence.

---

*Developed by Nikhil Wankhede*

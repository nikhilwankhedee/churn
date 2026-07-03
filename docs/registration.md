# Dataset Registration

Add new datasets to the framework in minutes.

## Quick Registration

```bash
# Step 1: Inspect and generate config
churn register path/to/data.csv --name my_dataset

# Step 2: Review and edit the generated config
cat configs/datasets/my_dataset.yaml

# Step 3: Place your data file
cp data.csv data/my_dataset.csv

# Step 4: Run
churn run my_dataset
```

## What the Wizard Does

1. **Reads** the first 10,000 rows of your CSV
2. **Infers** column roles (customer ID, timestamp, monetary value, etc.)
3. **Classifies** columns as numeric, datetime, or categorical
4. **Suggests** feature groups based on detected columns
5. **Generates** a YAML config following framework conventions

## CLI Options

```bash
churn register data.csv \
  --name my_dataset \
  --ecosystem transactional_marketplace \
  --customer-id user_id \
  --timestamp order_date \
  --source-url https://example.com/data \
  --citation "Author (2024)" \
  --output configs/datasets/my_dataset.yaml
```

## Configuration Structure

The generated YAML follows this structure:

```yaml
dataset:
  name: my_dataset
  ecosystem_type: transactional_marketplace

churn:
  strategy: inactivity
  prediction_window_days: 180

features:
  available_groups:
    - purchase
    - monetary
    - inactivity

schema:
  column_mapping:
    native_column_name: standardized_name

preprocessing:
  timestamp_columns:
    - order_date
```

## Column Mapping

Map your native column names to the standardized schema:

| Standard Name | Description | Required |
|--------------|-------------|----------|
| `customer_id` | Unique customer identifier | Yes |
| `event_time` | Transaction/event timestamp | Yes |
| `transaction_value` | Monetary value | No |
| `event_type` | Event type (view, purchase, etc.) | No |
| `review_score` | Review/rating score | No |
| `payment_type` | Payment method | No |

## Ecosystem Types

| Type | Description | Default Window |
|------|-------------|---------------|
| `transactional_marketplace` | E-commerce marketplaces | 180 days |
| `clickstream_commerce` | Web analytics / clickstream | 90 days |
| `habitual_retail` | Repeat-purchase retail | 180 days |
| `subscription` | Subscription services | Native label |

## Python API

```python
from src.api import ChurnFramework

fw = ChurnFramework()
config_path = fw.register_dataset(
    csv_path="data.csv",
    name="my_dataset",
    ecosystem="transactional_marketplace",
)
```

## Manual Configuration

You can also create configs manually. See `configs/datasets/olist.yaml` for a complete example.

## Validation

Always validate your config before running:

```bash
churn validate-config configs/datasets/my_dataset.yaml
```

---

*Developed by Nikhil Wankhede*

This checks for:
- Required sections present
- Valid churn strategy names
- Valid feature group names
- Reasonable prediction window
- File existence (if data files referenced)

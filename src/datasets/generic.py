"""
Generic dataset adapter — executes churn pipelines from manifest YAML alone.

This adapter loads, preprocesses, and standardizes any dataset described
by a manifest YAML file. It eliminates the need for hand-coded adapters
for most datasets.

The manifest becomes the single source of truth. This adapter reads:
- files section → which CSVs to load and how to join them
- schema section → column mapping to canonical names
- preprocessing section → timestamp parsing, imputation, filtering
- churn section → strategy, window, native labels
- features section → available feature groups

For datasets with genuinely custom logic (multi-table joins with complex
aggregation, domain-specific preprocessing), an optional plugin module
can be specified in adapter.plugin.
"""
import os
import importlib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.datasets.base import BaseDatasetAdapter
from src.utils import get_logger

logger = get_logger(__name__)


class GenericDatasetAdapter(BaseDatasetAdapter):
    """Manifest-driven adapter that executes without hand-coded Python.

    Usage:
        adapter = GenericDatasetAdapter(manifest_path="/path/to/manifest.yaml")
        adapter = GenericDatasetAdapter(manifest_dict={...})
        adapter.data_dir = "/path/to/data"
    """

    def __init__(
        self,
        manifest_path: Optional[str] = None,
        manifest_dict: Optional[Dict[str, Any]] = None,
    ):
        """Initialize from a manifest file or dict.

        Parameters
        ----------
        manifest_path : str, optional
            Path to the manifest YAML file.
        manifest_dict : dict, optional
            Already-loaded manifest dictionary.
        """
        if manifest_dict is not None:
            self._manifest = manifest_dict
        elif manifest_path is not None:
            self._manifest = self._load_manifest(manifest_path)
        else:
            raise ValueError(
                "GenericDatasetAdapter requires either manifest_path or manifest_dict"
            )

        self._plugin = None
        self._plugin_loaded = False

        # Set data_dir from manifest root_directory if available
        root_dir = self._manifest.get("root_directory")
        if root_dir:
            self._resolved_data_dir = str(root_dir)

    @staticmethod
    def _load_manifest(path: str) -> Dict[str, Any]:
        """Load a manifest YAML file."""
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def _ensure_plugin(self) -> None:
        """Load the optional plugin module if specified."""
        if self._plugin_loaded:
            return
        self._plugin_loaded = True

        plugin_path = self._manifest.get("adapter", {}).get("plugin")
        if not plugin_path:
            return

        try:
            if "." in plugin_path:
                module_path, class_name = plugin_path.rsplit(".", 1)
                mod = importlib.import_module(module_path)
                self._plugin = getattr(mod, class_name)()
            else:
                mod = importlib.import_module(plugin_path)
                self._plugin = mod
            logger.info("Loaded plugin: %s", plugin_path)
        except Exception as exc:
            logger.warning("Failed to load plugin '%s': %s", plugin_path, exc)

    # ── BaseDatasetAdapter interface ────────────────────────────────

    @property
    def dataset_name(self) -> str:
        return self._manifest.get("dataset", {}).get("name", "unknown")

    @property
    def ecosystem_type(self) -> str:
        return self._manifest.get("dataset", {}).get("ecosystem_type", "unknown")

    @property
    def churn_window_days(self) -> Optional[int]:
        churn = self._manifest.get("churn", {})
        if churn.get("uses_native_churn_label", False):
            return None
        return churn.get("prediction_window_days", 180)

    @property
    def uses_native_churn_label(self) -> bool:
        return self._manifest.get("churn", {}).get("uses_native_churn_label", False)

    def get_native_churn_labels(
        self, df: pd.DataFrame, cutoff_date: pd.Timestamp,
    ) -> pd.DataFrame:
        if self._plugin and hasattr(self._plugin, "get_native_churn_labels"):
            return self._plugin.get_native_churn_labels(df, cutoff_date)

        # Try to find a churn column in the data
        churn_col = None
        for candidate in ["Churn", "churn", "churned", "is_churned", "has_churned"]:
            if candidate in df.columns:
                churn_col = candidate
                break

        if churn_col is None or "customer_id" not in df.columns:
            raise ValueError(
                f"Dataset '{self.dataset_name}': no churn column found for "
                "native label extraction. Provide a plugin or add a 'churn' column."
            )

        labels = (
            df[["customer_id", churn_col]]
            .drop_duplicates(subset="customer_id")
            .copy()
        )
        # Normalize to 0/1
        try:
            labels[churn_col] = pd.to_numeric(
                labels[churn_col], errors="coerce"
            ).fillna(0).astype(int)
        except Exception:
            # Fallback: map string values
            labels[churn_col] = labels[churn_col].map(
                {"Yes": 1, "No": 0, "yes": 1, "no": 0, "1": 1, "0": 0,
                 True: 1, False: 0}
            ).fillna(0).astype(int)

        labels = labels.rename(columns={churn_col: "churn"})
        churn_rate = labels["churn"].mean()
        logger.info(
            "Native churn labels — rate: %.2f%% (%d / %d)",
            churn_rate * 100, int(labels["churn"].sum()), len(labels),
        )
        return labels

    @property
    def available_feature_groups(self) -> List[str]:
        return self._manifest.get("features", {}).get(
            "available_groups",
            ["purchase", "monetary", "inactivity"],
        )

    @property
    def required_files(self) -> List[str]:
        files = self._manifest.get("files", {})
        required = files.get("required", {})
        optional = files.get("optional", {})
        if isinstance(required, dict):
            return list(required.values())
        elif isinstance(required, list):
            return required
        return []

    @property
    def data_dir(self) -> str:
        """Return data directory, checking manifest root_directory first."""
        if hasattr(self, '_resolved_data_dir') and self._resolved_data_dir:
            return str(self._resolved_data_dir)

        # Check manifest root_directory
        root_dir = self._manifest.get("root_directory")
        if root_dir:
            self._resolved_data_dir = str(root_dir)
            return str(root_dir)

        # Fall back to centralized resolver
        from src.dataset_resolver import resolve_dataset_directory
        try:
            return resolve_dataset_directory(
                dataset_name=self.dataset_name,
                required_files=self.required_files or None,
            )
        except FileNotFoundError:
            from src.config import DATA_DIR
            return DATA_DIR

    @property
    def metadata(self) -> Dict[str, Any]:
        ds = self._manifest.get("dataset", {})
        churn = self._manifest.get("churn", {})
        return {
            "dataset_name": self.dataset_name,
            "ecosystem_type": self.ecosystem_type,
            "citation": ds.get("citation", ""),
            "source_url": ds.get("source_url", ""),
            "n_customers_approx": ds.get("n_customers_approx", 0),
            "n_orders_approx": ds.get("n_orders_approx", 0),
            "churn_window_days": self.churn_window_days,
            "churn_justification": churn.get("justification", ""),
            "uses_native_churn_label": self.uses_native_churn_label,
            "available_feature_groups": self.available_feature_groups,
        }

    # ── Data loading ────────────────────────────────────────────────

    def load_raw_data(self) -> pd.DataFrame:
        """Load all CSVs from manifest and merge them."""
        self._ensure_plugin()

        # Plugin can completely override loading
        if self._plugin and hasattr(self._plugin, "load_raw_data"):
            return self._plugin.load_raw_data(self.data_dir, self._manifest)

        files_section = self._manifest.get("files", {})
        required = files_section.get("required", {})
        optional = files_section.get("optional", {})

        if isinstance(required, dict):
            all_files = {**required}
        elif isinstance(required, list):
            all_files = {f"file_{i}": f for i, f in enumerate(required)}
        else:
            all_files = {}

        if isinstance(optional, dict):
            all_files.update(optional)

        if not all_files:
            raise ValueError(
                f"Dataset '{self.dataset_name}': no files defined in manifest"
            )

        # Load each CSV
        tables: Dict[str, pd.DataFrame] = {}
        for role, filename in all_files.items():
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.isfile(filepath):
                if role in (required if isinstance(required, dict) else {}):
                    raise FileNotFoundError(
                        f"Required file '{filename}' not found in {self.data_dir}"
                    )
                logger.warning("Optional file not found: %s — skipping", filename)
                continue

            try:
                # Check for encoding override in preprocessing
                encoding = self._manifest.get("preprocessing", {}).get("encoding")
                read_kwargs = {}
                if encoding:
                    read_kwargs["encoding"] = encoding

                df = pd.read_csv(filepath, **read_kwargs)
                tables[role] = df
                logger.info(
                    "Loaded %s (%s): %d rows x %d cols",
                    role, filename, df.shape[0], df.shape[1],
                )
            except Exception as exc:
                logger.error("Failed to load %s (%s): %s", role, filename, exc)
                if role in (required if isinstance(required, dict) else {}):
                    raise

        if not tables:
            raise FileNotFoundError(
                f"Dataset '{self.dataset_name}': no data files found in {self.data_dir}"
            )

        # Merge or concatenate tables
        merged = self._merge_tables(tables)

        # Plugin post-load hook
        if self._plugin and hasattr(self._plugin, "post_load"):
            merged = self._plugin.post_load(merged, self._manifest)

        return merged

    def _merge_tables(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple tables into a single DataFrame.

        Strategy:
        - Find the 'primary' table (usually the one with the most rows or
          the one named 'orders'/'events'/'data')
        - Left-join other tables on shared column names
        - Concatenate files with the same role (e.g., year_1 + year_2)
        """
        if len(tables) == 1:
            return list(tables.values())[0].copy()

        # Detect concatenation groups (tables with similar names)
        # e.g., year_1, year_2 → concat; orders, customers → join
        concat_groups: Dict[str, List[str]] = {}
        join_tables: Dict[str, pd.DataFrame] = {}

        for role, df in tables.items():
            # Check if this role should be concatenated with others
            base_role = role.rsplit("_", 1)[0] if role[-1].isdigit() and "_" in role else None
            if base_role and base_role in tables:
                if base_role not in concat_groups:
                    concat_groups[base_role] = []
                concat_groups[base_role].append(role)
            else:
                join_tables[role] = df

        # Concatenate groups
        for base_role, group_roles in concat_groups.items():
            frames = [tables[r] for r in [base_role] + group_roles if r in tables]
            if frames:
                join_tables[base_role] = pd.concat(frames, ignore_index=True)
                logger.info(
                    "Concatenated %d files for '%s': %d total rows",
                    len(frames), base_role, len(join_tables[base_role]),
                )

        if not join_tables:
            return pd.DataFrame()

        if len(join_tables) == 1:
            return list(join_tables.values())[0].copy()

        # Determine primary table (most rows)
        primary_role = max(join_tables, key=lambda r: len(join_tables[r]))
        result = join_tables[primary_role].copy()

        # Find join keys by looking for common column names
        for role, df in join_tables.items():
            if role == primary_role:
                continue
            common_cols = set(result.columns) & set(df.columns)
            # Prefer known join keys
            join_key = None
            for key_candidate in [
                "order_id", "customer_id", "user_id", "visitorid",
                "product_id", "item_id", "category_id",
            ]:
                if key_candidate in common_cols:
                    join_key = key_candidate
                    break
            if join_key is None and common_cols:
                join_key = max(common_cols, key=lambda c: len(set(result[c].dropna())))

            if join_key:
                result = result.merge(df, on=join_key, how="left")
                logger.info("Joined '%s' on '%s'", role, join_key)
            else:
                logger.warning(
                    "No common key found between '%s' and '%s' — skipping join",
                    primary_role, role,
                )

        return result

    # ── Preprocessing ───────────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply manifest-defined preprocessing."""
        self._ensure_plugin()

        if self._plugin and hasattr(self._plugin, "preprocess"):
            return self._plugin.preprocess(df, self._manifest)

        df = df.copy()
        preprocessing = self._manifest.get("preprocessing", {})

        # Parse timestamps
        ts_cols = preprocessing.get("timestamp_columns", [])
        for col in ts_cols:
            if col in df.columns:
                encoding = preprocessing.get("encoding")
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

        # Drop null timestamps
        drop_null_ts = preprocessing.get("drop_null_timestamp")
        if drop_null_ts and drop_null_ts in df.columns:
            before = len(df)
            df = df.dropna(subset=[drop_null_ts])
            dropped = before - len(df)
            if dropped:
                logger.info("Dropped %d rows with null %s", dropped, drop_null_ts)

        # Filter timestamp range
        ts_range = preprocessing.get("timestamp_range")
        if ts_range and drop_null_ts and drop_null_ts in df.columns:
            ts_min = pd.Timestamp(ts_range[0])
            ts_max = pd.Timestamp(ts_range[1])
            valid = (df[drop_null_ts] >= ts_min) & (df[drop_null_ts] <= ts_max)
            filtered = (~valid).sum()
            df = df[valid].copy()
            if filtered:
                logger.info("Filtered %d rows outside [%s, %s]", filtered, ts_min, ts_max)

        # Non-negative columns
        for col in preprocessing.get("non_negative_columns", []):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                df = df[df[col] >= 0].copy()

        # Outlier capping
        cap_pct = preprocessing.get("outlier_cap_percentile")
        if cap_pct:
            for col in preprocessing.get("non_negative_columns", []):
                if col in df.columns:
                    cap = df[col].quantile(cap_pct)
                    if cap > 0 and not np.isnan(cap):
                        df[col] = df[col].clip(upper=cap)

        # Median fill
        for col in preprocessing.get("median_fill", []):
            if col in df.columns:
                med = df[col].median()
                if pd.isna(med):
                    med = 0
                df[col] = df[col].fillna(med)

        # Zero fill
        for col in preprocessing.get("zero_fill", []):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Remove cancellations (e.g., Online Retail II)
        cancel_prefix = preprocessing.get("remove_cancellation_prefix")
        invoice_col = preprocessing.get("cancellation_column", "Invoice")
        if cancel_prefix and invoice_col in df.columns:
            df[invoice_col] = df[invoice_col].astype(str).str.strip()
            n_cancel = df[invoice_col].str.startswith(cancel_prefix).sum()
            if n_cancel:
                df = df[~df[invoice_col].str.startswith(cancel_prefix)].copy()
                logger.info("Removed %d cancellation invoices", n_cancel)

        # Min quantity/price filters
        min_qty = preprocessing.get("min_quantity")
        if min_qty is not None and "Quantity" in df.columns:
            df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
            df = df[df["Quantity"] >= min_qty].copy()

        min_price = preprocessing.get("min_price")
        if min_price is not None and "Price" in df.columns:
            df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)
            df = df[df["Price"] >= min_price].copy()

        # Timestamp column conversions for non-standard formats
        timestamp_unit = preprocessing.get("timestamp_unit")
        timestamp_col = preprocessing.get("timestamp_column_for_unit")
        if timestamp_unit and timestamp_col and timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(
                df[timestamp_col], unit=timestamp_unit, errors="coerce",
            )

        # Handle customer ID cleaning
        customer_col_candidates = ["customerID", "Customer ID", "customer_id"]
        for col in customer_col_candidates:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df = df[df[col] != ""].copy()
                break

        # Handle TotalCharges (Telco-specific but generic)
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"], errors="coerce"
            ).fillna(0)

        # Handle Churn column (Telco-specific but generic)
        churn_col = None
        for candidate in ["Churn", "churn", "churned"]:
            if candidate in df.columns:
                col_series = df[candidate]
                # Handle both object and StringDtype (pandas 3.x)
                if col_series.dtype == object or str(col_series.dtype) == "string":
                    mapped = col_series.map(
                        {"Yes": 1, "No": 0, "yes": 1, "no": 0}
                    )
                    df[candidate] = pd.to_numeric(mapped, errors="coerce").fillna(0).astype(int)
                    churn_col = candidate
                    break
                elif pd.api.types.is_numeric_dtype(col_series):
                    df[candidate] = col_series.fillna(0).astype(int)
                    churn_col = candidate
                    break

        # Handle SeniorCitizen (Telco-specific)
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = pd.to_numeric(
                df["SeniorCitizen"], errors="coerce"
            ).fillna(0).astype(int)

        # Drop rows with null customer ID
        for col in ["customer_id", "user_id", "visitorid", "customerID", "Customer ID"]:
            if col in df.columns:
                df = df.dropna(subset=[col])
                break

        return df

    # ── Schema standardization ──────────────────────────────────────

    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply manifest-defined column mapping and synthetic columns."""
        self._ensure_plugin()

        if self._plugin and hasattr(self._plugin, "standardize_schema"):
            return self._plugin.standardize_schema(df, self._manifest)

        df = df.copy()
        schema = self._manifest.get("schema", {})

        # Apply column mapping
        column_mapping = schema.get("column_mapping", {})
        if column_mapping:
            df = df.rename(columns=column_mapping, errors="ignore")

        # Apply event type mapping
        event_type_mapping = schema.get("event_type_mapping", {})
        if event_type_mapping and "event" in df.columns:
            df["event_type"] = df["event"].map(event_type_mapping).fillna("other")
        elif event_type_mapping and "event_type" in df.columns:
            df["event_type"] = df["event_type"].map(event_type_mapping).fillna("other")

        # Apply computed columns
        computed = schema.get("computed_columns", {})
        for target_col, expression in computed.items():
            try:
                # Simple expression evaluator for patterns like "Quantity * Price"
                if " * " in expression:
                    parts = expression.split(" * ")
                    col_a = parts[0].strip().strip("'\"")
                    col_b = parts[1].strip().strip("'\"")
                    if col_a in df.columns and col_b in df.columns:
                        df[target_col] = (
                            pd.to_numeric(df[col_a], errors="coerce").fillna(0)
                            * pd.to_numeric(df[col_b], errors="coerce").fillna(0)
                        )
                elif " + " in expression:
                    parts = expression.split(" + ")
                    col_a = parts[0].strip().strip("'\"")
                    col_b = parts[1].strip().strip("'\"")
                    if col_a in df.columns and col_b in df.columns:
                        df[target_col] = (
                            pd.to_numeric(df[col_a], errors="coerce").fillna(0)
                            + pd.to_numeric(df[col_b], errors="coerce").fillna(0)
                        )
            except Exception as exc:
                logger.warning("Computed column '%s' failed: %s", target_col, exc)

        # Apply synthetic columns
        synthetic = schema.get("synthetic_columns", {})
        for col, value in synthetic.items():
            if col not in df.columns:
                df[col] = value

        # Build synthetic event_time from tenure (Telco-style)
        if "event_time" not in df.columns and "tenure" in df.columns:
            tenure_months = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
            df["event_time"] = pd.Timestamp("2019-01-31") - pd.to_timedelta(
                tenure_months * 30, unit="D"
            )

        # Build synthetic event_time from days_since_prior_order (Instacart-style)
        if "event_time" not in df.columns:
            preprocessing = self._manifest.get("preprocessing", {})
            if "days_since_prior_order" in df.columns and "order_number" in df.columns:
                epoch = pd.Timestamp(
                    preprocessing.get("synthetic_timestamp_epoch", "2017-03-21")
                )
                df["days_since_prior_order"] = (
                    pd.to_numeric(df["days_since_prior_order"], errors="coerce").fillna(0)
                )
                df["days_from_end"] = (
                    df.groupby("user_id")["days_since_prior_order"]
                    .cumsum().fillna(0)
                )
                df["days_from_end"] = df.groupby("user_id")["days_from_end"].transform(
                    lambda x: x.max() - x
                )
                df["event_time"] = epoch - pd.to_timedelta(
                    df["days_from_end"], unit="D"
                )
            else:
                df["event_time"] = pd.Timestamp.now()

        # Ensure required canonical columns exist with defaults
        defaults = {
            "review_score": 0.0,
            "payment_type": "unknown",
            "delivery_delay": 0.0,
            "session_id": "unknown",
        }
        for col, default_val in defaults.items():
            if col not in df.columns:
                df[col] = default_val

        # If transaction_value is still missing, set to 0
        if "transaction_value" not in df.columns:
            df["transaction_value"] = 0.0

        logger.info("Standardized schema — columns: %s", list(df.columns))
        return df


def load_manifest(dataset_name: str) -> Dict[str, Any]:
    """Load a manifest YAML by dataset name.

    Looks in configs/datasets/{name}.yaml.

    Parameters
    ----------
    dataset_name : str
        Dataset name (e.g. 'olist', 'my_custom_dataset').

    Returns
    -------
    dict with manifest contents.

    Raises
    ------
    FileNotFoundError if manifest not found.
    """
    from src.config import get_configs_dir

    # Try standard configs location
    configs_dir = get_configs_dir()
    manifest_path = configs_dir / "datasets" / f"{dataset_name}.yaml"
    if manifest_path.exists():
        import yaml
        with open(manifest_path) as f:
            return yaml.safe_load(f) or {}

    # Try .dataset_registry for user-registered datasets
    try:
        from src.config import PROJECT_ROOT
        registry_path = Path(PROJECT_ROOT) / ".dataset_registry" / "manifests" / f"{dataset_name}.yaml"
        if registry_path.exists():
            import yaml
            with open(registry_path) as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass

    raise FileNotFoundError(
        f"No manifest found for dataset '{dataset_name}'. "
        f"Searched: {manifest_path}"
    )


def get_dataset_from_manifest(
    dataset_name: str,
    data_dir: Optional[str] = None,
) -> GenericDatasetAdapter:
    """Create a GenericDatasetAdapter from a manifest.

    Parameters
    ----------
    dataset_name : str
        Dataset name.
    data_dir : str, optional
        Explicit data directory.

    Returns
    -------
    GenericDatasetAdapter with data_dir configured.
    """
    manifest = load_manifest(dataset_name)
    adapter = GenericDatasetAdapter(manifest_dict=manifest)

    if data_dir is not None:
        adapter.data_dir = data_dir
    elif manifest.get("root_directory"):
        adapter.data_dir = manifest["root_directory"]

    return adapter

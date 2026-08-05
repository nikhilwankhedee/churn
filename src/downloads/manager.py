"""
Download Manager: list and obtain benchmark datasets.

Provides information about supported benchmark datasets and
guides users through obtaining them.
"""
import dataclasses
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class DatasetInfo:
    """Information about a benchmark dataset."""
    name: str
    description: str
    ecosystem_type: str
    n_customers: Optional[int] = None
    n_transactions: Optional[int] = None
    time_range: str = ""
    license: str = ""
    url: str = ""
    citation: str = ""
    available: bool = True
    requires_credentials: bool = False
    setup_instructions: str = ""


BENCHMARK_DATASETS: Dict[str, DatasetInfo] = {
    "olist": DatasetInfo(
        name="olist",
        description="Brazilian e-commerce public dataset by Olist",
        ecosystem_type="transactional_marketplace",
        n_customers=99441,
        n_transactions=100000,
        time_range="2016-09 to 2018-08",
        license="CC BY-NC-SA 4.0",
        url="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
        citation="Olist. Brazilian E-Commerce Public Dataset by Olist. Kaggle, 2018.",
        available=True,
    ),
    "rees46": DatasetInfo(
        name="rees46",
        description="E-commerce behavioral data from REES46",
        ecosystem_type="transactional_marketplace",
        n_customers=700000,
        n_transactions=50000000,
        time_range="2019-10 to 2020-01",
        license="Research use",
        url="https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store",
        citation="REES46. E-commerce behavior data from multi-category store. Kaggle, 2020.",
        available=True,
    ),
    "retailrocket": DatasetInfo(
        name="retailrocket",
        description="RetailRocket recommender system dataset",
        ecosystem_type="clickstream_commerce",
        n_customers=14000,
        n_transactions=2000000,
        time_range="2015-06 to 2016-06",
        license="CC BY-NC-SA 4.0",
        url="https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset",
        citation="RetailRocket. E-commerce Dataset. Kaggle, 2016.",
        available=True,
    ),
    "online_retail_ii": DatasetInfo(
        name="online_retail_ii",
        description="UCI Online Retail II dataset",
        ecosystem_type="habitual_retail",
        n_customers=4372,
        n_transactions=1067371,
        time_range="2009-12 to 2011-12",
        license="CC BY 4.0",
        url="https://archive.ics.uci.edu/ml/datasets/online+retail+ii",
        citation="Chen, D. et al. Online Retail II Data Set. UCI Machine Learning Repository, 2015.",
        available=True,
    ),
    "instacart": DatasetInfo(
        name="instacart",
        description="Instacart Market Basket Analysis",
        ecosystem_type="habitual_retail",
        n_customers=206209,
        n_transactions=3400000,
        time_range="2017 (anonymized)",
        license="CC0 1.0",
        url="https://www.kaggle.com/c/instacart-market-basket-analysis/data",
        citation="Instacart. Market Basket Analysis. Kaggle, 2017.",
        available=True,
    ),
    "telco": DatasetInfo(
        name="telco",
        description="IBM Telco Customer Churn dataset",
        ecosystem_type="subscription",
        n_customers=7043,
        n_transactions=7043,
        time_range="Cross-sectional",
        license="CC0 1.0",
        url="https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
        citation="IBM. Telco Customer Churn. IBM Sample Data, 2018.",
        available=True,
    ),
}


def list_datasets() -> List[DatasetInfo]:
    """List all supported benchmark datasets."""
    return list(BENCHMARK_DATASETS.values())


def get_dataset_info(name: str) -> Optional[DatasetInfo]:
    """Get information about a specific benchmark dataset."""
    return BENCHMARK_DATASETS.get(name.lower().strip())


def get_download_instructions(name: str) -> str:
    """Get download instructions for a dataset."""
    info = get_dataset_info(name)
    if info is None:
        return f"Unknown dataset: {name}"

    lines = [
        f"Dataset: {info.name}",
        f"Description: {info.description}",
        f"License: {info.license}",
        f"URL: {info.url}",
        "",
        "To use this dataset with ChurnLab:",
        f"  1. Download from: {info.url}",
        f"  2. Extract to your data directory",
        f"  3. Run: churn register <path_to_data>",
        f"  4. Run: churn benchmark {info.name}",
    ]

    if info.requires_credentials:
        lines.append("")
        lines.append("Note: This dataset requires credentials or registration.")
        lines.append(info.setup_instructions)

    return "\n".join(lines)

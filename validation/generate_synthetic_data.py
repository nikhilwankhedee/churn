#!/usr/bin/env python3
"""
Generate synthetic datasets that match the expected schemas for all 6 built-in
adapters plus 3 unknown datasets. Used for local validation when real data
is not available.

Each generator produces CSVs with realistic distributions, sufficient volume,
and correct column names to exercise the full pipeline.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RANDOM_SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

np.random.seed(RANDOM_SEED)


def _date_range(start, periods, freq="D"):
    return pd.date_range(start=start, periods=periods, freq=freq)


# ═══════════════════════════════════════════════════════════════
# 1. OLIST
# ═══════════════════════════════════════════════════════════════
def generate_olist(out_dir: Path):
    n_customers = 5000
    n_orders = 15000
    n_items = 20000
    n_reviews = 8000

    customer_ids = [f"cust_{i:05d}" for i in range(n_customers)]
    customer_unique = [f"cuniq_{i:05d}" for i in range(n_customers)]
    order_ids = [f"ord_{i:06d}" for i in range(n_orders)]

    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "customer_unique_id": customer_unique,
        "customer_zip_code_prefix": np.random.randint(10000, 99999, n_customers),
        "customer_city": np.random.choice(["sao paulo", "rio de janeiro", "belo horizonte"], n_customers),
        "customer_state": np.random.choice(["SP", "RJ", "MG"], n_customers),
    })

    base_date = datetime(2017, 1, 1)
    order_dates = [base_date + timedelta(days=int(d)) for d in np.random.uniform(0, 730, n_orders)]
    orders = pd.DataFrame({
        "order_id": order_ids,
        "customer_id": np.random.choice(customer_ids, n_orders),
        "order_status": np.random.choice(["delivered", "shipped", "canceled"], n_orders, p=[0.85, 0.10, 0.05]),
        "order_purchase_timestamp": order_dates,
        "order_approved_at": [d + timedelta(hours=np.random.uniform(0, 24)) for d in order_dates],
        "order_delivered_carrier_date": [d + timedelta(days=np.random.uniform(1, 5)) for d in order_dates],
        "order_delivered_customer_date": [d + timedelta(days=np.random.uniform(5, 30)) for d in order_dates],
        "order_estimated_delivery_date": [d + timedelta(days=15) for d in order_dates],
    })

    product_ids = [f"prod_{i:04d}" for i in range(500)]
    items = pd.DataFrame({
        "order_id": np.random.choice(order_ids, n_items),
        "order_item_id": np.random.randint(1, 5, n_items),
        "product_id": np.random.choice(product_ids, n_items),
        "seller_id": [f"sell_{i:03d}" for i in np.random.randint(0, 100, n_items)],
        "shipping_limit_date": [datetime(2017, 1, 1) + timedelta(days=int(d)) for d in np.random.uniform(0, 730, n_items)],
        "price": np.round(np.random.lognormal(3.5, 1.0, n_items), 2),
        "freight_value": np.round(np.random.lognormal(2.0, 0.8, n_items), 2),
    })

    review_order_ids = np.random.choice(order_ids, n_reviews)
    reviews = pd.DataFrame({
        "review_id": [f"rev_{i:05d}" for i in range(n_reviews)],
        "order_id": review_order_ids,
        "review_score": np.random.choice([1, 2, 3, 4, 5], n_reviews, p=[0.05, 0.08, 0.15, 0.30, 0.42]),
        "review_comment_title": np.random.choice(["great", "ok", "bad", None], n_reviews, p=[0.2, 0.3, 0.1, 0.4]),
        "review_comment_message": np.random.choice(["good product", "fast shipping", "bad quality", None], n_reviews, p=[0.3, 0.2, 0.1, 0.4]),
        "review_creation_date": [datetime(2017, 1, 1) + timedelta(days=int(d)) for d in np.random.uniform(0, 730, n_reviews)],
        "review_answer_timestamp": [datetime(2017, 1, 1) + timedelta(days=int(d) + 1) for d in np.random.uniform(0, 730, n_reviews)],
    })

    n_payments = n_orders
    payments = pd.DataFrame({
        "order_id": np.random.choice(order_ids, n_payments),
        "payment_sequential": np.random.randint(1, 4, n_payments),
        "payment_type": np.random.choice(["credit_card", "boleto", "voucher", "debit_card"], n_payments, p=[0.65, 0.20, 0.10, 0.05]),
        "payment_installments": np.random.choice([1, 2, 3, 4, 6, 10, 12], n_payments),
        "payment_value": np.round(np.random.lognormal(4.0, 1.2, n_payments), 2),
    })

    products = pd.DataFrame({
        "product_id": product_ids,
        "product_category_name": np.random.choice(["electronics", "furniture", "clothing", "toys", "sports"], 500),
        "product_name_lenght": np.random.randint(10, 80, 500),
        "product_description_lenght": np.random.randint(50, 2000, 500),
        "product_photos_qty": np.random.randint(1, 10, 500),
        "product_weight_g": np.random.lognormal(6.0, 1.5, 500).astype(int),
        "product_length_cm": np.random.randint(10, 100, 500),
        "product_height_cm": np.random.randint(5, 60, 500),
        "product_width_cm": np.random.randint(5, 60, 500),
    })

    sellers = pd.DataFrame({
        "seller_id": [f"sell_{i:03d}" for i in range(100)],
        "seller_zip_code_prefix": np.random.randint(10000, 99999, 100),
        "seller_city": np.random.choice(["sao paulo", "curitiba", "belo horizonte"], 100),
        "seller_state": np.random.choice(["SP", "PR", "MG"], 100),
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    customers.to_csv(out_dir / "olist_customers_dataset.csv", index=False)
    orders.to_csv(out_dir / "olist_orders_dataset.csv", index=False)
    items.to_csv(out_dir / "olist_order_items_dataset.csv", index=False)
    reviews.to_csv(out_dir / "olist_order_reviews_dataset.csv", index=False)
    payments.to_csv(out_dir / "olist_order_payments_dataset.csv", index=False)
    products.to_csv(out_dir / "olist_products_dataset.csv", index=False)
    sellers.to_csv(out_dir / "olist_sellers_dataset.csv", index=False)
    print(f"  Olist: {n_orders} orders, {n_customers} customers -> {out_dir}")


# ═══════════════════════════════════════════════════════════════
# 2. TELCO
# ═══════════════════════════════════════════════════════════════
def generate_telco(out_dir: Path):
    n = 3000
    tenure = np.random.randint(0, 73, n)
    monthly = np.round(np.random.uniform(20, 120, n), 2)
    total = np.round(monthly * tenure + np.random.normal(0, 50, n), 2)

    df = pd.DataFrame({
        "customerID": [f"CUST-{i:05d}" for i in range(n)],
        "gender": np.random.choice(["Male", "Female"], n),
        "SeniorCitizen": np.random.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": np.random.choice(["Yes", "No"], n),
        "Dependents": np.random.choice(["Yes", "No"], n, p=[0.30, 0.70]),
        "tenure": tenure,
        "PhoneService": np.random.choice(["Yes", "No"], n, p=[0.90, 0.10]),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n, p=[0.60, 0.40]),
        "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n),
        "MonthlyCharges": monthly,
        "TotalCharges": total.astype(str),
        "Churn": np.random.choice(["Yes", "No"], n, p=[0.27, 0.73]),
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "telco_customer_churn.csv", index=False)
    print(f"  Telco: {n} customers -> {out_dir}")


# ═══════════════════════════════════════════════════════════════
# 3. RETAILROCKET
# ═══════════════════════════════════════════════════════════════
def generate_retailrocket(out_dir: Path):
    n_events = 20000
    n_items = 500
    n_visitors = 3000

    base_ts = 1509494400000  # 2017-11-01 in milliseconds epoch
    events = pd.DataFrame({
        "timestamp": [int(base_ts + s * 1000) for s in np.random.uniform(0, 90 * 86400, n_events)],
        "visitorid": [f"v_{v}" for v in np.random.randint(1000, 1000 + n_visitors, n_events)],
        "event": np.random.choice(["view", "addtocart", "transaction"], n_events, p=[0.70, 0.15, 0.15]),
        "itemid": [f"i_{i}" for i in np.random.randint(0, n_items, n_events)],
        "transactionid": [str(np.random.randint(100000, 999999)) if e == "transaction" else "" for e in
                          np.random.choice(["view", "addtocart", "transaction"], n_events, p=[0.70, 0.15, 0.15])],
    })

    items = pd.DataFrame({
        "itemid": [f"i_{i}" for i in range(n_items)],
        "property": np.random.choice(["color", "size", "brand", "material"], n_items),
        "value": [f"val_{i}" for i in range(n_items)],
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_dir / "retailrocket_events.csv", index=False)
    items.to_csv(out_dir / "retailrocket_items.csv", index=False)
    print(f"  RetailRocket: {n_events} events, {n_visitors} visitors -> {out_dir}")


# ═══════════════════════════════════════════════════════════════
# 4. REES46
# ═══════════════════════════════════════════════════════════════
def generate_rees46(out_dir: Path):
    n_events = 25000
    n_users = 4000
    n_items = 800

    base_ts = 1546300800  # 2019-01-01 in seconds epoch
    events = pd.DataFrame({
        "timestamp": [int(base_ts + s) for s in np.random.uniform(0, 180 * 86400, n_events)],
        "user_id": [f"u_{u}" for u in np.random.randint(1000, 1000 + n_users, n_events)],
        "event_type": np.random.choice(["view", "cart", "purchase", "remove_from_cart"], n_events, p=[0.60, 0.15, 0.15, 0.10]),
        "item_id": [f"item_{i}" for i in np.random.randint(0, n_items, n_events)],
        "category_id": np.random.randint(0, 50, n_events),
        "price": np.round(np.random.lognormal(3.5, 1.0, n_events), 2),
    })

    users = pd.DataFrame({
        "user_id": [f"u_{u}" for u in range(1000, 1000 + n_users)],
        "age": np.random.randint(18, 70, n_users),
        "gender": np.random.choice(["M", "F"], n_users),
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_dir / "rees46_events.csv", index=False)
    users.to_csv(out_dir / "rees46_users.csv", index=False)
    print(f"  REES46: {n_events} events, {n_users} users -> {out_dir}")


# ═══════════════════════════════════════════════════════════════
# 5. INSTACART
# ═══════════════════════════════════════════════════════════════
def generate_instacart(out_dir: Path):
    n_orders = 10000
    n_products = 500
    n_users = 2000

    user_ids = np.random.randint(0, n_users, n_orders)
    order_numbers = np.random.randint(1, 20, n_orders)
    days_since = np.round(np.random.uniform(1, 30, n_orders), 1)
    # First orders (order_number == 1) must have NaN for days_since_prior_order
    days_since[order_numbers == 1] = np.nan

    orders = pd.DataFrame({
        "order_id": range(1, n_orders + 1),
        "user_id": [str(u) for u in user_ids],
        "eval_set": np.random.choice(["prior", "train", "test"], n_orders, p=[0.70, 0.15, 0.15]),
        "order_number": order_numbers,
        "order_dow": np.random.randint(0, 7, n_orders),
        "order_hour_of_day": np.random.randint(6, 23, n_orders),
        "days_since_prior_order": days_since,
    })

    products = pd.DataFrame({
        "product_id": range(n_products),
        "product_name": [f"product_{i}" for i in range(n_products)],
        "aisle_id": np.random.randint(0, 50, n_products),
        "department_id": np.random.randint(0, 10, n_products),
    })

    # Generate order_products__prior for engagement features
    n_prior = 30000
    order_products = pd.DataFrame({
        "order_id": np.random.choice(range(1, n_orders + 1), n_prior),
        "product_id": np.random.randint(0, n_products, n_prior),
        "add_to_cart_order": np.random.randint(1, 10, n_prior),
        "reordered": np.random.choice([0, 1], n_prior, p=[0.40, 0.60]),
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    orders.to_csv(out_dir / "instacart_orders.csv", index=False)
    products.to_csv(out_dir / "instacart_products.csv", index=False)
    order_products.to_csv(out_dir / "instacart_order_products__prior.csv", index=False)
    print(f"  Instacart: {n_orders} orders, {n_users} users -> {out_dir}")


# ═══════════════════════════════════════════════════════════════
# 6. ONLINE RETAIL II
# ═══════════════════════════════════════════════════════════════
def generate_online_retail_ii(out_dir: Path):
    n_per_file = 10000

    def _make_df(n, year_start):
        base_date = datetime(year_start, 12, 1)
        return pd.DataFrame({
            "Invoice": [f"{100000 + i}" for i in range(n)],
            "StockCode": [f"SC{i:05d}" for i in np.random.randint(0, 500, n)],
            "Description": np.random.choice(["widget", "gadget", "doohickey", "thingamajig"], n),
            "Quantity": np.random.randint(1, 50, n),
            "InvoiceDate": [(base_date + timedelta(days=int(d))).strftime("%Y-%m-%d %H:%M:%S")
                           for d in np.random.uniform(0, 365, n)],
            "UnitPrice": np.round(np.random.uniform(0.5, 100.0, n), 2),
            "Customer ID": np.random.randint(10000, 15000, n),
            "Country": np.random.choice(["United Kingdom", "Germany", "France", "EIRE"], n, p=[0.50, 0.20, 0.15, 0.15]),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    _make_df(n_per_file, 2009).to_csv(out_dir / "online_retail_II_2009_2010.csv", index=False)
    _make_df(n_per_file, 2010).to_csv(out_dir / "online_retail_II_2010_2011.csv", index=False)
    print(f"  Online Retail II: {n_per_file * 2} transactions -> {out_dir}")


# ═══════════════════════════════════════════════════════════════
# 7. UNKNOWN DATASETS (for Phase 4)
# ═══════════════════════════════════════════════════════════════

def generate_bank_marketing(out_dir: Path):
    """UCI Bank Marketing dataset — subscription-like with campaign contacts."""
    n = 4000
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, n),
        "job": np.random.choice(["admin.", "technician", "services", "management", "retired", "blue-collar"], n),
        "marital": np.random.choice(["married", "single", "divorced"], n, p=[0.60, 0.28, 0.12]),
        "education": np.random.choice(["primary", "secondary", "tertiary", "unknown"], n, p=[0.30, 0.40, 0.20, 0.10]),
        "default": np.random.choice(["yes", "no"], n, p=[0.02, 0.98]),
        "balance": np.round(np.random.lognormal(7.0, 1.5, n), 0).astype(int),
        "housing": np.random.choice(["yes", "no"], n, p=[0.55, 0.45]),
        "loan": np.random.choice(["yes", "no"], n, p=[0.15, 0.85]),
        "contact": np.random.choice(["cellular", "telephone", "unknown"], n, p=[0.65, 0.10, 0.25]),
        "day": np.random.randint(1, 31, n),
        "month": np.random.choice(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], n),
        "duration": np.random.randint(0, 3000, n),
        "campaign": np.random.randint(1, 10, n),
        "pdays": np.random.choice([-1] + list(range(1, 401)), n, p=[0.70] + [0.30 / 400] * 400),
        "previous": np.random.randint(0, 10, n),
        "poutcome": np.random.choice(["unknown", "failure", "success", "other"], n, p=[0.75, 0.15, 0.05, 0.05]),
        "y": np.random.choice(["yes", "no"], n, p=[0.12, 0.88]),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "bank_marketing.csv", index=False)
    print(f"  Bank Marketing: {n} contacts -> {out_dir}")


def generate_online_shoppers(out_dir: Path):
    """UCI Online Shoppers Intention — e-commerce session data."""
    n = 4000
    base_date = datetime(2018, 1, 1)
    df = pd.DataFrame({
        "customer_id": np.random.randint(1000, 5000, n),
        "event_time": [(base_date + timedelta(days=int(d))).strftime("%Y-%m-%d %H:%M:%S")
                       for d in np.random.uniform(0, 365, n)],
        "transaction_value": np.round(np.random.lognormal(3.0, 1.5, n), 2),
        "event_type": np.random.choice(["pageview", "transaction", "cart"], n, p=[0.70, 0.15, 0.15]),
        "product_specialist": np.random.randint(0, 20, n),
        "product_related_duration": np.round(np.random.exponential(200, n), 1),
        "bounce_rates": np.round(np.random.beta(2, 10, n), 4),
        "exit_rates": np.round(np.random.beta(2, 8, n), 4),
        "page_values": np.round(np.random.exponential(5, n), 2),
        "special_day": np.round(np.random.choice([0, 0.2, 0.5, 0.8, 1.0], n, p=[0.80, 0.05, 0.05, 0.05, 0.05]), 1),
        "operating_systems": np.random.choice([1, 2, 3, 4], n, p=[0.40, 0.35, 0.15, 0.10]),
        "browser": np.random.choice([1, 2, 3, 4, 5], n, p=[0.45, 0.30, 0.10, 0.10, 0.05]),
        "region": np.random.randint(1, 10, n),
        "traffic_type": np.random.randint(1, 20, n),
        "visitor_type": np.random.choice(["Returning_Visitor", "New_Visitor", "Other"], n, p=[0.70, 0.25, 0.05]),
        "weekend": np.random.choice([True, False], n, p=[0.23, 0.77]),
        "revenue": np.random.choice([True, False], n, p=[0.15, 0.85]),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "online_shoppers.csv", index=False)
    print(f"  Online Shoppers: {n} sessions -> {out_dir}")


def generate_ecommerce_brazil(out_dir: Path):
    """Brazilian e-commerce — transactional, different domain from Olist."""
    n_customers = 2000
    n_orders = 6000

    base_date = datetime(2018, 1, 1)
    customers = pd.DataFrame({
        "customer_id": [f"EC{i:05d}" for i in range(n_customers)],
        "customer_unique_id": [f"ECU{i:05d}" for i in range(n_customers)],
        "customer_city": np.random.choice(["sao paulo", "campinas", "rio de janeiro", "salvador"], n_customers),
        "customer_state": np.random.choice(["SP", "RJ", "BA", "MG"], n_customers),
    })

    orders = pd.DataFrame({
        "order_id": [f"EO{i:06d}" for i in range(n_orders)],
        "customer_id": np.random.choice([f"EC{i:05d}" for i in range(n_customers)], n_orders),
        "order_status": np.random.choice(["delivered", "shipped", "canceled"], n_orders, p=[0.88, 0.08, 0.04]),
        "order_purchase_timestamp": [(base_date + timedelta(days=int(d))).strftime("%Y-%m-%d %H:%M:%S")
                                     for d in np.random.uniform(0, 365, n_orders)],
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    customers.to_csv(out_dir / "ecommerce_brazil_customers.csv", index=False)
    orders.to_csv(out_dir / "ecommerce_brazil_orders.csv", index=False)
    print(f"  E-commerce Brazil: {n_orders} orders, {n_customers} customers -> {out_dir}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("Generating synthetic datasets...")
    print()

    builtin_dir = DATA_DIR / "builtin"
    unknown_dir = DATA_DIR / "unknown"

    print("Built-in datasets:")
    generate_olist(builtin_dir)
    generate_telco(builtin_dir)
    generate_retailrocket(builtin_dir)
    generate_rees46(builtin_dir)
    generate_instacart(builtin_dir)
    generate_online_retail_ii(builtin_dir)

    print()
    print("Unknown datasets:")
    generate_bank_marketing(unknown_dir / "bank_marketing")
    generate_online_shoppers(unknown_dir / "online_shoppers")
    generate_ecommerce_brazil(unknown_dir / "ecommerce_brazil")

    print()
    print(f"All synthetic data generated in {DATA_DIR}")


if __name__ == "__main__":
    main()

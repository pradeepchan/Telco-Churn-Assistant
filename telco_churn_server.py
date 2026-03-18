from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

mcp = FastMCP("telco-churn-server")

def load_data() -> pd.DataFrame:
    csv_path = "./data/telco-customer-churn.csv"
    df = pd.read_csv(csv_path)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.dropna(subset=["TotalCharges"]).copy()
    return df


def build_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)

    x = df.drop(columns=["customerID", "Churn"]).copy()

    numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")

    for col in x.columns:
        if pd.api.types.is_numeric_dtype(x[col]):
            x[col] = x[col].fillna(x[col].median())

    x = pd.get_dummies(x, drop_first=True)
    return x, y


def train_model() -> dict[str, Any]:
    model_path = ARTIFACT_DIR / "churn_model.joblib"
    columns_path = ARTIFACT_DIR / "training_columns.json"

    if model_path.exists() and columns_path.exists():
        model = joblib.load(model_path)
        training_columns = json.loads(columns_path.read_text())
        return {"model": model, "training_columns": training_columns}

    from sklearn.linear_model import LogisticRegression

    df = load_data()
    x, y = build_training_frame(df)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x, y)

    joblib.dump(model, model_path)
    columns_path.write_text(json.dumps(list(x.columns), indent=2))

    return {"model": model, "training_columns": list(x.columns)}


def get_raw_customer_row(customer_id: str) -> dict[str, Any]:
    df = load_data()
    match = df.loc[df["customerID"] == customer_id]
    if match.empty:
        raise ValueError(f"Customer ID {customer_id!r} not found.")

    row = match.iloc[0].to_dict()

    normalized = {}
    for k, v in row.items():
        if pd.isna(v):
            normalized[k] = None
        elif isinstance(v, (np.integer,)):
            normalized[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            normalized[k] = float(v)
        else:
            normalized[k] = v

    return normalized


def prepare_single_customer_features(customer_id: str) -> pd.DataFrame:
    raw = get_raw_customer_row(customer_id)
    row_df = pd.DataFrame([raw]).drop(columns=["customerID", "Churn"])

    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        if col in row_df.columns:
            row_df[col] = pd.to_numeric(row_df[col], errors="coerce")

    for col in row_df.columns:
        if pd.api.types.is_numeric_dtype(row_df[col]):
            row_df[col] = row_df[col].fillna(row_df[col].median())

    row_df = pd.get_dummies(row_df, drop_first=True)

    model_bundle = train_model()
    training_columns = model_bundle["training_columns"]

    for col in training_columns:
        if col not in row_df.columns:
            row_df[col] = 0

    row_df = row_df[training_columns]
    return row_df


def probability_to_segment(probability: float) -> str:
    if probability >= 0.75:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


@mcp.tool()
def dataset_overview() -> str:
    """
    Return dataset shape, target balance and sample columns.
    """
    df = load_data()
    churn_rate = float((df["Churn"].astype(str).str.lower() == "yes").mean())

    payload = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "churn_rate": round(churn_rate, 4),
        "sample_columns": df.columns[:10].tolist(),
        "model_type": "LogisticRegression",
    }
    return json.dumps(payload, indent=2)


@mcp.tool()
def list_customer_ids(limit: int = 10) -> str:
    """
    Return a few customer IDs for testing.
    """
    df = load_data()
    ids = df["customerID"].head(limit).tolist()
    return json.dumps({"customer_ids": ids}, indent=2)


@mcp.tool()
def get_customer_profile(customer_id: str) -> str:
    """
    Return the raw CSV row for a customer.
    """
    raw = get_raw_customer_row(customer_id)
    return json.dumps(raw, indent=2, default=str)


@mcp.tool()
def predict_churn(customer_id: str) -> str:
    """
    Predict churn probability using a trained logistic regression model.
    """
    model_bundle = train_model()
    model = model_bundle["model"]

    x = prepare_single_customer_features(customer_id)
    prob = float(model.predict_proba(x)[0, 1])
    pred = int(prob >= 0.5)
    segment = probability_to_segment(prob)

    payload = {
        "customer_id": customer_id,
        "predicted_label": pred,
        "predicted_churn_probability": round(prob, 4),
        "risk_segment": segment,
        "model_type": "LogisticRegression",
    }
    return json.dumps(payload, indent=2)


@mcp.tool()
def get_retention_offers(risk_segment: str) -> str:
    """
    Return retention offers for Low, Medium, or High risk.
    """
    offers = {
        "High": [
            "20% discount for 6 months",
            "Free tech support for 3 months",
            "Annual contract upgrade incentive",
        ],
        "Medium": [
            "10% discount for 3 months",
            "Bundle upgrade offer",
            "Loyalty points campaign",
        ],
        "Low": [
            "Standard loyalty communication",
            "Periodic engagement reminder",
            "No aggressive incentive required",
        ],
    }

    payload = {
        "risk_segment": risk_segment,
        "offers": offers.get(risk_segment, offers["Low"]),
    }
    return json.dumps(payload, indent=2)


if __name__ == "__main__":
    # Uses stdio transport for easy local integration with the Agents SDK.
    mcp.run()
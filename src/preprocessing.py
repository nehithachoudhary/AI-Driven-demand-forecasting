from pathlib import Path
import pandas as pd
from config import DATA_PATH, EXPECTED_COLUMNS, PROCESSED_DATA_PATH, ensure_directories


def validate_dataset(df: pd.DataFrame) -> dict:
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    numeric_columns = ["Inventory Level", "Units Sold", "Units Ordered", "Price", "Discount", "Competitor Pricing", "Epidemic", "Demand"]
    invalid_numeric = {column: int(pd.to_numeric(df[column], errors="coerce").isna().sum()) for column in numeric_columns}
    invalid_numeric = {key: value for key, value in invalid_numeric.items() if value}
    report = {
        "rows": len(df), "columns": len(df.columns), "missing_values": df.isna().sum().sum(),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_keys": int(df.duplicated(["Date", "Store ID", "Product ID"]).sum()),
        "invalid_numeric_values": invalid_numeric,
        "date_min": str(df["Date"].min()), "date_max": str(df["Date"].max()),
        "stores": int(df["Store ID"].nunique()), "products": int(df["Product ID"].nunique()),
        "categories": int(df["Category"].nunique()), "regions": int(df["Region"].nunique()),
    }
    return report


def load_and_validate(path: Path = DATA_PATH) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    report = validate_dataset(df)
    if report["missing_values"] or report["duplicate_rows"] or report["duplicate_keys"] or report["invalid_numeric_values"]:
        raise ValueError(f"Dataset validation failed: {report}")
    numeric_columns = ["Inventory Level", "Units Sold", "Units Ordered", "Price", "Discount", "Competitor Pricing", "Epidemic", "Demand"]
    if (df[numeric_columns] < 0).any().any():
        raise ValueError("Negative values found in non-negative numeric fields")
    return df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True), report


def check_continuity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (store, product), group in df.groupby(["Store ID", "Product ID"]):
        dates = group["Date"].sort_values()
        expected = pd.date_range(dates.min(), dates.max(), freq="D")
        rows.append({"Store ID": store, "Product ID": product, "observations": len(dates), "expected_days": len(expected), "missing_days": len(expected) - len(dates)})
    return pd.DataFrame(rows)


def preprocess(save: bool = True) -> tuple[pd.DataFrame, dict]:
    df, report = load_and_validate()
    report["continuity"] = check_continuity(df).to_dict(orient="records")
    if save:
        ensure_directories()
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        pd.DataFrame([report]).drop(columns=["continuity"]).to_json(PROCESSED_DATA_PATH.with_name("data_quality.json"), orient="records", indent=2)
    return df, report
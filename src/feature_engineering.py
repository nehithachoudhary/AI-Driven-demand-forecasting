import pandas as pd

LAG_FEATURES = ["demand_lag_1", "demand_lag_7", "demand_lag_14", "demand_lag_30", "units_sold_lag_1", "units_sold_lag_7", "units_sold_lag_14", "units_sold_lag_30"]
ROLLING_FEATURES = ["demand_rolling_mean_7", "demand_rolling_mean_14", "demand_rolling_mean_30", "demand_rolling_std_7", "demand_rolling_std_14", "demand_rolling_std_30"]
FEATURE_COLUMNS = ["year", "month", "day", "day_of_week", "week_of_year", "quarter", "day_of_year"] + LAG_FEATURES + ROLLING_FEATURES


def add_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    result = df.sort_values(["Store ID", "Product ID", "Date"]).copy()
    result["year"] = result["Date"].dt.year
    result["month"] = result["Date"].dt.month
    result["day"] = result["Date"].dt.day
    result["day_of_week"] = result["Date"].dt.dayofweek
    result["week_of_year"] = result["Date"].dt.isocalendar().week.astype(int)
    result["quarter"] = result["Date"].dt.quarter
    result["day_of_year"] = result["Date"].dt.dayofyear
    groups = result.groupby(["Store ID", "Product ID"], sort=False)
    for lag in (1, 7, 14, 30):
        result[f"demand_lag_{lag}"] = groups["Demand"].shift(lag)
        result[f"units_sold_lag_{lag}"] = groups["Units Sold"].shift(lag)
    history = groups["Demand"].shift(1)
    for window in (7, 14, 30):
        result[f"demand_rolling_mean_{window}"] = history.groupby([result["Store ID"], result["Product ID"]]).transform(lambda values: values.rolling(window).mean())
        result[f"demand_rolling_std_{window}"] = history.groupby([result["Store ID"], result["Product ID"]]).transform(lambda values: values.rolling(window).std())
    if drop_na:
        result = result.dropna(subset=FEATURE_COLUMNS + ["Demand"]).reset_index(drop=True)
    return result
import json
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, RESULTS_DIR, TEST_DAYS, ensure_directories
from src.feature_engineering import add_features
from src.forecasting import choose_best, train_lstm, train_sarima, train_tree_models
from src.preprocessing import preprocess

warnings.filterwarnings("ignore")


def make_eda(df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    daily = df.groupby("Date", as_index=False)["Demand"].sum()
    sns.set_theme(style="whitegrid")
    plots = [
        (daily, "Date", "Demand", "Overall Demand Trend", "overall_demand_trend.png"),
        (df.assign(Month=df["Date"].dt.to_period("M")).groupby("Month", as_index=False)["Demand"].sum(), "Month", "Demand", "Monthly Demand Trend", "monthly_demand_trend.png"),
        (df, "Store ID", "Demand", "Demand by Store", "store_demand.png"),
        (df, "Category", "Demand", "Demand by Category", "category_demand.png"),
    ]
    for frame, x, y, title, name in plots:
        plt.figure(figsize=(10, 5))
        if x in {"Store ID", "Category"}:
            frame.groupby(x, as_index=False)[y].sum().plot.bar(x=x, y=y, legend=False, ax=plt.gca(), color="#0f766e")
        else:
            plt.plot(frame[x].astype(str), frame[y], color="#0f766e")
        plt.title(title); plt.xlabel(x); plt.ylabel(y); plt.xticks(rotation=35); plt.tight_layout(); plt.savefig(FIGURES_DIR / name, dpi=120); plt.close()
    plt.figure(figsize=(8, 5)); sns.histplot(df["Demand"], bins=40, color="#f97316"); plt.title("Demand Distribution"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "demand_distribution.png", dpi=120); plt.close()
    plt.figure(figsize=(9, 7)); sns.heatmap(df.select_dtypes("number").corr(), cmap="vlag", center=0); plt.title("Numerical Feature Correlation"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=120); plt.close()


def main():
    ensure_directories()
    df, report = preprocess(save=True)
    make_eda(df)
    featured = add_features(df)
    train_end = featured["Date"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    results = train_tree_models(featured, train_end, MODELS_DIR)
    reference = df[(df["Store ID"] == "S001") & (df["Product ID"] == "P0001")].sort_values("Date")
    sarima_test, sarima_pred, metadata = train_sarima(reference, train_end, MODELS_DIR)
    results["SARIMA"] = (sarima_test, sarima_pred)
    lstm_status = "trained"
    try:
        lstm_test, lstm_pred = train_lstm(reference, train_end, MODELS_DIR)
        results["LSTM"] = (lstm_test, lstm_pred)
    except ModuleNotFoundError as exc:
        if exc.name != "tensorflow":
            raise
        lstm_status = "not trained: TensorFlow is not installed"
    comparison = choose_best(results)
    comparison.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    with open(METRICS_DIR / "data_quality.json", "w", encoding="utf-8") as file:
        json.dump({"dataset": report, "training_start": str(featured["Date"].min().date()), "training_end": str(train_end.date()), "testing_start": str((train_end + pd.Timedelta(days=1)).date()), "testing_end": str(featured["Date"].max().date()), "sarima_reference": metadata, "lstm_status": lstm_status, "best_model": comparison.iloc[0]["Model"]}, file, indent=2, default=str)
    for model_name, (actual_frame, prediction) in results.items():
        plot = pd.DataFrame({"Date": actual_frame["Date"].values, "Actual": actual_frame["Demand"].values, "Predicted": prediction})
        plot.to_csv(RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_predictions.csv", index=False)
        plt.figure(figsize=(11, 4)); plt.plot(plot["Date"], plot["Actual"], label="Actual"); plt.plot(plot["Date"], plot["Predicted"], label="Predicted"); plt.title(f"{model_name}: Actual vs Predicted"); plt.legend(); plt.tight_layout(); plt.savefig(FIGURES_DIR / f"{model_name.lower().replace(' ', '_')}_actual_vs_predicted.png", dpi=120); plt.close()
    print(comparison.to_string(index=False))
    print(f"Best model: {comparison.iloc[0]['Model']}")


if __name__ == "__main__":
    main()
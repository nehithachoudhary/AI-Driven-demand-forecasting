from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "sales_data.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DIR / "sales_clean.csv"
MODELS_DIR = BASE_DIR / "models"
CUSTOMER_MODELS_DIR = BASE_DIR / "customer_models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
FORECASTS_DIR = RESULTS_DIR / "forecasts"
DEFAULT_FORECAST_HORIZON = 7
DEFAULT_LEAD_TIME = 7
DEFAULT_SERVICE_LEVEL = 0.95
TEST_DAYS = 60
RANDOM_STATE = 42

EXPECTED_COLUMNS = [
    "Date", "Store ID", "Product ID", "Category", "Region",
    "Inventory Level", "Units Sold", "Units Ordered", "Price", "Discount",
    "Weather Condition", "Promotion", "Competitor Pricing", "Seasonality",
    "Epidemic", "Demand",
]


def ensure_directories():
    for path in (PROCESSED_DIR, MODELS_DIR, CUSTOMER_MODELS_DIR, RESULTS_DIR, FIGURES_DIR, METRICS_DIR, FORECASTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
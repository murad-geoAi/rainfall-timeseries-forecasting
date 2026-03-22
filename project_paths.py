from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GEOSPATIAL_DATA_DIR = DATA_DIR / "study_area"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVALUATION_OUTPUT_DIR = OUTPUTS_DIR / "evaluation"
FORECAST_OUTPUT_DIR = OUTPUTS_DIR / "forecasts"
FIGURES_OUTPUT_DIR = OUTPUTS_DIR / "figures"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METADATA_DIR = ARTIFACTS_DIR / "metadata"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"

DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

RAW_DAILY_DATA_PATH = RAW_DATA_DIR / "Rainfall_TimeSeries_Tabular_Dataset (1).csv"
MONTHLY_DATASET_PATH = PROCESSED_DATA_DIR / "monthly_rainfall_dataset.csv"
DAILY_CLIMATOLOGY_PATH = PROCESSED_DATA_DIR / "daily_rainfall_climatology.csv"

FUTURE_FORECASTS_PATH = FORECAST_OUTPUT_DIR / "future_forecasts.csv"

BACKTEST_PREDICTIONS_PATH = EVALUATION_OUTPUT_DIR / "backtest_predictions.csv"
EVALUATION_METRICS_PATH = EVALUATION_OUTPUT_DIR / "evaluation_metrics.csv"
EVALUATION_METRICS_BY_HORIZON_PATH = (
    EVALUATION_OUTPUT_DIR / "evaluation_metrics_by_horizon.csv"
)

MODEL_COMPARISON_FIGURE_PATH = FIGURES_OUTPUT_DIR / "model_comparison.png"
TEST_COMPARISON_FIGURE_PATH = FIGURES_OUTPUT_DIR / "test_sequence_comparison.png"
FUTURE_FORECAST_FIGURE_PATH = (
    FIGURES_OUTPUT_DIR / "future_forecast_march_october_2026.png"
)

BEST_MODEL_METADATA_PATH = METADATA_DIR / "best_model.json"


def ensure_project_directories() -> None:
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        GEOSPATIAL_DATA_DIR,
        EVALUATION_OUTPUT_DIR,
        FORECAST_OUTPUT_DIR,
        FIGURES_OUTPUT_DIR,
        MODELS_DIR,
        METADATA_DIR,
        CHECKPOINTS_DIR,
        DOCS_DIR,
        SCRIPTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

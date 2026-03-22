from __future__ import annotations

import json

import pandas as pd

from project_paths import (
    BEST_MODEL_METADATA_PATH,
    EVALUATION_METRICS_PATH,
    FUTURE_FORECASTS_PATH,
)


def main() -> None:
    metrics_path = EVALUATION_METRICS_PATH
    forecast_path = FUTURE_FORECASTS_PATH
    best_model_path = BEST_MODEL_METADATA_PATH

    missing = [path.name for path in (metrics_path, forecast_path, best_model_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing pipeline outputs: "
            + ", ".join(missing)
            + ". Run `python train.py` first."
        )

    metrics_df = pd.read_csv(metrics_path)
    forecast_df = pd.read_csv(forecast_path, parse_dates=["date"])
    with open(best_model_path, "r", encoding="utf-8") as handle:
        best_model = json.load(handle)

    print("Best validation model:", best_model["selected_model"])
    print("Model family:", best_model["model_family"])
    print("Forecast window:", best_model["forecast_start_month"], "to", best_model["forecast_end_month"])
    print("\nModel comparison:")
    print(
        metrics_df[
            [
                "model_name",
                "model_family",
                "validation_RMSE",
                "validation_MAE",
                "test_RMSE",
                "test_MAE",
                "selected_by_validation",
            ]
        ].to_string(index=False)
    )
    print("\nForecast summary:")
    print(
        forecast_df[
            [
                "date",
                "forecast_rainfall_mm",
                "lower_80_mm",
                "upper_80_mm",
                "pattern_label",
                "seasonal_phase",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parent
    metrics_path = project_root / "evaluation_metrics.csv"
    forecast_path = project_root / "future_forecasts.csv"
    best_model_path = project_root / "best_model.json"

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

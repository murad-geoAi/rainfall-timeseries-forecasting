from forecasting_pipeline import run_pipeline


def main() -> None:
    outputs = run_pipeline()
    best_model = outputs["best_model"]
    future_forecasts = outputs["future_forecasts"]

    print("Selected model:", best_model["selected_model"])
    print("Validation RMSE:", f"{best_model['validation_RMSE']:.2f}")
    print("Test RMSE:", f"{best_model['test_RMSE']:.2f}")
    print("\nForecast for March-October 2026:")
    print(
        future_forecasts[
            [
                "date",
                "forecast_rainfall_mm",
                "lower_80_mm",
                "upper_80_mm",
                "pattern_label",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

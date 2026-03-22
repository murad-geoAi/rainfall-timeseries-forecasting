from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_module import (
    ForecastConfig,
    TARGET_COLUMN,
    build_sequence_bundle,
    build_supervised_tabular_frame,
    build_tabular_feature_frame,
    inverse_transform_targets,
    load_monthly_dataframe,
    make_future_sequence_input,
    split_tabular_frame,
)
from daily_rainfall_profiles import save_daily_climatology
from lightning_module import (
    fit_sequence_fixed_epochs,
    predict_sequence_model,
    set_torch_seed,
    train_sequence_model,
)
from models import (
    SeasonalNaiveForecaster,
    build_sequence_model_factories,
    build_tabular_model_factories,
)
from project_paths import (
    BACKTEST_PREDICTIONS_PATH,
    BEST_MODEL_METADATA_PATH,
    DAILY_CLIMATOLOGY_PATH,
    EVALUATION_METRICS_BY_HORIZON_PATH,
    EVALUATION_METRICS_PATH,
    FIGURES_OUTPUT_DIR,
    FORECAST_OUTPUT_DIR,
    FUTURE_FORECAST_FIGURE_PATH,
    FUTURE_FORECASTS_PATH,
    MODEL_COMPARISON_FIGURE_PATH,
    MODELS_DIR,
    MONTHLY_DATASET_PATH,
    PROJECT_ROOT,
    TEST_COMPARISON_FIGURE_PATH,
    ensure_project_directories,
)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    set_torch_seed(seed)


def seasonal_naive_scale(series: pd.Series, seasonality: int = 12) -> float:
    values = series.to_numpy(dtype=float)
    if len(values) <= seasonality:
        return 1.0
    scale = np.mean(np.abs(values[seasonality:] - values[:-seasonality]))
    return float(scale) if scale > 0 else 1.0


def compute_metrics(actual: np.ndarray, predicted: np.ndarray, mase_scale: float) -> dict[str, float]:
    actual = actual.astype(float).ravel()
    predicted = np.clip(predicted.astype(float).ravel(), 0.0, None)
    errors = predicted - actual

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    actual_mean = float(np.mean(actual))
    ss_res = float(np.sum(np.square(actual - predicted)))
    ss_tot = float(np.sum(np.square(actual - actual_mean)))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    smape = float(
        100.0
        * np.mean(
            2.0
            * np.abs(predicted - actual)
            / (np.abs(actual) + np.abs(predicted) + 1e-6)
        )
    )
    wape = float(100.0 * np.sum(np.abs(errors)) / (np.sum(np.abs(actual)) + 1e-6))
    mase = float(mae / max(mase_scale, 1e-6))
    bias = float(np.mean(errors))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "sMAPE": smape,
        "WAPE": wape,
        "MASE": mase,
        "Bias": bias,
    }


def predictions_to_frame(
    model_name: str,
    model_family: str,
    split_name: str,
    origins: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    predicted = np.clip(predicted, 0.0, None)

    for row_index, origin_date in enumerate(pd.DatetimeIndex(origins)):
        for horizon_index in range(actual.shape[1]):
            target_date = origin_date + pd.DateOffset(months=horizon_index + 1)
            actual_value = float(actual[row_index, horizon_index])
            predicted_value = float(predicted[row_index, horizon_index])
            rows.append(
                {
                    "model_name": model_name,
                    "model_family": model_family,
                    "split": split_name,
                    "origin_date": origin_date,
                    "target_date": target_date,
                    "horizon": horizon_index + 1,
                    "actual_rain_mm": actual_value,
                    "predicted_rain_mm": predicted_value,
                    "residual_mm": actual_value - predicted_value,
                    "absolute_error_mm": abs(actual_value - predicted_value),
                }
            )

    return pd.DataFrame(rows)


def format_summary_row(
    model_name: str,
    model_family: str,
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    validation_best_epoch: int | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "model_name": model_name,
        "model_family": model_family,
        "validation_best_epoch": validation_best_epoch,
    }
    for metric_name, metric_value in validation_metrics.items():
        row[f"validation_{metric_name}"] = metric_value
    for metric_name, metric_value in test_metrics.items():
        row[f"test_{metric_name}"] = metric_value
    return row


def evaluate_tabular_models(
    monthly_df: pd.DataFrame,
    config: ForecastConfig,
    tabular_output_dir: Path,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    supervised_df = build_supervised_tabular_frame(monthly_df, config.forecast_horizon)
    feature_frame = build_tabular_feature_frame(monthly_df)
    tabular_split = split_tabular_frame(supervised_df, config, feature_frame=feature_frame)

    validation_scale = seasonal_naive_scale(monthly_df.loc[: config.train_end_ts, TARGET_COLUMN])
    test_scale = seasonal_naive_scale(monthly_df.loc[: config.validation_end_ts, TARGET_COLUMN])

    summary_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    model_factories = build_tabular_model_factories(
        random_state=config.random_seed,
        forecast_horizon=config.forecast_horizon,
    )

    for model_name, factory in model_factories.items():
        if model_name == "SeasonalNaive":
            validation_model = factory().fit(monthly_df[TARGET_COLUMN])
            validation_predictions = validation_model.predict(tabular_split.validation_origins)

            test_model = factory().fit(monthly_df[TARGET_COLUMN])
            test_predictions = test_model.predict(tabular_split.test_origins)
            joblib.dump(test_model, tabular_output_dir / f"{model_name}_trainval.joblib")
        else:
            validation_model = factory()
            validation_model.fit(tabular_split.X_train, tabular_split.y_train)
            validation_predictions = validation_model.predict(tabular_split.X_validation)

            test_model = factory()
            test_model.fit(tabular_split.X_train_validation, tabular_split.y_train_validation)
            test_predictions = test_model.predict(tabular_split.X_test)
            joblib.dump(test_model, tabular_output_dir / f"{model_name}_trainval.joblib")

        validation_metrics = compute_metrics(
            tabular_split.y_validation.to_numpy(),
            validation_predictions,
            validation_scale,
        )
        test_metrics = compute_metrics(
            tabular_split.y_test.to_numpy(),
            test_predictions,
            test_scale,
        )
        summary_rows.append(
            format_summary_row(
                model_name=model_name,
                model_family="tabular",
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
            )
        )

        prediction_frames.append(
            predictions_to_frame(
                model_name=model_name,
                model_family="tabular",
                split_name="validation",
                origins=tabular_split.validation_origins,
                actual=tabular_split.y_validation.to_numpy(),
                predicted=np.asarray(validation_predictions, dtype=float),
            )
        )
        prediction_frames.append(
            predictions_to_frame(
                model_name=model_name,
                model_family="tabular",
                split_name="test",
                origins=tabular_split.test_origins,
                actual=tabular_split.y_test.to_numpy(),
                predicted=np.asarray(test_predictions, dtype=float),
            )
        )

    return summary_rows, prediction_frames


def evaluate_sequence_models(
    monthly_df: pd.DataFrame,
    config: ForecastConfig,
    sequence_output_dir: Path,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    validation_bundle = build_sequence_bundle(
        monthly_df=monthly_df,
        fit_end=config.train_end_ts,
        seq_length=config.seq_length,
        forecast_horizon=config.forecast_horizon,
    )
    test_bundle = build_sequence_bundle(
        monthly_df=monthly_df,
        fit_end=config.validation_end_ts,
        seq_length=config.seq_length,
        forecast_horizon=config.forecast_horizon,
    )

    validation_train_mask = validation_bundle.origins <= config.train_end_ts
    validation_holdout_mask = (validation_bundle.origins > config.train_end_ts) & (
        validation_bundle.origins <= config.validation_end_ts
    )
    test_train_mask = test_bundle.origins <= config.validation_end_ts
    test_holdout_mask = test_bundle.origins > config.validation_end_ts

    validation_scale = seasonal_naive_scale(monthly_df.loc[: config.train_end_ts, TARGET_COLUMN])
    test_scale = seasonal_naive_scale(monthly_df.loc[: config.validation_end_ts, TARGET_COLUMN])

    summary_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    sequence_factories = build_sequence_model_factories(
        input_size=len(validation_bundle.feature_columns),
        forecast_horizon=config.forecast_horizon,
    )

    for model_name, factory in sequence_factories.items():
        model = factory()
        training_result = train_sequence_model(
            model=model,
            X_train=validation_bundle.X[validation_train_mask],
            y_train=validation_bundle.y[validation_train_mask],
            X_validation=validation_bundle.X[validation_holdout_mask],
            y_validation=validation_bundle.y[validation_holdout_mask],
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            max_epochs=config.max_epochs,
            patience=config.patience,
            seed=config.random_seed,
        )

        validation_predictions_scaled = predict_sequence_model(
            training_result.model,
            validation_bundle.X[validation_holdout_mask],
            batch_size=config.batch_size,
        )
        validation_predictions = inverse_transform_targets(
            validation_predictions_scaled,
            validation_bundle.y_scaler,
        )
        validation_actual = inverse_transform_targets(
            validation_bundle.y[validation_holdout_mask],
            validation_bundle.y_scaler,
        )

        test_model = fit_sequence_fixed_epochs(
            model=factory(),
            X_train=test_bundle.X[test_train_mask],
            y_train=test_bundle.y[test_train_mask],
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            epochs=training_result.best_epoch,
            seed=config.random_seed,
        )
        test_predictions_scaled = predict_sequence_model(
            test_model,
            test_bundle.X[test_holdout_mask],
            batch_size=config.batch_size,
        )
        test_predictions = inverse_transform_targets(
            test_predictions_scaled,
            test_bundle.y_scaler,
        )
        test_actual = inverse_transform_targets(
            test_bundle.y[test_holdout_mask],
            test_bundle.y_scaler,
        )

        torch.save(
            {
                "model_state_dict": test_model.state_dict(),
                "best_epoch": training_result.best_epoch,
                "feature_columns": test_bundle.feature_columns,
            },
            sequence_output_dir / f"{model_name}_trainval.pt",
        )

        summary_rows.append(
            format_summary_row(
                model_name=model_name,
                model_family="sequence",
                validation_metrics=compute_metrics(
                    validation_actual,
                    validation_predictions,
                    validation_scale,
                ),
                test_metrics=compute_metrics(
                    test_actual,
                    test_predictions,
                    test_scale,
                ),
                validation_best_epoch=training_result.best_epoch,
            )
        )

        prediction_frames.append(
            predictions_to_frame(
                model_name=model_name,
                model_family="sequence",
                split_name="validation",
                origins=validation_bundle.origins[validation_holdout_mask],
                actual=validation_actual,
                predicted=validation_predictions,
            )
        )
        prediction_frames.append(
            predictions_to_frame(
                model_name=model_name,
                model_family="sequence",
                split_name="test",
                origins=test_bundle.origins[test_holdout_mask],
                actual=test_actual,
                predicted=test_predictions,
            )
        )

    return summary_rows, prediction_frames


def build_metrics_by_horizon(
    predictions_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    validation_scale = seasonal_naive_scale(monthly_df.loc[: config.train_end_ts, TARGET_COLUMN])
    test_scale = seasonal_naive_scale(monthly_df.loc[: config.validation_end_ts, TARGET_COLUMN])

    for (model_name, model_family, split_name, horizon), group in predictions_df.groupby(
        ["model_name", "model_family", "split", "horizon"],
        sort=True,
    ):
        mase_scale = validation_scale if split_name == "validation" else test_scale
        metrics = compute_metrics(
            group["actual_rain_mm"].to_numpy(),
            group["predicted_rain_mm"].to_numpy(),
            mase_scale,
        )
        row: dict[str, object] = {
            "model_name": model_name,
            "model_family": model_family,
            "split": split_name,
            "horizon": horizon,
        }
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)


def choose_best_model(summary_df: pd.DataFrame) -> pd.Series:
    ranking = summary_df.sort_values(
        by=["validation_RMSE", "validation_MAE", "validation_WAPE"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return ranking.iloc[0]


def describe_pattern(z_score: float) -> str:
    if z_score <= -1.0:
        return "Much drier than usual"
    if z_score <= -0.3:
        return "Slightly drier than usual"
    if z_score < 0.3:
        return "Near normal"
    if z_score < 1.0:
        return "Slightly wetter than usual"
    return "Much wetter than usual"


def seasonal_phase(month_number: int) -> str:
    if month_number in (3, 4, 5):
        return "Pre-monsoon build-up"
    if month_number in (6, 7, 8, 9):
        return "Monsoon peak"
    return "Monsoon withdrawal"


def forecast_with_selected_model(
    best_model_row: pd.Series,
    monthly_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    config: ForecastConfig,
    models_dir: Path,
) -> pd.DataFrame:
    model_name = str(best_model_row["model_name"])
    model_family = str(best_model_row["model_family"])

    if model_family == "sequence":
        full_bundle = build_sequence_bundle(
            monthly_df=monthly_df,
            fit_end=monthly_df.index.max(),
            seq_length=config.seq_length,
            forecast_horizon=config.forecast_horizon,
        )
        sequence_factories = build_sequence_model_factories(
            input_size=len(full_bundle.feature_columns),
            forecast_horizon=config.forecast_horizon,
        )
        final_model = fit_sequence_fixed_epochs(
            model=sequence_factories[model_name](),
            X_train=full_bundle.X,
            y_train=full_bundle.y,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            epochs=int(best_model_row["validation_best_epoch"]),
            seed=config.random_seed,
        )
        future_input = make_future_sequence_input(
            monthly_df=monthly_df,
            x_scaler=full_bundle.x_scaler,
            seq_length=config.seq_length,
        )[None, ...]
        future_predictions_scaled = predict_sequence_model(
            final_model,
            future_input,
            batch_size=1,
        )
        future_predictions = inverse_transform_targets(
            future_predictions_scaled,
            full_bundle.y_scaler,
        )[0]
        torch.save(
            {
                "model_state_dict": final_model.state_dict(),
                "best_epoch": int(best_model_row["validation_best_epoch"]),
                "feature_columns": full_bundle.feature_columns,
            },
            models_dir / "best_model_final.pt",
        )
    else:
        supervised_df = build_supervised_tabular_frame(monthly_df, config.forecast_horizon)
        feature_frame = build_tabular_feature_frame(monthly_df)
        tabular_split = split_tabular_frame(
            supervised_df,
            config,
            feature_frame=feature_frame,
        )
        tabular_factories = build_tabular_model_factories(
            random_state=config.random_seed,
            forecast_horizon=config.forecast_horizon,
        )
        future_row = feature_frame.loc[[monthly_df.index.max()]].dropna()
        final_model = tabular_factories[model_name]()
        if isinstance(final_model, SeasonalNaiveForecaster):
            final_model.fit(monthly_df[TARGET_COLUMN])
            future_predictions = final_model.predict(
                pd.DatetimeIndex([monthly_df.index.max()])
            )[0]
        else:
            final_model.fit(
                supervised_df[tabular_split.feature_columns],
                supervised_df[tabular_split.target_columns],
            )
            future_predictions = final_model.predict(future_row)[0]
        joblib.dump(final_model, models_dir / "best_model_final.joblib")

    future_predictions = np.clip(np.asarray(future_predictions, dtype=float), 0.0, None)
    forecast_dates = pd.date_range(
        start=monthly_df.index.max() + pd.offsets.MonthBegin(1),
        periods=config.forecast_horizon,
        freq="MS",
    )

    validation_residuals = predictions_df[
        (predictions_df["model_name"] == model_name)
        & (predictions_df["split"] == "validation")
    ]
    interval_lookup = (
        validation_residuals.groupby("horizon")["residual_mm"]
        .quantile([0.1, 0.9])
        .unstack()
        .rename(columns={0.1: "residual_q10", 0.9: "residual_q90"})
    )

    climatology = (
        monthly_df.groupby(monthly_df.index.month)[TARGET_COLUMN]
        .agg(["mean", "std"])
        .rename(columns={"mean": "historical_month_avg_mm", "std": "historical_month_std_mm"})
    )

    forecast_rows: list[dict[str, object]] = []
    for horizon_step, forecast_date in enumerate(forecast_dates, start=1):
        historical_mean = float(climatology.loc[forecast_date.month, "historical_month_avg_mm"])
        historical_std = float(climatology.loc[forecast_date.month, "historical_month_std_mm"])
        residual_q10 = float(interval_lookup.loc[horizon_step, "residual_q10"])
        residual_q90 = float(interval_lookup.loc[horizon_step, "residual_q90"])
        forecast_value = float(future_predictions[horizon_step - 1])
        lower_80 = max(0.0, forecast_value + residual_q10)
        upper_80 = max(0.0, forecast_value + residual_q90)
        anomaly_mm = forecast_value - historical_mean
        anomaly_pct = 100.0 * anomaly_mm / historical_mean if historical_mean > 0 else np.nan
        z_score = anomaly_mm / historical_std if historical_std > 0 else 0.0

        forecast_rows.append(
            {
                "date": forecast_date,
                "label": forecast_date.strftime("%b %Y"),
                "best_model": model_name,
                "forecast_rainfall_mm": forecast_value,
                "lower_80_mm": lower_80,
                "upper_80_mm": upper_80,
                "historical_month_avg_mm": historical_mean,
                "historical_month_std_mm": historical_std,
                "anomaly_mm": anomaly_mm,
                "anomaly_pct": anomaly_pct,
                "pattern_label": describe_pattern(z_score),
                "seasonal_phase": seasonal_phase(forecast_date.month),
            }
        )

    return pd.DataFrame(forecast_rows)


def plot_model_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    ranking = summary_df.sort_values("validation_RMSE").reset_index(drop=True)
    x_positions = np.arange(len(ranking))
    width = 0.38

    plt.figure(figsize=(12, 6))
    plt.bar(
        x_positions - width / 2,
        ranking["validation_RMSE"],
        width=width,
        label="Validation RMSE",
    )
    plt.bar(
        x_positions + width / 2,
        ranking["test_RMSE"],
        width=width,
        label="Test RMSE",
    )
    plt.xticks(x_positions, ranking["model_name"], rotation=20, ha="right")
    plt.ylabel("RMSE (mm)")
    plt.title("Rainfall Forecast Model Comparison")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_test_comparison(
    predictions_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    top_models = (
        summary_df.sort_values("validation_RMSE")["model_name"].head(5).tolist()
    )
    test_predictions = predictions_df[
        (predictions_df["split"] == "test")
        & (predictions_df["model_name"].isin(top_models))
    ]

    aggregated = (
        test_predictions.groupby(["model_name", "target_date"], as_index=False)
        .agg(
            predicted_rain_mm=("predicted_rain_mm", "mean"),
            actual_rain_mm=("actual_rain_mm", "first"),
        )
        .sort_values("target_date")
    )

    plt.figure(figsize=(12, 6))
    actual_series = (
        aggregated[["target_date", "actual_rain_mm"]]
        .drop_duplicates()
        .sort_values("target_date")
    )
    plt.plot(
        actual_series["target_date"],
        actual_series["actual_rain_mm"],
        color="black",
        linewidth=2.5,
        label="Actual",
    )

    for model_name in top_models:
        model_series = aggregated[aggregated["model_name"] == model_name]
        plt.plot(
            model_series["target_date"],
            model_series["predicted_rain_mm"],
            linewidth=1.8,
            label=model_name,
        )

    plt.title("Test-Window Forecast Comparison")
    plt.ylabel("Rainfall (mm)")
    plt.xlabel("Target Month")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_future_forecast(forecast_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    x_values = forecast_df["date"]

    plt.plot(
        x_values,
        forecast_df["forecast_rainfall_mm"],
        marker="o",
        linewidth=2.5,
        label="Forecast",
    )
    plt.fill_between(
        x_values,
        forecast_df["lower_80_mm"],
        forecast_df["upper_80_mm"],
        alpha=0.2,
        label="80% interval",
    )
    plt.plot(
        x_values,
        forecast_df["historical_month_avg_mm"],
        linestyle="--",
        linewidth=1.8,
        label="Historical monthly average",
    )

    plt.title("Forecast Rainfall for March-October 2026")
    plt.xlabel("Month")
    plt.ylabel("Rainfall (mm)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_pipeline(config: ForecastConfig | None = None) -> dict[str, object]:
    config = config or ForecastConfig()
    set_global_seed(config.random_seed)
    ensure_project_directories()

    monthly_df = load_monthly_dataframe(PROJECT_ROOT / config.csv_path)
    monthly_df.to_csv(MONTHLY_DATASET_PATH)
    save_daily_climatology(
        csv_path=PROJECT_ROOT / config.csv_path,
        output_path=DAILY_CLIMATOLOGY_PATH,
    )

    tabular_rows, tabular_prediction_frames = evaluate_tabular_models(
        monthly_df=monthly_df,
        config=config,
        tabular_output_dir=MODELS_DIR,
    )
    sequence_rows, sequence_prediction_frames = evaluate_sequence_models(
        monthly_df=monthly_df,
        config=config,
        sequence_output_dir=MODELS_DIR,
    )

    summary_df = pd.DataFrame(tabular_rows + sequence_rows)
    summary_df = summary_df.sort_values(
        by=["validation_RMSE", "validation_MAE", "validation_WAPE"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    best_model_row = choose_best_model(summary_df)
    summary_df["selected_by_validation"] = summary_df["model_name"].eq(
        best_model_row["model_name"]
    )
    summary_df.to_csv(EVALUATION_METRICS_PATH, index=False)

    predictions_df = pd.concat(
        tabular_prediction_frames + sequence_prediction_frames,
        ignore_index=True,
    ).sort_values(["split", "model_name", "origin_date", "horizon"])
    predictions_df.to_csv(BACKTEST_PREDICTIONS_PATH, index=False)

    metrics_by_horizon_df = build_metrics_by_horizon(
        predictions_df=predictions_df,
        monthly_df=monthly_df,
        config=config,
    )
    metrics_by_horizon_df.to_csv(EVALUATION_METRICS_BY_HORIZON_PATH, index=False)

    forecast_df = forecast_with_selected_model(
        best_model_row=best_model_row,
        monthly_df=monthly_df,
        predictions_df=predictions_df,
        config=config,
        models_dir=MODELS_DIR,
    )
    forecast_df.to_csv(FUTURE_FORECASTS_PATH, index=False)

    best_model_payload = {
        "selected_model": str(best_model_row["model_name"]),
        "model_family": str(best_model_row["model_family"]),
        "train_origin_end": config.train_end,
        "validation_origin_end": config.validation_end,
        "last_observed_month": monthly_df.index.max().strftime("%Y-%m-%d"),
        "forecast_start_month": forecast_df.loc[0, "date"].strftime("%Y-%m-%d"),
        "forecast_end_month": forecast_df.loc[len(forecast_df) - 1, "date"].strftime(
            "%Y-%m-%d"
        ),
        "validation_RMSE": float(best_model_row["validation_RMSE"]),
        "test_RMSE": float(best_model_row["test_RMSE"]),
        "validation_best_epoch": (
            int(best_model_row["validation_best_epoch"])
            if pd.notna(best_model_row["validation_best_epoch"])
            else None
        ),
        "seq_length": config.seq_length,
        "forecast_horizon": config.forecast_horizon,
        "random_seed": config.random_seed,
    }
    with open(BEST_MODEL_METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(best_model_payload, handle, indent=2)

    plot_model_comparison(summary_df, MODEL_COMPARISON_FIGURE_PATH)
    plot_test_comparison(
        predictions_df=predictions_df,
        summary_df=summary_df,
        output_path=TEST_COMPARISON_FIGURE_PATH,
    )
    plot_future_forecast(
        forecast_df=forecast_df,
        output_path=FUTURE_FORECAST_FIGURE_PATH,
    )

    return {
        "best_model": best_model_payload,
        "evaluation_metrics": summary_df,
        "future_forecasts": forecast_df,
        "backtest_predictions": predictions_df,
    }

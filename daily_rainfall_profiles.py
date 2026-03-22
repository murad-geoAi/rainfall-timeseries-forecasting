from __future__ import annotations

import calendar
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAIN_THRESHOLD_MM = 0.1
RECENCY_DECAY = 0.97


@dataclass
class MonthProfileResult:
    target_date: pd.Timestamp
    monthly_total_mm: float
    monthly_total_source: str
    profile: pd.DataFrame
    top_chance_day: pd.Series
    top_amount_day: pd.Series
    expected_rainy_days: float
    monthly_context: dict[str, float | str | None]


def load_daily_rainfall_dataframe(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    daily_df = df[["date", "rain_mm"]].copy()
    daily_df["year"] = daily_df["date"].dt.year
    daily_df["month"] = daily_df["date"].dt.month
    daily_df["day"] = daily_df["date"].dt.day
    daily_df["weekday"] = daily_df["date"].dt.day_name()
    return daily_df.sort_values("date").reset_index(drop=True)


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(np.average(values.to_numpy(dtype=float), weights=weights.to_numpy(dtype=float)))


def build_daily_climatology(
    daily_df: pd.DataFrame,
    *,
    rain_threshold_mm: float = RAIN_THRESHOLD_MM,
    recency_decay: float = RECENCY_DECAY,
) -> pd.DataFrame:
    working_df = daily_df.copy()
    max_year = int(working_df["year"].max())
    working_df["weight"] = recency_decay ** (max_year - working_df["year"])
    working_df["rain_event"] = (working_df["rain_mm"] >= rain_threshold_mm).astype(float)

    rows: list[dict[str, float | int | str]] = []
    for (month, day), group in working_df.groupby(["month", "day"], sort=True):
        rainy_group = group[group["rain_event"] > 0]
        probability = _weighted_average(group["rain_event"], group["weight"])
        mean_rain = _weighted_average(group["rain_mm"], group["weight"])
        rainy_intensity = (
            _weighted_average(rainy_group["rain_mm"], rainy_group["weight"])
            if not rainy_group.empty
            else 0.0
        )
        risk_score = probability * max(rainy_intensity, mean_rain, 1e-6)

        rows.append(
            {
                "month": int(month),
                "day": int(day),
                "observations": int(len(group)),
                "years_covered": int(group["year"].nunique()),
                "rain_probability": probability,
                "rain_probability_pct": probability * 100.0,
                "weighted_mean_rain_mm": mean_rain,
                "weighted_rainy_day_intensity_mm": rainy_intensity,
                "risk_score": risk_score,
            }
        )

    climatology_df = pd.DataFrame(rows).sort_values(["month", "day"]).reset_index(drop=True)
    return climatology_df


def save_daily_climatology(
    csv_path: str | Path,
    output_path: str | Path,
    *,
    rain_threshold_mm: float = RAIN_THRESHOLD_MM,
    recency_decay: float = RECENCY_DECAY,
) -> pd.DataFrame:
    daily_df = load_daily_rainfall_dataframe(csv_path)
    climatology_df = build_daily_climatology(
        daily_df,
        rain_threshold_mm=rain_threshold_mm,
        recency_decay=recency_decay,
    )
    output_path = Path(output_path)
    climatology_df.to_csv(output_path, index=False)
    return climatology_df


def load_or_build_daily_climatology(
    csv_path: str | Path,
    climatology_path: str | Path | None = None,
    *,
    rain_threshold_mm: float = RAIN_THRESHOLD_MM,
    recency_decay: float = RECENCY_DECAY,
) -> pd.DataFrame:
    if climatology_path is not None:
        climatology_file = Path(climatology_path)
        if climatology_file.exists():
            return pd.read_csv(climatology_file)

    daily_df = load_daily_rainfall_dataframe(csv_path)
    return build_daily_climatology(
        daily_df,
        rain_threshold_mm=rain_threshold_mm,
        recency_decay=recency_decay,
    )


def _historical_month_total(monthly_df: pd.DataFrame, month_number: int) -> float:
    month_slice = monthly_df[monthly_df["date"].dt.month == month_number]
    return float(month_slice["rain_mm"].mean())


def resolve_monthly_total(
    target_date: pd.Timestamp,
    *,
    monthly_forecasts_df: pd.DataFrame,
    monthly_history_df: pd.DataFrame,
    custom_monthly_total_mm: float | None = None,
) -> tuple[float, str, dict[str, float | str | None]]:
    if custom_monthly_total_mm is not None:
        return (
            float(custom_monthly_total_mm),
            "Custom user input",
            {
                "best_model": None,
                "pattern_label": None,
                "anomaly_pct": None,
                "seasonal_phase": None,
            },
        )

    matching_forecast = monthly_forecasts_df[
        monthly_forecasts_df["date"] == target_date.normalize()
    ]
    if not matching_forecast.empty:
        forecast_row = matching_forecast.iloc[0]
        return (
            float(forecast_row["forecast_rainfall_mm"]),
            "Saved model forecast",
            {
                "best_model": forecast_row.get("best_model"),
                "pattern_label": forecast_row.get("pattern_label"),
                "anomaly_pct": (
                    float(forecast_row["anomaly_pct"])
                    if pd.notna(forecast_row.get("anomaly_pct"))
                    else None
                ),
                "seasonal_phase": forecast_row.get("seasonal_phase"),
            },
        )

    return (
        _historical_month_total(monthly_history_df, target_date.month),
        "Historical monthly average",
        {
            "best_model": None,
            "pattern_label": "Climatology baseline",
            "anomaly_pct": None,
            "seasonal_phase": None,
        },
    )


def _complete_month_profile(
    month_number: int,
    days_in_month: int,
    month_climatology_df: pd.DataFrame,
) -> pd.DataFrame:
    base_days = pd.DataFrame({"day": np.arange(1, days_in_month + 1, dtype=int)})
    merged = base_days.merge(month_climatology_df, on="day", how="left")

    fill_values = {
        "month": month_number,
        "observations": 0,
        "years_covered": 0,
        "rain_probability": 0.0,
        "rain_probability_pct": 0.0,
        "weighted_mean_rain_mm": 0.0,
        "weighted_rainy_day_intensity_mm": 0.0,
        "risk_score": 0.0,
    }
    merged = merged.fillna(fill_values)
    merged["month"] = merged["month"].astype(int)
    merged["day"] = merged["day"].astype(int)
    return merged


def build_month_profile(
    *,
    target_year: int,
    target_month: int,
    monthly_forecasts_df: pd.DataFrame,
    monthly_history_df: pd.DataFrame,
    daily_climatology_df: pd.DataFrame,
    custom_monthly_total_mm: float | None = None,
) -> MonthProfileResult:
    target_date = pd.Timestamp(year=target_year, month=target_month, day=1)
    monthly_total_mm, monthly_total_source, monthly_context = resolve_monthly_total(
        target_date,
        monthly_forecasts_df=monthly_forecasts_df,
        monthly_history_df=monthly_history_df,
        custom_monthly_total_mm=custom_monthly_total_mm,
    )

    days_in_month = calendar.monthrange(target_year, target_month)[1]
    month_climatology_df = daily_climatology_df[
        daily_climatology_df["month"] == target_month
    ].copy()
    profile_df = _complete_month_profile(target_month, days_in_month, month_climatology_df)

    profile_df["date"] = pd.date_range(target_date, periods=days_in_month, freq="D")
    profile_df["weekday"] = profile_df["date"].dt.day_name()
    profile_df["month_name"] = profile_df["date"].dt.strftime("%B")

    total_score = float(profile_df["risk_score"].sum())
    if total_score <= 0:
        profile_df["distribution_weight"] = 1.0 / len(profile_df)
    else:
        profile_df["distribution_weight"] = profile_df["risk_score"] / total_score

    profile_df["estimated_daily_rain_mm"] = (
        profile_df["distribution_weight"] * monthly_total_mm
    )
    profile_df["chance_rank"] = (
        profile_df["rain_probability_pct"].rank(method="dense", ascending=False).astype(int)
    )
    profile_df["amount_rank"] = (
        profile_df["estimated_daily_rain_mm"].rank(method="dense", ascending=False).astype(int)
    )

    top_chance_day = profile_df.sort_values(
        ["rain_probability_pct", "estimated_daily_rain_mm", "day"],
        ascending=[False, False, True],
    ).iloc[0]
    top_amount_day = profile_df.sort_values(
        ["estimated_daily_rain_mm", "rain_probability_pct", "day"],
        ascending=[False, False, True],
    ).iloc[0]

    expected_rainy_days = float(profile_df["rain_probability"].sum())

    return MonthProfileResult(
        target_date=target_date,
        monthly_total_mm=monthly_total_mm,
        monthly_total_source=monthly_total_source,
        profile=profile_df,
        top_chance_day=top_chance_day,
        top_amount_day=top_amount_day,
        expected_rainy_days=expected_rainy_days,
        monthly_context=monthly_context,
    )

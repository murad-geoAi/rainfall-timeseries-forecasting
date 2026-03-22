from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from project_paths import RAW_DAILY_DATA_PATH

TARGET_COLUMN = "rain_mm"
BASE_FEATURE_COLUMNS = [
    "dewpoint_c",
    "era5land_tp_mm",
    "soil_water_l1",
    "solar_rad_MJm2",
    "surface_pressure_hpa",
    "temp_c",
    "u10_ms",
    "v10_ms",
    "wind_speed_ms",
    TARGET_COLUMN,
]
MONTHLY_AGGREGATIONS = {
    "dewpoint_c": "mean",
    "era5land_tp_mm": "sum",
    "soil_water_l1": "mean",
    "solar_rad_MJm2": "mean",
    "surface_pressure_hpa": "mean",
    "temp_c": "mean",
    "u10_ms": "mean",
    "v10_ms": "mean",
    "wind_speed_ms": "mean",
    TARGET_COLUMN: "sum",
}
TABULAR_LAGS = (1, 2, 3, 6, 12, 24)
ROLLING_WINDOWS = (3, 6, 12)


@dataclass(frozen=True)
class ForecastConfig:
    csv_path: str = str(RAW_DAILY_DATA_PATH.relative_to(Path(__file__).resolve().parent))
    seq_length: int = 24
    forecast_horizon: int = 8
    train_end: str = "2021-12-01"
    validation_end: str = "2023-12-01"
    batch_size: int = 16
    max_epochs: int = 200
    patience: int = 20
    learning_rate: float = 1e-3
    random_seed: int = 42

    @property
    def train_end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.train_end)

    @property
    def validation_end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.validation_end)


@dataclass
class TabularSplit:
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_validation: pd.DataFrame
    y_validation: pd.DataFrame
    X_train_validation: pd.DataFrame
    y_train_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame
    validation_origins: pd.DatetimeIndex
    test_origins: pd.DatetimeIndex
    feature_columns: list[str]
    target_columns: list[str]
    feature_frame: pd.DataFrame


@dataclass
class SequenceBundle:
    X: np.ndarray
    y: np.ndarray
    origins: pd.DatetimeIndex
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    feature_columns: list[str]


def load_monthly_dataframe(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, parse_dates=["date"])
    monthly = (
        df.set_index("date")
        .sort_index()
        .resample("MS")
        .agg(MONTHLY_AGGREGATIONS)
        .ffill()
        .bfill()
    )
    return monthly


def build_tabular_feature_frame(monthly_df: pd.DataFrame) -> pd.DataFrame:
    feature_frame = pd.DataFrame(index=monthly_df.index)
    feature_frame["month_sin"] = np.sin(2 * np.pi * monthly_df.index.month / 12.0)
    feature_frame["month_cos"] = np.cos(2 * np.pi * monthly_df.index.month / 12.0)
    feature_frame["year"] = monthly_df.index.year
    feature_frame["trend"] = np.arange(len(monthly_df), dtype=float)

    for lag in TABULAR_LAGS:
        for column in BASE_FEATURE_COLUMNS:
            feature_frame[f"{column}_lag_{lag}"] = monthly_df[column].shift(lag)

    shifted_target = monthly_df[TARGET_COLUMN].shift(1)
    for window in ROLLING_WINDOWS:
        rolling_view = shifted_target.rolling(window)
        feature_frame[f"{TARGET_COLUMN}_roll_mean_{window}"] = rolling_view.mean()
        feature_frame[f"{TARGET_COLUMN}_roll_std_{window}"] = rolling_view.std()
        feature_frame[f"{TARGET_COLUMN}_roll_max_{window}"] = rolling_view.max()
        feature_frame[f"{TARGET_COLUMN}_roll_min_{window}"] = rolling_view.min()

    feature_frame["rain_lag_1_minus_2"] = (
        monthly_df[TARGET_COLUMN].shift(1) - monthly_df[TARGET_COLUMN].shift(2)
    )
    feature_frame["rain_lag_12_minus_24"] = (
        monthly_df[TARGET_COLUMN].shift(12) - monthly_df[TARGET_COLUMN].shift(24)
    )

    return feature_frame


def build_supervised_tabular_frame(
    monthly_df: pd.DataFrame, forecast_horizon: int
) -> pd.DataFrame:
    feature_frame = build_tabular_feature_frame(monthly_df)
    targets = pd.DataFrame(index=monthly_df.index)
    for horizon_step in range(1, forecast_horizon + 1):
        targets[f"target_t_plus_{horizon_step}"] = monthly_df[TARGET_COLUMN].shift(
            -horizon_step
        )

    supervised = feature_frame.join(targets).dropna().copy()
    return supervised


def split_tabular_frame(
    supervised_df: pd.DataFrame,
    config: ForecastConfig,
    feature_frame: pd.DataFrame | None = None,
) -> TabularSplit:
    target_columns = [
        f"target_t_plus_{horizon_step}"
        for horizon_step in range(1, config.forecast_horizon + 1)
    ]
    feature_columns = [
        column for column in supervised_df.columns if column not in target_columns
    ]
    origins = pd.DatetimeIndex(supervised_df.index)

    train_mask = origins <= config.train_end_ts
    validation_mask = (origins > config.train_end_ts) & (
        origins <= config.validation_end_ts
    )
    train_validation_mask = origins <= config.validation_end_ts
    test_mask = origins > config.validation_end_ts

    return TabularSplit(
        X_train=supervised_df.loc[train_mask, feature_columns],
        y_train=supervised_df.loc[train_mask, target_columns],
        X_validation=supervised_df.loc[validation_mask, feature_columns],
        y_validation=supervised_df.loc[validation_mask, target_columns],
        X_train_validation=supervised_df.loc[train_validation_mask, feature_columns],
        y_train_validation=supervised_df.loc[train_validation_mask, target_columns],
        X_test=supervised_df.loc[test_mask, feature_columns],
        y_test=supervised_df.loc[test_mask, target_columns],
        validation_origins=origins[validation_mask],
        test_origins=origins[test_mask],
        feature_columns=feature_columns,
        target_columns=target_columns,
        feature_frame=feature_frame if feature_frame is not None else pd.DataFrame(),
    )


def build_sequence_bundle(
    monthly_df: pd.DataFrame,
    fit_end: pd.Timestamp,
    seq_length: int,
    forecast_horizon: int,
) -> SequenceBundle:
    x_scaler = StandardScaler().fit(monthly_df.loc[:fit_end, BASE_FEATURE_COLUMNS])
    y_scaler = StandardScaler().fit(monthly_df.loc[:fit_end, [TARGET_COLUMN]])

    scaled_features = x_scaler.transform(monthly_df[BASE_FEATURE_COLUMNS])
    scaled_target = y_scaler.transform(monthly_df[[TARGET_COLUMN]])[:, 0]

    X_values: list[np.ndarray] = []
    y_values: list[np.ndarray] = []
    origin_dates: list[pd.Timestamp] = []

    for row_index in range(seq_length, len(monthly_df) - forecast_horizon + 1):
        X_values.append(scaled_features[row_index - seq_length : row_index])
        y_values.append(scaled_target[row_index : row_index + forecast_horizon])
        origin_dates.append(monthly_df.index[row_index - 1])

    return SequenceBundle(
        X=np.asarray(X_values, dtype=np.float32),
        y=np.asarray(y_values, dtype=np.float32),
        origins=pd.DatetimeIndex(origin_dates),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_columns=list(BASE_FEATURE_COLUMNS),
    )


def inverse_transform_targets(
    y_values: np.ndarray, y_scaler: StandardScaler
) -> np.ndarray:
    return y_scaler.inverse_transform(y_values.reshape(-1, 1)).reshape(y_values.shape)


def make_future_sequence_input(
    monthly_df: pd.DataFrame,
    x_scaler: StandardScaler,
    seq_length: int,
) -> np.ndarray:
    scaled_features = x_scaler.transform(monthly_df[BASE_FEATURE_COLUMNS])
    future_window = scaled_features[-seq_length:]
    return future_window.astype(np.float32)

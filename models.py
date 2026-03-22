from __future__ import annotations

import numpy as np
import pandas as pd
import torch.nn as nn
from typing import Callable

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class VanillaLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        sequence_output, _ = self.lstm(inputs)
        return self.head(sequence_output[:, -1, :])


class GRUModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        sequence_output, _ = self.gru(inputs)
        return self.head(sequence_output[:, -1, :])


class BiLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
            bidirectional=True,
        )
        self.head = nn.Linear(hidden_size * 2, output_size)

    def forward(self, inputs):
        sequence_output, _ = self.lstm(inputs)
        return self.head(sequence_output[:, -1, :])


class SeasonalNaiveForecaster:
    def __init__(self, forecast_horizon: int) -> None:
        self.forecast_horizon = forecast_horizon
        self.target_history_: pd.Series | None = None

    def fit(self, target_history: pd.Series) -> "SeasonalNaiveForecaster":
        self.target_history_ = target_history.sort_index().copy()
        return self

    def predict(self, origin_dates: pd.DatetimeIndex) -> np.ndarray:
        if self.target_history_ is None:
            raise RuntimeError("SeasonalNaiveForecaster must be fitted before predicting.")

        predictions = []
        for origin_date in pd.DatetimeIndex(origin_dates):
            row = []
            for horizon_step in range(1, self.forecast_horizon + 1):
                target_date = origin_date + pd.DateOffset(months=horizon_step)
                reference_date = target_date - pd.DateOffset(years=1)
                row.append(float(self.target_history_.loc[reference_date]))
            predictions.append(row)

        return np.asarray(predictions, dtype=float)


def build_tabular_model_factories(
    random_state: int, forecast_horizon: int
) -> dict[str, Callable[[], object]]:
    return {
        "SeasonalNaive": lambda: SeasonalNaiveForecaster(forecast_horizon),
        "Ridge": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=3.0)),
            ]
        ),
        "ElasticNet": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MultiOutputRegressor(
                        ElasticNet(
                            alpha=0.1,
                            l1_ratio=0.2,
                            max_iter=5000,
                            random_state=random_state,
                        ),
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
        ),
        "ExtraTrees": lambda: ExtraTreesRegressor(
            n_estimators=350,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
        ),
        "XGBoost": lambda: MultiOutputRegressor(
            XGBRegressor(
                n_estimators=250,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=1,
                verbosity=0,
            ),
            n_jobs=1,
        ),
    }


def build_sequence_model_factories(
    input_size: int, forecast_horizon: int
) -> dict[str, Callable[[], nn.Module]]:
    return {
        "VanillaLSTM": lambda: VanillaLSTM(
            input_size=input_size,
            output_size=forecast_horizon,
        ),
        "GRU": lambda: GRUModel(
            input_size=input_size,
            output_size=forecast_horizon,
        ),
        "BiLSTM": lambda: BiLSTM(
            input_size=input_size,
            output_size=forecast_horizon,
        ),
    }

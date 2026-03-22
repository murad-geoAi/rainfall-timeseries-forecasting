# Rainfall Time-Series Forecasting

This project forecasts monthly rainfall totals for **March 2026 to October 2026** using a reproducible, leakage-free pipeline built from the daily rainfall dataset in this repository.

## Objective

Forecast rainfall for:

- March 2026
- April 2026
- May 2026
- June 2026
- July 2026
- August 2026
- September 2026
- October 2026

The project now produces:

- clean monthly training data
- day-level rainfall chance profiles for each month
- chronological train/validation/test evaluation
- comparison across deep-learning and tabular baselines
- saved evaluation metrics
- saved backtest predictions
- a final selected model and future forecast table

## What Changed

The pipeline was rebuilt around the following rules:

- Daily data is resampled to monthly data in a consistent way.
- Forecast features use only information available at the forecast origin.
- Model selection is based on the **validation split**, not the test split.
- The final model is retrained on all available history through **February 2026** before generating future forecasts.
- Metrics are saved for both overall performance and horizon-by-horizon performance.
- A daily climatology artifact is generated so a Streamlit app can rank the most likely rainy days within any selected month.

## Models Compared

- `SeasonalNaive`
- `Ridge`
- `ElasticNet`
- `RandomForest`
- `ExtraTrees`
- `XGBoost`
- `VanillaLSTM`
- `GRU`
- `BiLSTM`

## Latest Run

The latest reproducible run selected **BiLSTM** on validation RMSE.

- Validation RMSE: `88.26`
- Test RMSE: `110.92`
- Last observed month in the data: `2026-02-01`
- Forecast window: `2026-03-01` to `2026-10-01`

Forecast snapshot from the latest run:

| Month | Forecast Rainfall (mm) | Pattern |
| --- | ---: | --- |
| Mar 2026 | 32.13 | Near normal |
| Apr 2026 | 180.56 | Slightly wetter than usual |
| May 2026 | 398.33 | Slightly wetter than usual |
| Jun 2026 | 571.47 | Near normal |
| Jul 2026 | 557.88 | Near normal |
| Aug 2026 | 504.08 | Slightly wetter than usual |
| Sep 2026 | 392.32 | Slightly drier than usual |
| Oct 2026 | 161.92 | Near normal |

## How To Run

From the project folder:

```bash
python train.py
python evaluate.py
python -m streamlit run streamlit_app.py
```

`train.py` rebuilds all outputs from scratch.

`evaluate.py` reads the saved outputs and prints a summary.

`streamlit_app.py` launches an interactive app where a user can:

- choose any month and year
- use the saved forecast total, climatology, or a custom monthly rainfall total
- see which day has the greatest rainfall chance
- inspect a ranked day-by-day rainfall profile
- download the daily profile as CSV

For the app, run `python train.py` first if you want the saved 2026 forecast to appear automatically. The app still works without that file by falling back to historical monthly averages or user input.

If `python -m streamlit` is not available in your active interpreter, install the dependencies from `requirements.txt` first. On this machine, the app was smoke-tested successfully through the local Anaconda Streamlit interpreter.

## Output Files

- `best_model.json`
- `evaluation_metrics.csv`
- `evaluation_metrics_by_horizon.csv`
- `backtest_predictions.csv`
- `future_forecasts.csv`
- `monthly_rainfall_dataset.csv`
- `daily_rainfall_climatology.csv`
- `model_comparison.png`
- `test_sequence_comparison.png`
- `future_forecast_march_october_2026.png`
- `streamlit_app.py`

## Notes For Presentation

- The selected model is chosen on validation performance to avoid test leakage.
- Some tabular models have very strong test scores, especially `ExtraTrees`, but the final model choice follows the leakage-free validation rule.
- The forecast file also includes an anomaly-based monthly pattern label and an 80% uncertainty interval.
- The Streamlit app estimates the most likely rainy days using historical day-of-month rainfall occurrence and rainy-day intensity, weighted slightly toward recent years.

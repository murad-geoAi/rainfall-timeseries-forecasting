# Rainfall Forecasting App

This project forecasts rainfall for **March to October 2026** and provides a clean Streamlit app to explore the **days within a month that have the highest chance of rainfall**.

## What The Project Does

- trains and compares multiple rainfall forecasting models
- selects the final model using a leakage-free validation workflow
- forecasts monthly rainfall for March-October 2026
- builds day-level rainfall chance profiles from historical daily rainfall behavior
- serves the results through a deployment-ready Streamlit app

## Current Best Model

Latest saved run:

- Model: `BiLSTM`
- Validation RMSE: `88.26`
- Test RMSE: `110.92`

Saved forecast file:

- [future_forecasts.csv](future_forecasts.csv)

## App

Main Streamlit entrypoint:

- [app.py](app.py)

App implementation:

- [streamlit_app.py](streamlit_app.py)

The app lets a user:

- choose any month and year
- use the saved forecast total or enter a custom monthly rainfall total
- see the day with the highest rainfall chance
- view the top rainy days for the month
- download the full day-by-day rainfall profile as CSV

## Install

For the Streamlit app only:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

For the full training and evaluation pipeline:

```bash
pip install -r requirements-training.txt
python train.py
python evaluate.py
```

## Which File To Run

To check the app interface locally in your terminal, run:

```bash
python -m streamlit run app.py
```

If that does not work in your active Python, use the interpreter where Streamlit is installed:

```bash
C:\ProgramData\Anaconda3\python.exe -m streamlit run app.py
```

For retraining the forecasting pipeline, run:

```bash
python train.py
```

For reading the saved evaluation summary, run:

```bash
python evaluate.py
```

## Streamlit Community Cloud

Deployment notes are here:

- [DEPLOY_STREAMLIT_CLOUD.md](DEPLOY_STREAMLIT_CLOUD.md)

The repository is already prepared for Community Cloud with:

- lightweight app dependencies in `requirements.txt`
- a simple app entrypoint in `app.py`
- pinned hosted Python version in `runtime.txt`
- Streamlit theme config in `.streamlit/config.toml`
- precomputed CSV artifacts so the hosted app does not need to retrain models

When you create the app on Streamlit Community Cloud:

- connect your GitHub repository
- keep the full project folder in the repo
- set the main file path to `app.py`

## Important Files

- [train.py](train.py)
- [evaluate.py](evaluate.py)
- [forecasting_pipeline.py](forecasting_pipeline.py)
- [daily_rainfall_profiles.py](daily_rainfall_profiles.py)
- [monthly_rainfall_dataset.csv](monthly_rainfall_dataset.csv)
- [daily_rainfall_climatology.csv](daily_rainfall_climatology.csv)
- [evaluation_metrics.csv](evaluation_metrics.csv)
- [future_forecast_march_october_2026.png](future_forecast_march_october_2026.png)

## LinkedIn Post

A ready-to-use LinkedIn post is included here:

- [LINKEDIN_POST.md](LINKEDIN_POST.md)

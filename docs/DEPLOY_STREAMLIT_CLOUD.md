# Streamlit Community Cloud Deployment

## Files Already Prepared

- `app.py` as the app entrypoint
- `requirements.txt` with lightweight app dependencies
- `.streamlit/config.toml` with theme and headless settings
- precomputed CSV artifacts for forecasts and daily rainfall profiles

## Deploy Steps

1. Push this project folder to GitHub.
2. Open Streamlit Community Cloud.
3. Click **New app**.
4. Select your repository and branch.
5. Set the main file path to `app.py`.
6. Open **Advanced settings** and choose Python `3.11` or `3.12`.
7. Deploy.

Artifacts that should stay in the repository for the hosted app:

- `outputs/forecasts/future_forecasts.csv`
- `data/processed/monthly_rainfall_dataset.csv`
- `data/processed/daily_rainfall_climatology.csv`
- `artifacts/metadata/best_model.json`

## Why This Setup Is Cloud-Friendly

- The app does not retrain models during startup.
- The app reads saved forecast files and climatology files from the repo.
- `requirements.txt` is kept light for hosting.
- Heavy model-training dependencies were moved to `requirements-training.txt`.
- The app avoids Altair so it is less sensitive to Python-version changes on Community Cloud.

## Important Note About Python Version

Streamlit Community Cloud now chooses Python from the deployment dialog, not from `runtime.txt`.

If your app was already deployed with a newer Python version, delete the app and redeploy it, then pick Python `3.11` or `3.12` in **Advanced settings**.

## Local Run

For the hosted app dependency set:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

For full model retraining as well:

```bash
pip install -r requirements-training.txt
python train.py
```

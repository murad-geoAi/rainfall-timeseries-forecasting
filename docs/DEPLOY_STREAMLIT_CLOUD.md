# Streamlit Community Cloud Deployment

## Files Already Prepared

- `app.py` as the app entrypoint
- `requirements.txt` with lightweight app dependencies
- `runtime.txt` with the Python version for deployment
- `.streamlit/config.toml` with theme and headless settings
- precomputed CSV artifacts for forecasts and daily rainfall profiles

## Deploy Steps

1. Push this project folder to GitHub.
2. Open Streamlit Community Cloud.
3. Click **New app**.
4. Select your repository and branch.
5. Set the main file path to `app.py`.
6. Deploy.

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
- `runtime.txt` makes the cloud Python version explicit.

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

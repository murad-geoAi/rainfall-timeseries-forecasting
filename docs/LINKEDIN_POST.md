# LinkedIn Post

Use one of the versions below depending on how much space and detail you want in your post.

## Main Post

Built a rainfall forecasting project that goes beyond predicting monthly totals.

This app forecasts rainfall for **March to October 2026** and also estimates **which days inside a selected month have the highest chance of rainfall**.

What I focused on in this project:

- building a reproducible and leakage-free forecasting pipeline
- comparing multiple models with time-aware validation
- selecting the final model based on validation performance, not test leakage
- generating forecast outputs and evaluation artifacts
- designing a clean Streamlit app for non-technical users

The final selected model was **BiLSTM**, with:

- Validation RMSE: `88.26`
- Test RMSE: `110.92`

The app allows a user to:

- choose any month and year
- use the saved forecast or enter a custom monthly rainfall total
- identify the day with the highest rain chance
- inspect the top rainy days for the month
- download the ranked daily rainfall profile as CSV

What I like most about this project is that it combines:

- time-series forecasting
- model evaluation and selection
- leakage-free ML workflow design
- climate data storytelling
- product thinking through a user-facing app

I also packaged it for **Streamlit Community Cloud** so it can be shared directly from GitHub.

If you'd like to explore the code or the app:

- GitHub: `[paste your GitHub repo link here]`
- App: `[paste your Streamlit app link here]`

I'd love feedback on the forecasting approach, interface design, or ideas for improving day-level rainfall ranking.

#MachineLearning #TimeSeries #Forecasting #DataScience #Streamlit #Python #DeepLearning #ClimateData #Rainfall #PortfolioProject

## Short Version

Built a rainfall forecasting app that predicts **monthly rainfall** and ranks the **most likely rainy days inside a selected month**.

Highlights:

- leakage-free forecasting pipeline
- multi-model comparison and validation-based selection
- BiLSTM chosen as the final model
- Streamlit app for exploring rainy-day patterns

This project helped me combine forecasting, evaluation, and product-style presentation in one workflow.

GitHub: `[paste repo link]`
App: `[paste app link]`

#MachineLearning #TimeSeries #Forecasting #Streamlit #Python #DataScience

## First Comment

Here are the project links:

- GitHub repo: `[paste repo link]`
- Live app: `[paste app link]`

The repository includes the training pipeline, evaluation results, forecast outputs, and the Streamlit app code.

from __future__ import annotations

import json
import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from daily_rainfall_profiles import (
    RAIN_THRESHOLD_MM,
    RECENCY_DECAY,
    build_daily_climatology,
    build_month_profile,
    load_daily_rainfall_dataframe,
)
from data_module import load_monthly_dataframe


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_CSV_PATH = PROJECT_ROOT / "Rainfall_TimeSeries_Tabular_Dataset (1).csv"
MONTHLY_DATASET_PATH = PROJECT_ROOT / "monthly_rainfall_dataset.csv"
FUTURE_FORECASTS_PATH = PROJECT_ROOT / "future_forecasts.csv"
DAILY_CLIMATOLOGY_PATH = PROJECT_ROOT / "daily_rainfall_climatology.csv"
BEST_MODEL_PATH = PROJECT_ROOT / "best_model.json"


def apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(44, 122, 123, 0.12), transparent 32%),
                linear-gradient(180deg, #f4efe7 0%, #f8f6f1 48%, #eef4f1 100%);
            color: #17333a;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        h1, h2, h3 {
            font-family: Georgia, "Times New Roman", serif;
            color: #12343b;
            letter-spacing: 0.01em;
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(18, 52, 59, 0.95), rgba(41, 94, 96, 0.92));
            border-radius: 24px;
            padding: 1.4rem 1.5rem;
            color: #f7f4ec;
            box-shadow: 0 18px 36px rgba(18, 52, 59, 0.16);
            margin-bottom: 1rem;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(18, 52, 59, 0.08);
            border-radius: 18px;
            padding: 1rem 1rem 0.8rem 1rem;
            box-shadow: 0 12px 24px rgba(18, 52, 59, 0.08);
        }
        .metric-label {
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #4b676d;
        }
        .metric-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #12343b;
            margin-top: 0.1rem;
        }
        .metric-note {
            color: #51656a;
            font-size: 0.92rem;
            margin-top: 0.3rem;
        }
        .note-panel {
            background: rgba(255, 255, 255, 0.78);
            border-left: 4px solid #2d7370;
            border-radius: 14px;
            padding: 0.9rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_static_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    daily_df = load_daily_rainfall_dataframe(RAW_CSV_PATH)

    if MONTHLY_DATASET_PATH.exists():
        monthly_history_df = pd.read_csv(MONTHLY_DATASET_PATH, parse_dates=["date"])
    else:
        monthly_history_df = (
            load_monthly_dataframe(RAW_CSV_PATH).reset_index().rename(columns={"index": "date"})
        )

    if FUTURE_FORECASTS_PATH.exists():
        monthly_forecasts_df = pd.read_csv(FUTURE_FORECASTS_PATH, parse_dates=["date"])
    else:
        monthly_forecasts_df = pd.DataFrame(
            columns=[
                "date",
                "best_model",
                "forecast_rainfall_mm",
                "pattern_label",
                "anomaly_pct",
                "seasonal_phase",
            ]
        )

    best_model = {}
    if BEST_MODEL_PATH.exists():
        with open(BEST_MODEL_PATH, "r", encoding="utf-8") as handle:
            best_model = json.load(handle)

    return daily_df, monthly_history_df, monthly_forecasts_df, best_model


@st.cache_data(show_spinner=False)
def load_default_daily_climatology() -> pd.DataFrame:
    if DAILY_CLIMATOLOGY_PATH.exists():
        return pd.read_csv(DAILY_CLIMATOLOGY_PATH)
    daily_df = load_daily_rainfall_dataframe(RAW_CSV_PATH)
    return build_daily_climatology(
        daily_df,
        rain_threshold_mm=RAIN_THRESHOLD_MM,
        recency_decay=RECENCY_DECAY,
    )


def build_metric_card(label: str, value: str, note: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """


def probability_chart(profile_df: pd.DataFrame, top_day_number: int):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    colors = [
        "#d0d8da" if day != top_day_number else "#2d7370"
        for day in profile_df["day"]
    ]
    ax.bar(profile_df["day"], profile_df["rain_probability_pct"], color=colors, width=0.82)
    ax.set_title("Rain Chance by Day of Month", fontsize=14)
    ax.set_xlabel("Day")
    ax.set_ylabel("Rain chance (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(0.25, profile_df["day"].max() + 0.75)
    return fig


def daily_amount_chart(profile_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(
        profile_df["day"],
        profile_df["estimated_daily_rain_mm"],
        color="#9c4f2d",
        linewidth=2.3,
        marker="o",
        markersize=4,
    )
    ax.fill_between(
        profile_df["day"],
        profile_df["estimated_daily_rain_mm"],
        color="#d59a6f",
        alpha=0.28,
    )
    ax.set_title("Estimated Daily Rainfall Share", fontsize=14)
    ax.set_xlabel("Day")
    ax.set_ylabel("Estimated rainfall (mm)")
    ax.grid(alpha=0.25)
    ax.set_xlim(1, profile_df["day"].max())
    return fig


def month_selector_label(month_number: int) -> str:
    return calendar.month_name[month_number]


def main() -> None:
    st.set_page_config(
        page_title="Rainfall Outlook Explorer",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_styles()

    daily_df, monthly_history_df, monthly_forecasts_df, best_model = load_static_inputs()

    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin-bottom:0.45rem;">Rainfall Outlook Explorer</h1>
            <div style="font-size:1.05rem; line-height:1.55;">
                Pick a month, optionally enter your own monthly rainfall total, and explore
                which days historically carry the strongest chance of rainfall. The app blends
                the saved monthly forecast with day-of-month rainfall behavior learned from the
                historical daily record.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Month Setup")
        target_year = int(
            st.number_input(
                "Year",
                min_value=2026,
                max_value=2100,
                value=2026,
                step=1,
            )
        )
        target_month = int(
            st.selectbox(
                "Month",
                options=list(range(1, 13)),
                format_func=month_selector_label,
                index=3,
            )
        )

        total_mode = st.radio(
            "Monthly rainfall total",
            options=["Use saved forecast or climatology", "Enter my own monthly total"],
        )
        custom_total_mm = None
        if total_mode == "Enter my own monthly total":
            custom_total_mm = float(
                st.number_input(
                    "Monthly rainfall total (mm)",
                    min_value=0.0,
                    value=180.0,
                    step=5.0,
                )
            )

        with st.expander("Advanced Controls", expanded=False):
            rain_threshold_mm = float(
                st.slider(
                    "Rain-day threshold (mm)",
                    min_value=0.0,
                    max_value=20.0,
                    value=float(RAIN_THRESHOLD_MM),
                    step=0.1,
                )
            )
            recency_decay = float(
                st.slider(
                    "Recent-year emphasis",
                    min_value=0.90,
                    max_value=1.00,
                    value=float(RECENCY_DECAY),
                    step=0.01,
                )
            )
            top_n_days = int(
                st.slider(
                    "Top rainy days to highlight",
                    min_value=3,
                    max_value=10,
                    value=5,
                    step=1,
                )
            )

        st.caption(
            "If the selected month exists in `future_forecasts.csv`, the app uses the saved model forecast. "
            "Otherwise it falls back to the historical monthly average unless you enter your own total."
        )

    if rain_threshold_mm == RAIN_THRESHOLD_MM and recency_decay == RECENCY_DECAY:
        daily_climatology_df = load_default_daily_climatology()
    else:
        daily_climatology_df = build_daily_climatology(
            daily_df,
            rain_threshold_mm=rain_threshold_mm,
            recency_decay=recency_decay,
        )

    month_result = build_month_profile(
        target_year=target_year,
        target_month=target_month,
        monthly_forecasts_df=monthly_forecasts_df,
        monthly_history_df=monthly_history_df,
        daily_climatology_df=daily_climatology_df,
        custom_monthly_total_mm=custom_total_mm,
    )
    profile_df = month_result.profile.copy()

    st.subheader(f"{month_result.target_date.strftime('%B %Y')} Summary")
    intro_col, context_col = st.columns([2.1, 1.4], gap="large")

    with intro_col:
        best_day_text = (
            f"{month_result.top_chance_day['date'].strftime('%d %b %Y')} "
            f"({month_result.top_chance_day['weekday']})"
        )
        st.markdown(
            f"""
            <div class="note-panel">
                <strong>Highest rainfall chance:</strong> {best_day_text}<br>
                <strong>Rain chance:</strong> {month_result.top_chance_day['rain_probability_pct']:.1f}%<br>
                <strong>Estimated daily rainfall:</strong> {month_result.top_chance_day['estimated_daily_rain_mm']:.1f} mm
            </div>
            """,
            unsafe_allow_html=True,
        )

    with context_col:
        model_label = best_model.get("selected_model", "No saved model metadata")
        pattern_label = month_result.monthly_context.get("pattern_label") or "User-defined monthly total"
        seasonal_phase = month_result.monthly_context.get("seasonal_phase") or "Day-of-month climatology"
        st.markdown(
            f"""
            <div class="note-panel">
                <strong>Monthly source:</strong> {month_result.monthly_total_source}<br>
                <strong>Monthly pattern:</strong> {pattern_label}<br>
                <strong>Context:</strong> {seasonal_phase}<br>
                <strong>Saved model:</strong> {model_label}
            </div>
            """,
            unsafe_allow_html=True,
        )

    metric_columns = st.columns(4, gap="medium")
    metric_columns[0].markdown(
        build_metric_card(
            "Monthly total",
            f"{month_result.monthly_total_mm:.1f} mm",
            month_result.monthly_total_source,
        ),
        unsafe_allow_html=True,
    )
    metric_columns[1].markdown(
        build_metric_card(
            "Expected rainy days",
            f"{month_result.expected_rainy_days:.1f}",
            f"Threshold: {rain_threshold_mm:.1f} mm",
        ),
        unsafe_allow_html=True,
    )
    metric_columns[2].markdown(
        build_metric_card(
            "Highest chance day",
            month_result.top_chance_day["date"].strftime("%d %b"),
            f"{month_result.top_chance_day['rain_probability_pct']:.1f}% rain chance",
        ),
        unsafe_allow_html=True,
    )
    metric_columns[3].markdown(
        build_metric_card(
            "Highest estimated amount",
            month_result.top_amount_day["date"].strftime("%d %b"),
            f"{month_result.top_amount_day['estimated_daily_rain_mm']:.1f} mm estimated",
        ),
        unsafe_allow_html=True,
    )

    if month_result.monthly_context.get("anomaly_pct") is not None:
        st.caption(
            f"Saved forecast anomaly for the selected month: "
            f"{float(month_result.monthly_context['anomaly_pct']):.1f}% versus the historical monthly average."
        )

    chart_col1, chart_col2 = st.columns(2, gap="large")
    with chart_col1:
        st.pyplot(
            probability_chart(profile_df, int(month_result.top_chance_day["day"])),
            use_container_width=True,
        )
    with chart_col2:
        st.pyplot(daily_amount_chart(profile_df), use_container_width=True)

    st.subheader("Top Rainy Days")
    top_days_df = (
        profile_df.sort_values(
            ["rain_probability_pct", "estimated_daily_rain_mm", "day"],
            ascending=[False, False, True],
        )
        .head(top_n_days)
        .loc[
            :,
            [
                "date",
                "weekday",
                "rain_probability_pct",
                "estimated_daily_rain_mm",
                "weighted_mean_rain_mm",
                "weighted_rainy_day_intensity_mm",
            ],
        ]
        .rename(
            columns={
                "date": "Date",
                "weekday": "Weekday",
                "rain_probability_pct": "Rain Chance (%)",
                "estimated_daily_rain_mm": "Estimated Rain (mm)",
                "weighted_mean_rain_mm": "Historical Mean Rain (mm)",
                "weighted_rainy_day_intensity_mm": "Rainy-Day Intensity (mm)",
            }
        )
    )
    st.dataframe(
        top_days_df.style.format(
            {
                "Rain Chance (%)": "{:.1f}",
                "Estimated Rain (mm)": "{:.1f}",
                "Historical Mean Rain (mm)": "{:.1f}",
                "Rainy-Day Intensity (mm)": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Full Day-by-Day Profile")
    export_df = profile_df[
        [
            "date",
            "weekday",
            "rain_probability_pct",
            "estimated_daily_rain_mm",
            "weighted_mean_rain_mm",
            "weighted_rainy_day_intensity_mm",
            "chance_rank",
            "amount_rank",
        ]
    ].rename(
        columns={
            "date": "date",
            "weekday": "weekday",
            "rain_probability_pct": "rain_chance_pct",
            "estimated_daily_rain_mm": "estimated_daily_rain_mm",
            "weighted_mean_rain_mm": "historical_mean_rain_mm",
            "weighted_rainy_day_intensity_mm": "historical_rainy_day_intensity_mm",
            "chance_rank": "chance_rank",
            "amount_rank": "amount_rank",
        }
    )
    st.dataframe(
        export_df.style.format(
            {
                "rain_chance_pct": "{:.1f}",
                "estimated_daily_rain_mm": "{:.1f}",
                "historical_mean_rain_mm": "{:.1f}",
                "historical_rainy_day_intensity_mm": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download day-level profile as CSV",
        data=csv_bytes,
        file_name=f"rainfall_profile_{target_year}_{target_month:02d}.csv",
        mime="text/csv",
    )

    st.caption(
        "Method note: monthly totals come from the saved forecast, a historical monthly average, "
        "or your custom input. Day rankings are derived from historical day-of-month rainfall occurrence "
        "and rainy-day intensity, weighted slightly toward recent years."
    )


if __name__ == "__main__":
    main()

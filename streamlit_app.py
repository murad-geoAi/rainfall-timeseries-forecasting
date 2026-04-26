from __future__ import annotations

import calendar
import html
import json
from textwrap import dedent

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
from project_paths import (
    BEST_MODEL_METADATA_PATH,
    DAILY_CLIMATOLOGY_PATH,
    FUTURE_FORECASTS_PATH,
    MONTHLY_DATASET_PATH,
    RAW_DAILY_DATA_PATH,
)

ACCENT = "#ff4d6d"
ACCENT_SOFT = "#ffe6eb"
SURFACE = "#ffffff"
PANEL = "#0f172a"
PANEL_SOFT = "#172033"
TEXT = "#101827"
MUTED = "#5b6678"


def apply_custom_styles() -> None:
    st.markdown(
        dedent(
            f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(255, 77, 109, 0.12), transparent 30%),
                radial-gradient(circle at top right, rgba(35, 50, 83, 0.10), transparent 22%),
                linear-gradient(180deg, #fbfaf8 0%, #ffffff 46%, #f5f7fb 100%);
            color: {TEXT};
        }}
        .block-container,
        [data-testid="stMainBlockContainer"] {{
            max-width: 1180px;
            padding-top: clamp(3.4rem, 7vh, 5rem);
            padding-bottom: 4rem;
        }}
        h1, h2, h3 {{
            color: {TEXT};
            font-family: Georgia, "Times New Roman", serif;
            letter-spacing: -0.03em;
        }}
        .hero-safe-space {{
            height: 0.35rem;
        }}
        .eyebrow {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background: {ACCENT_SOFT};
            color: {ACCENT};
            border: 1px solid rgba(255, 77, 109, 0.18);
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }}
        .hero-title {{
            font-size: clamp(2.9rem, 5.5vw, 4.8rem);
            line-height: 1.04;
            font-weight: 700;
            letter-spacing: -0.06em;
            padding-top: 0.08rem;
            margin: 0 0 0.8rem 0;
            max-width: 12ch;
        }}
        .hero-title .accent {{
            color: {ACCENT};
        }}
        .hero-subtitle {{
            color: {MUTED};
            font-size: 1.02rem;
            line-height: 1.58;
            max-width: 35rem;
            margin-bottom: 1rem;
        }}
        .dashboard-card {{
            background: linear-gradient(180deg, {PANEL} 0%, #121c2f 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            padding: 1.15rem 1.15rem 1rem 1.15rem;
            box-shadow: 0 28px 52px rgba(15, 23, 42, 0.18);
            color: #eef2ff;
        }}
        .dashboard-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }}
        .dashboard-label {{
            color: #94a3b8;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        .dashboard-month {{
            font-size: 1.7rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }}
        .dashboard-chip {{
            background: rgba(34, 197, 94, 0.14);
            color: #4ade80;
            border: 1px solid rgba(74, 222, 128, 0.18);
            padding: 0.38rem 0.64rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
        }}
        .dash-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin-bottom: 0.95rem;
        }}
        .dash-metric {{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 16px;
            padding: 0.85rem 0.85rem 0.75rem 0.85rem;
        }}
        .dash-metric-label {{
            color: #94a3b8;
            font-size: 0.74rem;
            margin-bottom: 0.25rem;
        }}
        .dash-metric-value {{
            font-size: 1.35rem;
            font-weight: 700;
            color: #f8fafc;
        }}
        .dash-list {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            padding: 0.65rem 0.85rem;
        }}
        .dash-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.56rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            gap: 1rem;
        }}
        .dash-row:last-child {{
            border-bottom: 0;
        }}
        .dash-row-date {{
            color: #e2e8f0;
            font-weight: 600;
        }}
        .dash-row-sub {{
            color: #94a3b8;
            font-size: 0.8rem;
            margin-top: 0.08rem;
        }}
        .dash-badge {{
            color: {ACCENT};
            background: rgba(255, 77, 109, 0.12);
            border: 1px solid rgba(255, 77, 109, 0.18);
            border-radius: 999px;
            padding: 0.28rem 0.58rem;
            font-size: 0.78rem;
            font-weight: 700;
            white-space: nowrap;
        }}
        .stat-card {{
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(16, 24, 39, 0.08);
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: 0 18px 34px rgba(16, 24, 39, 0.07);
        }}
        .stat-label {{
            color: {MUTED};
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            margin-bottom: 0.28rem;
        }}
        .stat-value {{
            color: {TEXT};
            font-weight: 700;
            font-size: 1.55rem;
        }}
        .stat-note {{
            color: {MUTED};
            font-size: 0.86rem;
            margin-top: 0.22rem;
        }}
        .section-title {{
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {ACCENT};
            font-weight: 700;
            margin: 1.6rem 0 0.3rem 0;
        }}
        .section-headline {{
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.04em;
            color: {TEXT};
            margin-bottom: 1.1rem;
        }}
        .micro-copy {{
            color: {MUTED};
            font-size: 0.93rem;
        }}
        .control-caption {{
            color: {MUTED};
            font-size: 0.8rem;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }}
        .stDownloadButton > button,
        .stButton > button {{
            border-radius: 999px;
            min-height: 2.9rem;
            padding: 0 1.1rem;
            font-weight: 700;
            border: 1px solid rgba(16, 24, 39, 0.08);
        }}
        .stDownloadButton > button {{
            background: {ACCENT};
            color: white;
            border-color: rgba(255, 77, 109, 0.18);
        }}
        [data-testid="stExpander"] {{
            border: 1px solid rgba(16, 24, 39, 0.08);
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.84);
        }}
        [data-testid="stDataFrame"] {{
            border-radius: 18px;
            overflow: hidden;
        }}
        @media (max-width: 900px) {{
            .block-container,
            [data-testid="stMainBlockContainer"] {{
                padding-top: 2.8rem;
            }}
            .hero-safe-space {{
                height: 0.15rem;
            }}
            .hero-title {{
                font-size: 2.6rem;
                max-width: none;
            }}
            .hero-subtitle {{
                max-width: none;
                font-size: 0.98rem;
            }}
            .dash-grid {{
                grid-template-columns: 1fr;
            }}
            .dashboard-month {{
                font-size: 1.45rem;
            }}
        }}
        </style>
        """
        ),
        unsafe_allow_html=True,
    )


def load_monthly_history_from_raw(csv_path: Path) -> pd.DataFrame:
    raw_df = pd.read_csv(csv_path, parse_dates=["date"])
    monthly_df = (
        raw_df.set_index("date")
        .sort_index()[["rain_mm"]]
        .resample("MS")
        .sum()
        .reset_index()
    )
    return monthly_df


@st.cache_data(show_spinner=False)
def load_static_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    daily_df = load_daily_rainfall_dataframe(RAW_DAILY_DATA_PATH)

    if MONTHLY_DATASET_PATH.exists():
        monthly_history_df = pd.read_csv(MONTHLY_DATASET_PATH, parse_dates=["date"])
    else:
        monthly_history_df = load_monthly_history_from_raw(RAW_DAILY_DATA_PATH)

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
    if BEST_MODEL_METADATA_PATH.exists():
        with open(BEST_MODEL_METADATA_PATH, "r", encoding="utf-8") as handle:
            best_model = json.load(handle)

    return daily_df, monthly_history_df, monthly_forecasts_df, best_model


@st.cache_data(show_spinner=False)
def load_default_daily_climatology() -> pd.DataFrame:
    if DAILY_CLIMATOLOGY_PATH.exists():
        return pd.read_csv(DAILY_CLIMATOLOGY_PATH)
    daily_df = load_daily_rainfall_dataframe(RAW_DAILY_DATA_PATH)
    return build_daily_climatology(
        daily_df,
        rain_threshold_mm=RAIN_THRESHOLD_MM,
        recency_decay=RECENCY_DECAY,
    )


def format_source_label(source_name: str) -> str:
    if source_name == "Saved model forecast":
        return "Forecast"
    if source_name == "Historical monthly average":
        return "Climatology"
    return "Custom total"


def render_stat_card(label: str, value: str, note: str) -> str:
    return dedent(
        f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        <div class="stat-note">{note}</div>
    </div>
    """
    )


def render_dashboard_card(
    month_result,
    best_model: dict,
    top_days: pd.DataFrame,
) -> str:
    month_name = month_result.target_date.strftime("%B %Y")
    source_label = format_source_label(month_result.monthly_total_source)
    rows_markup = []
    for _, row in top_days.head(3).iterrows():
        rows_markup.extend(
            [
                '<div class="dash-row">',
                "<div>",
                f'<div class="dash-row-date">{html.escape(row["Date"].strftime("%d %b"))}</div>',
                f'<div class="dash-row-sub">{html.escape(str(row["Weekday"]))}</div>',
                "</div>",
                f'<div class="dash-badge">{row["Rain Chance (%)"]:.0f}%</div>',
                "</div>",
            ]
        )

    dashboard_lines = [
        '<div class="dashboard-card">',
        '<div class="dashboard-top">',
        "<div>",
        '<div class="dashboard-label">Rainfall Dashboard</div>',
        f'<div class="dashboard-month">{html.escape(month_name)}</div>',
        "</div>",
        f'<div class="dashboard-chip">{html.escape(source_label)}</div>',
        "</div>",
        '<div class="dash-grid">',
        '<div class="dash-metric">',
        '<div class="dash-metric-label">Monthly total</div>',
        f'<div class="dash-metric-value">{month_result.monthly_total_mm:.0f} mm</div>',
        "</div>",
        '<div class="dash-metric">',
        '<div class="dash-metric-label">Peak day</div>',
        f'<div class="dash-metric-value">{html.escape(month_result.top_chance_day["date"].strftime("%d %b"))}</div>',
        "</div>",
        '<div class="dash-metric">',
        '<div class="dash-metric-label">Peak chance</div>',
        f'<div class="dash-metric-value">{month_result.top_chance_day["rain_probability_pct"]:.0f}%</div>',
        "</div>",
        "</div>",
        '<div class="dash-list">',
        '<div class="dashboard-label" style="margin-bottom:0.35rem;">Top rainy days</div>',
        *rows_markup,
        "</div>",
        "</div>",
    ]
    return "\n".join(dashboard_lines)


def build_chance_curve_figure(profile_df: pd.DataFrame, highlight_day: int):
    chart_df = profile_df.copy()
    highlight_df = chart_df[chart_df["day"] == highlight_day]
    fig, ax = plt.subplots(figsize=(7.6, 3.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.fill_between(
        chart_df["day"],
        chart_df["rain_probability_pct"],
        color=ACCENT,
        alpha=0.16,
    )
    ax.plot(
        chart_df["day"],
        chart_df["rain_probability_pct"],
        color=ACCENT,
        linewidth=2.8,
    )
    if not highlight_df.empty:
        ax.scatter(
            highlight_df["day"],
            highlight_df["rain_probability_pct"],
            s=72,
            color=PANEL,
            zorder=3,
        )

    ax.set_xlim(1, int(chart_df["day"].max()))
    ax.set_xlabel("")
    ax.set_ylabel("Rain chance (%)", color=MUTED)
    ax.grid(axis="y", color="#eceff5", linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d5dae2")
    ax.spines["bottom"].set_color("#d5dae2")
    ax.tick_params(axis="x", colors=MUTED)
    ax.tick_params(axis="y", colors=MUTED)
    fig.tight_layout()
    return fig


def build_top_days_figure(profile_df: pd.DataFrame, top_n_days: int):
    top_df = (
        profile_df.sort_values(
            ["rain_probability_pct", "estimated_daily_rain_mm", "day"],
            ascending=[False, False, True],
        )
        .head(top_n_days)
        .copy()
    )
    top_df["date_label"] = top_df["date"].dt.strftime("%d %b")
    top_df = top_df.sort_values("rain_probability_pct", ascending=True)

    fig_height = max(3.0, top_n_days * 0.72)
    fig, ax = plt.subplots(figsize=(6.2, fig_height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    bars = ax.barh(
        top_df["date_label"],
        top_df["rain_probability_pct"],
        color=ACCENT,
        alpha=0.92,
    )

    max_value = float(top_df["rain_probability_pct"].max()) if not top_df.empty else 0.0
    ax.set_xlim(0, max(10.0, max_value * 1.18))
    ax.set_xlabel("Rain chance (%)", color=MUTED)
    ax.set_ylabel("")
    ax.grid(axis="x", color="#eceff5", linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#d5dae2")
    ax.tick_params(axis="x", colors=MUTED)
    ax.tick_params(axis="y", colors=MUTED, length=0)

    for bar, value in zip(bars, top_df["rain_probability_pct"]):
        ax.text(
            bar.get_width() + max(0.6, max_value * 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}%",
            va="center",
            ha="left",
            color=TEXT,
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()
    return fig


def month_name(month_number: int) -> str:
    return calendar.month_name[month_number]


def main() -> None:
    st.set_page_config(
        page_title="Rainfall Day Finder",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_custom_styles()

    daily_df, monthly_history_df, monthly_forecasts_df, best_model = load_static_inputs()

    st.markdown("<div class='hero-safe-space'></div>", unsafe_allow_html=True)
    hero_left, hero_right = st.columns([1.06, 0.94], gap="large")

    with hero_left:
        st.markdown(
            dedent(
                """
            <div class="eyebrow">Rainfall Intelligence</div>
            <div class="hero-title">
                Find the <span class="accent">rainiest days</span> before the month begins
            </div>
            <div class="hero-subtitle">
                Pick a month. The app ranks the days most likely to receive rainfall based on historical data and deep learning forecasts.
            </div>
            """
            ),
            unsafe_allow_html=True,
        )

        control_row = st.columns([1.2, 0.85], gap="medium")
        with control_row[0]:
            st.markdown("<div class='control-caption'>Month</div>", unsafe_allow_html=True)
            target_month = int(
                st.selectbox(
                    "Month",
                    options=list(range(1, 13)),
                    format_func=month_name,
                    index=6,
                    label_visibility="collapsed",
                )
            )
        with control_row[1]:
            st.markdown("<div class='control-caption'>Year</div>", unsafe_allow_html=True)
            target_year = int(
                st.number_input(
                    "Year",
                    min_value=2001,
                    max_value=2100,
                    value=2026,
                    step=1,
                    label_visibility="collapsed",
                )
            )

        custom_total_mm = None

        with st.expander("Fine tune", expanded=False):
            advanced_cols = st.columns(3, gap="medium")
            with advanced_cols[0]:
                rain_threshold_mm = float(
                    st.slider(
                        "Rain-day threshold (mm)",
                        min_value=0.0,
                        max_value=20.0,
                        value=float(RAIN_THRESHOLD_MM),
                        step=0.1,
                    )
                )
            with advanced_cols[1]:
                recency_decay = float(
                    st.slider(
                        "Recent-year emphasis",
                        min_value=0.90,
                        max_value=1.00,
                        value=float(RECENCY_DECAY),
                        step=0.01,
                    )
                )
            with advanced_cols[2]:
                top_n_days = int(
                    st.slider(
                        "Top days to show",
                        min_value=3,
                        max_value=10,
                        value=5,
                        step=1,
                    )
                )
        if "top_n_days" not in locals():
            top_n_days = 5
        if "rain_threshold_mm" not in locals():
            rain_threshold_mm = float(RAIN_THRESHOLD_MM)
        if "recency_decay" not in locals():
            recency_decay = float(RECENCY_DECAY)

    if (
        rain_threshold_mm == float(RAIN_THRESHOLD_MM)
        and recency_decay == float(RECENCY_DECAY)
    ):
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
            ],
        ]
        .rename(
            columns={
                "date": "Date",
                "weekday": "Weekday",
                "rain_probability_pct": "Rain Chance (%)",
                "estimated_daily_rain_mm": "Est. Rain (mm)",
            }
        )
    )

    with hero_right:
        st.markdown(
            render_dashboard_card(
                month_result=month_result,
                best_model=best_model,
                top_days=top_days_df,
            ),
            unsafe_allow_html=True,
        )

    stat_columns = st.columns(4, gap="medium")
    stat_columns[0].markdown(
        render_stat_card(
            "Monthly total",
            f"{month_result.monthly_total_mm:.0f} mm",
            format_source_label(month_result.monthly_total_source),
        ),
        unsafe_allow_html=True,
    )
    stat_columns[1].markdown(
        render_stat_card(
            "Top day",
            month_result.top_chance_day["date"].strftime("%d %b"),
            month_result.top_chance_day["weekday"],
        ),
        unsafe_allow_html=True,
    )
    stat_columns[2].markdown(
        render_stat_card(
            "Peak chance",
            f"{month_result.top_chance_day['rain_probability_pct']:.0f}%",
            "Highest daily rain probability",
        ),
        unsafe_allow_html=True,
    )
    stat_columns[3].markdown(
        render_stat_card(
            "Rainy days",
            f"{month_result.expected_rainy_days:.1f}",
            f"Threshold {rain_threshold_mm:.1f} mm",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>Daily Pattern</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-headline'>Chance curve and top-day ranking</div>", unsafe_allow_html=True)

    charts_left, charts_right = st.columns([1.35, 1.0], gap="large")
    with charts_left:
        st.pyplot(
            build_chance_curve_figure(
                profile_df,
                int(month_result.top_chance_day["day"]),
            ),
            use_container_width=True,
            clear_figure=True,
        )
    with charts_right:
        st.pyplot(
            build_top_days_figure(profile_df, top_n_days),
            use_container_width=True,
            clear_figure=True,
        )

    table_header_left, table_header_right = st.columns([1, 1], gap="medium")
    with table_header_left:
        st.markdown("<div class='section-title'>Top Days</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-headline' style='font-size:1.55rem;'>Best rainfall windows</div>",
            unsafe_allow_html=True,
        )
    with table_header_right:
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
        st.download_button(
            "Download CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"rainfall_profile_{target_year}_{target_month:02d}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.dataframe(
        top_days_df,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
            "Rain Chance (%)": st.column_config.NumberColumn(
                "Chance (%)",
                format="%.1f",
            ),
            "Est. Rain (mm)": st.column_config.NumberColumn(
                "Est. Rain (mm)",
                format="%.1f",
            ),
        },
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Full day-by-day profile", expanded=False):
        st.dataframe(
            export_df,
            column_config={
                "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "rain_chance_pct": st.column_config.NumberColumn("Chance (%)", format="%.1f"),
                "estimated_daily_rain_mm": st.column_config.NumberColumn(
                    "Est. Rain (mm)",
                    format="%.1f",
                ),
                "historical_mean_rain_mm": st.column_config.NumberColumn(
                    "Hist. Mean (mm)",
                    format="%.1f",
                ),
                "historical_rainy_day_intensity_mm": st.column_config.NumberColumn(
                    "Rainy-Day Intensity (mm)",
                    format="%.1f",
                ),
            },
            use_container_width=True,
            hide_index=True,
        )
        pattern_label = month_result.monthly_context.get("pattern_label") or "Custom"
        anomaly_pct = month_result.monthly_context.get("anomaly_pct")
        footer_bits = [
            f"Pattern: {pattern_label}",
            f"Source: {format_source_label(month_result.monthly_total_source)}",
        ]
        if anomaly_pct is not None:
            footer_bits.append(f"Anomaly: {float(anomaly_pct):.1f}%")
        st.caption(" | ".join(footer_bits))


if __name__ == "__main__":
    main()

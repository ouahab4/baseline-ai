"""Baseline AI dashboard (COMPILIX INNOVATIONS).

Run:  streamlit run app/dashboard.py
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_PROCESSED, FIGURES, MODELS, REPORTS  # noqa: E402
from src.features import DAY_AHEAD_FEATURES, WEATHER_MODEL_FEATURES  # noqa: E402

st.set_page_config(page_title="Baseline AI", page_icon="⚡", layout="wide")

GREY, ACCENT, BLACK, RED = "#9aa0a6", "#1f6fd1", "#222222", "#d93025"


@st.cache_data
def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED / "predictions.parquet", filters=[("split", "==", "test")])
    df["building_id"] = df["building_id"].astype(str)
    return df


@st.cache_data
def load_daily() -> pd.DataFrame:
    d = pd.read_parquet(DATA_PROCESSED / "daily_anomalies.parquet")
    d["building_id"] = d["building_id"].astype(str)
    return d


@st.cache_data
def load_features(building: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / "test_features.parquet", filters=[("building_id", "==", building)])


@st.cache_resource
def load_model(name: str) -> lgb.Booster:
    return lgb.Booster(model_file=str(MODELS / f"{name}.txt"))


def metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    return float(np.sqrt(np.mean((y - p) ** 2)) / y.mean() * 100), float((y - p).sum() / (len(y) * y.mean()) * 100)


pred = load_predictions()
daily = load_daily()
comparison = pd.read_csv(REPORTS / "model_comparison.csv")
validation = json.loads((REPORTS / "validation.json").read_text())

# ---------------- Sidebar ----------------
st.sidebar.title("⚡ Baseline AI")
st.sidebar.caption("Hourly electricity forecasts for 591 real buildings. Test period: Jul–Dec 2017. "
                   "Developed by COMPILIX INNOVATIONS, SPASU BAC Students.")
sites = sorted(pred["site"].astype(str).unique())
site = st.sidebar.selectbox("Site", sites)
uses = sorted(pred.loc[pred["site"] == site, "primary_use"].astype(str).unique())
use = st.sidebar.selectbox("Building type", ["all"] + uses)
mask = pred["site"] == site
if use != "all":
    mask &= pred["primary_use"] == use
building = st.sidebar.selectbox("Building", sorted(pred.loc[mask, "building_id"].unique()))
b = pred[pred["building_id"] == building].sort_values("timestamp")

tab_overview, tab_forecast, tab_anom, tab_explain, tab_mv = st.tabs(
    ["Overview", "Forecast", "Anomalies", "Explain (SHAP)", "Savings (M&V)"])

# ---------------- Overview ----------------
with tab_overview:
    best = comparison.sort_values("median_cv_rmse_pct").iloc[0]
    base = comparison[comparison["column"] == "baseline_last_week"].iloc[0]
    weather = comparison[comparison["column"] == "lgbm_weather"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", best["model"].replace(" (day-ahead model)", ""), f"{best['median_cv_rmse_pct']:.1f}% CV-RMSE",
              delta_color="off")
    c2.metric("Error cut vs naive baseline",
              f"{(1 - best['median_cv_rmse_pct'] / base['median_cv_rmse_pct']) * 100:.0f}%")
    c3.metric("Within hourly accuracy limits (weather model)", f"{weather['ashrae_g14_pass_pct']:.0f}%")
    c4.metric("Injected faults detected", f"{validation['anomaly_detection']['fault_days_detected_pct']:.0f}%")

    st.subheader("Model comparison on unseen months (Jul–Dec 2017)")
    show = comparison.drop(columns="column").rename(columns={
        "median_cv_rmse_pct": "Median CV-RMSE %", "median_abs_nmbe_pct": "Median |NMBE| %",
        "median_mape_pct": "Median MAPE %", "rmsle": "RMSLE", "mae_kwh": "MAE kWh",
        "ashrae_g14_pass_pct": "Within accuracy limits %", "n_buildings": "Buildings"})
    st.dataframe(show, hide_index=True, width="stretch")
    col_a, col_b = st.columns(2)
    for col, fig in [(col_a, "model_comparison.png"), (col_b, "error_distribution.png")]:
        if (FIGURES / fig).exists():
            col.image(str(FIGURES / fig))

# ---------------- Forecast ----------------
with tab_forecast:
    dmin, dmax = b["timestamp"].min().date(), b["timestamp"].max().date()
    start = st.date_input("Start date", value=pd.Timestamp("2017-10-02").date(), min_value=dmin, max_value=dmax)
    days = st.slider("Days to show", 3, 60, 14)
    w = b[(b["timestamp"] >= pd.Timestamp(start)) & (b["timestamp"] < pd.Timestamp(start) + pd.Timedelta(days=days))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w["timestamp"], y=w["kwh"], name="Actual", line=dict(color=BLACK, width=1.6)))
    fig.add_trace(go.Scatter(x=w["timestamp"], y=w["lgbm_weather"], name="Weather model", line=dict(color=GREY)))
    fig.add_trace(go.Scatter(x=w["timestamp"], y=w["lgbm_day_ahead"], name="Day-ahead model", line=dict(color=ACCENT)))
    fig.update_layout(height=420, yaxis_title="kWh per hour", margin=dict(t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    c1, c2, c3 = st.columns(3)
    cv_w, nm_w = metrics(b["kwh"].to_numpy(), b["lgbm_weather"].to_numpy())
    cv_d, nm_d = metrics(b["kwh"].to_numpy(), b["lgbm_day_ahead"].to_numpy())
    c1.metric("Day-ahead CV-RMSE (whole test period)", f"{cv_d:.1f}%")
    c2.metric("Weather model CV-RMSE", f"{cv_w:.1f}%")
    c3.metric("Weather model NMBE", f"{nm_w:+.1f}%")
    fig_t = go.Figure(go.Scattergl(x=b["temperature"], y=b["kwh"], mode="markers",
                                   marker=dict(size=3, color=ACCENT, opacity=0.35)))
    fig_t.update_layout(height=300, xaxis_title="Outdoor temperature (°C)", yaxis_title="kWh per hour",
                        title="Consumption vs temperature (test period)", margin=dict(t=40, b=10))
    st.plotly_chart(fig_t, width="stretch")

# ---------------- Anomalies ----------------
with tab_anom:
    bd = daily[daily["building_id"] == building].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bd["date"], y=bd["actual"], name="Actual daily kWh",
                         marker_color=np.where(bd["anomaly"], RED, GREY)))
    fig.add_trace(go.Scatter(x=bd["date"], y=bd["expected"], name="Expected (weather model)",
                             line=dict(color=ACCENT, width=2)))
    fig.update_layout(height=380, yaxis_title="kWh per day", margin=dict(t=30, b=10), legend=dict(orientation="h"),
                      title="Red days used far more energy than the weather and calendar explain")
    st.plotly_chart(fig, width="stretch")
    n_anom = int(bd["anomaly"].sum())
    extra = float((bd.loc[bd["anomaly"], "actual"] - bd.loc[bd["anomaly"], "expected"]).sum())
    st.write(f"**{n_anom} anomalous days** for this building, **{extra:,.0f} kWh** above expectation.")

    st.subheader("Largest anomaly events across all buildings")
    events = pd.read_csv(REPORTS / "anomaly_events.csv")
    st.dataframe(events.head(25)[["building_id", "start", "end", "days", "extra_kwh", "excess_pct"]].round(0),
                 hide_index=True, width="stretch")
    v = validation["anomaly_detection"]
    st.caption(f"Validation: {v['fault_size']} faults injected into {v['buildings_tested']} buildings → "
               f"{v['fault_days_detected_pct']}% of fault days detected, {v['normal_days_flagged_pct']}% of normal days flagged.")

# ---------------- Explain ----------------
with tab_explain:
    model_name = st.radio("Model", ["lgbm_weather", "lgbm_day_ahead"], horizontal=True,
                          format_func=lambda m: "Weather model" if m == "lgbm_weather" else "Day-ahead model")
    feats = WEATHER_MODEL_FEATURES if model_name == "lgbm_weather" else DAY_AHEAD_FEATURES
    fx = load_features(building).sort_values("timestamp")
    when = st.select_slider("Hour to explain", options=list(fx["timestamp"]), value=fx["timestamp"].iloc[len(fx) // 2])
    row = fx[fx["timestamp"] == when]
    contrib = load_model(model_name).predict(row[feats], pred_contrib=True)[0]
    base_value, contrib = contrib[-1], contrib[:-1]
    expl = pd.DataFrame({"feature": feats, "value": row[feats].iloc[0].astype(str).to_numpy(), "shap": contrib})
    expl = expl.reindex(expl["shap"].abs().sort_values(ascending=False).index).head(12).iloc[::-1]
    fig = go.Figure(go.Bar(x=expl["shap"], y=expl["feature"] + " = " + expl["value"].str.slice(0, 18), orientation="h",
                           marker_color=np.where(expl["shap"] > 0, RED, ACCENT)))
    fig.update_layout(height=430, xaxis_title="Contribution to log(1 + kWh)  (red raises, blue lowers)",
                      margin=dict(t=30, b=10), title=f"Why the model predicts {np.expm1(base_value + contrib.sum()):,.1f} kWh "
                      f"at {pd.Timestamp(when):%Y-%m-%d %H:00}")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Average building-hour starts at {np.expm1(base_value):,.1f} kWh; each bar shows how one feature moves it.")
    col_a, col_b = st.columns(2)
    for col, fig_name in [(col_a, f"shap_summary_{model_name}.png"), (col_b, "shap_temperature_dependence.png")]:
        if (FIGURES / fig_name).exists():
            col.image(str(FIGURES / fig_name))

# ---------------- Savings (M&V) ----------------
with tab_mv:
    st.write("Simulate an energy-efficiency retrofit and check how the model measures the saving. "
             "The model is calibrated on the period before the retrofit, then predicts what the building "
             "*would have used* without it.")
    c1, c2 = st.columns(2)
    retrofit = c1.date_input("Retrofit date", value=pd.Timestamp("2017-10-01").date(),
                             min_value=pd.Timestamp("2017-08-01").date(), max_value=pd.Timestamp("2017-12-01").date())
    saving = c2.slider("True saving applied after the retrofit (%)", 0, 40, 15)
    g = b.copy()
    post = g["timestamp"] >= pd.Timestamp(retrofit)
    g.loc[post, "kwh"] *= 1 - saving / 100
    recent = ~post & (g["timestamp"] >= pd.Timestamp(retrofit) - pd.Timedelta(weeks=8))  # calibrate on 8 weeks before
    calibration = g.loc[recent, "kwh"].sum() / g.loc[recent, "lgbm_weather"].sum()
    g["baseline"] = g["lgbm_weather"] * calibration
    est = 1 - g.loc[post, "kwh"].sum() / g.loc[post, "baseline"].sum()
    gd = g.set_index("timestamp")[["kwh", "baseline"]].resample("D").sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gd["timestamp"], y=gd["baseline"], name="Expected without retrofit",
                             line=dict(color=GREY, width=2)))
    fig.add_trace(go.Scatter(x=gd["timestamp"], y=gd["kwh"], name="Actual (with simulated retrofit)",
                             line=dict(color=ACCENT, width=2)))
    fig.add_vline(x=pd.Timestamp(retrofit), line_dash="dash", line_color=BLACK)
    fig.update_layout(height=380, yaxis_title="kWh per day", margin=dict(t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("True saving", f"{saving:.1f}%")
    c2.metric("Saving measured by the model", f"{est * 100:.1f}%", f"{est * 100 - saving:+.1f} pts error", delta_color="off")
    c3.metric("Energy saved", f"{(g.loc[post, 'baseline'].sum() - g.loc[post, 'kwh'].sum()):,.0f} kWh")
    v = validation["savings_verification"]
    st.caption(f"Validated method (baseline retrained up to the retrofit): across {v['buildings_tested']} buildings with a 15% "
               f"simulated saving, the median error was {v['median_abs_error_pct_points']} points; "
               f"{v['within_10_points_pct']}% were within 10 points.")

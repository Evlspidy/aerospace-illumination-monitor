import io
import math
import struct
import wave
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh


# =========================================================
# AERO MISSION X | Software-only illumination simulation
# =========================================================

st.set_page_config(
    page_title="Aerospace Illumination Monitoring System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- THEME -------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root {
        --bg: #07111f; --panel: #0d1b2c; --cyan: #22d3ee;
        --purple: #c084fc; --muted: #9bb0c8;
    }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background:
          radial-gradient(ellipse at 82% 0%, rgba(31,91,139,.20), transparent 34%),
          linear-gradient(135deg, #050d18 0%, #091626 55%, #07111f 100%);
        color: #edf6ff;
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1600px; }
    h1, h2, h3 { color: #f3f8ff !important; }
    p, label, .stCaption { color: #b9c9dc; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1b2d, #071321);
        border-right: 1px solid rgba(34,211,238,.25);
    }
    [data-testid="stSidebar"] * { color: #dcecff; }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(16,36,58,.96), rgba(9,25,42,.96));
        border: 1px solid rgba(34,211,238,.32);
        border-radius: 13px; padding: 14px 16px;
        box-shadow: 0 0 18px rgba(34,211,238,.06);
    }
    div[data-testid="stMetric"] label { color: #a9bfd5 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f3f8ff !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid rgba(34,211,238,.27) !important;
        border-radius: 14px !important;
        background: linear-gradient(145deg, rgba(13,27,44,.92), rgba(8,20,34,.94));
    }
    .stButton > button, .stDownloadButton > button {
        border: 1px solid rgba(34,211,238,.65); border-radius: 9px;
        background: linear-gradient(100deg, #0d3651, #142747);
        color: #ecfbff; font-weight: 600;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #22d3ee; color: white;
        box-shadow: 0 0 14px rgba(34,211,238,.25);
    }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        background: #0b1c2e; border-color: rgba(34,211,238,.35);
    }
    hr { border-color: rgba(34,211,238,.22); }
    .small-label, .section-kicker {
        color: #22d3ee; font-size: .74rem; font-weight: 700;
        letter-spacing: 1.5px; text-transform: uppercase;
    }
    .hero {
        padding: 18px 22px; border: 1px solid rgba(34,211,238,.35);
        border-radius: 16px;
        background: linear-gradient(105deg, rgba(12,34,55,.98), rgba(15,24,49,.94));
        box-shadow: 0 0 24px rgba(34,211,238,.07); margin-bottom: 18px;
    }
    .hero-title { font-size: 1.65rem; font-weight: 800; color: #f4faff; }
    .hero-sub { color: #a9bfd5; font-size: .9rem; margin-top: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- CONSTANTS -----------------------
CABIN_THRESHOLD = 100
LAB_THRESHOLD = 300
CYAN = "#22d3ee"
PURPLE = "#c084fc"
GREEN = "#34d399"
RED = "#fb7185"

# -------------------- SESSION STATE ---------------------
defaults = {
    "history": [],
    "alert_log": [],
    "previous_alerts": [],
    "alarm_muted": False,
}
for key, default_value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ------------------------- HELPERS ----------------------
def create_alarm_sound():
    sample_rate = 44100
    duration = 1.0
    frequency = 880
    amplitude = 0.25
    frames = []
    for index in range(int(sample_rate * duration)):
        time_value = index / sample_rate
        pulse = 1 if (time_value % 0.6) < 0.25 else 0
        sample = amplitude * math.sin(2 * math.pi * frequency * time_value) * pulse
        frames.append(struct.pack("<h", int(sample * 32767)))

    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(b"".join(frames))
    return audio_buffer.getvalue()


def detect_anomaly(cabin_value, lab_value):
    threshold_anomaly = (
        cabin_value < CABIN_THRESHOLD or lab_value < LAB_THRESHOLD
    )
    ml_anomaly = False

    history = st.session_state.history
    if len(history) >= 10:
        training_df = pd.DataFrame(history)
        training_data = training_df[["Aircraft Cabin", "Laboratory"]].values
        model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
        )
        model.fit(training_data)
        ml_anomaly = model.predict([[cabin_value, lab_value]])[0] == -1

    return {
        "threshold_anomaly": threshold_anomaly,
        "ml_anomaly": ml_anomaly,
        "combined_anomaly": threshold_anomaly or ml_anomaly,
    }


def style_plot(figure, height=320):
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#dcecff", "family": "Inter"},
        margin={"l": 18, "r": 18, "t": 55, "b": 18},
        height=height,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "right", "x": 1},
        title_font={"size": 15, "color": "#edf6ff"},
    )
    figure.update_xaxes(gridcolor="rgba(148,170,195,.13)")
    figure.update_yaxes(gridcolor="rgba(148,170,195,.13)")
    return figure


def create_gauge(value, title, threshold, accent):
    safe_value = max(0, min(float(value), 1000))
    bar_color = GREEN if value >= threshold else RED
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=safe_value,
            title={"text": title, "font": {"size": 15, "color": "#dcecff"}},
            number={"suffix": " lux", "font": {"size": 27, "color": "#f4faff"}},
            gauge={
                "axis": {"range": [0, 1000], "tickcolor": "#8fa9c5"},
                "bar": {"color": bar_color, "thickness": 0.24},
                "bgcolor": "#10243a",
                "borderwidth": 1,
                "bordercolor": "rgba(34,211,238,.25)",
                "steps": [
                    {"range": [0, threshold], "color": "rgba(251,113,133,.18)"},
                    {"range": [threshold, 1000], "color": "rgba(52,211,153,.12)"},
                ],
                "threshold": {
                    "line": {"color": accent, "width": 3},
                    "thickness": 0.8,
                    "value": threshold,
                },
            },
        )
    )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin={"t": 45, "b": 10, "l": 20, "r": 20},
        font={"color": "#dcecff"},
    )
    return figure


# ------------------- SIDEBAR + INPUTS -------------------
st_autorefresh(interval=5000, limit=None, key="aero_refresh")

st.sidebar.markdown("## ✈️ Aerospace Illumination Monitoring")
st.sidebar.caption("Aerospace illumination monitoring")
st.sidebar.markdown("---")
st.sidebar.markdown("**MISSION STATUS**")
st.sidebar.success("SIMULATION ONLINE")
st.sidebar.markdown("**MONITORING CHANNELS**")
st.sidebar.markdown(
    "- Aircraft cabin illumination\n"
    "- Aerospace laboratory illumination\n"
    "- AI anomaly monitoring\n"
    "- Predictive trend analysis"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**DEMONSTRATION LIMITS**")
st.sidebar.metric("Cabin threshold", f"{CABIN_THRESHOLD} lux")
st.sidebar.metric("Laboratory threshold", f"{LAB_THRESHOLD} lux")
st.sidebar.caption("Illustrative software-demo values only.")

st.markdown(
    """
    <div class="hero">
      <div class="small-label">AEROSPACE ILLUMINATION MONITORING</div>
      <div class="hero-title">Aerospace Illumination Monitoring System</div>
      <div class="hero-sub">Aircraft Cabin | Aerospace Laboratory · Software Simulation</div>
    </div>
    """,
    unsafe_allow_html=True,
)

head_left, head_right = st.columns([3, 1])
with head_left:
    st.markdown("### Mission Overview")
    st.caption("Live simulated illumination telemetry and system condition.")
with head_right:
    st.markdown(
        "<div style='text-align:right;color:#34d399;font-weight:700;padding-top:12px'>"
        "● SIMULATION ACTIVE</div>",
        unsafe_allow_html=True,
    )

with st.container(border=True):
    st.markdown('<div class="section-kicker">01 / Input controls</div>',
                unsafe_allow_html=True)
    st.subheader("Illumination Input Simulation")
    input_left, input_right = st.columns(2)
    with input_left:
        st.markdown("**✈️ Aircraft Cabin**")
        cabin_light = st.number_input(
            "Cabin illumination (lux)", min_value=0, max_value=2000,
            value=350, step=10, key="cabin_input"
        )
    with input_right:
        st.markdown("**⌁ Aerospace Laboratory**")
        lab_light = st.number_input(
            "Laboratory illumination (lux)", min_value=0, max_value=2000,
            value=600, step=10, key="lab_input"
        )

# One sample is captured each five-second rerun. Inputs are simulated/manual.
st.session_state.history.append({
    "Time": datetime.now().strftime("%H:%M:%S"),
    "Aircraft Cabin": cabin_light,
    "Laboratory": lab_light,
})
st.session_state.history = st.session_state.history[-100:]

cabin_normal = cabin_light >= CABIN_THRESHOLD
lab_normal = lab_light >= LAB_THRESHOLD

active_alerts = []
if not cabin_normal:
    active_alerts.append("Low Aircraft Cabin Illumination")
if not lab_normal:
    active_alerts.append("Low Aerospace Laboratory Illumination")

# Record changes in alert state, not every refresh.
if active_alerts != st.session_state.previous_alerts:
    if active_alerts:
        event_name = "EMERGENCY ALERT"
        severity = "CRITICAL"
        detection_method = "Threshold-Based Detection"
        details = ", ".join(active_alerts)
    else:
        event_name = "SYSTEM RECOVERED"
        severity = "NORMAL"
        detection_method = "System Monitoring"
        details = "All readings are above demonstration thresholds"

    st.session_state.alert_log.append({
        "Date & Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "Event": event_name,
        "Severity": severity,
        "Detection Method": detection_method,
        "Cabin Illumination (lux)": cabin_light,
        "Laboratory Illumination (lux)": lab_light,
        "Details": details,
    })
    st.session_state.alert_log = st.session_state.alert_log[-50:]
    st.session_state.previous_alerts = active_alerts.copy()

if not active_alerts:
    st.session_state.alarm_muted = False

anomaly_result = detect_anomaly(cabin_light, lab_light)

# ----------------------- KPI ROW -------------------------
with st.container(border=True):
    st.markdown('<div class="section-kicker">02 / Primary telemetry</div>',
                unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("AIRCRAFT CABIN", f"{cabin_light} lux",
              "NORMAL" if cabin_normal else "LOW ILLUMINATION")
    k2.metric("AEROSPACE LAB", f"{lab_light} lux",
              "NORMAL" if lab_normal else "LOW ILLUMINATION")
    k3.metric("AI MONITORING",
              "ANOMALY" if anomaly_result["combined_anomaly"] else "NOMINAL",
              "Isolation Forest + thresholds")
    k4.metric("RECORDED SAMPLES", str(len(st.session_state.history)),
              "Latest 100 retained")

# ---------------------- LIVE STATUS ----------------------
with st.container(border=True):
    st.markdown('<div class="section-kicker">03 / System condition</div>',
                unsafe_allow_html=True)
    st.subheader("Live Monitoring Status")

    if anomaly_result["combined_anomaly"]:
        st.error("🚨 ANOMALY DETECTED — REVIEW REQUIRED")
        if anomaly_result["threshold_anomaly"]:
            st.warning(
                "Threshold detection: one or more readings are below the "
                "configured demonstration limit."
            )
        if anomaly_result["ml_anomaly"]:
            st.warning(
                "Isolation Forest: current reading differs from the "
                "learned historical pattern."
            )
    else:
        st.success("● AI MONITORING: NO ANOMALIES DETECTED")

    status_left, status_right = st.columns(2)
    with status_left:
        st.markdown("**✈️ Aircraft Cabin**")
        st.metric("Illumination intensity", f"{cabin_light} lux")
        if cabin_normal:
            st.success("CABIN STATUS: NORMAL")
        else:
            st.error("CABIN STATUS: LOW")

    with status_right:
        st.markdown("**⌁ Aerospace Laboratory**")
        st.metric("Illumination intensity", f"{lab_light} lux")
        if lab_normal:
            st.success("LAB STATUS: NORMAL")
        else:
            st.error("LAB STATUS: LOW")

# ------------------------- GAUGES -------------------------
with st.container(border=True):
    st.markdown('<div class="section-kicker">04 / Instrument panel</div>',
                unsafe_allow_html=True)
    st.subheader("Illumination Monitoring Gauges")
    gauge_left, gauge_right = st.columns(2)
    with gauge_left:
        st.plotly_chart(
            create_gauge(cabin_light, "AIRCRAFT CABIN",
                         CABIN_THRESHOLD, CYAN),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with gauge_right:
        st.plotly_chart(
            create_gauge(lab_light, "AEROSPACE LABORATORY",
                         LAB_THRESHOLD, PURPLE),
            use_container_width=True,
            config={"displayModeBar": False},
        )

# -------------------- LIVE TELEMETRY ---------------------
history_df = pd.DataFrame(st.session_state.history)
with st.container(border=True):
    st.markdown('<div class="section-kicker">05 / Live telemetry</div>',
                unsafe_allow_html=True)
    st.subheader("Illumination History — Real Time")
    if not history_df.empty:
        live_figure = go.Figure()
        live_figure.add_trace(go.Scatter(
            x=history_df["Time"], y=history_df["Aircraft Cabin"],
            mode="lines+markers", name="Aircraft Cabin",
            line={"color": CYAN, "width": 2.5}, marker={"size": 5},
        ))
        live_figure.add_trace(go.Scatter(
            x=history_df["Time"], y=history_df["Laboratory"],
            mode="lines+markers", name="Aerospace Laboratory",
            line={"color": PURPLE, "width": 2.5}, marker={"size": 5},
        ))
        live_figure.update_layout(
            title="Live Illumination (lux)", yaxis_title="Illumination (lux)"
        )
        st.plotly_chart(style_plot(live_figure, 360),
                        use_container_width=True,
                        config={"displayModeBar": False})

# ---------------------- COMPARISON -----------------------
with st.container(border=True):
    st.markdown('<div class="section-kicker">06 / Comparative analysis</div>',
                unsafe_allow_html=True)
    st.subheader("Cabin vs Laboratory Comparison")
    comparison_df = pd.DataFrame({
        "Location": ["Aircraft Cabin", "Aerospace Laboratory"],
        "Illumination (lux)": [cabin_light, lab_light],
    })
    comparison_figure = go.Figure(go.Bar(
        x=comparison_df["Location"],
        y=comparison_df["Illumination (lux)"],
        marker_color=[CYAN, PURPLE],
        text=comparison_df["Illumination (lux)"],
        textposition="outside",
    ))
    comparison_figure.update_layout(
        title="Current Illumination Levels", yaxis_title="Lux"
    )
    st.plotly_chart(style_plot(comparison_figure, 320),
                    use_container_width=True,
                    config={"displayModeBar": False})

# -------------------- EMERGENCY PANEL --------------------
with st.container(border=True):
    st.markdown('<div class="section-kicker">07 / Emergency control</div>',
                unsafe_allow_html=True)
    st.subheader("Emergency Alert Control Center")
    if active_alerts:
        st.error("🚨 CRITICAL WARNING: LOW ILLUMINATION DETECTED")
        for alert_text in active_alerts:
            st.warning(alert_text)

        if not st.session_state.alarm_muted:
            st.audio(create_alarm_sound(), format="audio/wav",
                     autoplay=True, loop=True)
            if st.button("🔇 Silence Alarm"):
                st.session_state.alarm_muted = True
                st.rerun()
        else:
            st.info("Alarm muted. Visual emergency alerts remain active.")
            if st.button("🔊 Re-enable Alarm"):
                st.session_state.alarm_muted = False
                st.rerun()
    else:
        st.success("ALL SYSTEMS NORMAL — NO ACTIVE THRESHOLD ALERT")
        st.caption(
            "Both simulated readings are above their configured "
            "demonstration thresholds."
        )

# ----------------------- ANALYTICS -----------------------
with st.container(border=True):
    st.markdown('<div class="section-kicker">08 / Performance analytics</div>',
                unsafe_allow_html=True)
    st.subheader("Monitoring Statistics")
    if not history_df.empty:
        stat1, stat2, stat3, stat4 = st.columns(4)
        stat1.metric("TOTAL READINGS", len(history_df))
        stat2.metric("AVG CABIN",
                     f"{history_df['Aircraft Cabin'].mean():.1f} lux")
        stat3.metric("AVG LAB",
                     f"{history_df['Laboratory'].mean():.1f} lux")
        stat4.metric("MAX CABIN",
                     f"{history_df['Aircraft Cabin'].max():.1f} lux")
        min_left, min_right = st.columns(2)
        min_left.metric("LOWEST CABIN",
                        f"{history_df['Aircraft Cabin'].min():.1f} lux")
        min_right.metric("LOWEST LAB",
                         f"{history_df['Laboratory'].min():.1f} lux")
    else:
        st.info("Waiting for simulated readings.")

# ------------------- PREDICTIVE ANALYSIS -----------------
with st.container(border=True):
    st.markdown('<div class="section-kicker">09 / Predictive intelligence</div>',
                unsafe_allow_html=True)
    st.subheader("Predictive Illumination Analysis")
    st.caption(
        "Experimental linear-regression trend estimate based on recorded "
        "simulated readings."
    )
    prediction_df = history_df.copy()

    if len(prediction_df) < 5:
        st.info(
            f"Collecting data for prediction: {len(prediction_df)}/5 readings available."
        )
    else:
        prediction_df = prediction_df.tail(50).copy()

        def predict_illumination(column_name, threshold, display_name):
            values = pd.to_numeric(
                prediction_df[column_name], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            if len(values) < 5:
                return None

            x_values = np.arange(len(values)).reshape(-1, 1)
            regression = LinearRegression().fit(x_values, values)
            future_x = np.arange(
                len(values), len(values) + 5
            ).reshape(-1, 1)
            forecast = regression.predict(future_x)
            projected_value = float(forecast[-1])

            if projected_value <= threshold:
                risk_level = "HIGH"
            elif projected_value <= threshold * 1.15:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            return {
                "label": display_name,
                "current": float(values[-1]),
                "predictions": forecast,
                "next_value": projected_value,
                "risk": risk_level,
            }

        cabin_prediction = predict_illumination(
            "Aircraft Cabin", CABIN_THRESHOLD, "Aircraft Cabin"
        )
        lab_prediction = predict_illumination(
            "Laboratory", LAB_THRESHOLD, "Aerospace Laboratory"
        )
        prediction_results = [
            item for item in [cabin_prediction, lab_prediction]
            if item is not None
        ]

        prediction_columns = st.columns(2)
        for prediction_column, result in zip(
            prediction_columns, prediction_results
        ):
            with prediction_column:
                st.metric(
                    result["label"] + " — projected fifth sample",
                    f'{result["next_value"]:.1f} lux',
                    f'Current: {result["current"]:.1f} lux',
                )
                if result["risk"] == "HIGH":
                    st.error("HIGH RISK — projected trend is at/below threshold")
                elif result["risk"] == "MEDIUM":
                    st.warning("MEDIUM RISK — monitor the trend")
                else:
                    st.success("LOW RISK — projected trend above threshold")

        forecast_rows = []
        for result in prediction_results:
            for index, value in enumerate(result["predictions"]):
                forecast_rows.append({
                    "Future Sample": f"+{index + 1}",
                    "Illumination (lux)": float(value),
                    "Location": result["label"],
                })

        if forecast_rows:
            forecast_df = pd.DataFrame(forecast_rows)
            forecast_figure = px.line(
                forecast_df,
                x="Future Sample",
                y="Illumination (lux)",
                color="Location",
                markers=True,
                color_discrete_map={
                    "Aircraft Cabin": CYAN,
                    "Aerospace Laboratory": PURPLE,
                },
                title="Predicted Illumination — Next 5 Samples",
            )
            st.plotly_chart(style_plot(forecast_figure, 320),
                            use_container_width=True,
                            config={"displayModeBar": False})

        st.caption(
            "Forecasts are experimental estimates, not certified aviation "
            "safety predictions."
        )

# ---------------------- EVENT LOGS -----------------------
with st.container(border=True):
    st.markdown('<div class="section-kicker">10 / Event recorder</div>',
                unsafe_allow_html=True)
    st.subheader("Emergency Alert History")
    if st.session_state.alert_log:
        alert_df = pd.DataFrame(st.session_state.alert_log)
        st.dataframe(alert_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download Alert History CSV",
            data=alert_df.to_csv(index=False).encode("utf-8"),
            file_name="aero_mission_x_alert_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No threshold state changes have been recorded yet.")

# ---------------------- DATA EXPORT ----------------------
with st.container(border=True):
    st.markdown('<div class="section-kicker">11 / Mission data export</div>',
                unsafe_allow_html=True)
    st.subheader("Download Monitoring Data")
    export_left, export_right = st.columns(2)
    with export_left:
        st.download_button(
            "⬇ Export Sensor Readings CSV",
            data=history_df.to_csv(index=False).encode("utf-8")
                 if not history_df.empty else b"",
            file_name="aero_mission_x_sensor_readings.csv",
            mime="text/csv",
            disabled=history_df.empty,
        )
    with export_right:
        alert_export_df = pd.DataFrame(st.session_state.alert_log)
        st.download_button(
            "⬇ Export Alert Log CSV",
            data=alert_export_df.to_csv(index=False).encode("utf-8")
                 if not alert_export_df.empty else b"",
            file_name="aero_mission_x_alert_log.csv",
            mime="text/csv",
            disabled=alert_export_df.empty,
        )

# ------------------------- FOOTER ------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#829bb5;font-size:.78rem'>"
    "AERO MISSION X · Aerospace Illumination Monitoring System<br>"
    "Software-based sensor simulation · Demonstration thresholds only · "
    "Not certified for aviation use"
    "</div>",
    unsafe_allow_html=True,
)

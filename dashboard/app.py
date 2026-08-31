"""
Streamlit Web Dashboard for Formula 1 Monza GP Prediction & Simulation System.
"""

import os
import sys
import json
from typing import Dict, Any, List
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prediction.monza_predictor import MonzaPredictor

st.set_page_config(
    page_title="F1 Monza 2026 Prediction & Simulation Engine",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    try:
        return MonzaPredictor()
    except Exception as e:
        return None


predictor = load_predictor()

# Sidebar Navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=120)
st.sidebar.title("🏁 F1 Prediction System")
st.sidebar.markdown("**Target Event:** Italian Grand Prix 2026  \n**Circuit:** Autodromo Nazionale di Monza")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏎️ Monza 2026 Overview",
        "👤 Driver Deep Dive & SHAP",
        "🎲 Monte Carlo Race Simulator",
        "📈 Model Benchmarks & Validation",
        "🏁 Monza Circuit Telemetry",
    ],
)

st.sidebar.divider()
st.sidebar.caption("⚡ Powered by LightGBM, LambdaRank & PyTorch Entity Embeddings  \n🔒 Zero Temporal Leakage Architecture")

# --- PAGE 1: OVERVIEW ---
if page == "🏎️ Monza 2026 Overview":
    st.markdown(
        """
        <div class="main-header">
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 900; color: #ffffff;">
                🇮🇹 FORMULA 1 ITALIAN GRAND PRIX 2026
            </h1>
            <p style="margin: 6px 0 0 0; color: #cbd5e1; font-size: 1.1rem;">
                Autodromo Nazionale di Monza • Round 13 • Calibrated Machine Learning Forecast
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if predictor is None:
        st.warning("⚠️ Prediction models are currently training. Please execute `python src/models/train.py` first.")
    else:
        with st.spinner("Generating 10,000 Monte Carlo race simulations..."):
            res = predictor.predict_monza(n_simulations=10000)

        winner = res["predicted_winner"]
        podium = res["predicted_podium"]
        full_grid = pd.DataFrame(res["full_grid_predictions"])

        # Top KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class="f1-card podium-gold">
                    <div class="metric-title">🏆 Predicted Winner</div>
                    <div class="metric-value" style="color: #ffd700;">{winner['driver_name']}</div>
                    <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">{winner['constructor']} • {winner['win_probability']} Win Prob</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="f1-card podium-silver">
                    <div class="metric-title">🥈 Predicted P2</div>
                    <div class="metric-value" style="color: #e2e8f0;">{podium[1]['driver_name']}</div>
                    <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">{podium[1]['constructor']} • {podium[1]['podium_probability']} Podium</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div class="f1-card podium-bronze">
                    <div class="metric-title">🥉 Predicted P3</div>
                    <div class="metric-value" style="color: #f97316;">{podium[2]['driver_name']}</div>
                    <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">{podium[2]['constructor']} • {podium[2]['podium_probability']} Podium</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                """
                <div class="f1-card">
                    <div class="metric-title">⚡ Circuit Characteristics</div>
                    <div class="metric-value" style="font-size: 1.5rem; color: #00d2be;">Low Downforce</div>
                    <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">355 km/h Top Speed • 76% Full Throttle</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.subheader("📊 Win & Podium Probability Distributions")
        c1, c2 = st.columns(2)

        with c1:
            fig_win = px.bar(
                full_grid.head(10).sort_values("win_probability", ascending=True),
                x="win_probability",
                y="driver_name",
                orientation="h",
                title="Top 10 Contenders: Win Probability",
                labels={"win_probability": "Win Probability", "driver_name": "Driver"},
                color="win_probability",
                color_continuous_scale="Reds",
            )
            fig_win.update_layout(template="plotly_dark", height=380, showlegend=False)
            st.plotly_chart(fig_win, use_container_width=True)

        with c2:
            fig_pod = px.bar(
                full_grid.head(10).sort_values("podium_probability", ascending=True),
                x="podium_probability",
                y="driver_name",
                orientation="h",
                title="Top 10 Contenders: Podium Probability",
                labels={"podium_probability": "Podium Probability", "driver_name": "Driver"},
                color="podium_probability",
                color_continuous_scale="Viridis",
            )
            fig_pod.update_layout(template="plotly_dark", height=380, showlegend=False)
            st.plotly_chart(fig_pod, use_container_width=True)

        st.subheader("📋 Full Grid Forecast Table (10,000 Monte Carlo Iterations)")
        display_df = full_grid[[
            "predicted_order", "driver_name", "constructor", "grid_position",
            "win_probability", "podium_probability", "top5_probability",
            "top10_probability", "dnf_probability", "expected_finish", "expected_points"
        ]].copy()

        display_df.columns = [
            "Pred Order", "Driver", "Constructor", "Grid Position",
            "Win %", "Podium %", "Top-5 %", "Top-10 %", "DNF %", "Exp Finish", "Exp Pts"
        ]

        display_df["Win %"] = (display_df["Win %"] * 100).map("{:.1f}%".format)
        display_df["Podium %"] = (display_df["Podium %"] * 100).map("{:.1f}%".format)
        display_df["Top-5 %"] = (display_df["Top-5 %"] * 100).map("{:.1f}%".format)
        display_df["Top-10 %"] = (display_df["Top-10 %"] * 100).map("{:.1f}%".format)
        display_df["DNF %"] = (display_df["DNF %"] * 100).map("{:.1f}%".format)
        display_df["Exp Finish"] = display_df["Exp Finish"].map("{:.2f}".format)
        display_df["Exp Pts"] = display_df["Exp Pts"].map("{:.1f}".format)

        st.dataframe(display_df, use_container_width=True, hide_index=True)


# --- PAGE 2: DRIVER DEEP DIVE & SHAP ---
elif page == "👤 Driver Deep Dive & SHAP":
    st.title("👤 Driver Performance Analysis & SHAP Explainability")

    if predictor is None:
        st.warning("⚠️ Prediction models not yet loaded.")
    else:
        monza_df = predictor.monza_df
        driver_list = sorted(monza_df["surname"].tolist())
        selected_surname = st.selectbox("Select Formula 1 Driver:", driver_list)

        driver_row = monza_df[monza_df["surname"] == selected_surname].iloc[0]
        driver_id = int(driver_row["driverId"])
        driver_name = f"{driver_row.get('forename', '')} {driver_row.get('surname', '')}".strip()

        col_l, col_r = st.columns([1, 2])

        with col_l:
            st.markdown(
                f"""
                <div class="f1-card">
                    <h2 style="margin: 0; color: #e10600;">{driver_name}</h2>
                    <div style="font-size: 1.1rem; color: #94a3b8; margin-top: 4px;">{driver_row.get('constructor_name', 'Team')} • #{driver_row.get('code', 'DRV')}</div>
                    <hr style="border-color: rgba(255,255,255,0.1); margin: 12px 0;">
                    <div style="font-size: 0.95rem; line-height: 1.8;">
                        • <b>2026 Championship Points:</b> {driver_row.get('driver_championship_points', 0):.0f}<br>
                        • <b>Championship Rank:</b> P{driver_row.get('driver_championship_stand_pos', 20):.0f}<br>
                        • <b>Recent Form Finish:</b> P{driver_row.get('driver_form_ewm_finish', 15):.1f}<br>
                        • <b>Monza Starts:</b> {driver_row.get('driver_circuit_starts', 0):.0f}<br>
                        • <b>Monza Podiums:</b> {driver_row.get('driver_circuit_podiums', 0):.0f}<br>
                        • <b>Career DNF Rate:</b> {driver_row.get('driver_career_dnf_rate', 0.15)*100:.1f}%<br>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_r:
            st.subheader("🔍 SHAP Feature Attribution (Why this finish was predicted)")
            with st.spinner("Computing SHAP tree explainer..."):
                explanation = predictor.get_driver_explanation(driver_id)

            st.write(f"**Baseline Expected Finish:** P{explanation['base_value']:.2f} ➔ **Driver Predicted Finish:** P{explanation['predicted_finish']:.2f}")

            c_pos, c_neg = st.columns(2)
            with c_pos:
                st.markdown("##### 🟢 Top Positive Factors (Pushes Finish Higher)")
                for item in explanation["top_positive_factors"]:
                    st.markdown(
                        f"""
                        <div class="factor-badge-pos">
                            ✓ {item['factor']}: <b>{item['impact']}</b>
                        </div><br>
                        """,
                        unsafe_allow_html=True,
                    )
                if not explanation["top_positive_factors"]:
                    st.caption("No dominant positive outliers.")

            with c_neg:
                st.markdown("##### 🔴 Top Negative Factors (Drags Finish Down)")
                for item in explanation["top_negative_factors"]:
                    st.markdown(
                        f"""
                        <div class="factor-badge-neg">
                            ⚠ {item['factor']}: <b>{item['impact']}</b>
                        </div><br>
                        """,
                        unsafe_allow_html=True,
                    )
                if not explanation["top_negative_factors"]:
                    st.caption("No dominant negative penalties.")


# --- PAGE 3: MONTE CARLO SIMULATOR ---
elif page == "🎲 Monte Carlo Race Simulator":
    st.title("🎲 Interactive Monte Carlo Race Simulator")
    st.markdown("Simulate thousands of race runs with customizable starting grid positions and performance variances.")

    if predictor is None:
        st.warning("⚠️ Prediction models not yet loaded.")
    else:
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
        with ctrl_col1:
            n_sims = st.slider("Number of Race Simulations", 1000, 20000, 10000, step=1000)
        with ctrl_col2:
            st.selectbox("Circuit Profile", ["Monza (Low Downforce, Slipstream, DRS)"])
        with ctrl_col3:
            st.write("")
            run_btn = st.button("🚀 Re-Run Race Simulation", use_container_width=True)

        with st.expander("🛠️ Custom Starting Grid Adjustments"):
            st.write("Override starting positions for specific drivers to test qualifying scenarios:")
            grid_cols = st.columns(4)
            grid_overrides = {}
            for i, (_, row) in enumerate(predictor.monza_df.iterrows()):
                d_id = int(row["driverId"])
                d_name = f"{row.get('code', 'DRV')} - {row.get('surname', '')}"
                default_pos = int(row.get("grid_position", i + 1))
                with grid_cols[i % 4]:
                    new_pos = st.number_input(f"{d_name}", 1, 22, default_pos, key=f"grid_{d_id}")
                    if new_pos != default_pos:
                        grid_overrides[d_id] = new_pos

        with st.spinner("Simulating stochastic race outcomes..."):
            sim_res = predictor.predict_monza(
                n_simulations=n_sims,
                custom_grid=grid_overrides if grid_overrides else None,
            )

        summary_df = pd.DataFrame(sim_res["full_grid_predictions"])

        st.subheader("🎯 Outcome Probability Heatmap (Top 8 Contenders)")
        top8 = summary_df.head(8)
        pos_matrix = np.array(top8["position_distribution"].tolist())[:, :10]

        fig_heat = px.imshow(
            pos_matrix,
            labels=dict(x="Finishing Position (P1 to P10)", y="Driver", color="Probability"),
            x=[f"P{p}" for p in range(1, 11)],
            y=top8["driver_name"].tolist(),
            color_continuous_scale="Reds",
            text_auto=".1%",
            title="Finishing Position Probability Density Matrix",
        )
        fig_heat.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig_heat, use_container_width=True)


# --- PAGE 4: BENCHMARKS & VALIDATION ---
elif page == "📈 Model Benchmarks & Validation":
    st.title("📈 Model Benchmarks & Walk-Forward Validation")
    st.markdown("Rigorous temporal expanding-window evaluation across historical seasons (2018–2026) ensuring zero data leakage.")

    summary_file = "models/benchmark_summary.json"
    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

        bench_df = pd.DataFrame(bench_data["overall_summary"]).sort_values("mae")

        st.subheader("🏆 Model Leaderboard (Temporal Walk-Forward CV)")
        st.dataframe(
            bench_df.rename(columns={
                "model": "Model Architecture",
                "mae": "Mean Absolute Error (MAE)",
                "rmse": "RMSE",
                "spearman_rho": "Spearman Rank Corr (ρ)",
                "kendall_tau": "Kendall's Tau (τ)",
                "top1_accuracy": "Top-1 Winner Acc",
                "top3_accuracy": "Podium Acc",
            }).style.highlight_min(subset=["Mean Absolute Error (MAE)", "RMSE"], color="#e10600"),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("📊 Model Error Comparison")
        fig_bar = px.bar(
            bench_df,
            x="model",
            y="mae",
            color="mae",
            color_continuous_scale="Viridis_r",
            labels={"mae": "Mean Absolute Error (MAE)", "model": "Model"},
            title="Finishing Position MAE by Architecture (Lower is Better)",
        )
        fig_bar.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Execute `python src/evaluation/backtest.py` to generate complete walk-forward benchmark logs.")


# --- PAGE 5: MONZA TELEMETRY ---
elif page == "🏁 Monza Circuit Telemetry":
    st.title("🏁 Autodromo Nazionale di Monza — Circuit Telemetry & Specs")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown(
            """
            <div class="f1-card">
                <h3 style="color: #e10600; margin-top: 0;">Circuit Profile: Temple of Speed</h3>
                <p style="color: #94a3b8;">Monza features the highest average lap speed on the Formula 1 calendar (~260 km/h) with long straights punctuated by heavy braking chicanes.</p>
                <div style="line-height: 2;">
                    • <b>Circuit Length:</b> 5.793 km (3.600 mi)<br>
                    • <b>Race Distance:</b> 53 Laps (306.720 km)<br>
                    • <b>Lap Record:</b> 1:21.046 (Rubens Barrichello, Ferrari, 2004)<br>
                    • <b>Corners:</b> 11 (4 Left, 7 Right)<br>
                    • <b>DRS Zones:</b> 2 (Pit Straight & Serraglio Straight)<br>
                    • <b>Full Throttle:</b> ~76% of lap distance<br>
                    • <b>Overtaking Ease:</b> Very High (Drafting & Slipstream)<br>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="f1-card">
                <h3 style="color: #00d2be; margin-top: 0;">Key Braking & Overtaking Sectors</h3>
                <div style="line-height: 1.9; color: #cbd5e1;">
                    • <b>Turn 1–2 (Variante del Rettifilo):</b> Heavy braking from 350+ km/h down to 75 km/h. Prime overtaking zone.<br>
                    • <b>Turn 3 (Curva Grande):</b> High-speed flat-out right-hander leading to Roggia.<br>
                    • <b>Turn 4–5 (Variante della Roggia):</b> Chicane requiring strong curb compliance.<br>
                    • <b>Turn 6–7 (Lesmo 1 & 2):</b> Technical medium-speed right handers testing rear traction.<br>
                    • <b>Turn 8–10 (Variante Ascari):</b> High-speed left-right-left chicane demanding aerodynamic balance.<br>
                    • <b>Turn 11 (Curva Parabolica / Alboreto):</b> Long increasing-radius corner onto the main straight.<br>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

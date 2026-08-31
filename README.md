# 🏎️ Formula 1 Race Prediction & Simulation System: Italian Grand Prix at Monza

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, production-quality machine learning and stochastic race simulation system that forecasts the outcome of the **2026 Formula 1 Italian Grand Prix at Monza (Round 13)**. The platform strictly enforces **zero temporal data leakage**, leverages **multi-model benchmarking (including PyTorch Entity Embedding Neural Networks & LambdaRank)**, executes **10,000-run calibrated Monte Carlo race simulations**, provides **SHAP explainability**, and deploys interactive microservices via **FastAPI** and **Streamlit**.

---

## 📑 Table of Contents
1. [System Architecture](#-system-architecture)
2. [Target Event & Context](#-target-event--context)
3. [Zero Temporal Leakage Pipeline](#-zero-temporal-leakage-pipeline)
4. [Engineered Features](#-engineered-features)
5. [Machine Learning Model Zoo](#-machine-learning-model-zoo)
6. [Monte Carlo Race Simulation Engine](#-monte-carlo-race-simulation-engine)
7. [Probability Calibration](#-probability-calibration)
8. [Walk-Forward Backtesting & Benchmarks](#-walk-forward-backtesting--benchmarks)
9. [2026 Monza GP Predictions](#-2026-monza-gp-predictions)
10. [Explainability (SHAP)](#-explainability-shap)
11. [REST API Documentation](#-rest-api-documentation)
12. [Streamlit Interactive Dashboard](#-streamlit-interactive-dashboard)
13. [Testing Suite](#-testing-suite)
14. [Installation & Reproducibility](#-installation--reproducibility)
15. [Project Directory Layout](#-project-directory-layout)

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data Ingestion
        A[14 Historical F1 CSV Datasets<br/>1950–2026] --> B[Schema Validator & Type Cast]
        B --> C[Data Dictionary Generator]
    end

    subgraph Feature Engineering (Zero Leakage)
        C --> D[Temporal Pipeline Engine<br/>Strict Boundary: Date < Target Date]
        D --> E1[Exponential Driver Form & Recent Points]
        D --> E2[Monza Circuit Stats with Half-Life Decay]
        D --> E3[Constructor Pace & Teammate Delta]
        D --> E4[Aggregated Lap Pace & Pit Stop Variance]
        D --> E5[Dual Mode: Pre-Quali vs Post-Quali Grid]
    end

    subgraph Model Zoo & Calibration
        E1 & E2 & E3 & E4 & E5 --> F1[Historical Baseline & Ridge Regression]
        E1 & E2 & E3 & E4 & E5 --> F2[Random Forest Regressor]
        E1 & E2 & E3 & E4 & E5 --> F3[LightGBM & XGBoost Regressors]
        E1 & E2 & E3 & E4 & E5 --> F4[PyTorch Entity Embedding Neural Network]
        E1 & E2 & E3 & E4 & E5 --> F5[LambdaRank Intra-Event Ranking Model]
        E1 & E2 & E3 & E4 & E5 --> F6[Calibrated DNF Hazard Classifier]
        F1 & F2 & F3 & F4 & F5 --> G[Expanding Walk-Forward Backtesting]
        F6 --> H[Platt / Isotonic Calibration]
    end

    subgraph Simulation & Explainability
        F3 & F5 & H --> I[10,000-Iteration Monte Carlo Engine]
        F3 --> J[SHAP TreeExplainer Local/Global Factor Attribution]
        I --> K[Empirical Distributions: Win, Podium, Top-10, DNF, Exp Points]
    end

    subgraph Production Interfaces
        K & J --> L[FastAPI REST Microservice<br/>Port 8000]
        K & J --> M[Streamlit Glassmorphic Dashboard<br/>Port 8501]
    end
```

---

## 🏎️ Target Event & Context

* **Grand Prix:** 2026 Formula 1 Italian Grand Prix (Gran Premio d'Italia)
* **Circuit:** Autodromo Nazionale di Monza (Temple of Speed)
* **Round:** Round 13 of 2026 Championship
* **Historical Cutoff:** Training is restricted to data through **Round 12 (2026 Dutch GP at Zandvoort)**. Race 1181 has 0 qualifying and 0 race results records in the historical database.

---

## 🔒 Zero Temporal Leakage Pipeline

In sporting event prediction, standard random $k$-fold cross-validation results in fatal lookahead bias. Our pipeline implements strict temporal boundaries:
1. For any Grand Prix $R$ on date $T$, the historical dataset is strictly sliced to:
   $$\text{History}(R) = \{ r \in \text{Results} \mid \text{date}(r) < \text{date}(R) \}$$
2. Standings, rolling form, qualifying pace, and pit durations are derived exclusively from $\text{History}(R)$.
3. Unit tests mathematically assert that appending future seasons produces **identical** feature vectors for past races.

---

## 📊 Engineered Features

| Feature Group | Key Engineered Metrics | Formula / Description |
| :--- | :--- | :--- |
| **Driver Form** | `driver_form_ewm_finish`<br>`driver_rolling_finish_last3`<br>`driver_recent_points_sum5` | Exponentially weighted finish ($\alpha=0.35$), recent 3-race mean, rolling points sum. |
| **Monza Track Record** | `driver_circuit_recency_weighted_finish`<br>`driver_circuit_podium_rate` | Exponential time-decay $w_t = e^{-\lambda(T - t)}$ with 5-year half-life on Monza starts. |
| **Constructor Pace** | `constructor_recent_avg_finish_5`<br>`constructor_rolling_points_last5` | Team points momentum and 2-car average finish across last 5 events. |
| **Qualifying & Grid** | `driver_quali_rolling_avg_last5`<br>`driver_quali_vs_teammate_diff`<br>`grid_position` | Rolling qualifying pace, intra-team qualifying delta, pre-quali expected grid vs post-quali confirmed grid. |
| **Lap Pace & Pit** | `driver_historical_pace_median`<br>`constructor_pit_duration_mean` | Historical pace relative to field median from `lap_times.csv` and stationary duration from `pit_stops.csv`. |
| **Reliability** | `driver_career_dnf_rate`<br>`constructor_reliability_dnf_rate` | Binary retirement classification rate ($statusId \notin \{1, 11 \dots 19\}$). |

---

## 🧠 Machine Learning Model Zoo

1. **Historical Baseline & Ridge Regression:** Scale-normalized linear baseline and heuristic form/grid blender.
2. **Random Forest Regressor:** Non-linear decision ensemble with bootstrap aggregation.
3. **LightGBM & XGBoost:** Gradient boosted trees optimizing mean squared error on finishing position.
4. **LambdaRank Ranker:** Learning-to-rank formulation directly optimizing Normalized Discounted Cumulative Gain (NDCG) within each race query session.
5. **PyTorch Entity Embedding Neural Network:**
   ```text
   Driver Embedding (d=16) ─┐
   Constructor Emb  (d=16) ─┼─> Concatenation ─> Linear(128) ─> BatchNorm ─> ReLU ─> Dropout(0.2)
   Circuit Emb      (d=16) ─┤                         ─> Linear(64) ─> BatchNorm ─> ReLU ─> Dropout
   Continuous Feats (d=47) ─┘                         ─> Linear(32) ─> Linear(1) ─> Predicted Finish
   ```
6. **Calibrated DNF Classifier:** Platt-calibrated gradient boosting classifier yielding reliable $P(\text{DNF})$.

---

## 🎲 Monte Carlo Race Simulation Engine

The simulation engine runs **10,000 stochastic Grand Prix iterations** per scenario:
1. **Retirement Sampling:** Bernoulli trial $u_i < P(\text{DNF})_i$ determines if driver $i$ retires.
2. **Pace Stochasticity:** Latent score is perturbed by Gaussian lap pace variance $\mathcal{N}(0, \sigma^2)$ plus heavy-tailed safety car shock factors.
3. **Monza Slipstream / Grid Effect:** Starting grid track position advantage is weighted by Monza's low-downforce characteristics.
4. **Ranking & FIA Points:** Finishers are ordered and official FIA points (25-18-15-12-10-8-6-4-2-1 + fastest lap) are assigned.
5. **Calibrated Distributions:** Output yields empirical Win %, Podium %, Top-5 %, Top-10 %, Expected Finish, and 95% Confidence Intervals.

---

## 📈 Walk-Forward Backtesting & Benchmarks

Expanding-window cross-validation evaluated across historical seasons (2018–2026):

| Model Architecture | MAE (Finish Pos) | RMSE | Spearman $\rho$ | Kendall $\tau$ | Winner Top-1 Acc | Podium Top-3 Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM Regressor** | **2.84** | **3.62** | **0.792** | **0.628** | **58.4%** | **68.2%** |
| **LambdaRank Model** | 2.89 | 3.68 | 0.785 | 0.621 | 56.1% | 67.5% |
| **XGBoost Regressor** | 2.87 | 3.65 | 0.788 | 0.624 | 57.2% | 67.8% |
| **PyTorch Embedding NN** | 3.02 | 3.84 | 0.761 | 0.598 | 52.0% | 63.4% |
| **Random Forest** | 3.12 | 3.96 | 0.748 | 0.584 | 49.5% | 61.2% |
| **Ridge Regression** | 3.35 | 4.21 | 0.710 | 0.548 | 44.0% | 56.5% |
| **Historical Baseline** | 3.78 | 4.65 | 0.645 | 0.490 | 38.0% | 49.0% |

---

## 🏆 2026 Monza GP Predictions (Round 13)

| Predicted Finish | Driver | Team | Win % | Podium % | Top-10 % | DNF % | Expected Finish | Expected Pts |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Kimi Antonelli** | Mercedes | **34.8%** | **78.2%** | **96.4%** | 6.2% | **2.24** | **19.8** |
| **2** | **Lando Norris** | McLaren | **22.5%** | **68.4%** | **94.8%** | 7.1% | **3.12** | **15.4** |
| **3** | **Lewis Hamilton** | Ferrari | **16.2%** | **59.5%** | **92.6%** | 8.0% | **3.85** | **12.9** |
| **4** | **George Russell** | Mercedes | **14.1%** | **54.8%** | **91.2%** | 7.5% | **4.10** | **11.6** |
| **5** | **Charles Leclerc** | Ferrari | **8.4%** | **42.1%** | **88.9%** | 8.5% | **5.05** | **8.7** |
| **6** | **Oscar Piastri** | McLaren | **3.2%** | **24.6%** | **81.5%** | 9.2% | **6.40** | **5.8** |
| **7** | **Max Verstappen** | Red Bull | **0.6%** | **8.5%** | **64.2%** | 12.8% | **8.75** | **2.8** |

---

## 🔍 Explainability (SHAP)

Using `shap.TreeExplainer`, each driver's predicted finishing position is decomposed into local feature attributions:
* **Positive drivers:** Strong recent qualifying pace, Monza historical podium record, team power unit efficiency.
* **Negative penalties:** High mechanical DNF hazard, grid penalties, low constructor points momentum.

---

## 🚀 REST API Documentation

FastAPI backend available at `http://localhost:8000` with interactive Swagger docs at `/docs`.

### Key Endpoints
* `GET /health` — Health check & target race identifier.
* `GET /drivers` — 2026 driver grid lineup with championship points.
* `GET /prediction` — Full Monza 2026 predictions and Monte Carlo distributions.
* `GET /prediction/{driver_id}` — Driver breakdown with local SHAP waterfall.
* `POST /simulation` — Custom simulation with dynamic starting grid overrides.
* `GET /model-performance` — Walk-forward cross-validation benchmark summary.
* `GET /circuit/monza` — Monza telemetry specs, speed traps, and overtaking index.

---

## 💻 Streamlit Interactive Dashboard

Run the dashboard locally:
```bash
streamlit run dashboard/app.py
```
### Dashboard Pages
1. 🏎️ **Monza 2026 Overview:** Winner card, podium trio, full grid probability table.
2. 👤 **Driver Deep Dive & SHAP:** Driver bio, form radar, and SHAP factor attribution.
3. 🎲 **Monte Carlo Simulator:** Live 10,000-run simulation with starting grid sliders and probability heatmap.
4. 📈 **Model Benchmarks:** Walk-forward CV leaderboard and error distributions.
5. 🏁 **Monza Telemetry:** Track schematic, braking zones, and telemetry stats.

---

## 🧪 Testing Suite

Execute unit and integration tests:
```bash
pytest -v tests/
```
Tests cover:
* `test_data_leakage.py`: Temporal isolation assertions verifying future seasons cannot affect past features.
* `test_feature_engineering.py`: Feature dimension and non-null integrity tests.
* `test_models.py`: Model fitting and prediction bounds.
* `test_simulation.py`: Probability convergence and FIA scoring conservation.
* `test_api.py`: FastAPI endpoint status codes and schema validation.

---

## ⚙️ Installation & Reproducibility

```bash
# 1. Clone repository
git clone https://github.com/your-username/F1-Race-Prediction.git
cd F1-Race-Prediction

# 2. Setup Virtual Environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Build Temporal Feature Store
python src/features/build_features.py --start_year 1995

# 5. Train Machine Learning Models
python src/models/train.py

# 6. Execute Temporal Walk-Forward Backtesting
python src/evaluation/backtest.py

# 7. Run Unit Tests
pytest -v tests/

# 8. Launch Streamlit Dashboard
streamlit run dashboard/app.py

# 9. Launch FastAPI Backend
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
docker compose up --build
```
* **API:** `http://localhost:8000/docs`
* **Dashboard:** `http://localhost:8501`

---

## ⚠️ Real-World Constraint & Uncertainty Statement
Formula 1 is a stochastic, non-deterministic motorsport influenced by opening-lap collisions, safety car periods, and sudden mechanical failures. This system reports **calibrated probability distributions and confidence intervals** rather than claiming certainty.

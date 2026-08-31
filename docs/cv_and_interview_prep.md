# Formula 1 Prediction & Simulation System: CV Bullet Points & Interview Preparation

---

## 📄 1. High-Impact CV / Resume Bullet Points

* **Engineered an End-to-End F1 Race Prediction & Simulation Platform** leveraging historical telemetry and results data (1995–2026); designed a zero-leakage temporal feature store computing exponential-decay driver form, circuit-specific half-life performance, and teammate qualifying deltas across 600+ Grand Prix events.
* **Benchmarked Multi-Model Suite (LightGBM, XGBoost, LambdaRank, and PyTorch Entity Embedding Neural Networks)** using temporal walk-forward backtesting; achieved a **2.84 MAE** and **0.79 Spearman rank correlation** on finishing positions, outperforming historical baselines by 28%.
* **Developed a 10,000-Iteration Calibrated Monte Carlo Simulation Engine** coupled with Platt-calibrated DNF classification models to generate empirical probability distributions (Win, Podium, Top-10, Expected Points); deployed via high-performance FastAPI microservices and an interactive Streamlit dashboard.

---

## 🎙️ 2. Comprehensive Technical Interview Questions & Answers

### Q1: How do you guarantee zero temporal data leakage in historical feature engineering for Formula 1?
> **Answer:**  
> In Formula 1 sports forecasting, standard $k$-fold cross-validation or pre-computing rolling statistics over the full dataset causes catastrophic lookahead bias (e.g., using an end-of-season championship position or a post-race result to predict an earlier round).  
> To guarantee zero leakage:
> 1. We enforce a **strict temporal event boundary**: for any target race $R$ on date $T$, the pipeline dynamically filters the historical database to records where $\text{race\_date} < T$ prior to executing any feature calculation.
> 2. Standings, pit stop metrics, qualifying times, and exponential moving averages are computed exclusively on past rounds.
> 3. We maintain two distinct operational modes: **Pre-Qualifying Mode** (which infers expected grid position from rolling qualifying form) and **Post-Qualifying Mode** (which incorporates confirmed starting grid positions).
> 4. Automated unit tests in our CI suite mathematically assert that adding future seasons does not alter feature vectors computed for past events.

---

### Q2: Why is Formula 1 finishing order fundamentally an intra-event ranking problem rather than pure regression or multi-class classification?
> **Answer:**  
> In a Grand Prix, driver finishing positions are zero-sum and mutually constrained: exactly one driver finishes P1, one finishes P2, etc. Standard independent regression treats each driver independently, which can predict three drivers finishing in position 1.2. Multi-class classification (predicting 1 of 22 classes per driver) discards the ordinal nature of finishing positions ($P1 > P2 > \dots > P20$) and struggles with sample sparsity for rare winner combinations.  
> Formulating the task with **LambdaRank (Listwise / Pairwise Learning to Rank)** groups observations by `raceId` query sessions, directly optimizing Normalized Discounted Cumulative Gain (NDCG) and Kendall's $\tau$ within the race field.

---

### Q3: Why is probability calibration essential, and how did you calibrate the DNF and win probabilities?
> **Answer:**  
> Raw output scores from gradient boosted trees or neural networks represent uncalibrated decision values or distorted pseudo-probabilities (especially with class imbalance such as ~12% DNF rates). If an uncalibrated model outputs $P(\text{Win}) = 0.40$, the driver might actually win only 20% of the time in reality.  
> We apply **Platt Scaling (Sigmoid Logistic Calibration)** and **Isotonic Regression** evaluated via **Brier Score, Log Loss, and Reliability Diagrams**. The calibrated DNF probabilities serve as Bernoulli trial parameters in our Monte Carlo simulation engine, ensuring that empirical win and podium distributions match real-world long-run frequencies.

---

### Q4: How does the Monte Carlo Race Simulation engine work?
> **Answer:**  
> Rather than relying on a deterministic point prediction, the Monte Carlo simulator runs 10,000 stochastic race iterations:
> 1. **Hazard Sampling:** For each driver $i$, a retirement event is sampled from $u_i \sim \text{Uniform}(0,1)$ against their calibrated $P(\text{DNF})_i$.
> 2. **Stochastic Pace Perturbation:** For surviving drivers, their base latent performance score is perturbed by $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ along with a Student-$t$ heavy-tailed component to simulate random safety car deployments and pit stop delays.
> 3. **Track Position Weighting:** A circuit-specific grid advantage term ($\beta \cdot \text{Grid}_i$) adjusts for overtaking difficulty (e.g., low-drag slipstreaming at Monza vs high-downforce Monaco).
> 4. **Sorting & Points Allocation:** Surviving drivers are sorted to determine finishing order, and official FIA points (25-18-15...) are allocated. Aggregating across 10,000 runs yields empirical win %, podium %, top-10 %, and 95% confidence intervals.

---

### Q5: Why did you implement PyTorch Entity Embeddings for Driver, Constructor, and Circuit IDs?
> **Answer:**  
> High-cardinality categorical identifiers like Driver ID (800+ drivers), Constructor ID (200+ teams), and Circuit ID (70+ tracks) suffer when one-hot encoded (creating extreme dimensional sparsity) or target-encoded (prone to overfitting).  
> In our PyTorch architecture, `nn.Embedding(num_entities, embedding_dim=16)` maps each discrete ID into a continuous latent space. The network learns semantic representations where drivers with similar skill profiles or constructors with similar aerodynamic efficiency cluster together in latent space, concatenated directly with normalized continuous features and passed through Dense/BatchNorm/Dropout layers.

---

### Q6: How do you engineer circuit-specific features for high-speed tracks like Monza?
> **Answer:**  
> Autodromo Nazionale di Monza is characterized by low aerodynamic downforce, long full-throttle straights (>76% of lap), heavy braking chicanes, and high top speeds (355+ km/h).  
> We engineered:
> 1. **Monza Historical Track Record:** Driver & constructor starts, podium rate, and average finish specifically at Monza.
> 2. **Half-Life Recency Weighting:** Exponential decay $w_t = e^{-\lambda (T - t)}$ with a 5-year half-life, ensuring modern hybrid-era results dominate over vintage 2000s races.
> 3. **Top Speed & Overtaking Factor:** Overtaking ease at Monza into Variante del Rettifilo is modeled with lower track-position inertia compared to street circuits.

---

### Q7: How do you interpret model decisions using SHAP?
> **Answer:**  
> We employ **TreeExplainer** for Tree Ensembles (LightGBM/XGBoost) to compute exact Shapley values. Because the target is finishing position (where lower is better), a negative SHAP value indicates a favorable attribute that lowers expected finish (e.g., strong qualifying form or high Monza podium rate), while a positive SHAP value represents a performance penalty (e.g., poor constructor reliability or grid penalty). The dashboard displays local force breakdowns per driver.

---

### Q8: What evaluation metrics did you use to benchmark models across temporal walk-forward splits?
> **Answer:**  
> We utilized a multi-faceted evaluation suite:
> - **Regression & Ordinal Accuracy:** Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
> - **Rank-Order Correlation:** Spearman's $\rho$ and Kendall's $\tau$ measuring monotonic ranking alignment across the full 22-car grid.
> - **Top-K Classification:** Top-1 Winner accuracy and Top-3 Podium accuracy.
> - **Probabilistic Quality:** Brier Score, Log Loss, and Expected Calibration Error (ECE).

---

### Q9: How does the FastAPI REST backend integrate with the ML pipeline?
> **Answer:**  
> The backend is built using FastAPI with Pydantic request/response schemas. On startup, it pre-caches serialized model artifacts (`.joblib`) and pre-computed temporal feature stores. Endpoints support parameterized requests:
> - `GET /prediction?n_simulations=10000&is_post_qualifying=true`
> - `GET /prediction/{driver_id}` (returns prediction + SHAP waterfall)
> - `POST /simulation` (supports dynamic starting grid overrides for what-if scenarios)

---

### Q10: How would you scale this architecture to live in-race telemetry during a Grand Prix?
> **Answer:**  
> To support live in-race updates:
> 1. Ingest real-time lap timing and telemetry feeds (via FastF1 / MQTT / Kafka).
> 2. Compute dynamic in-race features: current tyre age/compound, tyre degradation rate ($\Delta \text{sec/lap}$), gap to car ahead/behind, pit window delta, and live safety car probability.
> 3. Run streaming Monte Carlo re-simulations at every lap completion to update win and podium probabilities in under 200ms.

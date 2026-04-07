# Hackathon IndabaX Cameroon 2026 — PM2.5 Air Quality Prediction

> **Predicting PM2.5 air quality for 40 Cameroonian cities (2020–2025)**
> Physics-calibrated ERA5 proxy · XGBoost/LightGBM · Real-time inference · GitHub Actions

---

## Results at a Glance

| Metric | Value |
|---|---|
| Cities covered | 40 (all regions of Cameroon) |
| Period | 2020-01-01 → 2025-12-20 |
| Proxy calibration target | 32.5 µg/m³ (AQLI 2023 national mean) |
| XGBoost R² (test 2025) | **0.9929** |
| LightGBM R² (test 2025) | **0.9940** |
| Ensemble R² (test 2025) | **0.9939** |
| Ensemble RMSE | **1.55 µg/m³** |
| Real-time forecast horizon | 7–16 days (Open-Meteo) |

---

## Problem

No ground-truth PM2.5 measurements exist for Cameroon. The challenge: build a reliable air quality prediction system from scratch using only reanalysis data.

**Solution:** A two-stage pipeline:
1. Build a **physics-calibrated PM2.5 proxy** from ERA5 meteorological variables and NASA FIRMS fire data
2. Train **ML models** (XGBoost + LightGBM) on this proxy to predict future PM2.5

---

## PM2.5 Proxy Formula

```
PM2.5 = C_base × F_stagnation × F_wet × F_wind × F_harmattan × F_hygro × F_fire
```

| Factor | Formula | Physics |
|---|---|---|
| `F_stagnation` | `(1000/BLH)^0.6` clipped [0.3, 3.5] | Low boundary layer = pollution accumulation |
| `F_wet` | `1/(1 + 0.08×rain_mm)` | Wet scavenging by precipitation |
| `F_wind` | `exp(-0.035×wind_kmh)` suppressed during Harmattan | Turbulent dilution |
| `F_harmattan` | `1 + 1.4×is_dry×(lat-3)/8` | Saharan dust transport Nov–Mar |
| `F_hygro` | `min(1 + 0.004×max(0,RH-75), 1.3)` | Hygroscopic growth (Pöhlker 2023) |
| `F_fire` | `1 + 0.02×log(1+FRP_75km)` | Biomass burning (NASA FIRMS MODIS) |

`C_base` auto-calibrated so national annual mean = **32.5 µg/m³**.

---

## Pipeline

```bash
# 1. Extract ERA5 weather for 40 cities (2020–2025)
conda run -n hackathon_pm25 python notebooks/01_extract_era5_pm25_target.py
conda run -n hackathon_pm25 python notebooks/01b_patch_blh_gap.py

# 2. Build PM2.5 proxy
conda run -n hackathon_pm25 python notebooks/02_build_pm25_target.py

# 3. EDA + Feature engineering + Visualisations
conda run -n hackathon_pm25 python notebooks/03_eda_feature_analysis.py
conda run -n hackathon_pm25 python notebooks/03_feature_engineering.py
conda run -n hackathon_pm25 python notebooks/03b_visualisations.py

# 4. Extract NASA FIRMS fire data (requires FIRMS_MAP_KEY in .env)
conda run -n hackathon_pm25 python notebooks/04_extract_firms_fire.py

# 5. Validation vs CAMS + Monte Carlo uncertainty
conda run -n hackathon_pm25 python notebooks/05_validation_uncertainty.py

# 6. Train XGBoost + LightGBM models
conda run -n hackathon_pm25 python notebooks/06_model_xgboost.py

# 7. Real-time inference (7-day forecast for all 40 cities)
conda run -n hackathon_pm25 python notebooks/08_inference_realtime.py
```

---

## Project Structure

```
├── notebooks/          # Pipeline scripts (numbered, run sequentially)
├── data/
│   ├── pm25_proxy_era5.parquet          # ERA5 + PM2.5 proxy (87,240 rows)
│   ├── pm25_with_uncertainty.parquet    # Proxy + Monte Carlo IC90%
│   ├── firms_fire_daily.parquet         # NASA FIRMS FRP per city/day
│   ├── predictions_latest.parquet       # Real-time 7-day forecast
│   ├── alerts_latest.json               # Active WHO threshold alerts
│   └── figures/                         # Validation and EDA figures
├── models/
│   ├── xgboost_final.pkl                # Full model (lags + meteo, R²=0.993)
│   ├── lightgbm_final.pkl               # LightGBM model (R²=0.994)
│   └── xgboost_coldstart.pkl            # Meteo-only cold-start model
├── .github/workflows/
│   └── daily_refresh.yml                # GitHub Actions daily cron (06:00 UTC)
├── app.py                               # HuggingFace Spaces entry point
├── requirements_hf.txt                  # Pinned deps for HF Spaces
├── deploy_hf.py                         # Manual upload script (huggingface_hub)
├── RAPPORT_SCIENTIFIQUE.md
├── RAPPORT_VULGARISE.md
└── requirements.txt
```

---

## Key Scientific Choices

- **BLH_ALPHA = 0.6** — sub-linear exponent (Seinfeld & Pandis 2016, range 0.4–0.8)
- **RAIN_K = 0.08** — wet scavenging (Berge & Jakobsen 1998, range 0.05–0.12)
- **HARMATTAN_STRENGTH = 1.4** — seasonal ratio ×2.21 vs CAMS ×2.02 (Mbuh et al. 2021)
- **F_hygro cap = 1.3** — hygroscopic growth ceiling (Pöhlker et al. 2023)
- **F_fire radius = 75 km** — biomass burning transport (Gordon et al. 2023)
- **Validation vs CAMS**: r=0.339, NMB=+85% (expected — CAMS underestimates African PM2.5 by 20–50%)
- **Monte Carlo IC90%**: ~±8 µg/m³ at national mean — most sensitive parameter: RAIN_K (|r|=0.80)

---

## ML Validation Strategy

Expanding-window cross-validation (never random split — temporal data):

```
Fold 1: Train ≤2020 → Val 2021  (R²=0.970)
Fold 2: Train ≤2021 → Val 2022  (R²=0.994)
Fold 3: Train ≤2022 → Val 2023  (R²=0.994)
Fold 4: Train ≤2023 → Val 2024  (R²=0.994)
Test:   Train ≤2024 → Test 2025 (R²=0.993)
```

**Ablation study:** meteo-only XGBoost achieves R²=0.993 — the model learns physics (Harmattan, precipitation, BLH), not autocorrelation.
Top features: `is_true_harmattan` (41.6%), `climate_zone` (18.2%), `pm25_proxy_roll3_mean` (12.5%).

---

## Dashboard

**Live HuggingFace Space:** https://huggingface.co/spaces/QUASAR-30/pm25-cameroun

3 interactive pages:

| Page | Content |
|---|---|
| **TEMPS RÉEL** | Mapbox scatter map · WHO alert panel by region/threshold · 40-city HTML table with progress bars · CSV/JSON export |
| **PAR VILLE** | Time series 2020–2025 with Monte Carlo IC90% · 7-day Open-Meteo forecast · PM2.5 vs precip/BLH/wind correlation · Monthly seasonality |
| **CLASSEMENT** | Top 10 cities with error bars · Latitude vs PM2.5 gradient (R²=0.882) · Monthly heatmap · Annual trends by region · Model performance + feature importance + ablation study |

Updated daily at 06:00 UTC via GitHub Actions.

---

## Real-Time Pipeline

Automated daily forecast via GitHub Actions (`.github/workflows/daily_refresh.yml`):
- Fetches Open-Meteo 7-day weather forecast for all 40 cities
- Runs dual-model inference (full model or cold-start fallback)
- Updates `predictions_latest.parquet` and `alerts_latest.json`
- Commits and pushes results automatically at 06:00 UTC

---

## Setup

```bash
# Create environment
conda create -n hackathon_pm25 python=3.11
conda activate hackathon_pm25
pip install -r requirements.txt

# Configure secrets
cp .env.example .env
# Edit .env: add your FIRMS_MAP_KEY (free at firms.modaps.eosdis.nasa.gov/api/map_key/)
```

---

## References

- AQLI (2023). Cameroon annual mean PM2.5: 32.5 µg/m³
- Mbuh et al. (2021). Harmattan dust transport in northern Cameroon
- Pöhlker et al. (2023). Hygroscopic growth of African aerosols
- Gordon et al. (2023, GeoHealth). Biomass burning and PM2.5 mortality in Central Africa
- Seinfeld & Pandis (2016). Atmospheric Chemistry and Physics
- WHO (2021). Global Air Quality Guidelines

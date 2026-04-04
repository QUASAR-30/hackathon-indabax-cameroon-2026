# Rapport modèle PM2.5 — Hackathon IndabaX 2026


## XGBoost — Test 2025
- RMSE : 1.663 µg/m³
- MAE  : 0.926 µg/m³
- R²   : 0.9929
- MAPE : 2.5%

## LightGBM — Test 2025
- RMSE : 1.526 µg/m³
- MAE  : 0.861 µg/m³
- R²   : 0.9940
- MAPE : 2.3%

## Ensemble — Test 2025
- RMSE : 1.547 µg/m³
- MAE  : 0.852 µg/m³
- R²   : 0.9939
- MAPE : 2.2%

## Features (top 15 XGBoost)

- is_true_harmattan : 0.4161
- climate_zone : 0.1824
- pm25_proxy_roll3_mean : 0.1249
- harmattan_intensity : 0.0276
- lat_x_harmattan : 0.0219
- pm25_proxy_lag1 : 0.0217
- pm25_proxy_roll7_mean : 0.0172
- weather_proxy : 0.0155
- precip_log : 0.0152
- lat_norm : 0.0149
- precipitation_sum : 0.0130
- wind_x_wet : 0.0124
- blh_mean : 0.0113
- blh_log : 0.0105
- is_low_blh : 0.0094

## Ablation Study — Test 2025

Comparaison : Persistence vs Météo-only vs Full model

| Modèle | RMSE | MAE | R² | MAPE |
|--------|------|-----|-----|------|
| Persistence (PM2.5[t-1]) | 9.98 | 7.34 | 0.7453 | 28.2% |
| Météo-only XGBoost | 1.61 | 0.91 | 0.9934 | 2.4% |
| Full XGBoost | 1.66 | 0.93 | 0.9929 | 2.5% |

**ΔR² météo vs persistence** : +0.2481
**ΔR² full  vs météo-only**  : -0.0005
**Les lags contribuent -0% du gain total de R²**

Features retirées pour météo-only : 5 lags PM2.5
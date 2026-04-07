---
title: PM2.5 Cameroun — IndabaX 2026
emoji: 🌿
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: 1.56.0
app_file: app.py
pinned: true
license: mit
short_description: PM2.5 forecast for 40 Cameroonian cities
---

# PM2.5 Cameroun — Qualité de l'Air en Temps Réel

> **Hackathon IndabaX Cameroun 2026** — Prévisions PM2.5 pour 40 villes camerounaises  
> Pipeline ERA5 · NASA FIRMS · XGBoost/LightGBM · R²=0.994 · Mis à jour quotidiennement

## Dashboard

**3 pages interactives :**

- **TEMPS RÉEL** — Carte des concentrations PM2.5 · Alertes OMS · Prévisions 7 jours
- **PAR VILLE** — Série temporelle 2020–2025 avec IC90% Monte Carlo · Drivers météo
- **CLASSEMENT** — Top 40 villes · Gradient Nord-Sud · Saisonnalité · Performance modèles

## Modèles

| Modèle | R² | RMSE |
|---|---|---|
| XGBoost (full) | 0.9929 | 1.66 µg/m³ |
| LightGBM (full) | **0.9940** | **1.53 µg/m³** |
| Ensemble | 0.9939 | 1.55 µg/m³ |
| XGBoost cold-start (météo uniquement) | 0.9959 | — |

## Sources

- **ERA5** — Données météo historiques via Open-Meteo archive API
- **NASA FIRMS MODIS** — Feux de biomasse (FRP 75 km autour de chaque ville)
- **Open-Meteo forecast** — Prévisions 7–16 jours pour l'inférence temps réel
- **Proxy PM2.5** calibré sur AQLI 2023 (moyenne nationale Cameroun : 32.5 µg/m³)

## Mise à jour automatique

GitHub Actions cron **06:00 UTC** (07:00 WAT) — les prévisions sont poussées ici chaque matin.

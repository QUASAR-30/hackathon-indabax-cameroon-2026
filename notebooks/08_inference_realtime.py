"""
Pipeline d'inference temps reel — PM2.5 Cameroun
=================================================
Fetch Open-Meteo forecast -> feature engineering exact -> prediction ML -> alertes OMS

Strategie : concatene les 60 derniers jours historiques + forecast,
applique le meme feature engineering que 03_feature_engineering.py,
puis predit uniquement sur les jours forecast.

Usage :
    conda run -n hackathon_pm25 python notebooks/08_inference_realtime.py
    conda run -n hackathon_pm25 python notebooks/08_inference_realtime.py --city Maroua
    conda run -n hackathon_pm25 python notebooks/08_inference_realtime.py --cold-start
    conda run -n hackathon_pm25 python notebooks/08_inference_realtime.py --days 16

Outputs :
    data/predictions_latest.parquet
    data/alerts_latest.json
"""

import argparse
import json
import pickle
import sys
import warnings
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT / "notebooks"))

OMS_24H     = 15.0
BLH_MIN     = 250.0
HARMATTAN_STRENGTH = 1.4
RAIN_K      = 0.08
C_BASE_DEFAULT = 7.5

ALERT_LEVELS = [
    ("Bon",          0,       OMS_24H),
    ("Mauvais",      OMS_24H, 35),
    ("Tres Mauvais", 35,      55),
    ("Dangereux",    55,      75),
    ("Extreme",      75,      9999),
]

CITIES = [
    {"city": "Abong-Mbang",  "latitude": 3.9833,  "longitude": 13.1833, "region": "Est"},
    {"city": "Akonolinga",   "latitude": 3.7667,  "longitude": 12.2500, "region": "Centre"},
    {"city": "Ambam",        "latitude": 2.3833,  "longitude": 11.2833, "region": "Sud"},
    {"city": "Bafia",        "latitude": 4.7500,  "longitude": 11.2300, "region": "Centre"},
    {"city": "Bafoussam",    "latitude": 5.4667,  "longitude": 10.4167, "region": "Ouest"},
    {"city": "Bamenda",      "latitude": 5.9627,  "longitude": 10.1479, "region": "Nord-Ouest"},
    {"city": "Batouri",      "latitude": 4.4333,  "longitude": 14.3667, "region": "Est"},
    {"city": "Bertoua",      "latitude": 4.5778,  "longitude": 13.6833, "region": "Est"},
    {"city": "Buea",         "latitude": 4.1667,  "longitude":  9.2333, "region": "Sud-Ouest"},
    {"city": "Douala",       "latitude": 4.0483,  "longitude":  9.7043, "region": "Littoral"},
    {"city": "Dschang",      "latitude": 5.4500,  "longitude": 10.0500, "region": "Ouest"},
    {"city": "Ebolowa",      "latitude": 2.9000,  "longitude": 11.1500, "region": "Sud"},
    {"city": "Edea",         "latitude": 3.8000,  "longitude": 10.1333, "region": "Littoral"},
    {"city": "Foumban",      "latitude": 5.7167,  "longitude": 10.9000, "region": "Ouest"},
    {"city": "Garoua",       "latitude": 9.2992,  "longitude": 13.3954, "region": "Nord"},
    {"city": "Guider",       "latitude": 9.9300,  "longitude": 13.9400, "region": "Nord"},
    {"city": "Kousseri",     "latitude": 12.0667, "longitude": 15.0167, "region": "Extreme-Nord"},
    {"city": "Kribi",        "latitude": 2.9500,  "longitude":  9.9100, "region": "Sud"},
    {"city": "Kumba",        "latitude": 4.6333,  "longitude":  9.4500, "region": "Sud-Ouest"},
    {"city": "Kumbo",        "latitude": 6.2000,  "longitude": 10.6667, "region": "Nord-Ouest"},
    {"city": "Limbe",        "latitude": 4.0167,  "longitude":  9.2100, "region": "Sud-Ouest"},
    {"city": "Loum",         "latitude": 4.7167,  "longitude":  9.7333, "region": "Littoral"},
    {"city": "Mamfe",        "latitude": 5.7667,  "longitude":  9.3167, "region": "Sud-Ouest"},
    {"city": "Maroua",       "latitude": 10.5833, "longitude": 14.3167, "region": "Extreme-Nord"},
    {"city": "Mbalmayo",     "latitude": 3.5167,  "longitude": 11.5017, "region": "Centre"},
    {"city": "Mbengwi",      "latitude": 5.9833,  "longitude": 10.0167, "region": "Nord-Ouest"},
    {"city": "Mbouda",       "latitude": 5.6167,  "longitude": 10.2667, "region": "Ouest"},
    {"city": "Meiganga",     "latitude": 6.5167,  "longitude": 14.2833, "region": "Adamaoua"},
    {"city": "Mokolo",       "latitude": 10.7333, "longitude": 13.8000, "region": "Extreme-Nord"},
    {"city": "Ngaoundere",   "latitude": 7.3167,  "longitude": 13.5833, "region": "Adamaoua"},
    {"city": "Nkongsamba",   "latitude": 4.9500,  "longitude":  9.9333, "region": "Littoral"},
    {"city": "Poli",         "latitude": 8.4667,  "longitude": 13.2400, "region": "Nord"},
    {"city": "Sangmelima",   "latitude": 2.9333,  "longitude": 11.9833, "region": "Sud"},
    {"city": "Tibati",       "latitude": 6.4667,  "longitude": 12.6167, "region": "Adamaoua"},
    {"city": "Tignere",      "latitude": 7.3700,  "longitude": 12.6500, "region": "Adamaoua"},
    {"city": "Touboro",      "latitude": 7.7667,  "longitude": 15.3667, "region": "Nord"},
    {"city": "Wum",          "latitude": 6.3833,  "longitude": 10.0667, "region": "Nord-Ouest"},
    {"city": "Yagoua",       "latitude": 10.3333, "longitude": 15.2333, "region": "Extreme-Nord"},
    {"city": "Yaounde",      "latitude": 3.8667,  "longitude": 11.5167, "region": "Centre"},
    {"city": "Yokadouma",    "latitude": 3.5139,  "longitude": 15.0539, "region": "Est"},
]


# ── 1. FETCH OPEN-METEO ────────────────────────────────────────────────────────
def fetch_forecast(forecast_days=7):
    print(f"  Fetch Open-Meteo ({forecast_days} jours)...")
    lats = ",".join(str(c["latitude"]) for c in CITIES)
    lons = ",".join(str(c["longitude"]) for c in CITIES)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lats, "longitude": lons,
        "daily": ",".join([
            "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max",
            "temperature_2m_max", "temperature_2m_min", "shortwave_radiation_sum",
            "relative_humidity_2m_max", "relative_humidity_2m_min",
            "wind_direction_10m_dominant", "et0_fao_evapotranspiration",
        ]),
        "hourly": "boundary_layer_height",
        "forecast_days": forecast_days,
        "timezone": "Africa/Lagos",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        data = [data]

    rows = []
    for i, loc in enumerate(data):
        city_info = CITIES[i]
        daily = loc["daily"]
        hourly_blh = loc.get("hourly", {}).get("boundary_layer_height", [])
        n_days = len(daily["time"])

        for d in range(n_days):
            day_blh = [v for v in hourly_blh[d*24:(d+1)*24] if v is not None]
            blh_m  = max(float(np.mean(day_blh))  if day_blh else BLH_MIN, BLH_MIN)
            blh_mn = max(float(np.min(day_blh))   if day_blh else BLH_MIN, BLH_MIN)
            blh_mx = max(float(np.max(day_blh))   if day_blh else BLH_MIN, BLH_MIN)

            et0_list = daily.get("et0_fao_evapotranspiration") or [0]*n_days
            rows.append({
                "city": city_info["city"], "latitude": city_info["latitude"],
                "longitude": city_info["longitude"], "region": city_info["region"],
                "time": pd.to_datetime(daily["time"][d]),
                "precipitation_sum":          float(daily["precipitation_sum"][d] or 0),
                "wind_speed_10m_max":          float(daily["wind_speed_10m_max"][d] or 0),
                "wind_gusts_10m_max":          float(daily["wind_gusts_10m_max"][d] or 0),
                "temperature_2m_max":          float(daily["temperature_2m_max"][d] or 25),
                "temperature_2m_min":          float(daily["temperature_2m_min"][d] or 15),
                "shortwave_radiation_sum":     float(daily["shortwave_radiation_sum"][d] or 0),
                "relative_humidity_2m_max":    float(daily["relative_humidity_2m_max"][d] or 50),
                "relative_humidity_2m_min":    float(daily["relative_humidity_2m_min"][d] or 30),
                "wind_direction_10m_dominant": float(daily["wind_direction_10m_dominant"][d] or 180),
                "et0_fao_evapotranspiration":  float(et0_list[d] or 0),
                "blh_mean": blh_m, "blh_min": blh_mn, "blh_max": blh_mx,
                # proxy pour warm-up (sera ecrase par historique reel si dispo)
                "pm25_proxy": np.nan,
                "f_fire": 1.0,
            })

    df = pd.DataFrame(rows)
    print(f"  OK: {len(df)} lignes ({df['city'].nunique()} villes x {forecast_days}j)")
    return df


# ── 2. PROXY POUR FORECAST (warm-up lags) ────────────────────────────────────
def compute_proxy(df, c_base=C_BASE_DEFAULT):
    lat  = df["latitude"]
    rain = df["precipitation_sum"].clip(lower=0)
    wind = df["wind_speed_10m_max"].clip(lower=0)
    rh   = df["relative_humidity_2m_max"].clip(lower=0)
    blh  = df["blh_mean"].clip(lower=BLH_MIN)
    month = df["time"].dt.month
    wind_dir = df["wind_direction_10m_dominant"]

    is_dry  = month.isin([11, 12, 1, 2, 3]).astype(float)
    is_north = ((wind_dir >= 315) | (wind_dir <= 90))
    is_harm  = (is_dry > 0) & is_north

    f_stag  = ((1000 / blh) ** 0.6).clip(0.3, 3.5)
    f_wet   = 1 / (1 + RAIN_K * rain)
    f_wind  = np.where(is_harm, 1.0, np.exp(-0.035 * wind))
    f_harm  = 1 + HARMATTAN_STRENGTH * is_dry * (lat - 3) / 8
    f_hygro = (1 + 0.004 * (rh - 75).clip(lower=0)).clip(upper=1.3)
    f_hygro = np.where(rain > 1, 1.0, f_hygro)
    return pd.Series((c_base * f_stag * f_wet * f_wind * f_harm * f_hygro).values,
                     index=df.index)


# ── 3. FEATURE ENGINEERING (reproduit 03_feature_engineering.py exactement) ──
def build_features(df_forecast, df_history=None):
    """
    Concatene historique (60j) + forecast, applique le feature engineering complet,
    puis retourne uniquement les lignes forecast.
    """
    # Calculer proxy pour les jours forecast (warm-up lags)
    df_forecast = df_forecast.copy()
    df_forecast["pm25_proxy"] = compute_proxy(df_forecast)

    # Combiner historique + forecast
    if df_history is not None and not df_history.empty:
        needed_cols = ["city", "time", "latitude", "longitude", "region",
                       "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max",
                       "temperature_2m_max", "temperature_2m_min", "shortwave_radiation_sum",
                       "relative_humidity_2m_max", "relative_humidity_2m_min",
                       "wind_direction_10m_dominant", "et0_fao_evapotranspiration",
                       "blh_mean", "blh_min", "blh_max", "pm25_proxy"]
        hist_cols = [c for c in needed_cols if c in df_history.columns]
        hist = df_history[hist_cols].copy()
        for c in needed_cols:
            if c not in hist.columns:
                hist[c] = 0.0
        hist["f_fire"] = 1.0
        df = pd.concat([hist, df_forecast], ignore_index=True)
        forecast_times = set(df_forecast["time"].unique())
    else:
        df = df_forecast.copy()
        forecast_times = set(df["time"].unique())

    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    # ── Temporelles ──────────────────────────────────────────────────────────
    df["year"]        = df["time"].dt.year
    df["month"]       = df["time"].dt.month
    df["day_of_year"] = df["time"].dt.dayofyear
    df["day_of_week"] = df["time"].dt.dayofweek
    df["quarter"]     = df["time"].dt.quarter
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]     = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]     = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["dow_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ── Daylight (Spencer 1971) ───────────────────────────────────────────────
    doy = df["day_of_year"]
    lat_rad = np.radians(df["latitude"])
    B = 2 * np.pi * (doy - 1) / 365
    decl = (0.006918 - 0.399912*np.cos(B) + 0.070257*np.sin(B)
            - 0.006758*np.cos(2*B) + 0.000907*np.sin(2*B)
            - 0.002697*np.cos(3*B) + 0.00148*np.sin(3*B))
    cos_ha = (-np.tan(lat_rad) * np.tan(decl)).clip(-1, 1)
    df["daylight_duration"] = 2 * np.arccos(cos_ha) * (3600 * 12 / np.pi)

    # ── Harmattan ─────────────────────────────────────────────────────────────
    df["is_harmattan"]  = df["month"].isin([11, 12, 1, 2, 3]).astype(int)
    df["is_wet_season"] = 1 - df["is_harmattan"]
    conds = [df["month"].isin([12,1,2]), df["month"].isin([3,4,5,6]),
             df["month"].isin([7,8]),   df["month"].isin([9,10,11])]
    df["season_code"] = np.select(conds, [0,1,2,3], default=1)
    lat_n = ((df["latitude"] - 3) / (11 - 3)).clip(0, 1)
    df["harmattan_intensity"] = df["is_harmattan"] * lat_n
    wdir = df["wind_direction_10m_dominant"].fillna(180)
    is_north = ((wdir >= 315) | (wdir <= 90)).astype(int)
    df["is_true_harmattan"] = ((df["is_harmattan"] == 1) & (is_north == 1)).astype(int)

    # ── Meteo derivees ────────────────────────────────────────────────────────
    df["temp_amplitude"] = df["temperature_2m_max"] - df["temperature_2m_min"]
    df["is_dry_day"]     = (df["precipitation_sum"] < 0.1).astype(int)
    df["is_heavy_rain"]  = (df["precipitation_sum"] > 10).astype(int)
    df["rh_mean"]        = (df["relative_humidity_2m_max"] + df["relative_humidity_2m_min"]) / 2
    df["rh_amplitude"]   = df["relative_humidity_2m_max"] - df["relative_humidity_2m_min"]
    blh = df["blh_mean"].clip(lower=50)
    df["blh_log"]        = np.log(blh)
    df["blh_inv"]        = 1000 / blh
    df["is_low_blh"]     = (blh < 400).astype(int)
    df["precip_log"]     = np.log1p(df["precipitation_sum"].clip(lower=0))
    df["blh_min_log"]    = np.log1p(df["blh_min"].clip(lower=0))
    df["wind_log"]       = np.log1p(df["wind_speed_10m_max"].clip(lower=0))
    df["et0_log"]        = np.log1p(df["et0_fao_evapotranspiration"].clip(lower=0))
    df["stagnation_index"] = ((df["wind_speed_10m_max"] < 5).astype(int) +
                              df["is_low_blh"] + df["is_dry_day"])

    # ── Spatial ───────────────────────────────────────────────────────────────
    region_order = {"Sud": 0, "Sud-Ouest": 1, "Littoral": 2, "Centre": 3,
                    "Est": 4, "Ouest": 5, "Nord-Ouest": 6,
                    "Adamaoua": 7, "Nord": 8, "Extreme-Nord": 9}
    df["region_code"] = df["region"].map(region_order).fillna(5).astype(int)
    lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
    lon_min, lon_max = df["longitude"].min(), df["longitude"].max()
    df["lat_norm"] = (df["latitude"] - lat_min) / max(lat_max - lat_min, 1e-6)
    df["lon_norm"] = (df["longitude"] - lon_min) / max(lon_max - lon_min, 1e-6)
    df["climate_zone"] = pd.cut(df["latitude"],
                                 bins=[-np.inf, 5, 8, np.inf], labels=[0,1,2]
                                ).astype(float).astype(int)
    city_ids = {c: i for i, c in enumerate(sorted(df["city"].unique()))}
    df["city_id"] = df["city"].map(city_ids)

    # ── BLH regime ────────────────────────────────────────────────────────────
    df["blh_regime"] = pd.cut(df["blh_mean"],
                               bins=[-np.inf, 200, 500, 1500, np.inf],
                               labels=[0,1,2,3]).astype(float).astype(int)

    # ── Weather proxy ─────────────────────────────────────────────────────────
    precip = df["precipitation_sum"].fillna(0).clip(lower=0)
    precip_cat = pd.cut(precip, bins=[-np.inf, 0.1, 1.0, 10.0, np.inf],
                         labels=[0,1,2,3]).astype(float)
    df["precip_cat"] = precip_cat
    if "shortwave_radiation_sum" in df.columns:
        rad_norm = df.groupby("month")["shortwave_radiation_sum"].transform(
            lambda x: x / max(x.quantile(0.95), 1.0)
        ).clip(0, 1)
        df["rad_norm"] = rad_norm
        wconds = [(precip_cat==0)&(rad_norm>0.75), (precip_cat==0)&(rad_norm<=0.75),
                  precip_cat==1, precip_cat==2, precip_cat==3]
        df["weather_proxy"] = np.select(wconds, [0,1,2,3,4], default=1)
    else:
        df["rad_norm"] = 0.5
        df["weather_proxy"] = precip_cat

    # ── Interactions ─────────────────────────────────────────────────────────
    df["temp_amp_x_blh"]  = df["temp_amplitude"] * df["blh_mean"] / 1000
    df["wind_x_harmattan"]= df["wind_speed_10m_max"] * df["is_harmattan"]
    df["wind_x_wet"]      = df["wind_speed_10m_max"] * (1/(1+RAIN_K*precip))
    df["blh_x_precip"]    = df["blh_mean"] / 1000 * df["precipitation_sum"]
    df["lat_x_harmattan"] = df["lat_norm"] * df["harmattan_intensity"]

    # ── Target encoding par ville ─────────────────────────────────────────────
    city_te = df.groupby("city")["pm25_proxy"].mean().apply(np.log1p).to_dict()
    df["city_pm25_te"] = df["city"].map(city_te).fillna(0.0)

    # ── Lags ─────────────────────────────────────────────────────────────────
    LAG_VARS = ["pm25_proxy", "precipitation_sum", "wind_speed_10m_max",
                "blh_mean", "relative_humidity_2m_max", "temperature_2m_max"]
    LAG_DAYS = [1, 2, 3, 7, 14]
    df = df.sort_values(["city", "time"])
    for var in LAG_VARS:
        if var not in df.columns:
            continue
        for lag in LAG_DAYS:
            df[f"{var}_lag{lag}"] = df.groupby("city")[var].shift(lag)

    # ── Rolling ───────────────────────────────────────────────────────────────
    ROLL_VARS = ["pm25_proxy", "precipitation_sum", "wind_speed_10m_max",
                 "blh_mean", "temperature_2m_max",
                 "relative_humidity_2m_max", "shortwave_radiation_sum"]
    ROLL_WINS = [3, 7, 14, 30]
    for var in ROLL_VARS:
        if var not in df.columns:
            continue
        for win in ROLL_WINS:
            grp = df.groupby("city")[var]
            df[f"{var}_roll{win}_mean"] = grp.transform(
                lambda x: x.shift(1).rolling(win, min_periods=1).mean()
            )
            if win >= 7:
                df[f"{var}_roll{win}_std"] = grp.transform(
                    lambda x: x.shift(1).rolling(win, min_periods=2).std()
                )

    # Dry streak
    def dry_streak(series):
        streak, count = [], 0
        for val in series.shift(1).fillna(0):
            count = count + 1 if val == 1 else 0
            streak.append(count)
        return streak
    df["dry_streak"] = df.groupby("city")["is_dry_day"].transform(dry_streak)

    # Garder uniquement les jours forecast
    df_out = df[df["time"].isin(forecast_times)].copy()
    df_out = df_out.sort_values(["city", "time"]).reset_index(drop=True)
    return df_out


# ── 4. INFERENCE ──────────────────────────────────────────────────────────────
def load_models():
    with open(MODELS / "xgboost_final.pkl", "rb") as f:
        model_full = pickle.load(f)
    cold_path = MODELS / "xgboost_coldstart.pkl"
    model_cold, features_cold = model_full, None
    if cold_path.exists():
        with open(cold_path, "rb") as f:
            cold_data = pickle.load(f)
            model_cold   = cold_data["model"]
            features_cold = cold_data["features"]
    return model_full, model_cold, features_cold


def predict(df_features, model_full, model_cold, features_cold, use_cold=False):
    if use_cold and features_cold is not None:
        avail = [f for f in features_cold if f in df_features.columns]
        X = df_features.copy()
        for col in features_cold:
            if col not in X.columns:
                X[col] = 0.0
        X = X[features_cold].fillna(0)
        preds_log = model_cold.predict(X)
        model_used = "cold_start"
    else:
        features_full = list(model_full.feature_names_in_)
        X = df_features.copy()
        for col in features_full:
            if col not in X.columns:
                X[col] = 0.0
        X = X[features_full].fillna(0)
        preds_log = model_full.predict(X)
        model_used = "full"

    preds_pm25 = np.clip(np.expm1(preds_log), 0, None)
    result = df_features[["city","time","latitude","longitude","region"]].copy()
    result["pm25_pred"]    = np.round(preds_pm25, 1)
    result["model_used"]   = model_used
    result["forecast_date"]= date.today().isoformat()
    return result


# ── 5. ALERTES ────────────────────────────────────────────────────────────────
def get_level(pm25):
    for name, lo, hi in ALERT_LEVELS:
        if lo <= pm25 < hi:
            return name
    return "Extreme"


def generate_alerts(df_pred):
    TIPS = {
        "Mauvais":      "Limitez les activites exterieures pour enfants et personnes agees.",
        "Tres Mauvais": "Evitez les activites physiques en exterieur. Masque recommande.",
        "Dangereux":    "Restez a l interieur. Fenetres fermees. Evitez tout effort.",
        "Extreme":      "URGENCE: Confinement. Masque FFP2. Consultez un medecin si symptomes.",
    }
    alerts = []
    for _, row in df_pred.iterrows():
        pm = row["pm25_pred"]
        if pm >= OMS_24H:
            level = get_level(pm)
            alerts.append({
                "city":      row["city"],
                "date":      str(row["time"].date() if hasattr(row["time"],"date") else row["time"]),
                "pm25":      float(pm),
                "level":     level,
                "region":    row["region"],
                "latitude":  float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "message":   TIPS.get(level, ""),
            })
    return sorted(alerts, key=lambda x: -x["pm25"])


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Inference PM2.5 temps reel")
    parser.add_argument("--city",       help="Filtrer sur une ville")
    parser.add_argument("--cold-start", action="store_true", help="Modele meteo-only")
    parser.add_argument("--days",       type=int, default=7, help="Jours de prevision (max 16)")
    parser.add_argument("--no-save",    action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Pipeline Inference PM2.5 -- {date.today()}")
    print(f"  Mode : {'Cold-start (meteo-only)' if args.cold_start else 'Full model'}")
    print(f"{'='*60}\n")

    # 1. Fetch forecast
    df_forecast = fetch_forecast(forecast_days=args.days)
    if args.city:
        matches = [c["city"] for c in CITIES if args.city.lower() in c["city"].lower()]
        if matches:
            df_forecast = df_forecast[df_forecast["city"].isin(matches)]
            print(f"  Filtre: {matches}")

    # 2. Historique 60 jours pour lags corrects
    df_history = None
    proxy_path = DATA / "pm25_proxy_era5.parquet"
    if proxy_path.exists() and not args.cold_start:
        print("  Chargement historique (60 derniers jours)...")
        cols = ["city","time","latitude","longitude","region","pm25_proxy",
                "precipitation_sum","wind_speed_10m_max","wind_gusts_10m_max",
                "temperature_2m_max","temperature_2m_min","shortwave_radiation_sum",
                "relative_humidity_2m_max","relative_humidity_2m_min",
                "wind_direction_10m_dominant","blh_mean","blh_min","blh_max"]
        hist = pd.read_parquet(proxy_path, columns=[c for c in cols
                               if c in pd.read_parquet(proxy_path, columns=["city"]).columns
                               or c == "city"])
        # recharge avec toutes les colonnes disponibles
        hist = pd.read_parquet(proxy_path)
        hist["time"] = pd.to_datetime(hist["time"])
        cutoff = df_forecast["time"].min() - pd.Timedelta(days=65)
        df_history = hist[hist["time"] >= cutoff].copy()
        if "et0_fao_evapotranspiration" not in df_history.columns:
            df_history["et0_fao_evapotranspiration"] = 0.0
        if "wind_gusts_10m_max" not in df_history.columns:
            df_history["wind_gusts_10m_max"] = df_history.get("wind_speed_10m_max", 0)

        # Verifier si l'historique est suffisant (>= 14 jours avant forecast)
        hist_days = len(df_history) // max(df_history["city"].nunique(), 1)
        if hist_days < 14:
            print(f"  Historique insuffisant ({hist_days}j < 14j requis) -> bascule cold-start")
            args.cold_start = True
            df_history = None
        else:
            print(f"  OK: {len(df_history)} lignes ({df_history['time'].min().date()} -> {df_history['time'].max().date()})")

    # 3. Features
    print("  Feature engineering...")
    df_features = build_features(df_forecast, df_history)
    print(f"  OK: {len(df_features.columns)} features, {len(df_features)} lignes forecast")

    # 4. Inference
    print("  Chargement modeles...")
    model_full, model_cold, features_cold = load_models()
    print(f"  Predictions (mode: {'cold-start' if args.cold_start else 'full'})...")
    df_pred = predict(df_features, model_full, model_cold, features_cold,
                      use_cold=args.cold_start)

    # 5. Alertes
    alerts = generate_alerts(df_pred)
    n_danger = sum(1 for a in alerts if a["level"] in ["Dangereux","Extreme"])

    print(f"\n  Predictions: {len(df_pred)} (villes x jours)")
    print(f"  Alertes OMS: {len(alerts)} depassements")
    print(f"  Dangereux  : {n_danger} cas > 55 ug/m3")

    # Resume aujourd'hui
    today_pred = df_pred[df_pred["time"] == df_pred["time"].min()]
    print(f"\n  Previsions {today_pred['time'].iloc[0].date()} :")
    print(f"  {'Ville':<20} {'PM2.5':>8}  {'Niveau':<15}")
    print(f"  {'-'*48}")
    for _, row in today_pred.sort_values("pm25_pred", ascending=False).head(15).iterrows():
        level = get_level(row["pm25_pred"])
        flag  = " !!" if row["pm25_pred"] >= 55 else (" !" if row["pm25_pred"] >= OMS_24H else "")
        print(f"  {row['city']:<20} {row['pm25_pred']:>7.1f}  {level:<15}{flag}")

    # 6. Sauvegarde
    if not args.no_save:
        out_pred = DATA / "predictions_latest.parquet"
        df_pred.to_parquet(out_pred, index=False)
        print(f"\n  Saved: {out_pred}")

        out_alerts = DATA / "alerts_latest.json"
        with open(out_alerts, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at":  datetime.now().isoformat(),
                "forecast_days": args.days,
                "total_alerts":  len(alerts),
                "dangerous":     n_danger,
                "model":         "cold_start" if args.cold_start else "full",
                "alerts":        alerts,
            }, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {out_alerts}")
    print()


if __name__ == "__main__":
    main()

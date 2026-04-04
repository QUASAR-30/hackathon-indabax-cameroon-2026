"""
Extraction ERA5 — Variable cible PM2.5 proxy
Hackathon IndabaX Cameroun 2026

Stratégie validée :
  - Source : Open-Meteo Historical Weather API (ERA5, depuis 1940, gratuit)
  - V
  yer_height (agrégé en daily_mean + daily_min)
  - Période          : 2020-01-01 → 2025-12-20
  - Villes           : 40 villes du dataset hackathon

Problème dataset identifié :
  14/40 villes ont des coordonnées corrompues dans le fichier Excel
  (Excel a interprété des décimales comme des dates, pattern : YYYY-MM-DD → coordonnée = DD.MM)
  → Coordonnées corrigées via table de référence géographique ci-dessous
"""

import os
import time
import json
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_URL   = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2020-01-01"
END_DATE   = "2025-12-20"
OUTPUT_DIR = Path("data/era5_raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DAILY_VARS = ",".join([
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "temperature_2m_max",
    "temperature_2m_min",
    "shortwave_radiation_sum",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "et0_fao_evapotranspiration",   # déjà dans le dataset — cohérence
    "wind_direction_10m_dominant",
])
HOURLY_VARS = "boundary_layer_height"

DELAY_BETWEEN_REQUESTS = 1.2   # secondes (safe pour 600 req/min free tier)
MAX_RETRIES            = 3
TIMEOUT_SEC            = 120

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("data/extraction.log")]
)
log = logging.getLogger(__name__)

# ── Table de référence des villes (coordonnées corrigées) ──────────────────────
# Sources : géographie officielle Cameroun + correction du bug Excel
# (14 villes avaient des coordonnées encodées comme dates dans le fichier source)

CITIES = [
    # city              region          lat       lon
    # ── Adamaoua ──────────────────────────────────────────
    ("Tibati",          "Adamaoua",     6.4667,   12.6167),
    ("Tignere",         "Adamaoua",     7.37,     12.65),
    ("Ngaoundere",      "Adamaoua",     7.3167,   13.5833),
    ("Meiganga",        "Adamaoua",     6.5167,   14.2833),
    # ── Centre ────────────────────────────────────────────
    ("Bafia",           "Centre",       4.75,     11.23),
    ("Akonolinga",      "Centre",       3.7667,   12.25),
    ("Yaounde",         "Centre",       3.8667,   11.5167),  # lon corrigée (était 11.05)
    ("Mbalmayo",        "Centre",       3.5167,   11.5017),  # lon corrigée (était 11.05)
    # ── Est ───────────────────────────────────────────────
    ("Yokadouma",       "Est",          3.5139,   15.0539),  # lon corrigée (était 15.05 ✓)
    ("Batouri",         "Est",          4.4333,   14.3667),
    ("Bertoua",         "Est",          4.5778,   13.6833),
    ("Abong-Mbang",     "Est",          3.9833,   13.1833),
    # ── Extreme-Nord ──────────────────────────────────────
    ("Yagoua",          "Extreme-Nord", 10.3333,  15.2333),
    ("Maroua",          "Extreme-Nord", 10.5833,  14.3167),
    ("Kousseri",        "Extreme-Nord", 12.0667,  15.0167),  # les deux corrigées
    ("Mokolo",          "Extreme-Nord", 10.7333,  13.8000),  # lon corrigée (était 13.08)
    # ── Littoral ──────────────────────────────────────────
    ("Loum",            "Littoral",     4.7167,   9.7333),
    ("Douala",          "Littoral",     4.0483,   9.7043),   # les deux corrigées
    ("Edea",            "Littoral",     3.8000,   10.1333),  # lat corrigée (était 3.08)
    ("Nkongsamba",      "Littoral",     4.9500,   9.9333),
    # ── Nord ──────────────────────────────────────────────
    ("Garoua",          "Nord",         9.2992,   13.3954),  # les deux corrigées
    ("Guider",          "Nord",         9.9300,   13.9400),
    ("Poli",            "Nord",         8.4667,   13.2400),
    ("Touboro",         "Nord",         7.7667,   15.3667),
    # ── Nord-Ouest ────────────────────────────────────────
    ("Bamenda",         "Nord-Ouest",   5.9627,   10.1479),
    ("Kumbo",           "Nord-Ouest",   6.2000,   10.6667),  # lat corrigée (était 6.02)
    ("Wum",             "Nord-Ouest",   6.3833,   10.0667),  # lon corrigée (était 10.06 ✓)
    ("Mbengwi",         "Nord-Ouest",   5.9833,   10.0167),  # lon corrigée (était 10.01 ✓)
    # ── Ouest ─────────────────────────────────────────────
    ("Mbouda",          "Ouest",        5.6167,   10.2667),
    ("Foumban",         "Ouest",        5.7167,   10.9000),  # lon corrigée (était 10.09)
    ("Bafoussam",       "Ouest",        5.4667,   10.4167),
    ("Dschang",         "Ouest",        5.4500,   10.0500),  # lon corrigée (était 10.05 ✓)
    # ── Sud ───────────────────────────────────────────────
    ("Ebolowa",         "Sud",          2.9000,   11.1500),
    ("Kribi",           "Sud",          2.9500,   9.9100),
    ("Sangmelima",      "Sud",          2.9333,   11.9833),
    ("Ambam",           "Sud",          2.3833,   11.2833),
    # ── Sud-Ouest ─────────────────────────────────────────
    ("Kumba",           "Sud-Ouest",    4.6333,   9.4500),
    ("Buea",            "Sud-Ouest",    4.1667,   9.2333),
    ("Limbe",           "Sud-Ouest",    4.0167,   9.2100),   # lat corrigée (était 4.01)
    ("Mamfe",           "Sud-Ouest",    5.7667,   9.3167),
]

# ── Fonctions d'extraction ─────────────────────────────────────────────────────

def fetch_with_retry(url: str, params: dict, retries: int = MAX_RETRIES) -> dict | None:
    """GET avec retry automatique sur erreur réseau ou 429."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT_SEC)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("error"):
                    log.error(f"Erreur API : {data.get('reason')}")
                    return None
                return data
            elif resp.status_code == 429:
                wait = 60 * attempt
                log.warning(f"Rate limit (429) — attente {wait}s avant retry {attempt}/{retries}")
                time.sleep(wait)
            else:
                log.warning(f"HTTP {resp.status_code} — tentative {attempt}/{retries}")
                time.sleep(5 * attempt)
        except requests.exceptions.Timeout:
            log.warning(f"Timeout — tentative {attempt}/{retries}")
            time.sleep(10 * attempt)
        except requests.exceptions.ConnectionError as e:
            log.warning(f"Connexion erreur ({e}) — tentative {attempt}/{retries}")
            time.sleep(15 * attempt)
    return None


def fetch_daily(city: str, lat: float, lon: float,
                start: str, end: str, timezone: str = "auto") -> pd.DataFrame | None:
    """Récupère les variables daily ERA5 pour une ville et une période."""
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "daily":      DAILY_VARS,
        "timezone":   timezone,
    }
    data = fetch_with_retry(BASE_URL, params)
    if data is None:
        return None
    daily = data["daily"]
    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    df["city"] = city
    df["latitude"]  = lat
    df["longitude"] = lon
    return df


def fetch_blh_daily(city: str, lat: float, lon: float,
                    start: str, end: str, timezone: str = "auto") -> pd.DataFrame | None:
    """
    Récupère boundary_layer_height (hourly) et l'agrège en daily :
      - blh_mean  : moyenne journalière
      - blh_min   : minimum journalier (proxy accumulation nocturne)
      - blh_max   : maximum journalier (proxy dispersion diurne)
    """
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "hourly":     HOURLY_VARS,
        "timezone":   timezone,
    }
    data = fetch_with_retry(BASE_URL, params)
    if data is None:
        return None

    hourly = data["hourly"]
    df_h = pd.DataFrame(hourly)
    df_h["time"] = pd.to_datetime(df_h["time"])
    df_h["date"] = df_h["time"].dt.date

    # Agrégation daily
    df_daily = df_h.groupby("date")["boundary_layer_height"].agg(
        blh_mean="mean",
        blh_min="min",
        blh_max="max",
    ).reset_index()
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily.rename(columns={"date": "time"}, inplace=True)
    df_daily["city"] = city
    return df_daily


def extract_city(city: str, lat: float, lon: float) -> pd.DataFrame | None:
    """
    Extraction complète pour une ville : daily variables + BLH hourly → agrégé.
    Deux appels API par ville.
    """
    log.info(f"  Extraction daily    : {city} ({lat:.4f}, {lon:.4f})")
    df_daily = fetch_daily(city, lat, lon, START_DATE, END_DATE)
    if df_daily is None:
        log.error(f"  ECHEC daily : {city}")
        return None

    time.sleep(DELAY_BETWEEN_REQUESTS)

    log.info(f"  Extraction BLH      : {city}")
    df_blh = fetch_blh_daily(city, lat, lon, START_DATE, END_DATE)
    if df_blh is None:
        log.warning(f"  ECHEC BLH (ignoré) : {city} — blh_mean=NaN")
        df_daily["blh_mean"] = np.nan
        df_daily["blh_min"]  = np.nan
        df_daily["blh_max"]  = np.nan
    else:
        df_daily = df_daily.merge(df_blh[["time", "blh_mean", "blh_min", "blh_max"]],
                                  on="time", how="left")

    log.info(f"  ✅ {city} : {len(df_daily)} jours")
    return df_daily


# ── Pipeline principal ─────────────────────────────────────────────────────────

def main():
    log.info("=" * 65)
    log.info("Extraction ERA5 — Hackathon IndabaX Cameroun 2026")
    log.info(f"Période   : {START_DATE} → {END_DATE}")
    log.info(f"Villes    : {len(CITIES)}")
    log.info(f"Output    : {OUTPUT_DIR}")
    log.info("=" * 65)

    all_frames = []
    failed     = []

    for i, (city, region, lat, lon) in enumerate(CITIES, 1):
        slug     = city.lower().replace(' ', '_').replace('-', '_')
        out_file = OUTPUT_DIR / f"{slug}.parquet"
        out_csv  = OUTPUT_DIR / f"{slug}.csv"

        # Reprise si le fichier existe déjà (parquet ou csv)
        if out_file.exists():
            log.info(f"[{i:02d}/{len(CITIES)}] {city} — cache parquet")
            df = pd.read_parquet(out_file)
            all_frames.append(df)
            continue
        if out_csv.exists():
            log.info(f"[{i:02d}/{len(CITIES)}] {city} — cache csv")
            df = pd.read_csv(out_csv, parse_dates=["time"])
            all_frames.append(df)
            continue

        log.info(f"[{i:02d}/{len(CITIES)}] {city} ({region})")
        df = extract_city(city, lat, lon)

        if df is None:
            failed.append(city)
            log.error(f"  ❌ {city} ignoré")
        else:
            df["region"] = region
            # Sauvegarde parquet si pyarrow disponible, sinon CSV
            try:
                import pyarrow  # noqa
                df.to_parquet(out_file, index=False)
                log.info(f"  💾 Sauvegardé (parquet) : {out_file.name}")
            except ImportError:
                df.to_csv(out_csv, index=False)
                log.info(f"  💾 Sauvegardé (csv)     : {out_csv.name}")
            all_frames.append(df)

        # Délai inter-villes (2 appels par ville = 2× DELAY déjà inclus)
        time.sleep(DELAY_BETWEEN_REQUESTS)

    # ── Consolidation ──────────────────────────────────────────────────────────
    if all_frames:
        log.info("\nConsolidation du dataset...")
        df_all = pd.concat(all_frames, ignore_index=True)

        # Ordre cohérent avec le dataset original
        cols_order = [
            "city", "region", "latitude", "longitude", "time",
            "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max",
            "wind_direction_10m_dominant", "shortwave_radiation_sum",
            "relative_humidity_2m_max", "relative_humidity_2m_min",
            "et0_fao_evapotranspiration",
            "blh_mean", "blh_min", "blh_max",
        ]
        df_all = df_all[[c for c in cols_order if c in df_all.columns]]
        df_all = df_all.sort_values(["city", "time"]).reset_index(drop=True)

        try:
            import pyarrow  # noqa
            out_all = Path("data/era5_features_all_cities.parquet")
            df_all.to_parquet(out_all, index=False)
        except ImportError:
            out_all = Path("data/era5_features_all_cities.csv")
            df_all.to_csv(out_all, index=False)

        log.info(f"\n{'='*65}")
        log.info(f"Dataset consolidé : {out_all}")
        log.info(f"Shape             : {df_all.shape}")
        log.info(f"Villes extraites  : {df_all['city'].nunique()}/{len(CITIES)}")
        log.info(f"Période couverte  : {df_all['time'].min().date()} → {df_all['time'].max().date()}")
        log.info(f"Villes en échec   : {failed if failed else 'aucune'}")
        log.info(f"{'='*65}")

        # Aperçu rapide
        print("\n=== Aperçu du dataset ERA5 ===")
        print(df_all.head(10).to_string())
        print(f"\nShape : {df_all.shape}")
        print(f"Valeurs nulles BLH : {df_all['blh_mean'].isna().sum()}")
    else:
        log.error("Aucune donnée extraite !")

    if failed:
        log.warning(f"\nVilles en échec ({len(failed)}) : {failed}")
        log.warning("Relancer le script pour les réessayer (reprise automatique via cache).")


if __name__ == "__main__":
    main()

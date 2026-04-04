"""
Validation & Quantification d'Incertitude du Proxy PM2.5
Hackathon IndabaX Cameroun 2026

Ce script implémente trois analyses complémentaires :

Option C — Validation croisée vs produits globaux (CAMS/MERRA-2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compare notre proxy PM2.5 avec les produits de réanalyse globaux :
  · CAMS (Copernicus Atmosphere Monitoring Service) via Open-Meteo Air Quality API
  · Métriques : biais moyen, MAE, corrélation, ratio saisonnier
  · Objectif : confirmer que notre proxy est dans la même gamme de valeurs
    et capture les mêmes patterns saisonniers que les modèles de transport globaux

Option D — BLH "active hours" (approximation blh_max)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
La BLH de journée (10h–16h heure locale) est plus représentative de la
dispersion pendant les activités humaines que la BLH moyenne journalière
(qui inclut la nuit avec des inversions thermiques). blh_max ≈ BLH midi.
Comparaison blh_mean vs blh_max pour F_stagnation.

Option E — Monte Carlo (intervalles de confiance sur les paramètres)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Les paramètres physiques ont des plages documentées :
  · α (BLH) ∈ [0.4, 0.8]    — Seinfeld & Pandis 2016
  · a (rain) ∈ [0.05, 0.12]  — Berge & Jakobsen 1998
  · k (wind) ∈ [0.025, 0.05] — Pasquill-Gifford
  · H_strength ∈ [1.5, 2.5]  — Mbuh et al. 2021
  · γ (hygro) ∈ [0.002, 0.006] — Pöhlker et al. 2023
N=1000 tirages → distribution des PM2.5 → percentile 5 / 95 = IC90%
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import requests
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from io import StringIO

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Chemins ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("data")
FIG_DIR    = Path("data")

PROXY_FILE = DATA_DIR / "pm25_proxy_era5.parquet"


# ══════════════════════════════════════════════════════════════════════════════
# OPTION C — Validation vs CAMS
# ══════════════════════════════════════════════════════════════════════════════

# Villes de référence pour la validation (couvrent les 3 zones climatiques)
VALIDATION_CITIES = {
    "Yaounde":  {"lat": 3.867,  "lon": 11.517, "zone": "equatorial"},
    "Douala":   {"lat": 4.049,  "lon": 9.733,  "zone": "equatorial"},
    "Bafoussam":{"lat": 5.479,  "lon": 10.420, "zone": "transition"},
    "Bertoua":  {"lat": 4.579,  "lon": 13.686, "zone": "transition"},
    "Ngaoundere":{"lat": 7.321, "lon": 13.576, "zone": "transition"},
    "Garoua":   {"lat": 9.303,  "lon": 13.399, "zone": "sahelian"},
    "Maroua":   {"lat": 10.591, "lon": 14.315, "zone": "sahelian"},
}

CAMS_CACHE = DATA_DIR / "cams_validation.parquet"


def fetch_cams_pm25(city: str, lat: float, lon: float,
                    start: str = "2020-01-01", end: str = "2023-12-31") -> pd.DataFrame:
    """
    Récupère PM2.5 surface CAMS via Open-Meteo Air Quality API.
    Endpoint : https://air-quality-api.open-meteo.com/v1/air-quality
    Variables : pm2_5 (µg/m³, quotidien)

    Note : L'API Open-Meteo Air Quality donne accès aux données CAMS pour les dates
    historiques (archive limitée). Pour une validation complète, CAMS ADS recommandé.
    """
    base_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "hourly":     "pm2_5",
        "start_date": start,
        "end_date":   end,
        "timezone":   "Africa/Douala",
    }

    try:
        r = requests.get(base_url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        times  = data["hourly"]["time"]
        pm25   = data["hourly"]["pm2_5"]
        df = pd.DataFrame({"time": pd.to_datetime(times), "pm25_cams": pm25})
        df["city"] = city
        # Agréger horaire → journalier
        df["date"] = df["time"].dt.date
        df_daily = (df.groupby(["city", "date"])["pm25_cams"]
                    .mean()
                    .reset_index()
                    .rename(columns={"date": "time"}))
        df_daily["time"] = pd.to_datetime(df_daily["time"])
        return df_daily
    except Exception as e:
        log.warning(f"  CAMS échec pour {city} : {e}")
        return pd.DataFrame()


def option_c_validate_vs_cams(df_proxy: pd.DataFrame) -> dict:
    """
    Option C : Compare proxy PM2.5 vs CAMS pour les villes de référence.
    Télécharge CAMS si non en cache, calcule les métriques de validation.
    """
    log.info("=" * 65)
    log.info("OPTION C — Validation vs CAMS (Open-Meteo Air Quality API)")
    log.info("=" * 65)

    # Cache : évite de re-télécharger
    if CAMS_CACHE.exists():
        log.info(f"CAMS chargé depuis cache : {CAMS_CACHE}")
        df_cams = pd.read_parquet(CAMS_CACHE)
    else:
        log.info("Téléchargement CAMS pour 7 villes de référence (2020–2023)…")
        frames = []
        for city, info in VALIDATION_CITIES.items():
            log.info(f"  → {city} ({info['zone']})")
            df_city = fetch_cams_pm25(city, info["lat"], info["lon"])
            if not df_city.empty:
                df_city["climate_zone"] = info["zone"]
                frames.append(df_city)
            time.sleep(0.5)

        if not frames:
            log.error("Aucune donnée CAMS récupérée — vérifier la connexion internet")
            return {}

        df_cams = pd.concat(frames, ignore_index=True)
        df_cams.to_parquet(CAMS_CACHE, index=False)
        log.info(f"CAMS sauvegardé : {CAMS_CACHE}  {df_cams.shape}")

    # Fusion proxy ↔ CAMS
    df_val = df_proxy[df_proxy["city"].isin(VALIDATION_CITIES.keys())].copy()
    df_val["time"] = pd.to_datetime(df_val["time"])
    df_merged = df_val.merge(df_cams[["city", "time", "pm25_cams"]],
                              on=["city", "time"], how="inner")
    df_merged = df_merged.dropna(subset=["pm25_proxy", "pm25_cams"])

    if df_merged.empty:
        log.warning("Fusion proxy ↔ CAMS vide — pas de dates communes")
        return {}

    log.info(f"Paires valides proxy ↔ CAMS : {len(df_merged):,}")

    # ── Métriques globales ───────────────────────────────────────────────────
    proxy = df_merged["pm25_proxy"].values
    cams  = df_merged["pm25_cams"].values

    bias   = float(np.mean(proxy - cams))
    mae    = float(np.mean(np.abs(proxy - cams)))
    rmse   = float(np.sqrt(np.mean((proxy - cams) ** 2)))
    corr   = float(np.corrcoef(proxy, cams)[0, 1])
    # Normalized Mean Bias (NMB) — standard EPA/CAMS metric
    nmb    = float(np.sum(proxy - cams) / np.sum(cams) * 100)
    # Mean Absolute Percentage Error
    mape   = float(np.mean(np.abs((proxy - cams) / np.maximum(cams, 1))) * 100)

    print("\n" + "=" * 65)
    print("VALIDATION PROXY vs CAMS — Métriques")
    print("=" * 65)
    print(f"  Paires valides        : {len(df_merged):,}")
    print(f"  Biais moyen           : {bias:+.2f} µg/m³  (proxy - CAMS)")
    print(f"  MAE                   : {mae:.2f} µg/m³")
    print(f"  RMSE                  : {rmse:.2f} µg/m³")
    print(f"  Corrélation (r)       : {corr:.3f}")
    print(f"  NMB                   : {nmb:+.1f}%  (< ±30% = acceptable)")
    print(f"  MAPE                  : {mape:.1f}%")
    print()

    # ── Métriques par zone climatique ────────────────────────────────────────
    if "climate_zone" not in df_merged.columns:
        zone_map = {c: v["zone"] for c, v in VALIDATION_CITIES.items()}
        df_merged["climate_zone"] = df_merged["city"].map(zone_map)

    print("  Métriques par zone climatique :")
    for zone in ["equatorial", "transition", "sahelian"]:
        sub = df_merged[df_merged["climate_zone"] == zone]
        if len(sub) < 10:
            continue
        r_z = np.corrcoef(sub["pm25_proxy"], sub["pm25_cams"])[0, 1]
        b_z = (sub["pm25_proxy"] - sub["pm25_cams"]).mean()
        print(f"    {zone:<12} : r={r_z:.3f}  biais={b_z:+.2f} µg/m³  n={len(sub):,}")

    # ── Ratio saisonnier (test critique) ─────────────────────────────────────
    df_merged["month"] = pd.to_datetime(df_merged["time"]).dt.month
    dry_proxy = df_merged[df_merged["month"].isin([11,12,1,2,3])]["pm25_proxy"].mean()
    wet_proxy = df_merged[df_merged["month"].isin([5,6,7,8,9,10])]["pm25_proxy"].mean()
    dry_cams  = df_merged[df_merged["month"].isin([11,12,1,2,3])]["pm25_cams"].mean()
    wet_cams  = df_merged[df_merged["month"].isin([5,6,7,8,9,10])]["pm25_cams"].mean()
    print(f"\n  Ratio saisonnier sec/humide :")
    print(f"    Proxy : ×{dry_proxy/wet_proxy:.2f}  |  CAMS : ×{dry_cams/wet_cams:.2f}")
    print("=" * 65)

    # ── Figure de validation ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Option C — Validation Proxy PM2.5 vs CAMS", fontsize=12, fontweight="bold")

    # Scatter proxy vs CAMS
    ax = axes[0]
    sc = ax.scatter(cams, proxy, c=df_merged["month"], cmap="hsv",
                    alpha=0.3, s=5, vmin=1, vmax=12)
    lim = max(proxy.max(), cams.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x")
    ax.plot([0, lim], [0, lim * (1 + nmb/100)], "r:", lw=1, alpha=0.7, label=f"biais NMB={nmb:+.0f}%")
    ax.set_xlabel("CAMS PM2.5 (µg/m³)")
    ax.set_ylabel("Proxy PM2.5 (µg/m³)")
    ax.set_title(f"Scatter (r={corr:.3f})")
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label="Mois")

    # Saisonnalité comparative
    ax = axes[1]
    m_proxy = df_merged.groupby("month")["pm25_proxy"].mean()
    m_cams  = df_merged.groupby("month")["pm25_cams"].mean()
    months  = range(1, 13)
    ax.plot(months, [m_proxy.get(m, np.nan) for m in months], "o-",
            color="steelblue", lw=2, label="Proxy")
    ax.plot(months, [m_cams.get(m, np.nan)  for m in months], "s--",
            color="tomato",   lw=2, label="CAMS")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Saisonnalité comparative")
    ax.legend()

    # Biais par ville
    ax = axes[2]
    city_bias = df_merged.groupby("city").apply(
        lambda x: (x["pm25_proxy"] - x["pm25_cams"]).mean()
    ).sort_values()
    colors = ["tomato" if b > 0 else "steelblue" for b in city_bias.values]
    ax.barh(city_bias.index, city_bias.values, color=colors, alpha=0.8)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Biais moyen (proxy - CAMS) µg/m³")
    ax.set_title("Biais par ville")

    plt.tight_layout()
    fig_path = FIG_DIR / "validation_cams.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    log.info(f"Figure sauvegardée : {fig_path}")

    metrics = {"bias": bias, "mae": mae, "rmse": rmse, "corr": corr,
               "nmb": nmb, "mape": mape, "n_pairs": len(df_merged)}
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# OPTION D — BLH Active Hours (blh_max comme proxy de BLH diurne)
# ══════════════════════════════════════════════════════════════════════════════

def option_d_blh_active_hours(df_proxy: pd.DataFrame) -> pd.DataFrame:
    """
    Option D : Compare F_stagnation calculé avec blh_mean vs blh_max.

    Justification :
    - blh_mean inclut la nuit (inversion nocturne → BLH très bas)
    - La nuit, les émissions humaines sont faibles → la stagnation nocturne
      n'accumule pas autant de PM2.5 que la formule actuelle le suggère
    - blh_max ≈ BLH de midi (pic diurne) = dispersion pendant les activités

    Résultat attendu : pm25_blh_max légèrement inférieur à pm25_proxy actuel
    (stagnation moins surestimée), meilleure corrélation avec CAMS en saison sèche.
    """
    log.info("=" * 65)
    log.info("OPTION D — BLH Active Hours (blh_max vs blh_mean)")
    log.info("=" * 65)

    if "blh_max" not in df_proxy.columns:
        log.warning("blh_max absent du dataset — Option D ignorée")
        return df_proxy

    BLH_REF   = 1000.0
    BLH_MIN   = 150.0
    BLH_ALPHA = 0.60

    # F_stagnation avec blh_mean (actuel)
    blh_mean = df_proxy["blh_mean"].fillna(BLH_REF).clip(lower=BLH_MIN)
    F_stag_mean = ((BLH_REF / blh_mean) ** BLH_ALPHA).clip(0.3, 3.5)

    # F_stagnation avec blh_max (approximation heures actives)
    blh_max = df_proxy["blh_max"].fillna(BLH_REF).clip(lower=BLH_MIN)
    F_stag_max  = ((BLH_REF / blh_max) ** BLH_ALPHA).clip(0.3, 3.5)

    ratio = F_stag_mean.mean() / F_stag_max.mean()

    print("\n" + "=" * 65)
    print("OPTION D — Comparaison F_stagnation (mean vs max BLH)")
    print("=" * 65)
    print(f"  BLH mean — moyenne : {blh_mean.mean():.0f} m  |  F_stag moyen : {F_stag_mean.mean():.3f}")
    print(f"  BLH max  — moyenne : {blh_max.mean():.0f} m  |  F_stag moyen : {F_stag_max.mean():.3f}")
    print(f"  Ratio F_stag(mean) / F_stag(max) : {ratio:.3f}")
    print(f"  Impact sur PM2.5   : ×{1/ratio:.3f} si on passe à blh_max")
    print()

    # Analyse par saison
    df_proxy["month"] = pd.to_datetime(df_proxy["time"]).dt.month
    for season, months in [("Saison sèche (N-F-M)", [11,12,1,2,3]),
                            ("Saison humide (M-O)", [5,6,7,8,9,10])]:
        mask = df_proxy["month"].isin(months)
        r_mean = F_stag_mean[mask].mean()
        r_max  = F_stag_max[mask].mean()
        print(f"  {season} : F_stag_mean={r_mean:.3f}  F_stag_max={r_max:.3f}  ratio={r_mean/r_max:.3f}")

    print()

    # Corrélation entre les deux approches
    corr_stag = np.corrcoef(F_stag_mean, F_stag_max)[0, 1]
    print(f"  Corrélation F_stag_mean ↔ F_stag_max : r = {corr_stag:.4f}")
    print()

    # Recommandation
    if ratio > 1.15:
        print("  ► RECOMMANDATION : blh_max réduit significativement la surestimation")
        print("    de la stagnation nocturne. Considérer blh_max pour la production finale.")
    elif ratio > 1.05:
        print("  ► RECOMMANDATION : impact modéré (~5-15%). blh_max légèrement préférable")
        print("    mais blh_mean reste défendable (incertitude dans les plages documentées).")
    else:
        print("  ► blh_mean et blh_max donnent des résultats très similaires (<5%).")
        print("    Pas de changement nécessaire pour ce dataset.")

    print("=" * 65)

    # Figure comparative
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Option D — BLH Active Hours vs BLH Mean", fontsize=12, fontweight="bold")

    # Distribution des deux F_stagnation
    ax = axes[0]
    ax.hist(F_stag_mean, bins=60, alpha=0.6, color="steelblue",
            label=f"blh_mean (μ={F_stag_mean.mean():.3f})", density=True)
    ax.hist(F_stag_max,  bins=60, alpha=0.6, color="tomato",
            label=f"blh_max  (μ={F_stag_max.mean():.3f})", density=True)
    ax.set_xlabel("F_stagnation")
    ax.set_ylabel("Densité")
    ax.set_title("Distribution F_stagnation")
    ax.legend()

    # Saisonnalité
    ax = axes[1]
    months = range(1, 13)
    df_tmp = df_proxy.copy()
    df_tmp["F_stag_mean"] = F_stag_mean.values
    df_tmp["F_stag_max"]  = F_stag_max.values
    m_mean = df_tmp.groupby("month")["F_stag_mean"].mean()
    m_max  = df_tmp.groupby("month")["F_stag_max"].mean()
    ax.plot(months, [m_mean.get(m, np.nan) for m in months], "o-",
            color="steelblue", lw=2, label="F_stag(blh_mean)")
    ax.plot(months, [m_max.get(m, np.nan)  for m in months], "s--",
            color="tomato",   lw=2, label="F_stag(blh_max)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    ax.set_ylabel("F_stagnation moyen")
    ax.set_title("Saisonnalité du facteur de stagnation")
    ax.legend()

    plt.tight_layout()
    fig_path = FIG_DIR / "validation_blh_active.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    log.info(f"Figure sauvegardée : {fig_path}")

    return df_proxy


# ══════════════════════════════════════════════════════════════════════════════
# OPTION E — Monte Carlo (intervalles de confiance)
# ══════════════════════════════════════════════════════════════════════════════

# Plages de paramètres documentées dans la littérature
PARAM_RANGES = {
    "alpha":      (0.40, 0.80),   # BLH exposant — Seinfeld & Pandis 2016
    "rain_k":     (0.05, 0.12),   # lessivage humide — Berge & Jakobsen 1998
    "wind_k":     (0.025, 0.050), # dilution turbulente — Pasquill-Gifford
    "h_strength": (1.50, 2.50),   # Harmattan — Mbuh et al. 2021
    "rh_gamma":   (0.002, 0.006), # hygroscopique — Pöhlker et al. 2023
}

N_MC = 1000   # nombre de tirages Monte Carlo


def compute_proxy_fast(df: pd.DataFrame, alpha: float, rain_k: float,
                       wind_k: float, h_strength: float, rh_gamma: float,
                       blh_ref: float = 1000.0, blh_min: float = 150.0,
                       rh_threshold: float = 75.0,
                       target_mean: float = 32.5) -> np.ndarray:
    """Version vectorisée rapide du proxy pour Monte Carlo (pas de logging)."""
    BLH_ALPHA = alpha
    RAIN_K    = rain_k
    WIND_K    = wind_k
    H_STR     = h_strength
    RH_GAMMA  = rh_gamma

    blh     = df["blh_mean"].fillna(blh_ref).clip(lower=blh_min).values
    precip  = df["precipitation_sum"].fillna(0).clip(lower=0).values
    wind    = df["wind_speed_10m_max"].fillna(df["wind_speed_10m_max"].median()).clip(lower=0).values
    rh_max  = df["relative_humidity_2m_max"].fillna(70).clip(0, 100).values
    lat     = df["latitude"].values
    is_dry  = df["month"].isin([11, 12, 1, 2, 3]).astype(float).values

    lat_norm  = np.clip((lat - 3.0) / (11.0 - 3.0), 0, 1)
    is_harm   = np.zeros(len(df))
    if "wind_direction_10m_dominant" in df.columns:
        wd = df["wind_direction_10m_dominant"].fillna(0).values
        is_harm = ((wd >= 315) | (wd <= 90)).astype(float) * is_dry

    F1 = np.clip((blh_ref / np.maximum(blh, blh_min)) ** BLH_ALPHA, 0.3, 3.5)
    F2 = 1.0 / (1.0 + RAIN_K * precip)
    F3 = np.where(is_harm > 0, 1.0, np.exp(-WIND_K * wind))
    F4 = 1.0 + H_STR * is_dry * lat_norm
    F5 = np.clip(1.0 + RH_GAMMA * np.maximum(0, rh_max - rh_threshold), 1.0, 1.3)
    F5 = np.where(precip > 1.0, 1.0, F5)

    unnorm = F1 * F2 * F3 * F4 * F5
    C_base = target_mean / unnorm.mean()
    return np.clip(C_base * unnorm, 2.0, None)


def option_e_monte_carlo(df_proxy: pd.DataFrame, n_mc: int = N_MC) -> pd.DataFrame:
    """
    Option E : Monte Carlo sur les paramètres physiques.

    Pour chaque tirage aléatoire des paramètres dans leurs plages documentées,
    recalcule le proxy PM2.5 complet. Résultat : distribution de N_MC proxys
    → percentiles 5/95 = intervalle de confiance à 90%.
    """
    log.info("=" * 65)
    log.info(f"OPTION E — Monte Carlo ({n_mc} tirages)")
    log.info("=" * 65)

    # Assurer que les features temporelles sont présentes
    if "month" not in df_proxy.columns:
        df_proxy = df_proxy.copy()
        df_proxy["month"] = pd.to_datetime(df_proxy["time"]).dt.month

    rng = np.random.default_rng(seed=42)   # reproductible

    # Stockage des résultats : on garde uniquement mean / percentiles par ligne
    # (N×87240 serait trop lourd en RAM → on accumule seulement les statistiques)
    results_sum   = np.zeros(len(df_proxy))
    results_sum2  = np.zeros(len(df_proxy))
    percentile_store = np.zeros((len(df_proxy), n_mc), dtype=np.float32)

    log.info(f"Lancement des {n_mc} tirages Monte Carlo…")
    for i in range(n_mc):
        alpha      = rng.uniform(*PARAM_RANGES["alpha"])
        rain_k     = rng.uniform(*PARAM_RANGES["rain_k"])
        wind_k     = rng.uniform(*PARAM_RANGES["wind_k"])
        h_strength = rng.uniform(*PARAM_RANGES["h_strength"])
        rh_gamma   = rng.uniform(*PARAM_RANGES["rh_gamma"])

        pm25_i = compute_proxy_fast(df_proxy, alpha, rain_k, wind_k,
                                    h_strength, rh_gamma)
        percentile_store[:, i] = pm25_i.astype(np.float32)
        results_sum  += pm25_i
        results_sum2 += pm25_i ** 2

        if (i + 1) % 200 == 0:
            log.info(f"  {i+1}/{n_mc} tirages…")

    # Statistiques finales
    mc_mean = results_sum / n_mc
    mc_var  = results_sum2 / n_mc - mc_mean ** 2
    mc_std  = np.sqrt(np.maximum(mc_var, 0))
    mc_p05  = np.percentile(percentile_store, 5,  axis=1)
    mc_p95  = np.percentile(percentile_store, 95, axis=1)
    mc_p25  = np.percentile(percentile_store, 25, axis=1)
    mc_p75  = np.percentile(percentile_store, 75, axis=1)

    df_proxy = df_proxy.copy()
    df_proxy["pm25_mc_mean"] = mc_mean.round(3)
    df_proxy["pm25_mc_std"]  = mc_std.round(3)
    df_proxy["pm25_mc_p05"]  = mc_p05.round(3)
    df_proxy["pm25_mc_p95"]  = mc_p95.round(3)
    df_proxy["pm25_mc_p25"]  = mc_p25.round(3)
    df_proxy["pm25_mc_p75"]  = mc_p75.round(3)

    # ── Rapport ──────────────────────────────────────────────────────────────
    ic_width = (mc_p95 - mc_p05).mean()
    cv_mean  = (mc_std / mc_mean * 100).mean()

    print("\n" + "=" * 65)
    print("OPTION E — Monte Carlo : Résultats d'incertitude")
    print("=" * 65)
    print(f"  Tirages N               : {n_mc}")
    print(f"  PM2.5 proxy nominal     : {df_proxy['pm25_proxy'].mean():.2f} µg/m³")
    print(f"  PM2.5 MC mean           : {mc_mean.mean():.2f} µg/m³")
    print(f"  PM2.5 MC std (moyenne)  : {mc_std.mean():.2f} µg/m³")
    print(f"  IC90% largeur moyenne   : {ic_width:.2f} µg/m³")
    print(f"  CV moyen (std/mean)     : {cv_mean:.1f}%")
    print()
    print(f"  Paramètre le plus sensible (analyse de sensibilité) :")

    # Sensibilité : corrélation de rang entre chaque paramètre et le mean MC
    # (via re-tirage de 100 pour tester)
    rng2 = np.random.default_rng(seed=0)
    n_sens = 200
    param_names = list(PARAM_RANGES.keys())
    param_vals  = {p: [] for p in param_names}
    pm25_means  = []
    for _ in range(n_sens):
        params = {p: rng2.uniform(*PARAM_RANGES[p]) for p in param_names}
        for p in param_names:
            param_vals[p].append(params[p])
        pm25_i = compute_proxy_fast(df_proxy, **params)
        pm25_means.append(pm25_i.mean())

    pm25_arr = np.array(pm25_means)
    sensitivities = {}
    for p in param_names:
        arr = np.array(param_vals[p])
        corr = np.corrcoef(arr, pm25_arr)[0, 1]
        sensitivities[p] = abs(corr)
        sign = "↑" if np.corrcoef(arr, pm25_arr)[0, 1] > 0 else "↓"
        print(f"    {p:<12} : |r| = {abs(corr):.3f}  {sign}")

    most_sensitive = max(sensitivities, key=sensitivities.get)
    print(f"\n  → Paramètre le plus influent : {most_sensitive} (|r|={sensitivities[most_sensitive]:.3f})")
    print("=" * 65)

    # ── Figure incertitude ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Option E — Monte Carlo ({n_mc} tirages) : Incertitude PM2.5",
                 fontsize=12, fontweight="bold")

    # A. Distribution de la moyenne nationale MC
    ax = axes[0, 0]
    national_means = percentile_store.mean(axis=0)
    ax.hist(national_means, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(national_means.mean(), color="red", lw=2,
               label=f"Moyenne = {national_means.mean():.1f} µg/m³")
    ax.axvline(np.percentile(national_means, 5),  color="orange", lw=1.5, ls="--",
               label=f"P5={np.percentile(national_means,5):.1f}")
    ax.axvline(np.percentile(national_means, 95), color="orange", lw=1.5, ls="--",
               label=f"P95={np.percentile(national_means,95):.1f}")
    ax.set_xlabel("PM2.5 national moyen (µg/m³)")
    ax.set_ylabel("Fréquence")
    ax.set_title("A. Distribution de la moyenne nationale MC")
    ax.legend(fontsize=8)

    # B. Incertitude saisonnière
    ax = axes[0, 1]
    df_proxy["month"] = pd.to_datetime(df_proxy["time"]).dt.month
    months = range(1, 13)
    month_p05  = [df_proxy[df_proxy["month"]==m]["pm25_mc_p05"].mean()  for m in months]
    month_p95  = [df_proxy[df_proxy["month"]==m]["pm25_mc_p95"].mean()  for m in months]
    month_mean = [df_proxy[df_proxy["month"]==m]["pm25_mc_mean"].mean() for m in months]
    ax.fill_between(list(months), month_p05, month_p95, alpha=0.3,
                    color="steelblue", label="IC90%")
    ax.plot(list(months), month_mean, "o-", color="steelblue", lw=2, label="MC mean")
    ax.plot(list(months),
            [df_proxy[df_proxy["month"]==m]["pm25_proxy"].mean() for m in months],
            "k--", lw=1.5, label="Proxy nominal")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("B. Incertitude saisonnière (IC90%)")
    ax.legend(fontsize=8)

    # C. Sensibilité aux paramètres
    ax = axes[1, 0]
    sorted_params = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
    param_labels  = [p for p, _ in sorted_params]
    param_corrs   = [v for _, v in sorted_params]
    colors_s = ["tomato" if v > 0.5 else "steelblue" for v in param_corrs]
    ax.barh(param_labels, param_corrs, color=colors_s, alpha=0.8)
    ax.axvline(0.3, color="orange", ls="--", lw=1, label="seuil modéré")
    ax.set_xlabel("|Corrélation de rang|")
    ax.set_title("C. Analyse de sensibilité (influence sur PM2.5 moyen)")
    ax.legend(fontsize=8)

    # D. Carte de l'incertitude par ville
    ax = axes[1, 1]
    city_uncertainty = df_proxy.groupby("city").agg(
        pm25_proxy=("pm25_proxy", "mean"),
        pm25_p05=("pm25_mc_p05", "mean"),
        pm25_p95=("pm25_mc_p95", "mean"),
        latitude=("latitude", "first"),
    ).sort_values("latitude")
    ic_city = city_uncertainty["pm25_p95"] - city_uncertainty["pm25_p05"]
    ax.barh(range(len(ic_city)), ic_city.values, color="steelblue", alpha=0.7)
    ax.set_yticks(range(len(ic_city)))
    ax.set_yticklabels(city_uncertainty.index, fontsize=6)
    ax.set_xlabel("Largeur IC90% (µg/m³)")
    ax.set_title("D. Incertitude par ville (IC90%)")

    plt.tight_layout()
    fig_path = FIG_DIR / "validation_monte_carlo.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    log.info(f"Figure sauvegardée : {fig_path}")

    # Sauvegarde avec intervalles de confiance
    mc_out = DATA_DIR / "pm25_with_uncertainty.parquet"
    cols_to_save = ["city", "time", "latitude", "longitude", "month",
                    "pm25_proxy", "pm25_mc_mean", "pm25_mc_std",
                    "pm25_mc_p05", "pm25_mc_p25", "pm25_mc_p75", "pm25_mc_p95"]
    cols_to_save = [c for c in cols_to_save if c in df_proxy.columns]
    df_proxy[cols_to_save].to_parquet(mc_out, index=False)
    log.info(f"Proxy avec incertitude MC sauvegardé : {mc_out}  {df_proxy[cols_to_save].shape}")

    return df_proxy


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 65)
    log.info("Validation & Incertitude PM2.5 — Hackathon IndabaX 2026")
    log.info("=" * 65)

    # Chargement du proxy PM2.5 (généré par 02_build_pm25_target.py)
    if not PROXY_FILE.exists():
        raise FileNotFoundError(
            f"{PROXY_FILE} introuvable.\n"
            "Lancez d'abord : python notebooks/02_build_pm25_target.py"
        )

    df_proxy = pd.read_parquet(PROXY_FILE)
    df_proxy["time"]  = pd.to_datetime(df_proxy["time"])
    df_proxy["month"] = df_proxy["time"].dt.month
    log.info(f"Proxy chargé : {df_proxy.shape} | {df_proxy['city'].nunique()} villes")

    # ── Option C : Validation vs CAMS ───────────────────────────────────────
    log.info("\n>>> Option C — Validation vs CAMS")
    metrics_c = option_c_validate_vs_cams(df_proxy)

    # ── Option D : BLH active hours ──────────────────────────────────────────
    log.info("\n>>> Option D — BLH Active Hours")
    df_proxy = option_d_blh_active_hours(df_proxy)

    # ── Option E : Monte Carlo ───────────────────────────────────────────────
    log.info("\n>>> Option E — Monte Carlo (1000 tirages)")
    df_proxy = option_e_monte_carlo(df_proxy, n_mc=N_MC)

    log.info("=" * 65)
    log.info("Analyses terminées.")
    log.info(f"  Figures : {FIG_DIR}/validation_*.png")
    log.info(f"  Données : {DATA_DIR}/pm25_with_uncertainty.parquet")
    log.info("=" * 65)


if __name__ == "__main__":
    main()

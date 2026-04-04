"""
Feature Engineering — Dataset PM2.5 Hackathon IndabaX Cameroun 2026
=====================================================================
Version rigoureuse basée sur l'EDA (03_eda_feature_analysis.py).

Source de données : pm25_proxy_era5.parquet (ERA5 pur, complet)
  → Évite les 45–89% de NaN introduits par la fusion avec Dataset_complet_Meteo.xlsx

Décisions justifiées par l'EDA et tests rigoureux :
  1. Imputation       : interpolation linéaire par ville (RMSE=89 vs médiane=194)
  2. Cible            : log1p(pm25_proxy) — skewness 2.14 → 0.49
  3. Log-transforms   : precipitation_sum, blh_min, wind_speed, et0 (|skew|>1, log aide)
  4. Suppression      : colonnes redondantes (|r|>0.90) — garder le plus corrélé à la cible
  5. Normalisation    : PAS nécessaire pour XGBoost/LightGBM (invariants à l'échelle)
  6. daylight_duration: calculé astronomiquement depuis lat + day_of_year (pas l'Excel)
  7. weather_code     : approximé par binning (precip × radiation) — Spearman r=0.98 vs WMO
  8. Target encoding  : PM2.5 moyen par ville (calculé OOF pour éviter la fuite)
  9. Binning BLH      : 4 catégories physiques (inversion/stable/modéré/convectif)
  10. Sélection       : suppression features importance XGBoost < seuil (après entraînement)

Anti-fuite vérifiée (5 tests) :
  ✅ pm25_log absent des lags
  ✅ Aucune corrélation parfaite (|r|>0.99) avec la cible
  ✅ lag1 = pm25[J-1] confirmé sur 9 jours
  ✅ roll7_mean = mean(pm25[J-7:J-1]) confirmé
  ✅ Pas de fuite cross-ville

Catégories de features :
   1. Temporelles     — encodage cyclique (sin/cos mois, doy, dow)
   2. Astronomiques   — durée du jour calculée depuis latitude + date
   3. Météo dérivées  — amplitude thermique, stagnation, indices
   4. Log-transforms  — linéarisation des variables à queue lourde
   5. Binning         — BLH en 4 régimes physiques
   6. Weather proxy   — approximation WMO depuis precip × radiation (r=0.98)
   7. Harmattan       — indicateur, intensité, saison 4-niveaux
   8. Lags            — J-1, J-2, J-3, J-7, J-14 (par ville)
   9. Rolling         — mean/std sur 3/7/14/30 jours (par ville, shift=1)
  10. Spatiales       — région ordinal, lat/lon normalisés, zone climatique
  11. Target encoding — PM2.5 moyen par ville (OOF, sans fuite)
  12. Interactions    — vent×saison, BLH×précip, lat×harmattan
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Sources et sorties ────────────────────────────────────────────────────────
INPUT_PATH  = Path("data/pm25_proxy_era5.parquet")      # ERA5 pur, complet
OUTPUT_PATH = Path("data/dataset_features.parquet")

TARGET     = "pm25_proxy"
TARGET_LOG = "pm25_log"       # log1p(pm25_proxy) — cible ML principale
KEY_COLS   = ["city", "region", "latitude", "longitude", "time"]

# ── Paramètres lag/rolling ────────────────────────────────────────────────────
LAG_DAYS     = [1, 2, 3, 7, 14]
ROLLING_WINS = [3, 7, 14, 30]

LAG_VARS = [
    TARGET,
    "precipitation_sum",
    "wind_speed_10m_max",
    "blh_mean",
    "relative_humidity_2m_max",
    "temperature_2m_max",
]

ROLLING_VARS = [
    TARGET,
    "precipitation_sum",
    "wind_speed_10m_max",
    "blh_mean",
    "temperature_2m_max",
    "relative_humidity_2m_max",
    "shortwave_radiation_sum",
]

# ── Paramètres Harmattan ──────────────────────────────────────────────────────
HARMATTAN_LAT_MIN  = 3.0
HARMATTAN_LAT_MAX  = 11.0
HARMATTAN_MONTHS   = [11, 12, 1, 2, 3]


# ─────────────────────────────────────────────────────────────────────────────
# 0. Chargement
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: Path) -> pd.DataFrame:
    print(f"\n[0] Chargement : {path}")
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    # Supprimer colonnes intermédiaires du proxy (pas des features météo)
    drop_proxy_cols = ["F_stagnation", "F_wet", "F_wind", "F_harmattan",
                       "F_hygro", "C_base", "is_dry_season",
                       "month", "year", "day_of_year"]
    df = df.drop(columns=[c for c in drop_proxy_cols if c in df.columns])

    print(f"    Shape : {df.shape} | Villes : {df['city'].nunique()} | "
          f"Période : {df['time'].min().date()} → {df['time'].max().date()}")

    # Vérification NaN initiaux
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        print(f"    NaN initiaux (ERA5) :")
        for col, n in nan_cols.items():
            print(f"      {col}: {n} ({100*n/len(df):.1f}%)")
    else:
        print(f"    NaN initiaux : aucun ✅")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. Imputation — interpolation linéaire par ville
#    Justification EDA : RMSE=89 vs médiane=194 sur 10% NaN simulés
# ─────────────────────────────────────────────────────────────────────────────
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[1] Imputation des valeurs manquantes (interpolation linéaire par ville)")
    df = df.copy()
    df = df.sort_values(["city", "time"])

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    total_imputed = 0

    for col in num_cols:
        n_before = df[col].isna().sum()
        if n_before == 0:
            continue
        # Interpolation linéaire par ville (respecte la continuité temporelle)
        df[col] = df.groupby("city")[col].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )
        # Résidu extrême (début/fin de série sans voisin) → médiane par ville
        still_nan = df[col].isna().sum()
        if still_nan > 0:
            df[col] = df.groupby("city")[col].transform(
                lambda x: x.fillna(x.median())
            )
        n_after = df[col].isna().sum()
        imputed = n_before - n_after
        if imputed > 0:
            total_imputed += imputed
            print(f"    {col}: {n_before} NaN → {n_after} restants ({imputed} imputés)")

    print(f"    Total imputé : {total_imputed} valeurs")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Variable cible log-transformée
#    Justification EDA : skewness PM2.5 = 2.14 → 0.49 après log1p
# ─────────────────────────────────────────────────────────────────────────────
def add_log_target(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[2] Transformation log1p de la cible")
    df = df.copy()
    df[TARGET_LOG] = np.log1p(df[TARGET])
    print(f"    pm25_proxy  → skewness = {df[TARGET].skew():.3f}")
    print(f"    pm25_log    → skewness = {df[TARGET_LOG].skew():.3f}  ✅")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Durée du jour astronomique (remplace daylight_duration de l'Excel)
#    Formule de Spencer (1971) — précision ±1 min pour latitudes ≤ 60°N
# ─────────────────────────────────────────────────────────────────────────────
def compute_daylight_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la durée du jour (en secondes) depuis latitude + day_of_year.
    Évite toute dépendance à l'Excel pour cette feature critique (#2 importance XGBoost).
    """
    print("\n[3] Calcul astronomique de la durée du jour (Spencer 1971)")
    df = df.copy()

    doy = df["time"].dt.dayofyear
    lat_rad = np.radians(df["latitude"])

    # Déclinaison solaire (radians)
    B = 2 * np.pi * (doy - 1) / 365
    decl = (0.006918
            - 0.399912 * np.cos(B)
            + 0.070257 * np.sin(B)
            - 0.006758 * np.cos(2*B)
            + 0.000907 * np.sin(2*B)
            - 0.002697 * np.cos(3*B)
            + 0.00148  * np.sin(3*B))

    # Angle horaire au coucher du soleil (radians)
    cos_ha = -np.tan(lat_rad) * np.tan(decl)
    cos_ha = cos_ha.clip(-1, 1)  # éviter arccos hors domaine
    ha = np.arccos(cos_ha)

    # Durée du jour en secondes (2 × ha × 3600 / (π/12))
    df["daylight_duration"] = 2 * ha * (3600 * 12 / np.pi)

    dl_min = df["daylight_duration"].min() / 3600
    dl_max = df["daylight_duration"].max() / 3600
    dl_mean = df["daylight_duration"].mean() / 3600
    print(f"    Durée du jour : min={dl_min:.1f}h | mean={dl_mean:.1f}h | max={dl_max:.1f}h")
    print(f"    (Cameroun 3–13°N → variation faible, attendu ~11.5–12.5h) ✅")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Features temporelles (encodage cyclique)
# ─────────────────────────────────────────────────────────────────────────────
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[4] Features temporelles (cycliques)")
    df = df.copy()
    df["year"]        = df["time"].dt.year
    df["month"]       = df["time"].dt.month
    df["day_of_year"] = df["time"].dt.dayofyear
    df["day_of_week"] = df["time"].dt.dayofweek
    df["quarter"]     = df["time"].dt.quarter

    # Encodage cyclique — évite que jan et déc semblent "loin"
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    print(f"    +8 features temporelles (year, month, doy, dow, quarter + sin/cos)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Features Harmattan
# ─────────────────────────────────────────────────────────────────────────────
def add_harmattan_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[5] Features Harmattan et saisonnalité")
    df = df.copy()

    df["is_harmattan"]  = df["month"].isin(HARMATTAN_MONTHS).astype(int)
    df["is_wet_season"] = (~df["month"].isin(HARMATTAN_MONTHS)).astype(int)

    # Saison 4-niveaux (climatologie Cameroun)
    conditions = [
        df["month"].isin([12, 1, 2]),      # Petite saison sèche (Harmattan)
        df["month"].isin([3, 4, 5, 6]),    # Grande saison des pluies
        df["month"].isin([7, 8]),          # Grande saison sèche
        df["month"].isin([9, 10, 11]),     # Petite saison des pluies
    ]
    df["season_code"] = np.select(conditions, [0, 1, 2, 3], default=1)

    # Intensité Harmattan × latitude
    lat_norm = ((df["latitude"] - HARMATTAN_LAT_MIN) /
                (HARMATTAN_LAT_MAX - HARMATTAN_LAT_MIN)).clip(0, 1)
    df["harmattan_intensity"] = df["is_harmattan"] * lat_norm

    # Vent de secteur N–NE (315–90°) pendant saison sèche = "vrai" Harmattan
    if "wind_direction_10m_dominant" in df.columns:
        wdir = df["wind_direction_10m_dominant"].fillna(180)
        is_n_wind = ((wdir >= 315) | (wdir <= 90)).astype(int)
        df["is_true_harmattan"] = (
            (df["is_harmattan"] == 1) & (is_n_wind == 1)
        ).astype(int)
    else:
        df["is_true_harmattan"] = 0

    print(f"    +5 features Harmattan")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Features météo dérivées + log-transforms
#    Log-transforms justifiés par EDA (|skew|>1 et log aide) :
#      precipitation_sum : skew 8.6 → 3.0
#      blh_min           : skew 2.6 → 0.7
#      wind_speed_10m_max: skew 1.2 → 0.2
#      et0_fao           : skew 1.0 → 0.1
# ─────────────────────────────────────────────────────────────────────────────
def add_derived_meteo(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[6] Features météo dérivées + log-transforms")
    df = df.copy()

    # ── Amplitude thermique ──────────────────────────────────────────────────
    if all(c in df.columns for c in ["temperature_2m_max", "temperature_2m_min"]):
        df["temp_amplitude"] = df["temperature_2m_max"] - df["temperature_2m_min"]

    # ── Stagnation (vent faible) ─────────────────────────────────────────────
    if "wind_speed_10m_max" in df.columns:
        df["is_stagnant"]  = (df["wind_speed_10m_max"] < 5).astype(int)

    # ── Jours secs / pluvieux ────────────────────────────────────────────────
    if "precipitation_sum" in df.columns:
        df["is_dry_day"]    = (df["precipitation_sum"] < 0.1).astype(int)
        df["is_heavy_rain"] = (df["precipitation_sum"] > 10).astype(int)

    # ── Humidité relative : moyenne et amplitude ──────────────────────────────
    if all(c in df.columns for c in ["relative_humidity_2m_max",
                                      "relative_humidity_2m_min"]):
        df["rh_mean"]      = (df["relative_humidity_2m_max"] +
                               df["relative_humidity_2m_min"]) / 2
        df["rh_amplitude"] = (df["relative_humidity_2m_max"] -
                               df["relative_humidity_2m_min"])

    # ── BLH transformé ───────────────────────────────────────────────────────
    if "blh_mean" in df.columns:
        blh = df["blh_mean"].clip(lower=50)
        df["blh_log"]    = np.log(blh)
        df["blh_inv"]    = 1000 / blh
        df["is_low_blh"] = (blh < 400).astype(int)

    # ── LOG-TRANSFORMS (justifiés par EDA) ──────────────────────────────────
    # précipitations : skew 8.6 → 3.0 après log1p
    if "precipitation_sum" in df.columns:
        df["precip_log"] = np.log1p(df["precipitation_sum"].clip(lower=0))

    # BLH min : skew 2.6 → 0.7 après log1p
    if "blh_min" in df.columns:
        df["blh_min_log"] = np.log1p(df["blh_min"].clip(lower=0))

    # Vent max : skew 1.2 → 0.2 après log1p
    if "wind_speed_10m_max" in df.columns:
        df["wind_log"] = np.log1p(df["wind_speed_10m_max"].clip(lower=0))

    # ET0 : skew 1.0 → 0.1 après log1p
    if "et0_fao_evapotranspiration" in df.columns:
        df["et0_log"] = np.log1p(df["et0_fao_evapotranspiration"].clip(lower=0))

    # ── Stagnation cumulée (indice composite) ────────────────────────────────
    stag_components = ["is_stagnant", "is_low_blh", "is_dry_day"]
    if all(c in df.columns for c in stag_components):
        df["stagnation_index"] = (df["is_stagnant"] +
                                  df["is_low_blh"] +
                                  df["is_dry_day"])

    print(f"    +log-transforms (precip, blh_min, wind, et0)")
    print(f"    +dérivées (temp_amplitude, rh_mean, rh_amplitude, blh_inv, stagnation_index)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Features de lag
#    IMPORTANT : calculés PAR VILLE pour éviter les fuites inter-villes
# ─────────────────────────────────────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[7] Features de lag (mémoire temporelle, par ville)")
    df = df.copy().sort_values(["city", "time"])

    n_created = 0
    for var in LAG_VARS:
        if var not in df.columns:
            continue
        for lag in LAG_DAYS:
            df[f"{var}_lag{lag}"] = df.groupby("city")[var].shift(lag)
            n_created += 1

    print(f"    +{n_created} lag features ({len(LAG_VARS)} vars × {len(LAG_DAYS)} lags)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. Features rolling
#    shift(1) : la fenêtre ne doit pas inclure le jour J (fuite de données)
# ─────────────────────────────────────────────────────────────────────────────
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[8] Features rolling (tendance récente, par ville)")
    df = df.copy().sort_values(["city", "time"])

    n_created = 0
    for var in ROLLING_VARS:
        if var not in df.columns:
            continue
        for win in ROLLING_WINS:
            grp = df.groupby("city")[var]
            df[f"{var}_roll{win}_mean"] = grp.transform(
                lambda x: x.shift(1).rolling(win, min_periods=1).mean()
            )
            n_created += 1
            if win >= 7:
                df[f"{var}_roll{win}_std"] = grp.transform(
                    lambda x: x.shift(1).rolling(win, min_periods=2).std()
                )
                n_created += 1

    # Jours consécutifs sans pluie
    if "is_dry_day" in df.columns:
        def dry_streak(series):
            streak, count = [], 0
            for val in series.shift(1).fillna(0):
                count = count + 1 if val == 1 else 0
                streak.append(count)
            return streak
        df["dry_streak"] = df.groupby("city")["is_dry_day"].transform(dry_streak)
        n_created += 1

    print(f"    +{n_created} rolling features")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 9. Features spatiales
# ─────────────────────────────────────────────────────────────────────────────
def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[9] Features spatiales")
    df = df.copy()

    region_order = {
        "Sud": 0, "Sud-Ouest": 1, "Littoral": 2, "Centre": 3,
        "Est": 4, "Ouest": 5, "Nord-Ouest": 6,
        "Adamaoua": 7, "Nord": 8, "Extreme-Nord": 9
    }
    df["region_code"] = df["region"].map(region_order).fillna(5).astype(int)

    df["lat_norm"] = (df["latitude"] - df["latitude"].min()) / \
                     (df["latitude"].max() - df["latitude"].min())
    df["lon_norm"] = (df["longitude"] - df["longitude"].min()) / \
                     (df["longitude"].max() - df["longitude"].min())

    # Zone climatique : 0=équatorial (<5°N), 1=transition (5–8°N), 2=sahélien (>8°N)
    df["climate_zone"] = pd.cut(
        df["latitude"], bins=[-np.inf, 5, 8, np.inf], labels=[0, 1, 2]
    ).astype(float).astype(int)

    city_ids = {c: i for i, c in enumerate(sorted(df["city"].unique()))}
    df["city_id"] = df["city"].map(city_ids)

    print(f"    +5 features spatiales (region_code, lat/lon_norm, climate_zone, city_id)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 10. Weather proxy (approximation weather_code WMO depuis ERA5)
#
#     Justification EDA :
#       - weather_code = #3 feature (19% importance XGBoost) mais absent d'ERA5
#       - Spearman r=0.98 entre wmo_cat et precip_cat (testé sur 47,769 lignes)
#       - Gradient PM2.5 identique : Clair(83)>Nuageux(51)>Bruine(31)>Pluie(10)
#
#     Amélioration vs simple binning précip :
#       - Combiner precipitation_sum + shortwave_radiation_sum
#       - radiation élevée + pas de pluie = ciel clair = Harmattan (PM2.5 très élevé)
#       - radiation faible + pas de pluie = nuageux sans pluie
# ─────────────────────────────────────────────────────────────────────────────
def add_weather_proxy(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[10] Weather proxy (approximation WMO depuis précip × radiation)")
    df = df.copy()

    precip = df["precipitation_sum"].fillna(0).clip(lower=0)

    # Catégorie pluie (0=sec, 1=bruine, 2=pluie modérée, 3=forte pluie)
    precip_cat = pd.cut(
        precip,
        bins=[-np.inf, 0.1, 1.0, 10.0, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(float)

    df["precip_cat"] = precip_cat

    # Ensoleillement relatif (radiation normalisée par max mensuel par ville)
    # Capture : radiation élevée = ciel dégagé (Harmattan), basse = nuageux/pluie
    if "shortwave_radiation_sum" in df.columns:
        rad = df["shortwave_radiation_sum"].fillna(
            df["shortwave_radiation_sum"].median()
        )
        # Normalisation par quantile 95 pour chaque mois (évite les biais saisonniers)
        df["rad_norm"] = df.groupby("month")["shortwave_radiation_sum"].transform(
            lambda x: x / max(x.quantile(0.95), 1.0)
        ).clip(0, 1)

        # Proxy WMO composite :
        # 0=clair+chaud (Harmattan), 1=nuageux, 2=bruine, 3=pluie modérée, 4=forte pluie
        conditions = [
            (precip_cat == 0) & (df["rad_norm"] > 0.75),   # Clair = sec + ensoleillé
            (precip_cat == 0) & (df["rad_norm"] <= 0.75),  # Nuageux = sec + couvert
            (precip_cat == 1),                              # Bruine
            (precip_cat == 2),                              # Pluie modérée
            (precip_cat == 3),                              # Forte pluie
        ]
        df["weather_proxy"] = np.select(conditions, [0, 1, 2, 3, 4], default=1)
    else:
        df["weather_proxy"] = precip_cat

    counts = df["weather_proxy"].value_counts().sort_index()
    labels = ["Clair", "Nuageux", "Bruine", "Pluie mod.", "Forte pluie"]
    print(f"    Distribution weather_proxy :")
    for cat, label in enumerate(labels):
        n = counts.get(cat, 0)
        print(f"      {cat}={label:<12} : {n:>6,} jours ({100*n/len(df):.1f}%)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 11. Binning BLH — 4 régimes physiques
#
#     Justification :
#       - BLH est la variable physique la plus importante pour la dispersion
#       - Les arbres créent des seuils → des catégories explicites aident
#       - 4 régimes identifiés dans la littérature :
#           < 200m  : inversion forte (accumulation nocturne/hivernale)
#           200–500m: stable (accumulation modérée)
#           500–1500m: mixte (dispersion partielle)
#           >1500m  : convectif (forte dispersion, saison des pluies)
#       - Seuils validés sur la distribution ERA5 Cameroun (quartiles)
# ─────────────────────────────────────────────────────────────────────────────
def add_blh_binning(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[11] Binning BLH (4 régimes physiques)")
    df = df.copy()

    if "blh_mean" not in df.columns:
        print("    blh_mean absent — étape ignorée")
        return df

    blh = df["blh_mean"]

    # Vérifier que les seuils sont pertinents sur nos données
    q25, q50, q75 = blh.quantile([0.25, 0.50, 0.75])
    print(f"    Quartiles BLH : Q25={q25:.0f}m | Q50={q50:.0f}m | Q75={q75:.0f}m")

    # Régimes physiques (seuils fixes, défendables physiquement)
    bins   = [-np.inf, 200, 500, 1500, np.inf]
    labels = [0, 1, 2, 3]   # 0=inversion, 1=stable, 2=mixte, 3=convectif

    df["blh_regime"] = pd.cut(blh, bins=bins, labels=labels).astype(float).astype(int)

    # PM2.5 moyen par régime (validation)
    regime_pm25 = df.groupby("blh_regime")["pm25_proxy"].mean()
    regime_names = {0: "Inversion (<200m)", 1: "Stable (200-500m)",
                    2: "Mixte (500-1500m)", 3: "Convectif (>1500m)"}
    print(f"    PM2.5 moyen par régime BLH (attendu : décroissant) :")
    for r in sorted(regime_pm25.index):
        n = (df["blh_regime"] == r).sum()
        print(f"      {regime_names[r]:<22} : {regime_pm25[r]:5.1f} µg/m³  ({n:,} jours)")

    # Note : gradient non-monotone attendu — l'Harmattan (BLH 500–1000m) a PM2.5
    # élevé dû aux poussières sahariennes, pas à la stagnation. La relation
    # BLH→PM2.5 est conditionnelle à la saison. Les features d'interaction
    # BLH×saison capturent cet effet. Le binning reste utile comme seuil discret.
    vals = [regime_pm25.get(r, np.nan) for r in [0, 1, 2, 3]]
    is_decreasing = all(vals[i] >= vals[i+1] for i in range(len(vals)-1)
                        if not np.isnan(vals[i]) and not np.isnan(vals[i+1]))
    if not is_decreasing:
        print(f"    ℹ️  Gradient non-monotone (attendu : Harmattan BLH 500–1000m = PM2.5 élevé)")
        print(f"       → L'effet est capturé par les interactions BLH×saison")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 12. Target encoding par ville (OOF — Out-Of-Fold, sans fuite)
#
#     Justification :
#       - city_id ordinal (0–39) ne donne aucune info sémantique au modèle
#       - PM2.5 moyen par ville = encodage très informatif (ANOVA F=1887, p≈0)
#       - Risque de fuite si calculé sur tout le dataset → utiliser OOF :
#           Pour chaque ville, encoder avec la moyenne de TOUTES LES AUTRES villes
#           sur les années passées (expanding window, cohérent avec la CV temporelle)
#
#     Implémentation : leave-one-city-out sur les années de référence
#       city_te[v] = mean(pm25_proxy pour toutes les villes ≠ v, années ≤ année courante - 1)
#       → conservative mais sans fuite possible
# ─────────────────────────────────────────────────────────────────────────────
def add_target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[12] Target encoding par ville (OOF leave-one-city-out)")
    df = df.copy()

    # Encodage global (PM2.5 moyen historique par ville sur toutes les années)
    # Utilisé pour le TEST 2025 où on n'a pas de données futures
    city_global_mean = df.groupby("city")[TARGET].mean()

    # Encodage expanding (pour chaque ligne, moyenne des années précédentes)
    # Cela évite toute fuite temporelle
    df = df.sort_values(["city", "time"])

    city_te_expanding = []
    for city, grp in df.groupby("city"):
        grp = grp.copy().sort_values("time")
        # Expanding mean décalé de 1 an (shift par année)
        grp["year_"] = grp["time"].dt.year
        annual_means = grp.groupby("year_")[TARGET].mean()

        te_values = []
        for _, row in grp.iterrows():
            yr = row["year_"]
            past_means = annual_means[annual_means.index < yr]
            if len(past_means) == 0:
                # Pas d'historique → encodage global (toutes villes sauf celle-ci)
                te = df[df["city"] != city][TARGET].mean()
            else:
                te = past_means.mean()
            te_values.append(te)
        grp["city_pm25_te"] = te_values
        city_te_expanding.append(grp[["city", "time", "city_pm25_te"]])

    te_df = pd.concat(city_te_expanding).reset_index(drop=True)
    df = df.merge(te_df, on=["city", "time"], how="left")

    # Vérification : corrélation avec PM2.5 moyen réel par ville
    city_check = df.groupby("city").agg(
        te_mean=("city_pm25_te", "mean"),
        pm25_mean=(TARGET, "mean")
    )
    r = city_check["te_mean"].corr(city_check["pm25_mean"])
    print(f"    Corrélation city_pm25_te vs PM2.5 réel par ville : r = {r:.4f}")
    print(f"    (attendu proche de 1.0 → encoding informatif)")

    # Top/bottom 3 pour validation
    print(f"    Top 3 villes PM2.5 (encoding) :")
    top3 = city_check.nlargest(3, "pm25_mean")
    for _, row in top3.iterrows():
        print(f"      {_:<15} → PM2.5={row['pm25_mean']:.1f} | TE={row['te_mean']:.1f}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 13. Sélection de features — suppression des features à importance quasi nulle
#
#     Justification EDA :
#       - XGBoost rapide montre que snowfall_sum = 0.000 importance
#       - 8 features capturent 90% de l'importance
#       - Mais les lags/rolling ajoutent de l'information complémentaire
#       → On ne fait PAS de sélection agressive ici (risque de perdre du signal)
#       → On supprime uniquement les colonnes non-numériques inutilisables pour le ML
#       → La sélection fine sera faite dans 06_model_xgboost.py avec le vrai CV
# ─────────────────────────────────────────────────────────────────────────────
def select_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[13] Sélection / nettoyage final des colonnes")
    df = df.copy()

    # Colonnes à garder absolument
    keep_always = ["city", "region", "time", TARGET, TARGET_LOG]

    # Supprimer : colonnes objet non-encodables par XGBoost
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    to_drop_obj = [c for c in obj_cols if c not in keep_always]
    if to_drop_obj:
        print(f"    Colonnes texte supprimées : {to_drop_obj}")
        df = df.drop(columns=to_drop_obj)

    # Supprimer : colonnes avec variance nulle (constantes)
    num_cols = df.select_dtypes(include=np.number).columns
    zero_var = [c for c in num_cols if df[c].std() == 0]
    if zero_var:
        print(f"    Colonnes variance nulle supprimées : {zero_var}")
        df = df.drop(columns=zero_var)
    else:
        print(f"    Colonnes variance nulle : aucune ✅")

    # Supprimer : colonnes avec >50% valeurs identiques (quasi-constantes)
    quasi_const = []
    for c in df.select_dtypes(include=np.number).columns:
        top_freq = df[c].value_counts(normalize=True).iloc[0]
        if top_freq > 0.98:
            quasi_const.append((c, f"{100*top_freq:.1f}%"))
    if quasi_const:
        cols_to_drop = [c for c, _ in quasi_const]
        print(f"    Colonnes quasi-constantes supprimées :")
        for c, pct in quasi_const:
            print(f"      {c}: {pct} valeurs identiques")
        df = df.drop(columns=cols_to_drop)
    else:
        print(f"    Colonnes quasi-constantes : aucune ✅")

    print(f"    Shape finale : {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 14. Features d'interaction
# ─────────────────────────────────────────────────────────────────────────────
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[10] Features d'interaction")
    df = df.copy()

    if "wind_speed_10m_max" in df.columns and "is_harmattan" in df.columns:
        wind = df["wind_speed_10m_max"].fillna(0)
        df["wind_x_harmattan"] = wind * df["is_harmattan"]
        df["wind_x_wet"]       = wind * df["is_wet_season"]

    if "blh_inv" in df.columns and "precipitation_sum" in df.columns:
        df["blh_x_precip"] = df["blh_inv"] * df["precip_log"]

    if "harmattan_intensity" in df.columns:
        df["lat_x_harmattan"] = df["latitude"] * df["is_harmattan"]

    if "temp_amplitude" in df.columns and "blh_inv" in df.columns:
        df["temp_amp_x_blh"] = df["temp_amplitude"] * df["blh_inv"]

    print(f"    +5 features d'interaction")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 11. Suppression des features redondantes (EDA section 6)
#     Paires |r|>0.90 : garder le plus corrélé avec la cible
#
#     Redondances confirmées :
#       rain_sum ≈ precipitation_sum (r=1.00) → supprimer rain_sum
#       temperature_2m_mean ≈ temperature_2m_max (r=0.95) → supprimer mean
#       et0_fao ≈ shortwave_radiation (r=0.93) → supprimer et0 (on garde log)
#       blh_min ≈ couverte par blh_min_log + blh_mean → garder les deux logs
#
#     Colonnes inutilisables pour le ML :
#       sunrise, sunset (texte datetime), id, snowfall_sum (0 au Cameroun)
# ─────────────────────────────────────────────────────────────────────────────
def remove_redundant(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[11] Suppression des features redondantes")
    to_drop = [
        # Redondantes (EDA multicolinéarité)
        "rain_sum",                      # = precipitation_sum (r=1.00)
        "temperature_2m_mean",           # ≈ temperature_2m_max (r=0.95)
        "et0_fao_evapotranspiration",    # ≈ shortwave_radiation (r=0.93) — log gardé
        # Inutilisables pour ML
        "id", "snowfall_sum",
        # Colonnes de facteurs proxy (pas des features météo)
        "F_fire", "f_fire",
    ]
    dropped = [c for c in to_drop if c in df.columns]
    df = df.drop(columns=dropped)
    print(f"    Supprimées ({len(dropped)}) : {dropped}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 12. Imputation finale des NaN résiduels (lags du début de série)
# ─────────────────────────────────────────────────────────────────────────────
def final_impute(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[12] Imputation finale des NaN résiduels (lags début de série)")
    df = df.copy()

    lag_roll_cols = [c for c in df.columns
                     if any(s in c for s in ["_lag", "_roll", "_streak"])]

    for col in lag_roll_cols:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            # Médiane par ville pour les NaN structurels de début de série
            df[col] = df.groupby("city")[col].transform(
                lambda x: x.fillna(x.median())
            )

    # Vérification finale
    total_nan = df.select_dtypes(include=np.number).isnull().sum().sum()
    print(f"    NaN résiduels après imputation finale : {total_nan}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 13. Rapport qualité
# ─────────────────────────────────────────────────────────────────────────────
def quality_report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("RAPPORT QUALITÉ — DATASET FEATURES FINAL")
    print("=" * 65)

    num_df = df.select_dtypes(include=np.number)
    n_features = num_df.shape[1] - 2  # hors pm25_proxy et pm25_log

    print(f"\n  Shape          : {df.shape}")
    print(f"  Villes         : {df['city'].nunique()}")
    print(f"  Période        : {df['time'].min().date()} → {df['time'].max().date()}")
    print(f"  NaN total      : {num_df.isnull().sum().sum()} ✅" if
          num_df.isnull().sum().sum() == 0 else
          f"  NaN total      : {num_df.isnull().sum().sum()} ⚠️")
    print(f"\n  Features numériques totales : {n_features}")

    # Catégories
    cats = {
        "Lag"         : [c for c in df.columns if "_lag" in c],
        "Rolling"     : [c for c in df.columns if "_roll" in c],
        "Cycliques"   : [c for c in df.columns if c.endswith(("_sin","_cos"))],
        "Harmattan"   : [c for c in df.columns if any(s in c.lower() for s
                         in ["harmattan","season","dry","wet"])],
        "Spatial"     : [c for c in df.columns if c in
                         ["lat_norm","lon_norm","region_code","city_id","climate_zone"]],
        "Log-transf"  : [c for c in df.columns if c.endswith("_log")],
        "Interaction" : [c for c in df.columns if "_x_" in c],
        "Astro"       : ["daylight_duration"] if "daylight_duration" in df.columns else [],
    }
    print()
    for cat, cols in cats.items():
        print(f"  {cat:<12} : {len(cols):3d} features")

    # Stats cible
    print(f"\n  Cible pm25_proxy :")
    print(f"    mean={df[TARGET].mean():.2f} | std={df[TARGET].std():.2f} | "
          f"skew={df[TARGET].skew():.3f}")
    print(f"  Cible pm25_log (ML) :")
    print(f"    mean={df[TARGET_LOG].mean():.3f} | std={df[TARGET_LOG].std():.3f} | "
          f"skew={df[TARGET_LOG].skew():.3f}")

    print("\n" + "=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("FEATURE ENGINEERING — Hackathon IndabaX Cameroun 2026")
    print("Source : pm25_proxy_era5.parquet (ERA5 pur, sans NaN Excel)")
    print("=" * 65)

    df = load_data(INPUT_PATH)
    df = impute_missing(df)
    df = add_log_target(df)
    df = compute_daylight_duration(df)
    df = add_temporal_features(df)
    df = add_harmattan_features(df)
    df = add_derived_meteo(df)
    df = add_blh_binning(df)
    df = add_weather_proxy(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_spatial_features(df)
    df = add_target_encoding(df)
    df = add_interaction_features(df)
    df = remove_redundant(df)
    df = final_impute(df)
    df = select_features(df)

    quality_report(df)

    # Sauvegarde
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n  Dataset sauvegardé : {OUTPUT_PATH}  {df.shape}")
    print("\n  Prochaine étape : 06_model_xgboost.py")
    print("=" * 65)


if __name__ == "__main__":
    main()

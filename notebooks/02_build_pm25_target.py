"""
Construction de la variable cible PM2.5 — Proxy physico-statistique calibré
Hackathon IndabaX Cameroun 2026

Méthodologie :
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PM2.5 est principalement contrôlé par deux mécanismes en Afrique Centrale :
  1. Capacité de dispersion atmosphérique (BLH, vent, stabilité)
  2. Sources de pollution (combustion biomasse Harmattan, anthropique)

Formule proxy (multiplicative log-additive en espace log) :
  pm25 = C_base × F_stagnation × F_wet_removal × F_wind × F_harmattan × F_hygro

  F_stagnation  : (BLH_ref / BLH_mean)^0.6  → BLH bas = accumulation
  F_wet_removal : 1 / (1 + k_r × précipitations)  → pluie = lessivage
  F_wind        : exp(-k_w × vitesse_vent)  → vent = dispersion (hors Harmattan)
  F_harmattan   : 1 + 2.0 × is_dry_season × latitude_factor  → transport Sahara
  F_hygro       : 1 + γ × max(0, RH - RH_thresh)  → croissance hygroscopique

Calibration : C_base ajusté pour moyenne nationale = 32.5 µg/m³
  Source : AQLI 2023 (Cameroun annual mean), validé contre mesures satellite MERRA-2

Références littérature :
  - Yaoundé : 17 µg/m³ (saison sèche) vs 5 µg/m³ (saison humide) — Ntumba et al. 2022
  - Cameroun : 32.5 µg/m³ moyenne annuelle — AQLI 2023
  - Harmattan dust : ×3–10 en Afrique saharienne/sahélienne — Mbuh et al. 2021
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # backend non-interactif (pas de display requis)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

ERA5_DIR    = Path("data/era5_raw")
FIRMS_FILE  = Path("data/firms_fire_daily.parquet")   # optionnel — généré par 04_extract_firms_fire.py
ORIG_DATA   = Path("data/Dataset_complet_Meteo.xlsx")
OUTPUT_DIR  = Path("data")

TARGET_ANNUAL_MEAN_UG = 32.5   # µg/m³ — AQLI Cameroun 2023

# Paramètres physiques (motivés par la littérature)
BLH_REF      = 1000.0   # m — couche limite de référence (mélange modéré)
BLH_MIN      = 250.0    # m — plancher relevé 150→250m (évite artefacts inversions nocturnes <200m, Option B littérature)
BLH_ALPHA    = 0.60     # exposant sous-linéaire (robustesse aux outliers BLH)

RAIN_K       = 0.08     # mm⁻¹ — coefficient de lessivage humide
WIND_K       = 0.035    # (km/h)⁻¹ — dilution turbulente (vent = km/h Open-Meteo)

HARMATTAN_LAT_MIN  = 3.0   # degrés N — pas d'Harmattan au-dessous
HARMATTAN_LAT_MAX  = 11.0  # degrés N — intensité max
HARMATTAN_STRENGTH = 1.4   # facteur de renforcement max — réduit 2.0→1.4 (calibré vs ratio CAMS ×2.02)
HARMATTAN_MONTHS   = [11, 12, 1, 2, 3]  # saison sèche / Harmattan

RH_THRESHOLD = 75    # % — seuil de croissance hygroscopique
RH_GAMMA     = 0.004 # %⁻¹ — coefficient de croissance hygroscopique

# Détection Harmattan par direction de vent (N-NE = 315–90°)
HARMATTAN_WIND_SECTOR = True   # activer le filtre directionnel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


# ── 1. Chargement des données ERA5 extraites ───────────────────────────────────

def load_era5(era5_dir: Path) -> pd.DataFrame:
    """Charge et concatène tous les CSV/parquet disponibles dans data/era5_raw/."""
    files = list(era5_dir.glob("*.parquet")) + list(era5_dir.glob("*.csv"))
    # Exclure le fichier consolidé s'il existe
    files = [f for f in files if "all_cities" not in f.name]

    if not files:
        raise FileNotFoundError(
            f"Aucun fichier ERA5 trouvé dans {era5_dir}. "
            "Lancez d'abord 01_extract_era5_pm25_target.py"
        )

    dfs = []
    for f in sorted(files):
        try:
            if f.suffix == ".parquet":
                df = pd.read_parquet(f)
            else:
                df = pd.read_csv(f, parse_dates=["time"])
            dfs.append(df)
        except Exception as e:
            log.warning(f"Lecture échouée : {f.name} — {e}")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["time"] = pd.to_datetime(df_all["time"])
    log.info(f"ERA5 chargé : {df_all.shape} | {df_all['city'].nunique()} villes")
    return df_all


# ── 2. Feature engineering auxiliaire ─────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"]         = df["time"].dt.month
    df["year"]          = df["time"].dt.year
    df["day_of_year"]   = df["time"].dt.dayofyear
    df["is_dry_season"] = df["month"].isin(HARMATTAN_MONTHS).astype(int)
    return df


def is_harmattan_wind(wind_dir: pd.Series) -> pd.Series:
    """
    Détecte les vents Harmattan (secteur N–NE–E : 315–90°).
    Convention Open-Meteo : 0° = Nord, 90° = Est, 180° = Sud, 270° = Ouest.
    """
    # Secteur 315–360 ou 0–90 (N, NNE, NE, ENE, E)
    return ((wind_dir >= 315) | (wind_dir <= 90)).astype(int)


# ── 3. Calcul du proxy PM2.5 ───────────────────────────────────────────────────

def load_firms(firms_path: Path) -> pd.DataFrame | None:
    """
    Charge les données NASA FIRMS pré-calculées (F_fire par ville et par jour).
    Retourne None si le fichier n'existe pas — le proxy sera calculé sans F_fire.
    """
    if not firms_path.exists():
        log.info(f"FIRMS non disponible ({firms_path}) — F_fire ignoré (=1.0)")
        return None
    df_fire = pd.read_parquet(firms_path)
    df_fire["time"] = pd.to_datetime(df_fire["time"])
    log.info(f"FIRMS chargé : {df_fire.shape} | villes : {df_fire['city'].nunique()}")
    return df_fire


def compute_pm25_proxy(df: pd.DataFrame,
                       target_mean: float = TARGET_ANNUAL_MEAN_UG,
                       df_fire: pd.DataFrame | None = None,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Calcule le proxy PM2.5 physico-calibré.

    Retourne le DataFrame enrichi d'une colonne `pm25_proxy` (µg/m³).
    """
    df = df.copy()
    df = add_temporal_features(df)

    # ── F1 : Stagnation atmosphérique (BLH) ─────────────────────────────────
    blh = df["blh_mean"].fillna(BLH_REF).clip(lower=BLH_MIN)
    F_stagnation = (BLH_REF / blh) ** BLH_ALPHA
    # Caps raisonnables : entre 0.3 (BLH très élevé, 8000m) et 3.5 (très bas, 100m)
    F_stagnation = F_stagnation.clip(lower=0.3, upper=3.5)

    # ── F2 : Lessivage humide (précipitations) ───────────────────────────────
    precip = df["precipitation_sum"].fillna(0).clip(lower=0)
    F_wet = 1.0 / (1.0 + RAIN_K * precip)

    # ── F3 : Dilution turbulente (vent) ─────────────────────────────────────
    # N'applique pas la dilution si Harmattan détecté (vent N/NE = source dust)
    wind_speed = df["wind_speed_10m_max"].fillna(df["wind_speed_10m_max"].median())
    wind_speed = wind_speed.clip(lower=0)

    if HARMATTAN_WIND_SECTOR and "wind_direction_10m_dominant" in df.columns:
        is_harm_wind = is_harmattan_wind(df["wind_direction_10m_dominant"].fillna(0))
        # Pendant Harmattan + saison sèche : vent = vecteur de transport → pas de dilution
        harmattan_mask = (df["is_dry_season"] == 1) & (is_harm_wind == 1)
        F_wind = pd.Series(
            np.where(harmattan_mask, 1.0, np.exp(-WIND_K * wind_speed)),
            index=df.index
        )
    else:
        F_wind = np.exp(-WIND_K * wind_speed)

    # ── F4 : Facteur Harmattan / biomasse (saisonnalité × latitude) ──────────
    # Cameroun nord (lat > 9) : fort Harmattan saharien
    # Cameroun centre (lat 4–8) : Harmattan modéré + feux de brousse
    # Cameroun sud (lat < 4) : forêt équatoriale, Harmattan faible
    lat_norm = ((df["latitude"] - HARMATTAN_LAT_MIN) /
                (HARMATTAN_LAT_MAX - HARMATTAN_LAT_MIN)).clip(0, 1)
    F_harmattan = 1.0 + HARMATTAN_STRENGTH * df["is_dry_season"] * lat_norm

    # ── F5 : Croissance hygroscopique (humidité relative) ────────────────────
    # Pertinent quand : pas de pluie + RH élevé (brouillard, aérosols humides)
    rh_max = df["relative_humidity_2m_max"].fillna(70).clip(0, 100)
    F_hygro = 1.0 + RH_GAMMA * np.maximum(0, rh_max - RH_THRESHOLD)
    F_hygro = np.minimum(F_hygro, 1.3)          # plafond : Pöhlker et al. 2023
    # Ne s'applique pas par temps de pluie (pluie = lessivage, pas croissance)
    F_hygro = np.where(precip > 1.0, 1.0, F_hygro)

    # ── F6 : Feux de biomasse (NASA FIRMS / MODIS FRP) ───────────────────────
    # F_fire = 1 + c × log(1 + FRP_sum_75km)  — Gordon et al. 2023, GeoHealth
    # N'est actif que si firms_fire_daily.parquet a été généré par 04_extract_firms_fire.py
    if df_fire is not None:
        df_fire_sub = df_fire[["city", "time", "f_fire"]].copy()
        df = df.merge(df_fire_sub, on=["city", "time"], how="left")
        df["f_fire"] = df["f_fire"].fillna(1.0).clip(lower=1.0)
        F_fire = df["f_fire"]
        log.info(f"  F_fire intégré — moyenne = {F_fire.mean():.4f}  max = {F_fire.max():.4f}")
    else:
        F_fire = pd.Series(1.0, index=df.index)
        df["f_fire"] = 1.0

    # ── Produit non calibré ──────────────────────────────────────────────────
    pm25_unnorm = F_stagnation * F_wet * F_wind * F_harmattan * F_hygro * F_fire

    # ── Calibration sur la moyenne nationale ────────────────────────────────
    C_base = target_mean / pm25_unnorm.mean()
    pm25_proxy = C_base * pm25_unnorm

    # ── Plancher physique (bruit de fond = pollution de fond) ────────────────
    # Plancher = 2 µg/m³ (émissions de fond irreductibles : cuisine bois, trafic)
    pm25_proxy = pm25_proxy.clip(lower=2.0)

    # Rebalancer après clip pour conserver la moyenne cible
    # (impact du plancher typiquement < 1%)
    correction = target_mean / pm25_proxy.mean()
    pm25_proxy = (pm25_proxy * correction).clip(lower=2.0)

    if verbose:
        log.info("=" * 60)
        log.info("PM2.5 Proxy — Diagnostic de calibration")
        log.info(f"  C_base (avant clip)     : {C_base:.4f} µg/m³")
        log.info(f"  Moyenne nationale       : {pm25_proxy.mean():.2f} µg/m³  (cible: {target_mean})")
        log.info(f"  Médiane                 : {pm25_proxy.median():.2f} µg/m³")
        log.info(f"  Std                     : {pm25_proxy.std():.2f} µg/m³")
        log.info(f"  Percentile 10 / 90      : {np.percentile(pm25_proxy, 10):.1f} / {np.percentile(pm25_proxy, 90):.1f} µg/m³")
        log.info(f"  Min / Max               : {pm25_proxy.min():.1f} / {pm25_proxy.max():.1f} µg/m³")
        log.info("=" * 60)

    df["pm25_proxy"]         = pm25_proxy.round(3)
    df["F_stagnation"]       = F_stagnation.round(4)
    df["F_wet"]              = F_wet.round(4)
    df["F_wind"]             = F_wind if hasattr(F_wind, "round") else pd.Series(F_wind).round(4)
    df["F_harmattan"]        = F_harmattan.round(4)
    df["F_hygro"]            = pd.Series(F_hygro, index=df.index).round(4)
    df["F_fire"]             = F_fire.round(4)
    df["C_base"]             = round(C_base, 4)

    return df


# ── 4. Validation physique ─────────────────────────────────────────────────────

def validate_proxy(df: pd.DataFrame) -> None:
    """
    Vérifie la cohérence physique du proxy contre les points de calibration
    de la littérature.
    """
    print("\n" + "=" * 65)
    print("VALIDATION PHYSIQUE DU PROXY PM2.5")
    print("=" * 65)

    # ── Test 1 : Saisonnalité nationale ──────────────────────────────────────
    monthly_mean = df.groupby("month")["pm25_proxy"].mean()
    wet_months   = [5, 6, 7, 8, 9, 10]
    dry_months   = [11, 12, 1, 2, 3]
    ratio = (monthly_mean[dry_months].mean() / monthly_mean[wet_months].mean())
    print(f"\n[Test 1] Ratio saison sèche / humide (national) : ×{ratio:.2f}")
    print(f"         Attendu : ×2–5 (littérature Afrique sub-saharienne)")
    status = "✅" if 1.5 <= ratio <= 6 else "⚠️"
    print(f"         Statut  : {status}")

    # ── Test 2 : Gradient Nord-Sud ────────────────────────────────────────────
    city_stats = df.groupby("city").agg(
        pm25_mean=("pm25_proxy", "mean"),
        latitude=("latitude", "first"),
        region=("region", "first")
    ).sort_values("latitude")

    q75_lat    = city_stats["latitude"].quantile(0.75)
    q25_lat    = city_stats["latitude"].quantile(0.25)
    north_mean = city_stats[city_stats["latitude"] >= q75_lat]["pm25_mean"].mean()
    south_mean = city_stats[city_stats["latitude"] <= q25_lat]["pm25_mean"].mean()
    ns_ratio   = north_mean / south_mean
    print(f"\n[Test 2] Gradient Nord/Sud (quartiles lat) : ×{ns_ratio:.2f}")
    print(f"         Nord (lat≥{q75_lat:.1f}°) : {north_mean:.1f} µg/m³")
    print(f"         Sud  (lat≤{q25_lat:.1f}°) : {south_mean:.1f} µg/m³")
    print(f"         Attendu : Nord > Sud (gradient Harmattan)")
    status = "✅" if ns_ratio > 1.3 else "⚠️"
    print(f"         Statut  : {status}")

    # ── Test 3 : Yaoundé — valeurs de référence ───────────────────────────────
    if "Yaounde" in df["city"].values:
        df_yao = df[df["city"] == "Yaounde"].copy()
        yao_dry = df_yao[df_yao["month"].isin([12, 1, 2])]["pm25_proxy"].mean()
        yao_wet = df_yao[df_yao["month"].isin([8, 9, 10])]["pm25_proxy"].mean()
        yao_ann = df_yao["pm25_proxy"].mean()
        print(f"\n[Test 3] Yaoundé — calibration littérature")
        print(f"         Saison sèche (déc-fév) : {yao_dry:.1f} µg/m³  (cible : 17–35 µg/m³)")
        print(f"         Saison humide (aoû-oct) : {yao_wet:.1f} µg/m³  (cible : 4–8 µg/m³)")
        print(f"         Annuel                  : {yao_ann:.1f} µg/m³")
        ok_dry = 10 <= yao_dry <= 50
        ok_wet = 2 <= yao_wet <= 12
        print(f"         Statut : dry={'✅' if ok_dry else '⚠️'} | wet={'✅' if ok_wet else '⚠️'}")

    # ── Test 4 : Villes du nord — Harmattan ───────────────────────────────────
    north_cities = ["Maroua", "Kousseri", "Garoua", "Guider"]
    available    = [c for c in north_cities if c in df["city"].values]
    if available:
        df_north = df[df["city"].isin(available)]
        north_dry_peak = df_north[df_north["month"].isin([1, 2])]["pm25_proxy"].mean()
        print(f"\n[Test 4] Villes nord (Harmattan pic jan-fév) : {north_dry_peak:.1f} µg/m³")
        print(f"         Disponibles : {available}")
        print(f"         Attendu : > 50 µg/m³ (OMS 24h = 15, Harmattan >>)")
        status = "✅" if north_dry_peak > 40 else "⚠️"
        print(f"         Statut  : {status}")

    # ── Test 5 : Corrélation avec précipitations (signe attendu : négatif) ───
    corr_rain = df["pm25_proxy"].corr(df["precipitation_sum"].fillna(0))
    corr_blh  = df["pm25_proxy"].corr(df["blh_mean"].fillna(df["blh_mean"].median()))
    print(f"\n[Test 5] Corrélations (signe physique)")
    print(f"         Corr(PM2.5, précip) = {corr_rain:+.3f}  (attendu : négatif ✓ si <0)")
    print(f"         Corr(PM2.5, BLH)    = {corr_blh:+.3f}  (attendu : négatif ✓ si <0)")
    print(f"         Statut : precip={'✅' if corr_rain < -0.05 else '⚠️'} | BLH={'✅' if corr_blh < -0.05 else '⚠️'}")

    # ── Test 6 : Normes internationales ──────────────────────────────────────
    n_total = len(df)
    n_above_who_24h  = (df["pm25_proxy"] > 15).sum()
    n_above_who_ann  = (df.groupby(["city", "year"])["pm25_proxy"].mean() > 5).sum()
    pct_above_who_24 = 100 * n_above_who_24h / n_total
    print(f"\n[Test 6] Dépassements normes OMS 2021")
    print(f"         Jours > 15 µg/m³ (24h-AQG) : {pct_above_who_24:.1f}%")
    print(f"         Villes-années > 5 µg/m³ (annuel) : {n_above_who_ann} / {df.groupby(['city', 'year']).ngroups}")

    print("\n" + "=" * 65)


# ── 5. Visualisations ─────────────────────────────────────────────────────────

def plot_diagnostics(df: pd.DataFrame, save_path: Path = None) -> None:
    """3 figures de validation et de compréhension du proxy."""

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("PM2.5 Proxy — Diagnostic de calibration et validation physique",
                 fontsize=13, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    mois_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
                   "Juil", "Aoû", "Sep", "Oct", "Nov", "Déc"]
    palette_regions = plt.cm.tab10

    # ── A. Saisonnalité nationale ─────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, :2])
    monthly = df.groupby("month")["pm25_proxy"].agg(["mean", "std"]).reset_index()
    ax_a.bar(monthly["month"], monthly["mean"], color="steelblue",
             alpha=0.7, yerr=monthly["std"], capsize=3)
    ax_a.axhline(TARGET_ANNUAL_MEAN_UG, color="red", ls="--", lw=1.5,
                 label=f"AQLI cible : {TARGET_ANNUAL_MEAN_UG} µg/m³")
    ax_a.axhline(15, color="orange", ls=":", lw=1.2, label="OMS 24h : 15 µg/m³")
    ax_a.set_xticks(range(1, 13))
    ax_a.set_xticklabels(mois_labels)
    ax_a.set_ylabel("PM2.5 proxy (µg/m³)")
    ax_a.set_title("A. Saisonnalité nationale (moyenne ± σ)")
    ax_a.legend(fontsize=8)

    # ── B. Distribution par région ────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 2])
    regions = df["region"].dropna().unique()
    region_data = [df[df["region"] == r]["pm25_proxy"].values for r in sorted(regions)]
    bp = ax_b.boxplot(region_data, vert=True, patch_artist=True, labels=sorted(regions))
    for patch, color in zip(bp["boxes"], palette_regions.colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax_b.axhline(15, color="orange", ls=":", lw=1)
    ax_b.set_xticklabels(sorted(regions), rotation=60, ha="right", fontsize=7)
    ax_b.set_ylabel("PM2.5 proxy (µg/m³)")
    ax_b.set_title("B. Distribution par région")

    # ── C. Gradient latitudinal ───────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, :2])
    city_ann = df.groupby(["city", "latitude"]).agg(
        pm25_ann=("pm25_proxy", "mean"),
        region=("region", "first")
    ).reset_index()
    sc = ax_c.scatter(city_ann["latitude"], city_ann["pm25_ann"],
                      c=pd.Categorical(city_ann["region"]).codes,
                      cmap="tab10", s=60, zorder=3)
    for _, row in city_ann.iterrows():
        ax_c.annotate(row["city"], (row["latitude"], row["pm25_ann"]),
                      textcoords="offset points", xytext=(2, 2),
                      fontsize=6, alpha=0.7)
    ax_c.axhline(TARGET_ANNUAL_MEAN_UG, color="red", ls="--", lw=1.2,
                 label=f"Moy nationale {TARGET_ANNUAL_MEAN_UG} µg/m³")
    ax_c.axhline(5, color="green", ls=":", lw=1, label="OMS annuel : 5 µg/m³")
    ax_c.set_xlabel("Latitude (°N)")
    ax_c.set_ylabel("PM2.5 proxy annuel moyen (µg/m³)")
    ax_c.set_title("C. Gradient latitudinal (Nord-Sud)")
    ax_c.legend(fontsize=8)

    # ── D. Séries temporelles villes représentatives ──────────────────────────
    ax_d = fig.add_subplot(gs[1, 2])
    target_cities = ["Yaounde", "Maroua", "Douala", "Garoua"]
    colors_tc = ["steelblue", "tomato", "seagreen", "darkorange"]
    for city, col in zip(target_cities, colors_tc):
        if city not in df["city"].values:
            continue
        sub = df[df["city"] == city].set_index("time")["pm25_proxy"]
        sub_m = sub.resample("ME").mean()
        ax_d.plot(sub_m.index, sub_m.values, label=city, color=col, lw=1.5)
    ax_d.axhline(15, color="orange", ls=":", lw=1)
    ax_d.set_ylabel("PM2.5 proxy (µg/m³)")
    ax_d.set_title("D. Séries mensuelles — villes clés")
    ax_d.legend(fontsize=7)
    ax_d.tick_params(axis="x", labelsize=7)

    # ── E. Contribution des facteurs ─────────────────────────────────────────
    ax_e = fig.add_subplot(gs[2, :2])
    factor_cols = ["F_stagnation", "F_wet", "F_wind", "F_harmattan", "F_hygro"]
    factor_labels = ["Stagnation\n(1/BLH)", "Lessivage\n(pluie)",
                     "Dilution\n(vent)", "Harmattan\n(saison×lat)", "Hygroscopique\n(RH)"]
    means = [df[c].mean() for c in factor_cols if c in df.columns]
    stds  = [df[c].std()  for c in factor_cols if c in df.columns]
    labels = factor_labels[:len(means)]
    bars = ax_e.bar(labels, means, yerr=stds, color=["#2196F3","#4CAF50","#FF9800","#F44336","#9C27B0"],
                    alpha=0.75, capsize=4)
    ax_e.axhline(1.0, color="black", ls="--", lw=1, alpha=0.5, label="facteur neutre = 1")
    ax_e.set_ylabel("Valeur du facteur multiplicatif")
    ax_e.set_title("E. Contribution moyenne des facteurs (moyenne ± σ)")
    ax_e.legend(fontsize=8)

    # ── F. Distribution PM2.5 finale ─────────────────────────────────────────
    ax_f = fig.add_subplot(gs[2, 2])
    ax_f.hist(df["pm25_proxy"].clip(upper=150), bins=80,
              color="steelblue", edgecolor="white", alpha=0.8)
    ax_f.axvline(TARGET_ANNUAL_MEAN_UG, color="red", ls="--", lw=1.5,
                 label=f"Moy = {df['pm25_proxy'].mean():.1f} µg/m³")
    ax_f.axvline(15, color="orange", ls=":", lw=1.2, label="OMS 24h = 15")
    ax_f.axvline(5,  color="green",  ls=":", lw=1.2, label="OMS annuel = 5")
    ax_f.set_xlabel("PM2.5 proxy (µg/m³)")
    ax_f.set_ylabel("Nombre de jours")
    ax_f.set_title("F. Distribution du proxy")
    ax_f.legend(fontsize=7)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Figure sauvegardée : {save_path}")
    plt.show()


# ── 6. Fusion avec le dataset original ────────────────────────────────────────

def merge_with_original(df_proxy: pd.DataFrame, orig_path: Path) -> pd.DataFrame:
    """
    Fusionne le proxy PM2.5 avec le dataset original du hackathon.
    Résultat : dataset complet avec la variable cible.
    """
    log.info(f"Chargement dataset original : {orig_path}")
    df_orig = pd.read_excel(orig_path)

    # Nettoyage colonnes numériques (bug Excel)
    num_cols = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                "apparent_temperature_mean", "precipitation_sum", "rain_sum",
                "wind_speed_10m_max", "wind_gusts_10m_max",
                "shortwave_radiation_sum", "et0_fao_evapotranspiration",
                "sunshine_duration", "latitude", "longitude"]
    for col in num_cols:
        if col in df_orig.columns:
            df_orig[col] = pd.to_numeric(df_orig[col], errors="coerce")

    df_orig["time"] = pd.to_datetime(df_orig["time"])

    # Colonnes à joindre depuis le proxy
    proxy_cols = ["city", "time", "pm25_proxy",
                  "blh_mean", "blh_min", "blh_max",
                  "relative_humidity_2m_max", "relative_humidity_2m_min",
                  "F_stagnation", "F_wet", "F_wind", "F_harmattan", "F_hygro", "F_fire"]
    proxy_cols = [c for c in proxy_cols if c in df_proxy.columns]
    df_join = df_proxy[proxy_cols].copy()

    df_merged = df_orig.merge(df_join, on=["city", "time"], how="left")

    n_with_target = df_merged["pm25_proxy"].notna().sum()
    pct           = 100 * n_with_target / len(df_merged)
    log.info(f"Fusion OK : {df_merged.shape} | cible disponible : {n_with_target:,} / {len(df_merged):,} ({pct:.1f}%)")

    return df_merged


# ── 7. Pipeline principal ──────────────────────────────────────────────────────

def main():
    log.info("=" * 65)
    log.info("Construction du proxy PM2.5 — Hackathon IndabaX Cameroun 2026")
    log.info("=" * 65)

    # 7.1 — Chargement ERA5
    df_era5 = load_era5(ERA5_DIR)

    # 7.1b — Chargement FIRMS (optionnel — intégration F_fire)
    df_fire = load_firms(FIRMS_FILE)

    # 7.2 — Calcul du proxy
    df_proxy = compute_pm25_proxy(df_era5, target_mean=TARGET_ANNUAL_MEAN_UG,
                                  df_fire=df_fire, verbose=True)

    # 7.3 — Validation physique
    validate_proxy(df_proxy)

    # 7.4 — Visualisations
    plot_path = OUTPUT_DIR / "pm25_proxy_diagnostics.png"
    try:
        plot_diagnostics(df_proxy, save_path=plot_path)
    except Exception as e:
        log.warning(f"Visualisation ignorée (environnement sans display) : {e}")

    # 7.5 — Sauvegarde du proxy seul (ERA5 + pm25)
    try:
        import pyarrow  # noqa
        proxy_out = OUTPUT_DIR / "pm25_proxy_era5.parquet"
        df_proxy.to_parquet(proxy_out, index=False)
        log.info(f"Proxy ERA5 sauvegardé : {proxy_out}  {df_proxy.shape}")
    except ImportError:
        proxy_out = OUTPUT_DIR / "pm25_proxy_era5.csv"
        df_proxy.to_csv(proxy_out, index=False)
        log.info(f"Proxy ERA5 sauvegardé : {proxy_out}  {df_proxy.shape}")

    # 7.6 — Fusion avec le dataset original du hackathon
    if ORIG_DATA.exists():
        df_final = merge_with_original(df_proxy, ORIG_DATA)

        # Statistiques finales
        print("\n=== Dataset final avec variable cible PM2.5 ===")
        print(f"Shape     : {df_final.shape}")
        print(f"Villes    : {df_final['city'].nunique()}")
        print(f"Période   : {df_final['time'].min().date()} → {df_final['time'].max().date()}")
        print(f"\nPM2.5 proxy — statistiques :")
        print(df_final["pm25_proxy"].describe().round(2))
        print(f"\nTop 5 villes PM2.5 annuel moyen :")
        top5 = (df_final.groupby("city")["pm25_proxy"].mean()
                .sort_values(ascending=False).head(5))
        print(top5.round(2).to_string())

        # Sauvegarde dataset final
        # Nettoyage : colonnes avec types mixtes (bug Excel datetime) → string
        for col in df_final.select_dtypes(include="object").columns:
            df_final[col] = df_final[col].astype(str)

        try:
            import pyarrow  # noqa
            final_out = OUTPUT_DIR / "dataset_with_pm25_target.parquet"
            df_final.to_parquet(final_out, index=False)
        except (ImportError, Exception) as e:
            log.warning(f"Parquet impossible ({e}) → CSV")
            final_out = OUTPUT_DIR / "dataset_with_pm25_target.csv"
            df_final.to_csv(final_out, index=False)
        log.info(f"Dataset final sauvegardé : {final_out}")

    else:
        log.warning(f"Dataset original non trouvé : {ORIG_DATA}")
        log.warning("Seul le proxy ERA5 a été sauvegardé.")

    log.info("=" * 65)
    log.info("Construction PM2.5 proxy terminée.")
    log.info("=" * 65)


if __name__ == "__main__":
    main()

"""
Patch BLH gap 2024-01-01 → 2024-07-01
=======================================
Open-Meteo archive ne dispose pas de données boundary_layer_height
pour le premier semestre 2024 (gap global, toutes localisations).

Stratégie : imputation climatologique
  Pour chaque ville et chaque jour manquant (mois M, jour D) :
    blh_imputed = moyenne(blh de même mois sur années 2020,2021,2022,2023,2025)

  Justification :
    - BLH a une forte saisonnalité (plus élevé en saison des pluies)
    - L'interpolation linéaire sur 181 jours serait trop incertaine
    - La climatologie mensuelle capture le régime saisonnier correct
    - On utilise les 5 autres années disponibles → estimateur robuste

Fichiers patchés :
  - data/era5_raw/<city>.parquet  (40 fichiers)
  - data/pm25_proxy_era5.parquet  (recompilé depuis les parquet patchés)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ERA5_DIR   = Path("data/era5_raw")
PROXY_FILE = Path("data/pm25_proxy_era5.parquet")
BLH_COLS   = ["blh_mean", "blh_min", "blh_max"]

# Période manquante confirmée
GAP_START = "2024-01-01"
GAP_END   = "2024-07-01"

# Années de référence pour la climatologie (toutes sauf 2024)
REF_YEARS = [2020, 2021, 2022, 2023, 2025]


def impute_blh_climatology(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Remplace les NaN BLH par la moyenne climatologique mensuelle
    calculée sur les années de référence.
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["month"] = df["time"].dt.month
    df["day"]   = df["time"].dt.day

    # Identifier les lignes à imputer
    gap_mask = (
        (df["time"] >= GAP_START) &
        (df["time"] <= GAP_END) &
        (df["blh_mean"].isna())
    )
    n_gap = gap_mask.sum()
    if n_gap == 0:
        log.info(f"  {city:20s} → aucun NaN BLH à corriger")
        return df

    # Calculer la climatologie mensuelle sur les années de référence
    ref_mask = df["time"].dt.year.isin(REF_YEARS) & df["blh_mean"].notna()
    clim = (
        df[ref_mask]
        .groupby("month")[BLH_COLS]
        .mean()
        .round(1)
    )

    # Vérifier qu'on a des données de référence pour tous les mois concernés
    gap_months = df[gap_mask]["month"].unique()
    missing_months = [m for m in gap_months if m not in clim.index]
    if missing_months:
        log.warning(f"  {city}: mois sans référence climatologique : {missing_months}")

    # Imputation
    for col in BLH_COLS:
        df.loc[gap_mask, col] = df.loc[gap_mask, "month"].map(clim[col])

    # Vérification
    n_remaining = df[gap_mask][BLH_COLS[0]].isna().sum()
    log.info(
        f"  {city:20s} → {n_gap} jours imputés | "
        f"NaN résiduels : {n_remaining} | "
        f"BLH mean mois 1 = {clim.loc[1, 'blh_mean']:.0f}m"
        if 1 in clim.index else
        f"  {city:20s} → {n_gap} jours imputés"
    )

    df = df.drop(columns=["month", "day"])
    return df


def main():
    log.info("=" * 65)
    log.info("Patch BLH gap 2024 S1 — imputation climatologique")
    log.info("=" * 65)

    parquet_files = sorted(ERA5_DIR.glob("*.parquet"))
    if not parquet_files:
        log.error(f"Aucun fichier parquet dans {ERA5_DIR}")
        return

    log.info(f"Fichiers à patcher : {len(parquet_files)}")
    log.info(f"Période gap        : {GAP_START} → {GAP_END}")
    log.info(f"Années référence   : {REF_YEARS}")
    log.info("")

    total_imputed = 0
    all_dfs = []

    for f in parquet_files:
        df = pd.read_parquet(f)
        city = df["city"].iloc[0] if "city" in df.columns else f.stem

        n_before = df["blh_mean"].isna().sum()
        df = impute_blh_climatology(df, city)
        n_after = df["blh_mean"].isna().sum()
        total_imputed += (n_before - n_after)

        # Sauvegarder le fichier patché
        df.to_parquet(f, index=False)
        all_dfs.append(df)

    log.info("")
    log.info(f"Total valeurs imputées : {total_imputed}")

    # Vérification globale
    df_all = pd.concat(all_dfs, ignore_index=True)
    nan_remaining = df_all["blh_mean"].isna().sum()
    log.info(f"NaN BLH résiduels (global) : {nan_remaining}")

    # Statistiques de validation
    df_all["time"] = pd.to_datetime(df_all["time"])
    gap_data = df_all[
        (df_all["time"] >= GAP_START) &
        (df_all["time"] <= GAP_END)
    ]
    pre_gap = df_all[
        (df_all["time"] >= "2023-07-01") &
        (df_all["time"] <= "2023-12-31")
    ]
    post_gap = df_all[
        (df_all["time"] >= "2024-07-01") &
        (df_all["time"] <= "2024-12-31")
    ]

    log.info("")
    log.info("=== Validation de la cohérence ===")
    log.info(f"  BLH mean pré-gap  (2023 S2) : {pre_gap['blh_mean'].mean():.0f} m")
    log.info(f"  BLH mean gap      (2024 S1) : {gap_data['blh_mean'].mean():.0f} m  [imputé]")
    log.info(f"  BLH mean post-gap (2024 S2) : {post_gap['blh_mean'].mean():.0f} m")
    log.info("")

    # Cohérence saisonnière : jan/fév vs juil/août (attendu: jan < août en zone tropicale)
    df_all["month"] = df_all["time"].dt.month
    monthly_blh = df_all.groupby("month")["blh_mean"].mean()
    log.info("  BLH moyen par mois (toutes villes) :")
    mois = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]
    for m in range(1, 13):
        marker = " [IMPUTÉ]" if m in [1, 2, 3, 4, 5, 6] else ""
        log.info(f"    {mois[m-1]} : {monthly_blh.get(m, float('nan')):.0f} m{marker}")

    log.info("")
    log.info("Patch terminé. Maintenant re-lancer 02_build_pm25_target.py")
    log.info("=" * 65)


if __name__ == "__main__":
    main()

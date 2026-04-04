"""
Script d'exploration des données — Hackathon IndabaX Cameroun 2026
==================================================================
Lance ce script pour explorer et comprendre chaque fichier de données du projet.

Usage :
    conda run -n hackathon_pm25 python notebooks/explore_data.py
    conda run -n hackathon_pm25 python notebooks/explore_data.py --file proxy
    conda run -n hackathon_pm25 python notebooks/explore_data.py --file features
    conda run -n hackathon_pm25 python notebooks/explore_data.py --city Maroua
    conda run -n hackathon_pm25 python notebooks/explore_data.py --city Douala --year 2024

Fichiers disponibles :
    proxy     → data/pm25_proxy_era5.parquet       (source principale, 87 240 lignes)
    features  → data/dataset_features.parquet      (ML-ready, 87 240 × 144 variables)
    targets   → data/dataset_with_pm25_target.parquet (Excel fusionné + proxy)
    firms     → data/firms_fire_daily.parquet      (feux NASA FIRMS)
    unc       → data/pm25_with_uncertainty.parquet (proxy + intervalles de confiance MC)
    pred      → models/test_predictions_2025.parquet (prédictions ML 2025)
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ── Chemins ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

FILES = {
    "proxy": DATA / "pm25_proxy_era5.parquet",
    "features": DATA / "dataset_features.parquet",
    "targets": DATA / "dataset_with_pm25_target.parquet",
    "firms": DATA / "firms_fire_daily.parquet",
    "unc": DATA / "pm25_with_uncertainty.parquet",
    "pred": MODELS / "test_predictions_2025.parquet",
}

OMS_24H = 15.0
AQLI_REF = 32.5

# ── Helpers ────────────────────────────────────────────────────────────────────
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"


def sep(title="", char="─", width=70):
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n{BOLD}{CYAN}── {title} {'─' * pad}{RESET}")
    else:
        print(char * width)


def color_pm25(val):
    if val < OMS_24H:
        return f"{GREEN}{val:.1f}{RESET}"
    elif val < 35:
        return f"{YELLOW}{val:.1f}{RESET}"
    elif val < 75:
        return f"\033[33m{val:.1f}{RESET}"
    else:
        return f"{RED}{val:.1f}{RESET}"


def load(name):
    path = FILES[name]
    if not path.exists():
        print(f"{RED}✗ Fichier introuvable : {path}{RESET}")
        print("  Vérifie que le pipeline a été exécuté (voir CLAUDE.md).")
        return None
    df = pd.read_parquet(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    return df


# ── Vue d'ensemble de tous les fichiers ───────────────────────────────────────
def overview_all():
    sep("FICHIERS DE DONNÉES DU PROJET")
    print(f"{'Nom':<12} {'Fichier':<52} {'Lignes':>8} {'Colonnes':>8}  {'Statut'}")
    print("─" * 90)
    for name, path in FILES.items():
        if path.exists():
            df = pd.read_parquet(path)
            status = f"{GREEN}✓{RESET}"
            print(f"{name:<12} {str(path.relative_to(ROOT)):<52} {len(df):>8,} {len(df.columns):>8}  {status}")
        else:
            print(f"{name:<12} {str(path.relative_to(ROOT)):<52} {'—':>8} {'—':>8}  {RED}✗ manquant{RESET}")


# ── Exploration d'un fichier ──────────────────────────────────────────────────
def explore_file(name):
    df = load(name)
    if df is None:
        return

    sep(f"FICHIER : {name}  ({FILES[name].name})")

    print(f"\n{BOLD}Dimensions :{RESET} {len(df):,} lignes × {len(df.columns)} colonnes")

    if "time" in df.columns:
        print(f"{BOLD}Période    :{RESET} {df['time'].min().date()} → {df['time'].max().date()}")
    if "city" in df.columns:
        print(f"{BOLD}Villes     :{RESET} {df['city'].nunique()} villes")
        print(f"  {sorted(df['city'].unique())}")
    if "year" in df.columns or ("time" in df.columns):
        yr_col = "year" if "year" in df.columns else None
        if yr_col:
            print(f"{BOLD}Années     :{RESET} {sorted(df[yr_col].unique())}")

    sep("Colonnes et types")
    col_info = df.dtypes.reset_index()
    col_info.columns = ["Colonne", "Type"]
    col_info["NaN %"] = (df.isnull().mean() * 100).values.round(1)
    col_info["Exemple"] = [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "—"
                           for c in df.columns]

    # Highlight columns with NaN
    for _, row in col_info.iterrows():
        nan_str = f"{RED}{row['NaN %']:5.1f}%{RESET}" if row["NaN %"] > 5 else f"{row['NaN %']:5.1f}%"
        print(f"  {row['Colonne']:<45} {str(row['Type']):<10} NaN:{nan_str}  ex: {row['Exemple'][:40]}")

    sep("Statistiques PM2.5")
    if "pm25_proxy" in df.columns:
        pm = df["pm25_proxy"]
        print(f"  Moyenne    : {color_pm25(pm.mean())} µg/m³")
        print(f"  Médiane    : {color_pm25(pm.median())} µg/m³")
        print(f"  Max        : {color_pm25(pm.max())} µg/m³")
        print(f"  Min        : {pm.min():.1f} µg/m³")
        print(f"  Std        : {pm.std():.1f} µg/m³")
        pct_oms = (pm > OMS_24H).mean() * 100
        pct_danger = (pm > 75).mean() * 100
        print(f"\n  Jours > OMS 24h ({OMS_24H} µg/m³) : {YELLOW}{pct_oms:.1f}%{RESET} des observations")
        print(f"  Jours dangereux (>75 µg/m³)     : {RED}{pct_danger:.1f}%{RESET} des observations")


# ── Exploration d'une ville ───────────────────────────────────────────────────
def explore_city(city_name, year=None):
    df = load("proxy")
    if df is None:
        return

    # Recherche insensible à la casse
    all_cities = df["city"].unique()
    matches = [c for c in all_cities if city_name.lower() in c.lower()]

    if not matches:
        print(f"{RED}✗ Ville '{city_name}' introuvable.{RESET}")
        print(f"  Villes disponibles : {sorted(all_cities)}")
        return

    city = matches[0]
    if len(matches) > 1:
        print(f"{YELLOW}Plusieurs correspondances : {matches}. Utilisation de '{city}'.{RESET}")

    sub = df[df["city"] == city].copy()
    if year:
        sub = sub[sub["year"] == int(year)]

    sep(f"VILLE : {city}  ({year or '2020–2025'})")

    # Infos géographiques
    lat = sub["latitude"].iloc[0]
    lon = sub["longitude"].iloc[0]
    region = sub["region"].iloc[0] if "region" in sub.columns else "—"
    print(f"\n  Latitude   : {lat:.4f}°N")
    print(f"  Longitude  : {lon:.4f}°E")
    print(f"  Région     : {region}")
    print(f"  Période    : {sub['time'].min().date()} → {sub['time'].max().date()}")
    print(f"  Jours      : {len(sub)}")

    sep("PM2.5 — Statistiques")
    pm = sub["pm25_proxy"]
    print(f"  Moyenne annuelle : {color_pm25(pm.mean())} µg/m³  (AQLI réf: {AQLI_REF} µg/m³)")
    print(f"  Médiane          : {color_pm25(pm.median())} µg/m³")
    print(f"  Maximum          : {color_pm25(pm.max())} µg/m³")
    print(f"  Minimum          : {pm.min():.1f} µg/m³")

    pct_oms = (pm > OMS_24H).mean() * 100
    pct_75 = (pm > 75).mean() * 100
    print(f"\n  Jours > OMS 24h  : {YELLOW}{pct_oms:.0f}%{RESET} ({int(pct_oms/100*len(sub))} jours)")
    print(f"  Jours dangereux  : {RED}{pct_75:.0f}%{RESET} ({int(pct_75/100*len(sub))} jours)")

    sep("PM2.5 par mois (moyenne)")
    month_names = {1:"Jan",2:"Fév",3:"Mar",4:"Avr",5:"Mai",6:"Jun",
                   7:"Jul",8:"Aoû",9:"Sep",10:"Oct",11:"Nov",12:"Déc"}
    monthly = sub.groupby("month")["pm25_proxy"].mean()
    for m, val in monthly.items():
        bar_len = int(val / 3)
        bar = "█" * min(bar_len, 40)
        flag = f" {RED}← Harmattan{RESET}" if m in [12, 1, 2, 3] else ""
        print(f"  {month_names[m]:>3} : {bar:<40} {color_pm25(val)} µg/m³{flag}")

    sep("10 jours les plus pollués")
    cols_top10 = ["time", "pm25_proxy"]
    for c in ["precipitation_sum", "wind_speed_10m_max", "blh_mean"]:
        if c in sub.columns:
            cols_top10.append(c)
    top10 = sub.nlargest(10, "pm25_proxy")[cols_top10].reset_index(drop=True)
    top10["time"] = top10["time"].dt.date
    top10["pm25_proxy"] = top10["pm25_proxy"].round(1)
    top10.index += 1
    print(top10.to_string())

    # Prédictions ML si disponibles
    pred = load("pred")
    if pred is not None:
        city_pred = pred[pred["city"] == city]
        if not city_pred.empty:
            sep("Prédictions ML — 2025")
            pm_real = city_pred["pm25_proxy"]
            pm_ml = city_pred["pm25_pred_ensemble"]
            rmse = np.sqrt(((pm_real - pm_ml) ** 2).mean())
            corr = pm_real.corr(pm_ml)
            print(f"  RMSE     : {rmse:.2f} µg/m³")
            print(f"  R        : {corr:.4f}")
            print(f"  PM2.5 réel moyen 2025   : {color_pm25(pm_real.mean())} µg/m³")
            print(f"  PM2.5 prédit moyen 2025  : {color_pm25(pm_ml.mean())} µg/m³")


# ── Comparaison de villes ─────────────────────────────────────────────────────
def compare_cities():
    df = load("proxy")
    if df is None:
        return

    sep("COMPARAISON — 40 VILLES (classées par pollution)")
    city_stats = df.groupby("city").agg(
        region=("region", "first"),
        latitude=("latitude", "first"),
        pm25_mean=("pm25_proxy", "mean"),
        pm25_max=("pm25_proxy", "max"),
        jours_oms=("pm25_proxy", lambda x: (x > OMS_24H).sum()),
        jours_danger=("pm25_proxy", lambda x: (x > 75).sum()),
    ).reset_index().sort_values("pm25_mean", ascending=False)

    print(f"\n{'#':>3} {'Ville':<20} {'Région':<15} {'Lat':>6} {'Moy µg/m³':>10} "
          f"{'Max µg/m³':>10} {'J>OMS':>6} {'J>75':>6}")
    print("─" * 85)
    for i, row in enumerate(city_stats.itertuples(), 1):
        pm_str = color_pm25(row.pm25_mean)
        print(f"  {i:>2} {row.city:<20} {row.region:<15} {row.latitude:>6.2f} "
              f"{row.pm25_mean:>10.1f} {row.pm25_max:>10.0f} "
              f"{row.jours_oms:>6} {row.jours_danger:>6}")

    sep("Résumé national")
    print(f"  Moyenne nationale    : {color_pm25(df['pm25_proxy'].mean())} µg/m³")
    print(f"  Jours > OMS 24h     : {YELLOW}{(df['pm25_proxy'] > OMS_24H).mean()*100:.1f}%{RESET} des observations")
    print(f"  Jours dangereux >75  : {RED}{(df['pm25_proxy'] > 75).mean()*100:.1f}%{RESET} des observations")
    print(f"\n  Ville la + polluée   : {city_stats.iloc[0]['city']} "
          f"({color_pm25(city_stats.iloc[0]['pm25_mean'])} µg/m³)")
    print(f"  Ville la + propre    : {city_stats.iloc[-1]['city']} "
          f"({color_pm25(city_stats.iloc[-1]['pm25_mean'])} µg/m³)")


# ── Résumé ML ─────────────────────────────────────────────────────────────────
def ml_summary():
    df = load("pred")
    if df is None:
        return

    sep("RÉSULTATS ML — PRÉDICTIONS 2025")
    rmse_all = np.sqrt(((df["pm25_proxy"] - df["pm25_pred_ensemble"]) ** 2).mean())
    r2_all = 1 - ((df["pm25_proxy"] - df["pm25_pred_ensemble"]) ** 2).sum() / \
             ((df["pm25_proxy"] - df["pm25_proxy"].mean()) ** 2).sum()
    mape_all = (abs(df["pm25_proxy"] - df["pm25_pred_ensemble"]) /
                df["pm25_proxy"].clip(lower=1)).mean() * 100

    print(f"\n  RMSE Ensemble : {BOLD}{rmse_all:.2f} µg/m³{RESET}")
    print(f"  R²  Ensemble  : {BOLD}{r2_all:.4f}{RESET}")
    print(f"  MAPE Ensemble : {BOLD}{mape_all:.1f}%{RESET}")

    sep("RMSE par ville (Top 10 meilleures et pires)")
    city_rmse = df.groupby("city").apply(
        lambda g: pd.Series({
            "RMSE": np.sqrt(((g["pm25_proxy"] - g["pm25_pred_ensemble"]) ** 2).mean()),
            "R²": 1 - ((g["pm25_proxy"] - g["pm25_pred_ensemble"]) ** 2).sum() /
                  ((g["pm25_proxy"] - g["pm25_proxy"].mean()) ** 2).sum(),
            "PM2.5_moyen": g["pm25_proxy"].mean()
        })
    ).reset_index().sort_values("RMSE")

    print(f"\n{BOLD}5 villes avec le meilleur RMSE :{RESET}")
    for row in city_rmse.head(5).itertuples():
        print(f"  {row.city:<20} RMSE={GREEN}{row.RMSE:.2f}{RESET}  R²={row._3:.4f}  PM2.5 moy={row.PM2_5_moyen:.1f}")

    print(f"\n{BOLD}5 villes avec le moins bon RMSE :{RESET}")
    for row in city_rmse.tail(5).sort_values("RMSE", ascending=False).itertuples():
        print(f"  {row.city:<20} RMSE={YELLOW}{row.RMSE:.2f}{RESET}  R²={row._3:.4f}  PM2.5 moy={row.PM2_5_moyen:.1f}")
    print(f"\n  Note : RMSE plus élevé au nord = zones Harmattan plus intenses et variables.")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Explorer les données PM2.5 du projet IndabaX Cameroun 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python notebooks/explore_data.py                      # Vue d'ensemble de tous les fichiers
  python notebooks/explore_data.py --file proxy         # Explorer le fichier proxy ERA5
  python notebooks/explore_data.py --file features      # Explorer les features ML
  python notebooks/explore_data.py --city Maroua        # Stats pour Maroua (2020–2025)
  python notebooks/explore_data.py --city Douala --year 2023
  python notebooks/explore_data.py --compare            # Comparer toutes les villes
  python notebooks/explore_data.py --ml                 # Résultats des modèles ML
        """
    )
    parser.add_argument("--file", choices=list(FILES.keys()), help="Explorer un fichier spécifique")
    parser.add_argument("--city", help="Nom d'une ville (partiel accepté)")
    parser.add_argument("--year", type=int, help="Filtrer par année (avec --city)")
    parser.add_argument("--compare", action="store_true", help="Comparer toutes les villes")
    parser.add_argument("--ml", action="store_true", help="Afficher les résultats ML")

    args = parser.parse_args()

    # Pas d'argument → vue d'ensemble
    if not any([args.file, args.city, args.compare, args.ml]):
        print(f"\n{BOLD}{'═' * 70}{RESET}")
        print(f"{BOLD}  PM2.5 CAMEROUN — Exploration des données (IndabaX 2026){RESET}")
        print(f"{BOLD}{'═' * 70}{RESET}")
        overview_all()
        print(f"\n{CYAN}Utilise --help pour voir toutes les options d'exploration.{RESET}\n")
        return

    if args.file:
        explore_file(args.file)

    if args.city:
        explore_city(args.city, args.year)

    if args.compare:
        compare_cities()

    if args.ml:
        ml_summary()

    print()


if __name__ == "__main__":
    main()

"""
Export des données en Excel — Hackathon IndabaX Cameroun 2026
=============================================================
Génère des fichiers Excel lisibles et formatés depuis les parquet du projet.

Usage :
    conda run -n hackathon_pm25 python notebooks/export_to_excel.py
    conda run -n hackathon_pm25 python notebooks/export_to_excel.py --file proxy
    conda run -n hackathon_pm25 python notebooks/export_to_excel.py --all
    conda run -n hackathon_pm25 python notebooks/export_to_excel.py --summary

Options :
    (aucune)    → Exporte un résumé consolidé (recommandé pour débuter)
    --all       → Exporte tous les fichiers séparément
    --file NAME → Exporte un fichier précis (proxy / features / pred / unc / firms / targets)
    --summary   → Génère PM25_RESUME.xlsx avec plusieurs onglets de synthèse
    --city NOM  → Exporte les données d'une ville spécifique
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
EXPORT_DIR = ROOT / "data" / "exports_excel"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

OMS_24H = 15.0
AQLI_REF = 32.5

FILES = {
    "proxy":    DATA / "pm25_proxy_era5.parquet",
    "features": DATA / "dataset_features.parquet",
    "targets":  DATA / "dataset_with_pm25_target.parquet",
    "firms":    DATA / "firms_fire_daily.parquet",
    "unc":      DATA / "pm25_with_uncertainty.parquet",
    "pred":     MODELS / "test_predictions_2025.parquet",
}

MONTH_NAMES = {
    1:"Janvier", 2:"Février", 3:"Mars", 4:"Avril", 5:"Mai", 6:"Juin",
    7:"Juillet", 8:"Août", 9:"Septembre", 10:"Octobre", 11:"Novembre", 12:"Décembre"
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def load(name):
    path = FILES[name]
    if not path.exists():
        print(f"  ✗ Fichier manquant : {path}")
        return None
    df = pd.read_parquet(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    return df


def alert_level(val):
    if val < OMS_24H:   return "Bon/Moyen"
    elif val < 35:      return "Mauvais"
    elif val < 55:      return "Très Mauvais"
    elif val < 75:      return "Dangereux"
    else:               return "Extrême"


def apply_pm25_colors(ws, df, pm25_col, writer):
    """Colorise les cellules PM2.5 selon le niveau d'alerte."""
    import xlsxwriter
    wb = writer.book
    col_idx = df.columns.get_loc(pm25_col) + 1  # 1-indexed for xlsxwriter

    fmt_bon     = wb.add_format({"bg_color": "#c6efce", "font_color": "#276221"})
    fmt_mauvais = wb.add_format({"bg_color": "#ffeb9c", "font_color": "#9c6500"})
    fmt_tres    = wb.add_format({"bg_color": "#ffc7ce", "font_color": "#9c0006"})
    fmt_danger  = wb.add_format({"bg_color": "#d62728", "font_color": "#ffffff", "bold": True})
    fmt_extreme = wb.add_format({"bg_color": "#7b0000", "font_color": "#ffffff", "bold": True})

    for row_i, val in enumerate(df[pm25_col], start=2):  # row 1 = header
        if pd.isna(val):
            continue
        if val < OMS_24H:       fmt = fmt_bon
        elif val < 35:          fmt = fmt_mauvais
        elif val < 55:          fmt = fmt_tres
        elif val < 75:          fmt = fmt_danger
        else:                   fmt = fmt_extreme
        ws.write(row_i - 1, col_idx - 1, round(val, 2), fmt)


def write_df_sheet(writer, df, sheet_name, freeze=True, color_col=None):
    """Écrit un DataFrame dans un onglet Excel avec mise en forme."""
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]
    wb = writer.book

    # Header format
    hdr_fmt = wb.add_format({
        "bold": True, "bg_color": "#2E4057", "font_color": "#FFFFFF",
        "border": 1, "align": "center", "valign": "vcenter"
    })
    for col_i, col_name in enumerate(df.columns):
        ws.write(0, col_i, col_name, hdr_fmt)
        # Auto-width
        max_len = max(len(str(col_name)), df[col_name].astype(str).str.len().max() if len(df) > 0 else 10)
        ws.set_column(col_i, col_i, min(max_len + 2, 40))

    if freeze:
        ws.freeze_panes(1, 0)

    # Color PM2.5 column
    if color_col and color_col in df.columns:
        try:
            apply_pm25_colors(ws, df, color_col, writer)
        except Exception:
            pass  # silently skip if xlsxwriter coloring fails


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT RÉSUMÉ CONSOLIDÉ (recommandé)
# ══════════════════════════════════════════════════════════════════════════════
def export_summary():
    """Génère PM25_RESUME.xlsx avec 7 onglets de synthèse."""
    out = EXPORT_DIR / "PM25_RESUME.xlsx"
    print(f"\n  Génération de {out.name} ...")

    proxy = load("proxy")
    pred  = load("pred")
    unc   = load("unc")

    if proxy is None:
        print("  ✗ Fichier proxy manquant, impossible de générer le résumé.")
        return

    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:

        # ── Onglet 1 : À propos ──────────────────────────────────────────────
        about = pd.DataFrame({
            "Onglet": [
                "A_propos", "Villes_Stats", "Mensuel_Villes",
                "Serie_Nationale", "Predictions_2025", "Incertitude_MC", "Alertes_OMS"
            ],
            "Description": [
                "Ce fichier — guide de lecture",
                "Statistiques PM2.5 résumées pour chaque ville (2020–2025)",
                "PM2.5 moyen par ville et par mois (heatmap format tabulaire)",
                "Évolution nationale mensuelle 2020–2025",
                "Prédictions ML XGBoost/LightGBM pour 2025",
                "Intervalles de confiance Monte Carlo (P5, P25, P75, P95)",
                "Nombre de jours en dépassement OMS par ville et par année",
            ],
            "Source": [
                "—",
                "pm25_proxy_era5.parquet",
                "pm25_proxy_era5.parquet",
                "pm25_proxy_era5.parquet",
                "models/test_predictions_2025.parquet",
                "pm25_with_uncertainty.parquet",
                "pm25_proxy_era5.parquet",
            ],
            "Lignes": ["—", "40", "40 × 12", "72", "14 160", "87 240", "40 × 6"],
        })
        write_df_sheet(writer, about, "A_propos", color_col=None)

        # ── Onglet 2 : Stats par ville ────────────────────────────────────────
        city_stats = proxy.groupby("city").agg(
            Région=("region", "first"),
            Latitude=("latitude", "first"),
            Longitude=("longitude", "first"),
            PM25_Moyenne=("pm25_proxy", "mean"),
            PM25_Mediane=("pm25_proxy", "median"),
            PM25_Max=("pm25_proxy", "max"),
            PM25_Min=("pm25_proxy", "min"),
            PM25_Std=("pm25_proxy", "std"),
            Jours_total=("pm25_proxy", "count"),
            Jours_dessus_OMS=("pm25_proxy", lambda x: (x > OMS_24H).sum()),
            Jours_dangereux_75=("pm25_proxy", lambda x: (x > 75).sum()),
        ).reset_index()
        city_stats["Pct_dessus_OMS"] = (city_stats["Jours_dessus_OMS"] / city_stats["Jours_total"] * 100).round(1)
        city_stats["Niveau_alerte"] = city_stats["PM25_Moyenne"].apply(alert_level)
        city_stats = city_stats.sort_values("PM25_Moyenne", ascending=False).reset_index(drop=True)
        city_stats.index += 1
        city_stats = city_stats.reset_index().rename(columns={"index": "Rang"})
        for col in ["PM25_Moyenne", "PM25_Mediane", "PM25_Max", "PM25_Min", "PM25_Std"]:
            city_stats[col] = city_stats[col].round(1)
        write_df_sheet(writer, city_stats, "Villes_Stats", color_col="PM25_Moyenne")
        print("    ✓ Onglet Villes_Stats")

        # ── Onglet 3 : Heatmap mensuelle ─────────────────────────────────────
        monthly = proxy.groupby(["city", "month"])["pm25_proxy"].mean().unstack()
        monthly.columns = [MONTH_NAMES[m] for m in monthly.columns]
        lat_order = proxy.groupby("city")["latitude"].first().sort_values(ascending=False)
        monthly = monthly.loc[lat_order.index]
        monthly = monthly.round(1).reset_index()
        monthly.insert(1, "Région", proxy.groupby("city")["region"].first().reindex(monthly["city"]).values)
        monthly.insert(2, "Moy_annuelle", monthly.iloc[:, 3:].mean(axis=1).round(1))
        write_df_sheet(writer, monthly, "Mensuel_Villes", color_col="Moy_annuelle")
        print("    ✓ Onglet Mensuel_Villes")

        # ── Onglet 4 : Série nationale ───────────────────────────────────────
        proxy["year_month"] = proxy["time"].dt.to_period("M")
        national = proxy.groupby("year_month").agg(
            PM25_Moyenne=("pm25_proxy", "mean"),
            PM25_Mediane=("pm25_proxy", "median"),
            PM25_P10=("pm25_proxy", lambda x: x.quantile(0.10)),
            PM25_P90=("pm25_proxy", lambda x: x.quantile(0.90)),
            Nb_observations=("pm25_proxy", "count"),
        ).reset_index()
        national["year_month"] = national["year_month"].astype(str)
        national["OMS_24h_ref"] = OMS_24H
        national["AQLI_ref"] = AQLI_REF
        for col in ["PM25_Moyenne", "PM25_Mediane", "PM25_P10", "PM25_P90"]:
            national[col] = national[col].round(1)
        national = national.rename(columns={"year_month": "Mois"})
        write_df_sheet(writer, national, "Serie_Nationale", color_col="PM25_Moyenne")
        print("    ✓ Onglet Serie_Nationale")

        # ── Onglet 5 : Prédictions ML 2025 ───────────────────────────────────
        if pred is not None:
            pred_out = pred.copy()
            pred_out["Erreur_abs"] = (pred_out["pm25_proxy"] - pred_out["pm25_pred_ensemble"]).abs().round(2)
            pred_out["Erreur_pct"] = (pred_out["Erreur_abs"] / pred_out["pm25_proxy"].clip(lower=1) * 100).round(1)
            pred_out["Niveau_reel"] = pred_out["pm25_proxy"].apply(alert_level)
            pred_out["Niveau_predit"] = pred_out["pm25_pred_ensemble"].apply(alert_level)
            for col in ["pm25_proxy", "pm25_pred_xgb", "pm25_pred_lgb", "pm25_pred_ensemble"]:
                pred_out[col] = pred_out[col].round(1)
            pred_out["time"] = pred_out["time"].dt.date
            pred_out = pred_out.rename(columns={
                "city": "Ville", "time": "Date",
                "pm25_proxy": "PM25_Reel", "pm25_pred_xgb": "PM25_XGBoost",
                "pm25_pred_lgb": "PM25_LightGBM", "pm25_pred_ensemble": "PM25_Ensemble"
            })
            write_df_sheet(writer, pred_out, "Predictions_2025", color_col="PM25_Reel")
            print("    ✓ Onglet Predictions_2025")
        else:
            pd.DataFrame({"Info": ["Fichier models/test_predictions_2025.parquet manquant"]}).to_excel(
                writer, sheet_name="Predictions_2025", index=False)

        # ── Onglet 6 : Incertitude Monte Carlo ────────────────────────────────
        if unc is not None:
            unc_out = unc.copy()
            unc_out["time"] = unc_out["time"].dt.date
            for col in ["pm25_proxy", "pm25_mc_mean", "pm25_mc_std",
                        "pm25_mc_p05", "pm25_mc_p25", "pm25_mc_p75", "pm25_mc_p95"]:
                if col in unc_out.columns:
                    unc_out[col] = unc_out[col].round(1)
            unc_out = unc_out.rename(columns={
                "city": "Ville", "time": "Date",
                "pm25_proxy": "PM25_Proxy", "pm25_mc_mean": "MC_Moyenne",
                "pm25_mc_std": "MC_Std", "pm25_mc_p05": "IC90_Bas",
                "pm25_mc_p25": "IC50_Bas", "pm25_mc_p75": "IC50_Haut",
                "pm25_mc_p95": "IC90_Haut"
            })
            write_df_sheet(writer, unc_out, "Incertitude_MC", color_col="PM25_Proxy")
            print("    ✓ Onglet Incertitude_MC")

        # ── Onglet 7 : Alertes OMS par année ─────────────────────────────────
        alertes = proxy.groupby(["city", "year"]).agg(
            Région=("region", "first"),
            PM25_Annuel=("pm25_proxy", "mean"),
            J_gt_OMS_15=("pm25_proxy", lambda x: (x > 15).sum()),
            J_gt_35=("pm25_proxy",    lambda x: (x > 35).sum()),
            J_gt_55=("pm25_proxy",    lambda x: (x > 55).sum()),
            J_gt_75=("pm25_proxy",    lambda x: (x > 75).sum()),
        ).reset_index()
        alertes["PM25_Annuel"] = alertes["PM25_Annuel"].round(1)
        alertes["Pct_J_gt_OMS"] = (alertes["J_gt_OMS_15"] / 365 * 100).round(1)
        alertes["Niveau_annuel"] = alertes["PM25_Annuel"].apply(alert_level)
        alertes = alertes.rename(columns={"city": "Ville", "year": "Année"})
        write_df_sheet(writer, alertes, "Alertes_OMS", color_col="PM25_Annuel")
        print("    ✓ Onglet Alertes_OMS")

    print(f"\n  ✅ Fichier généré : {out}")
    print(f"     Taille : {out.stat().st_size / 1024:.0f} Ko")


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT D'UN FICHIER BRUT
# ══════════════════════════════════════════════════════════════════════════════
def export_file(name):
    df = load(name)
    if df is None:
        return

    out = EXPORT_DIR / f"{name}.xlsx"
    print(f"\n  Export de '{name}' → {out.name}  ({len(df):,} lignes × {len(df.columns)} colonnes)")

    if len(df) > 500_000:
        print(f"  ⚠ Fichier très large. Export limité aux 200 000 premières lignes.")
        df = df.head(200_000)

    if "time" in df.columns:
        df["time"] = df["time"].dt.date

    # Round floats
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(3)

    pm_col = "pm25_proxy" if "pm25_proxy" in df.columns else None

    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        write_df_sheet(writer, df, name, color_col=pm_col)

    print(f"  ✅ Fichier généré : {out}  ({out.stat().st_size / 1024:.0f} Ko)")


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT DONNÉES D'UNE VILLE
# ══════════════════════════════════════════════════════════════════════════════
def export_city(city_name):
    proxy = load("proxy")
    pred  = load("pred")
    unc   = load("unc")

    if proxy is None:
        return

    # Recherche insensible à la casse
    all_cities = proxy["city"].unique()
    matches = [c for c in all_cities if city_name.lower() in c.lower()]
    if not matches:
        print(f"  ✗ Ville '{city_name}' introuvable. Villes disponibles : {sorted(all_cities)}")
        return
    city = matches[0]
    safe_name = city.replace(" ", "_").replace("-", "_")

    out = EXPORT_DIR / f"ville_{safe_name}.xlsx"
    print(f"\n  Export de la ville '{city}' → {out.name}")

    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:

        # Onglet 1 : Données journalières complètes
        city_df = proxy[proxy["city"] == city].copy()
        city_df["time"] = city_df["time"].dt.date
        float_cols = city_df.select_dtypes("float").columns
        city_df[float_cols] = city_df[float_cols].round(2)
        city_df["Niveau_alerte"] = city_df["pm25_proxy"].apply(alert_level)
        write_df_sheet(writer, city_df, "Données_journalières", color_col="pm25_proxy")
        print(f"    ✓ Données_journalières ({len(city_df)} lignes)")

        # Onglet 2 : Résumé mensuel
        monthly = city_df.copy()
        monthly["month"] = pd.to_datetime(monthly["time"]).dt.month
        monthly["year"]  = pd.to_datetime(monthly["time"]).dt.year
        monthly_agg = monthly.groupby(["year", "month"]).agg(
            PM25_Moy=("pm25_proxy", "mean"),
            PM25_Max=("pm25_proxy", "max"),
            PM25_Min=("pm25_proxy", "min"),
            Jours_gt_OMS=("pm25_proxy", lambda x: (x > OMS_24H).sum()),
            Jours_gt_75=("pm25_proxy",  lambda x: (x > 75).sum()),
            Nb_jours=("pm25_proxy", "count"),
        ).reset_index()
        monthly_agg["Mois_nom"] = monthly_agg["month"].map(MONTH_NAMES)
        monthly_agg["Niveau"] = monthly_agg["PM25_Moy"].apply(alert_level)
        for col in ["PM25_Moy", "PM25_Max", "PM25_Min"]:
            monthly_agg[col] = monthly_agg[col].round(1)
        write_df_sheet(writer, monthly_agg, "Résumé_mensuel", color_col="PM25_Moy")
        print(f"    ✓ Résumé_mensuel")

        # Onglet 3 : Prédictions ML 2025
        if pred is not None:
            city_pred = pred[pred["city"] == city].copy()
            if not city_pred.empty:
                city_pred["time"] = city_pred["time"].dt.date
                city_pred["Erreur"] = (city_pred["pm25_proxy"] - city_pred["pm25_pred_ensemble"]).round(2)
                for col in ["pm25_proxy", "pm25_pred_xgb", "pm25_pred_lgb", "pm25_pred_ensemble"]:
                    city_pred[col] = city_pred[col].round(1)
                write_df_sheet(writer, city_pred, "Prédictions_2025", color_col="pm25_proxy")
                print(f"    ✓ Prédictions_2025 ({len(city_pred)} lignes)")

        # Onglet 4 : Incertitude MC
        if unc is not None:
            city_unc = unc[unc["city"] == city].copy()
            if not city_unc.empty:
                city_unc["time"] = city_unc["time"].dt.date
                float_cols_unc = city_unc.select_dtypes("float").columns
                city_unc[float_cols_unc] = city_unc[float_cols_unc].round(1)
                write_df_sheet(writer, city_unc, "Incertitude_MC", color_col="pm25_proxy")
                print(f"    ✓ Incertitude_MC")

    print(f"\n  ✅ Fichier généré : {out}  ({out.stat().st_size / 1024:.0f} Ko)")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Exporter les données PM2.5 en Excel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python notebooks/export_to_excel.py                  # Résumé consolidé (recommandé)
  python notebooks/export_to_excel.py --summary        # Idem
  python notebooks/export_to_excel.py --all            # Tous les fichiers séparément
  python notebooks/export_to_excel.py --file proxy     # Données proxy ERA5 brutes
  python notebooks/export_to_excel.py --file pred      # Prédictions ML 2025
  python notebooks/export_to_excel.py --city Maroua    # Fichier dédié à Maroua
  python notebooks/export_to_excel.py --city Douala    # Fichier dédié à Douala
        """
    )
    parser.add_argument("--summary", action="store_true", help="Résumé consolidé (7 onglets)")
    parser.add_argument("--all", action="store_true", help="Tous les fichiers séparément")
    parser.add_argument("--file", choices=list(FILES.keys()), help="Un fichier précis")
    parser.add_argument("--city", help="Données d'une ville spécifique")

    args = parser.parse_args()

    print(f"\n{'═' * 60}")
    print(f"  Export Excel — PM2.5 Cameroun (IndabaX 2026)")
    print(f"  Destination : {EXPORT_DIR}")
    print(f"{'═' * 60}")

    # Aucun argument → résumé consolidé par défaut
    if not any([args.summary, args.all, args.file, args.city]):
        export_summary()
        return

    if args.summary:
        export_summary()

    if args.all:
        for name in FILES.keys():
            if FILES[name].exists():
                export_file(name)

    if args.file:
        export_file(args.file)

    if args.city:
        export_city(args.city)

    print(f"\n  Tous les fichiers sont dans : {EXPORT_DIR}\n")


if __name__ == "__main__":
    main()

"""
Modèle prédictif PM2.5 — XGBoost + LightGBM avec validation temporelle
Hackathon IndabaX Cameroun 2026

Architecture de validation :
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expanding-window cross-validation (jamais de mélange aléatoire) :
  Fold 1  : Train 2020        → Val 2021
  Fold 2  : Train 2020–2021   → Val 2022
  Fold 3  : Train 2020–2022   → Val 2023
  Fold 4  : Train 2020–2023   → Val 2024
  Test final : Train 2020–2024 → Test 2025

Pourquoi expanding-window ?
  - Les séries temporelles ont une dépendance temporelle (autocorrélation)
  - Le shuffle aléatoire crée une fuite d'information (data leakage)
  - L'expanding-window simule la vraie mise en production (on prédit l'avenir)

Features utilisées (87 240 × 144 — dataset_features.parquet) :
  - Météo ERA5 : BLH, précipitations, vent, RH, température (30 features)
  - Facteurs proxy : F_stagnation, F_wet, F_wind, F_harmattan, F_hygro
  - Lags PM2.5 : lag1, lag2, lag3, lag7, lag14 (autorégressif)
  - Rolling stats : fenêtres 3/7/14/30 jours
  - Encodages cycliques : mois, jour de l'année, jour de la semaine
  - Features spatiales : latitude, longitude, région, zone climatique
  - Interactions : wind_x_harmattan, blh_x_precip, etc.

Cible :
  pm25_proxy (µg/m³) — variable cible construite dans 02_build_pm25_target.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Chemins ────────────────────────────────────────────────────────────────────
FEATURES_FILE = Path("data/dataset_features.parquet")
OUTPUT_DIR    = Path("models")
FIG_DIR       = Path("data")

# ── Configuration modèle ───────────────────────────────────────────────────────
TARGET = "pm25_proxy"

# Features à EXCLURE : target, identifiants, features dérivées de la target
EXCLUDE_COLS = {
    TARGET, "city", "time", "region",
    # Facteurs du proxy (ils sont dans la target, pas indépendants)
    "F_stagnation", "F_wet", "F_wind", "F_harmattan", "F_hygro", "F_fire",
    "C_base", "f_fire",
    # Colonnes redondantes ou non-features
    "is_dry_season",  # déjà encodé dans d'autres features
    # CRITIQUE : pm25_log = log1p(pm25_proxy) — transform directe de la target
    # L'inclure comme feature serait une fuite parfaite (leakage)
    "pm25_log",
}

# Préfixes des features lags/rolling du proxy — retirés pour le modèle météo-only
LAG_PREFIXES = ("pm25_proxy_lag", "pm25_proxy_rm")

# Folds expanding-window
FOLDS = [
    {"name": "Fold1", "train_end": 2020, "val_year": 2021},
    {"name": "Fold2", "train_end": 2021, "val_year": 2022},
    {"name": "Fold3", "train_end": 2022, "val_year": 2023},
    {"name": "Fold4", "train_end": 2023, "val_year": 2024},
]
TEST_YEAR = 2025


# ══════════════════════════════════════════════════════════════════════════════
# 1. Chargement et préparation des données
# ══════════════════════════════════════════════════════════════════════════════

def load_features(path: Path) -> pd.DataFrame:
    """Charge le dataset features et prépare les types."""
    log.info(f"Chargement features : {path}")
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    log.info(f"Dataset : {df.shape} | {df['city'].nunique()} villes | {df['year'].nunique()} années")
    log.info(f"Période : {df['time'].min().date()} → {df['time'].max().date()}")
    return df


def select_features(df: pd.DataFrame) -> list[str]:
    """
    Sélectionne les colonnes features (toutes sauf exclusions et non-numériques).
    """
    feature_cols = []
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        if df[col].dtype in [object, "category"]:
            continue
        if df[col].isna().mean() > 0.5:   # exclure colonnes >50% NaN
            continue
        feature_cols.append(col)

    log.info(f"Features sélectionnées : {len(feature_cols)}")
    return sorted(feature_cols)


def prepare_fold(df: pd.DataFrame, train_end_year: int, val_year: int,
                 feature_cols: list[str]):
    """
    Prépare les splits train/val pour un fold expanding-window.
    Train : toutes les années ≤ train_end_year
    Val   : val_year uniquement
    """
    train_mask = df["year"] <= train_end_year
    val_mask   = df["year"] == val_year

    X_train = df.loc[train_mask, feature_cols].fillna(0)
    y_train = df.loc[train_mask, TARGET]
    X_val   = df.loc[val_mask,   feature_cols].fillna(0)
    y_val   = df.loc[val_mask,   TARGET]

    return X_train, y_train, X_val, y_val


# ══════════════════════════════════════════════════════════════════════════════
# 2. Métriques d'évaluation
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, name: str = "") -> dict:
    """Calcule RMSE, MAE, R², MAPE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

    if name:
        log.info(f"  [{name}] RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}  MAPE={mape:.1f}%")

    return {"name": name, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


# ══════════════════════════════════════════════════════════════════════════════
# 3. Entraînement XGBoost
# ══════════════════════════════════════════════════════════════════════════════

def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Entraîne XGBoost avec early stopping sur le fold de validation.

    Hyperparamètres motivés :
    - n_estimators=2000 : beaucoup d'arbres + early stopping
    - learning_rate=0.05 : faible pour regularisation naturelle
    - max_depth=6 : profondeur modérée (évite overfitting)
    - subsample=0.8 + colsample=0.8 : bagging style stochastic gradient
    - reg_alpha=0.1 + reg_lambda=1.0 : L1+L2 regularisation
    - tree_method='hist' : rapide sur CPU
    """
    try:
        import xgboost as xgb
    except ImportError:
        log.error("XGBoost non installé : conda run -n hackathon_pm25 pip install xgboost")
        return None

    model = xgb.XGBRegressor(
        n_estimators          = 2000,
        learning_rate         = 0.05,
        max_depth             = 6,
        min_child_weight      = 3,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        reg_alpha             = 0.1,
        reg_lambda            = 1.0,
        gamma                 = 0.1,
        tree_method           = "hist",
        device                = "cpu",
        random_state          = 42,
        n_jobs                = -1,
        verbosity             = 0,
        early_stopping_rounds = 50,   # XGBoost 2.x : paramètre du constructeur
    )

    model.fit(
        X_train, y_train,
        eval_set = [(X_val, y_val)],
        verbose  = False,
    )

    log.info(f"  XGBoost : best_iteration={model.best_iteration}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. Entraînement LightGBM
# ══════════════════════════════════════════════════════════════════════════════

def train_lightgbm(X_train, y_train, X_val, y_val):
    """
    Entraîne LightGBM — souvent plus rapide et parfois plus précis que XGBoost
    sur les datasets tabulaires avec beaucoup de features.

    LightGBM utilise le leaf-wise growth (vs level-wise pour XGBoost)
    → plus efficace mais besoin de contrôler num_leaves.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        log.warning("LightGBM non installé : conda run -n hackathon_pm25 pip install lightgbm")
        return None

    model = lgb.LGBMRegressor(
        n_estimators      = 2000,
        learning_rate     = 0.05,
        num_leaves        = 63,       # 2^max_depth - 1 = 63 pour depth~6
        max_depth         = -1,       # LightGBM gère via num_leaves
        min_child_samples = 20,
        subsample         = 0.8,
        subsample_freq    = 1,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1,
    )

    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False),
                 lgb.log_evaluation(period=-1)]

    model.fit(
        X_train, y_train,
        eval_set   = [(X_val, y_val)],
        callbacks  = callbacks,
    )

    log.info(f"  LightGBM : best_iteration={model.best_iteration_}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 5. Cross-validation expanding-window
# ══════════════════════════════════════════════════════════════════════════════

def run_cross_validation(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Exécute la validation croisée expanding-window pour XGBoost et LightGBM.
    Retourne les métriques par fold et les modèles finaux.
    """
    log.info("=" * 65)
    log.info("Validation croisée expanding-window")
    log.info("=" * 65)

    results_xgb = []
    results_lgb = []
    models_xgb  = []
    models_lgb  = []

    for fold in FOLDS:
        log.info(f"\n{'─'*45}")
        log.info(f"  {fold['name']} : Train ≤{fold['train_end']} → Val {fold['val_year']}")

        X_train, y_train, X_val, y_val = prepare_fold(
            df, fold["train_end"], fold["val_year"], feature_cols
        )
        log.info(f"  Train : {X_train.shape[0]:,} lignes | Val : {X_val.shape[0]:,} lignes")

        # XGBoost
        log.info("  → XGBoost...")
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
        if xgb_model:
            pred_xgb = xgb_model.predict(X_val)
            m = compute_metrics(y_val, pred_xgb, f"XGB {fold['name']}")
            m["fold"] = fold["name"]
            results_xgb.append(m)
            models_xgb.append(xgb_model)

        # LightGBM
        log.info("  → LightGBM...")
        lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
        if lgb_model:
            pred_lgb = lgb_model.predict(X_val)
            m = compute_metrics(y_val, pred_lgb, f"LGB {fold['name']}")
            m["fold"] = fold["name"]
            results_lgb.append(m)
            models_lgb.append(lgb_model)

    # Résumé CV
    if results_xgb:
        avg_xgb = {
            "rmse": np.mean([r["rmse"] for r in results_xgb]),
            "mae":  np.mean([r["mae"]  for r in results_xgb]),
            "r2":   np.mean([r["r2"]   for r in results_xgb]),
            "mape": np.mean([r["mape"] for r in results_xgb]),
        }
        log.info(f"\nXGBoost CV moyen : RMSE={avg_xgb['rmse']:.2f}  MAE={avg_xgb['mae']:.2f}  R²={avg_xgb['r2']:.4f}")

    if results_lgb:
        avg_lgb = {
            "rmse": np.mean([r["rmse"] for r in results_lgb]),
            "mae":  np.mean([r["mae"]  for r in results_lgb]),
            "r2":   np.mean([r["r2"]   for r in results_lgb]),
            "mape": np.mean([r["mape"] for r in results_lgb]),
        }
        log.info(f"LightGBM CV moyen : RMSE={avg_lgb['rmse']:.2f}  MAE={avg_lgb['mae']:.2f}  R²={avg_lgb['r2']:.4f}")

    return {
        "results_xgb": results_xgb,
        "results_lgb": results_lgb,
        "models_xgb": models_xgb,
        "models_lgb": models_lgb,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. Modèle final + test 2025
# ══════════════════════════════════════════════════════════════════════════════

def train_final_and_test(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Entraîne le modèle final sur toutes les données 2020-2024,
    évalue sur 2025 (test set tenu à l'écart).
    """
    log.info("=" * 65)
    log.info("Modèle final : Train 2020–2024 → Test 2025")
    log.info("=" * 65)

    train_mask = df["year"] <= 2024
    test_mask  = df["year"] == TEST_YEAR

    X_train = df.loc[train_mask, feature_cols].fillna(0)
    y_train = df.loc[train_mask, TARGET]
    X_test  = df.loc[test_mask,  feature_cols].fillna(0)
    y_test  = df.loc[test_mask,  TARGET]

    log.info(f"Train : {X_train.shape[0]:,} | Test : {X_test.shape[0]:,}")

    # On utilise les 10% finaux de train comme validation pour early stopping
    n_val   = max(1000, int(0.1 * len(X_train)))
    X_tr    = X_train.iloc[:-n_val]
    y_tr    = y_train.iloc[:-n_val]
    X_es    = X_train.iloc[-n_val:]
    y_es    = y_train.iloc[-n_val:]

    final_models = {}

    # XGBoost final
    log.info("→ XGBoost final...")
    xgb_final = train_xgboost(X_tr, y_tr, X_es, y_es)
    if xgb_final:
        pred_test_xgb = xgb_final.predict(X_test)
        m_xgb = compute_metrics(y_test, pred_test_xgb, "XGB Test 2025")
        final_models["xgb"] = {"model": xgb_final, "metrics": m_xgb,
                                "predictions": pred_test_xgb}

    # LightGBM final
    log.info("→ LightGBM final...")
    lgb_final = train_lightgbm(X_tr, y_tr, X_es, y_es)
    if lgb_final:
        pred_test_lgb = lgb_final.predict(X_test)
        m_lgb = compute_metrics(y_test, pred_test_lgb, "LGB Test 2025")
        final_models["lgb"] = {"model": lgb_final, "metrics": m_lgb,
                                "predictions": pred_test_lgb}

    # Ensemble (moyenne simple) si les deux sont disponibles
    if "xgb" in final_models and "lgb" in final_models:
        pred_ensemble = 0.5 * pred_test_xgb + 0.5 * pred_test_lgb
        m_ens = compute_metrics(y_test, pred_ensemble, "Ensemble Test 2025")
        final_models["ensemble"] = {"metrics": m_ens, "predictions": pred_ensemble}

    # Importance des features (XGBoost)
    if "xgb" in final_models:
        importances = pd.Series(
            xgb_final.feature_importances_,
            index=feature_cols
        ).sort_values(ascending=False)
        final_models["feature_importance"] = importances
        log.info("\nTop 15 features les plus importantes (XGBoost):")
        for feat, imp in importances.head(15).items():
            log.info(f"  {feat:<40} : {imp:.4f}")

    # Ajouter les prédictions au dataset test
    df_test = df.loc[test_mask].copy()
    if "xgb" in final_models:
        df_test["pm25_pred_xgb"] = final_models["xgb"]["predictions"]
    if "lgb" in final_models:
        df_test["pm25_pred_lgb"] = final_models["lgb"]["predictions"]
    if "ensemble" in final_models:
        df_test["pm25_pred_ensemble"] = final_models["ensemble"]["predictions"]
    final_models["df_test_predictions"] = df_test

    return final_models


# ══════════════════════════════════════════════════════════════════════════════
# 7. Visualisations
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(cv_results: dict, final_models: dict, df: pd.DataFrame,
                 feature_cols: list[str]) -> None:
    """Figures de diagnostic : CV, scatter, importance, séries temporelles."""

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("PM2.5 Prediction — XGBoost + LightGBM (Hackathon IndabaX 2026)",
                 fontsize=13, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── A. Métriques CV par fold ──────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, :2])
    folds_names = [r["fold"] for r in cv_results.get("results_xgb", [])]
    if folds_names:
        x = np.arange(len(folds_names))
        w = 0.35
        rmse_xgb = [r["rmse"] for r in cv_results["results_xgb"]]
        rmse_lgb = [r["rmse"] for r in cv_results.get("results_lgb", [])]
        ax_a.bar(x - w/2, rmse_xgb, w, label="XGBoost", color="steelblue", alpha=0.8)
        if rmse_lgb:
            ax_a.bar(x + w/2, rmse_lgb, w, label="LightGBM", color="tomato", alpha=0.8)
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(folds_names)
        ax_a.set_ylabel("RMSE (µg/m³)")
        ax_a.set_title("A. RMSE par fold (expanding-window CV)")
        ax_a.legend()
        # Annoter R²
        for i, (r_xgb, r_lgb) in enumerate(zip(
            cv_results["results_xgb"],
            cv_results.get("results_lgb", [{}]*len(folds_names))
        )):
            ax_a.text(i - w/2, r_xgb["rmse"] + 0.3, f'R²={r_xgb["r2"]:.3f}',
                      ha="center", fontsize=7, color="steelblue")
            if r_lgb:
                ax_a.text(i + w/2, r_lgb["rmse"] + 0.3, f'R²={r_lgb["r2"]:.3f}',
                          ha="center", fontsize=7, color="tomato")

    # ── B. Scatter prédictions vs réel (test 2025) ────────────────────────────
    ax_b = fig.add_subplot(gs[0, 2])
    df_test = final_models.get("df_test_predictions", pd.DataFrame())
    if not df_test.empty and "pm25_pred_xgb" in df_test.columns:
        y_true = df_test[TARGET]
        y_pred = df_test.get("pm25_pred_ensemble", df_test["pm25_pred_xgb"])
        lim    = max(y_true.max(), y_pred.max()) * 1.05
        ax_b.scatter(y_true, y_pred, alpha=0.15, s=5, color="steelblue")
        ax_b.plot([0, lim], [0, lim], "r--", lw=1.5, label="y=x")
        r2 = final_models.get("ensemble", final_models.get("xgb", {})).get(
            "metrics", {}).get("r2", 0)
        ax_b.set_xlabel("PM2.5 réel (µg/m³)")
        ax_b.set_ylabel("PM2.5 prédit (µg/m³)")
        ax_b.set_title(f"B. Scatter Test 2025 (R²={r2:.3f})")
        ax_b.legend(fontsize=8)

    # ── C. Importance des features (top 20) ────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, :2])
    if "feature_importance" in final_models:
        imp = final_models["feature_importance"].head(20)
        colors = ["tomato" if "pm25_proxy" in f or "lag" in f
                  else "steelblue" for f in imp.index]
        ax_c.barh(range(len(imp)), imp.values[::-1], color=colors[::-1], alpha=0.8)
        ax_c.set_yticks(range(len(imp)))
        ax_c.set_yticklabels(imp.index[::-1], fontsize=8)
        ax_c.set_xlabel("Importance XGBoost (gain)")
        ax_c.set_title("C. Top 20 features importantes")

    # ── D. Série temporelle : Yaoundé (test 2025) ────────────────────────────
    ax_d = fig.add_subplot(gs[1, 2])
    if not df_test.empty and "pm25_pred_xgb" in df_test.columns:
        df_yao = df_test[df_test["city"] == "Yaounde"].copy()
        if not df_yao.empty:
            df_yao = df_yao.sort_values("time")
            ax_d.plot(df_yao["time"], df_yao[TARGET], "k-", lw=1.5,
                      alpha=0.8, label="Réel")
            ax_d.plot(df_yao["time"], df_yao.get("pm25_pred_ensemble",
                       df_yao["pm25_pred_xgb"]),
                      "r--", lw=1.5, alpha=0.8, label="Prédit")
            ax_d.set_ylabel("PM2.5 (µg/m³)")
            ax_d.set_title("D. Yaoundé — Test 2025")
            ax_d.legend(fontsize=8)
            ax_d.tick_params(axis="x", labelsize=7)

    # ── E. Erreur résiduelle par mois ──────────────────────────────────────────
    ax_e = fig.add_subplot(gs[2, :2])
    if not df_test.empty and "pm25_pred_xgb" in df_test.columns:
        df_test = df_test.copy()
        df_test["month"] = pd.to_datetime(df_test["time"]).dt.month
        pred_col = "pm25_pred_ensemble" if "pm25_pred_ensemble" in df_test.columns \
                   else "pm25_pred_xgb"
        df_test["residual"] = df_test[pred_col] - df_test[TARGET]
        monthly_res = df_test.groupby("month")["residual"].agg(["mean", "std"])
        months = range(1, 13)
        ax_e.bar(months, monthly_res["mean"].reindex(months, fill_value=0),
                 yerr=monthly_res["std"].reindex(months, fill_value=0),
                 color=["tomato" if v > 0 else "steelblue"
                        for v in monthly_res["mean"].reindex(months, fill_value=0)],
                 alpha=0.8, capsize=3)
        ax_e.axhline(0, color="black", lw=1)
        ax_e.set_xticks(range(1, 13))
        ax_e.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
        ax_e.set_ylabel("Résidu moyen (prédit − réel) µg/m³")
        ax_e.set_title("E. Biais mensuel (Test 2025)")

    # ── F. RMSE par ville (heatmap barres) ───────────────────────────────────
    ax_f = fig.add_subplot(gs[2, 2])
    if not df_test.empty and "pm25_pred_xgb" in df_test.columns:
        pred_col = "pm25_pred_ensemble" if "pm25_pred_ensemble" in df_test.columns \
                   else "pm25_pred_xgb"
        city_rmse = df_test.groupby("city").apply(
            lambda x: np.sqrt(mean_squared_error(x[TARGET], x[pred_col]))
        ).sort_values(ascending=True)
        city_lat = df_test.groupby("city")["latitude"].first()
        colors_c = plt.cm.RdYlGn_r(
            (city_rmse - city_rmse.min()) / (city_rmse.max() - city_rmse.min() + 1e-8)
        )
        ax_f.barh(range(len(city_rmse)), city_rmse.values, color=colors_c, alpha=0.8)
        ax_f.set_yticks(range(len(city_rmse)))
        ax_f.set_yticklabels(city_rmse.index, fontsize=6)
        ax_f.set_xlabel("RMSE (µg/m³)")
        ax_f.set_title("F. RMSE par ville (Test 2025)")

    plt.tight_layout()
    fig_path = FIG_DIR / "model_diagnostics.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    log.info(f"Figure sauvegardée : {fig_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7b. Ablation Study : Full vs Météo-only vs Persistence
# ══════════════════════════════════════════════════════════════════════════════

def get_meteo_only_features(feature_cols: list[str]) -> list[str]:
    """Retourne les features sans aucun lag/rolling dérivé du proxy PM2.5."""
    return [f for f in feature_cols
            if not any(f.startswith(p) for p in LAG_PREFIXES)]


def run_persistence_baseline(df: pd.DataFrame) -> dict:
    """
    Baseline de persistence : PM2.5[t] = PM2.5[t-1].
    Utilise la colonne pm25_proxy_lag1 — valeur du jour précédent,
    disponible pour chaque ligne sans regarder dans le futur.
    """
    test_mask = df["year"] == TEST_YEAR
    y_true = df.loc[test_mask, TARGET]
    if "pm25_proxy_lag1" in df.columns:
        y_pred = df.loc[test_mask, "pm25_proxy_lag1"]
    else:
        y_pred = y_true.shift(1)
    valid = y_true.notna() & y_pred.notna()
    return compute_metrics(y_true[valid], y_pred[valid], "Persistence Test 2025")


def run_ablation_study(df: pd.DataFrame, feature_cols: list[str],
                       full_test_metrics: dict) -> dict:
    """
    Compare 3 approches sur Test 2025 :
      1. Persistence — PM2.5[t] = PM2.5[t-1]  (aucun ML)
      2. Météo-only  — XGBoost sans lags PM2.5
      3. Full model  — XGBoost avec tous les features (déjà calculé)
    """
    log.info("=" * 65)
    log.info("ABLATION STUDY : Full vs Météo-only vs Persistence")
    log.info("=" * 65)

    ablation = {}

    # 1. Persistence
    log.info("\n[1/3] Baseline persistence (PM2.5[t-1])...")
    ablation["persistence"] = run_persistence_baseline(df)

    # 2. Météo-only
    meteo_features = get_meteo_only_features(feature_cols)
    n_removed = len(feature_cols) - len(meteo_features)
    log.info(f"\n[2/3] Météo-only XGBoost "
             f"({len(meteo_features)} features, {n_removed} lags retirés)...")

    train_mask = df["year"] <= 2024
    test_mask  = df["year"] == TEST_YEAR

    X_train_m = df.loc[train_mask, meteo_features].fillna(0)
    y_train_m = df.loc[train_mask, TARGET]
    X_test_m  = df.loc[test_mask,  meteo_features].fillna(0)
    y_test_m  = df.loc[test_mask,  TARGET]

    n_val = max(1000, int(0.1 * len(X_train_m)))
    xgb_meteo = train_xgboost(
        X_train_m.iloc[:-n_val], y_train_m.iloc[:-n_val],
        X_train_m.iloc[-n_val:], y_train_m.iloc[-n_val:]
    )
    if xgb_meteo:
        pred_meteo = xgb_meteo.predict(X_test_m)
        ablation["meteo_only"] = compute_metrics(y_test_m, pred_meteo,
                                                  "Météo-only Test 2025")
        ablation["meteo_features_n"]  = len(meteo_features)
        ablation["lag_features_n"]    = n_removed

        # Importance météo-only (top 10)
        imp_m = pd.Series(xgb_meteo.feature_importances_,
                          index=meteo_features).sort_values(ascending=False)
        ablation["meteo_importance"] = imp_m
        log.info("  Top 10 features météo-only :")
        for feat, imp in imp_m.head(10).items():
            log.info(f"    {feat:<40} : {imp:.4f}")

    # 3. Full model (résultats déjà calculés)
    log.info("\n[3/3] Full model (résultats existants)...")
    ablation["full"] = full_test_metrics

    # ── Résumé comparatif ──────────────────────────────────────────────────
    log.info("\n" + "═" * 65)
    log.info("  RÉSUMÉ ABLATION — Test 2025")
    log.info(f"  {'Modèle':<26} {'RMSE':>7} {'MAE':>7} {'R²':>8} {'MAPE':>7}")
    log.info("  " + "─" * 55)
    for label, key in [("Persistence (lag1)",   "persistence"),
                        ("Météo-only XGBoost",   "meteo_only"),
                        ("Full XGBoost",         "full")]:
        if key in ablation and ablation[key]:
            m = ablation[key]
            log.info(f"  {label:<26} {m['rmse']:>7.2f} {m['mae']:>7.2f} "
                     f"{m['r2']:>8.4f} {m['mape']:>6.1f}%")

    # Gains incrémentaux
    if all(k in ablation for k in ["persistence", "meteo_only", "full"]):
        dr2_m = ablation["meteo_only"]["r2"] - ablation["persistence"]["r2"]
        dr2_f = ablation["full"]["r2"] - ablation["meteo_only"]["r2"]
        log.info(f"\n  ΔR²  météo vs persistence  : {dr2_m:+.4f}")
        log.info(f"  ΔR²  full  vs météo-only   : {dr2_f:+.4f}")
        log.info(f"  → Les lags ajoutent {dr2_f/(dr2_m+dr2_f)*100:.0f}% du gain total de R²")
    log.info("═" * 65)

    return ablation


def plot_ablation(ablation: dict) -> None:
    """Figure comparative 3-panels : RMSE, R², MAPE pour les 3 modèles."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Ablation Study — Contribution des lags PM2.5 vs Météo pure (Test 2025)",
        fontsize=12, fontweight="bold"
    )

    labels, rmse_v, r2_v, mape_v = [], [], [], []
    palette = ["#95a5a6", "#3498db", "#e74c3c"]

    for label, key in [("Persistence\n(PM2.5[t-1])",  "persistence"),
                        ("Météo-only\n(sans lags)",     "meteo_only"),
                        ("Full model\n(lags + météo)",  "full")]:
        if key in ablation and ablation[key]:
            labels.append(label)
            rmse_v.append(ablation[key]["rmse"])
            r2_v.append(ablation[key]["r2"])
            mape_v.append(ablation[key]["mape"])

    x = np.arange(len(labels))
    cols = palette[:len(labels)]

    for ax, vals, ylabel, title, fmt in [
        (axes[0], rmse_v, "RMSE (µg/m³)", "RMSE ↓", "{:.1f}"),
        (axes[1], r2_v,   "R²",            "R² ↑",   "{:.3f}"),
        (axes[2], mape_v, "MAPE (%)",      "MAPE ↓", "{:.1f}%"),
    ]:
        bars = ax.bar(x, vals, color=cols, alpha=0.85, edgecolor="white", width=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    fmt.format(v), ha="center", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylabel == "R²":
            ax.set_ylim(0, 1.08)

    plt.tight_layout()
    path = FIG_DIR / "ablation_study.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log.info(f"Figure ablation : {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 8. Sauvegarde des modèles
# ══════════════════════════════════════════════════════════════════════════════

def save_models(final_models: dict, cv_results: dict) -> None:
    """Sauvegarde les modèles et les métriques."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import joblib

        if "xgb" in final_models:
            joblib.dump(final_models["xgb"]["model"], OUTPUT_DIR / "xgboost_final.pkl")
            log.info(f"XGBoost sauvegardé : {OUTPUT_DIR}/xgboost_final.pkl")

        if "lgb" in final_models:
            joblib.dump(final_models["lgb"]["model"], OUTPUT_DIR / "lightgbm_final.pkl")
            log.info(f"LightGBM sauvegardé : {OUTPUT_DIR}/lightgbm_final.pkl")

    except ImportError:
        log.warning("joblib non disponible — modèles non sauvegardés")

    # Métriques CV au format parquet
    rows = cv_results.get("results_xgb", []) + cv_results.get("results_lgb", [])
    if rows:
        df_cv = pd.DataFrame(rows)
        df_cv.to_parquet(OUTPUT_DIR / "cv_metrics.parquet", index=False)

    # Prédictions test
    df_test = final_models.get("df_test_predictions")
    if df_test is not None and not df_test.empty:
        cols = ["city", "time", TARGET] + [c for c in df_test.columns
                                            if "pm25_pred" in c]
        df_test[cols].to_parquet(OUTPUT_DIR / "test_predictions_2025.parquet", index=False)
        log.info(f"Prédictions test 2025 sauvegardées : {OUTPUT_DIR}/test_predictions_2025.parquet")

    # Rapport texte
    report_lines = ["# Rapport modèle PM2.5 — Hackathon IndabaX 2026\n"]
    for model_name, key in [("XGBoost", "xgb"), ("LightGBM", "lgb"), ("Ensemble", "ensemble")]:
        if key in final_models and "metrics" in final_models[key]:
            m = final_models[key]["metrics"]
            report_lines.append(f"\n## {model_name} — Test 2025")
            report_lines.append(f"- RMSE : {m['rmse']:.3f} µg/m³")
            report_lines.append(f"- MAE  : {m['mae']:.3f} µg/m³")
            report_lines.append(f"- R²   : {m['r2']:.4f}")
            report_lines.append(f"- MAPE : {m['mape']:.1f}%")

    report_lines.append("\n## Features (top 15 XGBoost)\n")
    if "feature_importance" in final_models:
        for feat, imp in final_models["feature_importance"].head(15).items():
            report_lines.append(f"- {feat} : {imp:.4f}")

    with open(OUTPUT_DIR / "model_report.md", "w") as f:
        f.write("\n".join(report_lines))
    log.info(f"Rapport sauvegardé : {OUTPUT_DIR}/model_report.md")


def append_ablation_to_report(ablation: dict) -> None:
    """Ajoute les résultats de l'ablation study au rapport texte."""
    path = OUTPUT_DIR / "model_report.md"
    lines = ["\n\n## Ablation Study — Test 2025\n",
             "Comparaison : Persistence vs Météo-only vs Full model\n",
             "| Modèle | RMSE | MAE | R² | MAPE |",
             "|--------|------|-----|-----|------|"]
    for label, key in [("Persistence (PM2.5[t-1])", "persistence"),
                        ("Météo-only XGBoost",        "meteo_only"),
                        ("Full XGBoost",              "full")]:
        if key in ablation and ablation[key]:
            m = ablation[key]
            lines.append(f"| {label} | {m['rmse']:.2f} | {m['mae']:.2f} "
                         f"| {m['r2']:.4f} | {m['mape']:.1f}% |")

    if all(k in ablation for k in ["persistence", "meteo_only", "full"]):
        dr2_m = ablation["meteo_only"]["r2"] - ablation["persistence"]["r2"]
        dr2_f = ablation["full"]["r2"]       - ablation["meteo_only"]["r2"]
        lines += [
            f"\n**ΔR² météo vs persistence** : {dr2_m:+.4f}",
            f"**ΔR² full  vs météo-only**  : {dr2_f:+.4f}",
            f"**Les lags contribuent {dr2_f/(dr2_m+dr2_f)*100:.0f}% du gain total de R²**",
            f"\nFeatures retirées pour météo-only : {ablation.get('lag_features_n', '?')} lags PM2.5",
        ]

    with open(path, "a") as f:
        f.write("\n".join(lines))
    log.info(f"Ablation ajoutée au rapport : {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 65)
    log.info("Modèle PM2.5 — XGBoost + LightGBM — Hackathon IndabaX 2026")
    log.info("=" * 65)

    # 1. Chargement
    df = load_features(FEATURES_FILE)

    # 2. Sélection features
    feature_cols = select_features(df)

    # Rapport rapide sur le dataset
    print(f"\n{'='*65}")
    print(f"Dataset : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    print(f"Features : {len(feature_cols)} variables prédictives")
    print(f"Target   : {TARGET}  (µg/m³)")
    print(f"Villes   : {df['city'].nunique()}")
    print(f"Années   : {sorted(df['year'].unique())}")
    nan_pct = df[feature_cols].isna().mean().mean() * 100
    print(f"NaN mean : {nan_pct:.2f}% (remplacés par 0)")
    print(f"{'='*65}\n")

    # 3. Cross-validation
    cv_results = run_cross_validation(df, feature_cols)

    # 4. Modèle final + test 2025
    final_models = train_final_and_test(df, feature_cols)

    # 5. Visualisations
    log.info("\nGénération des figures de diagnostic...")
    try:
        plot_results(cv_results, final_models, df, feature_cols)
    except Exception as e:
        log.warning(f"Visualisation partielle : {e}")

    # 6. Sauvegarde
    save_models(final_models, cv_results)

    # 7. Ablation study
    log.info("\nLancement ablation study...")
    full_metrics = final_models.get("xgb", {}).get("metrics", {})
    ablation = run_ablation_study(df, feature_cols, full_metrics)
    try:
        plot_ablation(ablation)
    except Exception as e:
        log.warning(f"Figure ablation partielle : {e}")
    append_ablation_to_report(ablation)

    # 8. Résumé final
    print(f"\n{'='*65}")
    print("RÉSULTATS FINAUX — TEST 2025")
    print(f"{'='*65}")
    for name, key in [("XGBoost", "xgb"), ("LightGBM", "lgb"), ("Ensemble", "ensemble")]:
        if key in final_models and "metrics" in final_models[key]:
            m = final_models[key]["metrics"]
            print(f"  {name:<12} : RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R²={m['r2']:.4f}  MAPE={m['mape']:.1f}%")
    print(f"\n{'─'*65}")
    print("ABLATION STUDY — TEST 2025")
    print(f"{'─'*65}")
    for label, key in [("Persistence",  "persistence"),
                        ("Météo-only",   "meteo_only"),
                        ("Full XGBoost", "full")]:
        if key in ablation and ablation[key]:
            m = ablation[key]
            print(f"  {label:<14} : RMSE={m['rmse']:.2f}  R²={m['r2']:.4f}  MAPE={m['mape']:.1f}%")
    print(f"{'='*65}")
    log.info("Pipeline modèle terminé.")


if __name__ == "__main__":
    main()

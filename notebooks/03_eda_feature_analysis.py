"""
EDA & Feature Engineering Analysis — Hackathon IndabaX Cameroun 2026
=====================================================================
Objectif : prendre des décisions éclairées sur :
  1. Valeurs manquantes  → quelle stratégie d'imputation ?
  2. Distributions       → log, normalisation, binning nécessaires ?
  3. Corrélations        → quelles features sont vraiment pertinentes ?
  4. Multicolinéarité    → quelles features sont redondantes ?
  5. Importance          → ranking rapide XGBoost pour sélection
  6. Outliers            → faut-il les traiter ?

Toutes les décisions sont justifiées par des tests ou visualisations.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

INPUT_PATH  = Path("data/dataset_with_pm25_target.parquet")
OUTPUT_DIR  = Path("data/eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "pm25_proxy"

# Palette cohérente
BLUE   = "#2196F3"
RED    = "#F44336"
GREEN  = "#4CAF50"
ORANGE = "#FF9800"
PURPLE = "#9C27B0"

print("=" * 70)
print("EDA & FEATURE ENGINEERING ANALYSIS")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 0. Chargement
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_parquet(INPUT_PATH)
df["time"] = pd.to_datetime(df["time"])

# Forcer numérique sur toutes les colonnes sauf clés
key_cols = ["city", "region", "time"]
for col in df.columns:
    if col not in key_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values(["city", "time"]).reset_index(drop=True)

print(f"\n[0] Dataset chargé : {df.shape}")
print(f"    Villes : {df['city'].nunique()} | "
      f"Période : {df['time'].min().date()} → {df['time'].max().date()}")
print(f"    Colonnes numériques : {df.select_dtypes(include=np.number).shape[1]}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. VALEURS MANQUANTES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[1] ANALYSE DES VALEURS MANQUANTES")
print("=" * 70)

nan_df = pd.DataFrame({
    "n_missing" : df.isnull().sum(),
    "pct_missing": (df.isnull().sum() / len(df) * 100).round(2),
    "dtype"     : df.dtypes
})
nan_df = nan_df[nan_df["n_missing"] > 0].sort_values("pct_missing", ascending=False)

if nan_df.empty:
    print("  Aucune valeur manquante dans le dataset d'entrée.")
else:
    print(f"  Colonnes avec NaN ({len(nan_df)}) :")
    print(nan_df.to_string())

    # Visualisation
    fig, ax = plt.subplots(figsize=(10, max(4, len(nan_df) * 0.4)))
    nan_df["pct_missing"].plot(kind="barh", ax=ax, color=RED, alpha=0.7)
    ax.axvline(5, color=ORANGE, ls="--", lw=1.5, label="5%")
    ax.axvline(20, color=RED, ls="--", lw=1.5, label="20%")
    ax.set_xlabel("% valeurs manquantes")
    ax.set_title("Valeurs manquantes par colonne")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_missing_values.png", dpi=120)
    plt.close()
    print(f"\n  → Figure sauvegardée : {OUTPUT_DIR}/01_missing_values.png")

# Pattern des NaN : aléatoire ou structurel ?
print("\n  Pattern des NaN (par ville × colonne) :")
nan_cols_with_missing = [c for c in df.columns if df[c].isna().any()
                         and c not in key_cols]
if nan_cols_with_missing:
    for col in nan_cols_with_missing[:5]:
        nan_by_city = df.groupby("city")[col].apply(lambda x: x.isna().sum())
        cities_with_nan = (nan_by_city > 0).sum()
        print(f"    {col:40s} : {cities_with_nan}/40 villes concernées")


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEST D'IMPUTATION — médiane vs mean vs ffill vs interpolate
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[2] TEST DE STRATÉGIE D'IMPUTATION")
print("=" * 70)

# On choisit une colonne numérique complète et on y injecte des NaN artificiels
# pour comparer les méthodes sur des valeurs connues
test_col = "blh_mean" if "blh_mean" in df.columns else df.select_dtypes(include=np.number).columns[0]

print(f"\n  Colonne de test : {test_col}")
print(f"  Mécanisme de NaN simulé : MCAR (Missing Completely At Random) — 10% des valeurs")

np.random.seed(42)
df_test = df[[test_col, "city", "time"]].copy()
mask = np.random.rand(len(df_test)) < 0.10
true_values = df_test.loc[mask, test_col].copy()
df_test.loc[mask, test_col] = np.nan

results = {}

# Médiane globale
imp_median_global = df_test[test_col].median()
pred_global = df_test[test_col].fillna(imp_median_global)
results["Médiane globale"] = np.sqrt(np.mean((pred_global[mask] - true_values)**2))

# Médiane par ville
imp_city_median = df_test.groupby("city")[test_col].transform("median")
pred_city_med = df_test[test_col].fillna(imp_city_median)
results["Médiane par ville"] = np.sqrt(np.mean((pred_city_med[mask] - true_values)**2))

# Moyenne par ville
imp_city_mean = df_test.groupby("city")[test_col].transform("mean")
pred_city_mean = df_test[test_col].fillna(imp_city_mean)
results["Moyenne par ville"] = np.sqrt(np.mean((pred_city_mean[mask] - true_values)**2))

# Forward fill par ville
pred_ffill = df_test.groupby("city")[test_col].transform(lambda x: x.ffill())
pred_ffill = pred_ffill.fillna(imp_city_median)  # NaN en tête de série
results["Forward fill (ffill)"] = np.sqrt(np.mean((pred_ffill[mask] - true_values)**2))

# Interpolation linéaire par ville
pred_interp = df_test.groupby("city")[test_col].transform(
    lambda x: x.interpolate(method="linear", limit_direction="both")
)
pred_interp = pred_interp.fillna(imp_city_median)
results["Interpolation linéaire"] = np.sqrt(np.mean((pred_interp[mask] - true_values)**2))

print(f"\n  Résultats RMSE (plus bas = meilleur) pour '{test_col}' :")
print(f"  {'Méthode':<30} {'RMSE':>10}")
print(f"  {'-'*42}")
best_method = min(results, key=results.get)
for method, rmse in sorted(results.items(), key=lambda x: x[1]):
    marker = " ← MEILLEUR" if method == best_method else ""
    print(f"  {method:<30} {rmse:>10.2f}{marker}")

# Visualisation comparaison imputations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter : vraies valeurs vs imputées
methods_to_plot = {
    "Médiane par ville": pred_city_med,
    "Forward fill": pred_ffill,
    "Interpolation": pred_interp,
}
colors_imp = [BLUE, GREEN, ORANGE]
ax = axes[0]
for (name, pred), col in zip(methods_to_plot.items(), colors_imp):
    ax.scatter(true_values.values, pred[mask].values,
               alpha=0.3, s=10, color=col, label=f"{name} (RMSE={results.get(name, results.get('Forward fill (ffill)', 0)):.0f})")
lims = [true_values.min(), true_values.max()]
ax.plot(lims, lims, "k--", lw=1.5, label="Parfait (y=x)")
ax.set_xlabel(f"Vraie valeur ({test_col})")
ax.set_ylabel("Valeur imputée")
ax.set_title("Vraies valeurs vs imputées (10% NaN artificiels)")
ax.legend(fontsize=8)

# Bar chart RMSE
ax2 = axes[1]
methods = list(results.keys())
rmses = [results[m] for m in methods]
colors_bar = [GREEN if m == best_method else BLUE for m in methods]
ax2.barh(methods, rmses, color=colors_bar, alpha=0.8)
ax2.set_xlabel("RMSE")
ax2.set_title("Comparaison des stratégies d'imputation")
ax2.axvline(min(rmses), color=GREEN, ls="--", lw=1.5, alpha=0.7)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "02_imputation_comparison.png", dpi=120)
plt.close()
print(f"\n  → Figure sauvegardée : {OUTPUT_DIR}/02_imputation_comparison.png")
print(f"\n  DÉCISION : utiliser '{best_method}' pour l'imputation des valeurs manquantes")


# ─────────────────────────────────────────────────────────────────────────────
# 3. DISTRIBUTION DE LA VARIABLE CIBLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[3] DISTRIBUTION DE LA VARIABLE CIBLE (pm25_proxy)")
print("=" * 70)

target = df[TARGET].dropna()
skewness = target.skew()
kurtosis = target.kurtosis()
_, pval_norm = stats.shapiro(target.sample(min(5000, len(target)), random_state=42))
skewness_log = np.log1p(target).skew()

print(f"\n  Distribution originale :")
print(f"    Moyenne   : {target.mean():.2f} µg/m³")
print(f"    Médiane   : {target.median():.2f} µg/m³")
print(f"    Std       : {target.std():.2f} µg/m³")
print(f"    Skewness  : {skewness:.3f}  (|>1| = très asymétrique)")
print(f"    Kurtosis  : {kurtosis:.3f}  (>3 = queues lourdes)")
print(f"    Test normalité (Shapiro-Wilk p-value) : {pval_norm:.2e}")

print(f"\n  Après transformation log1p :")
print(f"    Skewness log(PM2.5) : {skewness_log:.3f}")

if abs(skewness) > 1:
    print(f"\n  → PM2.5 est asymétrique (skew={skewness:.2f})")
    if abs(skewness_log) < abs(skewness):
        print(f"  → Log1p RÉDUIT la skewness : {skewness:.2f} → {skewness_log:.2f}")
        print(f"  DÉCISION : entraîner le modèle sur log(PM2.5) puis anti-log à la prédiction")
    else:
        print(f"  → Log1p n'améliore pas suffisamment. Garder l'échelle originale.")
else:
    print(f"  → Skewness modérée ({skewness:.2f}), transformation log non indispensable")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# Histogramme brut
axes[0].hist(target.clip(upper=200), bins=80, color=BLUE, edgecolor="white", alpha=0.8)
axes[0].axvline(target.mean(), color=RED, ls="--", lw=2, label=f"Mean={target.mean():.1f}")
axes[0].axvline(target.median(), color=GREEN, ls="--", lw=2, label=f"Median={target.median():.1f}")
axes[0].set_title(f"Distribution PM2.5\nskew={skewness:.2f}, kurt={kurtosis:.2f}")
axes[0].set_xlabel("PM2.5 proxy (µg/m³)")
axes[0].legend(fontsize=8)

# Log1p
log_target = np.log1p(target)
axes[1].hist(log_target, bins=80, color=ORANGE, edgecolor="white", alpha=0.8)
axes[1].set_title(f"Distribution log(PM2.5+1)\nskew={skewness_log:.2f}")
axes[1].set_xlabel("log(PM2.5 + 1)")

# QQ-plot vs normal
stats.probplot(target.sample(3000, random_state=42), plot=axes[2])
axes[2].set_title("QQ-plot vs distribution normale")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "03_target_distribution.png", dpi=120)
plt.close()
print(f"\n  → Figure sauvegardée : {OUTPUT_DIR}/03_target_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. DISTRIBUTIONS DES FEATURES — skewness, outliers
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[4] DISTRIBUTIONS DES FEATURES (skewness & outliers)")
print("=" * 70)

num_cols = [c for c in df.select_dtypes(include=np.number).columns
            if c not in [TARGET, "latitude", "longitude", "city_id"]
            and not c.startswith("F_")
            and df[c].notna().sum() > 1000]

skew_results = []
for col in num_cols:
    s = df[col].dropna()
    if len(s) < 100:
        continue
    skew_val = s.skew()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    n_outliers = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
    skew_log = np.log1p(s.clip(lower=0)).skew() if s.min() >= 0 else np.nan
    skew_results.append({
        "feature": col,
        "skewness": round(skew_val, 3),
        "skewness_log1p": round(skew_log, 3) if not np.isnan(skew_log) else np.nan,
        "n_outliers": n_outliers,
        "pct_outliers": round(100 * n_outliers / len(s), 1),
        "log_helps": abs(skew_log) < abs(skew_val) if not np.isnan(skew_log) else False
    })

skew_df = pd.DataFrame(skew_results).sort_values("skewness", key=abs, ascending=False)

high_skew = skew_df[abs(skew_df["skewness"]) > 1]
print(f"\n  Features très asymétriques (|skew| > 1) : {len(high_skew)}")
if not high_skew.empty:
    print(high_skew[["feature", "skewness", "skewness_log1p", "log_helps",
                      "n_outliers", "pct_outliers"]].to_string(index=False))

print(f"\n  Features avec outliers > 5% : ")
high_outliers = skew_df[skew_df["pct_outliers"] > 5]
if not high_outliers.empty:
    print(high_outliers[["feature", "pct_outliers", "skewness"]].to_string(index=False))
else:
    print("  Aucune.")

# Visualisation top features skewées
top_skew_cols = skew_df["feature"].head(min(9, len(skew_df))).tolist()
if top_skew_cols:
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(top_skew_cols):
        s = df[col].dropna().clip(
            lower=df[col].quantile(0.01),
            upper=df[col].quantile(0.99)
        )
        axes[i].hist(s, bins=50, color=BLUE, edgecolor="white", alpha=0.7)
        sk = df[col].skew()
        axes[i].set_title(f"{col}\nskew={sk:.2f}", fontsize=9)
        axes[i].tick_params(labelsize=7)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Distributions des features les plus asymétriques", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_feature_distributions.png", dpi=120)
    plt.close()
    print(f"\n  → Figure sauvegardée : {OUTPUT_DIR}/04_feature_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CORRÉLATIONS AVEC LA CIBLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[5] CORRÉLATIONS AVEC LA CIBLE (Pearson + Spearman)")
print("=" * 70)

corr_results = []
for col in num_cols:
    if col == TARGET:
        continue
    s = df[[col, TARGET]].dropna()
    if len(s) < 500:
        continue
    pearson_r, pearson_p = stats.pearsonr(s[col], s[TARGET])
    spearman_r, spearman_p = stats.spearmanr(s[col], s[TARGET])
    corr_results.append({
        "feature": col,
        "pearson_r": round(pearson_r, 3),
        "spearman_r": round(spearman_r, 3),
        "pearson_p": pearson_p,
        "spearman_sig": "✅" if spearman_p < 0.01 else "⚠️",
        "abs_spearman": abs(spearman_r)
    })

corr_df = pd.DataFrame(corr_results).sort_values("abs_spearman", ascending=False)

print(f"\n  Top 20 features les plus corrélées avec PM2.5 (Spearman) :")
print(f"  {'Feature':<40} {'Pearson':>8} {'Spearman':>9} {'Sig':>4}")
print(f"  {'-'*65}")
for _, row in corr_df.head(20).iterrows():
    print(f"  {row['feature']:<40} {row['pearson_r']:>8.3f} {row['spearman_r']:>9.3f} {row['spearman_sig']:>4}")

print(f"\n  Features faiblement corrélées (|Spearman| < 0.05) → candidats à supprimer :")
weak = corr_df[corr_df["abs_spearman"] < 0.05]
if not weak.empty:
    print(f"  {', '.join(weak['feature'].tolist())}")
else:
    print("  Aucune feature avec corrélation négligeable.")

# Visualisation
top_corr = corr_df.head(25)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Pearson vs Spearman
colors_corr = [GREEN if v > 0 else RED for v in top_corr["spearman_r"]]
axes[0].barh(top_corr["feature"], top_corr["spearman_r"], color=colors_corr, alpha=0.8)
axes[0].axvline(0, color="black", lw=0.8)
axes[0].set_xlabel("Spearman r avec PM2.5")
axes[0].set_title("Top 25 features — Corrélation Spearman avec PM2.5")
axes[0].tick_params(axis="y", labelsize=8)

# Scatter PM2.5 vs meilleure feature
best_feat = corr_df.iloc[0]["feature"]
sample = df[[best_feat, TARGET]].dropna().sample(min(5000, len(df)), random_state=42)
axes[1].scatter(sample[best_feat], sample[TARGET], alpha=0.2, s=8, color=BLUE)
axes[1].set_xlabel(best_feat)
axes[1].set_ylabel("PM2.5 proxy")
axes[1].set_title(f"Scatter : PM2.5 vs {best_feat}\n(Spearman r={corr_df.iloc[0]['spearman_r']:.3f})")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "05_correlations.png", dpi=120)
plt.close()
print(f"\n  → Figure sauvegardée : {OUTPUT_DIR}/05_correlations.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MULTICOLINÉARITÉ
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[6] MULTICOLINÉARITÉ ENTRE FEATURES")
print("=" * 70)

# Matrice de corrélation sur les features existantes
cols_for_corr = [c for c in num_cols if c != TARGET and df[c].notna().sum() > 1000]
corr_matrix = df[cols_for_corr].corr(method="spearman").abs()

# Paires très corrélées (>0.90) — redondantes
upper_tri = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
very_high_corr = []
for col in upper_tri.columns:
    for idx in upper_tri.index:
        val = upper_tri.loc[idx, col]
        if val > 0.90:
            very_high_corr.append((idx, col, round(val, 3)))

very_high_corr.sort(key=lambda x: x[2], reverse=True)
print(f"\n  Paires de features avec corrélation Spearman > 0.90 (redondantes) :")
if very_high_corr:
    print(f"  {'Feature A':<35} {'Feature B':<35} {'|r|':>5}")
    print(f"  {'-'*77}")
    for a, b, r in very_high_corr[:20]:
        print(f"  {a:<35} {b:<35} {r:>5.3f}")
    print(f"\n  → {len(very_high_corr)} paires redondantes trouvées")
    print(f"  → Pour chaque paire, garder celle avec la plus forte corrélation avec la CIBLE")
else:
    print("  Aucune paire avec |r| > 0.90")

# Heatmap sur un sous-ensemble des features les plus importantes
top_feats_for_heatmap = corr_df["feature"].head(20).tolist()
if top_feats_for_heatmap:
    sub_corr = df[top_feats_for_heatmap].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(14, 12))
    mask_tri = np.triu(np.ones_like(sub_corr), k=1)
    sns.heatmap(sub_corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={"size": 7}, linewidths=0.5)
    ax.set_title("Matrice de corrélation Spearman — Top 20 features", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_correlation_matrix.png", dpi=120)
    plt.close()
    print(f"\n  → Figure sauvegardée : {OUTPUT_DIR}/06_correlation_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. FEATURE IMPORTANCE — XGBoost rapide
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[7] IMPORTANCE DES FEATURES — XGBoost rapide")
print("=" * 70)

try:
    from xgboost import XGBRegressor

    feature_cols = [c for c in num_cols if c != TARGET
                    and not c.startswith("F_")
                    and df[c].notna().mean() > 0.5]

    df_ml = df[feature_cols + [TARGET]].dropna(subset=[TARGET])
    X = df_ml[feature_cols].fillna(df_ml[feature_cols].median())
    y = df_ml[TARGET]

    # Train sur 2020-2023, test sur 2024
    train_mask = df_ml.index.isin(df[df["time"].dt.year <= 2023].index)
    X_train, y_train = X[train_mask], y[train_mask]

    xgb = XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0
    )
    xgb.fit(X_train, y_train)

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": xgb.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"\n  Modèle rapide entraîné ({X_train.shape[0]:,} samples)")
    print(f"\n  Top 30 features par importance XGBoost :")
    print(f"  {'Rang':>4} {'Feature':<40} {'Importance':>10}")
    print(f"  {'-'*58}")
    for i, (_, row) in enumerate(imp_df.head(30).iterrows(), 1):
        print(f"  {i:>4} {row['feature']:<40} {row['importance']:>10.4f}")

    # Features avec importance quasi nulle
    near_zero = imp_df[imp_df["importance"] < 0.001]
    print(f"\n  Features importance < 0.001 (candidates à supprimer) : {len(near_zero)}")
    if not near_zero.empty:
        print(f"  {', '.join(near_zero['feature'].tolist())}")

    # Importance cumulée
    imp_df["cumulative"] = imp_df["importance"].cumsum()
    n_90pct = (imp_df["cumulative"] < 0.90).sum() + 1
    print(f"\n  Nombre de features pour capturer 90% de l'importance : {n_90pct}")

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    top30 = imp_df.head(30)
    colors_imp = [GREEN if i < n_90pct else BLUE for i in range(len(top30))]
    axes[0].barh(top30["feature"][::-1], top30["importance"][::-1],
                 color=colors_imp[::-1], alpha=0.8)
    axes[0].set_xlabel("Importance XGBoost")
    axes[0].set_title("Top 30 features — Importance XGBoost")
    axes[0].tick_params(axis="y", labelsize=8)

    # Courbe d'importance cumulée
    axes[1].plot(range(1, len(imp_df)+1), imp_df["cumulative"],
                 color=BLUE, lw=2)
    axes[1].axhline(0.90, color=RED, ls="--", lw=1.5, label="90%")
    axes[1].axhline(0.95, color=ORANGE, ls="--", lw=1.5, label="95%")
    axes[1].axvline(n_90pct, color=GREEN, ls=":", lw=1.5,
                    label=f"{n_90pct} features → 90%")
    axes[1].set_xlabel("Nombre de features (triées par importance)")
    axes[1].set_ylabel("Importance cumulée")
    axes[1].set_title("Courbe d'importance cumulée")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_feature_importance.png", dpi=120)
    plt.close()
    print(f"\n  → Figure sauvegardée : {OUTPUT_DIR}/07_feature_importance.png")

except ImportError:
    print("  XGBoost non disponible — étape ignorée")


# ─────────────────────────────────────────────────────────────────────────────
# 8. ANALYSE PAR GROUPE — features catégorielles
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[8] ANALYSE PAR GROUPE (régions, zones climatiques)")
print("=" * 70)

print("\n  PM2.5 moyen par région :")
reg_stats = df.groupby("region")[TARGET].agg(["mean", "std", "median", "count"])
reg_stats = reg_stats.sort_values("mean", ascending=False)
print(reg_stats.round(2).to_string())

# Test ANOVA : est-ce que la région est significative ?
groups = [df[df["region"] == r][TARGET].dropna().values
          for r in df["region"].dropna().unique()]
f_stat, p_val = stats.f_oneway(*groups)
print(f"\n  ANOVA test (région → PM2.5) : F={f_stat:.1f}, p={p_val:.2e}")
if p_val < 0.001:
    print(f"  → La région est un prédicteur HAUTEMENT SIGNIFICATIF du PM2.5 ✅")
    print(f"  → Justifie l'encodage de la région comme feature")

print("\n  PM2.5 moyen par mois (saisonnalité nationale) :")
month_stats = df.groupby(df["time"].dt.month)[TARGET].mean().round(2)
print(month_stats.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# Boxplot par région
regions_sorted = reg_stats.index.tolist()
data_by_region = [df[df["region"] == r][TARGET].dropna().clip(upper=150).values
                  for r in regions_sorted]
bp = axes[0].boxplot(data_by_region, labels=regions_sorted,
                      patch_artist=True, vert=True)
for patch, color in zip(bp["boxes"], plt.cm.tab10.colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_xticklabels(regions_sorted, rotation=45, ha="right", fontsize=8)
axes[0].set_ylabel("PM2.5 proxy (µg/m³)")
axes[0].set_title(f"PM2.5 par région\nANOVA F={f_stat:.0f}, p={p_val:.1e}")
axes[0].axhline(32.5, color=RED, ls="--", lw=1.5, label="AQLI 32.5")
axes[0].legend(fontsize=8)

# Saisonnalité
mois_labels = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]
axes[1].bar(range(1, 13), month_stats.values, color=BLUE, alpha=0.8)
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(mois_labels)
axes[1].set_ylabel("PM2.5 moyen (µg/m³)")
axes[1].set_title("Saisonnalité nationale du PM2.5")
axes[1].axhline(32.5, color=RED, ls="--", lw=1.5, label="AQLI 32.5")
axes[1].legend()

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "08_group_analysis.png", dpi=120)
plt.close()
print(f"\n  → Figure sauvegardée : {OUTPUT_DIR}/08_group_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. BESOIN DE NORMALISATION ?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[9] NORMALISATION — nécessaire pour XGBoost/LightGBM ?")
print("=" * 70)

print("""
  XGBoost et LightGBM sont des modèles à base d'arbres.
  Les arbres prennent des décisions par seuils sur une feature à la fois.
  Ils sont INVARIANTS à la mise à l'échelle (scaling, normalisation).
  → StandardScaler ou MinMaxScaler N'AMÉLIORENT PAS les performances
    pour XGBoost/LightGBM sur des features numériques.

  Exception importante : si on ajoute des embeddings ou des réseaux de
  neurones dans le pipeline, la normalisation devient nécessaire.

  Conclusion : PAS DE NORMALISATION pour notre pipeline XGBoost/LightGBM.
  Mais les transformations LOG restent utiles car elles changent la FORME
  de la relation (linéarisent une relation exponentielle), pas l'échelle.
""")


# ─────────────────────────────────────────────────────────────────────────────
# 10. SYNTHÈSE & RECOMMANDATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[10] SYNTHÈSE & DÉCISIONS ÉCLAIRÉES")
print("=" * 70)

print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  DÉCISIONS DE FEATURE ENGINEERING BASÉES SUR LES DONNÉES        │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  1. IMPUTATION DES NaN                                           │
  │     → Méthode retenue : '{best_method}'          │
  │     → Justifié par RMSE le plus bas sur 10% NaN simulés         │
  │                                                                  │
  │  2. TRANSFORMATION DE LA CIBLE                                   │
  │     → PM2.5 skew = {skewness:.2f} → {'log1p recommandé' if abs(skewness) > 1 else 'transformation non nécessaire'}          │
  │     → skew après log1p = {skewness_log:.2f}                          │
  │                                                                  │
  │  3. FEATURES SKEWÉES (log1p sur les inputs)                      │
  │     → {len(high_skew)} features avec |skew| > 1 → log1p sur précipitations,  │
  │       BLH, FRP, radiation (valeurs toujours ≥ 0)                │
  │                                                                  │
  │  4. NORMALISATION                                                │
  │     → PAS NÉCESSAIRE pour XGBoost/LightGBM                      │
  │                                                                  │
  │  5. FEATURES REDONDANTES                                         │
  │     → {len(very_high_corr)} paires avec |r| > 0.90 → supprimer le moins           │
  │       corrélé avec la cible dans chaque paire                   │
  │                                                                  │
  │  6. ENCODAGE                                                     │
  │     → Région : ordinal (du sud au nord) + one-hot si besoin     │
  │     → Saisons : binaire + cyclique sin/cos ✓ déjà prévu         │
  │                                                                  │
  │  7. IMPORTANCE XGBoost                                           │
  │     → {n_90pct if 'n_90pct' in dir() else '?'} features capturent 90% de l'importance           │
  │     → Supprimer les features avec importance < 0.001            │
  │                                                                  │
  └─────────────────────────────────────────────────────────────────┘
""")

print(f"  Figures sauvegardées dans : {OUTPUT_DIR}/")
print(f"  Fichiers générés :")
for f in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"    {f.name}")

print("\n" + "=" * 70)
print("EDA TERMINÉE — Utilisez ces résultats pour affiner 03_feature_engineering.py")
print("=" * 70)

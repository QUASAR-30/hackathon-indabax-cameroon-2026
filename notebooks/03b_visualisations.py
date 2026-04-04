"""
Visualisations du dataset PM2.5 — Hackathon IndabaX Cameroun 2026
==================================================================
Source : data/dataset_features.parquet (dataset final post feature engineering)

Figures produites :
  01 — Distribution PM2.5 (brut vs log, QQ-plot)
  02 — Saisonnalité nationale (mensuelle + par région)
  03 — Gradient géographique Nord-Sud (carte de bulles)
  04 — Heatmap villes × mois (PM2.5 moyen)
  05 — Séries temporelles villes représentatives (2020–2025)
  06 — Boxplots par région (avec normes OMS)
  07 — Corrélations top features avec PM2.5
  08 — Importance des features (XGBoost rapide sur données réelles)
  09 — Distribution des facteurs proxy (F_stagnation, F_wet, etc.)
  10 — Comparaison inter-annuelle (2020–2025)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

INPUT_PATH = Path("data/dataset_features.parquet")
OUT_DIR    = Path("data/visualisations")
OUT_DIR.mkdir(exist_ok=True)

TARGET     = "pm25_proxy"
TARGET_LOG = "pm25_log"

# Palette
BLUE   = "#2196F3"; RED    = "#F44336"; GREEN  = "#4CAF50"
ORANGE = "#FF9800"; PURPLE = "#9C27B0"; TEAL   = "#009688"
COLORS_REG = plt.cm.tab10.colors

print("=" * 65)
print("VISUALISATIONS DATASET PM2.5 — IndabaX Cameroun 2026")
print("=" * 65)

# ─── Chargement ──────────────────────────────────────────────────────────────
df = pd.read_parquet(INPUT_PATH)
df["time"] = pd.to_datetime(df["time"])
df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month
print(f"\nDataset : {df.shape} | {df['city'].nunique()} villes | "
      f"{df['time'].min().date()} → {df['time'].max().date()}")

# ─── Figure 01 : Distribution PM2.5 ─────────────────────────────────────────
print("\n[01] Distribution PM2.5...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Distribution de la variable cible PM2.5", fontsize=13, fontweight="bold")

target = df[TARGET].dropna()
skew_raw = target.skew()
skew_log = np.log1p(target).skew()

# Histogramme brut
axes[0].hist(target.clip(upper=200), bins=80, color=BLUE, edgecolor="white", alpha=0.85)
axes[0].axvline(target.mean(),   color=RED,    ls="--", lw=2, label=f"Moyenne = {target.mean():.1f}")
axes[0].axvline(target.median(), color=GREEN,  ls="--", lw=2, label=f"Médiane = {target.median():.1f}")
axes[0].axvline(15,              color=ORANGE, ls=":",  lw=1.5, label="OMS 24h = 15 µg/m³")
axes[0].axvline(5,               color=TEAL,   ls=":",  lw=1.5, label="OMS annuel = 5 µg/m³")
axes[0].set_xlabel("PM2.5 (µg/m³)"); axes[0].set_ylabel("Nombre de jours")
axes[0].set_title(f"Distribution brute\nskewness = {skew_raw:.2f}"); axes[0].legend(fontsize=8)

# Histogramme log1p
log_target = np.log1p(target)
axes[1].hist(log_target, bins=80, color=ORANGE, edgecolor="white", alpha=0.85)
axes[1].axvline(log_target.mean(),   color=RED,   ls="--", lw=2, label=f"Moyenne = {log_target.mean():.2f}")
axes[1].axvline(log_target.median(), color=GREEN, ls="--", lw=2, label=f"Médiane = {log_target.median():.2f}")
axes[1].set_xlabel("log(PM2.5 + 1)"); axes[1].set_ylabel("Nombre de jours")
axes[1].set_title(f"Après log1p (cible ML)\nskewness = {skew_log:.2f}"); axes[1].legend(fontsize=8)

# QQ-plot
stats.probplot(target.sample(5000, random_state=42), plot=axes[2])
axes[2].set_title("QQ-plot vs distribution normale\n(queue droite lourde = skew positif)")

plt.tight_layout()
fig.savefig(OUT_DIR / "01_distribution_pm25.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → 01_distribution_pm25.png")

# ─── Figure 02 : Saisonnalité ────────────────────────────────────────────────
print("[02] Saisonnalité...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Saisonnalité du PM2.5", fontsize=13, fontweight="bold")

mois = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]

# Mensuel national
monthly = df.groupby("month")[TARGET].agg(["mean","std","median"]).reset_index()
ax = axes[0]
bars = ax.bar(monthly["month"], monthly["mean"], color=BLUE, alpha=0.75,
              yerr=monthly["std"], capsize=3, label="Moyenne ± σ")
ax.plot(monthly["month"], monthly["median"], "o--", color=RED, lw=2,
        markersize=5, label="Médiane")
ax.fill_between(monthly["month"],
                monthly["mean"] - monthly["std"],
                monthly["mean"] + monthly["std"],
                alpha=0.15, color=BLUE)
ax.axhline(32.5, color=RED,    ls="--", lw=1.5, label="AQLI Cameroun 32.5 µg/m³")
ax.axhline(15,   color=ORANGE, ls=":",  lw=1.5, label="OMS 24h : 15 µg/m³")
ax.axhline(5,    color=GREEN,  ls=":",  lw=1.2, label="OMS annuel : 5 µg/m³")
ax.set_xticks(range(1,13)); ax.set_xticklabels(mois)
ax.set_ylabel("PM2.5 (µg/m³)"); ax.set_title("Saisonnalité nationale (moyenne ± σ)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Saisonnalité par zone climatique
zones = {0: "Équatorial (<5°N)", 1: "Transition (5–8°N)", 2: "Sahélien (>8°N)"}
colors_z = [TEAL, ORANGE, RED]
ax2 = axes[1]
for z, (zone_id, zone_name) in enumerate(zones.items()):
    sub = df[df["climate_zone"] == zone_id]
    mo = sub.groupby("month")[TARGET].mean()
    ax2.plot(mo.index, mo.values, "o-", color=colors_z[z], lw=2.5,
             markersize=6, label=zone_name)
ax2.axhline(32.5, color="gray", ls="--", lw=1.2, alpha=0.7)
ax2.axhline(15,   color=ORANGE, ls=":",  lw=1.2)
ax2.set_xticks(range(1,13)); ax2.set_xticklabels(mois)
ax2.set_ylabel("PM2.5 (µg/m³)")
ax2.set_title("Saisonnalité par zone climatique")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "02_saisonnalite.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → 02_saisonnalite.png")

# ─── Figure 03 : Gradient géographique ──────────────────────────────────────
print("[03] Gradient géographique...")
city_stats = df.groupby("city").agg(
    pm25_mean=(TARGET, "mean"),
    pm25_std=(TARGET, "std"),
    latitude=("latitude", "first"),
    longitude=("longitude", "first"),
    region=("region", "first"),
    climate_zone=("climate_zone", "first")
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Gradient géographique du PM2.5", fontsize=13, fontweight="bold")

# Scatter lat vs PM2.5
ax = axes[0]
for z, (zone_id, zone_name) in enumerate(zones.items()):
    sub = city_stats[city_stats["climate_zone"] == zone_id]
    ax.scatter(sub["latitude"], sub["pm25_mean"],
               s=80, color=colors_z[z], label=zone_name, zorder=3, alpha=0.85)
    for _, row in sub.iterrows():
        ax.annotate(row["city"], (row["latitude"], row["pm25_mean"]),
                    textcoords="offset points", xytext=(4, 2), fontsize=7, alpha=0.8)
# Ligne de tendance
z_fit = np.polyfit(city_stats["latitude"], city_stats["pm25_mean"], 1)
x_fit = np.linspace(city_stats["latitude"].min(), city_stats["latitude"].max(), 100)
ax.plot(x_fit, np.polyval(z_fit, x_fit), "k--", lw=1.5, alpha=0.5,
        label=f"Tendance linéaire")
ax.axhline(32.5, color=RED, ls="--", lw=1.2, alpha=0.6, label="AQLI 32.5")
ax.set_xlabel("Latitude (°N)"); ax.set_ylabel("PM2.5 moyen annuel (µg/m³)")
ax.set_title("Gradient latitudinal Nord-Sud")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Carte de bulles (lon × lat, taille = PM2.5)
ax2 = axes[1]
sc = ax2.scatter(
    city_stats["longitude"], city_stats["latitude"],
    s=city_stats["pm25_mean"] * 3,
    c=city_stats["pm25_mean"],
    cmap="YlOrRd", alpha=0.85, edgecolors="gray", lw=0.5,
    vmin=10, vmax=80
)
plt.colorbar(sc, ax=ax2, label="PM2.5 moyen (µg/m³)")
for _, row in city_stats.nlargest(5, "pm25_mean").iterrows():
    ax2.annotate(row["city"], (row["longitude"], row["latitude"]),
                 textcoords="offset points", xytext=(5, 3), fontsize=8,
                 fontweight="bold", color="darkred")
ax2.set_xlabel("Longitude (°E)"); ax2.set_ylabel("Latitude (°N)")
ax2.set_title("Carte PM2.5 moyen par ville\n(taille ∝ PM2.5)")
ax2.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "03_gradient_geographique.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → 03_gradient_geographique.png")

# ─── Figure 04 : Heatmap villes × mois ──────────────────────────────────────
print("[04] Heatmap villes × mois...")
pivot = df.pivot_table(values=TARGET, index="city", columns="month", aggfunc="mean")
# Trier les villes par latitude (nord → sud)
city_lat = df.groupby("city")["latitude"].first().sort_values(ascending=False)
pivot = pivot.reindex(city_lat.index)
pivot.columns = mois

fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(
    pivot, annot=True, fmt=".0f", cmap="YlOrRd",
    linewidths=0.3, linecolor="white",
    annot_kws={"size": 7},
    cbar_kws={"label": "PM2.5 moyen (µg/m³)"},
    vmin=5, vmax=120, ax=ax
)
ax.set_title("Heatmap PM2.5 moyen — Villes (du nord au sud) × Mois",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Mois"); ax.set_ylabel("Ville (triée par latitude, nord → sud)")
plt.tight_layout()
fig.savefig(OUT_DIR / "04_heatmap_villes_mois.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → 04_heatmap_villes_mois.png")

# ─── Figure 05 : Séries temporelles ─────────────────────────────────────────
print("[05] Séries temporelles...")
cities_plot = [
    ("Maroua",    RED,    "Sahélien (Extrême-Nord)"),
    ("Garoua",    ORANGE, "Sahélien (Nord)"),
    ("Ngaoundere",PURPLE, "Transition (Adamaoua)"),
    ("Yaounde",   BLUE,   "Équatorial (Centre)"),
    ("Douala",    TEAL,   "Équatorial (Littoral)"),
    ("Kribi",     GREEN,  "Équatorial (Sud)"),
]

fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=True)
fig.suptitle("Séries temporelles PM2.5 — Villes représentatives (2020–2025)",
             fontsize=13, fontweight="bold")
axes = axes.flatten()

for i, (city, color, label) in enumerate(cities_plot):
    sub = df[df["city"] == city].set_index("time")[TARGET]
    if len(sub) == 0:
        continue
    # Série journalière et lissage 30j
    sub_30 = sub.rolling(30, center=True, min_periods=15).mean()
    axes[i].fill_between(sub.index, sub.values, alpha=0.2, color=color)
    axes[i].plot(sub.index, sub.values, color=color, lw=0.5, alpha=0.6)
    axes[i].plot(sub_30.index, sub_30.values, color=color, lw=2.5, label="Lissage 30j")
    axes[i].axhline(15,   color=ORANGE, ls=":", lw=1.2, alpha=0.8)
    axes[i].axhline(32.5, color=RED,    ls="--", lw=1.2, alpha=0.6)
    # Fond grisé = périodes Harmattan (nov-mar)
    for yr in range(2020, 2026):
        for m_start, m_end in [(f"{yr}-11-01", f"{yr+1}-03-31")]:
            try:
                axes[i].axvspan(pd.Timestamp(m_start), pd.Timestamp(m_end),
                                alpha=0.07, color="brown")
            except Exception:
                pass
    axes[i].set_ylabel("PM2.5 (µg/m³)")
    axes[i].set_title(f"{city} — {label}", fontsize=10)
    axes[i].legend(fontsize=8)
    axes[i].grid(alpha=0.25)
    ann_mean = sub.mean()
    axes[i].annotate(f"Moy. ann. = {ann_mean:.1f} µg/m³",
                     xy=(0.02, 0.93), xycoords="axes fraction",
                     fontsize=8, color=color, fontweight="bold")

# Légende commune
for ax in axes:
    ax.tick_params(axis="x", labelsize=8)

fig.text(0.5, 0.01, "Date", ha="center", fontsize=10)
fig.text(0.01, 0.5,
         "Zones grisées = saison Harmattan (nov–mar) | Orange : OMS 24h=15 | Rouge tirets : AQLI=32.5",
         ha="left", va="center", fontsize=8, style="italic", rotation=0)
plt.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(OUT_DIR / "05_series_temporelles.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → 05_series_temporelles.png")

# ─── Figure 06 : Boxplots par région ─────────────────────────────────────────
print("[06] Boxplots par région...")
region_order = ["Sud","Est","Centre","Littoral","Sud-Ouest","Nord-Ouest",
                "Ouest","Adamaoua","Nord","Extreme-Nord"]
region_order = [r for r in region_order if r in df["region"].unique()]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Distribution PM2.5 par région", fontsize=13, fontweight="bold")

# Boxplot
data_reg = [df[df["region"]==r][TARGET].clip(upper=180).values for r in region_order]
bp = axes[0].boxplot(data_reg, labels=region_order, patch_artist=True,
                     vert=True, showfliers=False,
                     medianprops={"color":"black","lw":2})
for patch, color in zip(bp["boxes"], plt.cm.RdYlGn_r(np.linspace(0.1,0.9,len(region_order)))):
    patch.set_facecolor(color); patch.set_alpha(0.8)
axes[0].axhline(15,   color=ORANGE, ls=":", lw=1.5, label="OMS 24h = 15 µg/m³")
axes[0].axhline(32.5, color=RED,    ls="--", lw=1.5, label="AQLI = 32.5 µg/m³")
axes[0].set_xticklabels(region_order, rotation=40, ha="right", fontsize=9)
axes[0].set_ylabel("PM2.5 (µg/m³)")
axes[0].set_title("Boxplot par région (sud → nord)")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

# Violin plot
parts = axes[1].violinplot(data_reg, positions=range(1, len(region_order)+1),
                            showmeans=True, showmedians=True)
for pc in parts["bodies"]:
    pc.set_alpha(0.6); pc.set_facecolor(BLUE)
axes[1].set_xticks(range(1, len(region_order)+1))
axes[1].set_xticklabels(region_order, rotation=40, ha="right", fontsize=9)
axes[1].axhline(15,   color=ORANGE, ls=":", lw=1.5)
axes[1].axhline(32.5, color=RED,    ls="--", lw=1.5)
axes[1].set_ylabel("PM2.5 (µg/m³)")
axes[1].set_title("Violin plot par région")
axes[1].grid(alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "06_boxplots_regions.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → 06_boxplots_regions.png")

# ─── Figure 07 : Corrélations top features ───────────────────────────────────
print("[07] Corrélations features...")
num_cols = [c for c in df.select_dtypes(include=np.number).columns
            if c not in [TARGET, TARGET_LOG, "year", "city_id"]
            and df[c].notna().sum() > 1000]

corr_data = []
for col in num_cols:
    s = df[[col, TARGET]].dropna()
    if len(s) < 500: continue
    sp_r, sp_p = stats.spearmanr(s[col], s[TARGET])
    pe_r, _    = stats.pearsonr(s[col], s[TARGET])
    corr_data.append({"feature": col, "spearman": sp_r, "pearson": pe_r,
                      "abs_sp": abs(sp_r)})

corr_df = pd.DataFrame(corr_data).sort_values("abs_sp", ascending=False).head(25)

fig, axes = plt.subplots(1, 2, figsize=(16, 9))
fig.suptitle("Corrélations des features avec PM2.5", fontsize=13, fontweight="bold")

colors_bar = [GREEN if v > 0 else RED for v in corr_df["spearman"]]
axes[0].barh(corr_df["feature"][::-1], corr_df["spearman"][::-1],
             color=colors_bar[::-1], alpha=0.8)
axes[0].axvline(0,    color="black", lw=0.8)
axes[0].axvline(0.3,  color="gray",  ls="--", lw=0.8, alpha=0.6)
axes[0].axvline(-0.3, color="gray",  ls="--", lw=0.8, alpha=0.6)
axes[0].set_xlabel("Corrélation de Spearman avec PM2.5")
axes[0].set_title("Top 25 features — Spearman r")
axes[0].tick_params(axis="y", labelsize=8)

# Pearson vs Spearman scatter
axes[1].scatter(corr_df["pearson"], corr_df["spearman"],
                s=60, color=BLUE, alpha=0.8)
for _, row in corr_df.head(8).iterrows():
    axes[1].annotate(row["feature"], (row["pearson"], row["spearman"]),
                     textcoords="offset points", xytext=(5,2), fontsize=7)
lim = max(abs(corr_df["pearson"]).max(), abs(corr_df["spearman"]).max()) + 0.05
axes[1].plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.4, label="y=x (linéaire)")
axes[1].set_xlabel("Pearson r"); axes[1].set_ylabel("Spearman r")
axes[1].set_title("Pearson vs Spearman\n(écart = non-linéarité)")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "07_correlations_features.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → 07_correlations_features.png")

# ─── Figure 08 : Importance XGBoost ──────────────────────────────────────────
print("[08] Importance XGBoost (entraînement rapide)...")
try:
    from xgboost import XGBRegressor

    feature_cols = [c for c in df.select_dtypes(include=np.number).columns
                    if c not in [TARGET, TARGET_LOG]
                    and df[c].notna().mean() > 0.8]

    df_ml = df[feature_cols + [TARGET_LOG]].dropna(subset=[TARGET_LOG])
    X = df_ml[feature_cols].fillna(df_ml[feature_cols].median())
    y = df_ml[TARGET_LOG]

    train_mask = df_ml.index.isin(df[df["year"] <= 2023].index)
    xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                       subsample=0.8, colsample_bytree=0.7,
                       random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(X[train_mask], y[train_mask])

    imp_df = pd.DataFrame({"feature": feature_cols,
                           "importance": xgb.feature_importances_}
                          ).sort_values("importance", ascending=False)
    imp_df["cumulative"] = imp_df["importance"].cumsum()
    n_90 = (imp_df["cumulative"] < 0.90).sum() + 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.suptitle("Importance des features — XGBoost (Train 2020–2023)",
                 fontsize=13, fontweight="bold")

    top30 = imp_df.head(30)
    colors_imp = [GREEN if i < n_90 else BLUE for i in range(len(top30))]
    axes[0].barh(top30["feature"][::-1], top30["importance"][::-1],
                 color=colors_imp[::-1], alpha=0.85)
    axes[0].set_xlabel("Importance XGBoost (gain)")
    axes[0].set_title(f"Top 30 features\n(vert = dans les {n_90} features → 90% importance)")
    axes[0].tick_params(axis="y", labelsize=8)

    axes[1].plot(range(1, len(imp_df)+1), imp_df["cumulative"],
                 color=BLUE, lw=2.5)
    axes[1].fill_between(range(1, len(imp_df)+1), imp_df["cumulative"],
                         alpha=0.15, color=BLUE)
    axes[1].axhline(0.90, color=RED,    ls="--", lw=1.5, label="90%")
    axes[1].axhline(0.95, color=ORANGE, ls="--", lw=1.5, label="95%")
    axes[1].axvline(n_90, color=GREEN,  ls=":",  lw=2,
                    label=f"{n_90} features → 90%")
    axes[1].set_xlabel("Nombre de features (triées par importance)")
    axes[1].set_ylabel("Importance cumulée"); axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Courbe d'importance cumulée")
    axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "08_importance_xgboost.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   → 08_importance_xgboost.png  ({n_90} features → 90% importance)")
except ImportError:
    print("   XGBoost non disponible — étape ignorée")

# ─── Figure 09 : Comparaison inter-annuelle ───────────────────────────────────
print("[09] Comparaison inter-annuelle...")
annual_stats = df.groupby(["year","region"])[TARGET].mean().reset_index()
years = sorted(df["year"].unique())
regions_sorted = sorted(df["region"].unique())

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Évolution inter-annuelle du PM2.5 (2020–2025)",
             fontsize=13, fontweight="bold")

# Tendance nationale par année
yr_national = df.groupby("year")[TARGET].agg(["mean","std","median"])
axes[0].plot(yr_national.index, yr_national["mean"], "o-",
             color=BLUE, lw=2.5, markersize=8, label="Moyenne nationale")
axes[0].fill_between(yr_national.index,
                     yr_national["mean"] - yr_national["std"],
                     yr_national["mean"] + yr_national["std"],
                     alpha=0.2, color=BLUE)
axes[0].plot(yr_national.index, yr_national["median"], "s--",
             color=GREEN, lw=2, markersize=6, label="Médiane nationale")
axes[0].axhline(32.5, color=RED, ls="--", lw=1.5, label="AQLI 32.5")
axes[0].axhline(15, color=ORANGE, ls=":", lw=1.2, label="OMS 24h")
for yr, row in yr_national.iterrows():
    axes[0].annotate(f"{row['mean']:.1f}", (yr, row["mean"]),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=9, fontweight="bold", color=BLUE)
axes[0].set_xlabel("Année"); axes[0].set_ylabel("PM2.5 (µg/m³)")
axes[0].set_title("Tendance nationale annuelle")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
axes[0].set_xticks(years)

# Heatmap région × année
pivot_yr = annual_stats.pivot(index="region", columns="year", values=TARGET)
pivot_yr = pivot_yr.reindex([r for r in region_order if r in pivot_yr.index])
sns.heatmap(pivot_yr, annot=True, fmt=".0f", cmap="YlOrRd",
            linewidths=0.5, linecolor="white",
            annot_kws={"size": 10},
            cbar_kws={"label": "PM2.5 moyen (µg/m³)"},
            vmin=10, vmax=80, ax=axes[1])
axes[1].set_title("PM2.5 moyen par région et par année")
axes[1].set_xlabel("Année"); axes[1].set_ylabel("Région")

plt.tight_layout()
fig.savefig(OUT_DIR / "09_evolution_annuelle.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → 09_evolution_annuelle.png")

# ─── Figure 10 : Features météo clés vs PM2.5 ────────────────────────────────
print("[10] Relations features météo vs PM2.5...")
meteo_pairs = [
    ("relative_humidity_2m_max", "Humidité max (%)"),
    ("precipitation_sum", "Précipitations (mm)"),
    ("blh_mean", "Hauteur couche limite (m)"),
    ("wind_speed_10m_max", "Vitesse vent max (km/h)"),
    ("daylight_duration", "Durée du jour (s)"),
    ("shortwave_radiation_sum", "Radiation solaire (W/m²)"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Relations features météo clés vs PM2.5",
             fontsize=13, fontweight="bold")
axes = axes.flatten()

sample = df.sample(min(8000, len(df)), random_state=42)
for i, (feat, label) in enumerate(meteo_pairs):
    if feat not in df.columns:
        axes[i].set_visible(False)
        continue
    x = sample[feat].dropna()
    y = sample.loc[x.index, TARGET]
    valid = x.notna() & y.notna()
    x, y = x[valid], y[valid]
    r, _ = stats.spearmanr(x, y)
    axes[i].scatter(x.clip(upper=x.quantile(0.99)), y.clip(upper=150),
                    alpha=0.15, s=8, color=BLUE)
    # Binned trend
    try:
        bins = pd.qcut(x, q=20, duplicates="drop")
        trend = y.groupby(bins).median()
        bin_mids = [b.mid for b in trend.index.categories
                    if hasattr(b, "mid")][:len(trend)]
        if bin_mids:
            axes[i].plot(bin_mids, trend.values[:len(bin_mids)],
                         "r-", lw=2.5, label=f"Tendance médiane")
    except Exception:
        pass
    axes[i].set_xlabel(label, fontsize=9)
    axes[i].set_ylabel("PM2.5 (µg/m³)", fontsize=9)
    axes[i].set_title(f"Spearman r = {r:.3f}", fontsize=10)
    axes[i].legend(fontsize=8)
    axes[i].grid(alpha=0.25)

plt.tight_layout()
fig.savefig(OUT_DIR / "10_meteo_vs_pm25.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → 10_meteo_vs_pm25.png")

# ─── Résumé ───────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("TOUTES LES FIGURES GÉNÉRÉES")
print("=" * 65)
figs = sorted(OUT_DIR.glob("*.png"))
for f in figs:
    size_kb = f.stat().st_size // 1024
    print(f"  {f.name:<45} {size_kb:>5} KB")
print(f"\n  Dossier : {OUT_DIR.resolve()}")
print("=" * 65)

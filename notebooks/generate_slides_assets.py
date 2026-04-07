"""
Génération des assets visuels pour Google Slides — L'Aube Africaine
====================================================================
Usage:
    conda run -n hackathon_pm25 pip install qrcode[pil]
    conda run -n hackathon_pm25 python notebooks/generate_slides_assets.py

Output: data/slides_assets/
"""

import sys, subprocess
from pathlib import Path

try:
    import qrcode
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "qrcode[pil]", "-q"])
    import qrcode

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
OUT  = DATA / "slides_assets"
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette TERRA ─────────────────────────────────────────────────────────────
TERRA  = "#8B3A1E"
TERRA2 = "#6B2A10"
OCHRE  = "#C8941A"
SAND   = "#D4B896"
CREAM  = "#F5EDD9"
CREAM2 = "#EDE0C4"
GOOD   = "#2E7D32"
BAD    = "#C62828"
WARN   = "#E65100"
MUTED  = "#7A6040"
TEXT   = "#2C1A08"
WHITE  = "#FFFFFF"

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
})

# ── Équipe ────────────────────────────────────────────────────────────────────
TEAM_NAME    = "TERRA  ·  L'Aube Africaine"
TEAM_MEMBERS = [
    "TIOGUIM Prince",
    "Membre 2",       # <- remplacer
    "Membre 3",       # <- remplacer
]
EVENT = "IndabaX Cameroon 2026 · Cameroun"
HF_URL = "https://huggingface.co/spaces/QUASAR-30/pm25-cameroun"
GH_URL = "https://github.com/QUASAR-30/hackathon-indabax-cameroon-2026"


def save(fig, name, dpi=180):
    path = OUT / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"  OK  {name}")


def rounded_box(ax, x, y, w, h, fc, ec, lw=1.5, radius=0.02, transform=None):
    if transform is None:
        transform = ax.transAxes
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       transform=transform, zorder=2)
    ax.add_patch(p)
    return p


# ══════════════════════════════════════════════════════════════════════════════
# QR CODES
# ══════════════════════════════════════════════════════════════════════════════
def make_qr(url, filename, label, sub_url):
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M,
                       box_size=12, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color=TEXT, back_color="white")

    fig = plt.figure(figsize=(3.6, 4.0), facecolor=WHITE)
    # Cadre extérieur
    ax_outer = fig.add_axes([0, 0, 1, 1])
    ax_outer.set_xlim(0, 1); ax_outer.set_ylim(0, 1); ax_outer.axis("off")
    rounded_box(ax_outer, 0.03, 0.03, 0.94, 0.94, WHITE, TERRA, lw=3)

    # Bande titre
    rounded_box(ax_outer, 0.03, 0.82, 0.94, 0.15, TERRA, TERRA, lw=0)
    ax_outer.text(0.5, 0.895, label, ha="center", va="center",
                  fontsize=11, fontweight="bold", color=WHITE)

    # QR code
    ax_qr = fig.add_axes([0.1, 0.22, 0.80, 0.60])
    ax_qr.imshow(img)
    ax_qr.axis("off")

    # URL sous le QR
    ax_outer.text(0.5, 0.12, sub_url, ha="center", va="center",
                  fontsize=8.5, color=TERRA, fontweight="bold")

    save(fig, filename)


print("\n== QR Codes")
make_qr(HF_URL, "qr_hf.png",     "Dashboard Live",  "huggingface.co/spaces/QUASAR-30/pm25-cameroun")
make_qr(GH_URL, "qr_github.png", "Code Source",      "github.com/QUASAR-30/hackathon-indabax-cameroon-2026")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Bandeau équipe
# ══════════════════════════════════════════════════════════════════════════════
print("\n== Slide 1 — Equipe")

fig = plt.figure(figsize=(13, 3.2), facecolor=TERRA2)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 13); ax.set_ylim(0, 3.2); ax.axis("off")
ax.set_facecolor(TERRA2)

# Bande gauche colorée
ax.add_patch(plt.Rectangle((0.22, 0.25), 0.08, 2.7, color=OCHRE, zorder=3))

# Nom équipe
ax.text(0.55, 2.4, TEAM_NAME,
        fontsize=28, fontweight="bold", color=OCHRE, va="center")

# Membres
ax.text(0.55, 1.72,
        "   ·   ".join(TEAM_MEMBERS),
        fontsize=16, color=CREAM, va="center")

# Event
ax.text(0.55, 1.1,
        EVENT, fontsize=12, color=SAND, va="center", alpha=0.9)

# Séparateur vertical
ax.add_patch(plt.Rectangle((8.2, 0.3), 0.03, 2.6, color=OCHRE, alpha=0.4, zorder=2))

# Stats — 3 cartes blanches à droite
stats = [
    ("40",        "villes\n10 régions"),
    ("R² 0.994",  "LightGBM\ntest 2025"),
    ("7 jours",   "prévisions\nlive · 07h WAT"),
]
for i, (val, lbl) in enumerate(stats):
    cx = 8.5 + i * 1.5
    # Carte blanche
    ax.add_patch(FancyBboxPatch((cx, 0.45), 1.3, 2.3,
                                boxstyle="round,pad=0.05",
                                facecolor=WHITE, edgecolor=OCHRE,
                                linewidth=2, zorder=3))
    ax.text(cx + 0.65, 1.95, val,
            ha="center", va="center",
            fontsize=13, fontweight="bold", color=TERRA, zorder=4)
    ax.text(cx + 0.65, 1.15, lbl,
            ha="center", va="center",
            fontsize=9, color=MUTED, multialignment="center",
            linespacing=1.5, zorder=4)

save(fig, "s1_cover_team.png", dpi=200)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — KPI Cards
# ══════════════════════════════════════════════════════════════════════════════
print("\n== Slide 2 — KPIs")

kpis = [
    ("89.8 %",     "des jours dépassent\nle seuil OMS 24h (15 µg/m³)", "ERA5 · 2020–2025",  BAD),
    ("32.5 µg/m³", "PM2.5 annuel moyen\nnational · 6.5× le seuil OMS",  "AQLI 2023",         BAD),
    ("2.7 ans",    "d'espérance de vie perdus\nen moyenne / Camerounais", "AQLI 2023",        WARN),
    ("139 µg/m³",  "pic Harmattan à Maroua\n= 28× le seuil OMS",         "ERA5 · Extrême-Nord", BAD),
]

fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
fig.patch.set_facecolor(CREAM)
fig.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.92, bottom=0.06)

for ax, (val, desc, source, color) in zip(axes, kpis):
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    # Bande top colorée
    ax.add_patch(plt.Rectangle((0, 0.87), 1, 0.13, color=color, zorder=2, transform=ax.transAxes, clip_on=False))
    # Bordure gauche colorée
    ax.add_patch(plt.Rectangle((0, 0), 0.025, 0.87, color=color, alpha=0.25, transform=ax.transAxes))

    # Valeur principale
    ax.text(0.5, 0.67, val, transform=ax.transAxes,
            ha="center", fontsize=22, fontweight="bold", color=color, va="center")

    # Description
    ax.text(0.5, 0.43, desc, transform=ax.transAxes,
            ha="center", fontsize=10.5, color=TEXT, va="center",
            multialignment="center", linespacing=1.5)

    # Ligne séparatrice
    ax.plot([0.05, 0.95], [0.20, 0.20], color=SAND, linewidth=0.8, transform=ax.transAxes)

    # Source
    ax.text(0.5, 0.10, source, transform=ax.transAxes,
            ha="center", fontsize=9, color=MUTED, style="italic", va="center")

save(fig, "s2_kpis.png")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Gradient Nord → Sud
# ══════════════════════════════════════════════════════════════════════════════
print("\n== Slide 2 — Gradient Nord -> Sud")

# Données réelles si disponibles
proxy_path = DATA / "pm25_proxy_era5.parquet"
if proxy_path.exists():
    try:
        df = pd.read_parquet(proxy_path, columns=["city", "pm25_proxy", "latitude"])
        city_means = (df.groupby("city")
                       .agg(pm=("pm25_proxy", "mean"), lat=("latitude", "first"))
                       .sort_values("lat", ascending=False)
                       .reset_index())
        # Sélection : 5 nord + 3 milieu + 5 sud
        n = len(city_means)
        idx = list(range(5)) + list(range(n//2-1, n//2+2)) + list(range(n-5, n))
        sel = city_means.iloc[sorted(set(idx))].sort_values("lat", ascending=False)
        cities_g = list(zip(sel["city"], sel["pm"], sel["lat"]))
        print("    (donnees reelles ERA5)")
    except Exception as e:
        cities_g = None
        print(f"    (fallback: {e})")
else:
    cities_g = None

if cities_g is None:
    cities_g = [
        ("Mokolo", 52.7, 10.73), ("Maroua", 50.6, 10.58), ("Kousseri", 47.3, 12.07),
        ("Garoua", 40.5, 9.30),  ("Yagoua", 39.1, 10.33), ("Ngaoundere", 30.1, 7.32),
        ("Bafoussam", 28.4, 5.47),("Bamenda", 26.8, 5.96), ("Yaounde", 24.2, 3.87),
        ("Douala", 23.5, 4.05),  ("Kribi", 22.3, 2.95),   ("Ambam", 21.1, 2.38),
    ]

cities_g = sorted(cities_g, key=lambda x: -x[1])
g_labels = [c[0] for c in cities_g]
g_vals   = [c[1] for c in cities_g]

def pm_color(v):
    if v >= 45: return BAD
    if v >= 35: return WARN
    if v >= 25: return "#F9A825"
    return GOOD

g_colors = [pm_color(v) for v in g_vals]

fig, ax = plt.subplots(figsize=(14, 5.0))
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)

bars = ax.bar(range(len(g_labels)), g_vals,
              color=g_colors, width=0.60, edgecolor=WHITE, linewidth=1.2, zorder=3)

# Valeurs EN DESSOUS du haut de chaque barre (à l'intérieur si barre assez grande)
for bar, v in zip(bars, g_vals):
    y_text = bar.get_height() - 2.5 if bar.get_height() > 6 else bar.get_height() + 0.8
    color_text = WHITE if bar.get_height() > 6 else TEXT
    ax.text(bar.get_x() + bar.get_width()/2, y_text,
            f"{v:.1f}", ha="center", va="top" if bar.get_height() > 6 else "bottom",
            fontsize=10, fontweight="bold", color=color_text, zorder=4)

# Ligne OMS — avec annotation complètement à gauche hors des barres
ax.axhline(15, color=OCHRE, linestyle="--", linewidth=2.0, zorder=2, alpha=0.9)
ax.text(-0.55, 16.5, "Seuil OMS\n15 µg/m³",
        ha="left", va="bottom", fontsize=9.5, color=OCHRE, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE, edgecolor=OCHRE,
                  linewidth=1, alpha=0.95))

ax.set_xticks(range(len(g_labels)))
ax.set_xticklabels(g_labels, rotation=30, ha="right", fontsize=11, color=TEXT)
ax.set_ylabel("PM2.5 moyen (µg/m³)", fontsize=12, color=MUTED, labelpad=10)
ax.set_ylim(0, max(g_vals) * 1.18)
ax.set_xlim(-0.8, len(g_labels) - 0.2)
ax.set_title("Gradient de pollution Nord  →  Sud   ·   Moyenne annuelle 2020–2025",
             fontsize=14, fontweight="bold", color=TERRA, pad=14)

ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.tick_params(axis="y", colors=MUTED, labelsize=10)
ax.tick_params(axis="x", which="both", bottom=False)
ax.spines["left"].set_color(SAND)
ax.spines["left"].set_visible(True)
ax.spines["bottom"].set_color(SAND)
ax.spines["bottom"].set_visible(True)

# Note facteur en bas à droite
fig.text(0.98, 0.01, f"Facteur x{g_vals[0]/g_vals[-1]:.1f} entre nord et sud",
         ha="right", fontsize=9.5, color=MUTED, style="italic")

# Légende régions
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor=BAD,      label=">= 45 µg/m³  Extreme-Nord / Nord"),
    Patch(facecolor=WARN,     label="35–45 µg/m³  Nord / Adamaoua"),
    Patch(facecolor="#F9A825", label="25–35 µg/m³  Adamaoua / Ouest"),
    Patch(facecolor=GOOD,     label="< 25 µg/m³   Centre / Sud"),
]
ax.legend(handles=legend_elems, loc="upper right", fontsize=9,
          framealpha=0.9, edgecolor=SAND, frameon=True)

fig.tight_layout(rect=[0, 0.02, 1, 1])
save(fig, "s2_gradient.png")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Pipeline (sans emojis)
# ══════════════════════════════════════════════════════════════════════════════
print("\n== Slide 3 — Pipeline")

steps = [
    ("ERA5",   "Archive\nMétéo",    "Satellite · 2020–2025\n40 villes · BLH/Vent/Pluie",  TERRA, False),
    ("+",      "",                  "",                                                      None,  True),
    ("FIRMS",  "NASA\nFIRMS",       "Feux biomasse\nMODIS · FRP 75 km",                    TERRA, False),
    ("->",     "",                  "",                                                      None,  True),
    ("PROXY",  "Proxy\nPhysique",   "6 facteurs multiplies\nCalibre 32.5 µg/m3",           OCHRE, False),
    ("->",     "",                  "",                                                      None,  True),
    ("139F",   "Feature\nEngineering","Lags · Rolling\nHarmattan · Spatial\nAnti-leak 5/5", TERRA, False),
    ("->",     "",                  "",                                                      None,  True),
    ("ML",     "XGBoost +\nLightGBM","R²=0.994\nExpanding-window CV\nAblation study",      OCHRE, False),
    ("->",     "",                  "",                                                      None,  True),
    ("OMS",    "Alertes\nOMS",      "7 jours · 40 villes\nGitHub Actions · 07h WAT",        TERRA, False),
]

fig = plt.figure(figsize=(20, 4.2))
fig.patch.set_facecolor(WHITE)
ax = fig.add_axes([0.01, 0.05, 0.98, 0.82])
ax.set_facecolor(WHITE)
ax.axis("off")
ax.set_xlim(0, 20)
ax.set_ylim(0, 4.2)

BOX_W, BOX_H = 2.4, 3.0
x = 0.3

for abbr, title, sub, color, is_sep in steps:
    if is_sep:
        sym = "+" if abbr == "+" else ">"
        ax.text(x + 0.15, 2.1, sym,
                ha="center", va="center",
                fontsize=22, fontweight="bold",
                color=TERRA if sym == "+" else SAND)
        x += 0.5
        continue

    is_hl = (color == OCHRE)
    bg = "#FFF8EC" if is_hl else WHITE
    ec = OCHRE if is_hl else TERRA
    lw = 2.5 if is_hl else 1.8

    # Boîte
    ax.add_patch(FancyBboxPatch((x, 0.5), BOX_W, BOX_H,
                                boxstyle="round,pad=0.07",
                                facecolor=bg, edgecolor=ec, linewidth=lw, zorder=2))

    # Cercle d'icône (remplacement emoji)
    circle = plt.Circle((x + BOX_W/2, 0.5 + BOX_H - 0.55), 0.42,
                         color=ec, zorder=3)
    ax.add_patch(circle)
    ax.text(x + BOX_W/2, 0.5 + BOX_H - 0.55, abbr,
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=WHITE, zorder=4)

    # Titre
    ax.text(x + BOX_W/2, 0.5 + BOX_H - 1.25, title,
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=TEXT, multialignment="center", linespacing=1.35)

    # Barre séparatrice
    ax.plot([x + 0.2, x + BOX_W - 0.2], [0.5 + BOX_H - 1.65, 0.5 + BOX_H - 1.65],
            color=SAND, linewidth=0.8)

    # Sous-texte
    ax.text(x + BOX_W/2, 0.5 + BOX_H - 2.35, sub,
            ha="center", va="center", fontsize=9, color=MUTED,
            multialignment="center", linespacing=1.45)

    x += BOX_W + 0.5

ax.set_title("Pipeline complet — de la donnée satellite à l'alerte OMS",
             fontsize=13.5, fontweight="bold", color=TERRA, y=1.04)
save(fig, "s3_pipeline.png")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Tableau de performances
# ══════════════════════════════════════════════════════════════════════════════
print("\n== Slide 4 — Tableau performances")

rows = [
    ["Persistence J-1 (baseline)",    "0.745",   "—",    "—",    "—"],
    ["XGBoost (complet)",              "0.9929",  "1.66", "0.93", "2.5 %"],
    ["LightGBM  *",                    "0.9940",  "1.53", "0.86", "2.3 %"],
    ["Ensemble 50/50",                 "0.9939",  "1.55", "0.85", "2.2 %"],
    ["Cold-start  (meteo-only)",       "0.9959",  "—",    "—",    "—"],
]
cols = ["Modele", "R²", "RMSE (µg/m³)", "MAE (µg/m³)", "MAPE"]
best_rows = {2, 3}  # LightGBM + Ensemble

fig, ax = plt.subplots(figsize=(12, 3.6))
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)
ax.axis("off")

col_w  = [0.34, 0.16, 0.19, 0.17, 0.14]
x_pos  = [sum(col_w[:i]) for i in range(len(col_w))]
row_h  = 0.115
header_y = 0.82

# En-tête
for j, (col, xp, cw) in enumerate(zip(cols, x_pos, col_w)):
    ax.add_patch(FancyBboxPatch((xp + 0.005, header_y), cw - 0.010, row_h,
                                boxstyle="square,pad=0",
                                facecolor=TERRA, edgecolor="none",
                                transform=ax.transAxes, zorder=2))
    ax.text(xp + cw/2, header_y + row_h/2, col,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=WHITE, zorder=3)

# Lignes
for i, row in enumerate(rows):
    y = header_y - (i + 1) * row_h
    is_best = i in best_rows
    bg = "#FFF3E0" if is_best else (CREAM if i % 2 == 1 else WHITE)
    for j, (val, xp, cw) in enumerate(zip(row, x_pos, col_w)):
        ax.add_patch(FancyBboxPatch((xp + 0.005, y + 0.002), cw - 0.010, row_h - 0.004,
                                    boxstyle="square,pad=0",
                                    facecolor=bg, edgecolor="none",
                                    transform=ax.transAxes, zorder=2))
        # Couleur R² vert pour bons scores
        fc = TEXT
        fw = "normal"
        if j == 1 and i >= 1:   # colonne R²
            fc = GOOD; fw = "bold"
        if i == best_rows and j > 0:
            fw = "bold"
        ax.text(xp + cw/2, y + row_h/2, val,
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=11, color=fc, fontweight=fw, zorder=3)

    # Indicateur meilleure ligne
    if is_best:
        ax.text(0.002, y + row_h/2, ">>",
                transform=ax.transAxes,
                ha="left", va="center",
                fontsize=9, color=OCHRE, fontweight="bold", zorder=3)

ax.set_title("Performances — Test 2025  (jamais vu pendant l'entraînement)",
             fontsize=13, fontweight="bold", color=TERRA,
             x=0.5, y=0.97, transform=ax.transAxes, ha="center")

fig.text(0.5, 0.01,
         "Litterature africaine PM2.5 : R² 0.70–0.85 · RMSE 2.7–6.5 µg/m3  |  * meilleur modele individuel",
         ha="center", fontsize=9, color=MUTED, style="italic")
save(fig, "s4_perf_table.png")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Ablation Study
# ══════════════════════════════════════════════════════════════════════════════
print("\n== Slide 4 — Ablation")

ablation = [
    ("Persistence (J-1)\nbaseline trivial",   0.745,  BAD),
    ("Complet\n(lags + meteo)",                0.9929, TERRA),
    ("Meteo-only [*]\n(cold-start)",           0.9959, GOOD),
]

fig, ax = plt.subplots(figsize=(11, 3.8))
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)

# Utiliser les ticks Y comme labels — bien plus propre
bar_ys   = [0.7, 1.8, 2.9]
bar_lbls = [a[0] for a in ablation]

for (lbl, val, col), y in zip(ablation, bar_ys):
    # Fond crème (pleine largeur visible)
    ax.barh(y, 0.30, height=0.55, left=0.70, color=CREAM2,
            edgecolor="none", zorder=1)
    # Barre colorée
    ax.barh(y, val - 0.70, height=0.55, left=0.70, color=col,
            edgecolor="none", zorder=2, alpha=0.92)
    # Valeur DANS la barre (vers la fin)
    ax.text(min(val - 0.005, 0.998), y,
            f"R² = {val:.4f}",
            va="center", ha="right",
            fontsize=12, fontweight="bold", color=WHITE, zorder=3)

ax.set_xlim(0.68, 1.01)
ax.set_ylim(0.1, 3.6)
ax.set_yticks(bar_ys)
ax.set_yticklabels(bar_lbls, fontsize=12, color=TEXT, linespacing=1.4)
ax.tick_params(axis="y", which="both", left=False, pad=10)
ax.set_xlabel("R²  (test 2025)", fontsize=11, color=MUTED, labelpad=8)
ax.spines["bottom"].set_visible(True)
ax.spines["bottom"].set_color(SAND)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="x", colors=MUTED, labelsize=10)
ax.set_title("Ablation Study — Ce que le modele apprend vraiment",
             fontsize=13, fontweight="bold", color=TERRA, pad=14)

fig.text(0.5, 0.01,
         "[*] Meteo-only = R² 0.9959  :  le modele apprend la physique (Harmattan, BLH, pluie) — pas l'autocorrelation",
         ha="center", fontsize=9.5, color=TERRA, style="italic")

fig.tight_layout(rect=[0, 0.07, 1, 1])
save(fig, "s4_ablation.png")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Cross-Validation
# ══════════════════════════════════════════════════════════════════════════════
print("\n== Slide 4 — CV Folds")

folds    = ["Fold 1\nVal 2021", "Fold 2\nVal 2022", "Fold 3\nVal 2023",
            "Fold 4\nVal 2024", "Test 2025\n(holdout final)"]
r2_xgb   = [0.970, 0.994, 0.994, 0.994, 0.9929]
r2_lgb   = [0.967, 0.994, 0.994, 0.994, 0.9940]
x        = np.arange(len(folds))
w        = 0.34

fig, ax  = plt.subplots(figsize=(10, 4.8))
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)
fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.22)

b1 = ax.bar(x - w/2, r2_xgb, w, label="XGBoost", color=TERRA,  alpha=0.92, edgecolor=WHITE, linewidth=0.5)
b2 = ax.bar(x + w/2, r2_lgb, w, label="LightGBM", color=OCHRE, alpha=0.92, edgecolor=WHITE, linewidth=0.5)

# Valeurs DANS les barres (pas au-dessus pour éviter les chevauchements)
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        # texte en blanc à l'intérieur si barre assez haute
        if h > 0.962:
            ax.text(bar.get_x() + bar.get_width()/2,
                    ax.get_ylim()[0] + (h - ax.get_ylim()[0]) * 0.5,
                    f"{h:.3f}",
                    ha="center", va="center",
                    fontsize=9, color=WHITE, fontweight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.0008,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=9, color=TEXT, fontweight="bold")

# Mise en évidence holdout (sans transform= pour utiliser transData par défaut)
ax.add_patch(FancyBboxPatch((3.55, 0.938), 0.9, 0.066,
                             boxstyle="round,pad=0.005",
                             facecolor=OCHRE, alpha=0.12,
                             edgecolor=OCHRE, linewidth=1.5,
                             transform=ax.transData))

ax.set_xticks(x)
ax.set_xticklabels(folds, fontsize=10.5, color=TEXT)
ax.set_ylim(0.940, 1.003)
ax.set_ylabel("R²", fontsize=12, color=MUTED, labelpad=8)
ax.set_title("Cross-Validation Expanding-Window — Jamais de random split",
             fontsize=13, fontweight="bold", color=TERRA, pad=12)
ax.legend(fontsize=11, framealpha=0.9, edgecolor=SAND, loc="lower right")
ax.yaxis.set_minor_locator(MultipleLocator(0.005))
ax.tick_params(axis="y", colors=MUTED, labelsize=10)
ax.tick_params(axis="x", which="both", bottom=False)
ax.spines["left"].set_visible(True); ax.spines["left"].set_color(SAND)
ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(SAND)

# Annotation propre sous le graphe
fig.text(0.5, 0.04,
         "Cadre en orange = Test 2025 holdout — jamais vu pendant l'entraînement",
         ha="center", fontsize=9.5, color=MUTED, style="italic")
save(fig, "s4_cv_folds.png")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
print("\n== Slide 4 — Feature Importance")

feats = [
    ("is_true_harmattan",         41.6, TERRA),
    ("climate_zone",              18.2, TERRA),
    ("pm25_proxy_roll3_mean",     12.5, OCHRE),
    ("harmattan_intensity",        2.8, TERRA),
    ("lat_x_harmattan",            2.2, OCHRE),
    ("pm25_proxy_lag1",            2.2, OCHRE),
    ("pm25_proxy_roll7_mean",      1.7, OCHRE),
    ("weather_proxy",              1.5, GOOD),
    ("precip_log",                 1.5, GOOD),
    ("lat_norm",                   1.5, OCHRE),
]

fig, ax = plt.subplots(figsize=(10, 5.0))
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)

y = np.arange(len(feats))[::-1]
for (lbl, val, col), yi in zip(feats, y):
    # Fond gris
    ax.barh(yi, 45, height=0.55, left=0, color=CREAM2, edgecolor="none", zorder=1)
    # Barre
    ax.barh(yi, val, height=0.55, left=0, color=col,
            edgecolor=WHITE, linewidth=0.4, alpha=0.90, zorder=2)
    # Valeur à droite de la barre
    ax.text(val + 0.5, yi, f"{val:.1f} %",
            va="center", fontsize=10.5, fontweight="bold", color=col)

ax.set_yticks(y)
ax.set_yticklabels([f[0] for f in feats],
                   fontsize=10.5, color=TEXT,
                   fontfamily="monospace")
ax.set_xlim(0, 50)
ax.set_xlabel("Importance — gain (%)", fontsize=11, color=MUTED, labelpad=8)
ax.set_title("Top 10 Features — XGBoost (gain)",
             fontsize=13, fontweight="bold", color=TERRA, pad=12)
ax.tick_params(axis="x", colors=MUTED, labelsize=10)
ax.tick_params(axis="y", which="both", left=False)
ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(SAND)
ax.xaxis.set_minor_locator(MultipleLocator(5))

# Note is_true_harmattan
ax.text(42.5, len(feats) - 1,
        "Signal\nphysique\ndominant",
        ha="center", va="center",
        fontsize=9, color=TERRA, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE,
                  edgecolor=TERRA, linewidth=1, alpha=0.9))

fig.tight_layout()
save(fig, "s4_feature_imp.png")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — Timeline déploiement (redesign complet)
# ══════════════════════════════════════════════════════════════════════════════
print("\n== Slide 6 — Timeline")

phases = [
    ("Phase 1  ·  M1 – M2", [
        "Alertes SMS via Orange API",
        "Partenariat ONACC",
        "Pilote Extreme-Nord",
    ], TERRA),
    ("Phase 2  ·  M3 – M6", [
        "Integration Ministere\nde la Sante Publique",
        "API publique\npour chercheurs",
    ], OCHRE),
    ("Phase 3  ·  M7+", [
        "Extension regionale :",
        "Tchad · RCA · Gabon",
        "Meme pipeline ERA5",
    ], GOOD),
]

# Layout : 1 bloc "Deja operationnel" + 3 boites de phases, alignment propre
# Pas de fleche qui traverse les boites — juste des fleches entre les boites
FIG_W, FIG_H = 16, 4.8
BOX_Y0, BOX_H_OP = 0.5, 3.8
BOX_H_PH = 3.8
BW_OP = 2.8   # largeur bloc operationnel
BW_PH = 2.9   # largeur boites phases
GAP   = 0.55  # espace entre boites

fig = plt.figure(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(WHITE)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)
ax.set_facecolor(WHITE); ax.axis("off")

# ── Titre ────────────────────────────────────────────────────────────────────
ax.text(FIG_W/2, FIG_H - 0.3,
        "Plan de Deploiement — De l'hackathon a la production nationale",
        ha="center", va="center",
        fontsize=13.5, fontweight="bold", color=TERRA)

# ── Bloc "Deja operationnel" ──────────────────────────────────────────────────
ox = 0.3
ax.add_patch(FancyBboxPatch((ox, BOX_Y0), BW_OP, BOX_H_OP,
                             boxstyle="round,pad=0.08",
                             facecolor=TERRA2, edgecolor=OCHRE,
                             linewidth=2.5, zorder=2))

# Header du bloc operationnel
ax.add_patch(FancyBboxPatch((ox, BOX_Y0 + BOX_H_OP - 0.65), BW_OP, 0.65,
                             boxstyle="square,pad=0",
                             facecolor=TERRA, edgecolor="none", zorder=3))
ax.text(ox + BW_OP/2, BOX_Y0 + BOX_H_OP - 0.32,
        "Deja operationnel",
        ha="center", va="center",
        fontsize=11.5, fontweight="bold", color=OCHRE, zorder=4)

# Badge LIVE
badge_y = BOX_Y0 + BOX_H_OP - 1.1
ax.add_patch(FancyBboxPatch((ox + 0.3, badge_y - 0.15), BW_OP - 0.6, 0.32,
                             boxstyle="round,pad=0.04",
                             facecolor=GOOD, edgecolor="none", zorder=3))
ax.text(ox + BW_OP/2, badge_y + 0.01, "LIVE  ·  GitHub Actions",
        ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=WHITE, zorder=4)

done_items = [
    "40 villes · 10 regions",
    "7j previsions / matin",
    "Dashboard HF public",
    "Export CSV / JSON",
    "Open-source",
]
y_start_done = BOX_Y0 + BOX_H_OP - 1.6
for i, it in enumerate(done_items):
    yi = y_start_done - i * 0.45
    # Puce verte
    ax.add_patch(plt.Circle((ox + 0.35, yi), 0.10, color=GOOD, zorder=3))
    ax.text(ox + 0.35, yi, "v",
            ha="center", va="center",
            fontsize=7, fontweight="bold", color=WHITE, zorder=4)
    ax.text(ox + 0.58, yi, it,
            ha="left", va="center",
            fontsize=10, color=CREAM, zorder=3)

# ── Fleches et boites phases ──────────────────────────────────────────────────
px_start = ox + BW_OP + GAP

for pi, (phase_title, items, col) in enumerate(phases):
    px = px_start + pi * (BW_PH + GAP)

    # Fleche entre boites (sauf avant la premiere)
    arr_x = px - GAP
    ax.annotate("", xy=(px, BOX_Y0 + BOX_H_PH/2),
                xytext=(arr_x, BOX_Y0 + BOX_H_PH/2),
                arrowprops=dict(arrowstyle="-|>", color=SAND, lw=2.0,
                                mutation_scale=16))

    # Boite blanche
    ax.add_patch(FancyBboxPatch((px, BOX_Y0), BW_PH, BOX_H_PH,
                                 boxstyle="round,pad=0.08",
                                 facecolor=WHITE, edgecolor=col,
                                 linewidth=2.5, zorder=2))

    # Header coloré avec numero
    ax.add_patch(FancyBboxPatch((px, BOX_Y0 + BOX_H_PH - 0.70), BW_PH, 0.70,
                                 boxstyle="square,pad=0",
                                 facecolor=col, edgecolor="none", zorder=3))
    ax.text(px + 0.28, BOX_Y0 + BOX_H_PH - 0.35,
            str(pi + 1),
            ha="center", va="center",
            fontsize=14, fontweight="bold", color=WHITE, alpha=0.5, zorder=4)
    ax.text(px + BW_PH/2 + 0.1, BOX_Y0 + BOX_H_PH - 0.35, phase_title,
            ha="center", va="center",
            fontsize=11, fontweight="bold", color=WHITE, zorder=4)

    # Items — espacés proprement dans la zone disponible
    # Zone items : de BOX_Y0+0.3 a BOX_Y0+BOX_H_PH-0.85
    item_zone_h  = BOX_H_PH - 1.05
    item_zone_y0 = BOX_Y0 + 0.3
    n_items = len(items)
    step = item_zone_h / (n_items + 0.5)

    for i, item in enumerate(items):
        iy = item_zone_y0 + item_zone_h - (i + 0.7) * step
        ax.text(px + BW_PH/2, iy, item,
                ha="center", va="center",
                fontsize=10.5, color=TEXT,
                multialignment="center", linespacing=1.4, zorder=3)

save(fig, "s6_timeline.png", dpi=180)


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*56}")
print(f"  Assets generes : data/slides_assets/")
print(f"{'='*56}")
total = sum(f.stat().st_size for f in sorted(OUT.glob("*.png"))) // 1024
print(f"  {len(list(OUT.glob('*.png')))} fichiers PNG  ·  {total} Ko\n")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name}")
print(f"\n  NB: Mettre a jour TEAM_MEMBERS (ligne 44) avec les 3 noms")

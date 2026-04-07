"""
Dashboard PM2.5 — Qualité de l'air au Cameroun
Hackathon IndabaX Cameroun 2026

Aesthetic: TERRA — African Environmental Field Report
Warm cream/parchment editorial aesthetic. Terracotta headers. Playfair Display serif.
Feels like an urgent scientific bulletin from an African research station.
Unexpected: light, warm, editorial — the opposite of every "dark sci-fi" dashboard.

Usage:
  conda activate hackathon_pm25
  streamlit run notebooks/07_dashboard.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parent.parent
DATA_DIR         = ROOT / "data"
PREDICTIONS_FILE = DATA_DIR / "predictions_latest.parquet"
ALERTS_FILE      = DATA_DIR / "alerts_latest.json"
UNCERTAINTY_FILE = DATA_DIR / "pm25_with_uncertainty.parquet"
PROXY_FILE       = DATA_DIR / "pm25_proxy_era5.parquet"

# ══════════════════════════════════════════════════════════════════════════════
# AQI CONFIG (WHO 2021)
# ══════════════════════════════════════════════════════════════════════════════

AQI_THRESHOLDS = [
    (0,    12.0,  "Bon",           "#2E7D32", "#F1F8E9", "#81C784"),
    (12.0, 35.4,  "Modéré",        "#F9A825", "#FFFDE7", "#FFD54F"),
    (35.4, 55.4,  "Mauvais",       "#E64A19", "#FBE9E7", "#FF8A65"),
    (55.4, 150.0, "Très mauvais",  "#B71C1C", "#FFEBEE", "#EF9A9A"),
    (150,  9999,  "Dangereux",     "#4A148C", "#F3E5F5", "#CE93D8"),
]
AQI_ORDER  = [t[2] for t in AQI_THRESHOLDS]
AQI_COLORS = {t[2]: t[3] for t in AQI_THRESHOLDS}
AQI_META   = {t[2]: {"border": t[3], "bg": t[4], "soft": t[5]} for t in AQI_THRESHOLDS}

LEVEL_MAP = {
    "bon": "Bon", "modere": "Modéré", "modéré": "Modéré",
    "mauvais": "Mauvais", "tres mauvais": "Très mauvais",
    "très mauvais": "Très mauvais", "dangereux": "Dangereux",
}


def to_aqi(v: float) -> str:
    for lo, hi, label, *_ in AQI_THRESHOLDS:
        if lo <= v < hi:
            return label
    return "Dangereux"


def norm_level(raw: str) -> str:
    return LEVEL_MAP.get(raw.lower().strip(), raw)


MONTHS_FR = ["Jan","Fév","Mar","Avr","Mai","Jui","Jul","Aoû","Sep","Oct","Nov","Déc"]

REGION_PALETTE = {
    "Adamaoua":    "#8B3A1E",
    "Centre":      "#C8941A",
    "Est":         "#5C7A2E",
    "Extreme-Nord":"#B71C1C",
    "Littoral":    "#1565C0",
    "Nord":        "#E64A19",
    "Nord-Ouest":  "#6A1B9A",
    "Ouest":       "#00695C",
    "Sud":         "#4E342E",
    "Sud-Ouest":   "#2E7D32",
}

HEATMAP_COLORSCALE = [
    [0.00, "#F5EDD9"],
    [0.08, "#7A9E6E"],   # vert sauge désaturé (amélioration 2 — lisibilité)
    [0.24, "#F9A825"],
    [0.37, "#E64A19"],
    [0.50, "#B71C1C"],
    [1.00, "#4A148C"],
]


def terra_layout(title: str = "", height: int = 440) -> dict:
    """Rich Plotly layout preset for TERRA theme."""
    return dict(
        title=dict(
            text=title,
            font=dict(family="Playfair Display, Georgia, serif",
                      size=16, color="#5C1F0A"),
            x=0.0, xanchor="left", pad=dict(l=0, t=4),
        ),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFCF5",
        font=dict(family="Barlow Condensed, Arial Narrow, sans-serif",
                  size=12, color="#1A0F05"),
        xaxis=dict(gridcolor="#E8DCC8", gridwidth=1, linecolor="#E8DCC8",
                   tickfont=dict(color="#1A0F05", size=11), zeroline=False),
        yaxis=dict(gridcolor="#E8DCC8", gridwidth=1, linecolor="#E8DCC8",
                   tickfont=dict(color="#1A0F05", size=11), zeroline=False),
        legend=dict(bgcolor="rgba(245,237,217,0.9)", bordercolor="#E8DCC8",
                    borderwidth=1,
                    font=dict(family="Barlow Condensed, sans-serif",
                              size=11, color="#1A0F05")),
        hoverlabel=dict(bgcolor="#F5EDD9", bordercolor="#8B3A1E",
                        font=dict(family="Barlow Condensed, sans-serif",
                                  size=12, color="#1A0F05")),
        margin=dict(l=10, r=10, t=56, b=10),
    )


# ══════════════════════════════════════════════════════════════════════════════
# TERRA CSS — Warm Editorial Theme
# ══════════════════════════════════════════════════════════════════════════════

TERRA_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400&family=Barlow+Condensed:wght@400;600;700&family=Barlow:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

/* ── Variables ── */
:root {
    --cream:      #F5EDD9;
    --parchment:  #EDE0C4;
    --warm-white: #FAF6EE;
    --terra:      #8B3A1E;
    --terra-dark: #5C1F0A;
    --ochre:      #C8941A;
    --ochre-light:#FFF3CD;
    --ink:        #1A0F05;
    --ink-mid:    #4A3728;
    --ink-soft:   #8B7355;
    --border:     #D4C4A0;
    --card:       #FFFCF5;
    --shadow:     rgba(90,50,20,0.12);
}

/* ── App background — warm cream with grain ── */
.stApp {
    background-color: var(--cream) !important;
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E") !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem !important; max-width: 100% !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--terra-dark) !important;
    border-right: none !important;
}
section[data-testid="stSidebar"] * { color: var(--parchment) !important; }
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem !important; }

/* ── Nav radio (styled as tabs) ── */
div[data-testid="stRadio"][data-nav-tabs] { display: none; } /* fallback hide raw widget */
.nav-tabs-container {
    border-bottom: 2px solid var(--terra);
    margin-bottom: 0;
    display: flex;
    gap: 0;
    padding-top: 0.5rem;
}
.nav-tab-btn {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--ink-soft) !important;
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    border-bottom: 3px solid transparent !important;
    padding: 0.7rem 1.8rem !important;
    cursor: pointer !important;
    transition: color 0.15s, border-color 0.15s;
    margin-bottom: -2px;
}
.nav-tab-btn:hover {
    color: var(--terra) !important;
}
.nav-tab-btn.active {
    color: var(--terra) !important;
    background: var(--warm-white) !important;
    border-bottom: 3px solid var(--terra) !important;
}
/* ── Streamlit radio styled as tabs ── */
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    gap: 0 !important;
    padding: 0 !important;
    flex-wrap: nowrap !important;
    background: var(--terra-dark) !important;
    border-radius: 0 !important;
    margin: 0 -2rem !important;
    padding-left: 2rem !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    display: flex !important;
    align-items: center !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: rgba(245,237,217,0.55) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
    padding: 0.75rem 1.6rem !important;
    cursor: pointer !important;
    gap: 0 !important;
    transition: color 0.15s, background 0.15s !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
    color: rgba(245,237,217,0.85) !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
    color: var(--cream) !important;
    background: rgba(245,237,217,0.12) !important;
    border-bottom: 3px solid var(--ochre) !important;
    font-weight: 700 !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}
/* ── Analyser button ── */
div[data-testid="stButton"].analyser-btn > button {
    background: var(--terra) !important;
    color: var(--cream) !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 6px 14px !important;
}
div[data-testid="stButton"].analyser-btn > button:hover {
    background: var(--terra-dark) !important;
}

/* ── Selectbox / inputs ── */
div[data-testid="stSelectbox"] > div > div {
    background: var(--warm-white) !important;
    color: var(--ink) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: 'Barlow', sans-serif !important;
    font-size: 14px !important;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--ink-soft) !important;
}
div[data-testid="stNumberInput"] input {
    background: var(--warm-white) !important;
    border: 1.5px solid var(--border) !important;
    color: var(--ink) !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Expander ── */
details {
    background: var(--cream) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 4px !important;
    margin-top: 1rem !important;
}
details summary {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--terra) !important;
    background: var(--cream) !important;
    padding: 0.7rem 1rem !important;
}
details[open] summary {
    border-bottom: 1.5px solid var(--border) !important;
}
/* Contenu de l'expander */
details > div {
    background: var(--cream) !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    background: var(--warm-white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}

/* ── Alerts / info boxes ── */
div[data-testid="stAlert"] {
    background: var(--ochre-light) !important;
    border: 1.5px solid var(--ochre) !important;
    border-radius: 4px !important;
    color: var(--ink) !important;
    font-family: 'Barlow', sans-serif !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--parchment); }
::-webkit-scrollbar-thumb { background: var(--terra); border-radius: 3px; }

/* ── Animations ── */
@keyframes pulse {
    0%, 100% { opacity:1; transform:scale(1); }
    50%       { opacity:0.5; transform:scale(0.85); }
}
</style>
"""

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner="Chargement des prévisions temps réel…")
def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(PREDICTIONS_FILE)
    df["time"]          = pd.to_datetime(df["time"])
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    df["aqi_category"]  = df["pm25_pred"].apply(to_aqi)
    return df


@st.cache_data(ttl=300, show_spinner="Chargement des alertes OMS…")
def load_alerts() -> dict:
    with open(ALERTS_FILE) as f:
        data = json.load(f)
    for a in data.get("alerts", []):
        a["level"] = norm_level(a.get("level", ""))
    return data


@st.cache_data(ttl=3600, show_spinner="Chargement des données historiques 2020–2025…")
def load_historical() -> pd.DataFrame:
    path = UNCERTAINTY_FILE if UNCERTAINTY_FILE.exists() else PROXY_FILE
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["time"]   = pd.to_datetime(df["time"])
    df["month"]  = df["time"].dt.month
    df["year"]   = df["time"].dt.year
    # Attach region if missing (from proxy file)
    if "region" not in df.columns and PROXY_FILE.exists():
        reg_map = (pd.read_parquet(PROXY_FILE, columns=["city","region"])
                   .drop_duplicates("city").set_index("city")["region"])
        df["region"] = df["city"].map(reg_map)
    return df


@st.cache_data(ttl=3600, show_spinner="Chargement des données météo…")
def load_proxy_meteo() -> pd.DataFrame:
    if not PROXY_FILE.exists():
        return pd.DataFrame()
    cols = ["city","time","precipitation_sum","blh_mean",
            "wind_speed_10m_max","pm25_proxy"]
    df = pd.read_parquet(PROXY_FILE, columns=cols)
    df["time"] = pd.to_datetime(df["time"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def banner_html(date_str: str, n_cities: int, generated: str) -> str:
    return (
        '<div style="background:linear-gradient(135deg,#5C1F0A 0%,#8B3A1E 50%,#7A2E10 100%);padding:28px 36px 22px;margin:0 -2rem 2rem;position:relative;overflow:hidden;">'
        '<div style="position:absolute;top:0;right:0;bottom:0;width:300px;background-image:repeating-linear-gradient(45deg,rgba(255,255,255,0.03) 0px,rgba(255,255,255,0.03) 1px,transparent 1px,transparent 12px);"></div>'
        '<div style="position:absolute;bottom:-20px;right:40px;font-size:120px;opacity:0.06;line-height:1;font-family:\'Playfair Display\',serif;color:white;">PM</div>'
        '<div style="display:inline-flex;align-items:center;gap:6px;background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);border-radius:2px;padding:3px 10px;margin-bottom:12px;">'
        '<span style="width:7px;height:7px;background:#00C853;border-radius:50%;display:inline-block;box-shadow:0 0 6px #00C853;animation:pulse 2s infinite;"></span>'
        '<span style="font-family:\'Barlow Condensed\',sans-serif;font-size:10px;font-weight:700;letter-spacing:0.18em;color:rgba(255,255,255,0.8);text-transform:uppercase;">EN DIRECT</span>'
        '</div>'
        '<h1 style="font-family:\'Playfair Display\',serif;font-size:42px;font-weight:900;color:#F5EDD9;margin:0 0 6px;line-height:1.1;letter-spacing:-0.01em;">Qualit&#233; de l\'Air &#8212; Cameroun</h1>'
        f'<p style="font-family:Barlow,sans-serif;font-size:13px;color:rgba(245,237,217,0.6);margin:0;letter-spacing:0.04em;">Pr&#233;visions PM2.5 &middot; {date_str} &ensp;&middot;&ensp; {n_cities} villes surveill&#233;es &ensp;&middot;&ensp; XGBoost / LightGBM + ERA5 &ensp;&middot;&ensp; IndabaX Cameroun 2026</p>'
        '</div>'
    )


def metric_card_html(label: str, value: str, sub: str = "",
                     accent: str = "#8B3A1E", urgent: bool = False) -> str:
    border_top = f"3px solid {accent}"
    shadow = f"0 2px 16px rgba(139,58,30,0.18), 0 0 0 1px {accent}30" if urgent else \
             "0 2px 8px rgba(90,50,20,0.10)"
    return f"""
    <div style="
        background: #FFFCF5;
        border: 1px solid #D4C4A0;
        border-top: {border_top};
        border-radius: 3px;
        padding: 18px 20px 14px;
        box-shadow: {shadow};
        height: 100%;
        position: relative;
    ">
        <div style="
            font-family:'Barlow Condensed',sans-serif;
            font-size:10px; font-weight:700;
            letter-spacing:0.18em; text-transform:uppercase;
            color:#8B7355; margin-bottom:10px;
        ">{label}</div>
        <div style="
            font-family:'DM Mono',monospace;
            font-size:32px; font-weight:500;
            color:{accent}; line-height:1;
            letter-spacing:-0.02em;
        ">{value}</div>
        {"" if not sub else f'<div style="font-family:Barlow,sans-serif;font-size:11px;color:#8B7355;margin-top:7px;line-height:1.4;">{sub}</div>'}
    </div>
    """


def alert_card_html(alert: dict) -> str:
    meta   = AQI_META.get(alert["level"], AQI_META["Dangereux"])
    pm25   = alert.get("pm25", 0)
    city   = alert.get("city", "")
    region = alert.get("region", "")
    msg    = alert.get("message", "")
    lvl    = alert.get("level", "")
    return f"""
    <div style="
        background: {meta['bg']};
        border: 1px solid {meta['border']}44;
        border-left: 4px solid {meta['border']};
        border-radius: 0 3px 3px 0;
        padding: 11px 14px;
        margin-bottom: 8px;
    ">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <div style="
                    font-family:'Playfair Display',serif;
                    font-size:15px; font-weight:700;
                    color:{meta['border']}; margin-bottom:2px;
                ">{city}</div>
                <div style="
                    font-family:'Barlow',sans-serif;
                    font-size:11px; color:#8B7355;
                ">{region}</div>
            </div>
            <div style="text-align:right;">
                <div style="
                    font-family:'DM Mono',monospace;
                    font-size:16px; font-weight:500;
                    color:{meta['border']};
                ">{pm25:.1f}</div>
                <div style="
                    font-family:'Barlow Condensed',sans-serif;
                    font-size:9px; font-weight:700;
                    letter-spacing:0.12em; text-transform:uppercase;
                    color:{meta['border']}; opacity:0.8;
                ">µg/m³</div>
            </div>
        </div>
        <div style="
            margin-top:6px;
            display:flex; justify-content:space-between; align-items:center;
        ">
            <span style="
                font-family:'Barlow',sans-serif;
                font-size:11px; color:#8B7355;
                font-style:italic;
            ">{msg[:60]}{'…' if len(msg)>60 else ''}</span>
            <span style="
                font-family:'Barlow Condensed',sans-serif;
                font-size:9px; font-weight:700;
                letter-spacing:0.1em; text-transform:uppercase;
                background:{meta['border']}; color:white;
                padding:2px 7px; border-radius:2px;
                white-space:nowrap; margin-left:8px;
            ">{lvl}</span>
        </div>
    </div>
    """


def section_title_html(text: str, sub: str = "") -> str:
    return f"""
    <div style="margin-bottom:1rem;">
        <div style="
            font-family:'Barlow Condensed',sans-serif;
            font-size:10px; font-weight:700;
            letter-spacing:0.2em; text-transform:uppercase;
            color:#8B7355; margin-bottom:4px;
        ">— {sub} —</div>
        <h3 style="
            font-family:'Playfair Display',serif;
            font-size:22px; font-weight:700;
            color:#1A0F05; margin:0;
            border-bottom: 2px solid #8B3A1E;
            padding-bottom: 6px;
        ">{text}</h3>
    </div>
    """ if sub else f"""
    <h3 style="
        font-family:'Playfair Display',serif;
        font-size:20px; font-weight:700;
        color:#1A0F05; margin:0 0 1rem;
        border-bottom: 2px solid #8B3A1E;
        padding-bottom: 6px;
        display:inline-block;
    ">{text}</h3>
    """


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TEMPS RÉEL
# ══════════════════════════════════════════════════════════════════════════════

def page_realtime():
    df_all = load_predictions()
    alerts = load_alerts()

    # ── Banner ────────────────────────────────────────────────────────────────
    forecast_date = df_all["forecast_date"].max().strftime("%d %B %Y")
    st.markdown(banner_html(forecast_date, df_all["city"].nunique(),
                            alerts.get("generated_at", "")[:10]),
                unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    with c1:
        regions    = ["Toutes"] + sorted(df_all["region"].unique())
        region_sel = st.selectbox("Région", regions, key="r1")
    with c2:
        dates    = sorted(df_all["time"].dt.date.unique())
        date_sel = st.selectbox("Date de prévision", dates,
                                index=min(1, len(dates)-1),
                                format_func=lambda d: d.strftime("%A %d %b %Y"), key="d1")
    with c3:
        models    = ["Tous"] + sorted(df_all["model_used"].unique())
        model_sel = st.selectbox("Modèle", models, key="m1")
    with c4:
        seuil = st.number_input("Seuil µg/m³", value=35.4, step=5.0, key="s1")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Filter ────────────────────────────────────────────────────────────────
    dff = df_all[df_all["time"].dt.date == date_sel].copy()
    if region_sel != "Toutes":
        dff = dff[dff["region"] == region_sel]
    if model_sel != "Tous":
        dff = dff[dff["model_used"] == model_sel]

    # ── Metrics ───────────────────────────────────────────────────────────────
    n_alert  = int((dff["pm25_pred"] >= seuil).sum())
    pm25_avg = dff["pm25_pred"].mean() if not dff.empty else 0
    pm25_max = dff["pm25_pred"].max()  if not dff.empty else 0
    city_max = dff.loc[dff["pm25_pred"].idxmax(), "city"] if not dff.empty else "—"

    max_color   = "#B71C1C" if pm25_max >= 55.4 else "#E64A19" if pm25_max >= 35.4 else "#8B3A1E"
    alert_color = "#B71C1C" if n_alert > 0 else "#2E7D32"

    _tip_style = (
        "font-family:'Barlow',sans-serif;font-size:11px;"
        "color:#4A3728;background:rgba(212,196,160,0.35);"
        "border-radius:0 0 3px 3px;padding:5px 10px;"
        "border:1px solid #D4C4A0;border-top:none;margin-top:-2px;"
    )
    m1, m2, m3, m4 = st.columns(4, gap="small")
    with m1:
        st.markdown(metric_card_html("Villes surveillées", str(len(dff)),
                    "40 villes · 10 régions", "#8B3A1E"), unsafe_allow_html=True)
        st.markdown(f'<div style="{_tip_style}">ℹ Réseau couvrant les 10 régions du Cameroun.</div>',
                    unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card_html("PM2.5 moyen", f"{pm25_avg:.1f}",
                    "µg/m³ · moyenne nationale", "#C8941A"), unsafe_allow_html=True)
        st.markdown(f'<div style="{_tip_style}">ℹ PM2.5 = particules ≤ 2.5 µm. OMS : &lt; 15 µg/m³ (24h), &lt; 5 µg/m³ (annuel).</div>',
                    unsafe_allow_html=True)
    with m3:
        st.markdown(metric_card_html("PM2.5 maximum", f"{pm25_max:.1f}",
                    f"µg/m³ · {city_max}", max_color,
                    urgent=pm25_max >= 55.4), unsafe_allow_html=True)
        tip3 = "⚠ Dépasse le seuil OMS ×3 — populations vulnérables à risque." if pm25_max >= 55.4 \
               else "ℹ Ville la plus polluée ce jour. Seuil alerte : 35 µg/m³."
        st.markdown(f'<div style="{_tip_style}">{tip3}</div>', unsafe_allow_html=True)
    with m4:
        st.markdown(metric_card_html("Villes en alerte", str(n_alert),
                    f"≥ {seuil:.0f} µg/m³", alert_color,
                    urgent=n_alert > 0), unsafe_allow_html=True)
        st.markdown(f'<div style="{_tip_style}">ℹ Villes dépassant {seuil:.0f} µg/m³. Ajustable via le champ Seuil.</div>',
                    unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Map + Alerts ──────────────────────────────────────────────────────────
    map_col, alert_col = st.columns([3, 1], gap="medium")

    with map_col:
        st.markdown(section_title_html("Carte des concentrations PM2.5",
                    "Vue géographique"), unsafe_allow_html=True)
        if dff.empty:
            st.warning("Aucune donnée pour cette sélection.")
        else:
            fig = px.scatter_mapbox(
                dff,
                lat="latitude",
                lon="longitude",
                color="aqi_category",
                size="pm25_pred",
                size_max=28,
                hover_name="city",
                hover_data={
                    "pm25_pred":    ":.1f",
                    "region":       True,
                    "model_used":   True,
                    "aqi_category": True,
                    "latitude":     False,
                    "longitude":    False,
                },
                color_discrete_map=AQI_COLORS,
                category_orders={"aqi_category": AQI_ORDER},
                zoom=5,
                center={"lat": 5.5, "lon": 12.5},
                mapbox_style="carto-positron",
                height=500,
                labels={"pm25_pred": "PM2.5 (µg/m³)", "aqi_category": "Catégorie AQI"},
            )
            fig.update_layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                uirevision="constant",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(
                    title=dict(text="CATÉGORIE AQI",
                               font=dict(size=10, family="Barlow Condensed",
                                         color="#8B7355")),
                    font=dict(size=12, family="Barlow Condensed", color="#1A0F05"),
                    bgcolor="rgba(255,252,245,0.95)",
                    bordercolor="#D4C4A0",
                    borderwidth=1,
                    x=0.01, y=0.99,
                ),
            )
            st.plotly_chart(fig, width="stretch",
                            config={"scrollZoom": True, "displayModeBar": False})

    # ── Session state : pré-sélection ville la plus polluée pour P2 ─────────────
    if not dff.empty and "city_sel" not in st.session_state:
        st.session_state["city_sel"] = city_max

    with alert_col:
        st.markdown(section_title_html("Alertes OMS", "Dépassements"), unsafe_allow_html=True)

        all_alerts = alerts.get("alerts", [])
        date_str   = date_sel.strftime("%Y-%m-%d")
        day_alerts = [a for a in all_alerts if a.get("date", "") == date_str]
        if region_sel != "Toutes":
            day_alerts = [a for a in day_alerts if a.get("region") == region_sel]
        day_alerts = sorted(
            [a for a in day_alerts if a.get("pm25", 0) >= seuil],
            key=lambda x: x.get("pm25", 0), reverse=True
        )

        if not day_alerts:
            st.markdown("""
            <div style="
                text-align:center; padding:40px 16px;
                color:#2E7D32; font-family:'Playfair Display',serif;
                font-style:italic; font-size:14px;
                background:#F1F8E9; border:1px solid #A5D6A7;
                border-radius:3px;
            ">Aucune ville<br>au-dessus du seuil.</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                f"<p style='font-family:Barlow,sans-serif;font-size:12px;"
                f"color:#B71C1C;margin-bottom:10px;font-weight:500;'>"
                f"⚠ {len(day_alerts)} ville(s) dépassent {seuil:.0f} µg/m³</p>",
                unsafe_allow_html=True,
            )
            with st.container(height=380):
                st.markdown(
                    "".join(alert_card_html(a) for a in day_alerts),
                    unsafe_allow_html=True,
                )
            # ── Bouton natif → P2 pour la ville la plus polluée ──────────────
            top_alert_city = day_alerts[0].get("city", "") if day_alerts else ""
            if top_alert_city:
                if st.button(
                    f"→ Analyser {top_alert_city}",
                    key="btn_top_alert",
                    help=f"Voir l'analyse historique complète de {top_alert_city}",
                    width="stretch",
                ):
                    st.session_state["city_sel"]    = top_alert_city
                    st.session_state["_nav_request"] = "PAR VILLE"
                    st.rerun()

    # ── Table ─────────────────────────────────────────────────────────────────
    with st.expander(" TABLEAU COMPLET DES PRÉDICTIONS — trié par PM2.5 décroissant"):
        if not dff.empty:
            tbl = dff[["city","region","pm25_pred","aqi_category","model_used"]].copy()
            tbl = tbl.sort_values("pm25_pred", ascending=False).reset_index(drop=True)
            pm25_max_tbl = tbl["pm25_pred"].max()
            rows_html = ""
            for _, row in tbl.iterrows():
                bar_pct   = int(row["pm25_pred"] / max(pm25_max_tbl, 1) * 100)
                aqi_meta  = AQI_META.get(row["aqi_category"], AQI_META["Bon"])
                bar_color = aqi_meta["border"]
                rows_html += (
                    f'<tr style="border-bottom:1px solid #E8DCC8;">'
                    f'<td style="padding:7px 10px;font-family:Barlow Condensed,sans-serif;'
                    f'font-size:13px;font-weight:600;color:#1A0F05;">{row["city"]}</td>'
                    f'<td style="padding:7px 10px;font-family:Barlow,sans-serif;'
                    f'font-size:12px;color:#8B7355;">{row["region"]}</td>'
                    f'<td style="padding:7px 10px;min-width:160px;">'
                    f'<div style="display:flex;align-items:center;gap:8px;">'
                    f'<div style="flex:1;background:#E8DCC8;border-radius:2px;height:6px;">'
                    f'<div style="background:{bar_color};width:{bar_pct}%;height:6px;border-radius:2px;"></div></div>'
                    f'<span style="font-family:DM Mono,monospace;font-size:12px;'
                    f'color:{bar_color};font-weight:500;white-space:nowrap;">{row["pm25_pred"]:.1f} µg/m³</span>'
                    f'</div></td>'
                    f'<td style="padding:7px 10px;">'
                    f'<span style="background:{aqi_meta["border"]};color:#fff;'
                    f'font-family:Barlow Condensed,sans-serif;font-size:10px;font-weight:700;'
                    f'letter-spacing:0.08em;padding:2px 7px;border-radius:2px;">{row["aqi_category"]}</span>'
                    f'</td>'
                    f'<td style="padding:7px 10px;font-family:Barlow,sans-serif;'
                    f'font-size:11px;color:#8B7355;">{row["model_used"]}</td>'
                    f'</tr>'
                )
            st.markdown(
                f'<div style="overflow-x:auto;border:1px solid #D4C4A0;border-radius:4px;">'
                f'<table style="width:100%;border-collapse:collapse;background:#FFFCF5;">'
                f'<thead><tr style="background:#F5EDD9;border-bottom:2px solid #8B3A1E;">'
                f'<th style="padding:8px 10px;text-align:left;font-family:Barlow Condensed,sans-serif;'
                f'font-size:11px;font-weight:700;letter-spacing:0.12em;color:#5C1F0A;">VILLE</th>'
                f'<th style="padding:8px 10px;text-align:left;font-family:Barlow Condensed,sans-serif;'
                f'font-size:11px;font-weight:700;letter-spacing:0.12em;color:#5C1F0A;">RÉGION</th>'
                f'<th style="padding:8px 10px;text-align:left;font-family:Barlow Condensed,sans-serif;'
                f'font-size:11px;font-weight:700;letter-spacing:0.12em;color:#5C1F0A;">PM2.5</th>'
                f'<th style="padding:8px 10px;text-align:left;font-family:Barlow Condensed,sans-serif;'
                f'font-size:11px;font-weight:700;letter-spacing:0.12em;color:#5C1F0A;">CATÉGORIE AQI</th>'
                f'<th style="padding:8px 10px;text-align:left;font-family:Barlow Condensed,sans-serif;'
                f'font-size:11px;font-weight:700;letter-spacing:0.12em;color:#5C1F0A;">MODÈLE</th>'
                f'</tr></thead><tbody>{rows_html}</tbody></table></div>',
                unsafe_allow_html=True,
            )
            # ── Export CSV + JSON ──────────────────────────────────────────────
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            exp1, exp2 = st.columns(2)
            with exp1:
                csv_data = tbl.rename(columns={
                    "city":"Ville","region":"Région","pm25_pred":"PM2.5 µg/m³",
                    "aqi_category":"Catégorie AQI","model_used":"Modèle"
                }).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Exporter CSV — prédictions",
                    data=csv_data,
                    file_name=f"pm25_cameroun_{date_sel}.csv",
                    mime="text/csv",
                )
            with exp2:
                import json as _json
                alerts_export = alerts.get("alerts", [])
                st.download_button(
                    "⬇ Exporter JSON — alertes OMS",
                    data=_json.dumps(alerts_export, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name=f"alertes_oms_{date_sel}.json",
                    mime="application/json",
                )

            # ── Navigation rapide → P2 depuis le tableau ─────────────────────
            st.markdown("<hr style='margin:14px 0 10px;border-color:#D4C4A0;'>", unsafe_allow_html=True)
            st.markdown(
                "<p style='font-family:Barlow Condensed,sans-serif;font-size:11px;"
                "font-weight:700;letter-spacing:0.12em;text-transform:uppercase;"
                "color:#8B7355;margin-bottom:8px;'>Analyser une ville en détail</p>",
                unsafe_allow_html=True,
            )
            all_cities_sorted = tbl["city"].tolist()
            default_city = st.session_state.get("city_sel", all_cities_sorted[0])
            default_idx  = all_cities_sorted.index(default_city) if default_city in all_cities_sorted else 0
            nav_col1, nav_col2 = st.columns([3, 1])
            with nav_col1:
                sel_city = st.selectbox(
                    "Ville",
                    all_cities_sorted,
                    index=default_idx,
                    label_visibility="collapsed",
                    key="nav_city_select",
                )
            with nav_col2:
                if st.button("→ Analyser", key="btn_nav_city", width="stretch"):
                    st.session_state["city_sel"]     = sel_city
                    st.session_state["_nav_request"] = "PAR VILLE"
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE PAR VILLE
# ══════════════════════════════════════════════════════════════════════════════

# Shared plotly layout defaults for the TERRA theme
_PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FFFCF5",
    font=dict(family="Barlow, sans-serif", color="#4A3728", size=11),
    margin=dict(l=50, r=20, t=30, b=40),
)

_AXIS_STYLE = dict(
    gridcolor="#E8DCC8",
    linecolor="#D4C4A0",
    showgrid=True,
    zeroline=False,
)


def page_city():
    df_hist = load_historical()
    df_pred = load_predictions()

    if df_hist.empty:
        st.warning("Fichier historique introuvable (`pm25_with_uncertainty.parquet`).")
        return

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #5C1F0A 0%, #8B3A1E 50%, #7A2E10 100%);
        padding: 20px 36px 16px; margin: 0 -2rem 1.5rem;
    ">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:10px;
                    font-weight:700;letter-spacing:0.18em;text-transform:uppercase;
                    color:rgba(245,237,217,0.5);margin-bottom:4px;">
            — Analyse locale —
        </div>
        <h2 style="font-family:'Playfair Display',serif;font-size:30px;font-weight:900;
                   color:#F5EDD9;margin:0;line-height:1.1;">
            Série temporelle par ville
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # Merge proxy meteo with uncertainty data
    df_proxy = load_proxy_meteo()  # loads pm25_proxy_era5.parquet (has meteo cols)
    has_proxy = not df_proxy.empty

    # ── City selector ─────────────────────────────────────────────────────────
    cities = sorted(df_hist["city"].unique())
    if "city_sel" not in st.session_state:
        st.session_state["city_sel"] = "Maroua" if "Maroua" in cities else cities[0]
    sel_col, info_col = st.columns([2, 3])
    with sel_col:
        city = st.selectbox(
            "Choisir une ville",
            cities,
            key="city_sel",
        )
    with info_col:
        city_data  = df_hist[df_hist["city"] == city]
        city_mean  = city_data["pm25_proxy"].mean()
        city_max   = city_data["pm25_proxy"].max()
        city_lat   = city_data["latitude"].iloc[0]
        city_region = df_pred[df_pred["city"] == city]["region"].iloc[0] \
                      if not df_pred[df_pred["city"] == city].empty else ""
        cat = to_aqi(city_mean)
        meta = AQI_META.get(cat, AQI_META["Bon"])
        # Rang national (parmi les 40 villes)
        all_city_means = df_hist.groupby("city")["pm25_proxy"].mean().sort_values(ascending=False)
        rank = int(all_city_means.index.tolist().index(city)) + 1 if city in all_city_means.index else "—"
        # % jours au-dessus de la norme OMS 24h (15 µg/m³)
        pct_above_who = 100 * (city_data["pm25_proxy"] > 15).mean()
        nat_mean = df_hist["pm25_proxy"].mean()
        vs_nat = city_mean - nat_mean
        vs_nat_str = f"+{vs_nat:.1f}" if vs_nat >= 0 else f"{vs_nat:.1f}"
        st.markdown(f"""
        <div style="display:flex;gap:12px;align-items:stretch;padding:10px 0;flex-wrap:wrap;">
            <div style="background:{meta['bg']};border:1px solid {meta['border']}40;
                        border-left:4px solid {meta['border']};border-radius:0 3px 3px 0;
                        padding:10px 18px;min-width:180px;">
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:10px;
                            font-weight:700;letter-spacing:0.15em;text-transform:uppercase;
                            color:{meta['border']};margin-bottom:3px;">{cat}</div>
                <div style="font-family:'DM Mono',monospace;font-size:22px;
                            font-weight:500;color:{meta['border']};">
                    {city_mean:.1f} µg/m³
                </div>
                <div style="font-family:'Barlow',sans-serif;font-size:11px;
                            color:#8B7355;">Moyenne 2020–2025 · {city_region} · {city_lat:.1f}°N</div>
            </div>
            <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
                <div style="background:#FFFCF5;border:1px solid #D4C4A0;border-radius:3px;
                            padding:8px 14px;text-align:center;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:10px;
                                font-weight:700;letter-spacing:0.12em;color:#8B7355;margin-bottom:2px;">RANG NATIONAL</div>
                    <div style="font-family:'DM Mono',monospace;font-size:20px;
                                color:#5C1F0A;font-weight:500;">#{rank}<span style="font-size:11px;color:#8B7355;">/40</span></div>
                </div>
                <div style="background:#FFFCF5;border:1px solid #D4C4A0;border-radius:3px;
                            padding:8px 14px;text-align:center;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:10px;
                                font-weight:700;letter-spacing:0.12em;color:#8B7355;margin-bottom:2px;">JOURS > OMS 15</div>
                    <div style="font-family:'DM Mono',monospace;font-size:20px;
                                color:#B71C1C;font-weight:500;">{pct_above_who:.0f}<span style="font-size:11px;color:#8B7355;">%</span></div>
                </div>
                <div style="background:#FFFCF5;border:1px solid #D4C4A0;border-radius:3px;
                            padding:8px 14px;text-align:center;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:10px;
                                font-weight:700;letter-spacing:0.12em;color:#8B7355;margin-bottom:2px;">VS NATIONAL</div>
                    <div style="font-family:'DM Mono',monospace;font-size:20px;
                                color:{'#B71C1C' if vs_nat>0 else '#2E7D32'};font-weight:500;">
                        {vs_nat_str}<span style="font-size:11px;color:#8B7355;"> µg/m³</span></div>
                </div>
                <div style="background:#FFFCF5;border:1px solid #D4C4A0;border-radius:3px;
                            padding:8px 14px;text-align:center;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:10px;
                                font-weight:700;letter-spacing:0.12em;color:#8B7355;margin-bottom:2px;">MAXIMUM OBSERVÉ</div>
                    <div style="font-family:'DM Mono',monospace;font-size:20px;
                                color:#8B3A1E;font-weight:500;">{city_max:.0f}<span style="font-size:11px;color:#8B7355;"> µg/m³</span></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Time series with IC90% ─────────────────────────────────────────────────
    st.markdown(section_title_html("PM2.5 · 2020–2025 avec IC90% Monte Carlo",
                "Série temporelle"), unsafe_allow_html=True)

    d = city_data.copy().sort_values("time")
    d_pred_city = df_pred[df_pred["city"] == city].sort_values("time")
    today_str   = str(pd.Timestamp.now().date())

    # Weekly resample for performance while keeping detail
    d_weekly = d.set_index("time").resample("W").agg({
        "pm25_proxy":  "mean",
        "pm25_mc_p05": "mean",
        "pm25_mc_p25": "mean",
        "pm25_mc_p75": "mean",
        "pm25_mc_p95": "mean",
    }).reset_index()

    fig_ts = go.Figure()

    # IC90% band — fill="tonexty": p05 first (invisible anchor), then p95 fills to it
    fig_ts.add_trace(go.Scatter(
        x=d_weekly["time"], y=d_weekly["pm25_mc_p05"],
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip", name="IC p05",
    ))
    fig_ts.add_trace(go.Scatter(
        x=d_weekly["time"], y=d_weekly["pm25_mc_p95"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(139,58,30,0.08)",
        showlegend=True, name="IC 90%", hoverinfo="skip",
    ))
    # IC50% band
    fig_ts.add_trace(go.Scatter(
        x=d_weekly["time"], y=d_weekly["pm25_mc_p25"],
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip", name="IC p25",
    ))
    fig_ts.add_trace(go.Scatter(
        x=d_weekly["time"], y=d_weekly["pm25_mc_p75"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(139,58,30,0.15)",
        showlegend=True, name="IC 50%", hoverinfo="skip",
    ))
    # Proxy line
    fig_ts.add_trace(go.Scatter(
        x=d_weekly["time"], y=d_weekly["pm25_proxy"],
        mode="lines", name="PM2.5 historique",
        line=dict(color="#8B3A1E", width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>PM2.5 : %{y:.1f} µg/m³<extra></extra>",
    ))
    # Forecast overlay
    if not d_pred_city.empty:
        # Pont de connexion : dernier point historique → première prévision
        last_hist = d_weekly.dropna(subset=["pm25_proxy"]).iloc[-1]
        first_pred = d_pred_city.iloc[0]
        fig_ts.add_trace(go.Scatter(
            x=[last_hist["time"], first_pred["time"]],
            y=[last_hist["pm25_proxy"], first_pred["pm25_pred"]],
            mode="lines", name="Connexion",
            line=dict(color="#2E7D32", width=1.5, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
        fig_ts.add_trace(go.Scatter(
            x=d_pred_city["time"], y=d_pred_city["pm25_pred"],
            mode="markers+lines", name="Prévisions J+7",
            line=dict(color="#2E7D32", width=2, dash="dot"),
            marker=dict(size=10, color="#2E7D32", symbol="diamond",
                        line=dict(color="white", width=1.5)),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Prévision : %{y:.1f} µg/m³<extra></extra>",
        ))
    # Today vline — use add_shape+add_annotation (add_vline with string date fails on some Plotly versions)
    today_ts = pd.Timestamp(today_str)
    fig_ts.add_shape(
        type="line",
        x0=today_ts, x1=today_ts, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="rgba(90,50,20,0.4)", width=1.5, dash="dash"),
    )
    fig_ts.add_annotation(
        x=today_ts, xref="x", y=0.97, yref="paper",
        text="Aujourd'hui", showarrow=False,
        font=dict(color="#8B7355", size=10, family="Barlow Condensed"),
        xanchor="left", yanchor="top",
    )
    # WHO guideline
    fig_ts.add_hline(y=15, line=dict(color="#C8941A", dash="dot", width=1.5),
                     annotation_text="OMS 15 µg/m³",
                     annotation_font=dict(color="#C8941A", size=10, family="Barlow Condensed"),
                     annotation_position="top right")

    fig_ts.update_layout(
        **_PLOT_LAYOUT,
        height=340,
        hovermode="x unified",
        uirevision=city,
        hoverlabel=dict(bgcolor="#FFFCF5", bordercolor="#D4C4A0",
                        font=dict(color="#1A0F05", size=12, family="Barlow")),
        legend=dict(orientation="h", y=1.02, x=0,
                    font=dict(size=11, family="Barlow Condensed"),
                    bgcolor="rgba(255,252,245,0.9)", bordercolor="#D4C4A0", borderwidth=1),
        xaxis=dict(**_AXIS_STYLE, showspikes=True, spikecolor="#D4C4A0",
                   spikethickness=1),
        yaxis=dict(**_AXIS_STYLE, title="PM2.5 (µg/m³)"),
    )
    st.plotly_chart(fig_ts, width="stretch",
                    config={"scrollZoom": True, "displayModeBar": False})

    # ── Meteorological drivers ────────────────────────────────────────────────
    if has_proxy:
        st.markdown(section_title_html("Facteurs météorologiques & PM2.5",
                    "Drivers physiques — corrélation visuelle"), unsafe_allow_html=True)

        dm = df_proxy[df_proxy["city"] == city].copy().sort_values("time")
        if not dm.empty:
            # Monthly aggregation for meteo
            dm_m = dm.set_index("time").resample("ME").agg({
                "precipitation_sum": "sum",
                "blh_mean": "mean",
                "wind_speed_10m_max": "mean",
                "pm25_proxy": "mean",
            }).reset_index()

            # 2 rows: [PM2.5 + précip secondary_y] / [BLH + vent secondary_y]
            fig_meteo = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.08,
                specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
                subplot_titles=[
                    "PM2.5 & Précipitations (pluie vers le bas = lavage atmosphérique)",
                    "BLH & Vitesse du vent",
                ],
            )

            # Row 1 — PM2.5 primary
            fig_meteo.add_trace(go.Scatter(
                x=dm_m["time"], y=dm_m["pm25_proxy"],
                name="PM2.5", mode="lines",
                line=dict(color="#8B3A1E", width=2),
                hovertemplate="<b>%{x|%b %Y}</b><br>PM2.5 : %{y:.1f} µg/m³<extra></extra>",
            ), row=1, col=1, secondary_y=False)

            # Row 1 — Précipitations secondary, INVERTED (rain down = PM2.5 down)
            fig_meteo.add_trace(go.Bar(
                x=dm_m["time"], y=dm_m["precipitation_sum"],
                name="Précipitations", marker_color="rgba(21,101,192,0.35)",
                marker_line_width=0,
                hovertemplate="<b>%{x|%b %Y}</b><br>Précip : %{y:.0f} mm<extra></extra>",
            ), row=1, col=1, secondary_y=True)

            # Row 2 — BLH primary
            fig_meteo.add_trace(go.Scatter(
                x=dm_m["time"], y=dm_m["blh_mean"],
                name="BLH moyen", mode="lines", fill="tozeroy",
                fillcolor="rgba(200,148,26,0.10)",
                line=dict(color="#C8941A", width=1.5),
                hovertemplate="<b>%{x|%b %Y}</b><br>BLH : %{y:.0f} m<extra></extra>",
            ), row=2, col=1, secondary_y=False)

            # Row 2 — Wind secondary
            fig_meteo.add_trace(go.Scatter(
                x=dm_m["time"], y=dm_m["wind_speed_10m_max"],
                name="Vent max", mode="lines",
                line=dict(color="#4A148C", width=1.5, dash="dot"),
                hovertemplate="<b>%{x|%b %Y}</b><br>Vent : %{y:.1f} km/h<extra></extra>",
            ), row=2, col=1, secondary_y=True)

            fig_meteo.update_layout(
                **_PLOT_LAYOUT,
                height=440,
                hovermode="x unified",
                uirevision=city,
                hoverlabel=dict(bgcolor="#EDE0C4", bordercolor="#8B3A1E",
                                font=dict(color="#1A0F05", size=11, family="Barlow")),
                showlegend=True,
                legend=dict(orientation="h", y=1.02, x=0,
                            font=dict(size=10, family="Barlow Condensed"),
                            bgcolor="rgba(255,252,245,0.9)", bordercolor="#D4C4A0", borderwidth=1),
            )
            # Axis styles
            for r in (1, 2):
                fig_meteo.update_xaxes(**_AXIS_STYLE, row=r, col=1)
                fig_meteo.update_yaxes(**_AXIS_STYLE, row=r, col=1, secondary_y=False)
            # Precipitation axis INVERTED on secondary
            fig_meteo.update_yaxes(
                autorange="reversed", showgrid=False,
                tickfont=dict(color="rgba(21,101,192,0.7)", size=9),
                title_text="Précip. ↓ (mm)",
                title_font=dict(color="rgba(21,101,192,0.7)", size=10),
                row=1, col=1, secondary_y=True,
            )
            fig_meteo.update_yaxes(
                **_AXIS_STYLE, title_text="BLH (m)", row=2, col=1, secondary_y=False,
            )
            fig_meteo.update_yaxes(
                showgrid=False,
                tickfont=dict(color="rgba(74,20,140,0.7)", size=9),
                title_text="Vent (km/h)",
                title_font=dict(color="rgba(74,20,140,0.7)", size=10),
                row=2, col=1, secondary_y=True,
            )
            st.plotly_chart(fig_meteo, width="stretch",
                            config={"displayModeBar": False})

    # ── Seasonality ───────────────────────────────────────────────────────────
    st.markdown(section_title_html("Saisonnalité mensuelle", "Profil climatique"),
                unsafe_allow_html=True)

    d_season = d.copy()
    d_season["month_num"] = d_season["time"].dt.month
    monthly_mean = (d_season.groupby("month_num")["pm25_proxy"]
                    .agg(["mean", "std"]).reset_index())
    monthly_mean["month_label"] = monthly_mean["month_num"].apply(
        lambda m: MONTHS_FR[m-1])

    # National monthly for comparison
    hist_all = load_historical()
    nat_monthly = (hist_all.assign(month_num=hist_all["time"].dt.month)
                   .groupby("month_num")["pm25_proxy"].mean().reset_index())
    nat_monthly.columns = ["month_num", "pm25_nat"]

    DRY_MONTHS = {11, 12, 1, 2, 3}  # Harmattan / saison sèche

    bar_colors = [
        "rgba(139,58,30,0.80)" if m in DRY_MONTHS else "rgba(21,101,192,0.55)"
        for m in monthly_mean["month_num"]
    ]

    fig_season = go.Figure()
    # ± 1 std envelope
    fig_season.add_trace(go.Scatter(
        x=monthly_mean["month_label"], y=monthly_mean["mean"] + monthly_mean["std"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_season.add_trace(go.Scatter(
        x=monthly_mean["month_label"], y=monthly_mean["mean"] - monthly_mean["std"],
        fill="tonexty", fillcolor="rgba(139,58,30,0.10)",
        line=dict(width=0), showlegend=True, name="± 1 écart-type", hoverinfo="skip",
    ))
    # Bars colored by season
    fig_season.add_trace(go.Bar(
        x=monthly_mean["month_label"], y=monthly_mean["mean"],
        name=city, marker_color=bar_colors, marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>PM2.5 : %{y:.1f} µg/m³<extra></extra>",
    ))
    # National average line
    nat_sorted = nat_monthly.sort_values("month_num")
    fig_season.add_trace(go.Scatter(
        x=[MONTHS_FR[int(m)-1] for m in nat_sorted["month_num"]],
        y=nat_sorted["pm25_nat"],
        name="Moyenne nationale",
        mode="lines+markers",
        line=dict(color="#C8941A", width=2, dash="dot"),
        marker=dict(size=5),
        hovertemplate="<b>%{x}</b><br>National : %{y:.1f} µg/m³<extra></extra>",
    ))
    fig_season.add_hline(y=15, line=dict(color="#C8941A", dash="dot", width=1),
                         annotation_text="OMS 15",
                         annotation_font=dict(color="#C8941A", size=9))
    # Season annotations
    y_top = monthly_mean["mean"].max() * 1.15
    fig_season.add_annotation(x="Jan", y=y_top, text="◆ Harmattan / saison sèche",
        showarrow=False, font=dict(color="rgba(139,58,30,0.8)", size=10, family="Barlow Condensed"),
        xanchor="left")
    fig_season.add_annotation(x="Jun", y=y_top, text="◆ Saison des pluies",
        showarrow=False, font=dict(color="rgba(21,101,192,0.8)", size=10, family="Barlow Condensed"),
        xanchor="center")

    fig_season.update_layout(
        **_PLOT_LAYOUT,
        height=320,
        barmode="overlay",
        hovermode="x unified",
        xaxis=dict(**_AXIS_STYLE, categoryorder="array", categoryarray=MONTHS_FR,
                   tickfont=dict(size=12, color="#1A0F05")),   # amélioration 5 — 12px min
        yaxis=dict(**_AXIS_STYLE, title="PM2.5 (µg/m³)"),
        legend=dict(orientation="h", y=1.05, x=0,
                    font=dict(size=11, family="Barlow Condensed"),
                    bgcolor="rgba(255,252,245,0.9)", bordercolor="#D4C4A0", borderwidth=1),
    )
    st.plotly_chart(fig_season, width="stretch",
                    config={"displayModeBar": False})

    # ── Export CSV ville ───────────────────────────────────────────────────────
    city_csv = city_data[["time","pm25_proxy"]].rename(
        columns={"time":"Date","pm25_proxy":"PM2.5 µg/m³"}
    ).to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇ Exporter données historiques — {city}",
        data=city_csv,
        file_name=f"pm25_{city.lower().replace(' ','_')}_2020_2025.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CLASSEMENT & STATISTIQUES NATIONALES
# ══════════════════════════════════════════════════════════════════════════════

def page_ranking():
    df_hist = load_historical()
    if df_hist.empty:
        st.warning("Fichier historique introuvable.")
        return

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="background:linear-gradient(135deg,#5C1F0A 0%,#8B3A1E 50%,#7A2E10 100%);'
        'padding:20px 36px 16px;margin:0 -2rem 1.5rem;">'
        '<div style="font-family:Barlow Condensed,sans-serif;font-size:10px;font-weight:700;'
        'letter-spacing:0.18em;text-transform:uppercase;color:rgba(245,237,217,0.5);'
        'margin-bottom:4px;">— Vue d\'ensemble —</div>'
        '<h2 style="font-family:Playfair Display,serif;font-size:30px;font-weight:900;'
        'color:#F5EDD9;margin:0;line-height:1.1;">Classement &amp; Statistiques nationales</h2>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Precompute city stats (ascending=True for horizontal bar — plotly renders bottom→top)
    city_stats = (df_hist.groupby("city")["pm25_proxy"]
                  .agg(mean="mean", std="std")
                  .reset_index()
                  .sort_values("mean", ascending=False)
                  .reset_index(drop=True))
    city_stats["aqi"] = city_stats["mean"].apply(to_aqi)

    df_hist["month_num"] = df_hist["time"].dt.month
    df_hist["year"]      = df_hist["time"].dt.year

    # ── Row 1: Top 10 + Gradient Nord-Sud ─────────────────────────────────────
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        top10 = city_stats.head(10).sort_values("mean", ascending=True)
        bar_colors = [AQI_COLORS.get(row["aqi"], "#8B3A1E") for _, row in top10.iterrows()]

        fig_top = go.Figure()
        fig_top.add_trace(go.Bar(
            orientation="h",
            x=top10["mean"], y=top10["city"],
            error_x=dict(type="data", array=top10["std"],
                         color="rgba(26,15,5,0.3)", thickness=1.5, width=4),
            marker=dict(color=bar_colors, line=dict(color="rgba(26,15,5,0.1)", width=0.5)),
            text=[f"  {v:.1f}" for v in top10["mean"]],
            textposition="outside",
            textfont=dict(family="Barlow Condensed", size=11, color="#1A0F05"),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>PM2.5 moy. : %{x:.1f} µg/m³<extra></extra>",
        ))
        # AQI legend patches
        for label, color in AQI_COLORS.items():
            fig_top.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                marker=dict(symbol="square", size=10, color=color),
                name=label, showlegend=True))
        # WHO 15 line (add_shape avoids string-date bug)
        fig_top.add_shape(type="line", x0=15, x1=15, y0=-0.5, y1=9.5,
                          line=dict(color="#2E7D32", width=1.5, dash="dot"))
        fig_top.add_annotation(x=15, y=9.5, text="OMS 15", showarrow=False,
                               xanchor="left", yanchor="top", xshift=4,
                               font=dict(color="#2E7D32", size=10, family="Barlow Condensed"))

        lay = terra_layout("Top 10 villes — PM2.5 moyen 2020–2025", height=420)
        lay["xaxis"].update(title=dict(text="PM2.5 moyen (µg/m³)",
                                       font=dict(size=11, color="#1A0F05")),
                            range=[0, top10["mean"].max() + top10["std"].max() + 18])
        lay["yaxis"].update(tickfont=dict(size=12, family="Barlow Condensed", color="#1A0F05"))
        lay["legend"].update(title=dict(text="Catégorie AQI",
                                        font=dict(size=11, color="#1A0F05")),
                             x=0.62, y=0.02)
        fig_top.update_layout(**lay)
        st.plotly_chart(fig_top, width="stretch", config={"displayModeBar": False})

    with col_r:
        lat_stats = (df_hist.groupby("city")
                     .agg(lat=("latitude","first"), pm25=("pm25_proxy","mean"))
                     .reset_index())
        lat_stats["city"]  = lat_stats.index if "city" not in lat_stats.columns \
                             else lat_stats["city"]
        lat_stats = lat_stats.reset_index(drop=True)
        lat_stats["aqi"]   = lat_stats["pm25"].apply(to_aqi)

        # Regression
        x_arr  = lat_stats["lat"].values
        y_arr  = lat_stats["pm25"].values
        coeffs = np.polyfit(x_arr, y_arr, 1)
        x_fit  = np.linspace(x_arr.min() - 0.3, x_arr.max() + 0.3, 100)
        y_fit  = np.polyval(coeffs, x_fit)
        y_pred = np.polyval(coeffs, x_arr)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
        r2     = 1 - ss_res / ss_tot
        rmse   = np.sqrt(ss_res / len(y_arr))

        fig_grad = go.Figure()
        # Regression band
        fig_grad.add_trace(go.Scatter(
            x=np.concatenate([x_fit, x_fit[::-1]]),
            y=np.concatenate([y_fit + rmse, (y_fit - rmse)[::-1]]),
            fill="toself", fillcolor="rgba(200,148,26,0.10)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        # Regression line
        fig_grad.add_trace(go.Scatter(
            x=x_fit, y=y_fit, mode="lines",
            line=dict(color="#C8941A", width=2, dash="dash"),
            name=f"Tendance (R²={r2:.2f})", hoverinfo="skip",
        ))
        # Points per AQI category
        for cat in AQI_ORDER:
            sub = lat_stats[lat_stats["aqi"] == cat]
            if sub.empty:
                continue
            fig_grad.add_trace(go.Scatter(
                x=sub["lat"], y=sub["pm25"], mode="markers", name=cat,
                marker=dict(color=AQI_COLORS[cat],
                            size=sub["pm25"].apply(lambda v: 6 + v / 4),
                            line=dict(color="rgba(26,15,5,0.3)", width=0.8), opacity=0.85),
                text=sub["city"],
                hovertemplate="<b>%{text}</b><br>Lat : %{x:.2f}°N<br>"
                              "PM2.5 : %{y:.1f} µg/m³<extra></extra>",
            ))
        fig_grad.add_annotation(xref="paper", yref="paper", x=0.02, y=0.97,
            text=f"<b>R² = {r2:.3f}</b>", showarrow=False,
            font=dict(size=13, color="#5C1F0A", family="Barlow Condensed"),
            bgcolor="#F5EDD9", bordercolor="#8B3A1E", borderwidth=1, borderpad=4,
            xanchor="left", yanchor="top")
        fig_grad.add_annotation(xref="paper", yref="paper", x=0.98, y=0.06,
            text="Plus au nord → Harmattan plus fort<br>→ PM2.5 plus élevé",
            showarrow=False,
            font=dict(size=10, color="#8B3A1E", family="Barlow Condensed"),
            align="right", xanchor="right", yanchor="bottom")
        # Amélioration 3 — annotations villes extrêmes (top 3 + bottom 3)
        lat_stats_sorted = lat_stats.sort_values("pm25", ascending=False)
        for _, row_ann in pd.concat([lat_stats_sorted.head(3), lat_stats_sorted.tail(3)]).iterrows():
            fig_grad.add_annotation(
                x=row_ann["lat"], y=row_ann["pm25"],
                text=row_ann["city"],
                showarrow=True, arrowhead=1, arrowsize=0.8,
                arrowcolor="rgba(139,58,30,0.5)", arrowwidth=1,
                font=dict(size=9, color="#5C1F0A", family="Barlow Condensed"),
                bgcolor="rgba(245,237,217,0.85)", bordercolor="#D4C4A0",
                borderwidth=1, borderpad=2,
                ax=18, ay=-18,
            )

        lay2 = terra_layout("Gradient Nord–Sud — latitude vs PM2.5", height=420)
        lay2["xaxis"].update(title=dict(text="Latitude (°N)",
                                        font=dict(size=11, color="#1A0F05")))
        lay2["yaxis"].update(title=dict(text="PM2.5 moyen (µg/m³)",
                                        font=dict(size=11, color="#1A0F05")))
        lay2["legend"].update(title=dict(text="Catégorie AQI",
                                         font=dict(size=11, color="#1A0F05")))
        fig_grad.update_layout(**lay2)
        st.plotly_chart(fig_grad, width="stretch", config={"displayModeBar": False})

    # ── Row 2: Heatmap saisonnalité + évolution inter-annuelle ────────────────
    col_a, col_b = st.columns([11, 9], gap="large")

    with col_a:
        top15_cities = city_stats.head(15)["city"].tolist()
        heat_data = (df_hist[df_hist["city"].isin(top15_cities)]
                     .groupby(["city","month_num"])["pm25_proxy"].mean().reset_index())
        heat_pivot = (heat_data.pivot(index="city", columns="month_num", values="pm25_proxy")
                      .reindex(columns=range(1, 13)))
        # Sort rows by descending mean PM2.5
        city_order = city_stats[city_stats["city"].isin(top15_cities)]["city"].tolist()
        heat_pivot = heat_pivot.reindex(city_order)
        z_max = min(150, float(np.nanmax(heat_pivot.values)) * 1.05)

        fig_heat = go.Figure(go.Heatmap(
            z=heat_pivot.values,
            x=[MONTHS_FR[m-1] for m in range(1, 13)],
            y=heat_pivot.index.tolist(),
            colorscale=HEATMAP_COLORSCALE,
            zmin=0, zmax=z_max,
            hoverongaps=False,
            xgap=1.5, ygap=1.5,
            colorbar=dict(
                title=dict(text="PM2.5<br>(µg/m³)",
                           font=dict(family="Barlow Condensed, sans-serif",
                                     size=11, color="#1A0F05"),
                           side="right"),
                tickfont=dict(family="Barlow Condensed", size=10, color="#1A0F05"),
                thickness=12, len=0.85,
                outlinecolor="#E8DCC8", outlinewidth=0.5, bgcolor="#FFFCF5",
            ),
            hovertemplate="<b>%{y}</b> · %{x}<br>PM2.5 : %{z:.1f} µg/m³<extra></extra>",
        ))
        # Dry season bracket
        for m_idx, m_name in enumerate(MONTHS_FR):
            if m_name in ("Nov", "Déc", "Jan", "Fév", "Mar"):
                fig_heat.add_shape(type="rect",
                    x0=m_idx-0.5, x1=m_idx+0.5, y0=-0.5, y1=len(top15_cities)-0.5,
                    line=dict(color="rgba(139,58,30,0.35)", width=1),
                    fillcolor="rgba(0,0,0,0)", layer="above")
        fig_heat.add_annotation(x="Nov", y=len(top15_cities)-0.3,
            text="◄ Harmattan / saison sèche", showarrow=False,
            font=dict(size=9, color="#8B3A1E", family="Barlow Condensed"),
            yanchor="bottom", xanchor="left")

        lay3 = terra_layout("Saisonnalité PM2.5 — 15 villes les plus polluées", height=490)
        lay3["xaxis"].update(side="bottom", tickfont=dict(size=12, color="#1A0F05"), showgrid=False,
                             categoryorder="array", categoryarray=MONTHS_FR)  # amélioration 5
        lay3["yaxis"].update(autorange="reversed", tickfont=dict(size=11, color="#1A0F05"),
                             showgrid=False, dtick=1, automargin=True)
        lay3["margin"].update(l=170, r=80, b=40, t=60)
        fig_heat.update_layout(**lay3)
        st.plotly_chart(fig_heat, width="stretch", config={"displayModeBar": False})

    with col_b:
        years_all = sorted(df_hist["year"].unique())
        yearly = (df_hist.groupby(["region","year"])["pm25_proxy"].mean().reset_index()
                  if "region" in df_hist.columns else
                  df_hist.groupby("year")["pm25_proxy"].mean().reset_index())

        fig_yr = go.Figure()
        # WHO reference
        fig_yr.add_trace(go.Scatter(
            x=years_all, y=[15]*len(years_all), mode="lines",
            line=dict(color="#2E7D32", width=1.5, dash="dot"),
            name="OMS 15 µg/m³", hoverinfo="skip",
        ))
        if "region" in yearly.columns:
            for reg in sorted(yearly["region"].unique()):
                sub = yearly[yearly["region"] == reg].sort_values("year")
                color = REGION_PALETTE.get(reg, "#8B7355")
                fig_yr.add_trace(go.Scatter(
                    x=sub["year"], y=sub["pm25_proxy"],
                    mode="lines+markers", name=reg,
                    line=dict(color=color, width=2),
                    marker=dict(color=color, size=6,
                                line=dict(color="rgba(255,255,255,0.5)", width=1)),
                    hovertemplate=f"<b>{reg}</b> : %{{y:.1f}} µg/m³<extra></extra>",
                    showlegend=False,
                ))
                # Label fin de ligne à droite
                last = sub.iloc[-1]
                fig_yr.add_annotation(
                    x=last["year"], y=last["pm25_proxy"],
                    text=f"<b>{reg}</b>",
                    showarrow=False, xanchor="left", xshift=6,
                    font=dict(size=10, color=color, family="Barlow Condensed"),
                )
        else:
            fig_yr.add_trace(go.Scatter(
                x=yearly["year"], y=yearly["pm25_proxy"],
                mode="lines+markers", name="Nationale",
                line=dict(color="#8B3A1E", width=2), marker=dict(size=7)))

        lay4 = terra_layout("Évolution inter-annuelle — PM2.5 par région", height=490)
        lay4["xaxis"].update(title=dict(text="Année", font=dict(size=11, color="#1A0F05")),
                             tickmode="array", tickvals=years_all,
                             ticktext=[str(y) for y in years_all])
        lay4["yaxis"].update(title=dict(text="PM2.5 moyen (µg/m³)",
                                        font=dict(size=11, color="#1A0F05")))
        lay4["showlegend"] = False
        lay4["hovermode"] = "x unified"
        lay4["margin"].update(r=120)   # marge droite réduite — labels inline
        fig_yr.update_layout(**lay4)
        st.plotly_chart(fig_yr, width="stretch", config={"displayModeBar": False})

    # ── Export CSV classement ────────────────────────────────────────────────
    rank_csv = city_stats[["city","mean","std","aqi"]].rename(columns={
        "city":"Ville","mean":"PM2.5 moyen µg/m³","std":"Écart-type","aqi":"Catégorie AQI"
    }).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Exporter CSV — classement 40 villes",
        data=rank_csv,
        file_name="pm25_classement_cameroun_2020_2025.csv",
        mime="text/csv",
    )

    # ── Stats panel ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        '<div style="font-family:Playfair Display,Georgia,serif;font-size:20px;'
        'color:#5C1F0A;margin:0 0 16px;padding-bottom:8px;'
        'border-bottom:2px solid #8B3A1E;">Performance des modèles de prédiction</div>',
        unsafe_allow_html=True,
    )

    # Model cards
    mc1, mc2, mc3 = st.columns(3)
    def _model_card(col, name, r2, rmse, best=False):
        border = "2px solid #8B3A1E" if best else "1px solid #E8DCC8"
        badge  = ('<span style="background:#8B3A1E;color:#fff;font-size:10px;'
                  'padding:2px 8px;border-radius:10px;font-family:Barlow Condensed,'
                  'sans-serif;margin-left:8px;">Meilleur</span>') if best else ""
        col.markdown(
            f'<div style="background:#FFFCF5;border:{border};border-radius:8px;'
            f'padding:16px 20px;text-align:center;">'
            f'<div style="font-family:Playfair Display,serif;font-size:15px;'
            f'color:#5C1F0A;font-weight:700;margin-bottom:12px;">{name}{badge}</div>'
            f'<div style="display:flex;justify-content:space-around;gap:12px;">'
            f'<div><div style="font-family:Barlow Condensed,sans-serif;font-size:11px;'
            f'color:#8B7355;margin-bottom:2px;">R²</div>'
            f'<div style="font-family:Barlow Condensed,sans-serif;font-size:24px;'
            f'font-weight:600;color:#8B3A1E;">{r2:.4f}</div></div>'
            f'<div style="border-left:1px solid #E8DCC8;"></div>'
            f'<div><div style="font-family:Barlow Condensed,sans-serif;font-size:11px;'
            f'color:#8B7355;margin-bottom:2px;">RMSE</div>'
            f'<div style="font-family:Barlow Condensed,sans-serif;font-size:24px;'
            f'font-weight:600;color:#C8941A;">{rmse:.2f}'
            f'<span style="font-size:13px"> µg/m³</span></div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    _model_card(mc1, "XGBoost",  0.9929, 1.66)
    _model_card(mc2, "LightGBM", 0.9940, 1.53, best=True)
    _model_card(mc3, "Ensemble", 0.9939, 1.55)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    fi_col, cv_col = st.columns([3, 2], gap="large")

    with fi_col:
        st.markdown(
            '<div style="font-family:Barlow Condensed,sans-serif;font-size:13px;'
            'color:#8B3A1E;font-weight:600;margin-bottom:10px;letter-spacing:0.5px;">'
            'IMPORTANCE DES VARIABLES</div>',
            unsafe_allow_html=True,
        )
        feat_items = [
            ("is_true_harmattan",        41.6, "#8B3A1E"),
            ("climate_zone",             18.2, "#C8941A"),
            ("pm25_proxy_roll3_mean",    12.5, "#C8941A"),
            ("blh_mean",                  7.8, "#8B7355"),
            ("precipitation_sum",         5.4, "#8B7355"),
            ("wind_speed_10m_max",        4.1, "#8B7355"),
            ("latitude",                  3.3, "#8B7355"),
        ]
        max_fi = feat_items[0][1]
        for feat, pct, color in feat_items:
            bar_w = int(pct / max_fi * 100)
            st.markdown(
                f'<div style="margin-bottom:8px;">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
                f'<span style="font-family:Barlow Condensed,sans-serif;font-size:12px;'
                f'color:#1A0F05;">{feat}</span>'
                f'<span style="font-family:Barlow Condensed,sans-serif;font-size:12px;'
                f'color:{color};font-weight:600;">{pct:.1f}%</span></div>'
                f'<div style="background:#E8DCC8;border-radius:3px;height:8px;">'
                f'<div style="background:{color};width:{bar_w}%;height:8px;'
                f'border-radius:3px;"></div></div></div>',
                unsafe_allow_html=True,
            )

    with cv_col:
        st.markdown(
            '<div style="font-family:Barlow Condensed,sans-serif;font-size:13px;'
            'color:#8B3A1E;font-weight:600;margin-bottom:10px;letter-spacing:0.5px;">'
            'VALIDATION CROISÉE EXPANDING-WINDOW</div>',
            unsafe_allow_html=True,
        )
        cv_data = [
            ("Fold 1", "2021",  0.970, 0.967),
            ("Fold 2", "2022",  0.994, 0.994),
            ("Fold 3", "2023",  0.994, 0.994),
            ("Fold 4", "2024",  0.994, 0.994),
            ("Test",   "2025",  0.993, 0.994),
        ]
        max_xgb = max(r[2] for r in cv_data)
        max_lgb = max(r[3] for r in cv_data)
        def _cv_row_bg(fold, xgb, lgb):
            if fold == "Test":
                return "#FBE9E7"    # rouge pâle — test set
            if xgb == max_xgb and lgb == max_lgb:
                return "#FFF8E1"    # or pâle — meilleur fold
            return "#FFFCF5"
        def _cv_row_html(fold, val, xgb, lgb):
            bg   = _cv_row_bg(fold, xgb, lgb)
            best_badge = ("&nbsp;<span style='font-size:9px;background:#C8941A;color:#fff;"
                          "padding:1px 5px;border-radius:2px;'>MEILLEUR</span>"
                          if xgb == max_xgb and lgb == max_lgb else "")
            test_badge = ("&nbsp;<span style='font-size:9px;background:#8B3A1E;color:#fff;"
                          "padding:1px 5px;border-radius:2px;'>TEST</span>"
                          if fold == "Test" else "")
            xgb_w = "font-weight:700;" if xgb == max_xgb else ""
            lgb_w = "font-weight:700;" if lgb == max_lgb else ""
            return (
                f'<tr style="background:{bg};">'
                f'<td style="padding:6px 8px;font-weight:600;">{fold}{best_badge}{test_badge}</td>'
                f'<td style="padding:6px 8px;color:#8B7355;">{val}</td>'
                f'<td style="padding:6px 8px;text-align:right;color:#8B3A1E;{xgb_w}">{xgb:.3f}</td>'
                f'<td style="padding:6px 8px;text-align:right;color:#C8941A;{lgb_w}">{lgb:.3f}</td>'
                f'</tr>'
            )
        rows_html = "".join(_cv_row_html(*row) for row in cv_data)
        st.markdown(
            f'<table style="width:100%;border-collapse:collapse;'
            f'font-family:Barlow Condensed,sans-serif;font-size:12px;">'
            f'<thead><tr style="border-bottom:2px solid #8B3A1E;">'
            f'<th style="padding:6px 8px;text-align:left;color:#5C1F0A;">Fold</th>'
            f'<th style="padding:6px 8px;text-align:left;color:#5C1F0A;">Val</th>'
            f'<th style="padding:6px 8px;text-align:right;color:#8B3A1E;">XGB R²</th>'
            f'<th style="padding:6px 8px;text-align:right;color:#C8941A;">LGB R²</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        # Ablation
        st.markdown(
            '<div style="font-family:Barlow Condensed,sans-serif;font-size:13px;'
            'color:#8B3A1E;font-weight:600;margin-bottom:10px;letter-spacing:0.5px;">'
            'ÉTUDE D\'ABLATION</div>',
            unsafe_allow_html=True,
        )
        for abl_name, abl_r2, abl_color in [
            ("Persistance (baseline)",    0.745, "#B71C1C"),
            ("Météo uniquement (XGBoost)",0.993, "#2E7D32"),
            ("Full (météo + lags)",       0.993, "#2E7D32"),
        ]:
            bar_w = int(abl_r2 * 100)
            st.markdown(
                f'<div style="margin-bottom:10px;">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
                f'<span style="font-family:Barlow Condensed,sans-serif;font-size:11px;'
                f'color:#1A0F05;">{abl_name}</span>'
                f'<span style="font-family:Barlow Condensed,sans-serif;font-size:11px;'
                f'color:{abl_color};font-weight:600;">R²={abl_r2:.3f}</span></div>'
                f'<div style="background:#E8DCC8;border-radius:3px;height:6px;">'
                f'<div style="background:{abl_color};width:{bar_w}%;height:6px;'
                f'border-radius:3px;"></div></div></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div style="font-family:Barlow Condensed,sans-serif;font-size:10px;'
            'color:#E64A19;margin-top:6px;font-style:italic;">'
            '→ Le modèle apprend la physique, pas l\'autocorrélation.</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:12px 0 20px;">
            <div style="font-size:42px;">🌿</div>
            <div style="
                font-family:'Playfair Display',serif;
                font-size:20px;font-weight:700;
                color:#F5EDD9;margin-top:6px;
            ">PM2.5 Cameroun</div>
            <div style="
                font-family:'Barlow Condensed',sans-serif;
                font-size:10px;font-weight:700;letter-spacing:0.18em;
                text-transform:uppercase;color:rgba(245,237,217,0.5);
                margin-top:3px;
            ">INDABAX 2026</div>
        </div>
        """, unsafe_allow_html=True)

        # AQI Legend
        st.markdown("""
        <div style="
            font-family:'Barlow Condensed',sans-serif;
            font-size:10px;font-weight:700;letter-spacing:0.18em;
            text-transform:uppercase;color:rgba(245,237,217,0.4);
            margin-bottom:10px;border-top:1px solid rgba(255,255,255,0.1);
            padding-top:16px;
        ">Seuils OMS 2021 · 24h</div>
        """, unsafe_allow_html=True)

        ranges = ["0–12", "12–35", "35–55", "55–150", "≥ 150"]
        for (_, hi, label, color, bg, soft), rng in zip(AQI_THRESHOLDS, ranges):
            st.markdown(f"""
            <div style="
                display:flex;align-items:center;justify-content:space-between;
                padding:6px 10px;margin-bottom:4px;
                background:rgba(255,255,255,0.06);border-radius:3px;
                border-left:3px solid {color};
            ">
                <span style="font-family:'Barlow',sans-serif;font-size:12px;
                             color:rgba(245,237,217,0.85);">{label}</span>
                <span style="font-family:'DM Mono',monospace;font-size:10px;
                             color:rgba(245,237,217,0.45);">{rng}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="
            margin-top:20px;padding-top:16px;
            border-top:1px solid rgba(255,255,255,0.1);
            font-family:'Barlow',sans-serif;font-size:12px;
            color:rgba(245,237,217,0.45);line-height:1.8;
        ">
            <b style="color:rgba(245,237,217,0.7);">Modèles</b><br>
            XGBoost · LightGBM<br>
            R² = 0.994 · RMSE = 1.55 µg/m³<br><br>
            <b style="color:rgba(245,237,217,0.7);">Sources</b><br>
            ERA5 · Open-Meteo<br>
            NASA FIRMS MODIS<br><br>
            <b style="color:rgba(245,237,217,0.7);">Mise à jour</b><br>
            Quotidienne · 06:00 UTC<br>
            GitHub Actions
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="TERRA — PM2.5 Cameroun",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(TERRA_CSS, unsafe_allow_html=True)
    render_sidebar()

    # ── Navigation par session_state (permet navigation P1→P2 programmatique) ─
    # _nav_request : clé écrite par les boutons (jamais liée à un widget)
    # _nav_page    : clé liée au widget radio — on l'écrit AVANT le render (autorisé)
    _TAB_LABELS = ["TEMPS RÉEL", "PAR VILLE", "CLASSEMENT"]

    # Consommer la demande de navigation et injecter dans la clé du widget
    # AVANT que st.radio soit instancié → écriture autorisée par Streamlit
    _pending = st.session_state.pop("_nav_request", None)
    if _pending in _TAB_LABELS:
        st.session_state["_nav_page"] = _pending

    if "_nav_page" not in st.session_state:
        st.session_state["_nav_page"] = _TAB_LABELS[0]

    active_page = st.radio(
        "Navigation",
        _TAB_LABELS,
        horizontal=True,
        label_visibility="collapsed",
        key="_nav_page",          # widget lit session_state["_nav_page"] à chaque render
    )

    if active_page == "TEMPS RÉEL":
        page_realtime()
    elif active_page == "PAR VILLE":
        page_city()
    else:
        page_ranking()


if __name__ == "__main__":
    main()

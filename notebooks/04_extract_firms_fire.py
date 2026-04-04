"""
Extraction des données feux NASA FIRMS — Facteur F_fire pour le proxy PM2.5
Hackathon IndabaX Cameroun 2026

Justification scientifique :
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Les feux de biomasse sont le facteur manquant le plus important dans
notre proxy PM2.5 pour le Cameroun. Gordon et al. (2023, GeoHealth)
montrent que les émissions des feux contribuent massivement au PM2.5
en Afrique Centrale, avec un impact sur la mortalité comparable aux
sources anthropiques. Le FRP (Fire Radiative Power) mesuré par MODIS/VIIRS
est un proxy direct de la quantité de particules émises.

Formule :
  F_fire = 1 + c × log(1 + FRP_radius_sum)
  c = 0.02  (calibration conservatrice — Afrique Centrale)
  FRP agrégé dans un rayon de 75 km autour de chaque ville

Prérequis :
  Un MAP_KEY NASA FIRMS GRATUIT — obtenu en 5 minutes à :
  https://firms.modaps.eosdis.nasa.gov/api/map_key/
  → entrer l'email, recevoir la clé par mail, copier dans MAP_KEY ci-dessous

Données :
  MODIS Collection 6.1 (Standard Processing) — archive 2000–présent
  Résolution spatiale : ~1 km par pixel feu
  Colonne FRP : Fire Radiative Power en Megawatts (MW)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from io import StringIO

# Chargement des variables d'environnement depuis .env (si python-dotenv disponible)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass   # python-dotenv optionnel — os.environ suffit si la variable est déjà exportée

warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#   CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAP_KEY = os.environ.get("FIRMS_MAP_KEY", "")
if not MAP_KEY:
    raise EnvironmentError(
        "Variable FIRMS_MAP_KEY manquante.\n"
        "  → Créez un fichier .env à la racine du projet avec :\n"
        "      FIRMS_MAP_KEY=votre_cle_32_caracteres\n"
        "  → Ou exportez-la dans votre shell : export FIRMS_MAP_KEY=...\n"
        "  → Clé gratuite sur : https://firms.modaps.eosdis.nasa.gov/api/map_key/"
    )

START_YEAR  = 2020
END_YEAR    = 2025
RADIUS_KM   = 75      # rayon d'agrégation autour de chaque ville (km)
C_FIRE      = 0.02    # coefficient d'impact feu (calibré Afrique Centrale)
                      # Plage plausible : 0.01–0.05 (Gordon et al. 2023)
                      # c=0.02 → pic de feux (FRP=1000 MW) donne F_fire ≈ 1.14

DATA_SOURCE = "MODIS_SP"    # MODIS Standard Processing (meilleur pour historique)
#               Alternatives : "VIIRS_SNPP_SP" (plus précis depuis 2012)
#               Pour 2020-2025, les deux fonctionnent

# Bounding box du Cameroun (marge +1° pour inclure les feux transfrontaliers)
CAMEROON_BBOX = "7,1,17,14"   # west,south,east,north (lon_min,lat_min,lon_max,lat_max)
CHUNK_DAYS    = 5              # FIRMS area/csv : max 5 jours par requête pour archive SP

OUTPUT_DIR  = Path("data/firms_raw")
OUTPUT_FILE = Path("data/firms_fire_daily.parquet")
ERA5_FILE   = Path("data/era5_features_all_cities.parquet")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


# ── Coordonnées des 40 villes ──────────────────────────────────────────────────
# (Reprises depuis 01_extract_era5_pm25_target.py)
CITIES = [
    ("Tibati",        "Adamaoua",    6.4667,  12.6167),
    ("Tignere",       "Adamaoua",    7.3700,  12.6500),
    ("Ngaoundere",    "Adamaoua",    7.3167,  13.5833),
    ("Meiganga",      "Adamaoua",    6.5167,  14.2833),
    ("Bafia",         "Centre",      4.7500,  11.2300),
    ("Akonolinga",    "Centre",      3.7667,  12.2500),
    ("Yaounde",       "Centre",      3.8667,  11.5167),
    ("Mbalmayo",      "Centre",      3.5167,  11.5017),
    ("Yokadouma",     "Est",         3.5139,  15.0539),
    ("Batouri",       "Est",         4.4333,  14.3667),
    ("Bertoua",       "Est",         4.5833,  13.6833),
    ("Abong-Mbang",   "Est",         3.9833,  13.1833),
    ("Yagoua",        "Extreme-Nord",10.3500,  15.2333),
    ("Maroua",        "Extreme-Nord",10.5833,  14.3167),
    ("Kousseri",      "Extreme-Nord",12.0833,  15.0333),
    ("Mokolo",        "Extreme-Nord", 9.3500,  13.7333),
    ("Loum",          "Littoral",    4.7167,   9.7333),
    ("Douala",        "Littoral",    4.0483,   9.7043),
    ("Edea",          "Littoral",    3.8000,  10.1333),
    ("Nkongsamba",    "Littoral",    4.9500,   9.9333),
    ("Garoua",        "Nord",        9.2992,  13.3954),
    ("Guider",        "Nord",        9.9333,  13.9500),
    ("Poli",          "Nord",        8.4833,  13.2333),
    ("Touboro",       "Nord",        7.7667,  15.3667),
    ("Bamenda",       "Nord-Ouest",  5.9631,  10.1597),
    ("Kumbo",         "Nord-Ouest",  6.2000,  10.6667),
    ("Wum",           "Nord-Ouest",  6.3833,  10.0667),
    ("Mbengwi",       "Nord-Ouest",  5.9833,  10.0167),
    ("Mbouda",        "Ouest",       5.6167,  10.2667),
    ("Foumban",       "Ouest",       5.7167,  10.9000),
    ("Bafoussam",     "Ouest",       5.4765,  10.4162),
    ("Dschang",       "Ouest",       5.4500,  10.0500),
    ("Ebolowa",       "Sud",         2.9000,  11.1500),
    ("Kribi",         "Sud",         2.9333,   9.9167),
    ("Sangmelima",    "Sud",         2.9333,  11.9833),
    ("Ambam",         "Sud",         2.3833,  11.2833),
    ("Kumba",         "Sud-Ouest",   4.6333,   9.4500),
    ("Buea",          "Sud-Ouest",   4.1667,   9.2333),
    ("Limbe",         "Sud-Ouest",   4.0167,   9.2100),
    ("Mamfe",         "Sud-Ouest",   5.7667,   9.3167),
]


# ── 1. Distance Haversine ──────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2_arr, lon2_arr):
    """
    Distance en km entre (lat1, lon1) et un tableau de points (lat2, lon2).
    Vectorisé pour filtrer rapidement les pixels feux dans le rayon.
    """
    R = 6371.0
    dlat = np.radians(lat2_arr - lat1)
    dlon = np.radians(lon2_arr - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2_arr))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


# ── 2. Téléchargement FIRMS par année ─────────────────────────────────────────

def download_firms_year(year: int, source: str = DATA_SOURCE) -> pd.DataFrame:
    """
    Télécharge toutes les détections de feux pour le Cameroun
    pour une année donnée via l'API FIRMS.

    FIRMS Area API (bbox) :
    https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{BBOX}/{DAY_RANGE}/{DATE}
      BBOX    : west,south,east,north (degrés décimaux)
      DAY_RANGE : 1–5 pour les archives SP (Standard Processing)

    Stratégie : tranches de 5 jours pour respecter la limite SP et les timeouts.
    Rate limit : 5000 transactions / 10 min (largement suffisant).
    """
    cache_path = OUTPUT_DIR / f"firms_{source}_CMR_{year}.parquet"
    if cache_path.exists():
        log.info(f"  [{year}] Cache trouvé : {cache_path.name}")
        return pd.read_parquet(cache_path)

    base_url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{source}/{CAMEROON_BBOX}"
    start = pd.Timestamp(f"{year}-01-01")
    end   = pd.Timestamp(f"{year}-12-31")
    dates = pd.date_range(start, end, freq=f"{CHUNK_DAYS}D")

    all_dfs = []
    for i, date in enumerate(dates):
        # Nombre de jours pour cette tranche (max CHUNK_DAYS, ajusté fin d'année)
        remaining = (end - date).days + 1
        n_days    = min(CHUNK_DAYS, remaining)
        url       = f"{base_url}/{n_days}/{date.strftime('%Y-%m-%d')}"

        for attempt in range(1, 4):
            try:
                r = requests.get(url, timeout=120)   # 120s : SP archive peut être lente
                if r.status_code == 429:
                    log.warning(f"  Rate limit — attente 60s (tranche {i+1})")
                    time.sleep(60)
                    continue
                if r.status_code == 400:
                    # "No data" ou période sans feux
                    break
                r.raise_for_status()
                if r.text.strip() and "latitude" in r.text:
                    df = pd.read_csv(StringIO(r.text))
                    all_dfs.append(df)
                    log.info(f"  [{year}] tranche {i+1}/{len(dates)} : {len(df)} pixels")
                else:
                    log.info(f"  [{year}] tranche {i+1}/{len(dates)} : 0 pixels (zone vide)")
                break
            except Exception as e:
                log.warning(f"  Erreur tranche {i+1} (essai {attempt}/3) : {e}")
                time.sleep(15 * attempt)

        time.sleep(1.0)  # pause polie entre requêtes

    if not all_dfs:
        log.warning(f"  [{year}] Aucune donnée reçue")
        return pd.DataFrame()

    df_year = pd.concat(all_dfs, ignore_index=True)
    # Colonnes minimales nécessaires
    needed = {"latitude", "longitude", "acq_date", "frp"}
    if not needed.issubset(df_year.columns):
        log.warning(f"  [{year}] Colonnes manquantes : {needed - set(df_year.columns)}")
        return pd.DataFrame()

    df_year["acq_date"] = pd.to_datetime(df_year["acq_date"])
    df_year = df_year.drop_duplicates(
        subset=["latitude", "longitude", "acq_date"]
    )

    log.info(f"  [{year}] {len(df_year):,} pixels feux téléchargés")

    # Sauvegarde cache
    df_year.to_parquet(cache_path, index=False)
    return df_year


# ── 3. Agrégation spatiale : FRP par ville par jour ──────────────────────────

def aggregate_frp_by_city(df_fires: pd.DataFrame,
                           cities: list,
                           radius_km: float = RADIUS_KM) -> pd.DataFrame:
    """
    Pour chaque ville et chaque jour, somme le FRP de tous les pixels
    feux détectés dans un rayon de `radius_km` km autour de la ville.

    Pourquoi 75 km ?
    - 50 km : capture les feux très locaux mais rate le transport régional
    - 100 km : trop de bruit dans les zones denses
    - 75 km : compromis documenté pour les zones sub-sahariennes
      (durée de transport ~6-24h pour particules fines = ~50-100 km)

    Retourne un DataFrame : city | time | frp_sum | n_fires
    """
    if df_fires.empty:
        return pd.DataFrame(columns=["city", "time", "frp_sum", "n_fires"])

    fire_lats = df_fires["latitude"].values
    fire_lons = df_fires["longitude"].values
    fire_frp  = df_fires["frp"].fillna(0).values
    fire_date = df_fires["acq_date"].values

    records = []
    for city, region, lat, lon in cities:
        # Distance de tous les pixels feux à cette ville
        dists = haversine_km(lat, lon, fire_lats, fire_lons)
        mask  = dists <= radius_km

        if mask.sum() == 0:
            continue

        df_local = pd.DataFrame({
            "date": fire_date[mask],
            "frp":  fire_frp[mask]
        })
        df_agg = df_local.groupby("date").agg(
            frp_sum=("frp", "sum"),
            n_fires=("frp", "count")
        ).reset_index()
        df_agg["city"] = city
        records.append(df_agg)

    if not records:
        return pd.DataFrame(columns=["city", "time", "frp_sum", "n_fires"])

    result = pd.concat(records, ignore_index=True)
    result = result.rename(columns={"date": "time"})
    result["time"] = pd.to_datetime(result["time"])
    return result


# ── 4. Calcul de F_fire ────────────────────────────────────────────────────────

def compute_f_fire(frp_sum: pd.Series, c: float = C_FIRE) -> pd.Series:
    """
    F_fire = 1 + c × log(1 + FRP_sum)

    Justification :
    - log(1+x) : forme sous-linéaire — évite que les feux extrêmes
      dominent tout (saturation physique : la fumée elle-même bloque le rayonnement)
    - c = 0.02 : pour FRP=1000 MW (grand feu régional), F_fire ≈ 1.14
                 pour FRP=100  MW (feu modéré),         F_fire ≈ 1.09
                 pour FRP=0    (pas de feu),             F_fire = 1.00

    Plage de c documentée : 0.01–0.05 selon les régions africaines.
    Recommandation : calibrer c en maximisant la corrélation proxy vs CAMS.
    """
    return 1.0 + c * np.log1p(frp_sum.fillna(0).clip(lower=0))


# ── 5. Pipeline principal ──────────────────────────────────────────────────────

def main():
    # Vérification MAP_KEY
    if MAP_KEY == "VOTRE_MAP_KEY_ICI":
        print("""
╔══════════════════════════════════════════════════════════════╗
║  MAP_KEY manquante !                                         ║
║                                                              ║
║  1. Allez sur : https://firms.modaps.eosdis.nasa.gov/api/map_key/
║  2. Entrez votre email → recevez la clé par mail (5 min)     ║
║  3. Remplacez MAP_KEY = "VOTRE_MAP_KEY_ICI" dans ce script   ║
╚══════════════════════════════════════════════════════════════╝
""")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("Extraction NASA FIRMS — Feux de biomasse Cameroun 2020-2025")
    log.info(f"  Source  : {DATA_SOURCE}")
    log.info(f"  Rayon   : {RADIUS_KM} km | c_fire = {C_FIRE}")
    log.info("=" * 65)

    # ── 5.1 Téléchargement FIRMS par année ─────────────────────────────────
    all_fires = []
    for year in range(START_YEAR, END_YEAR + 1):
        log.info(f"[Année {year}]")
        df_year = download_firms_year(year, DATA_SOURCE)
        if not df_year.empty:
            all_fires.append(df_year)

    if not all_fires:
        log.error("Aucune donnée FIRMS téléchargée. Vérifiez votre MAP_KEY.")
        return

    df_all_fires = pd.concat(all_fires, ignore_index=True)
    log.info(f"Total pixels feux : {len(df_all_fires):,}")

    # ── 5.2 Agrégation spatiale par ville/jour ──────────────────────────────
    log.info("Agrégation spatiale (rayon 75 km)...")
    df_frp = aggregate_frp_by_city(df_all_fires, CITIES, RADIUS_KM)
    log.info(f"  Ville-jours avec feux : {len(df_frp):,}")

    # ── 5.3 Expansion vers le dataset complet (tous les jours, toutes villes)
    # Créer l'index complet ville × date
    all_cities_names = [c[0] for c in CITIES]
    all_dates = pd.date_range(
        f"{START_YEAR}-01-01", f"{END_YEAR}-12-20", freq="D"
    )
    idx = pd.MultiIndex.from_product(
        [all_cities_names, all_dates], names=["city", "time"]
    )
    df_full = pd.DataFrame(index=idx).reset_index()
    df_full = df_full.merge(df_frp, on=["city", "time"], how="left")
    df_full["frp_sum"]  = df_full["frp_sum"].fillna(0)
    df_full["n_fires"]  = df_full["n_fires"].fillna(0).astype(int)

    # ── 5.4 Calcul F_fire ───────────────────────────────────────────────────
    df_full["f_fire"] = compute_f_fire(df_full["frp_sum"], C_FIRE)

    # ── 5.5 Statistiques ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STATISTIQUES F_fire")
    print("=" * 65)
    print(f"Jours sans feu (FRP=0)      : {(df_full['frp_sum']==0).sum():,} "
          f"({100*(df_full['frp_sum']==0).mean():.1f}%)")
    print(f"Jours avec feu              : {(df_full['frp_sum']>0).sum():,}")
    print(f"FRP max observé             : {df_full['frp_sum'].max():.0f} MW")
    print(f"F_fire moyen                : {df_full['f_fire'].mean():.4f}")
    print(f"F_fire max (pic)            : {df_full['f_fire'].max():.4f}")
    print()
    # Top 5 villes les plus touchées par les feux
    city_fire = df_full[df_full["frp_sum"]>0].groupby("city")["frp_sum"].mean()
    print("Top 5 villes — FRP moyen (jours avec feux) :")
    print(city_fire.sort_values(ascending=False).head(5).round(1).to_string())
    print("=" * 65)

    # ── 5.6 Sauvegarde ──────────────────────────────────────────────────────
    df_full.to_parquet(OUTPUT_FILE, index=False)
    log.info(f"Sauvegardé : {OUTPUT_FILE}  {df_full.shape}")

    print("""
╔══════════════════════════════════════════════════════════════╗
║  PROCHAINE ÉTAPE                                             ║
║                                                              ║
║  Relancer 02_build_pm25_target.py — il va automatiquement    ║
║  détecter data/firms_fire_daily.parquet et intégrer F_fire   ║
║  dans le calcul du proxy PM2.5.                              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()

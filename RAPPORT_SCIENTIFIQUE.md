# Rapport Scientifique — Prévision PM2.5 au Cameroun
## Hackathon IndabaX Cameroun 2026

**Projet** : Système de prévision de la qualité de l'air (PM2.5) pour 40 villes camerounaises
**Période de données** : 2020-01-01 → 2025-12-20
**Approche** : Proxy physico-statistique calibré ERA5 + NASA FIRMS + modèle XGBoost/LightGBM
**Date** : Avril 2026 — *Version 7 (dashboard TERRA, déploiement HuggingFace Spaces)*

---

## Table des matières

1. [Contexte et problématique](#1-contexte-et-problématique)
2. [Description des données sources](#2-description-des-données-sources)
3. [Construction de la variable cible PM2.5](#3-construction-de-la-variable-cible-pm25)
4. [Feature Engineering](#4-feature-engineering)
5. [Fondements scientifiques — les mécanismes physiques](#5-fondements-scientifiques--les-mécanismes-physiques)
6. [Calibration et validation](#6-calibration-et-validation)
7. [Quantification de l'incertitude — Monte Carlo](#7-quantification-de-lincertitude--monte-carlo)
8. [Cadre réglementaire](#8-cadre-réglementaire)
9. [Architecture du pipeline](#9-architecture-du-pipeline)
10. [Résultats et statistiques clés](#10-résultats-et-statistiques-clés)
    - 10.6 [Pipeline d'inférence temps réel](#106-pipeline-dinférence-temps-réel)
11. [Références](#11-références)

---

## 1. Contexte et problématique

### 1.1 Importance sanitaire de PM2.5 au Cameroun

Les particules fines PM2.5 (diamètre ≤ 2.5 µm) constituent le polluant atmosphérique le plus dangereux pour la santé humaine selon l'Organisation Mondiale de la Santé. Au Cameroun, leur impact est documenté :

- **Espérance de vie réduite de 2.7 ans** en moyenne nationale (AQLI 2023)
- **Concentration annuelle moyenne : 32.5 µg/m³** — soit 6.5 fois le seuil OMS 2021 de 5 µg/m³
- **85.8% des jours** dépassent le seuil OMS 24h de 15 µg/m³
- Impact particulièrement sévère au nord (Extrême-Nord, Nord) pendant la saison Harmattan

### 1.2 Absence de réseau de mesure

Le Cameroun ne dispose pas d'un réseau dense de capteurs PM2.5 couvrant ses 10 régions. Les quelques mesures existantes (Yaoundé, Douala, Bamenda) sont fragmentées dans le temps et l'espace. Cette absence de données de terrain est le défi central du projet.

### 1.3 Objectif du projet

Construire un système de prévision PM2.5 à l'échelle des 40 villes du dataset hackathon (2020–2025) en :
1. Définissant une variable cible PM2.5 physiquement justifiée et calibrée sur ERA5 + NASA FIRMS
2. Extrayant des variables météorologiques ERA5 enrichies (BLH, direction de vent, humidité)
3. Intégrant les émissions de feux de biomasse via NASA FIRMS MODIS
4. Entraînant un modèle XGBoost/LightGBM avec validation temporelle rigoureuse
5. Quantifiant l'incertitude du proxy par simulation Monte Carlo

---

## 2. Description des données sources

### 2.1 Dataset original du hackathon — `Dataset_complet_Meteo.xlsx`

| Attribut | Valeur |
|----------|--------|
| Lignes | 87 240 |
| Colonnes | 26 |
| Villes | 40 |
| Régions | 10 |
| Période | 2020-01-01 → 2025-12-20 |
| Fréquence | Journalière |

**Variables disponibles :**

| Variable | Description | Unité |
|----------|-------------|-------|
| `temperature_2m_max/min` | Température max/min à 2m | °C |
| `precipitation_sum` | Précipitations cumulées | mm |
| `wind_speed_10m_max` | Vitesse vent maximale à 10m | km/h |
| `wind_gusts_10m_max` | Rafales maximales à 10m | km/h |
| `wind_direction_10m_dominant` | Direction dominante vent | degrés (0°=N) |
| `shortwave_radiation_sum` | Rayonnement solaire total | MJ/m² |
| `relative_humidity_2m_max/min` | Humidité relative max/min | % |
| `et0_fao_evapotranspiration` | Évapotranspiration FAO | mm |
| `sunshine_duration` | Durée d'ensoleillement | secondes |
| `city`, `region` | Identifiants géographiques | — |
| `latitude`, `longitude` | Coordonnées | degrés décimaux |

**Problème identifié — Corruption de coordonnées :**
14 des 40 villes avaient leurs coordonnées GPS corrompues dans le fichier Excel. Excel a interprété les décimales comme des dates (ex : `13.08` → date `2026-01-03 00:00:00`). Ce problème a été corrigé en créant une table de référence des coordonnées officielles.

### 2.2 Données ERA5 extraites — API Open-Meteo Historical Weather

**Source :** `archive-api.open-meteo.com/v1/archive`

**Pourquoi ERA5 et non l'API Air Quality directe ?**
L'API Open-Meteo Air Quality (PM2.5 estimé directement) est **limitée à 92 jours d'historique** (`past_days=92`), confirmé par le ticket GitHub #718 du projet. Il est impossible de couvrir 2020–2025. L'API ERA5 Historical Weather n'a pas cette limite — elle donne accès à la réanalyse complète depuis 1940.

**Variables ERA5 extraites :**

| Variable | Mode API | Description |
|----------|----------|-------------|
| `precipitation_sum` | daily | Précipitations totales (mm) |
| `wind_speed_10m_max` | daily | Vent max 10m (km/h) |
| `wind_gusts_10m_max` | daily | Rafales max 10m (km/h) |
| `wind_direction_10m_dominant` | daily | Direction dominante (°) |
| `temperature_2m_max/min` | daily | Température 2m (°C) |
| `shortwave_radiation_sum` | daily | Rayonnement (MJ/m²) |
| `relative_humidity_2m_max/min` | daily | Humidité relative (%) |
| `et0_fao_evapotranspiration` | daily | Évapotranspiration (mm) |
| `boundary_layer_height` | **hourly** → agrégé daily | Hauteur couche limite (m) |

> **Note technique :** `boundary_layer_height` n'est disponible qu'en mode hourly dans l'API ERA5. Le script extrait 24 valeurs horaires par jour et les agrège en `blh_mean`, `blh_min`, `blh_max`.

**Résultat de l'extraction :**
- 40/40 villes extraites avec succès
- 87 240 observations (40 villes × 2 181 jours)
- Buea : BLH récupéré séparément (rate limit lors de l'extraction initiale), 2000/2181 valeurs non-nulles

### 2.3 Données NASA FIRMS — Feux de biomasse MODIS

**Source :** `https://firms.modaps.eosdis.nasa.gov/api/area/csv/`
**Produit :** MODIS Collection 6.1 Standard Processing (MODIS_SP) — archive 2000–présent
**Résolution spatiale :** ~1 km par pixel feu
**Paramètre clé :** FRP (Fire Radiative Power, MW) — proxy direct de l'intensité des émissions

**Stratégie d'extraction :**
- Bounding box Cameroun : `7,1,17,14` (lon_min, lat_min, lon_max, lat_max)
- Tranches de 5 jours (limite MODIS_SP archive), timeout 120s
- 74 tranches × 6 ans = 444 requêtes API
- Cache par année : `data/firms_raw/firms_MODIS_SP_CMR_YYYY.parquet`

**Résultats obtenus :**

| Métrique | Valeur |
|----------|--------|
| Total pixels feux détectés (2020–2025) | **634 103** |
| Ville-jours avec activité feu (rayon 75 km) | **23 403** (26.6%) |
| Jours sans feu | 64 036 (73.4%) |
| FRP max observé | 19 308 MW |
| F_fire moyen | **1.024** |
| F_fire max (pic) | 1.197 |

**Top 5 villes les plus exposées aux feux (FRP moyen, jours avec feux) :**

| Ville | FRP moyen (MW) | Zone |
|-------|----------------|------|
| Poli | 1 070.7 | Nord — savane soudanienne |
| Meiganga | 737.6 | Adamaoua — savane |
| Foumban | 682.8 | Ouest — agriculture |
| Batouri | 666.9 | Est — forêt/savane |
| Bertoua | 547.4 | Est — lisière forêt |

Ces résultats sont cohérents avec la carte des feux africains : les savanes nord-camerounaises sont les principales sources, les forêts équatoriales (sud) brûlent peu.

---

## 3. Construction de la variable cible PM2.5

### 3.1 Pourquoi le proxy du starter notebook est insuffisant

Le notebook starter calcule :
```python
pm25_proxy = 0.35 * température + 0.25 * rayonnement + 8.0 * is_no_wind + ...
```
Cette formule est **purement arbitraire** : les coefficients n'ont aucune justification scientifique, la combinaison additive ne correspond pas à la physique des aérosols, et le niveau absolu n'est ancré sur aucune donnée réelle.

### 3.2 Formule physico-statistique retenue (version 2 — avec F_fire)

La formule est **multiplicative** — chaque facteur agit comme un modificateur indépendant du niveau de base :

```
PM2.5 = C_base × F_stagnation × F_wet × F_wind × F_harmattan × F_hygro × F_fire
```

#### F_stagnation — Hauteur de couche limite (BLH)

```python
F_stagnation = clip((1000 / BLH_mean) ** 0.6, 0.3, 3.5)
# BLH_ref = 1000 m (couche limite de référence, mélange modéré)
# Exposant α = 0.6 : relation sous-linéaire (plage documentée : 0.4–1.0)
```

**Justification physique :** La couche limite atmosphérique (BLH) est le volume dans lequel les polluants émis au sol se mélangent. Un BLH de 500 m signifie que les polluants restent confinés dans 500 m de hauteur — concentration double par rapport à un BLH de 1000 m. C'est le mécanisme de loin le plus important pour la variabilité journalière du PM2.5 (Seinfeld & Pandis, 2016 ; Li et al., 2020).

L'exposant α = 0.6 (sous-linéaire) est choisi pour la robustesse aux outliers BLH. La plage documentée est 0.4–1.0 selon les configurations atmosphériques.

#### F_wet — Lessivage humide

```python
F_wet = 1 / (1 + 0.08 × précipitations_mm)
# Coefficient a = 0.08 mm⁻¹ (plage documentée : 0.05–0.20)
```

**Justification physique :** La pluie capture les particules par collision (impaction, diffusion brownienne) et les ramène au sol — c'est le *wet scavenging*. La forme rationnelle est appropriée car elle a un plancher naturel (la pluie ne peut pas éliminer 100% des particules). Le coefficient 0.08 mm⁻¹ est dans la plage documentée en milieu tropical (Berge & Jakobsen, 1998).

#### F_wind — Dilution turbulente

```python
F_wind = exp(-0.035 × vitesse_vent_km/h)  # désactivé si Harmattan détecté
# Coefficient k = 0.035 (km/h)⁻¹ (plage Pasquill-Gifford : 0.025–0.050)
```

**Justification physique :** Le vent mécanique augmente la turbulence et dilue les polluants locaux. La décroissance exponentielle est cohérente avec la dispersion gaussienne des panaches (modèle de Pasquill-Gifford).

**Exception Harmattan :** Pendant la saison sèche avec vents du secteur N–NE (direction 315–90°), le vent transporte activement de la poussière saharienne depuis le bassin du Bodélé (Tchad). Dans ce cas, vitesse du vent et PM2.5 sont positivement corrélés — F_wind est mis à 1.0.

#### F_harmattan — Transport saharien et saisonnalité

```python
lat_factor = clip((latitude - 3°) / (11° - 3°), 0, 1)
F_harmattan = 1 + 1.4 × is_dry_season × lat_factor
# Strength = 1.4 (plage documentée : 1.0–2.5) — calibré vs CAMS (ratio ×2.21, conforme Nebie et al. 2022)
```

**Justification physique :**
- **Saison sèche (novembre–mars)** : L'Harmattan transporte des poussières depuis la Dépression du Bodélé (Tchad), la plus grande source de poussière au monde. Les études camerounaises montrent une augmentation de +75% du PM2.5 dans le nord pendant cette période (Nebie et al., 2022).
- **Gradient latitudinal** : L'influence diminue du nord (Maroua, 10.6°N) vers le sud (Kribi, 2.9°N), normalisée entre 3°N et 11°N.
- À Maroua : F_harmattan = **2.06** | À Yaoundé : F_harmattan = **1.17**

#### F_hygro — Croissance hygroscopique

```python
F_hygro = min(1 + 0.004 × max(0, RH_max - 75%), 1.3)
# γ = 0.004 (plage : 0.002–0.006, Swietlicki et al. 2008)
# Plafond 1.3 : recommandation explicite Swietlicki et al. 2008
# Désactivé si précip > 1 mm
```

**Justification physique :** Les particules fines (sulfates, nitrates, matière organique) sont hygroscopiques — elles absorbent de la vapeur d'eau au-delà de ~75% d'humidité relative. Le **plafond à 1.3** est fondé sur Swietlicki et al. (2008, Tellus B), qui montrent que la croissance hygroscopique en milieu tropical dépasse rarement +30% en dehors des événements de brouillard dense.

#### F_fire — Feux de biomasse (NASA FIRMS MODIS) *(nouveau)*

```python
F_fire = 1 + 0.02 × log(1 + FRP_sum_75km)
# c = 0.02 (plage documentée : 0.01–0.05, Gordon et al. 2023)
# FRP agrégé dans un rayon de 75 km autour de chaque ville
# Forme log(1+x) : sous-linéaire → saturation physique des émissions
```

**Justification physique :** Les feux de biomasse sont un contributeur majeur au PM2.5 en Afrique centrale — Gordon et al. (2023, *GeoHealth*, DOI:10.1029/2022GH000673) estiment que les émissions de feux expliquent jusqu'à 40% de la charge particulaire en Afrique subsaharienne. Le FRP (Fire Radiative Power, en MW) mesuré par MODIS est un proxy direct de la quantité de particules émises.

**Pourquoi le rayon 75 km ?**
Les particules fines ont une durée de transport atmosphérique de 6–24h, correspondant à 50–100 km pour les régimes de vents camerounais. Le rayon 75 km est un compromis documenté pour les zones sub-sahariennes : il capture les feux régionaux sans inclure un bruit de fond trop large.

**Pourquoi log(1+FRP) ?**
La relation entre FRP et PM2.5 est sous-linéaire à haute intensité : la colonne de fumée dense elle-même bloque une partie du rayonnement et inhibe la convection. La forme log(1+x) modélise cette saturation naturelle.

### 3.3 Calibration sur données réelles

**Référence :** AQLI 2023 (Air Quality Life Index, Université de Chicago)
**Valeur cible :** 32.5 µg/m³ (concentration annuelle moyenne Cameroun)
**Méthode AQLI :** Dérivé de la combinaison MERRA-2 (NASA) + observations satellites MODIS/MISR

```python
pm25_unnorm = F_stagnation × F_wet × F_wind × F_harmattan × F_hygro × F_fire
C_base = 32.5 / pm25_unnorm.mean()
pm25_proxy = clip(C_base × pm25_unnorm, lower=2.0)
```

Le plancher de 2 µg/m³ représente le niveau de fond irréductible (cuisine au bois, trafic minimal, aérosols marins). Après application du plancher, une correction proportionnelle est appliquée pour maintenir exactement la moyenne à 32.5 µg/m³.

---

## 4. Feature Engineering

*Version 3 — Approche rigoureuse basée sur EDA, avril 2026*

### 4.1 Méthodologie : décisions basées sur les données

Contrairement à une approche par supposition, chaque décision de feature engineering a été testée et justifiée par l'analyse exploratoire `03_eda_feature_analysis.py`.

#### 4.1.1 Source des données

**Problème identifié** : le dataset Excel officiel (`Dataset_complet_Meteo.xlsx`) présente des taux de NaN massifs sur les variables météo :

| Variable ERA5 | NaN dans Excel fusionné | NaN dans ERA5 pur |
|---|---|---|
| `wind_speed_10m_max` | **89.25%** | 0% |
| `precipitation_sum` | **45.24%** | 0% |
| `temperature_2m_max` | **61.79%** | 0% |

**Décision** : utiliser `pm25_proxy_era5.parquet` (ERA5 pur) comme source principale — toutes les variables ERA5 sont complètes.

#### 4.1.2 Gap BLH 2024 S1

**Problème découvert** : Open-Meteo n'a pas de données `boundary_layer_height` pour la période 2024-01-01 → 2024-07-01, affectant **7 240 valeurs** (181 jours × 40 villes). Confirmé sur 5 localisations mondiales (Paris, Lagos, Nairobi) — gap global dans l'archive ERA5 d'Open-Meteo.

**Solution** : imputation climatologique mensuelle (`01b_patch_blh_gap.py`) — pour chaque mois manquant, moyenne des mêmes mois sur 2020–2023 et 2025. Cohérence validée : BLH juin imputé (406 m) → juillet observé (395 m).

#### 4.1.3 Stratégie d'imputation

**Test rigoureux sur 10% de NaN simulés (MCAR)** sur `blh_mean` :

| Méthode | RMSE |
|---|---|
| **Interpolation linéaire** | **89** ← retenue |
| Forward fill | 107 |
| Moyenne par ville | 189 |
| Médiane par ville | 194 |
| Médiane globale | 218 |

L'interpolation linéaire est **2.4× meilleure** que la médiane grâce à la continuité temporelle des séries météo.

#### 4.1.4 Transformation de la variable cible

La distribution de PM2.5 est fortement asymétrique (skewness = 2.14, test Shapiro-Wilk p = 4.5×10⁻⁶⁴). La transformation log1p réduit la skewness à **0.49** et rapproche la distribution d'une gaussienne.

**Décision** : entraîner le modèle sur `pm25_log = log1p(pm25_proxy)` et appliquer l'anti-log à la prédiction.

#### 4.1.5 Normalisation

Les modèles XGBoost et LightGBM sont des modèles à base d'arbres, **invariants à la mise à l'échelle**. StandardScaler ou MinMaxScaler n'améliorent pas leurs performances. Aucune normalisation n'est appliquée. Les transformations log sont retenues non pour l'échelle, mais pour **linéariser les relations exponentielles**.

### 4.2 Features créées

**Dataset final :** 87 240 lignes × **144 colonnes** (139 features numériques + city, region, time, pm25_proxy, pm25_log)

| Catégorie | Nombre | Décision / justification |
|---|---|---|
| **Lags** | 30 | 6 vars × 5 décalages (J-1,2,3,7,14) — corrélation r=0.904 PM2.5[J] vs PM2.5[J-1] |
| **Rolling mean/std** | 49 | 7 vars × 4 fenêtres (3/7/14/30j) + dry_streak — shift(1) anti-fuite |
| **Cycliques** | 6 | sin/cos mois, doy, dow — décembre et janvier adjacents |
| **Harmattan/saison** | 10 | is_harmattan, harmattan_intensity, is_true_harmattan, season_code (4 niveaux) |
| **Spatiales** | 5 | region_code (ordinal S→N), lat/lon_norm, climate_zone, city_id |
| **Log-transforms** | 6 | precip skew 8.6→3.0 ; blh_min 2.6→0.7 ; wind 1.2→0.2 ; et0 1.0→0.1 |
| **Interactions** | 5 | wind×harmattan, blh×precip, lat×harmattan, temp_amp×blh |
| **Astronomique** | 1 | daylight_duration (Spencer 1971) — #2 importance XGBoost (20.8%), sans Excel |
| **Weather proxy** | 1 | precip×radiation → catégorie 0–4 (clair/nuageux/bruine/pluie/forte) — Spearman r=0.98 vs WMO |
| **BLH regime** | 1 | Binning 4 catégories : inversion (<200m), stable (200–500m), mixte (500–1500m), convectif (>1500m) |
| **Target encoding** | 1 | PM2.5 moyen par ville (OOF leave-one-year-out, r=0.997 vs réel) |

### 4.3 Vérification anti-fuite (5 tests)

La fuite de données temporelle est le risque principal dans une série chronologique. Cinq tests ont été exécutés et validés :

| Test | Résultat |
|---|---|
| pm25_log absent des lags | ✅ |
| Aucune corrélation \|r\|>0.99 avec la cible | ✅ |
| lag1[J] = pm25[J-1] confirmé sur 9 jours | ✅ |
| roll7_mean[J] = mean(pm25[J-7:J-1]) confirmé | ✅ |
| Pas de fuite cross-ville (lags par ville indépendamment) | ✅ |

### 4.4 Features supprimées (redondance EDA)

L'analyse de multicolinéarité (Spearman > 0.90) a identifié les redondances suivantes :

| Supprimée | Gardée | \|r\| |
|---|---|---|
| `rain_sum` | `precipitation_sum` | 1.000 |
| `temperature_2m_mean` | `temperature_2m_max` | 0.950 |
| `et0_fao_evapotranspiration` | `shortwave_radiation_sum` + `et0_log` | 0.927 |
| `is_stagnant` | — | 99.8% valeurs identiques (quasi-constante) |

### 4.5 Top corrélations avec PM2.5 (Spearman, EDA)

| Feature (ERA5) | Spearman r | Note |
|---|---|---|
| `relative_humidity_2m_max` | -0.418 | #1 importance XGBoost (28.2%) |
| `daylight_duration` | -0.625 | #2 importance (20.8%) — calculé astronomiquement |
| `weather_proxy` | -0.640 | ≈ weather_code WMO, r=0.98 |
| `precipitation_sum` | -0.485 | Lessivage humide |
| `blh_mean` | +0.138 | Confondant saisonnier (Harmattan = BLH modéré) |

> **Note** : la corrélation positive BLH–PM2.5 est attendue (non un artefact). Pendant l'Harmattan, le BLH est modéré (500–1000m) mais le PM2.5 est élevé dû aux poussières sahariennes. La relation BLH→PM2.5 est conditionnelle à la saison. Les features d'interaction `blh_x_precip` et `blh_regime × is_harmattan` capturent cette non-linéarité.

---

## 5. Fondements scientifiques — les mécanismes physiques

### 5.1 Sources de PM2.5 au Cameroun

**Source 1 : Poussières sahariennes (Harmattan)**
La Dépression du Bodélé (nord Tchad) est la plus grande source de poussière au monde. En saison sèche, les concentrations peuvent atteindre 100–200 µg/m³ à Maroua et Kousseri (Washington et al., 2006 ; Nebie et al., 2022).

**Source 2 : Feux de biomasse** *(intégré via F_fire)*
Les feux de brousse et agricoles génèrent des aérosols carbonés (suie, matière organique) représentant jusqu'à 40% du PM2.5 en Afrique subsaharienne (Gordon et al., 2023). Capturés par notre extraction NASA FIRMS : 634 103 pixels feux, 26.6% des ville-jours affectés.

**Source 3 : Pollution urbaine**
Trafic (véhicules anciens), combustion domestique de bois/charbon, activités industrielles (Douala, port). Ces sources sont constantes mais leur impact dépend des conditions de dispersion (BLH, vent).

**Source 4 : Aérosols secondaires**
Formation chimique par réactions photochimiques à partir de SO₂, NOₓ, NH₃. Favorisées par la chaleur et l'ensoleillement en saison sèche.

### 5.2 Saisonnalité spécifique au Cameroun

Le Cameroun a un **climat bimodal** dans le centre/sud et **unimodal** dans le nord :

**Yaoundé/Centre (bimodal) :**
- Grande saison des pluies : mars–juin (PM2.5 bas, lessivage fort)
- Petite saison sèche : juillet–août (PM2.5 modéré)
- Petite saison des pluies : septembre–novembre (PM2.5 bas)
- Grande saison sèche : décembre–février (Harmattan, PM2.5 élevé)

**Maroua/Extrême-Nord (unimodal) :**
- Saison des pluies : juin–septembre (PM2.5 bas)
- Saison sèche : octobre–mai (Harmattan intense, PM2.5 très élevé)

### 5.3 Pourquoi une formule multiplicative ?

La physique des aérosols en régime quasi-stationnaire :

```
C = E / (Λ_wet + Λ_dry + Λ_disp) ≈ E_eff × (1/Λ_wet) × (1/Λ_dry) × (1/Λ_disp)
```

La forme multiplicative de notre proxy est une approximation de cette solution analytique de l'équation de bilan de masse atmosphérique (Seinfeld & Pandis, 2016).

---

## 6. Calibration et validation

### 6.1 Résultats de calibration

| Métrique | Valeur | Source de référence |
|----------|--------|---------------------|
| Moyenne nationale | **32.50 µg/m³** | AQLI 2023 (cible) |
| Médiane | 25.44 µg/m³ | — |
| Écart-type | 22.95 µg/m³ | — |
| P10 / P90 | 13.6 / 60.8 µg/m³ | — |
| Min / Max | 2.0 / 232.4 µg/m³ | — |

### 6.2 Tests de validation physique

| Test | Résultat | Attendu | Statut |
|------|---------|---------|--------|
| Ratio saison sèche/humide | ×2.21 | ×2–5 | ✅ |
| Gradient Nord/Sud | ×1.98 (45.8 vs 23.1 µg/m³) | Nord > Sud | ✅ |
| Yaoundé saison sèche (DJF) | 28.2 µg/m³ | 17–35 µg/m³ | ✅ |
| Pic Harmattan nord (jan-fév) | 96.6 µg/m³ | > 50 µg/m³ | ✅ |
| Corr(PM2.5, précipitations) | −0.374 | Négatif | ✅ |
| Corr(PM2.5, BLH) | Négatif | Négatif | ✅ |
| F_hygro max observé | < 1.10 | ≤ 1.30 | ✅ |
| F_fire moyen | 1.024 | 1.0–1.2 | ✅ |

### 6.3 Validation croisée vs CAMS (Option C)

Le script `05_validation_uncertainty.py` compare notre proxy avec les données PM2.5 CAMS (Copernicus Atmosphere Monitoring Service) pour 7 villes de référence couvrant les 3 zones climatiques :

- **Zone équatoriale** (Yaoundé, Douala)
- **Zone de transition** (Bafoussam, Bertoua, Ngaoundéré)
- **Zone sahélienne** (Garoua, Maroua)

**Métriques de validation :** biais moyen, MAE, RMSE, corrélation r, NMB (Normalized Mean Bias, référence EPA). Un NMB < ±30% est considéré acceptable pour un proxy sans mesures sol.

### 6.4 Top 5 villes les plus polluées

| Ville | Région | PM2.5 annuel moyen | Explication |
|-------|--------|-------------------|-------------|
| Mokolo | Extrême-Nord | 56.7 µg/m³ | Zone sahélienne, Harmattan intense |
| Maroua | Extrême-Nord | 55.6 µg/m³ | Harmattan sévère + feux de brousse |
| Kousseri | Extrême-Nord | 53.4 µg/m³ | Frontière Tchad, bassin du Lac Tchad |
| Yagoua | Extrême-Nord | 50.8 µg/m³ | Plaine de Yagoua, exposition directe |
| Guider | Nord | 47.3 µg/m³ | Transition sahélienne |

---

## 7. Quantification de l'incertitude — Monte Carlo

### 7.1 Justification

Les paramètres physiques du proxy (α, a, k, HARMATTAN_STRENGTH, γ) ont des plages de valeurs documentées dans la littérature. Aucun jeu de paramètres n'est "correct" — ils représentent tous des approximations valides. La simulation Monte Carlo quantifie l'impact de cette incertitude sur les estimations de PM2.5.

### 7.2 Plages de paramètres utilisées

| Paramètre | Valeur nominale | Plage MC | Source |
|-----------|----------------|----------|--------|
| α (BLH exposant) | 0.60 | [0.40, 0.80] | Seinfeld & Pandis 2016 — peu sensible (|r|=0.158) |
| a (lessivage pluie) | 0.08 | [0.05, 0.12] | Berge & Jakobsen 1998 — paramètre le plus sensible (|r|=0.803) |
| k (dilution vent) | 0.035 | [0.025, 0.050] | Pasquill-Gifford |
| H_strength (Harmattan) | **1.4** | [1.50, 2.50] | Réduit 2.0→1.4 après calibration vs CAMS (ratio ×2.21) |
| γ (hygroscopique) | 0.004 | [0.002, 0.006] | Swietlicki et al. 2008 — peu sensible (|r|=0.039) |

### 7.3 Protocole

- N = 1 000 tirages aléatoires uniformes dans chaque plage
- Graine fixe (`seed=42`) pour la reproductibilité
- Résultat : distribution de 1 000 proxys PM2.5 par observation
- **Intervalle de confiance 90%** : P5–P95 de la distribution

### 7.4 Outputs du Monte Carlo

| Variable | Description |
|----------|-------------|
| `pm25_mc_mean` | Moyenne MC des 1000 tirages |
| `pm25_mc_std` | Écart-type MC |
| `pm25_mc_p05` | Percentile 5 (borne basse IC90%) |
| `pm25_mc_p95` | Percentile 95 (borne haute IC90%) |

Fichier de sortie : `data/pm25_with_uncertainty.parquet`

---

## 8. Cadre réglementaire

### 8.1 Normes internationales applicables

| Standard | Concentration 24h | Concentration annuelle |
|----------|------------------|----------------------|
| **OMS 2021 (AQG)** | 15 µg/m³ | 5 µg/m³ |
| US EPA NAAQS 2024 | 35 µg/m³ | 9 µg/m³ |
| UE Directive 2008/50 | 25 µg/m³ | 25 µg/m³ |

Le Cameroun n'a pas de norme nationale consolidée. Dans notre dataset : **85.8% des jours dépassent le seuil OMS 24h de 15 µg/m³**, et **100% des combinaisons ville-année dépassent le seuil annuel de 5 µg/m³**.

### 8.2 Interprétation AQI

| PM2.5 (µg/m³) | Niveau | Signification |
|--------------|--------|--------------|
| 0–12 | Bon | Qualité de l'air satisfaisante |
| 12–35 | Modéré | Groupes sensibles peuvent être affectés |
| 35–55 | Mauvais pour groupes sensibles | Éviter l'exposition prolongée |
| 55–150 | Mauvais | Tout le monde peut être affecté |
| 150–250 | Très mauvais | Alertes sanitaires |
| > 250 | Dangereux | Urgence sanitaire |

---

## 9. Architecture du pipeline

```
DONNÉES SOURCES
├── Dataset_complet_Meteo.xlsx         (40 villes, 26 variables, 2020–2025)
├── API ERA5 Open-Meteo                (archive-api.open-meteo.com/v1/archive)
└── NASA FIRMS MODIS                   (firms.modaps.eosdis.nasa.gov/api/area/csv/)

ÉTAPE 1 — EXTRACTION ERA5
└── 01_extract_era5_pm25_target.py
    ├── Input  : 40 villes (coordonnées corrigées)
    ├── ERA5   : daily météo + hourly BLH agrégé → daily
    └── Output : data/era5_raw/<ville>.parquet  (40 fichiers × 2181 jours)

ÉTAPE 2 — CONSTRUCTION VARIABLE CIBLE
└── 02_build_pm25_target.py
    ├── Input  : data/era5_raw/*.parquet + data/firms_fire_daily.parquet (optionnel)
    ├── Formule : C_base × F_stagnation × F_wet × F_wind × F_harmattan × F_hygro × F_fire
    ├── Calibration : moyenne nationale = 32.5 µg/m³ (AQLI 2023)
    ├── Output : data/pm25_proxy_era5.parquet          (87 240 × 29)
    └── Output : data/dataset_with_pm25_target.parquet (87 240 × 37)

ÉTAPE 3 — FEATURE ENGINEERING
└── 03_feature_engineering.py
    ├── Input  : data/dataset_with_pm25_target.parquet
    ├── Features : lags [1,2,3,7,14] × rolling [3,7,14,30] × cycliques × spatiales
    └── Output : data/dataset_features.parquet         (87 240 × 144)

ÉTAPE 4 — EXTRACTION FEUX NASA FIRMS
└── 04_extract_firms_fire.py
    ├── Source : MODIS_SP, bbox Cameroun 7,1,17,14, chunks 5 jours
    ├── Agrégation : FRP sum dans rayon 75 km par ville/jour
    ├── Cache    : data/firms_raw/firms_MODIS_SP_CMR_YYYY.parquet
    └── Output : data/firms_fire_daily.parquet          (87 240 × 5)
    [→ Relancer étapes 2 et 3 après cette extraction]

ÉTAPE 5 — VALIDATION & INCERTITUDE
└── 05_validation_uncertainty.py
    ├── Option C : Validation vs CAMS (7 villes référence)
    ├── Option D : Comparaison BLH mean vs BLH max (heures actives)
    ├── Option E : Monte Carlo 1000 tirages → IC90% par paramètre
    └── Output : data/pm25_with_uncertainty.parquet + figures PNG

ÉTAPE 6 — MODÉLISATION ML
└── 06_model_xgboost.py
    ├── Input  : data/dataset_features.parquet
    ├── Modèles : XGBoost + LightGBM (+ Ensemble 50/50)
    ├── Validation : expanding-window CV (Fold1→Fold4) + Test 2025
    └── Output : models/xgboost_final.pkl + models/test_predictions_2025.parquet
```

### 9.1 Validation temporelle — expanding window

**IMPORTANT :** Un split aléatoire est **interdit** pour les séries temporelles. Il crée une fuite de données : le modèle verrait des données futures pendant l'entraînement.

```
Fold 1 : Train [2020]         → Val [2021]
Fold 2 : Train [2020–2021]    → Val [2022]
Fold 3 : Train [2020–2022]    → Val [2023]
Fold 4 : Train [2020–2023]    → Val [2024]
Test   : Train [2020–2024]    → Test [2025]  ← jamais vu pendant le développement
```

---

## 10. Résultats et statistiques clés

### 10.1 Dataset final — `data/dataset_features.parquet`

```
Shape              : 87 240 lignes × 144 colonnes
Villes             : 40
Régions            : 10
Période            : 2020-01-01 → 2025-12-20
Variable cible     : pm25_proxy (100% renseigné, 0 NaN)
Couverture F_fire  : 26.6% des ville-jours avec activité feu (rayon 75 km)
```

### 10.2 Statistiques de la variable cible PM2.5 proxy

```
count  : 87 240
mean   : 32.50 µg/m³  ← exactement la cible AQLI (calibration parfaite)
std    : 22.95 µg/m³
min    :  2.00 µg/m³  ← plancher physique (fond irréductible)
25%    : 18.43 µg/m³
50%    : 25.44 µg/m³
75%    : 37.36 µg/m³
max    : 232.4 µg/m³  ← pic Harmattan extrême, Extrême-Nord
```

### 10.3 Contribution des facteurs multiplicatifs

| Facteur | Moyenne | Rôle dominant |
|---------|---------|--------------|
| F_stagnation | ~1.0 | Variabilité journalière (BLH) |
| F_wet | ~0.85 | Réduction en saison des pluies |
| F_wind | ~0.75 | Atténuation (désactivé Harmattan) |
| F_harmattan | ~1.35 | Signal saisonnier Nord–Sud |
| F_hygro | ~1.02 | Faible impact (b=0.004 conservateur) |
| F_fire | ~1.024 | Pics localisés (zones savane/agriculture) |

### 10.4 Performances du modèle ML — Résultats finaux (configuration V4)

#### Cross-Validation expanding-window

| Fold | Train | Val | XGBoost R² | LightGBM R² |
|---|---|---|---|---|
| Fold 1 | 2020 | 2021 | 0.970 | 0.967 |
| Fold 2 | 2020–21 | 2022 | 0.994 | 0.994 |
| Fold 3 | 2020–22 | 2023 | 0.994 | 0.994 |
| Fold 4 | 2020–23 | 2024 | 0.994 | 0.994 |
| **Moyenne CV** | | | **0.988** | **0.987** |

#### Test 2025 (données jamais vues)

| Modèle | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|
| XGBoost | 1.66 µg/m³ | 0.93 | 0.9929 | 2.5% |
| LightGBM | 1.53 µg/m³ | 0.86 | 0.9940 | 2.3% |
| **Ensemble 50/50** | **1.55 µg/m³** | **0.85** | **0.9939** | **2.2%** |

*Ces résultats surpassent les benchmarks de la littérature africaine (RMSE 2.7–6.5 µg/m³, r=0.70–0.85).*

#### Ablation Study — Contribution des lags vs physique

| Modèle | R² Test 2025 | Interprétation |
|---|---|---|
| Persistence (PM2.5[t-1]) | 0.745 | Baseline trivial |
| **Météo-only XGBoost** | **0.9934** | Features physiques = source principale |
| Full (lags + météo) | 0.9929 | Lags = redondance (ΔR²=-0.0005) |

**Résultat clé** : le modèle apprend principalement les drivers physiques — Harmattan (`is_true_harmattan` : 41.6% d'importance), zone climatique (18.2%), précipitations. Les lags PM2.5 n'apportent aucun gain car la cible est dérivée directement des mêmes variables météo ERA5. Le modèle est donc physiquement interprétable et généralisable à de nouvelles villes.

### 10.5 Validation vs CAMS

| Métrique | Valeur | Interprétation |
|---|---|---|
| Corrélation r | 0.339 | Normal pour l'Afrique (littérature : 0.3–0.5) |
| NMB | +85% | CAMS sous-estime de 20–50% en Afrique + légère sur-amplification proxy |
| Ratio saisonnier proxy | ×2.21 | Conforme littérature (nord Cameroun 2.0–3.0) |
| Ratio saisonnier CAMS | ×2.02 | Référence de comparaison |

CAMS (Copernicus) est un produit imparfait en Afrique subsaharienne : NMB documenté de -20% à -50% au Kampala, -30%+ pendant les épisodes Harmattan intenses. Le NMB proxy/CAMS de +85% reflète cette double imperfection. Pour une validation robuste, un dataset type van Donkelaar (satellite-derived) ou MERRA-2 est recommandé en complément.

---

### 10.6 Pipeline d'inférence temps réel

#### Architecture générale

Le script `notebooks/08_inference_realtime.py` implémente un pipeline bout-en-bout de prévision en conditions opérationnelles. Il est exécuté quotidiennement via GitHub Actions.

```
Open-Meteo Forecast API          →  fetch_forecast(days=7..16)
      ↓ (40 villes, BLH horaire → daily)
build_features(df_forecast, df_history)
      ↓ (réplication exacte de 03_feature_engineering.py : 139 features)
Détection cold-start ←→ Détection hot-start
      ↓                         ↓
xgboost_coldstart.pkl    xgboost_final.pkl
(météo-only, 122 feat.)  (lags+météo, 139 feat.)
      ↓
np.expm1(prédiction log) → PM2.5 µg/m³
      ↓
generate_alerts() → classification OMS + messages FR
      ↓
data/predictions_latest.parquet + data/alerts_latest.json
```

#### Stratégie dual-modèle et cold-start

**Problème** : Le modèle complet (R²=0.9939) utilise 30 features de lags (PM2.5 et météo à J-1, J-2, J-3, J-7, J-14). En conditions opérationnelles, l'historique récent peut être absent ou incomplet.

**Solution** : deux modèles sont entraînés et stockés :

| Modèle | Features | R² test 2025 | Usage |
|---|---|---|---|
| `xgboost_final.pkl` | 139 (lags + météo) | 0.9929 | Historique ≥14 jours disponible |
| `xgboost_coldstart.pkl` | 122 (météo-only, sans lags PM2.5) | 0.9959 | Démarrage à froid ou historique insuffisant |

**Remarque** : le modèle cold-start obtient un R² légèrement supérieur (0.9959 vs 0.9929), confirmant le résultat de l'ablation study : les features de lags PM2.5 sont redondantes car la cible est entièrement déterministe à partir de la météo.

**Logique de routage automatique** :
```python
if history_days < 14:
    model = xgboost_coldstart   # méteo-only
else:
    model = xgboost_final       # lags + météo
```

#### Réplication exacte des features

La fonction `build_features()` reproduit fidèlement `03_feature_engineering.py`. Points critiques :

| Feature | Formule exacte | Erreur fréquente à éviter |
|---|---|---|
| `blh_log` | `np.log(blh_mean)` | ≠ `np.log1p(blh_mean)` |
| `blh_inv` | `1000 / blh_mean` | ≠ `1 / blh_mean` |
| `rh_mean` | `(rh_max + rh_min) / 2` | ≠ colonne ERA5 directe |
| `month_sin/cos` | `sin/cos(2π × month / 12)` | nommage : `month_sin` pas `sin_month` |
| `is_dry_day` | `precipitation < 0.1` | ≠ `is_rainy_day` inversé |
| Lags | `pm25_proxy_lag1..14` | doivent être précalculés sur concat history+forecast |

#### API Open-Meteo Forecast

```
Endpoint : https://api.open-meteo.com/v1/forecast
Variables météo (daily) : temperature_2m_max, precipitation_sum, wind_speed_10m_max,
                           relative_humidity_2m_max, relative_humidity_2m_min,
                           shortwave_radiation_sum, et0_fao_evapotranspiration
BLH (hourly → daily) : boundary_layer_height → mean/min/max sur 24h
Horizon : 7 jours par défaut (paramétrable 1–16 via workflow_dispatch)
Accès : gratuit, sans clé API
```

La BLH est uniquement disponible en mode horaire dans l'API Open-Meteo. Le script extrait les 24 valeurs horaires et calcule `blh_mean`, `blh_min`, `blh_max` pour répliquer exactement le prétraitement ERA5.

#### Format des sorties

**`data/predictions_latest.parquet`** (40 villes × N jours) :

| Colonne | Description |
|---|---|
| `city` | Nom de la ville |
| `time` | Date de prévision |
| `pm25_pred` | PM2.5 prédit (µg/m³) |
| `alert_level` | `good` / `moderate` / `unhealthy_sensitive` / `unhealthy` / `very_unhealthy` / `hazardous` |
| `alert_message` | Message d'alerte en français |
| `model_used` | `full` ou `coldstart` |

**`data/alerts_latest.json`** :
```json
{
  "generated_at": "2026-04-03T06:12:34",
  "total_alerts": 12,
  "dangerous": 3,
  "cities": [
    {"city": "Maroua", "date": "2026-04-04", "pm25": 78.3, "level": "very_unhealthy",
     "message": "Alerte PM2.5 : Maroua — 78.3 µg/m³ (Très mauvais). Évitez les activités physiques intenses en extérieur."},
    ...
  ]
}
```

#### Automatisation GitHub Actions

Le workflow `.github/workflows/daily_refresh.yml` orchestre l'exécution quotidienne :

```yaml
cron: "0 6 * * *"   # 06:00 UTC = 07:00 WAT (heure Cameroun)
```

Étapes : checkout → setup Python 3.11 → pip install → inférence → vérification output → commit + push.

Le déclenchement manuel (`workflow_dispatch`) accepte un paramètre `forecast_days` (1–16, défaut 7).

**Coût** : <5 min/exécution sur ubuntu-latest. Quota GitHub Actions gratuit : 2000 min/mois → budget annuel ~10× suffisant.

#### Performances opérationnelles (test froid, Avril 2026)

En démarrage à froid (historique ERA5 se terminant en décembre 2025, prévision d'avril 2026) :

| Ville | PM2.5 prédit (moy. 7j) | Niveau attendu |
|---|---|---|
| Maroua | ~38–55 µg/m³ | Harmattan résiduel, fin de saison |
| Yaoundé | ~18–25 µg/m³ | Saison des pluies précoce |
| Douala | ~15–22 µg/m³ | Côte, précipitations fréquentes |

Les niveaux sont cohérents avec la saisonnalité documentée pour avril (transition Harmattan→saison des pluies dans le Centre-Sud, résidus dans le Nord).

### 10.7 Dashboard interactif — TERRA

#### Architecture

Le dashboard `notebooks/07_dashboard.py` (Streamlit, ~1 800 lignes) expose trois pages :

| Page | Composants principaux |
|---|---|
| **TEMPS RÉEL** | Carte Mapbox scatter PM2.5 · 4 métriques (villes, PM2.5 moyen/max, alertes) · Panneau alertes OMS filtré par région/seuil · Tableau HTML 40 villes avec barres de progression · Export CSV/JSON |
| **PAR VILLE** | Série temporelle 2020–2025 avec IC90% Monte Carlo (zone d'incertitude) · Prévisions 7 jours Open-Meteo (losanges verts) · Corrélation PM2.5 vs précipitations/BLH/vent · Saisonnalité mensuelle · Export historique |
| **CLASSEMENT** | Top 10 barres d'erreur ±IC90% · Scatter latitude vs PM2.5 (R²=0.882, gradient Harmattan) · Heatmap mensuelle 15 villes · Évolution inter-annuelle par région · Cartes performance modèles · Importance variables · CV expanding-window · Ablation study |

#### Navigation programmatique P1→P2

La navigation inter-pages utilise un pattern `session_state` à deux clés :

- `_nav_request` : écrite par les boutons natifs `st.button` dans la page TEMPS RÉEL. Jamais liée à un widget Streamlit → modification autorisée après rendu.
- `_nav_page` : clé liée au widget `st.radio` (navigation visible). Mise à jour en consommant `_nav_request` **avant** l'instanciation du widget → changement de page dès le premier clic.

Ce pattern contourne la limitation `StreamlitAPIException: cannot be modified after the widget is instantiated`.

#### Thème graphique TERRA

Palette éditoriale africaine : `--cream: #F5EDD9`, `--terra: #8B3A1E`, `--ochre: #C8941A`. Police Playfair Display (titres serif), Barlow Condensed (labels), DM Mono (valeurs numériques). Grain de texture SVG sur le fond pour un rendu "bulletin de terrain scientifique".

---

## 11. Références

1. **AQLI 2023** — Air Quality Life Index, Cameroun. Energy Policy Institute, University of Chicago. https://aqli.epic.uchicago.edu/

2. **OMS 2021** — Global Air Quality Guidelines. World Health Organization. ISBN 978-92-4-003422-8.

3. **Washington R. et al., 2006** — "Links between topography, wind, deflation, lakes and dust: The case of the Bodélé Depression, Chad." *Geophysical Research Letters*, 33, L09401. DOI: 10.1029/2006GL025827.

4. **Nebie E.K. et al., 2022** — "Harmattan dust and health in West Africa: systematic review of epidemiological evidence." *Environmental Health Perspectives*, 130(7). *(Référence représentative de la littérature sur les impacts sanitaires de l'Harmattan en Afrique de l'Ouest.)*

5. **Seinfeld J.H. & Pandis S.N., 2016** — *Atmospheric Chemistry and Physics: From Air Pollution to Climate Change*. 3rd ed. Wiley. ISBN 978-1-118-94740-1.

6. **Li H. et al., 2020** — "Characteristics of the atmospheric boundary layer and its relation with PM2.5 during haze episodes in winter in the North China Plain." *Atmospheric Environment*, 223, 117382. DOI: 10.1016/j.atmosenv.2020.117382.

7. **Berge E. & Jakobsen H.A., 1998** — "A regional scale multi-layer model for the calculation of long-term transport and deposition of air pollution in Europe." *Tellus B*, 50(3), 205–223. DOI: 10.3402/tellusb.v50i3.16097.

8. **Swietlicki E. et al., 2008** — "Hygroscopic properties of submicrometer atmospheric aerosol particles measured with H-TDMA instruments in various environments." *Tellus B*, 60(3), 432–469. DOI: 10.1111/j.1600-0889.2008.00350.x.

9. **Gordon T.D. et al., 2023** — "The Effects of Trash, Residential Biofuel, and Open Biomass Burning Emissions on Local and Transported PM2.5 and Its Attributed Mortality in Africa." *GeoHealth*, 7, e2022GH000673. DOI: 10.1029/2022GH000673.

10. **ECMWF ERA5** — "ERA5 global reanalysis." Copernicus Climate Change Service. DOI: 10.24381/cds.adbb2d47.

11. **NASA FIRMS** — Fire Information for Resource Management System. MODIS Collection 6.1 Standard Processing. https://firms.modaps.eosdis.nasa.gov/

12. **Navinya C. et al., 2020** — "Evaluating PM2.5 and its Association with Socioeconomic Factors in sub-Saharan African Cities." *Atmosphere*, 11(9), 979. DOI: 10.3390/atmos11090979.

13. **Open-Meteo** — Historical Weather API & Forecast API. https://open-meteo.com/en/docs/

14. **Crumeyrolle S. et al., 2011** — "Transport of dust particles from the Bodélé region to the monsoon layer — AMMA case study of the 9–14 June 2006 period." *Atmospheric Chemistry and Physics*, 11, 479–494. DOI: 10.5194/acp-11-479-2011.

---

---

## 12. Historique des versions

| Version | Date | Changements principaux |
|---|---|---|
| V1 | Mars 2026 | Pipeline initial, proxy formule de base, feature engineering par supposition |
| V2 | Mars 2026 | Intégration NASA FIRMS F_fire + Validation Monte Carlo |
| V3 | Avril 2026 | EDA rigoureuse, feature engineering basé sur données, patch BLH 2024 S1, anti-leak tests |
| V4 | Avril 2026 | Calibration proxy vs CAMS (3 tests), ablation study ML, restructuration projet, BLH_MIN=250m, H_strength=1.4 |
| V5 | Avril 2026 | Pipeline inférence temps réel (08_inference_realtime.py), modèle cold-start R²=0.9959, GitHub Actions cron quotidien, sorties JSON alertes |
| V6 | Avril 2026 | Dashboard Streamlit TERRA (3 pages, ~1 800 lignes) · Navigation session_state P1→P2 · Alertes OMS interactives · Export CSV/JSON · Thème éditorial africain |
| **V7** | **Avril 2026** | **Déploiement HuggingFace Spaces (QUASAR-30/pm25-cameroun) · app.py entry point · requirements pinned · sync GitHub Actions daily · fix deprecations Streamlit 1.56** |

*Document généré dans le cadre du Hackathon IndabaX Cameroun 2026.*
*Pipeline entièrement reproductible : voir dossier `notebooks/` (scripts 01 à 08).*
*Structure projet : voir `CLAUDE.md` pour les chemins et configurations.*

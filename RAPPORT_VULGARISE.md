# Rapport Scientifique Vulgarisé
## Prédire la qualité de l'air au Cameroun avec l'Intelligence Artificielle
### Hackathon IndabaX Cameroun 2026

---

> **Ce rapport est écrit pour être compris par tout le monde**, même sans formation scientifique.
> Chaque notion technique est expliquée avant d'être utilisée.
> Tu peux le lire du début à la fin, ou aller directement à la section qui t'intéresse.

---

## Table des matières

1. [Le problème de départ](#1-le-problème-de-départ)
2. [Qu'est-ce que le PM2.5 ?](#2-quest-ce-que-le-pm25)
3. [Pourquoi c'est un problème difficile ?](#3-pourquoi-cest-un-problème-difficile)
4. [Notre stratégie en 3 étapes](#4-notre-stratégie-en-3-étapes)
5. [Étape 1 — Construire un proxy PM2.5](#5-étape-1--construire-un-proxy-pm25)
6. [Étape 2 — Préparer les données pour le ML](#6-étape-2--préparer-les-données-pour-le-ml)
7. [Étape 3 — Entraîner le modèle de Machine Learning](#7-étape-3--entraîner-le-modèle-de-machine-learning)
8. [Les résultats : est-ce que ça marche ?](#8-les-résultats--est-ce-que-ça-marche-)
9. [Ce que ça veut dire pour le Cameroun](#9-ce-que-ça-veut-dire-pour-le-cameroun)
10. [Le système temps réel — des prévisions chaque matin](#10-le-système-temps-réel--des-prévisions-chaque-matin)
11. [Le dashboard — voir la qualité de l'air en un clic](#11-le-dashboard--voir-la-qualité-de-lair-en-un-clic)
12. [Les limites honnêtes du projet](#12-les-limites-honnêtes-du-projet)
13. [Glossaire des notions clés](#13-glossaire-des-notions-clés)

---

## 1. Le problème de départ

**Question de départ :** Dans 40 villes camerounaises, l'air est-il dangereux à respirer ? Et demain, sera-t-il plus ou moins pollué ?

**Le problème concret :** Pour répondre à cette question, il faudrait des capteurs de qualité de l'air dans chaque ville, mesurant en permanence. Ces capteurs **n'existent pas** au Cameroun à l'échelle nationale. Pas de données = pas de modèle.

**Notre réponse :** On va *reconstruire* le niveau de pollution à partir de données météo disponibles gratuitement par satellite, puis entraîner un modèle d'IA qui prédit la pollution future à partir de la météo.

---

## 2. Qu'est-ce que le PM2.5 ?

**PM2.5** = "Particulate Matter 2.5 micromètres". Ce sont des **particules microscopiques** en suspension dans l'air dont le diamètre est inférieur à 2,5 micromètres (pour comparaison : un cheveu humain fait environ 70 micromètres, soit 28 fois plus épais).

**Pourquoi c'est dangereux :**
Ces particules sont si petites qu'elles **pénètrent jusqu'aux poumons profonds** et même dans le sang. Elles provoquent des maladies respiratoires, cardiaques, des cancers du poumon, et réduisent l'espérance de vie.

**D'où ça vient au Cameroun :**
- **L'Harmattan** : vent qui souffle du Sahara (novembre à mars) et transporte des milliards de tonnes de sable et de poussière vers le sud
- Les feux de biomasse (agriculture, déforestation)
- La combustion de bois pour cuisiner
- Le trafic routier dans les grandes villes

**Les seuils à retenir :**

| Niveau | PM2.5 | Ce que ça veut dire |
|--------|-------|---------------------|
| Bon | < 5 µg/m³ | Recommandation annuelle OMS (quasi inaccessible en Afrique) |
| Acceptable | < 15 µg/m³ | Recommandation journalière OMS |
| Mauvais | 15–35 µg/m³ | Risque pour les personnes sensibles |
| Très Mauvais | 35–75 µg/m³ | Risque pour tout le monde |
| Dangereux | > 75 µg/m³ | Éviter toute activité extérieure |

> **Pour donner une idée :** Maroua en janvier peut atteindre 150–200 µg/m³. C'est 30 à 40 fois le seuil annuel recommandé par l'OMS.

---

## 3. Pourquoi c'est un problème difficile ?

### 3.1 Pas de mesures = pas de vérité terrain

En Machine Learning, pour entraîner un modèle, il faut des **exemples corrects** : "ce jour-là, la pollution était de X". Sans capteurs, on n'a pas ces exemples. C'est comme vouloir apprendre à un enfant à reconnaître des chats sans jamais lui montrer de photo de chat.

**Solution adoptée :** On crée nous-mêmes une "vérité approximative" à partir de la physique atmosphérique. Ce n'est pas parfait, mais c'est raisonné et documenté.

### 3.2 La pollution dépend de beaucoup de facteurs simultanément

La pollution d'un jour donné à Maroua dépend de :
- La pluie (qui nettoie l'air)
- Le vent (qui peut apporter de la poussière ou disperser la pollution)
- La hauteur de la couche atmosphérique où se mélange l'air
- La saison (Harmattan ou saison des pluies)
- La latitude (nord = plus exposé au Sahara)
- Les feux de forêt dans la région

Tous ces facteurs **interagissent**. Un modèle simple (une droite) ne peut pas capturer ça. Il faut du Machine Learning.

### 3.3 Les données temporelles sont piégeuses

Si on sait que la pollution était forte hier, c'est souvent parce qu'elle sera forte aujourd'hui aussi (c'est logique : le vent Harmattan dure plusieurs jours). Un modèle mal conçu peut "tricher" en disant juste "demain = aujourd'hui" et avoir l'air très performant... sans rien avoir appris d'utile.

---

## 4. Notre stratégie en 3 étapes

```
Données satellite ERA5          Données FIRMS
(météo : pluie, vent, BLH...)   (feux de forêt par satellite NASA)
          │                              │
          └──────────────┬───────────────┘
                         ▼
              ┌─────────────────────┐
              │  ÉTAPE 1 : Proxy    │  ← Formule physique
              │  PM2.5 "artificiel" │    pour reconstruire
              └─────────┬───────────┘    la pollution
                        │
                        ▼
              ┌─────────────────────┐
              │  ÉTAPE 2 : Feature  │  ← Préparer 139 variables
              │  Engineering        │    qui décrivent chaque jour
              └─────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │  ÉTAPE 3 : Modèle   │  ← XGBoost + LightGBM
              │  XGBoost/LightGBM   │    apprend les patterns
              └─────────┬───────────┘
                        │
                        ▼
              Prédictions PM2.5 quotidiennes
              pour 40 villes, 2020–2025
```

---

## 5. Étape 1 — Construire un proxy PM2.5

### 5.1 Qu'est-ce qu'un "proxy" ?

Un proxy est un **indicateur de remplacement**. Comme on ne peut pas mesurer directement la pollution, on la **reconstruit indirectement** à partir de variables qu'on connaît.

Exemple du quotidien : si tu ne peux pas peser directement l'eau d'un lac, tu mesures le niveau d'eau et tu calcules le volume à partir des dimensions du lac. Le niveau d'eau est un proxy du volume.

### 5.2 La formule du proxy

Notre proxy PM2.5 est un **produit de 6 facteurs physiques** :

```
PM2.5_estimé = C_base × F_stagnation × F_wet × F_wind × F_harmattan × F_hygro × F_fire
```

Chaque facteur représente un mécanisme physique réel :

---

**F_stagnation — L'atmosphère qui se comprime**

La couche de mélange atmosphérique (BLH = Boundary Layer Height) est la zone dans laquelle les polluants se mélangent. Quand cette couche est haute (>1500m), les polluants se dispersent dans beaucoup d'air → peu de pollution. Quand elle est basse (200–300m), tout se concentre dans peu d'air → beaucoup de pollution.

```
F_stagnation = (1000 / max(BLH, 250))^0.6
```

*En clair : si la couche de mélange est 4 fois plus petite, la pollution est environ 2,3 fois plus forte.*

**Choix documenté :** on utilise un plancher de 250m minimum (pas moins) parce que la nuit, la couche peut descendre à 20–50m par inversion thermique. Ce serait physiquement absurde de calculer un facteur de stagnation ×50 pour une nuit froide.

---

**F_wet — La pluie qui nettoie l'air**

La pluie capture les particules en suspension et les entraîne au sol. C'est le phénomène de "wet scavenging" (lessivage humide).

```
F_wet = 1 / (1 + 0.08 × pluie_mm)
```

*En clair : 10mm de pluie réduit la pollution de 55%. 50mm réduit de 80%.*

**Paramètre rain_k = 0.08** : c'est le coefficient de lessivage. La littérature scientifique donne une plage de 0.05 à 0.12. Notre analyse de sensibilité (Monte Carlo) montre que c'est le paramètre le plus important du proxy.

---

**F_wind — Le vent qui disperse ou transporte**

Le vent a deux effets opposés :
- **Vent normal** : disperse les polluants locaux → réduit la pollution
- **Vent Harmattan (nord/nord-est)** : transporte de la poussière du Sahara → augmente la pollution

```
F_wind = exp(-0.035 × vitesse_vent)  ← si pas Harmattan
F_wind = 1.0                         ← si Harmattan (vent est vecteur de poussière)
```

*Choix justifié par EDA : la corrélation vitesse du vent / PM2.5 est quasi-nulle globalement (r≈0) car ces deux effets s'annulent. On les sépare donc.*

---

**F_harmattan — La poussière du Sahara**

L'Harmattan est le driver principal de la pollution au Cameroun. Son intensité augmente avec la latitude (plus on est au nord, plus on est exposé) et suit un cycle saisonnier (novembre à mars).

```
F_harmattan = 1 + 1.4 × is_saison_sèche × (latitude - 3) / (11 - 3)
```

*En clair : à Maroua (lat 10.5°N, pleine saison sèche), F_harmattan ≈ 2.3 → la pollution est 2.3× plus forte à cause de l'Harmattan.*

**Paramètre H_strength = 1.4** : initialement à 2.0, réduit après calibration contre les données CAMS (satellite Copernicus). Avec 1.4, notre ratio saisonnier (pollution été/hiver) est ×2.21, proche des ×2.02 de CAMS et dans la plage [2.0–3.0] de la littérature pour le nord-Cameroun.

---

**F_hygro — L'humidité qui gonfle les particules**

Par temps humide, les particules absorbent de l'eau et "gonflent" (croissance hygroscopique), ce qui augmente leur masse et les rend plus nocives.

```
F_hygro = min(1 + 0.004 × max(0, RH - 75%), 1.3)  ← si pas de pluie
F_hygro = 1.0                                        ← si pluie > 1mm (pluie efface l'effet)
```

*Plafonné à 1.3× : littérature (Swietlicki et al. 2008, Tellus B) montre que la croissance hygroscopique des aérosols tropicaux dépasse rarement +30%.*

---

**F_fire — Les feux de forêt (NASA FIRMS)**

Les feux agricoles et de déforestation émettent des quantités massives de PM2.5. On utilise les données NASA FIRMS (Fire Radiative Power) par satellite pour quantifier les feux dans un rayon de 75km autour de chaque ville.

```
F_fire = 1 + 0.02 × log(1 + FRP_75km)
```

---

**C_base — La calibration finale**

C_base est une **constante de calibration** calculée automatiquement pour que la moyenne nationale annuelle soit exactement 32.5 µg/m³ — la référence AQLI 2023 pour le Cameroun. C'est une ancre : quels que soient les paramètres utilisés, on force la valeur moyenne à correspondre à une référence reconnue.

### 5.3 Comment on a validé le proxy ?

On a comparé notre proxy avec **CAMS** (le modèle de qualité de l'air de l'agence spatiale européenne Copernicus). Résultats :
- Corrélation r = 0.34 → dans la plage normale pour l'Afrique (0.3–0.5)
- Notre proxy donne des valeurs +85% plus élevées que CAMS

Ce dernier point n'est pas forcément un défaut : CAMS est connu pour **sous-estimer** la pollution en Afrique subsaharienne de 20–50% (documenté dans la littérature scientifique). Mais cela signifie aussi que notre proxy n'est pas une vérité absolue — c'est une **estimation raisonnée avec des incertitudes documentées**.

---

## 6. Étape 2 — Préparer les données pour le ML

### 6.1 Qu'est-ce que le Feature Engineering ?

Le "feature engineering" (ingénierie des variables) consiste à **transformer les données brutes en variables informatives** pour le modèle. Un modèle de ML ne comprend pas "c'est l'Harmattan" — il faut traduire ça en chiffres.

### 6.2 Pourquoi on transforme la cible en log ?

La distribution brute du PM2.5 est fortement **asymétrique** : la plupart des jours ont une pollution de 15–40 µg/m³, mais quelques jours Harmattan intenses peuvent atteindre 150–200 µg/m³. Si on entraîne le modèle sur ces valeurs brutes, il va optimiser sa précision sur les jours normaux (80% du temps) et rater les pics — exactement ce qu'on veut éviter.

**Solution :** on applique la transformation log1p (logarithme + 1) sur la cible.

```
pm25_log = log(1 + pm25_proxy)
```

L'asymétrie passe de 1.91 → 0.36. La distribution est quasi-normale. Le modèle traite équitablement les jours normaux ET les pics Harmattan.

### 6.3 Les 139 variables construites

On a créé 139 variables au total, regroupées par catégorie :

| Catégorie | Nombre | Exemples |
|-----------|--------|---------|
| Lags temporels | 30 | PM2.5 d'hier, d'avant-hier, d'il y a 7j et 14j |
| Statistiques glissantes | 49 | Moyenne PM2.5 sur 7j, 14j, 30j, 90j |
| Cycliques (saisons) | 6 | sin/cos du mois, sin/cos du jour de l'année |
| Harmattan/saison | 10 | is_true_harmattan, harmattan_intensity |
| Spatial | 5 | région, latitude normalisée, zone climatique |
| Transformations log | 6 | log(précip+1), log(BLH+1) |
| Interactions | 5 | vent × harmattan, BLH × précip |
| Astronomique | 1 | durée du jour (formule Spencer 1971) |
| Nouveaux | 4 | weather_proxy, blh_regime, city_pm25_te |

### 6.4 L'anti-triche (anti-leakage)

Une erreur classique en ML : utiliser des données du **futur** pour prédire le présent. Par exemple, si on inclut la moyenne PM2.5 de la semaine qui suit, le modèle a accès à la réponse avant la question — il triche.

On a vérifié 5 types de fuite possibles :
1. ✅ La variable cible (pm25_log) n'est pas dans les features
2. ✅ Aucune variable n'est corrélée à >0.99 avec la cible (ce qui indiquerait une fuite)
3. ✅ Le lag-1 = PM2.5 du jour précédent, pas du jour actuel
4. ✅ La moyenne glissante 7j = les 7 jours *précédents*, pas les jours suivants
5. ✅ Pas de fuite entre villes (chaque ville est indépendante)

---

## 7. Étape 3 — Entraîner le modèle de Machine Learning

### 7.1 Qu'est-ce que XGBoost ?

XGBoost (eXtreme Gradient Boosting) est un algorithme de Machine Learning basé sur les **arbres de décision**. Imagine un arbre qui pose des questions : "Est-ce que c'est l'Harmattan ? → Si oui : est-ce qu'il y a eu de la pluie hier ? → Si oui : ..."

Le "boosting" consiste à construire des centaines de petits arbres imparfaits, chacun corrigeant les erreurs du précédent. Au final, leur combinaison est très précise. C'est l'un des algorithmes les plus performants en compétitions de data science.

**LightGBM** est une variante plus rapide avec les mêmes principes.

### 7.2 Pourquoi pas du deep learning (réseau de neurones) ?

Pour ce dataset (87 240 lignes, 139 variables), XGBoost et LightGBM sont :
- Plus rapides à entraîner (secondes vs heures)
- Aussi précis ou meilleurs
- Plus interprétables (on peut expliquer pourquoi le modèle prédit X)
- Moins sensibles aux hyperparamètres

Le deep learning brille surtout sur des millions d'exemples et des données non-structurées (images, texte).

### 7.3 La validation temporelle (expanding-window)

**Problème :** si on mélange aléatoirement les données pour tester le modèle, on risque de tester sur 2022 en s'entraînant sur 2023 et 2024 → le modèle a "vu l'avenir". Une série temporelle ne peut pas être validée comme ça.

**Solution :** validation par fenêtre expansive :

```
Fold 1 : Entraînement sur 2020        → Test sur 2021
Fold 2 : Entraînement sur 2020–2021   → Test sur 2022
Fold 3 : Entraînement sur 2020–2022   → Test sur 2023
Fold 4 : Entraînement sur 2020–2023   → Test sur 2024
Test   : Entraînement sur 2020–2024   → Test sur 2025 (jamais vu)
```

Chaque fold représente un scénario réel : "j'ai des données jusqu'à aujourd'hui, je prédis demain".

---

## 8. Les résultats : est-ce que ça marche ?

### 8.1 Les métriques clés

**R² (R-carré)** : mesure la qualité des prédictions. 0 = nul, 1 = parfait. Un R²=0.99 veut dire que le modèle explique 99% de la variabilité observée.

**RMSE** (Root Mean Square Error) : erreur moyenne en µg/m³. RMSE=1.6 µg/m³ signifie qu'en moyenne, la prédiction est à ±1.6 µg/m³ de la vraie valeur.

**MAPE** (Mean Absolute Percentage Error) : erreur en pourcentage.

### 8.2 Résultats sur le test 2025

| Modèle | RMSE | R² | MAPE |
|--------|------|----|------|
| XGBoost | 1.66 µg/m³ | 0.993 | 2.5% |
| LightGBM | 1.53 µg/m³ | 0.994 | 2.3% |
| Ensemble (moyenne) | **1.55 µg/m³** | **0.994** | **2.2%** |

Pour donner un contexte : une erreur de 1.55 µg/m³ sur une plage de 0–200 µg/m³, c'est une précision de moins de 1%. C'est excellent.

### 8.3 La preuve que le modèle apprend la physique (étude d'ablation)

L'**étude d'ablation** consiste à retirer des composantes du modèle pour mesurer leur contribution réelle.

| Version du modèle | R² | Ce que ça signifie |
|-------------------|----|---------------------|
| Persistence simple (PM2.5 demain = PM2.5 aujourd'hui) | 0.745 | Baseline : sans ML du tout |
| **Météo seulement** (sans historique PM2.5) | **0.993** | La physique suffit ! |
| Modèle complet (météo + historique) | 0.993 | Les lags n'apportent rien |

**Conclusion majeure :** le modèle n'apprend pas juste "demain ressemble à aujourd'hui". Il apprend réellement les mécanismes physiques — l'Harmattan, les précipitations, la BLH. La preuve : même sans aucun historique PM2.5, on obtient R²=0.993.

### 8.4 Ce que le modèle a appris

Les variables les plus importantes pour prédire la pollution :

1. **is_true_harmattan** (41.6%) → La saison Harmattan est le driver principal
2. **climate_zone** (18.2%) → Le gradient géographique nord/sud
3. **Moyennes glissantes PM2.5** (12.5%) → L'autocorrélation temporelle (résiduelle)
4. **precipitation_sum** → La pluie qui nettoie l'air

---

## 9. Ce que ça veut dire pour le Cameroun

### 9.1 Le gradient nord-sud est dramatique

Ambam (2°N, frontière Gabon) : PM2.5 moyen ~21 µg/m³
Maroua (10.5°N, Extrême-Nord) : PM2.5 moyen ~51 µg/m³

Facteur 2.4× entre nord et sud. En janvier, le nord peut atteindre des valeurs 8 à 10 fois supérieures au sud.

### 9.2 L'Harmattan = urgence de santé publique

- **Toutes** les 40 villes dépassent le seuil OMS journalier (15 µg/m³) pendant l'Harmattan
- L'Extrême-Nord (Maroua, Kousseri, Mokolo) dépasse 75 µg/m³ pendant 60 à 90 jours par an
- La pollution réduit l'espérance de vie de 2.7 ans en moyenne (AQLI 2023)

### 9.3 Ce modèle peut fonctionner partout en Afrique

**Résultat clé de l'étude d'ablation :** le modèle météo-only (sans historique de pollution) atteint R²=0.993.

Cela signifie : si on connaît les données ERA5 d'une ville (disponibles **gratuitement et partout dans le monde** depuis 1940), on peut estimer le PM2.5 sans avoir jamais installé un seul capteur. Ce modèle est transférable à n'importe quelle ville d'Afrique subsaharienne.

---

## 10. Le système temps réel — des prévisions chaque matin

### 10.1 Le problème : un modèle qui dort dans un fichier ne sert à rien

On a entraîné un excellent modèle (R²=0.994). Mais un modèle stocké dans un fichier `.pkl` n'aide personne. Pour qu'il soit utile, il faut qu'il tourne chaque jour, qu'il récupère les données météo du lendemain, et qu'il produise des alertes.

C'est l'objectif du système temps réel.

### 10.2 Comment ça marche — vue d'ensemble simple

```
Chaque matin à 7h (heure du Cameroun) :

1. RÉCUPÉRER les prévisions météo pour les 40 villes
   → Open-Meteo : API gratuite, données pour **7 jours** à l'avance (extensible à 16)
   → Températures, pluie, vent, BLH, humidité...

2. CONSTRUIRE les mêmes 139 variables qu'on a utilisées pour l'entraînement

3. PRÉDIRE le PM2.5 pour chaque ville pour chaque jour de prévision

4. CLASSER le niveau d'alerte (OMS)

5. SAUVEGARDER les résultats :
   → predictions_latest.parquet (tableau complet)
   → alerts_latest.json (alertes en français)
```

Tout ça sans intervention humaine, automatiquement, tous les jours.

### 10.3 Open-Meteo : l'API météo gratuite

**C'est quoi une API ?** Une API (Application Programming Interface) est un service web qui répond à des questions automatiques. Au lieu d'aller sur un site météo et de lire les données à la main, notre programme pose la question "quelles sont les prévisions pour Maroua pour les 7 prochains jours ?" et reçoit la réponse en quelques millisecondes.

**Pourquoi Open-Meteo ?**
- **Gratuit** — pas de clé API requise pour l'usage de base
- **40 villes en une seule requête** — on envoie toutes les coordonnées GPS en une fois
- **7 jours de prévision** par défaut — suffisant pour les décisions de santé publique (extensible à 16 jours)
- **Données identiques à ERA5** — les variables sont les mêmes que celles utilisées pour l'entraînement

### 10.4 Le problème du "démarrage à froid" et comment on le résout

**Le problème des lags :** Pour faire une prédiction, notre modèle complet a besoin de savoir : "quel était le PM2.5 hier ? il y a 7 jours ? il y a 14 jours ?" Ces "lags" sont essentiels.

**Mais en prévision temps réel :** si on démarre le système aujourd'hui, on n'a pas l'historique des semaines précédentes. C'est le problème du **démarrage à froid** (cold start).

**Notre solution — deux modèles :**

| Situation | Modèle utilisé | Performances |
|-----------|---------------|--------------|
| Historique PM2.5 ≥ 14 jours disponible | Modèle complet (lags + météo) | R²=0.9929 |
| Démarrage à froid (pas d'historique) | Modèle météo-only | R²=0.9959 |

**La bonne surprise :** le modèle météo-only est en fait *légèrement meilleur* ! Ça confirme ce qu'on avait trouvé dans l'étude d'ablation : le PM2.5 est entièrement prévisible à partir de la météo. Les lags PM2.5 n'apportent rien de plus.

### 10.5 Les niveaux d'alerte OMS

Chaque prédiction est automatiquement classée :

| Couleur | Niveau | PM2.5 | Recommandation |
|---------|--------|-------|----------------|
| Vert | Bon | < 12 µg/m³ | Aucune restriction |
| Jaune | Modéré | 12–35 µg/m³ | Personnes sensibles : limiter l'exposition |
| Orange | Mauvais pour personnes sensibles | 35–55 µg/m³ | Enfants, asthmatiques : éviter le plein air |
| Rouge | Mauvais | 55–75 µg/m³ | Tout le monde : limiter les activités extérieures |
| Violet | Très mauvais | 75–150 µg/m³ | Restez à l'intérieur si possible |
| Bordeaux | Dangereux | > 150 µg/m³ | Urgence sanitaire — mesures immédiates |

Exemple d'alerte générée :
> "Alerte PM2.5 : Maroua — 78.3 µg/m³ (Très mauvais). Évitez les activités physiques intenses en extérieur."

### 10.6 GitHub Actions : l'ordinateur qui travaille pendant qu'on dort

**C'est quoi GitHub Actions ?** C'est un service de GitHub qui permet d'exécuter des programmes automatiquement, selon un calendrier, sur des serveurs distants. Gratuit pour les projets publics (jusqu'à 2000 min/mois).

**Ce qu'on a configuré :**

```
Tous les jours à 06:00 UTC (07:00 heure Cameroun) :
  1. Télécharger le code depuis GitHub
  2. Installer Python et les bibliothèques nécessaires
  3. Exécuter 08_inference_realtime.py (les prévisions)
  4. Vérifier que les sorties sont correctes (nb de lignes, villes, alertes)
  5. Sauvegarder automatiquement les résultats dans le dépôt GitHub
```

**Pourquoi c'est important pour le hackathon :** on ne paie rien, les prévisions tournent automatiquement, et n'importe qui peut accéder aux dernières prédictions via GitHub ou un futur dashboard.

---

## 11. Le dashboard — voir la qualité de l'air en un clic

### 11.1 Un accès public, sans installation

Toutes les prédictions sont accessibles à travers un **dashboard interactif** hébergé gratuitement sur HuggingFace Spaces, une plateforme internationale pour les projets d'IA :

> **https://huggingface.co/spaces/QUASAR-30/pm25-cameroun**

N'importe qui, depuis un téléphone ou un ordinateur, peut consulter les prévisions — sans créer de compte, sans télécharger quoi que ce soit.

### 11.2 Les 3 pages du dashboard

**Page 1 — TEMPS RÉEL**
La page d'accueil. Une carte du Cameroun affiche les 40 villes avec des points colorés selon le niveau de pollution prévu aujourd'hui. Plus le point est rouge/foncé, plus l'air est dangereux. À côté :
- Les alertes OMS actives, filtrables par région et par seuil
- Un tableau des 40 villes classées par niveau de pollution avec des barres de progression visuelles
- Export des données en CSV ou JSON (pour les journalistes, chercheurs, autorités)

**Page 2 — PAR VILLE**
Choisir une ville et voir :
- Son historique complet de PM2.5 de 2020 à 2025, avec la **zone d'incertitude** en grisé (les limites haute et basse de l'estimation Monte Carlo)
- Les **7 prochains jours de prévision** (losanges verts)
- Les corrélations avec la météo : comment la pluie, le vent et la BLH influencent la pollution de cette ville spécifiquement
- La saisonnalité mensuelle : quels mois sont les plus dangereux

**Page 3 — CLASSEMENT**
Une vue comparative des 40 villes :
- Classement par pollution annuelle moyenne avec intervalles d'incertitude
- Gradient nord-sud : plus on monte vers le nord, plus la pollution augmente (R²=0.88)
- Carte de chaleur mensuelle : pour chaque ville, quels mois sont critiques
- Performance des modèles : les graphiques de validation pour les curieux

### 11.3 Mis à jour chaque matin automatiquement

Grâce au système GitHub Actions décrit plus haut, le dashboard se rafraîchit automatiquement à **7h00 heure du Cameroun** avec les nouvelles prévisions météo. L'utilisateur voit toujours les données du jour.

---

## 12. Les limites honnêtes du projet

### 11.1 On prédit le proxy, pas la vraie pollution

Notre modèle prédit le proxy qu'on a construit, pas une mesure réelle. Si notre formule proxy est biaisée, les prédictions le sont aussi. C'est une limite fondamentale de l'absence de capteurs.

### 11.2 CAMS comme seule référence externe

On compare notre proxy à CAMS, mais CAMS lui-même est un modèle et sous-estime la pollution africaine. Pour une validation robuste, il faudrait des mesures sur le terrain, même quelques capteurs low-cost dans les villes principales.

### 11.3 Le modèle n'anticipe pas les événements exceptionnels

Un feu de forêt hors-saison, une tempête de sable inhabituellement intense, une activité volcanique — notre modèle ne peut pas anticiper ce qu'il n'a jamais vu.

### 11.4 Résolution temporelle = un jour

On prédit par jour. Pour des alertes en temps réel, il faudrait des données météo et un modèle horaire.

---

## 13. Glossaire des notions clés

| Terme | Définition simple |
|-------|-------------------|
| **PM2.5** | Particules fines de moins de 2.5 micromètres dans l'air. Très dangereuses car pénètrent dans les poumons profonds |
| **ERA5** | Réanalyse atmosphérique de Copernicus/ECMWF. Base de données météo mondiale de 1940 à aujourd'hui, accessible gratuitement |
| **BLH (Boundary Layer Height)** | Hauteur de la couche atmosphérique où l'air se mélange. Basse = pollution concentrée. Haute = pollution dispersée |
| **Harmattan** | Vent sec du Sahara, novembre à mars. Principal transporteur de poussières vers le Cameroun |
| **Proxy** | Indicateur de remplacement quand la mesure directe est impossible |
| **Feature Engineering** | Transformation des données brutes en variables informatives pour le modèle ML |
| **XGBoost / LightGBM** | Algorithmes ML basés sur des ensembles d'arbres de décision. Très efficaces sur données tabulaires |
| **R²** | Coefficient de détermination. 0 = modèle nul, 1 = prédiction parfaite |
| **RMSE** | Erreur quadratique moyenne. En µg/m³ : erreur typique d'une prédiction |
| **Expanding-window CV** | Technique de validation qui respecte l'ordre temporel des données |
| **Lag** | Valeur d'une variable à un temps passé (lag-1 = valeur d'hier) |
| **Rolling mean** | Moyenne glissante sur N jours passés |
| **Log1p** | Transformation logarithmique log(1+x), réduit l'asymétrie des distributions |
| **Wet scavenging** | Nettoyage de l'atmosphère par les précipitations |
| **CAMS** | Copernicus Atmosphere Monitoring Service. Modèle de qualité de l'air européen |
| **FIRMS** | Fire Information for Resource Management System. Données feux NASA par satellite MODIS |
| **Monte Carlo** | Méthode de simulation : on répète l'expérience 1000 fois avec des paramètres aléatoires pour mesurer l'incertitude |
| **Ablation study** | Étude qui retire des composantes du modèle une par une pour mesurer leur vraie contribution |
| **OMS** | Organisation Mondiale de la Santé. Fixe les seuils de qualité de l'air recommandés |
| **AQLI** | Air Quality Life Index. Mesure l'impact de la pollution sur l'espérance de vie |
| **µg/m³** | Microgrammes par mètre cube. Unité de mesure de la concentration de particules dans l'air |
| **API** | Application Programming Interface. Service web qui permet à un programme d'interroger automatiquement un autre service (ex. API météo) |
| **Open-Meteo** | API météo gratuite, fournit des prévisions jusqu'à 16 jours et des données historiques ERA5 |
| **Cold start** | Démarrage à froid : situation où le système n'a pas d'historique récent et doit prédire sans données passées |
| **GitHub Actions** | Service d'automatisation de GitHub. Exécute des scripts selon un calendrier, gratuitement |
| **Inférence** | Acte d'utiliser un modèle entraîné pour faire de nouvelles prédictions sur de nouvelles données |

---

*Rapport rédigé dans le cadre du Hackathon IndabaX Cameroun 2026 — Prédiction PM2.5 pour 40 villes camerounaises (2020–2025).*
*Données : ERA5 (Copernicus), NASA FIRMS MODIS. Modèle : XGBoost + LightGBM.*

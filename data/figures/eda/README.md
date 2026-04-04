# Guide de lecture — Figures EDA (Analyse Exploratoire)

Ces 8 figures documentent les **décisions prises avant le feature engineering**.
Chaque figure répond à une question précise. Rien n'est fait par supposition.

---

## 01_missing_values.png — Valeurs manquantes par colonne

**Ce que tu vois :** Barres horizontales, longueur = % de NaN par variable.

**Comment lire :**
- Barres dépassant la ligne rouge (20%) = problème majeur
- `wind_speed_10m_max` : ~89% de NaN → quasi-inutilisable dans ce dataset
- `temperature_2m_min/max` : ~60% de NaN
- Les variables ERA5 pures (blh_mean, blh_min, precipitation_sum) : 0–5% de NaN → fiables

**Ce que ça implique :**
Le dataset Excel officiel est fortement dégradé sur les colonnes météo. C'est pourquoi
tout le pipeline utilise `pm25_proxy_era5.parquet` (ERA5 pur, 0 NaN) comme source,
et non le dataset Excel fusionné.

---

## 02_imputation_comparison.png — Comparaison des stratégies d'imputation

**Ce que tu vois :**
- Gauche : scatter "vraie valeur vs valeur imputée" pour 3 méthodes
- Droite : barres RMSE (plus court = meilleur)

**Comment lire :**
- Points proches de la diagonale y=x = bonne imputation
- Barres RMSE : interpolation linéaire (vert) est la plus courte
- L'écart entre méthodes est très large : RMSE 89 vs 194 (médiane) vs 221 (moyenne globale)

**Ce que ça implique :**
L'interpolation linéaire par ville est choisie pour imputer les NaN résiduels.
Elle exploite la continuité temporelle (les valeurs voisines dans le temps sont proches),
ce que la médiane ou la moyenne globale ne font pas.

---

## 03_target_distribution.png — Distribution de la variable cible PM2.5

**Ce que tu vois :** 3 panneaux — histogramme brut, histogramme log1p, QQ-plot.

**Comment lire :**
- Histogramme brut : longue queue droite (skewness=2.14) → non-normal → mauvais pour ML
- Histogramme log1p : beaucoup plus symétrique (skewness=0.49) → quasi-normal → bon pour ML
- QQ-plot : points suivent la diagonale rouge après transformation → distribution acceptable
- Lignes rouges/oranges : seuils OMS (5 µg/m³ annuel, 15 µg/m³ 24h)

**Ce que ça implique :**
La cible ML est `pm25_log = log1p(pm25_proxy)` et non `pm25_proxy` directement.
Entraîner sur une cible asymétrique biaise le modèle vers les valeurs moyennes
et sous-estime les pics (exactement ce qu'on ne veut pas pour les alertes).

---

## 04_feature_distributions.png — Distributions des features

**Ce que tu vois :** Histogrammes des variables météo brutes.

**Comment lire :**
- Barres très concentrées sur la gauche avec queue longue = distribution log-normale = candidat à la transformation log
- Variables symmétriques = pas besoin de transformation

**Ce que ça implique :**
`precipitation_sum` (skew 8.6), `blh_min` (2.6), `wind_speed` (1.8) sont log-transformées
pour réduire l'asymétrie et améliorer le signal pour le modèle ML.

---

## 05_correlations.png — Corrélations Spearman avec PM2.5

**Ce que tu vois :**
- Gauche : barres horizontales, rouge = corrélation négative, vert = positive
- Droite : scatter PM2.5 vs weather_code (exemple de relation non-linéaire)

**Comment lire :**
- Corrélation Spearman (contrairement à Pearson) capture les relations monotones non-linéaires
- Barres rouges longues = la variable baisse quand PM2.5 monte (ex: humidité → pluie → moins de pollution)
- Barres vertes = la variable monte avec PM2.5 (ex: direction vent N/NE = Harmattan)
- r < 0.1 ou > -0.1 = signal faible ou inexistant

**Ce que ça implique :**
Les variables à faible corrélation ne sont pas supprimées — elles peuvent avoir un signal
non-monotone ou interagir avec d'autres variables. XGBoost capture ces interactions.

---

## 06_correlation_matrix.png — Matrice de multicolinéarité

**Ce que tu vois :** Heatmap colorée variable × variable. Rouge = forte corrélation positive, bleu = négative.

**Comment lire :**
- Diagonale = toujours 1.0 (variable corrélée avec elle-même)
- Carrés rouge foncé hors diagonale = paires de variables très redondantes
- Ex : temperature_2m_max et temperature_2m_min → r > 0.95 → redondantes

**Ce que ça implique :**
Pour les modèles basés sur les arbres (XGBoost/LightGBM), la multicolinéarité ne
dégrade pas les performances mais dilue l'importance des features. On garde les deux
variables d'une paire colinéaire car l'une peut capturer des nuances que l'autre manque.

---

## 07_feature_importance.png — Importance XGBoost sur données brutes

**Ce que tu vois :**
- Gauche : barres classées par importance (gain = contribution à la réduction d'erreur)
- Droite : courbe cumulative → quelle fraction du signal est capturée par les N premières features

**Comment lire :**
- Barres longues = features qui réduisent le plus l'erreur de prédiction
- Courbe cumulative : ligne verte = 90% du signal capturé avec seulement 8 features
- Cela signifie que 80%+ des variables sont redondantes ou à faible signal

**Ce que ça implique :**
Sur les données brutes (sans feature engineering), `relative_humidity_2m_max` est la
feature la plus importante. Après feature engineering, `is_true_harmattan` prendra la
première place — ce qui est plus interprétable physiquement.

---

## 08_group_analysis.png — Analyse par région et saisonnalité

**Ce que tu vois :**
- Gauche : boxplots PM2.5 par région (du sud au nord)
- Droite : saisonnalité nationale mensuelle

**Comment lire :**
**Boxplot :** la boîte centrale = 50% des observations (Q1–Q3), la ligne centrale = médiane,
les moustaches = 1.5×IQR, les points = outliers.
- Boîte haute et large = PM2.5 élevé et variable (nord en saison Harmattan)
- Boîte basse et étroite = PM2.5 faible et stable (sud équatorial)
- ANOVA F=1887 p≈0 : la région explique statistiquement une grande part de la variance

**Saisonnalité :** barres hautes Jan–Mar = Harmattan, creux Jul–Sep = saison des pluies.
La ligne rouge pointillée = référence AQLI 32.5 µg/m³.

**Ce que ça implique :**
La région (latitude) et la saison sont les deux drivers structurants principaux.
Ces deux dimensions doivent absolument être représentées dans les features ML :
→ `lat_norm`, `region_code`, `is_harmattan`, `harmattan_intensity`.

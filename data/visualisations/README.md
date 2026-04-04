# Guide de lecture — Visualisations du dataset PM2.5

Ces 10 figures donnent une **vue complète du signal PM2.5 sur les 40 villes camerounaises
(2020–2025)**. Elles sont générées depuis le proxy final calibré (V4).

Ordre de lecture recommandé : 01 → 03 → 04 → 02 → 05 → 09 → 06 → 10 → 07 → 08

---

## 01_distribution_pm25.png — Distribution de la variable cible

**Ce que tu vois :** Histogramme brut, histogramme log1p, QQ-plot.

**Comment lire :**
- Histogramme bleu (brut) : asymétrie forte vers la droite = quelques villes/jours ont des
  valeurs extrêmes (Harmattan intense). Skewness = 1.91.
- Histogramme orange (log1p) : après transformation logarithmique, la distribution est
  quasi-symétrique. Skewness = 0.36. C'est la cible utilisée pour entraîner le ML.
- QQ-plot : si les points bleus suivent la ligne rouge → distribution normale.
  La légère déviation en haut à droite = quelques jours extrêmes persistent.

**Ce que ça implique :**
La transformation log1p est nécessaire pour que le modèle ML traite équitablement les
jours normaux ET les pics. Sans elle, le modèle optimiserait la précision sur les jours
courants (80% du temps) et raterait systématiquement les épisodes Harmattan intenses.

---

## 02_saisonnalite.png — Saisonnalité nationale et par zone climatique

**Ce que tu vois :** 2 graphes — moyenne nationale par mois, moyenne par zone et par mois.

**Comment lire :**
- Barres bleues = moyenne nationale mensuelle (avec écart-type en barres d'erreur)
- Ligne rouge pointillée = référence AQLI 32.5 µg/m³
- Ligne rouge fine = OMS 24h (15 µg/m³)
- Les 3 courbes colorées (droite) = zones équatoriale, transition, sahélienne

**Ce que ça implique :**
- Pic Jan–Mar = Harmattan : vents secs du Sahara apportent des poussières
- Creux Jun–Sep = saison des pluies : la pluie "lave" l'atmosphère
- L'écart entre zones est immense : sahélien (rouge) peut atteindre 90+ µg/m³ en Harmattan,
  équatorial (bleu) reste sous 35 µg/m³ toute l'année
- **Implication santé** : toutes les barres dépassent largement les seuils OMS

---

## 03_gradient_geographique.png — Gradient PM2.5 nord-sud

**Ce que tu vois :** Scatter latitude vs PM2.5, carte bubble des 40 villes.

**Comment lire :**
- Scatter (gauche) : chaque point = une ville, couleur = zone climatique
  - Axe X = latitude (°N) : gauche = sud, droite = nord
  - Axe Y = PM2.5 moyen annuel
  - Ligne pointillée grise = tendance linéaire
- Carte bubble (droite) : taille et couleur de chaque cercle = niveau PM2.5

**Ce que ça implique :**
Le gradient est quasi-linéaire : chaque degré de latitude nord = +3–4 µg/m³ de PM2.5.
Maroua (12°N) ≈ 75 µg/m³, Ambam (2°N) ≈ 20 µg/m³. L'écart facteur 3–4 entre
nord et sud est le signal le plus fort du dataset. C'est le reflet de la proximité
au Sahara et de l'intensité du Harmattan.

---

## 04_heatmap_villes_mois.png — Heatmap 40 villes × 12 mois

**Ce que tu vois :** Tableau coloré, villes en lignes (nord en haut), mois en colonnes.

**Comment lire :**
- Couleur rouge foncé = PM2.5 très élevé (>80 µg/m³), jaune = modéré, vert clair = faible
- Lignes du haut (Kousseri, Mokolo, Yagoua) = Extrême-Nord = rouge intense Jan–Mar
- Lignes du bas (Sangmelima, Ebolowa, Ambam) = Sud équatorial = vert clair toute l'année
- Colonnes Jan–Mar = Harmattan = rouge presque partout

**Ce que ça implique :**
On voit d'un coup d'œil toute la structure spatio-temporelle :
- La structure horizontale (couleurs qui varient par mois) = saisonnalité Harmattan
- La structure verticale (dégradé du rouge au vert) = gradient géographique nord-sud
- Les villes "grises" (valeurs intermédiaires stables) = zone de transition (Adamaoua)

---

## 05_series_temporelles.png — Séries 2020–2025 pour 6 villes

**Ce que tu vois :** 6 graphes individuels, un par ville représentative.

**Comment lire :**
- Ligne colorée = PM2.5 journalier
- Ligne épaisse = lissage 30 jours (tendance)
- Zones beiges ombragées = périodes Harmattan (Nov–Mar)
- Lignes horizontales = seuils OMS (rouge=24h, orange=annuel)

**Ce que ça implique :**
- Maroua/Garoua (nord) : pics annuels réguliers à 100–200 µg/m³, parfaitement alignés
  sur les ombrages Harmattan → le proxy capture très bien les épisodes
- Yaoundé/Douala (équatorial) : signal plat autour de 20–35 µg/m³, faibles variations
- **Aucune rupture visible en 2024** malgré le patch climatologique BLH → la correction
  est transparente dans les séries

---

## 06_boxplots_regions.png — Distribution PM2.5 par région

**Ce que tu vois :** Boxplots et violins, régions classées du sud au nord.

**Comment lire :**
**Boxplot :** boîte = Q1–Q3 (50% des jours), ligne centrale = médiane,
moustaches = 1.5×IQR, points au-delà = jours extrêmes.
**Violin :** la largeur = densité des observations à ce niveau.

- Régions du haut (Extrême-Nord, Nord) : boîtes hautes, violins bimodaux
  (deux bosses = saison Harmattan + saison pluies)
- Régions du bas (Sud, Est) : boîtes basses, violins unimodaux et étroits

**Ce que ça implique :**
La distribution bimodale dans les régions du nord est un signal clé : ces régions
ont deux "états" bien distincts (propre vs pollué). Cela justifie la feature
`is_harmattan` (variable binaire) plutôt qu'une variable continue seulement.

---

## 07_correlations_features.png — Corrélations des features avec PM2.5

**Ce que tu vois :** Barres Spearman (top 25), nuage Pearson vs Spearman.

**Comment lire :**
- Barres vertes = corrélation positive (variable augmente avec PM2.5)
- Barres rouges = corrélation négative (variable diminue quand PM2.5 augmente)
- Plus la barre est longue, plus la relation est forte (en valeur absolue)
- Nuage droit : points proches de la diagonale = relation linéaire
  Points éloignés de la diagonale (en bas à gauche) = relation non-linéaire

**Ce que ça implique :**
- `pm25_proxy_rm7_mean` (lag 7j) en tête : l'autocorrélation temporelle est forte
  MAIS l'ablation study montre que c'est de la redondance, pas de la causalité
- `harmattan_intensity` et `lat_x_harmattan` : fortes corrélations positives →
  les features Harmattan créées capturent bien le signal
- `precipitation_sum` : Spearman -0.66 → le meilleur prédicteur météo pur
- Points loin de la diagonale dans le nuage = justifient Spearman > Pearson pour ces données

---

## 08_importance_xgboost.png — Importance features XGBoost

**Ce que tu vois :** Top 30 features par importance, courbe cumulative.

**Comment lire :**
- Barres vertes = features dans les 13 premières (qui totalisent 90% de l'importance)
- Barres bleues = features au-delà du seuil 90%
- Courbe droite : ligne verte = nombre de features pour atteindre 90% du signal (ici 13)

**Ce que ça implique :**
Avec seulement 13 features sur 139, le modèle capture 90% du signal.
- `is_true_harmattan` domine (27%) → le Harmattan est le driver #1
- `climate_zone` #2 (20%) → le gradient géographique est le driver #2
- Les features météo (precip_cat, weather_proxy) sont dans les 13 premières →
  la physique est bien apprise, pas juste l'autocorrélation

---

## 09_evolution_annuelle.png — Tendance nationale 2020–2025

**Ce que tu vois :** Courbe nationale par année, heatmap région × année.

**Comment lire :**
- Courbe gauche : chaque point = moyenne annuelle nationale PM2.5
  - Ligne bleue = moyenne, verte = médiane
  - Lignes de référence = AQLI 32.5 et OMS 24h
- Heatmap droite : cases = PM2.5 moyen par (région, année)
  - Rouge = PM2.5 élevé, blanc = modéré, axe Y = régions, axe X = années

**Ce que ça implique :**
La moyenne nationale est stable autour de 32–33 µg/m³ sur 2020–2025.
**Pas de dérive artificielle** du proxy → la calibration C_base est robuste.
La heatmap révèle une légère tendance à la hausse dans l'Extrême-Nord (2024–2025
légèrement plus rouge que 2020–2021) — peut refléter de vraies tendances climatiques
ou une variabilité naturelle interannuelle.

---

## 10_meteo_vs_pm25.png — Relations entre variables météo et PM2.5

**Ce que tu vois :** 6 scatter plots, chacun = une variable météo vs PM2.5.

**Comment lire :**
- Points bleus = observations journalières (87 240 points au total)
- Courbe rouge = tendance médiane binée (moyenne glissante robuste)
- r = corrélation de Spearman en haut de chaque graphe

**Ce que ça implique pour chaque variable :**

| Variable | r | Interprétation |
|---|---|---|
| Humidité max | -0.41 | Saison humide = air humide = moins de dust. Relation non-linéaire (plateau à haute humidité) |
| Précipitations | -0.66 | Plus fort signal météo direct. Dès 5–10mm, PM2.5 chute abruptement (wet scavenging) |
| BLH | -0.13 | Signal faible : Harmattan a BLH modérée mais PM2.5 très élevé (effet dust confondant) |
| Vitesse vent | -0.018 | Quasi-nul : vent Harmattan (fort + poussiéreux) et vent convectif (fort + propre) s'annulent |
| Durée du jour | -0.60 | Jours courts = hiver = Harmattan. Corrélation indirecte via la saisonnalité |
| Radiation solaire | +0.49 | Paradoxalement positif : saison sèche = ciel clair ET poussiéreux simultanément |

**La leçon clé :** les relations brutes sont souvent contre-intuitives ou faibles
car plusieurs mécanismes se superposent. C'est pourquoi les interactions
(wind × harmattan, BLH × précip) sont plus informatives que les variables seules.

# Guide de lecture — Figures Modèle ML

Ces figures documentent les **performances et comportements du modèle XGBoost/LightGBM**.
Elles répondent à : "le modèle prédit-il bien, et apprend-il vraiment la physique ?"

---

## model_diagnostics.png — Diagnostics complets du modèle (6 panneaux)

### Panneau A — RMSE par fold (expanding-window CV)

**Comment lire :**
- Chaque groupe de barres = un fold de validation
- Bleu = XGBoost, orange = LightGBM
- La hauteur = RMSE (µg/m³) — plus bas = meilleur
- Les annotations en rouge = R² correspondant

**Ce que ça implique :**
- Fold 1 (train=2020 seulement) a un RMSE plus élevé (~3.4) : normal car 1 an de données
  est insuffisant pour apprendre les patterns complets.
- Folds 2–4 convergent à RMSE ~1.5–1.7 µg/m³ : le modèle se stabilise dès qu'il a
  2+ années de données. C'est un signe de **bonne généralisation temporelle**.
- XGBoost et LightGBM donnent des résultats très proches — pas de modèle nettement supérieur.

**Règle d'interprétation :** si Fold4 était beaucoup plus mauvais que Fold2, cela
indiquerait un concept drift (les patterns changent dans le temps). Ici ce n'est pas le cas.

---

### Panneau B — Scatter Test 2025 (R²=0.994)

**Comment lire :**
- Axe X = PM2.5 réel (proxy ERA5 qu'on cherche à prédire)
- Axe Y = PM2.5 prédit par le modèle ensemble
- Ligne rouge pointillée = diagonale parfaite y=x
- Chaque point = une paire (ville, jour) en 2025

**Ce que ça implique :**
Les points très serrés autour de la diagonale sur toute la plage [0–160 µg/m³]
signifient que le modèle prédit bien pour toutes les conditions : saison sèche,
saison des pluies, nord et sud. Pas de biais systématique visible.
R²=0.994 signifie que 99.4% de la variance est expliquée — excellent.

**Attention :** ce R² élevé est attendu car la cible est construite depuis ERA5
et le modèle utilise des features ERA5. Ce n'est pas une validation contre des mesures
réelles. Voir `data/figures/validation/` pour la validation externe vs CAMS.

---

### Panneau C — Top 20 features importantes (XGBoost)

**Comment lire :**
- Barres = "gain" XGBoost = contribution moyenne à la réduction de l'erreur
- Rouge = features lags/rolling PM2.5 (autocorrélation temporelle)
- Bleu = features météo/physiques
- `is_true_harmattan` en tête = 41.6% d'importance

**Ce que ça implique :**
Le fait que `is_true_harmattan` domine (et non un lag PM2.5) signifie que le modèle
apprend réellement le mécanisme physique Harmattan → PM2.5 élevé.
`climate_zone` en 2ème place confirme l'importance du gradient géographique nord/sud.
Les lags PM2.5 apparaissent (~3ème–6ème) mais leur contribution réelle est faible
(confirmé par l'ablation study — voir ablation_study.png).

---

### Panneau D — Série temporelle Yaoundé 2025

**Comment lire :**
- Ligne noire = PM2.5 réel (proxy ERA5)
- Ligne rouge pointillée = PM2.5 prédit par le modèle
- Les deux courbes doivent se superposer au maximum

**Ce que ça implique :**
Pour Yaoundé (ville équatoriale, PM2.5 modéré), le modèle suit bien les variations
journalières. Les légères différences sur les pics courts (quelques jours) sont normales
— ces fluctuations rapides sont difficiles à prévoir depuis les seules données météo.

---

### Panneau E — Biais mensuel (Test 2025)

**Comment lire :**
- Barres = biais moyen (prédit − réel) par mois
- Barres rouges = sur-estimation, bleues = sous-estimation
- Barres d'erreur = ± 1 écart-type inter-villes
- Ligne noire = biais zéro (idéal)

**Ce que ça implique :**
Le biais est proche de zéro pour tous les mois → pas de biais saisonnier systématique.
La légère sous-estimation en début Harmattan (Oct–Nov) est typique : le modèle est
"en retard" sur les transitions abruptes. Les barres d'erreur larges reflètent
la grande variabilité entre les 40 villes (nord vs sud).

---

### Panneau F — RMSE par ville (Test 2025)

**Comment lire :**
- Barres horizontales = RMSE pour chaque ville, classées du plus faible au plus fort
- Couleur verte = faible RMSE (bonne prédiction), rouge = RMSE élevé
- Villes du bas = meilleures prédictions, villes du haut = plus difficiles

**Ce que ça implique :**
Les villes avec RMSE élevé (~3–4 µg/m³) sont principalement les villes du nord
(Maroua, Kousseri) où les pics Harmattan sont les plus intenses et les plus variables.
Les villes du sud ont RMSE < 1 µg/m³ — très faciles à prédire car peu variables.
**Ce gradient est physiquement cohérent** : là où le signal est plus fort, l'erreur
absolue est aussi plus grande.

---

## ablation_study.png — Contribution des lags vs physique météo

**Ce que tu vois :** 3 métriques (RMSE, R², MAPE) pour 3 versions du modèle.

**Comment lire :**
- 3 barres par métrique = 3 modèles testés
  - Gris : Persistence seule = "PM2.5 demain = PM2.5 aujourd'hui" (aucun ML)
  - Bleu : Météo-only = XGBoost avec UNIQUEMENT les features météo (sans lags PM2.5)
  - Rouge : Full model = XGBoost avec tout (météo + lags PM2.5)

**Règles de lecture :**
- RMSE et MAPE : barres plus courtes = meilleur
- R² : barres plus hautes = meilleur

**Ce que ça implique — le résultat le plus important du projet :**

| Question | Réponse |
|---|---|
| La persistence est-elle un bon modèle ? | Non : RMSE=10, R²=0.745, MAPE=28% |
| Les features météo suffisent-elles ? | Oui : RMSE=1.6, R²=0.993, MAPE=2.4% |
| Les lags PM2.5 apportent-ils quelque chose ? | Presque rien : ΔR²=-0.0005 |

La météo-only atteint R²=0.993 sans aucun historique PM2.5.
Cela confirme que le modèle apprend réellement la physique (Harmattan, pluies, BLH)
et non juste "demain ressemble à aujourd'hui".

**Conséquence pratique :** ce modèle peut être appliqué à une nouvelle ville
sans aucun historique PM2.5, à condition d'avoir les données ERA5 — ce qui est
disponible partout dans le monde.

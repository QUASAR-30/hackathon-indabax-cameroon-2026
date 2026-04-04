# Guide de lecture — Figures Validation & Incertitude

Ces 3 figures évaluent la **crédibilité du proxy PM2.5** construit depuis ERA5.
Elles répondent à : "notre proxy est-il physiquement cohérent et bien calibré ?"

---

## validation_cams.png — Validation vs CAMS (Option C)

**Ce que tu vois :** 3 panneaux — scatter proxy/CAMS, saisonnalité comparative, biais par ville.

### Panneau gauche — Scatter Proxy vs CAMS

**Comment lire :**
- Axe X = PM2.5 CAMS (modèle de transport atmosphérique Copernicus)
- Axe Y = PM2.5 notre proxy ERA5
- Diagonale pointillée noire = ligne parfaite y=x (proxy = CAMS)
- Couleur des points = mois (bleu foncé = début d'année = Harmattan, jaune = été)
- r=0.339 = corrélation de Spearman entre les deux produits
- Biais NMB=+85% = notre proxy est en moyenne 85% au-dessus de CAMS

**Ce que ça implique :**
Le nuage de points au-dessus de la diagonale confirme que notre proxy donne des valeurs
plus élevées que CAMS. Cela est attendu car CAMS sous-estime systématiquement le PM2.5
en Afrique subsaharienne (NMB documenté de -20% à -50% dans la littérature).
r=0.339 est dans la fourchette normale pour l'Afrique (0.3–0.5 selon la littérature).
**Ce n'est pas un échec — c'est une limite connue de CAMS en contexte africain.**

### Panneau central — Saisonnalité comparative

**Comment lire :**
- Courbe bleue = notre proxy (valeurs moyennes mensuelles)
- Courbe orange = CAMS (valeurs moyennes mensuelles)
- Les deux courbes doivent avoir la même forme (pic Jan–Fév, creux Jul–Sep)

**Ce que ça implique :**
Les deux produits montrent le même cycle saisonnier (Harmattan hiver, pluies été).
Notre proxy amplifie davantage les pics (~55 µg/m³ vs ~12 µg/m³ pour CAMS en janvier).
Le ratio saisonnier proxy (×2.21) est dans les bornes de la littérature pour le nord Cameroun (2.0–3.0).

### Panneau droit — Biais par ville

**Comment lire :**
- Barres = biais moyen (proxy − CAMS) par ville de validation
- Plus la barre est longue, plus le proxy sur-estime CAMS pour cette ville
- Maroua (nord) a le plus grand biais, Bafoussam (centre) le plus petit

**Ce que ça implique :**
Le biais est gradient nord/sud : il augmente avec la latitude. C'est cohérent avec
le Harmattan plus intense au nord. La configuration HARMATTAN_STRENGTH=1.4 a été
choisie pour limiter ce biais tout en restant physiquement réaliste.

---

## validation_blh_active.png — BLH Mean vs BLH Max (Option D)

**Ce que tu vois :** 2 panneaux — distributions de F_stagnation, saisonnalité de F_stagnation.

### Panneau gauche — Distributions F_stagnation

**Comment lire :**
- Bleu = F_stagnation calculé avec blh_mean (moyenne 24h) → moyenne 1.665
- Orange = F_stagnation calculé avec blh_max (BLH de journée) → moyenne 0.855
- L'écart entre les deux distributions = facteur ~2

**Ce que ça implique :**
La BLH nocturne peut descendre à 20–50m (inversion thermique stable) ce qui donne
F_stagnation = (1000/50)^0.6 ≈ 10 — une valeur absurde.
Utiliser blh_mean inclut ces valeurs nocturnes extrêmes, gonflant artificiellement le proxy.
C'est pourquoi BLH_MIN = 250m a été introduit (plancher recommandé par la littérature)
pour "couper" les valeurs nocturnes irréalistes sans passer entièrement à blh_max.

### Panneau droit — Saisonnalité de F_stagnation

**Comment lire :**
- Courbe bleue (blh_mean) : monte à ~2.0 en saison sèche (Jan–Mar), descend en saison des pluies
- Courbe orange (blh_max) : quasi-plate à ~0.8–1.0 toute l'année

**Ce que ça implique :**
Si on utilise blh_max, le facteur de stagnation perd tout son signal saisonnier
(la courbe est plate). On perdrait l'information sur la variabilité de la dispersion.
Si on utilise blh_mean sans plancher, on sur-amplifie les nuits d'hiver.
**La solution retenue (BLH_MIN=250m) préserve le signal saisonnier tout en évitant les extrêmes.**

---

## validation_monte_carlo.png — Monte Carlo 1000 tirages (Option E)

**Ce que tu vois :** 4 panneaux — distribution nationale MC, IC90% saisonnier, sensibilité, IC90% par ville.

### Panneau A — Distribution de la moyenne nationale

**Comment lire :**
- Chaque barre = fréquence d'une valeur de la moyenne nationale PM2.5 sur 1000 simulations
- Les 1000 simulations varient tous les paramètres physiques dans leurs plages littérature
- Ligne rouge = proxy nominal (32.5 µg/m³) — doit être au centre de la distribution
- Lignes orange = P5 et P95 = intervalle de confiance 90%

**Ce que ça implique :**
La distribution est très concentrée autour de 32.5 µg/m³ (faible dispersion).
La calibration C_base est robuste — elle s'ajuste automatiquement à chaque tirage.
L'incertitude sur la **moyenne nationale** est très faible : CV ≈ 8.9%.

### Panneau B — Incertitude saisonnière IC90%

**Comment lire :**
- Ligne centrale = proxy nominal
- Zone bleue = IC90% (entre P5 et P95 des 1000 tirages)
- Zones larges = mois où l'incertitude est grande
- Zones étroites = mois où le proxy est robuste quel que soit le paramètre

**Ce que ça implique :**
L'incertitude est plus large en saison Harmattan (Nov–Mar) car c'est là où les
paramètres HARMATTAN_STRENGTH et RAIN_K ont le plus d'impact.
En saison des pluies, le proxy est plus certain car la pluie (F_wet) domine et
son effet est bien contraint.

### Panneau C — Analyse de sensibilité

**Comment lire :**
- Barres = corrélation de rang entre chaque paramètre et la moyenne nationale PM2.5
- Plus la barre est longue, plus le paramètre influence le résultat final
- Ligne orange = seuil "modéré" (|r|=0.3)
- rain_k (rouge) dépasse largement tous les autres

**Ce que ça implique :**
`rain_k` (coefficient de lessivage par la pluie) est de loin le paramètre le plus
influent (|r|=0.803). Une incertitude de ±50% sur rain_k change le PM2.5 moyen
de ~30%. En revanche, l'exposant BLH (alpha=0.60) a peu d'impact (|r|=0.158),
ce qui justifie de ne pas chercher à l'optimiser finement.

### Panneau D — IC90% par ville

**Comment lire :**
- Barres horizontales = largeur de l'IC90% pour chaque ville
- Ville avec barre longue = proxy moins certain pour cette ville
- Classées approximativement nord (haut, barres longues) → sud (bas, barres courtes)

**Ce que ça implique :**
Les villes du nord (Maroua, Kousseri) ont une incertitude plus grande (~15 µg/m³)
car elles sont plus exposées au Harmattan — le paramètre h_strength y a plus d'impact.
Les villes équatoriales du sud ont des IC90% étroits (~4–6 µg/m³) — le proxy y est
plus robuste car dominé par les précipitations (paramètre mieux contraint).

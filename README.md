
# Campagne_Market – Analyse marketing & segmentation client (README)

> **TL;DR**  
> Projet data-marketing (6 jours, équipe de 5) sous **Python + Jupyter** pour :  
> 1) **Mesurer l’efficacité des campagnes** (KPIs + ROI),  
> 2) **Segmenter la clientèle** (clustering),  
> 3) **Prédire la réponse client** (classification),  
> 4) **Raconter une histoire business** claire en soutenance (15’ + 10’ Q&A).

---

## 1) Contexte & rôle

La Direction Marketing d’une multinationale de la grande distribution a entendu parler de vos prouesses de data analyste.  
Objectifs :  
- **Chiffrer** l’efficacité des campagnes marketing.  
- **Cerner** la cible client et adapter la stratégie.  
- **Développer** une démarche analytique **méthodique** et **cohérente**, adaptée à des enjeux professionnels.

**Durée & outils** : 6 jours en groupe de 5 – **Python** (Jupyter Notebook).

---

## 2) Livrables attendus

- Un **Notebook** (ou plusieurs) rassemblant :  
  - Analyse exploratoire (EDA) et **data quality**.  
  - **KPIs** marketing (incluant campagnes 1..5).  
  - **Feature engineering** (justifié).  
  - **Segmentation client** (clustering) + interprétation.  
  - **Modèle(s) prédictif(s)** de réponse client (classification) + évaluation.  
  - Synthèse business & **reco actionnables**.
- **Soutenance** (15 min) + **Q&A** (10 min).  
- Un **README** (ce fichier) + un **requirements.txt** + scripts reproductibles.  
- Optionnel : un **rapport PDF** d’aide à la décision pour les directions (Marketing / Ventes).

---

## 3) Données & dictionnaire (exemple type)

Fichier principal : `data/Campagne_Market.csv` (ou `camp_market_clean.csv`).  
> **Note** : adaptez les noms de colonnes aux vôtres. Ci-dessous un **exemple** courant pour un dataset marketing de retail.

| Colonne (ex)              | Description                                                                                 |
|---------------------------|---------------------------------------------------------------------------------------------|
| `CustomerID`              | Identifiant client unique                                                                  |
| `Age`                     | Âge (années)                                                                               |
| `Sexe`                    | Sexe                                                                                       |
| `Revenu`                  | Revenu annuel (€)                                                                          |
| `Ville` / `Region`        | Zone géographique                                                                          |
| `Freq`                    | **Fréquence d’achats** sur la période d’observation                                        |
| `CA` / `Monetary`         | **Chiffre d’affaires** (dépenses totales sur la période)                                   |
| `Recency`                 | **Récence** (jours depuis le dernier achat)                                                |
| `Nb_achats_en_ligne`      | Nombre d’achats effectués **en ligne**                                                     |
| `Nb_achats_magasin`       | Nombre d’achats effectués **en magasin**                                                   |
| `Promo_ratio`             | Part des achats faits **avec promotion** (entre 0 et 1)                                    |
| `Campagne_1` … `Campagne_5` | Indique l’exposition à la campagne 1..5 (binaire 0/1 ou intensité)                      |
| `Accept_1` … `Accept_5`   | **Réponse/acceptation** à la campagne 1..5 (0/1)                                           |
| `Canal_pref`              | Canal préféré (web / store / app / email / etc.)                                           |
| `Panier_moyen`            | Panier moyen par achat (peut être calculé)                                                 |
| `Satisfaction`            | Score (enquête) – optionnel                                                                |

**Qualité & RGPD**  
- Vérifier doublons, valeurs manquantes, types, outliers.  
- Aucune donnée **PII sensible** (emails bruts, n° tél, adresses exactes) dans les livrables.  
- Tracer toute **anonymisation** réalisée.

---

## 4) KPIs – définitions & formules (sans groupe test/contrôle)

> Ces KPIs sont calculés **par client**, **par segment**, **par campagne**, puis **agrégés** (moyenne, médiane, pondération par CA). Adaptez au périmètre désiré.

### 4.1 Comportement d’achat (RFM étendu)
- **Fréquence** : `Freq` (nombre d’achats sur la période).  
- **Monetary / CA** : `CA` (somme des dépenses).  
- **Récence** : `Recency` (jours depuis dernier achat, plus petit = plus récent).
- **Panier moyen** : `Panier_moyen = CA / max(Freq, 1)`.
- **Part online** : `Part_online = Nb_achats_en_ligne / max(Freq, 1)`.  
  > **Astuce** : protéger contre la division par 0 (utiliser `max(Freq,1)`).  
- **Promo_ratio moyen** : `Promo_ratio_moy = mean(Promo_ratio)` sur la population considérée.

### 4.2 Campagnes (1..5)
- **Taux d’acceptation campagne _i_** :  
  `Accept_rate_i = (nb d’Accept_i == 1) / (nb de clients exposés à Campagne_i)`
- **CA des acceptants (campagne _i_)** :  
  `CA_acceptants_i = somme(CA des clients avec Accept_i == 1)`  
  (Variante : CA **incrémental** si vous avez un historique pré-campagne.)
- **Contribution des acceptants** :  
  `Part_CA_acceptants_i = CA_acceptants_i / CA_total_population_exposée_i`
- **Coût campagne i** : `Cout_i` (fourni en paramètre ou variable projet).
- **Marge unitaire** : `marge` (ex. 0.30) – **à expliciter** dans le projet.
- **ROI naïf** (sans groupe contrôle) :  
  `ROI_i = ((marge * CA_acceptants_i) - Cout_i) / Cout_i`
  - **Limite** : absence d’uplift véritable (pas de causalité prouvée).
- **Proxy “uplift” sans contrôle** (optionnel, à expliciter) :  
  - **Baseline temporelle** : comparer au **CA avant campagne** sur la même window.  
  - **Contrefactuel par matching** : comparer acceptants vs **non-acceptants comparables** (propensity score / nearest neighbors).

### 4.3 KPIs portfolio / segment
- **Poids segment** : part clients & part CA.  
- **Valeur vie simplifiée (LTV naïve)** :  
  `LTV = (marge * CA / horizon) * horizon = marge * CA` (si horizon = 1 période)  
  - Pour un horizon > 1, projeter la **Fréquence** & **Panier moyen** et actualiser (taux d’actualisation).

> **Script utile** : `kpi_terminal.py` (CLI)  
> Exemple :  
> ```bash
> python kpi_terminal.py --file data/camp_market_clean.csv --cout-campagne 12000 --marge 0.30 --horizon 2
> ```

---

## 5) Méthodologie (workplan)

1. **Data quality & EDA**  
   - Types, NA, doublons, distributions, corrélations (num & cat via Cramér’s V).  
   - Segments évidents : High-Value, Récents, Promo-lovers, Online-first…
2. **Feature engineering**  
   - RFM, `Part_online`, `Panier_moyen`, `Taux_promo_personnel` (`Promo_ratio`), canaux, saisonnalité, interactions (ex. revenu×promo).
   - Encodages : One‑Hot / Target enc. Standardisation si modèles linéaires.
3. **Segmentation (clustering)**  
   - KMeans (ou GMM/Hierarchical). Choix de `k` via silhouette, Davies‑Bouldin.  
   - Profils & **personas marketing** + **reco actionnables** par segment.
4. **Prédiction de réponse**  
   - Cible : `Accept_i` (binaire).  
   - Modèles : **Logistic Regression**, **RandomForest**, **XGBoost**.  
   - Équilibre : stratification, pénalité de classe, SMOTE (avec prudence).  
   - Métriques : **AUC**, **Recall** classe positive, **PR-AUC**, **F1**.  
   - **Interprétabilité** : coefficients / **SHAP** (features clés).
5. **Évaluation économique**  
   - Courbe **gain/cost** par seuil, **profit curve**, **lift** (si possible).  
   - Simulation de ciblage (budget, coût contact, marge unitaire).

---

## 7) Installation & exécution

### 7.1 Prérequis
- Python ≥ 3.10
- pip, venv (ou conda)
- JupyterLab/Notebook

### 7.2 Environnement
```bash
# cloner le dépôt
git clone <votre_repo.git>
cd <votre_repo>

# créer et activer l'environnement
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# installer les dépendances
pip install -r requirements.txt

# lancer Jupyter
jupyter lab
```

### 7.3 Scripts utiles
```bash
# KPIs terminal (sans groupe test/contrôle)
python scripts/kpi_terminal.py --file data/camp_market_clean.csv --cout-campagne 12000 --marge 0.30 --horizon 2

# Pré-traitements
python scripts/preprocess.py --in data/Campagne_Market.csv --out data/camp_market_clean.csv

# Clustering (KMeans par défaut)
python scripts/cluster.py --file data/camp_market_clean.csv --k 5 --out reports/figures/segments.png

# Entraînement classification (Accept_i)
python scripts/train_clf.py --file data/camp_market_clean.csv --target Accept_1 --model xgb --out models/accept_1.pkl

# Générer un court rapport (HTML/MD)
python scripts/report.py --file data/camp_market_clean.csv --out reports/rapport.html
```

---

## 8) Évaluation & métriques

- **Data quality** : couverture NA, cohérence, duplications.  
- **KPIs** : définitions, justesse, interprétations business.  
- **Clustering** : silhouette, stabilité, lisibilité des segments (features clés).  
- **Classification** : AUC / PR‑AUC / Recall+, **calibration** (Brier).  
- **Économie** : ROI (hypothèses de marge/coût explicites), **profit curves**.  
- **Robustesse** : validation croisée, fuite de données évitée, seed fixée.

---

## 9) Limites & risques

- **Pas de groupe contrôle** → causalité non garantie (ROI **naïf**).  
- **Fuite temporelle** si on mélange avant/après dans le training.  
- **Biais & équité** : vérifier que le ciblage ne pénalise pas des sous‑groupes.  
- **RGPD** : anonymiser, minimiser, documenter les finalités marketing.

---

## 10) Plan de soutenance (15’)

1. **Problème & données** (1’30) : contexte, qualité, limites.  
2. **KPIs clés** (3’) : 3–5 messages forts, un graphique par KPI.  
3. **Segments** (4’) : carte des personas + actions proposées.  
4. **Prédiction** (4’) : modèle retenu, métriques, seuil de décision.  
5. **Business case** (1’30) : ROI estimé + plan test‑and‑learn (A/B / uplift).  
**Backup** : hypothèses, ablations, comparatifs de modèles.

**Barème à garder en tête** : livraison fonctionnelle, doc, portabilité, clarté, adaptabilité, insights, actions business, storytelling, prétraitement/harmonisation, confidentialité/éthique, variété/pertinence/lisibilité des viz, modèles ML, créativité, outillage, présentation.

---

## 11) Roadmap 6 jours (suggestion)

- **J1** : cadrage, EDA, data quality, premiers KPIs.  
- **J2** : feature engineering, KPIs campagnes, premières reco.  
- **J3** : clustering & personas, itérations.  
- **J4** : modèles de réponse, sélection & interprétation.  
- **J5** : simulation ROI/targeting, story & slides.  
- **J6** : répétitions, polishing, livrables finaux.

---

## 12) Contribution (workflow équipe de 5)

- Branches : `main` (stable), `develop`, `feature/<topic>`.  
- PR avec **review** croisée, **lint** (black, flake8), **tests** unitaires min.  
- Commit messages clairs (Conventional Commits).

---

## 13) Licence

MIT (à adapter selon les contraintes de l’entreprise/école).

---

## 14) Contacts

- **PO / PM** : …  
- **Référent data** : …  
- **Référent modèle** : …  
- **Référent BI / viz** : …  
- **Référent soutenance** : …

---

### Annexes – Formules rapides

- `Part_online = Nb_achats_en_ligne / max(Freq, 1)`  
- `Promo_ratio_moy = mean(Promo_ratio)`  
- `Panier_moyen = CA / max(Freq, 1)`  
- `Accept_rate_i = nb(Accept_i == 1) / nb(exposés Campagne_i)`  

> **Conseil** : Affichez toujours **l’hypothèse** derrière chaque KPI (périmètre, fenêtre temporelle, base clients) pour éviter les mauvaises interprétations.

````markdown
# Bootcamp-Data-Rush-4 — Marketing Analytics

Ce projet propose un mini-pipeline d’analyse marketing : **feature engineering**, **clustering clients** (coude, silhouette, PCA), et **tableau de bord KPI** au rendu terminal soigné.

## Sommaire
- [Prérequis](#prérequis)
- [Installation & environnement](#installation--environnement)
- [Arborescence](#arborescence)
- [Je veux juste tout lancer (pipeline)](#je-veux-juste-tout-lancer-pipeline)
- [1) Feature engineering](#1-feature-engineering)
- [2) Clustering clients](#2-clustering-clients)
- [3) KPI terminal](#3-kpi-terminal)
- [Makefile (raccourcis)](#makefile-raccourcis)
- [Schéma des données (FR ↔ EN)](#schéma-des-données-fr--en)
- [Sorties & visualisations](#sorties--visualisations)
- [Bonnes pratiques](#bonnes-pratiques)
- [Dépannage (FAQ)](#dépannage-faq)
- [Licence](#licence)

---

## Prérequis
- **Python 3.10+** (testé en 3.11/3.13)
- macOS / Linux / Windows (PowerShell/WSL ok)
- Terminal avec support **Unicode** recommandé (pour les cadres jolis dans le script KPI)

## Installation & environnement

```bash
# 1) Cloner le projet
git clone <votre-repo> Bootcamp-Data-Rush-4
cd Bootcamp-Data-Rush-4

# 2) Créer et activer un venv
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 3) Mettre à jour pip + installer dépendances
python -m pip install --upgrade pip
pip install -r requirements.txt
````

> Astuce : sur Apple Silicon, utilisez un Python natif arm64 (Homebrew Python 3.11+).

---

## Arborescence

```
Bootcamp-Data-Rush-4/
├─ data/
│  └─ camp_market_clean.csv
├─ processed/
│  └─ features_enriched.csv
├─ outputs/
│  ├─ 01_coude_inertie.png
│  ├─ 02_silhouette.png
│  ├─ 03_projection_pca.png
│  └─ 04_resume_clusters.csv
├─ scripts/
│  ├─ features_engineering.py
│  ├─ clustering_clients.py
│  ├─ KPIs.py
│  └─ kpi_terminal.py
├─ Makefile
├─ requirements.txt
└─ README.md
```

---

## Je veux juste tout lancer (pipeline)

### Option A — Makefile (recommandé)

```bash
# Enchaîne features → clustering
make all \
  INPUT=data/camp_market_clean.csv \
  OUTPUT=processed/features_enriched.csv \
  OVERWRITE=1 \
  KMIN=2 KMAX=8 OUT=outputs RS=42
```

### Option B — Scripts (manuel)

```bash
# 1) Feature engineering
python scripts/features_engineering.py \
  -i data/camp_market_clean.csv \
  -o processed/features_enriched.csv \
  --overwrite

# 2) Clustering
python scripts/clustering_clients.py \
  -f processed/features_enriched.csv \
  --k-min 2 --k-max 8 -o outputs --random-state 42
```

---

## 1) Feature engineering

**Script :** `scripts/features_engineering.py`
**Entrée :** `data/camp_market_clean.csv`
**Sortie :** `processed/features_enriched.csv`

### Commande minimale

```bash
python scripts/features_engineering.py -i data/camp_market_clean.csv -o processed/features_enriched.csv --overwrite
```

### Options

| Option           |                        Par défaut | Description                        |
| ---------------- | --------------------------------: | ---------------------------------- |
| `-i`, `--input`  |      `data/camp_market_clean.csv` | Fichier d’entrée (CSV nettoyé).    |
| `-o`, `--output` | `processed/features_enriched.csv` | Fichier de sortie (CSV enrichi).   |
| `--sep`          |                               `,` | Séparateur CSV.                    |
| `--encoding`     |                           `utf-8` | Encodage CSV.                      |
| `--ref-year`     |                            `2014` | Année de référence pour l’âge.     |
| `--overwrite`    |                           `False` | Écraser si le fichier existe déjà. |

### Variables créées

* `Depense_totale = MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts + MntGoldProds`
* `Depense_plaisir = MntWines + MntSweetProducts + MntGoldProds`
* `Part_plaisir = Depense_plaisir / Depense_totale`
* `Taux_depense_sur_revenu = Depense_totale / Income`
* `Total_achats = NumWebPurchases + NumCatalogPurchases + NumStorePurchases`
* `Part_achats_en_ligne = NumWebPurchases / Total_achats`
* `Part_achats_catalogue = NumCatalogPurchases / Total_achats`
* `Taux_visite_achat_web = NumWebPurchases / NumWebVisitsMonth`
* `Nb_enfants_total = Kidhome + Teenhome`
* `Age = 2014 - Year_Birth` (ou `--ref-year`)
* `Score_engagement_marketing = AcceptedCmp1 + ... + AcceptedCmp5`
* `Taux_acceptation_campagne = Score_engagement_marketing / 5`

> Le script gère automatiquement les versions FR ↔ EN des noms de colonnes (voir tableau plus bas).

---

## 2) Clustering clients

**Script :** `scripts/clustering_clients.py`
**Entrée :** `processed/features_enriched.csv`
**Sorties :** diagnostics coude/silhouette, projection PCA, résumé CSV, dataset labellisé

### Commande minimale

```bash
python scripts/clustering_clients.py \
  -f processed/features_enriched.csv \
  --k-min 2 --k-max 8 -o outputs --random-state 42
```

### Options

| Option               |                        Par défaut | Description                                      |
| -------------------- | --------------------------------: | ------------------------------------------------ |
| `-f`, `--file`       | `processed/features_enriched.csv` | CSV en entrée (features).                        |
| `--features <liste>` |                (liste prédéfinie) | Colonnes utilisées pour KMeans.                  |
| `--k-min`            |                               `2` | Nombre minimal de clusters.                      |
| `--k-max`            |                               `8` | Nombre maximal de clusters.                      |
| `-o`, `--out-dir`    |                         `outputs` | Dossier de sortie (PNG/CSV).                     |
| `--random-state`     |                              `42` | Graine aléatoire.                                |
| `--no-scale`         |                           `False` | Désactive la standardisation (`StandardScaler`). |

### Comportement

* Calcule **inertie (coude)** et **silhouette** pour k ∈ [`k-min`, `k-max`].
* Choisit **k** par défaut via le **meilleur score silhouette** (sinon `k-min`).
* Réalise une **PCA 2D** pour projeter les points et sauvegarder la projection.
* Génère un **résumé par cluster** : dépenses moyennes, part plaisir, enfants, achats, part web, âge, score marketing + **ligne lisible** avec émojis dans le terminal.

**Interprétation silhouette**

* `> 0.50` : séparation **très bonne**
* `0.25–0.50` : **acceptable**
* `< 0.25` : **faible** (clusters possiblement artificiels)

---

## 3) KPI terminal

**Script :** `scripts/kpi_terminal.py`
Affiche un **tableau de bord** dans le terminal.

### Commande type

```bash
python scripts/kpi_terminal.py \
  --file data/camp_market_clean.csv \
  --cout-campagne 12000 \
  --marge 0.30 \
  --horizon 2 \
  --width 100
```

### Options

| Option            |    Par défaut | Description                     |
| ----------------- | ------------: | ------------------------------- |
| `--file`          | (obligatoire) | CSV nettoyé (colonnes FR).      |
| `--cout-campagne` |         `0.0` | Coût total de la campagne.      |
| `--marge`         |        `0.30` | Marge pour CLV lite.            |
| `--horizon`       |         `2.0` | Horizon (années) pour CLV lite. |
| `--width`         |          `96` | Largeur d’affichage terminal.   |

**Notes :**

* Si la colonne `Groupe` (test/contrôle) **n’existe pas**, elle est **créée automatiquement** (split stable 50/50 basé sur `Identifiant`) et un avertissement est affiché.
* KPI : effectifs TRT/CTL, taux de réponse, uplift, revenu incrémental, CPA & ROI (si coût > 0), top CLV, clients à risque, synthèse.

---

## Makefile (raccourcis)

```make
# Lancer uniquement les features
make features INPUT=data/camp_market_clean.csv OUTPUT=processed/features_enriched.csv OVERWRITE=1

# Lancer uniquement le clustering
make cluster FILE=processed/features_enriched.csv KMIN=2 KMAX=8 OUT=outputs RS=42

# Tout enchaîner
make all INPUT=data/camp_market_clean.csv OUTPUT=processed/features_enriched.csv OVERWRITE=1 KMIN=2 KMAX=8 OUT=outputs RS=42
```

Variables disponibles : `INPUT, OUTPUT, SEP, ENC, REFYEAR, OVERWRITE, FILE, KMIN, KMAX, OUT, RS, NOSCALE`

---

## Schéma des données (FR ↔ EN)

Le projet reconnaît automatiquement les colonnes en **FR** ou en **EN** (MarTech UCI). Principales correspondances :

| EN                    | FR                      |
| --------------------- | ----------------------- |
| `MntWines`            | `Montant_vin`           |
| `MntFruits`           | `Montant_fruits`        |
| `MntMeatProducts`     | `Montant_viande`        |
| `MntFishProducts`     | `Montant_poisson`       |
| `MntSweetProducts`    | `Montant_sucreries`     |
| `MntGoldProds`        | `Montant_luxe`          |
| `Income`              | `Revenu`                |
| `NumWebPurchases`     | `Nb_achats_en_ligne`    |
| `NumCatalogPurchases` | `Achats_catalogue`      |
| `NumStorePurchases`   | `Achats_magasin`        |
| `NumWebVisitsMonth`   | `Nb_visites_web_mois`   |
| `Kidhome`             | `Enfant_charge`         |
| `Teenhome`            | `Ado_charge`            |
| `Year_Birth`          | `Année_naissance`       |
| `AcceptedCmp1..5`     | `Accepte_Campagne_1..5` |

---

## Sorties & visualisations

Dans `outputs/` :

* `01_coude_inertie.png` — courbe inertie vs k
* `02_silhouette.png` — score silhouette moyen vs k
* `03_projection_pca.png` — projection PCA 2D colorée par cluster
* `04_resume_clusters.csv` — résumé par cluster (moyennes & indicateurs)

Dans `processed/` :

* `features_enriched.csv` — CSV enrichi (toutes features)
* `features_enriched_labeled.csv` — CSV enrichi + colonne `cluster` (si généré)

---

## Bonnes pratiques

* **Nettoyez** les valeurs aberrantes (revenu/achats extrêmes) avant clustering.
* **Standardisez** les features pour KMeans (activé par défaut).
* Inspectez **silhouette** : ne pas sur-interpréter un k avec score faible.
* Ajustez la **liste de features** (`--features`) selon la stratégie marketing.

---

## Dépannage (FAQ)

**No such file or directory …**
→ Vérifiez les chemins (`-i`, `-f`, `-o`) et l’existence des dossiers. Les scripts créent `processed/` et `outputs/` s’ils manquent.

**Colonnes manquantes**
→ Renommez vos colonnes ou utilisez le mapping FR/EN. Si vos noms diffèrent, adaptez le mapping dans `features_engineering.py`.

**Unicode moche**
→ Votre terminal n’est pas en UTF-8. Le script dégrade en ASCII automatiquement. Activez UTF-8 si possible (iTerm/Windows Terminal).

**ROI / CPA à « - »**
→ Fournissez `--cout-campagne` > 0 pour calculer CPA/ROI.

---

## Licence

Usage pédagogique / interne. Adaptez selon votre contexte.

---

💡 **Aide**

```bash
python scripts/features_engineering.py --help
python scripts/clustering_clients.py   --help
python scripts/kpi_terminal.py         --help
```

Bonnes analyses ! 🚀
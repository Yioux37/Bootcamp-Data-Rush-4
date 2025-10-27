#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
features_engineering.py — création d'indicateurs utiles à partir du dataset marketing.

Exemple d'usage :
  python features_engineering.py --input data/camp_market_clean.csv --output processed/features_enriched.csv --ref-year 2014
"""

import argparse
import math
import sys
import numpy as np
import pandas as pd

# utils

def ensure_numeric(df: pd.DataFrame, cols):
    """Force cols en float (gère virgules décimales, espaces, symboles)."""
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].astype(str)
        s = s.str.replace(",", ".", regex=False)
        s = s.str.replace(r"[^0-9\.\-eE+]", "", regex=True)
        df[c] = pd.to_numeric(s, errors="coerce")
    return df

def safe_div(num, den):
    if den is None or (isinstance(den, (int, float, np.floating)) and (den == 0 or np.isnan(den))):
        return np.nan
    try:
        return num / den
    except Exception:
        return np.nan

# mapping colonnes
# Le script accepte deux schémas :
# - EN (Kaggle/Portugal) : MntWines, MntFruits, ..., NumWebPurchases, Income, etc.
# - FR (ton CSV) : Montant_vin, Montant_fruits, ..., Nb_achats_en_ligne, Revenu, etc.
COLMAPS = [
    # Schéma EN
    {
        "vin": "MntWines",
        "fruits": "MntFruits",
        "viande": "MntMeatProducts",
        "poisson": "MntFishProducts",
        "sucreries": "MntSweetProducts",
        "luxe": "MntGoldProds",
        "achats_web": "NumWebPurchases",
        "achats_catalogue": "NumCatalogPurchases",
        "achats_magasin": "NumStorePurchases",
        "visites_web": "NumWebVisitsMonth",
        "enfant": "Kidhome",
        "ado": "Teenhome",
        "annee_naissance": "Year_Birth",
        "revenu": "Income",
        "cmp1": "AcceptedCmp1",
        "cmp2": "AcceptedCmp2",
        "cmp3": "AcceptedCmp3",
        "cmp4": "AcceptedCmp4",
        "cmp5": "AcceptedCmp5",
    },
    # Schéma FR
    {
        "vin": "Montant_vin",
        "fruits": "Montant_fruits",
        "viande": "Montant_viande",
        "poisson": "Montant_poisson",
        "sucreries": "Montant_sucreries",
        "luxe": "Montant_luxe",
        "achats_web": "Nb_achats_en_ligne",
        "achats_catalogue": "Achats_catalogue",
        "achats_magasin": "Achats_magasin",
        "visites_web": "Nb_visites_web_mois",
        "enfant": "Enfant_charge",
        "ado": "Ado_charge",
        "annee_naissance": "Année_naissance",
        "revenu": "Revenu",
        "cmp1": "Accepte_Campagne_1",
        "cmp2": "Accepte_Campagne_2",
        "cmp3": "Accepte_Campagne_3",
        "cmp4": "Accepte_Campagne_4",
        "cmp5": "Accepte_Campagne_5",
    },
]

def pick_schema(df: pd.DataFrame):
    """Choisit automatiquement le mapping le plus compatible avec le DF."""
    best = None
    best_score = -1
    for m in COLMAPS:
        keys = list(m.values())
        score = sum(1 for k in keys if k in df.columns)
        if score > best_score:
            best = m
            best_score = score
    return best

# core

def engineer_features(df: pd.DataFrame, ref_year: int = 2014) -> pd.DataFrame:
    m = pick_schema(df)
    if m is None:
        raise ValueError("Impossible de détecter le schéma de colonnes (FR/EN).")

    # colonnes nécessaires minimum
    required = [
        m["vin"], m["fruits"], m["viande"], m["poisson"], m["sucreries"], m["luxe"],
        m["achats_web"], m["achats_catalogue"], m["achats_magasin"],
        m["visites_web"], m["enfant"], m["ado"], m["annee_naissance"], m["revenu"],
        m["cmp1"], m["cmp2"], m["cmp3"], m["cmp4"], m["cmp5"]
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    # force numériques
    num_cols = [
        m["vin"], m["fruits"], m["viande"], m["poisson"], m["sucreries"], m["luxe"],
        m["achats_web"], m["achats_catalogue"], m["achats_magasin"], m["visites_web"],
        m["enfant"], m["ado"], m["annee_naissance"], m["revenu"],
        m["cmp1"], m["cmp2"], m["cmp3"], m["cmp4"], m["cmp5"],
    ]
    df = ensure_numeric(df, num_cols)

    # 1) Dépense_totale
    df["Depense_totale"] = (
        df[m["vin"]]
        + df[m["fruits"]]
        + df[m["viande"]]
        + df[m["poisson"]]
        + df[m["sucreries"]]
        + df[m["luxe"]]
    )

    # 2) Depense_plaisir
    df["Depense_plaisir"] = df[m["vin"]] + df[m["sucreries"]] + df[m["luxe"]]

    # 3) Part_plaisir
    df["Part_plaisir"] = df["Depense_plaisir"] / df["Depense_totale"].replace(0, np.nan)

    # 4) Taux_depense_sur_revenu
    df["Taux_depense_sur_revenu"] = df["Depense_totale"] / df[m["revenu"]].replace(0, np.nan)

    # 5) Total_achats
    df["Total_achats"] = df[m["achats_web"]] + df[m["achats_catalogue"]] + df[m["achats_magasin"]]

    # 6) Part_achats_en_ligne
    df["Part_achats_en_ligne"] = df[m["achats_web"]] / df["Total_achats"].replace(0, np.nan)

    # 7) Part_achats_catalogue
    df["Part_achats_catalogue"] = df[m["achats_catalogue"]] / df["Total_achats"].replace(0, np.nan)

    # 8) Taux_visite_achat_web
    df["Taux_visite_achat_web"] = df[m["achats_web"]] / df[m["visites_web"]].replace(0, np.nan)

    # 9) Nb_enfants_total
    df["Nb_enfants_total"] = df[m["enfant"]] + df[m["ado"]]

    # 10) Age
    df["Age"] = ref_year - df[m["annee_naissance"]]

    # 11) Score_engagement_marketing
    df["Score_engagement_marketing"] = (
        df[m["cmp1"]] + df[m["cmp2"]] + df[m["cmp3"]] + df[m["cmp4"]] + df[m["cmp5"]]
    )

    # 12) Taux_acceptation_campagne
    df["Taux_acceptation_campagne"] = df["Score_engagement_marketing"] / 5.0

    return df

# CLI

def main():
    p = argparse.ArgumentParser(description="Création d'indicateurs (feature engineering)")
    p.add_argument("--input", required=True, help="Chemin du CSV d'entrée")
    p.add_argument("--output", required=True, help="Chemin du CSV enrichi à écrire")
    p.add_argument("--ref-year", type=int, default=2014, help="Année de référence pour le calcul de l'âge (défaut 2014)")
    args = p.parse_args()

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"[ERREUR] Lecture CSV: {e}")
        sys.exit(1)

    try:
        df_out = engineer_features(df, ref_year=args.ref_year)
    except Exception as e:
        print(f"[ERREUR] Calcul des features: {e}")
        sys.exit(2)

    try:
        df_out.to_csv(args.output, index=False)
    except Exception as e:
        print(f"[ERREUR] Écriture CSV: {e}")
        sys.exit(3)

    # Rendu terminal concis
    n_rows, n_cols = df_out.shape
    created_cols = [
        "Depense_totale","Depense_plaisir","Part_plaisir","Taux_depense_sur_revenu",
        "Total_achats","Part_achats_en_ligne","Part_achats_catalogue","Taux_visite_achat_web",
        "Nb_enfants_total","Age","Score_engagement_marketing","Taux_acceptation_campagne"
    ]

    print("\n" + "="*72)
    print(" FEATURES CRÉÉES ".center(72, "="))
    print("="*72)
    for c in created_cols:
        miss = df_out[c].isna().mean()
        print(f" - {c:28s} | NaN: {miss:6.2%} | exemple: {df_out[c].dropna().head(1).to_list()[0] if df_out[c].notna().any() else '—'}")
    print("-"*72)
    print(f"Lignes: {n_rows:,}  |  Colonnes: {n_cols:,}".replace(",", " "))
    print(f"Fichier écrit -> {args.output}")
    print("="*72 + "\n")

if __name__ == "__main__":
    main()

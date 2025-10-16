#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from datetime import datetime
import numpy as np
import pandas as pd


# =========
# Aliases : harmonisation des noms réels -> noms attendus par le script
# =========
# On renomme le DataFrame pour que le reste du code utilise des noms "propres".
COLUMN_ALIASES = {
    # Achats / canaux
    "Nb_achats_en_ligne": "Achats_en_ligne",
    # (Ajouter d'autres alias si nécessaire)
    # "Visites_web_mois": "Nb_visites_web_mois",
    # "Date_client": "Date_acquisition_client",
}

# Colonnes des montants produits (utilisées pour Monetary et nb produits)
PRODUCT_AMOUNT_COLS = [
    "Montant_vin",
    "Montant_fruits",
    "Montant_viande",
    "Montant_poisson",
    "Montant_sucreries",
    "Montant_luxe",
]

# Colonnes d’achats par canal
PURCHASE_CHANNEL_COLS = [
    "Achats_en_ligne",
    "Achats_catalogue",
    "Achats_magasin",
]

# Quelques colonnes attendues pour des contrôles simples
SOFT_REQUIRED_COLS = [
    "Identifiant",
    "Année_naissance",
    "Education",
    "Situation_matrimoniale",
    "Revenu",
    "Enfant_charge",
    "Ado_charge",
    "Date_acquisition_client",
    "Nombre_jours_depuis_dernier_achat",
    "Nb_visites_web_mois",
    "Response",
]


def apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes du DF d’après COLUMN_ALIASES si elles existent."""
    rename_map = {src: dst for src, dst in COLUMN_ALIASES.items() if src in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def check_columns(df: pd.DataFrame) -> None:
    """Avertit (sans casser) si des colonnes utiles manquent."""
    missing_soft = [c for c in SOFT_REQUIRED_COLS if c not in df.columns]
    if missing_soft:
        print(
            f"[AVERTISSEMENT] Colonnes courantes manquantes (le script continue): {missing_soft}"
        )

    missing_purchase = [c for c in PURCHASE_CHANNEL_COLS if c not in df.columns]
    if missing_purchase:
        raise ValueError(
            "Colonnes d’achats par canal manquantes : "
            f"{missing_purchase}. Corrige les en-têtes ou mets un alias."
        )

    missing_products = [c for c in PRODUCT_AMOUNT_COLS if c not in df.columns]
    if missing_products:
        raise ValueError(
            "Colonnes de montants produits manquantes : "
            f"{missing_products}. Corrige les en-têtes ou mets un alias."
        )


def coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse les dates et remplace les dates placeholder 1970-01-01 par NaT."""
    if "Date_acquisition_client" in df.columns:
        df["Date_acquisition_client"] = pd.to_datetime(
            df["Date_acquisition_client"], errors="coerce"
        )
        # Beaucoup de jeux posent 1970-01-01 comme valeur sentinelle -> on la traite comme manquante
        mask_1970 = df["Date_acquisition_client"] == pd.Timestamp("1970-01-01")
        df.loc[mask_1970, "Date_acquisition_client"] = pd.NaT
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée les features RFM + dérivées (tenure, nb produits, etc.)."""

    # Total achats (fréquence)
    df["Nb_achats_total"] = df[PURCHASE_CHANNEL_COLS].sum(axis=1)

    # Monetary : somme des dépenses catégories
    df["Depense_totale"] = df[PRODUCT_AMOUNT_COLS].sum(axis=1)

    # Recency (jours) : on le prend tel quel si présent
    if "Nombre_jours_depuis_dernier_achat" in df.columns:
        df["Recence_jours"] = pd.to_numeric(
            df["Nombre_jours_depuis_dernier_achat"], errors="coerce"
        )
    else:
        df["Recence_jours"] = np.nan

    # Tenure (ancienneté en jours) depuis l’acquisition jusqu’à la date de référence
    # On prend "aujourd’hui" comme référence; tu peux figer une date si besoin.
    today = pd.Timestamp(datetime.utcnow().date())
    if "Date_acquisition_client" in df.columns:
        df["Tenure_jours"] = (today - df["Date_acquisition_client"]).dt.days
    else:
        df["Tenure_jours"] = np.nan

    # Nombre de catégories achetées (>0)
    df["Nb_categories_achetees"] = (df[PRODUCT_AMOUNT_COLS] > 0).sum(axis=1)

    # RFM quantiles (scores 1..5, 5 = meilleur)
    def qcut_score(s, q=5, ascending=True):
        # robustesse : si série constante -> score médian
        if s.nunique(dropna=True) <= 1:
            return pd.Series(3, index=s.index)
        labels = list(range(1, q + 1))
        if ascending:
            return pd.qcut(s.rank(method="first"), q=q, labels=labels).astype(int)
        else:
            # inverser : pour Recency, plus petit = meilleur
            return pd.qcut(s.rank(method="first"), q=q, labels=labels[::-1]).astype(int)

    # R (petite recence = mieux)
    df["R_score"] = qcut_score(df["Recence_jours"], q=5, ascending=False)

    # F (plus de fréquences = mieux)
    df["F_score"] = qcut_score(df["Nb_achats_total"], q=5, ascending=True)

    # M (plus de dépenses = mieux)
    df["M_score"] = qcut_score(df["Depense_totale"], q=5, ascending=True)

    df["RFM_score"] = df["R_score"] * 100 + df["F_score"] * 10 + df["M_score"]

    return df


def safe_rate(numer, denom):
    numer = float(numer)
    denom = float(denom)
    return numer / denom if denom != 0 else 0.0


def compute_kpis(df: pd.DataFrame, control_col: str | None, segments: list[str]) -> pd.DataFrame:
    """
    Calcule des KPI de base globalement, par segments, et (si présent) par groupe test/contrôle.
    """
    rows = []

    def push_row(scope: str, scope_value: str | None, data: pd.DataFrame):
        n = len(data)
        response_rate = data["Response"].mean() if "Response" in data.columns else np.nan
        avg_spend = data["Depense_totale"].mean()
        freq = data["Nb_achats_total"].mean()
        visits = data["Nb_visites_web_mois"].mean() if "Nb_visites_web_mois" in data.columns else np.nan
        rows.append(
            {
                "Niveau": scope,
                "Valeur": scope_value if scope_value is not None else "Global",
                "Nb_clients": n,
                "Taux_reponse": response_rate,
                "Depense_moy": avg_spend,
                "Freq_achats_moy": freq,
                "Visites_web_moy": visits,
                "R_score_moy": data["R_score"].mean(),
                "F_score_moy": data["F_score"].mean(),
                "M_score_moy": data["M_score"].mean(),
                "RFM_score_moy": data["RFM_score"].mean(),
            }
        )

    # Global
    push_row("Global", None, df)

    # Par segments univariés
    for seg in segments:
        if seg not in df.columns:
            print(f"[AVERTISSEMENT] Segment '{seg}' introuvable, ignoré.")
            continue
        for val, grp in df.groupby(seg):
            push_row(seg, str(val), grp)

    # Test / Contrôle (si demandé et présent)
    if control_col:
        if control_col not in df.columns:
            print(f"[AVERTISSEMENT] Colonne de contrôle '{control_col}' absente, section test/contrôle ignorée.")
        else:
            # KPI par groupe
            for val, grp in df.groupby(control_col):
                push_row(f"{control_col}", str(val), grp)

            # Si binaire (2 groupes), on donne un lift simple sur le taux de réponse
            if df[control_col].nunique(dropna=True) == 2 and "Response" in df.columns:
                piv = df.groupby(control_col)["Response"].mean().rename("taux")
                if len(piv) == 2:
                    a, b = piv.index.tolist()
                    lift = (piv[a] - piv[b]) / piv[b] if piv[b] != 0 else np.nan
                    rows.append(
                        {
                            "Niveau": f"Lift_{control_col}",
                            "Valeur": f"{a} vs {b}",
                            "Nb_clients": np.nan,
                            "Taux_reponse": lift,
                            "Depense_moy": np.nan,
                            "Freq_achats_moy": np.nan,
                            "Visites_web_moy": np.nan,
                            "R_score_moy": np.nan,
                            "F_score_moy": np.nan,
                            "M_score_moy": np.nan,
                            "RFM_score_moy": np.nan,
                        }
                    )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Campagne marketing • Clustering & KPI"
    )
    parser.add_argument("--input", dest="input_path", required=True, help="Chemin vers le CSV d’entrée (données clients).")
    parser.add_argument("--output_dir", default="processed", help="Dossier de sortie (défaut: processed).")
    parser.add_argument("--campaign-cost", type=float, default=0.0, help="Coût total de la campagne (optionnel).")
    parser.add_argument("--control-col", default=None, help="Nom de la colonne de groupe (ex: Groupe Test/Contrôle).")
    parser.add_argument("--segments", default="", help="Liste de colonnes segments séparées par des virgules.")
    args = parser.parse_args()

    input_path = args.input_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("\n— Lecture du fichier:", input_path)
    df = pd.read_csv(input_path)
    print(f"— Dimensions: {df.shape[0]:,} lignes x {df.shape[1]} colonnes")

    # Harmonisation noms
    df = apply_aliases(df)

    # Vérifs colonnes
    check_columns(df)

    # Dates / tenure
    df = coerce_dates(df)

    print("— Génération des features (RFM, nb produits, canaux, tenure)")
    df = add_features(df)

    # Parse segments
    segments = [c.strip() for c in args.segments.split(",") if c.strip()]

    print("— Calcul des KPI")
    kpis = compute_kpis(df, control_col=args.control_col, segments=segments)

    # Ajout rapide d'info coût campagne (global) si fourni
    if args.campaign_cost and args.campaign_cost > 0 and "Response" in df.columns:
        n_targeted = len(df)
        responses = df["Response"].sum()
        cost_per_target = args.campaign_cost / n_targeted if n_targeted else np.nan
        cost_per_response = args.campaign_cost / responses if responses else np.nan
        print(
            f"— Coût campagne: {args.campaign_cost:,.2f} | "
            f"Cout/contact: {cost_per_target:,.2f} | "
            f"Cout/réponse: {cost_per_response if not np.isnan(cost_per_response) else 'NA'}"
        )
        # On append une ligne d’info budget en bas du tableau KPI
        kpis = pd.concat(
            [
                kpis,
                pd.DataFrame(
                    [
                        {
                            "Niveau": "Budget",
                            "Valeur": "Campagne",
                            "Nb_clients": n_targeted,
                            "Taux_reponse": df["Response"].mean(),
                            "Depense_moy": np.nan,
                            "Freq_achats_moy": np.nan,
                            "Visites_web_moy": np.nan,
                            "R_score_moy": np.nan,
                            "F_score_moy": np.nan,
                            "M_score_moy": np.nan,
                            "RFM_score_moy": np.nan,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    # Sauvegardes
    features_path = os.path.join(output_dir, "features.csv")
    kpis_path = os.path.join(output_dir, "kpis.csv")

    df.to_csv(features_path, index=False)
    kpis.to_csv(kpis_path, index=False)

    print(f"\n✓ Features écrites dans: {features_path}")
    print(f"✓ KPI écrits dans: {kpis_path}\n")
    print("Terminé.")


if __name__ == "__main__":
    main()
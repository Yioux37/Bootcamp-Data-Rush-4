#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KPIs.py — Calcul de KPI marketing sur le fichier camp_market_clean.csv

Sorties (dans --output, défaut: data/processed) :
- kpis_summary.csv        : KPI globaux (campagne entière)
- kpis_by_segment.csv     : KPI par segments (si --segments fourni)
- kpis_optional.csv       : KPI complémentaires globaux
- kpis_report.xlsx        : Excel avec toutes les feuilles ci-dessus

Options utiles :
--input <fichier.csv>          : chemin du CSV (défaut: data/raw/camp_market_clean.csv)
--output <dossier>             : dossier de sortie (défaut: data/processed)
--campaign-cost <float>        : coût total de la campagne (pour CAC/ROI)
--margin <float>               : marge unitaire (0.3 = 30%), pour CLV (défaut 0.3)
--clv-years <int>              : horizon CLV en années (défaut 3)
--control-col <str>            : nom de la colonne Test/Control (pour uplift)
--segments <col1 col2 ...>     : colonnes de segmentation (ex: Education Situation_matrimoniale)

Hypothèses raisonnables (adaptables si besoin) :
- Recette (CA) client = somme des colonnes Montant_*.
- Nb commandes client = Achats_catalogue + Achats_magasin + Nb_achats_en_ligne.
- Acheteur = (CA>0) ou (Nb commandes>0).
- Conversion = acheteurs / exposés (lignes du fichier).
- Taux de réponse = moyenne de la colonne 'Response' si présente (sinon NaN).
- AOV (Panier moyen) = CA total / nb commandes (si nb commandes > 0).
- CLV ≈ AOV * fréquence annuelle * marge * horizon (fréquence annualisée depuis Date_acquisition_client).
- Uplift : si --control-col est fourni et contient 2 groupes (ex: "Test", "Control"),
  on calcule l’écart Test−Control sur Taux de réponse et Conversion.

Le script est tolérant aux colonnes manquantes (remplace par NaN quand nécessaire).
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Constantes : noms de colonnes attendues (tel que dans ton CSV)
AMOUNT_COLS = [
    "Montant_vin",
    "Montant_fruits",
    "Montant_viande",
    "Montant_poisson",
    "Montant_sucreries",
    "Montant_luxe",
]

ORDER_PARTS = [
    "Achats_catalogue",
    "Achats_magasin",
    "Nb_achats_en_ligne",
]

OPTIONAL_COLS = {
    "response": "Response",
    "recency_days": "Nombre_jours_depuis_dernier_achat",
    "web_visits": "Nb_visites_web_mois",
    "promo_orders": "Nb_achats_promo",
    "acq_date": "Date_acquisition_client",
    # colonnes d’acceptation de campagnes (si présentes dans ton fichier)
    "acc3": "Accepte_Campagne_3",
    "acc4": "Accepte_Campagne_4",
    "acc5": "Accepte_Campagne_5",
    "acc1": "Accepte_Campagne_1",
    "acc2": "Accepte_Campagne_2",
}


# Fonctions utilitaires
def safe_sum(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Somme des colonnes présentes uniquement (ignore celles manquantes)."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index)
    return df[present].sum(axis=1, skipna=True)


def safe_fillna(series: pd.Series, val: float = 0.0) -> pd.Series:
    return series.fillna(val) if series is not None else series


def parse_date_safe(s: pd.Series) -> pd.Series:
    """Parse de dates tolérant (YYYY-mm-dd), retourne NaT quand impossible."""
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([np.nan] * len(s)), errors="coerce")


def annualize_frequency(orders: pd.Series, acq_date: pd.Series) -> pd.Series:
    """
    Approx. fréquence annuelle :
    freq = nb_commandes / max(âge_client_en_années, 1/12)
    """
    now = pd.Timestamp(datetime.utcnow().date())
    age_years = ((now - acq_date).dt.days / 365.25).clip(lower=1 / 12)  # évite division par 0
    return (orders / age_years).replace([np.inf, -np.inf], np.nan)


# Calculs de KPI globaux
def compute_global_kpis(
    df: pd.DataFrame,
    campaign_cost: float = 0.0,
    margin: float = 0.3,
    clv_years: int = 3,
) -> Dict[str, float]:
    n = len(df)  # exposés

    # CA et nb commandes
    revenue = safe_sum(df, AMOUNT_COLS)
    orders = safe_sum(df, ORDER_PARTS)

    # acheteur si CA>0 ou commandes>0
    buyers_mask = (safe_fillna(revenue, 0) > 0) | (safe_fillna(orders, 0) > 0)
    buyers = buyers_mask.sum()

    # conversion
    conv = buyers / n if n > 0 else np.nan

    # taux de réponse (si colonne response)
    if OPTIONAL_COLS["response"] in df.columns:
        resp_rate = df[OPTIONAL_COLS["response"]].mean()
    else:
        resp_rate = np.nan

    total_revenue = safe_fillna(revenue, 0).sum()
    total_orders = int(safe_fillna(orders, 0).sum())

    aov = total_revenue / total_orders if total_orders > 0 else np.nan
    cac = campaign_cost / buyers if buyers > 0 else np.nan
    roi = (total_revenue - campaign_cost) / campaign_cost if campaign_cost > 0 else np.nan

    # CLV proxy : AOV * fréquence annuelle * marge * horizon
    if OPTIONAL_COLS["acq_date"] in df.columns:
        acq = parse_date_safe(df[OPTIONAL_COLS["acq_date"]])
        freq_annual = annualize_frequency(safe_fillna(orders, 0), acq)
        aov_per_client = (safe_fillna(revenue, 0) / safe_fillna(orders, 0)).replace([np.inf, -np.inf], np.nan)
        clv_client = aov_per_client * freq_annual * margin * clv_years
        clv_avg = clv_client.replace([np.inf, -np.inf], np.nan).mean()
    else:
        clv_avg = np.nan

    return {
        "exposes": n,
        "acheteurs": int(buyers),
        "conversion": conv,
        "taux_reponse": resp_rate,
        "ca_total": total_revenue,
        "nb_commandes": total_orders,
        "aov": aov,
        "cac": cac,
        "roi": roi,
        "clv_moyen_proxy": clv_avg,
        "cout_campagne": campaign_cost,
    }


# Uplift (Test vs Control)
def compute_uplift(df: pd.DataFrame, control_col: str) -> Dict[str, float]:
    if control_col not in df.columns:
        return {"uplift_conversion": np.nan, "uplift_taux_reponse": np.nan}

    groups = df[control_col].dropna().unique()
    if len(groups) < 2:
        return {"uplift_conversion": np.nan, "uplift_taux_reponse": np.nan}

    # Heuristique : on considère que la modalité qui contient "test" est Test, sinon on prend la 1ère comme Control
    gvalues = [str(g).lower() for g in groups]
    if any("test" in g for g in gvalues) and any("control" in g for g in gvalues):
        test_label = [g for g in df[control_col].unique() if str(g).lower().find("test") >= 0][0]
        control_label = [g for g in df[control_col].unique() if str(g).lower().find("control") >= 0][0]
    else:
        # fallback : tri alphabétique
        sorted_vals = sorted(df[control_col].dropna().unique(), key=lambda x: str(x))
        control_label, test_label = sorted_vals[0], sorted_vals[1]

    def _metrics(sub: pd.DataFrame) -> Tuple[float, float]:
        # conversion
        revenue = safe_sum(sub, AMOUNT_COLS)
        orders = safe_sum(sub, ORDER_PARTS)
        buyers = ((safe_fillna(revenue, 0) > 0) | (safe_fillna(orders, 0) > 0)).sum()
        conv = buyers / len(sub) if len(sub) > 0 else np.nan

        # taux reponse
        if OPTIONAL_COLS["response"] in sub.columns:
            resp = sub[OPTIONAL_COLS["response"]].mean()
        else:
            resp = np.nan
        return conv, resp

    conv_test, resp_test = _metrics(df[df[control_col] == test_label])
    conv_ctrl, resp_ctrl = _metrics(df[df[control_col] == control_label])

    return {
        "uplift_conversion": (conv_test - conv_ctrl) if pd.notna(conv_test) and pd.notna(conv_ctrl) else np.nan,
        "uplift_taux_reponse": (resp_test - resp_ctrl) if pd.notna(resp_test) and pd.notna(resp_ctrl) else np.nan,
        "label_test": str(test_label),
        "label_control": str(control_label),
        "conv_test": conv_test,
        "conv_control": conv_ctrl,
        "resp_test": resp_test,
        "resp_control": resp_ctrl,
    }


# KPI par segments
def compute_segment_kpis(
    df: pd.DataFrame,
    segments: List[str],
    campaign_cost: float = 0.0,
    margin: float = 0.3,
    clv_years: int = 3,
) -> pd.DataFrame:
    segs = [c for c in segments if c in df.columns]
    if not segs:
        return pd.DataFrame()

    df = df.copy()
    df["__revenue__"] = safe_sum(df, AMOUNT_COLS)
    df["__orders__"]  = safe_sum(df, ORDER_PARTS)
    df["__buyer__"]   = ((safe_fillna(df["__revenue__"], 0) > 0) |
                         (safe_fillna(df["__orders__"], 0)  > 0)).astype(int)
    df["__response__"] = df[OPTIONAL_COLS["response"]] if OPTIONAL_COLS["response"] in df.columns else np.nan

    total_rows   = len(df)
    cost_per_row = (campaign_cost / total_rows) if total_rows > 0 else 0.0

    agg = (df
           .groupby(segs, dropna=False)
           .agg(exposes=("__buyer__", "size"),
                acheteurs=("__buyer__", "sum"),
                ca_total=("__revenue__", "sum"),
                nb_commandes=("__orders__", "sum"),
                taux_reponse=("__response__", "mean"))
           .reset_index())

    agg["conversion"]            = agg["acheteurs"] / agg["exposes"]
    agg["aov"]                   = agg["ca_total"] / agg["nb_commandes"].replace(0, np.nan)
    agg["cout_campagne_alloue"]  = agg["exposes"] * cost_per_row
    agg["roi"]                   = (agg["ca_total"] - agg["cout_campagne_alloue"]) / agg["cout_campagne_alloue"].replace(0, np.nan)

    if OPTIONAL_COLS["acq_date"] in df.columns:
        acq = parse_date_safe(df[OPTIONAL_COLS["acq_date"]])
        df["__freq_ann__"] = annualize_frequency(safe_fillna(df["__orders__"], 0), acq)
        df["__aov_cli__"]  = (safe_fillna(df["__revenue__"], 0) / safe_fillna(df["__orders__"], 0)).replace([np.inf, -np.inf], np.nan)
        df["__clv_cli__"]  = df["__aov_cli__"] * df["__freq_ann__"] * margin * clv_years
        clv_by_seg = (df.groupby(segs, dropna=False)["__clv_cli__"]
                        .mean()
                        .reset_index()
                        .rename(columns={"__clv_cli__": "clv_moyen_proxy"}))
        agg = agg.merge(clv_by_seg, on=segs, how="left")
    else:
        agg["clv_moyen_proxy"] = np.nan

    return agg.sort_values(segs + ["exposes"], ascending=[True]*len(segs) + [False])

# KPI optionnels/complémentaires
def compute_optional_kpis(df: pd.DataFrame) -> Dict[str, float]:
    out = {}

    # Recency (jours depuis dernier achat)
    if OPTIONAL_COLS["recency_days"] in df.columns:
        out["recency_median_jours"] = df[OPTIONAL_COLS["recency_days"]].median()
        out["recency_mean_jours"] = df[OPTIONAL_COLS["recency_days"]].mean()
    else:
        out["recency_median_jours"] = np.nan
        out["recency_mean_jours"] = np.nan

    # Visites web
    if OPTIONAL_COLS["web_visits"] in df.columns:
        out["visites_web_moy/mois"] = df[OPTIONAL_COLS["web_visits"]].mean()
    else:
        out["visites_web_moy/mois"] = np.nan

    # Part des commandes sous promo
    orders = safe_sum(df, ORDER_PARTS)
    if OPTIONAL_COLS["promo_orders"] in df.columns:
        promo = df[OPTIONAL_COLS["promo_orders"]].fillna(0)
        out["part_commandes_promo"] = (promo.sum() / orders.sum()) if orders.sum() > 0 else np.nan
    else:
        out["part_commandes_promo"] = np.nan

    # Taux d’acceptation par campagne (si colonnes présentes)
    for key in ("acc1", "acc2", "acc3", "acc4", "acc5"):
        col = OPTIONAL_COLS[key]
        if col in df.columns:
            out[f"taux_accept_{col}"] = df[col].mean()
        else:
            out[f"taux_accept_{col}"] = np.nan

    return out

# I/O
def read_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    df = pd.read_csv(path)
    return df


def write_outputs(
    out_dir: Path,
    global_kpis: Dict[str, float],
    segment_df: pd.DataFrame,
    optional_kpis: Dict[str, float],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV globaux
    pd.DataFrame([global_kpis]).to_csv(out_dir / "kpis_summary.csv", index=False)

    # CSV par segments (si non vide)
    if segment_df is not None and not segment_df.empty:
        segment_df.to_csv(out_dir / "kpis_by_segment.csv", index=False)

    # CSV optionnels
    pd.DataFrame([optional_kpis]).to_csv(out_dir / "kpis_optional.csv", index=False)

    # Excel
    with pd.ExcelWriter(out_dir / "kpis_report.xlsx", engine="xlsxwriter") as xw:
        pd.DataFrame([global_kpis]).to_excel(xw, sheet_name="Global", index=False)
        if segment_df is not None and not segment_df.empty:
            segment_df.to_excel(xw, sheet_name="Par_segment", index=False)
        pd.DataFrame([optional_kpis]).to_excel(xw, sheet_name="Optionnels", index=False)


# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calcul de KPI marketing sur camp_market_clean.csv")
    parser.add_argument("--input", default="data/raw/camp_market_clean.csv", help="Chemin du CSV d'entrée.")
    parser.add_argument("--output", default="data/processed", help="Dossier de sortie.")
    parser.add_argument("--campaign-cost", type=float, default=0.0, help="Coût total de la campagne (float).")
    parser.add_argument("--margin", type=float, default=0.30, help="Marge unitaire (0.30 = 30%).")
    parser.add_argument("--clv-years", type=int, default=3, help="Horizon CLV en années.")
    parser.add_argument("--control-col", default=None, help="Colonne Test/Control pour calculer l’uplift.")
    parser.add_argument("--segments", nargs="*", default=[], help="Liste de colonnes de segmentation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output)

    # Chargement
    df = read_input(input_path)

    # KPI globaux
    global_kpis = compute_global_kpis(
        df,
        campaign_cost=args.campaign_cost,
        margin=args.margin,
        clv_years=args.clv_years,
    )

    # Uplift (si demandé)
    if args.control_col:
        uplift = compute_uplift(df, args.control_col)
        global_kpis.update(uplift)

    # KPI par segments
    seg_df = compute_segment_kpis(
        df,
        segments=args.segments,
        campaign_cost=args.campaign_cost,
        margin=args.margin,
        clv_years=args.clv_years,
    )

    # KPI optionnels
    optional_kpis = compute_optional_kpis(df)

    # Exports
    write_outputs(out_dir, global_kpis, seg_df, optional_kpis)

    # Affichage console synthétique
    print("\n=== KPI GLOBAUX ===")
    for k, v in global_kpis.items():
        print(f"{k:>24} : {v}")

    if args.segments:
        print("\n=== KPI PAR SEGMENT ===")
        print(seg_df.head(20).to_string(index=False))

    print("\n=== KPI OPTIONNELS ===")
    for k, v in optional_kpis.items():
        print(f"{k:>24} : {v}")

    print(f"\nFichiers exportés dans: {out_dir.resolve()}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
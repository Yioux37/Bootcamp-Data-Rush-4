#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
kpi_terminal.py — KPIs marketing avec rendu terminal soigné (FR).
Usage:
  python kpi_terminal.py --file data/camp_market_clean.csv --cout-campagne 10000 --marge 0.3 --horizon 2
"""

import argparse
from datetime import datetime
import math
import sys
import re
import numpy as np
import pandas as pd

# ---------- Helpers d'affichage (pas de dépendance externe) ----------

USE_UNICODE = sys.stdout.encoding and "UTF" in sys.stdout.encoding.upper()

def _hline(width, heavy=False):
    if USE_UNICODE:
        h = "═" if heavy else "─"
        return h * width
    return "-" * width

def _box(title, lines, width=80):
    title = f" {title.strip()} "
    if USE_UNICODE:
        tl, tr, bl, br = "╔", "╗", "╚", "╝"
        top = tl + _hline(width - 2, True) + tr
        bot = bl + _hline(width - 2, True) + br
        title_bar = "╠" + title.center(width - 2, "═") + "╣"
        content = []
        for ln in lines:
            s = ln[:width - 4]
            content.append("║ " + s.ljust(width - 4) + " ║")
        return "\n".join([top, title_bar] + content + [bot])
    else:
        top = "+" + _hline(width - 2) + "+"
        title_bar = "+" + title.center(width - 2, "-") + "+"
        bot = "+" + _hline(width - 2) + "+"
        content = []
        for ln in lines:
            s = ln[:width - 4]
            content.append("| " + s.ljust(width - 4) + " |")
        return "\n".join([top, title_bar] + content + [bot])

def fmt_num(x, decimals=2):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "-"
        if isinstance(x, (int, np.integer)):
            return f"{int(x):,}".replace(",", " ")
        return f"{float(x):,.{decimals}f}".replace(",", " ")
    except Exception:
        return "-"

def fmt_pct(x, decimals=2):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "-"
        return f"{100*float(x):.{decimals}f}%"
    except Exception:
        return "-"

def safe_div(a, b):
    try:
        if b == 0 or b is None or (isinstance(b, float) and math.isnan(b)):
            return float("nan")
        return a / b
    except Exception:
        return float("nan")

def print_kv_table(title, rows, width=80, key_w=34):
    # rows: list of (label, value_str)
    lines = []
    for k, v in rows:
        k = str(k)
        v = str(v)
        if len(k) > key_w:
            k = k[:key_w-1] + "…"
        pad = " " * max(1, key_w - len(k))
        lines.append(f"{k}{pad} : {v}")
    print(_box(title, lines, width=width))
    print()

def print_table(title, headers, rows, width=80, col_w=None):
    if col_w is None:
        n = len(headers)
        col_w = [max(len(str(h)), 10) for h in headers]
        for r in rows:
            for i, c in enumerate(r):
                col_w[i] = max(col_w[i], len(str(c)))
        total = sum(col_w) + 3*(len(headers)-1)
        target = min(max(total, 40), width-4)
        if total > target:
            ratio = target/total
            col_w = [max(8, int(w*ratio)) for w in col_w]

    def fmt_row(cells):
        out = []
        for i, c in enumerate(cells):
            s = str(c)
            if len(s) > col_w[i]:
                s = s[:col_w[i]-1] + "…"
            out.append(s.ljust(col_w[i]))
        return " | ".join(out)

    lines = []
    lines.append(fmt_row(headers))
    lines.append("-" * len(lines[0]))
    for r in rows:
        lines.append(fmt_row(r))
    print(_box(title, lines, width=width))
    print()

# ---------- Utilitaires de nettoyage ----------

def ensure_numeric(df, cols):
    """
    Force les colonnes indiquées en float.
    Gère les virgules décimales, espaces, symboles monétaires, etc.
    """
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].astype(str)
        # virgules -> points
        s = s.str.replace(",", ".", regex=False)
        # enlève tout sauf chiffres, point, signe, exposant
        s = s.str.replace(r'[^0-9\.\-eE+]', '', regex=True)
        df[c] = pd.to_numeric(s, errors="coerce")
    return df

# ---------- Calculs KPI ----------

def build_kpis(df, cout_campagne=0.0, marge=0.30, horizon=2.0):
    # Colonnes attendues (FR)
    required = [
        "Identifiant","Année_naissance","Education","Situation_matrimoniale","Revenu",
        "Enfant_charge","Ado_charge","Date_acquisition_client","Nombre_jours_depuis_dernier_achat",
        "Montant_vin","Montant_fruits","Montant_viande","Montant_poisson","Montant_sucreries","Montant_luxe",
        "Nb_achats_promo","Nb_achats_en_ligne","Achats_catalogue","Achats_magasin","Nb_visites_web_mois",
        "Accepte_Campagne_3","Accepte_Campagne_4","Accepte_Campagne_5","Accepte_Campagne_1","Accepte_Campagne_2",
        "Reclamation_client","Z_CostContact","Z_Revenue","Response"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    # Dates
    if not pd.api.types.is_datetime64_any_dtype(df["Date_acquisition_client"]):
        df["Date_acquisition_client"] = pd.to_datetime(df["Date_acquisition_client"], errors="coerce")

    # Force numérique sur toutes les colonnes quantitatives
    numeric_cols = [
        "Identifiant","Année_naissance","Revenu","Enfant_charge","Ado_charge",
        "Nombre_jours_depuis_dernier_achat","Nb_achats_promo","Nb_achats_en_ligne",
        "Achats_catalogue","Achats_magasin","Nb_visites_web_mois",
        "Z_CostContact","Z_Revenue","Response",
        "Montant_vin","Montant_fruits","Montant_viande","Montant_poisson","Montant_sucreries","Montant_luxe",
    ]
    df = ensure_numeric(df, numeric_cols)

    # Response propre (0/1)
    if "Response" in df.columns:
        df["Response"] = df["Response"].fillna(0).astype(float)

    today = pd.Timestamp.today().normalize()

    # Features de base
    df["Age"] = (today.year - df["Année_naissance"]).astype(float)
    df["Tenure_j"] = (today - df["Date_acquisition_client"]).dt.days.astype(float)
    df["Recency_j"] = df["Nombre_jours_depuis_dernier_achat"].astype(float)

    spend_cols = [
        "Montant_vin","Montant_fruits","Montant_viande",
        "Montant_poisson","Montant_sucreries","Montant_luxe"
    ]
    df["Spend_total"] = df[spend_cols].sum(axis=1)

    df["Freq"] = df["Achats_catalogue"] + df["Achats_magasin"] + df["Nb_achats_en_ligne"]
    df["AOV"] = df["Spend_total"] / df["Freq"].replace(0, np.nan)

    # Mix canal
    df["Part_online"]    = df["Nb_achats_en_ligne"] / df["Freq"].replace(0, np.nan)
    df["Part_magasin"]   = df["Achats_magasin"] / df["Freq"].replace(0, np.nan)
    df["Part_catalogue"] = df["Achats_catalogue"] / df["Freq"].replace(0, np.nan)

    # Mix catégories
    for cat in ["vin","fruits","viande","poisson","sucreries","luxe"]:
        df[f"Part_{cat}"] = df[f"Montant_{cat}"] / df["Spend_total"].replace(0, np.nan)

    # Cross-sell & promo
    df["Cross_sell"] = (df[spend_cols] > 0).sum(axis=1) / len(spend_cols)
    df["Promo_ratio"] = df["Nb_achats_promo"] / df["Freq"].replace(0, np.nan)

    # Charges & réclamations
    df["Charge_index"] = df["Enfant_charge"] + df["Ado_charge"]
    df["Has_claim"] = (df["Reclamation_client"] > 0).astype(int)

    # Freq annuelle (proxy) pour CLV
    df["Tenure_annees"] = (df["Tenure_j"] / 365).clip(lower=1/12)  # >= 1 mois
    df["Freq_par_an"] = df["Freq"] / df["Tenure_annees"]
    df["CLV_lite"] = df["AOV"] * df["Freq_par_an"] * float(marge) * float(horizon)

    # Colonne de contrôle (TRT/CTL) auto si absente
    if "Groupe" not in df.columns:
        # hash stable de l'identifiant -> split 50/50 reproductible
        rng = pd.util.hash_pandas_object(df["Identifiant"].fillna(0).astype(int), index=False)
        df["Groupe"] = (rng % 2 == 0).astype(int)  # 1=TRT, 0=CTL

    # Agrégations clés
    n_total = len(df)
    n_trt = int((df["Groupe"] == 1).sum())
    n_ctl = int((df["Groupe"] == 0).sum())

    conv_total = float(df["Response"].mean())
    conv_trt = float(df.loc[df["Groupe"] == 1, "Response"].mean())
    conv_ctl = float(df.loc[df["Groupe"] == 0, "Response"].mean())

    uplift_abs = conv_trt - conv_ctl
    uplift_rel = uplift_abs / conv_ctl if conv_ctl and not math.isnan(conv_ctl) and conv_ctl > 0 else float("nan")

    rev_trt = float(df.loc[df["Groupe"] == 1, "Spend_total"].mean())
    rev_ctl = float(df.loc[df["Groupe"] == 0, "Spend_total"].mean())
    inc_rev_p_cust = rev_trt - rev_ctl
    inc_rev_total = inc_rev_p_cust * n_trt

    # CPA & ROI
    acheteurs_trt = int(df.loc[df["Groupe"] == 1, "Response"].sum())
    cpa = safe_div(cout_campagne, acheteurs_trt) if cout_campagne > 0 else float("nan")
    roi = safe_div((inc_rev_total - cout_campagne), cout_campagne) if cout_campagne > 0 else float("nan")

    # Récap global
    global_kpi = {
        "Clients (total)": n_total,
        "Test (TRT)": n_trt,
        "Contrôle (CTL)": n_ctl,
        "Taux de réponse (global)": conv_total,
        "Freq moyenne": float(df["Freq"].mean()),
        "AOV moyen": float(df["AOV"].mean()),
        "Dépense totale moyenne": float(df["Spend_total"].mean()),
        "CLV lite moyenne": float(df["CLV_lite"].mean()),
        "Promo_ratio moyen": float(df["Promo_ratio"].mean()),
        "Part online moyenne": float(df["Part_online"].mean()),
        "Réclamants (%)": float(df["Has_claim"].mean()),
    }

    campagne_kpi = {
        "Conv TRT": conv_trt,
        "Conv CTL": conv_ctl,
        "Uplift absolu": uplift_abs,
        "Uplift relatif": uplift_rel,
        "Δ Revenu moyen/cliente": inc_rev_p_cust,
        "Incrément total (≈)": inc_rev_total,
        "CPA (si coût>0)": cpa,
        "ROI incrémental": roi,
        "Acheteurs TRT (#)": acheteurs_trt
    }

    # Segments rapides (assure dtype numérique pour nlargest)
    df["CLV_lite"] = pd.to_numeric(df["CLV_lite"], errors="coerce").fillna(0.0)
    top_val = df.nlargest(5, "CLV_lite")[["Identifiant","CLV_lite","Spend_total","Freq","AOV","Part_online","Promo_ratio"]]
    risques = df.sort_values("Recency_j", ascending=False).head(5)[["Identifiant","Recency_j","Spend_total","Freq","Part_online"]]

    return df, global_kpi, campagne_kpi, top_val, risques

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="KPIs marketing – rendu terminal soigné")
    parser.add_argument("--file", required=True, help="Chemin du CSV nettoyé")
    parser.add_argument("--cout-campagne", type=float, default=0.0, help="Coût total de la campagne (monétaire)")
    parser.add_argument("--marge", type=float, default=0.30, help="Marge (%) utilisée pour CLV lite (ex: 0.30)")
    parser.add_argument("--horizon", type=float, default=2.0, help="Horizon (années) pour CLV lite (ex: 2)")
    parser.add_argument("--width", type=int, default=96, help="Largeur d'affichage")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.file, parse_dates=["Date_acquisition_client"])
    except Exception as e:
        print(f"Erreur de lecture CSV: {e}")
        sys.exit(1)

    # Info colonne de contrôle
    warn = None
    if "Groupe" not in df.columns:
        warn = "[INFO] Colonne de contrôle 'Groupe' absente -> création automatique (aléatoire 50/50)."

    try:
        df_out, global_kpi, campagne_kpi, top_val, risques = build_kpis(
            df,
            cout_campagne=args.cout_campagne,
            marge=args.marge,
            horizon=args.horizon
        )
    except Exception as e:
        print(f"Erreur de calcul KPI: {e}")
        sys.exit(2)

    width = args.width

    # Header
    header_lines = [
        f"Fichier : {args.file}",
        f"Date exécution : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Paramètres : coût={fmt_num(args.cout_campagne)} | marge={fmt_pct(args.marge)} | horizon={fmt_num(args.horizon,1)} an(s)"
    ]
    print(_box("TABLEAU DE BORD KPI – TERMINAL", header_lines, width=width))
    print()

    if warn:
        print(_box("NOTE", [warn], width=width))
        print()

    # Bloc Global
    rows_global = [
        ("Clients (total)", fmt_num(global_kpi["Clients (total)"], 0)),
        ("Test (TRT)", fmt_num(global_kpi["Test (TRT)"], 0)),
        ("Contrôle (CTL)", fmt_num(global_kpi["Contrôle (CTL)"], 0)),
        ("Taux de réponse (global)", fmt_pct(global_kpi["Taux de réponse (global)"])),
        ("Freq moyenne", fmt_num(global_kpi["Freq moyenne"], 2)),
        ("AOV moyen", fmt_num(global_kpi["AOV moyen"], 2)),
        ("Dépense totale moyenne", fmt_num(global_kpi["Dépense totale moyenne"], 2)),
        ("CLV lite moyenne", fmt_num(global_kpi["CLV lite moyenne"], 2)),
        ("Promo_ratio moyen", fmt_pct(global_kpi["Promo_ratio moyen"])),
        ("Part online moyenne", fmt_pct(global_kpi["Part online moyenne"])),
        ("Réclamants (%)", fmt_pct(global_kpi["Réclamants (%)"])),
    ]
    print_kv_table("KPI GLOBAUX", rows_global, width=width)

    # Bloc Campagne
    rows_campagne = [
        ("Conv TRT", fmt_pct(campagne_kpi["Conv TRT"])),
        ("Conv CTL", fmt_pct(campagne_kpi["Conv CTL"])),
        ("Uplift absolu", fmt_pct(campagne_kpi["Uplift absolu"])),
        ("Uplift relatif", fmt_pct(campagne_kpi["Uplift relatif"]) if not math.isnan(campagne_kpi["Uplift relatif"]) else "-"),
        ("Δ Revenu moyen / client", fmt_num(campagne_kpi["Δ Revenu moyen/cliente"], 2)),
        ("Incrément total (≈)", fmt_num(campagne_kpi["Incrément total (≈)"], 2)),
        ("Acheteurs TRT (#)", fmt_num(campagne_kpi["Acheteurs TRT (#)"], 0)),
        ("CPA", fmt_num(campagne_kpi["CPA (si coût>0)"], 2) if not math.isnan(campagne_kpi["CPA (si coût>0)"]) else "-"),
        ("ROI incrémental", fmt_pct(campagne_kpi["ROI incrémental"]) if not math.isnan(campagne_kpi["ROI incrémental"]) else "-"),
    ]
    print_kv_table("KPI CAMPAGNE / EXPÉRIMENTATION", rows_campagne, width=width)

    # Top valeur & Risque
    headers_top = ["Identifiant","CLV_lite","Spend","Freq","AOV","Part_online","Promo_ratio"]
    rows_top = [
        [
            int(r.Identifiant) if not pd.isna(r.Identifiant) else "-",
            fmt_num(r.CLV_lite,2),
            fmt_num(r.Spend_total,2),
            fmt_num(r.Freq,2),
            fmt_num(r.AOV,2),
            fmt_pct(r.Part_online),
            fmt_pct(r.Promo_ratio)
        ]
        for _, r in top_val.iterrows()
    ]
    print_table("TOP 5 CLIENTS PAR CLV (LITE)", headers_top, rows_top, width=width)

    headers_risk = ["Identifiant","Recency_j","Spend","Freq","Part_online"]
    rows_risk = [
        [
            int(r.Identifiant) if not pd.isna(r.Identifiant) else "-",
            fmt_num(r.Recency_j,0),
            fmt_num(r.Spend_total,2),
            fmt_num(r.Freq,2),
            fmt_pct(r.Part_online)
        ]
        for _, r in risques.iterrows()
    ]
    print_table("TOP 5 À RISQUE (RECENCY ÉLEVÉE)", headers_risk, rows_risk, width=width)

    # Mini synthèse actionnable
    synth = [
        "• Priorités : améliorer Conv TRT vs CTL, réduire Promo_ratio si érosion marge, stimuler réachat (Recency).",
        "• Ciblage : exploiter TOP CLV pour cross-sell premium, relancer clients à risque avec offre non-promo.",
        "• Canal : si Part_online faible, tester parcours web/app et incentives non monétaires."
    ]
    print(_box("SYNTHÈSE ACTIONNABLE (RÉSUMÉ)", synth, width=width))

if __name__ == "__main__":
    main()
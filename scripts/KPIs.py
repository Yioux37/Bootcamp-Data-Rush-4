#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
kpi_terminal.py — KPIs marketing (FR) avec rendu terminal soigné, SANS groupe test/contrôle.
Ajoute l'analyse des campagnes 1..5 : taux d'acceptation, CA des acceptants, etc.

Usage (exemples) :
  python kpi_terminal.py --file data/camp_market_clean.csv --cout-campagne 12000 --marge 0.30 --horizon 2
  python kpi_terminal.py --file data/camp_market_clean.csv --width 110
"""

import argparse
from datetime import datetime
import math
import sys
import numpy as np
import pandas as pd

# Helpers d'affichage (ASCII/Unicode, aucune dépendance externe)

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
            s = ("" if ln is None else str(ln))[:width - 4]
            content.append("║ " + s.ljust(width - 4) + " ║")
        return "\n".join([top, title_bar] + content + [bot])
    else:
        top = "+" + _hline(width - 2) + "+"
        title_bar = "+" + title.center(width - 2, "-") + "+"
        bot = "+" + _hline(width - 2) + "+"
        content = []
        for ln in lines:
            s = ("" if ln is None else str(ln))[:width - 4]
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
    lines = []
    for k, v in rows:
        k = str(k); v = str(v)
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

# Nettoyage & conversions

def ensure_numeric(df, cols):
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].astype(str)
        s = s.str.replace(",", ".", regex=False)
        s = s.str.replace(r'[^0-9\.\-eE+]', '', regex=True)
        df[c] = pd.to_numeric(s, errors="coerce")
    return df

# Indicateurs de qualité & fiabilité

def iqr_outlier_mask(series):
    q1, q3 = np.nanpercentile(series, [25, 75])
    iqr = q3 - q1
    low = q1 - 1.5*iqr
    high = q3 + 1.5*iqr
    return (series < low) | (series > high)

def reliability_score(metrics):
    score = 100.0
    score -= max(0, (1 - metrics["coverage"])) * 30.0
    score -= min(30.0, metrics["bad_dates_pct"] * 100 * 0.5)
    score -= min(20.0, (metrics["dup_ids"] > 0) * 10.0 + min(metrics["dup_ids"], 10) * 1.0)
    score -= min(10.0, metrics["zero_freq_pct"] * 100 * 0.3)
    score -= min(20.0, metrics["outlier_pct"] * 100 * 0.5)
    return max(0.0, min(100.0, score))

# KPI principaux + Campagnes

def build_kpis(df, cout_campagne=0.0, marge=0.30, horizon=2.0):
    required = [
        "Identifiant","Année_naissance","Education","Situation_matrimoniale","Revenu",
        "Enfant_charge","Ado_charge","Date_acquisition_client","Nombre_jours_depuis_dernier_achat",
        "Montant_vin","Montant_fruits","Montant_viande","Montant_poisson","Montant_sucreries","Montant_luxe",
        "Nb_achats_promo","Nb_achats_en_ligne","Achats_catalogue","Achats_magasin","Nb_visites_web_mois",
        "Accepte_Campagne_3","Accepte_Campagne_4","Accepte_Campagne_5","Accepte_Campagne_1","Accepte_Campagne_2",
        "Reclamation_client","Z_CostContact","Z_Revenue","Response"
    ]
    present = [c for c in required if c in df.columns]
    coverage = len(present) / len(required)

    if "Date_acquisition_client" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date_acquisition_client"]):
        df["Date_acquisition_client"] = pd.to_datetime(df["Date_acquisition_client"], errors="coerce")

    numeric_cols = [
        "Identifiant","Année_naissance","Revenu","Enfant_charge","Ado_charge",
        "Nombre_jours_depuis_dernier_achat","Nb_achats_promo","Nb_achats_en_ligne",
        "Achats_catalogue","Achats_magasin","Nb_visites_web_mois",
        "Z_CostContact","Z_Revenue","Response",
        "Montant_vin","Montant_fruits","Montant_viande","Montant_poisson","Montant_sucreries","Montant_luxe",
        "Accepte_Campagne_1","Accepte_Campagne_2","Accepte_Campagne_3","Accepte_Campagne_4","Accepte_Campagne_5",
    ]
    df = ensure_numeric(df, numeric_cols)

    if "Response" in df.columns:
        df["Response"] = df["Response"].fillna(0).astype(float)

    today = pd.Timestamp.today().normalize()
    if "Année_naissance" in df.columns:
        df["Age"] = (today.year - df["Année_naissance"]).astype(float)
    if "Date_acquisition_client" in df.columns:
        df["Tenure_j"] = (today - df["Date_acquisition_client"]).dt.days.astype(float)
    if "Nombre_jours_depuis_dernier_achat" in df.columns:
        df["Recency_j"] = df["Nombre_jours_depuis_dernier_achat"].astype(float)

    spend_cols = [c for c in [
        "Montant_vin","Montant_fruits","Montant_viande",
        "Montant_poisson","Montant_sucreries","Montant_luxe"
    ] if c in df.columns]
    df["Spend_total"] = df[spend_cols].sum(axis=1)

    purch_cols = [c for c in ["Achats_catalogue","Achats_magasin","Nb_achats_en_ligne"] if c in df.columns]
    df["Freq"] = df[purch_cols].sum(axis=1)
    df["AOV"] = df["Spend_total"] / df["Freq"].replace(0, np.nan)

    df["Part_online"]    = df["Nb_achats_en_ligne"] / df["Freq"].replace(0, np.nan)
    df["Part_magasin"]   = df["Achats_magasin"] / df["Freq"].replace(0, np.nan)
    df["Part_catalogue"] = df["Achats_catalogue"] / df["Freq"].replace(0, np.nan)
    for cat in ["vin","fruits","viande","poisson","sucreries","luxe"]:
        col = f"Montant_{cat}"
        if col in df.columns:
            df[f"Part_{cat}"] = df[col] / df["Spend_total"].replace(0, np.nan)

    if spend_cols:
        df["Cross_sell"] = (df[spend_cols] > 0).sum(axis=1) / len(spend_cols)
    else:
        df["Cross_sell"] = np.nan
    df["Promo_ratio"] = df["Nb_achats_promo"] / df["Freq"].replace(0, np.nan)

    df["Charge_index"] = df["Enfant_charge"] + df["Ado_charge"]
    df["Has_claim"] = (df["Reclamation_client"] > 0).astype(int)

    df["Tenure_annees"] = (df.get("Tenure_j", pd.Series(np.nan, index=df.index)) / 365).clip(lower=1/12)
    df["Freq_par_an"] = df["Freq"] / df["Tenure_annees"]
    df["CLV_lite"] = df["AOV"] * df["Freq_par_an"] * float(marge) * float(horizon)

    # Qualité & Fiabilité
    n_total = len(df)
    bad_dates_pct = float(df["Date_acquisition_client"].isna().mean()) if "Date_acquisition_client" in df.columns else 1.0
    dup_ids = int(df["Identifiant"].duplicated().sum()) if "Identifiant" in df.columns else 0
    zero_freq_pct = float((df["Freq"].fillna(0) == 0).mean()) if "Freq" in df.columns else 1.0
    outlier_mask = iqr_outlier_mask(df["Spend_total"].fillna(0).astype(float))
    outlier_pct = float(outlier_mask.mean()) if n_total > 0 else 0.0

    rel_inputs = {
        "coverage": coverage,
        "bad_dates_pct": bad_dates_pct,
        "dup_ids": dup_ids,
        "zero_freq_pct": zero_freq_pct,
        "outlier_pct": outlier_pct
    }
    score_rel = reliability_score(rel_inputs)
    label_rel = ("Excellente" if score_rel >= 90 else
                 "Bonne" if score_rel >= 75 else
                 "Moyenne" if score_rel >= 60 else
                 "Faible")

    # KPI Globaux
    global_kpi = {
        "Clients (total)": n_total,
        "Taux de réponse (global)": float(df["Response"].mean()) if "Response" in df.columns else float("nan"),
        "Freq moyenne": float(df["Freq"].mean()),
        "AOV moyen": float(df["AOV"].mean()),
        "Dépense totale moyenne": float(df["Spend_total"].mean()),
        "CLV lite moyenne": float(df["CLV_lite"].mean()),
        "Promo_ratio moyen": float(df["Promo_ratio"].mean()),
        "Part online moyenne": float(df["Part_online"].mean()),
        "Réclamants (%)": float(df["Has_claim"].mean()),
    }

    # IC95% pour le taux de réponse global (Wilson)
    def wilson_ci(p, n, z=1.96):
        if n == 0 or p != p:
            return (float("nan"), float("nan"))
        denom = 1 + z**2 / n
        center = (p + z**2/(2*n)) / denom
        margin = z * math.sqrt((p*(1-p)/n) + (z**2/(4*n**2))) / denom
        return (max(0.0, center - margin), min(1.0, center + margin))

    p = global_kpi["Taux de réponse (global)"]
    n = n_total
    ci_low, ci_high = wilson_ci(p, n)

    #  Campagnes 1..5 
    camp_cols = [c for c in ["Accepte_Campagne_1","Accepte_Campagne_2","Accepte_Campagne_3","Accepte_Campagne_4","Accepte_Campagne_5"] if c in df.columns]
    camp_rows = []
    for c in camp_cols:
        k = int(c.split("_")[-1])
        acc = df[c].fillna(0).astype(int)
        n_accept = int(acc.sum())
        rate = float(acc.mean()) if len(acc) else float("nan")

        mask = acc == 1
        ca_total_accept = float(df.loc[mask, "Spend_total"].sum())
        ca_moy_accept = float(df.loc[mask, "Spend_total"].mean()) if n_accept > 0 else float("nan")
        ca_moy_par_client_global = safe_div(ca_total_accept, n_total)

        freq_acc = float(df.loc[mask, "Freq"].mean()) if n_accept > 0 else float("nan")
        aov_acc  = float(df.loc[mask, "AOV"].mean())  if n_accept > 0 else float("nan")
        resp_acc = float(df.loc[mask, "Response"].mean()) if "Response" in df.columns and n_accept > 0 else float("nan")

        camp_rows.append({
            "Campagne": k,
            "Taux_acceptation": rate,
            "Acceptants": n_accept,
            "CA_total_acceptants": ca_total_accept,
            "CA_moyen_acceptant": ca_moy_accept,
            "CA_moyen_par_client_global": ca_moy_par_client_global,
            "Freq_moy_acc": freq_acc,
            "AOV_moy_acc": aov_acc,
            "Taux_resp_des_acceptants": resp_acc
        })

    camp_df = pd.DataFrame(camp_rows).sort_values(["Taux_acceptation","CA_total_acceptants"], ascending=[False, False]) if camp_rows else pd.DataFrame()

    # Segments 
    df["CLV_lite"] = pd.to_numeric(df["CLV_lite"], errors="coerce").fillna(0.0)
    top_val = df.nlargest(5, "CLV_lite")[["Identifiant","CLV_lite","Spend_total","Freq","AOV","Part_online","Promo_ratio"]]
    risques = df.sort_values("Recency_j", ascending=False).head(5)[["Identifiant","Recency_j","Spend_total","Freq","Part_online"]]

    quality = {
        "coverage": coverage,
        "bad_dates_pct": bad_dates_pct,
        "dup_ids": dup_ids,
        "zero_freq_pct": zero_freq_pct,
        "outlier_pct": outlier_pct,
        "score": score_rel,
        "label": label_rel
    }
    ci = {"p": p, "n": n, "low": ci_low, "high": ci_high}
    return df, global_kpi, quality, ci, top_val, risques, camp_df

# Main 

def main():
    parser = argparse.ArgumentParser(description="KPIs marketing – rendu terminal soigné (sans groupes) + analyse campagnes")
    parser.add_argument("--file", required=True, help="Chemin du CSV nettoyé")
    parser.add_argument("--cout-campagne", type=float, default=0.0, help="Coût total de la campagne (monétaire)")
    parser.add_argument("--marge", type=float, default=0.30, help="Marge (%) utilisée pour CLV lite (ex: 0.30)")
    parser.add_argument("--horizon", type=float, default=2.0, help="Horizon (années) pour CLV lite (ex: 2)")
    parser.add_argument("--width", type=int, default=100, help="Largeur d'affichage")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.file, parse_dates=["Date_acquisition_client"])
    except Exception as e:
        print(f"Erreur de lecture CSV: {e}")
        sys.exit(1)

    try:
        df_out, global_kpi, quality, ci, top_val, risques, camp_df = build_kpis(
            df,
            cout_campagne=args.cout_campagne,
            marge=args.marge,
            horizon=args.horizon
        )
    except Exception as e:
        print(f"Erreur de calcul KPI: {e}")
        sys.exit(2)

    width = args.width

    header_lines = [
        f"Fichier : {args.file}",
        f"Date exécution : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Paramètres : coût={fmt_num(args.cout_campagne)} | marge={fmt_pct(args.marge)} | horizon={fmt_num(args.horizon,1)} an(s)"
    ]
    print(_box("TABLEAU DE BORD KPI – TERMINAL (GLOBAL + CAMPAGNES)", header_lines, width=width))
    print()

    qual_lines = [
        f"Couverture colonnes clés           : {fmt_pct(quality['coverage'])}",
        f"Dates invalides                    : {fmt_pct(quality['bad_dates_pct'])}",
        f"Identifiants dupliqués (#)         : {fmt_num(quality['dup_ids'], 0)}",
        f"Freq = 0 (impact AOV/parts)        : {fmt_pct(quality['zero_freq_pct'])}",
        f"Outliers dépense totale            : {fmt_pct(quality['outlier_pct'])}",
        f"Score de fiabilité (0-100)         : {fmt_num(quality['score'], 0)}  ({quality['label']})",
    ]
    print(_box("QUALITÉ DES DONNÉES & FIABILITÉ", qual_lines, width=width))
    print()

    rows_global = [
        ("Clients (total)", fmt_num(global_kpi["Clients (total)"], 0)),
        ("Taux de réponse (global)", fmt_pct(global_kpi["Taux de réponse (global)"])),
        ("Freq moyenne", fmt_num(global_kpi["Freq moyenne"], 2)),
        ("AOV moyen", fmt_num(global_kpi["AOV moyen"], 2)),
        ("Dépense totale moyenne", fmt_num(global_kpi["Dépense totale moyenne"], 2)),
        ("CLV lite moyenne", fmt_num(global_kpi["CLV lite moyenne"], 2)),
        ("Promo_ratio moyen", fmt_pct(global_kpi["Promo_ratio moyen"])),
        ("Part online moyenne", fmt_pct(global_kpi["Part online moyenne"])),
        ("Réclamants (%)", fmt_pct(global_kpi["Réclamants (%)"])),
    ]
    print_kv_table("KPI GLOBAUX (SANS TRT/CTL)", rows_global, width=width)

    ic_lines = [
        f"Taux de réponse (global) : {fmt_pct(ci['p'])}  (IC95% {fmt_pct(ci['low'])} ; {fmt_pct(ci['high'])})",
        f"Taille d'échantillon (n) : {fmt_num(ci['n'], 0)}",
    ]
    print(_box("CONFIANCE STATISTIQUE (IC 95%) — TAUX GLOBAL", ic_lines, width=width))
    print()

    if not camp_df.empty:
        headers_c = [
            "Campagne","Taux_acceptation","Acceptants",
            "CA_total_acceptants","CA_moyen_acceptant","CA_moyen_par_client",
            "Freq_moy_acc","AOV_moy_acc","Taux_resp_acc"
        ]
        rows_c = []
        for _, r in camp_df.iterrows():
            rows_c.append([
                int(r.Campagne),
                fmt_pct(r.Taux_acceptation),
                fmt_num(r.Acceptants, 0),
                fmt_num(r.CA_total_acceptants, 2),
                fmt_num(r.CA_moyen_acceptant, 2),
                fmt_num(r.CA_moyen_par_client_global, 2),
                fmt_num(r.Freq_moy_acc, 2),
                fmt_num(r.AOV_moy_acc, 2),
                fmt_pct(r.Taux_resp_des_acceptants),
            ])
        print_table("KPI PAR CAMPAGNE (HISTORIQUE)", headers_c, rows_c, width=width)

        best_rate = camp_df.sort_values("Taux_acceptation", ascending=False).iloc[0]
        best_ca   = camp_df.sort_values("CA_total_acceptants", ascending=False).iloc[0]
        best_cam_lines = [
            f"• Plus haut taux d’acceptation : Campagne {int(best_rate.Campagne)} → {fmt_pct(best_rate.Taux_acceptation)} (acceptants={fmt_num(best_rate.Acceptants,0)})",
            f"• Plus gros CA des acceptants : Campagne {int(best_ca.Campagne)} → {fmt_num(best_ca.CA_total_acceptants,2)}",
            f"• CA moyen/acceptant le + élevé : Campagne {int(camp_df.iloc[camp_df['CA_moyen_acceptant'].idxmax()].Campagne)} "
            f"→ {fmt_num(camp_df['CA_moyen_acceptant'].max(),2)}",
        ]
        print(_box("SYNTHÈSE — MEILLEURES CAMPAGNES", best_cam_lines, width=width))
        print()
    else:
        print(_box("KPI PAR CAMPAGNE (HISTORIQUE)", ["Colonnes Accepte_Campagne_1..5 absentes."], width=width))
        print()

    headers_top = ["Identifiant","CLV_lite","Spend","Freq","AOV","Part_online","Promo_ratio"]
    rows_top = [
        [
            "-" if pd.isna(r.Identifiant) else int(r.Identifiant),
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
            "-" if pd.isna(r.Identifiant) else int(r.Identifiant),
            fmt_num(r.Recency_j,0),
            fmt_num(r.Spend_total,2),
            fmt_num(r.Freq,2),
            fmt_pct(r.Part_online)
        ]
        for _, r in risques.iterrows()
    ]
    print_table("TOP 5 À RISQUE (RECENCY ÉLEVÉE)", headers_risk, rows_risk, width=width)

    synth = [
        "• Accélérer la meilleure campagne (haut taux ou haut CA) et répliquer ses leviers.",
        "• Tester des optimisations sur les campagnes faibles (créa, ciblage, canal).",
        "• Activer upsell/cross-sell sur TOP CLV ; relancer clients à recency élevée avec offres non-promo."
    ]
    print(_box("SYNTHÈSE ACTIONNABLE (RÉSUMÉ)", synth, width=width))

if __name__ == "__main__":
    main()
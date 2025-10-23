#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
kpi_terminal.py — KPIs marketing avec rendu terminal soigné (FR) + Qualité & Confiance.
Usage:
  python kpi_terminal.py --file data/camp_market_clean.csv --cout-campagne 10000 --marge 0.3 --horizon 2
"""

import argparse
from datetime import datetime
import math
import sys
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
        s = s.str.replace(",", ".", regex=False)                  # virgules -> points
        s = s.str.replace(r'[^0-9\.\-eE+]', '', regex=True)       # enlève tout sauf chiffres/point/signe/exposant
        df[c] = pd.to_numeric(s, errors="coerce")
    return df

# ---------- Confiance statistique (IC, tests) ----------

def wilson_ci(k, n, z=1.96):
    """IC 95% de proportion (Wilson). Retourne (low, high) en proportion [0..1]."""
    if n is None or n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    half = (z * math.sqrt((p*(1-p) + z**2/(4*n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def two_prop_diff_ci(k1, n1, k0, n0, z=1.96):
    """IC 95% de la différence de proportions p1-p0 (approx normale)."""
    if n1<=0 or n0<=0:
        return (float("nan"), float("nan"), float("nan"))
    p1 = k1/n1; p0 = k0/n0
    d = p1 - p0
    se = math.sqrt(p1*(1-p1)/n1 + p0*(1-p0)/n0)
    if se == 0 or math.isnan(se):
        return (d, float("nan"), float("nan"))
    ci_low, ci_high = d - z*se, d + z*se
    # p-value bilatérale
    z_stat = d / se
    # CDF normale via erf
    phi = 0.5 * (1 + math.erf(abs(z_stat)/math.sqrt(2)))
    pval = 2 * (1 - phi)
    return (d, ci_low, ci_high, pval, se)

# ---------- Qualité & Fiabilité des données ----------

def data_quality_report(df):
    n = len(df)
    if n == 0:
        return {
            "coverage_core": 0, "date_invalid_rate": 1, "dup_rate": 0,
            "freq_zero_rate": 0, "outlier_rate": 0, "score": 0, "label": "N/A"
        }

    spend_cols = ["Montant_vin","Montant_fruits","Montant_viande","Montant_poisson","Montant_sucreries","Montant_luxe"]
    numeric_cols = [
        "Identifiant","Année_naissance","Revenu","Enfant_charge","Ado_charge",
        "Nombre_jours_depuis_dernier_achat","Nb_achats_promo","Nb_achats_en_ligne",
        "Achats_catalogue","Achats_magasin","Nb_visites_web_mois",
        "Z_CostContact","Z_Revenue","Response"
    ] + spend_cols + ["Freq","AOV","Spend_total"]

    # Couverture (non-NA) des colonnes clés
    coverages = []
    for c in numeric_cols:
        if c in df.columns:
            coverages.append(df[c].notna().mean())
    coverage_core = float(np.mean(coverages)) if coverages else 0.0

    # Dates invalides
    date_invalid_rate = float(df["Date_acquisition_client"].isna().mean()) if "Date_acquisition_client" in df.columns else 1.0

    # Doublons d'identifiants
    dup_count = int(df["Identifiant"].duplicated().sum()) if "Identifiant" in df.columns else 0
    dup_rate = dup_count / n

    # Freq = 0 (AOV/parts impossibles)
    freq_zero_rate = float((df["Freq"]==0).mean()) if "Freq" in df.columns else 1.0

    # Outliers dépense totale (IQR)
    outlier_rate = float("nan")
    if "Spend_total" in df.columns:
        q1 = df["Spend_total"].quantile(0.25)
        q3 = df["Spend_total"].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5*iqr
        high = q3 + 1.5*iqr
        outlier_rate = float(((df["Spend_total"] < low) | (df["Spend_total"] > high)).mean())

    # Score de fiabilité (0-100) — heuristique simple et transparente
    score = 100.0
    score -= (1 - coverage_core) * 40
    score -= date_invalid_rate * 10
    score -= min(dup_rate*100, 10)
    score -= min(freq_zero_rate*100, 15)
    if not math.isnan(outlier_rate):
        score -= min(outlier_rate*100, 15)
    score = float(max(0.0, min(100.0, score)))

    if score >= 90: label = "Excellente"
    elif score >= 75: label = "Bonne"
    elif score >= 60: label = "Moyenne"
    else: label = "Faible"

    return {
        "coverage_core": coverage_core,
        "date_invalid_rate": date_invalid_rate,
        "dup_rate": dup_rate,
        "dup_count": dup_count,
        "freq_zero_rate": freq_zero_rate,
        "outlier_rate": outlier_rate if not math.isnan(outlier_rate) else 0.0,
        "score": score,
        "label": label
    }

# ---------- Calculs KPI ----------

def build_kpis(df, cout_campagne=0.0, marge=0.30, horizon=2.0):
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

    # Numériques
    numeric_cols = [
        "Identifiant","Année_naissance","Revenu","Enfant_charge","Ado_charge",
        "Nombre_jours_depuis_dernier_achat","Nb_achats_promo","Nb_achats_en_ligne",
        "Achats_catalogue","Achats_magasin","Nb_visites_web_mois",
        "Z_CostContact","Z_Revenue","Response",
        "Montant_vin","Montant_fruits","Montant_viande","Montant_poisson","Montant_sucreries","Montant_luxe",
    ]
    df = ensure_numeric(df, numeric_cols)

    if "Response" in df.columns:
        df["Response"] = df["Response"].fillna(0).astype(float)

    today = pd.Timestamp.today().normalize()

    # Features de base
    df["Age"] = (today.year - df["Année_naissance"]).astype(float)
    df["Tenure_j"] = (today - df["Date_acquisition_client"]).dt.days.astype(float)
    df["Recency_j"] = df["Nombre_jours_depuis_dernier_achat"].astype(float)

    spend_cols = ["Montant_vin","Montant_fruits","Montant_viande","Montant_poisson","Montant_sucreries","Montant_luxe"]
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

    # Colonne Groupe auto si absente
    if "Groupe" not in df.columns:
        rng = pd.util.hash_pandas_object(df["Identifiant"].fillna(0).astype(int), index=False)
        df["Groupe"] = (rng % 2 == 0).astype(int)  # 1=TRT, 0=CTL

    # Agrégations clés
    n_total = len(df)
    n_trt = int((df["Groupe"] == 1).count() and (df["Groupe"] == 1).sum())
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

    # Segments rapides
    df["CLV_lite"] = pd.to_numeric(df["CLV_lite"], errors="coerce").fillna(0.0)
    top_val = df.nlargest(5, "CLV_lite")[["Identifiant","CLV_lite","Spend_total","Freq","AOV","Part_online","Promo_ratio"]]
    risques = df.sort_values("Recency_j", ascending=False).head(5)[["Identifiant","Recency_j","Spend_total","Freq","Part_online"]]

    # Rapport de qualité
    quality = data_quality_report(df)

    # Stat tests & IC
    k_trt = int(df.loc[df["Groupe"] == 1, "Response"].sum())
    k_ctl = int(df.loc[df["Groupe"] == 0, "Response"].sum())

    ci_trt = wilson_ci(k_trt, n_trt)
    ci_ctl = wilson_ci(k_ctl, n_ctl)
    d, d_lo, d_hi, pval, se = two_prop_diff_ci(k_trt, n_trt, k_ctl, n_ctl)

    stats_conf = {
        "Conv TRT": (conv_trt, ci_trt),
        "Conv CTL": (conv_ctl, ci_ctl),
        "Uplift absolu": (uplift_abs, (d_lo, d_hi)),
        "z (uplift)": (d / se if se and se>0 else float("nan")),
        "p-value (bilatérale)": pval
    }

    # Récap global & campagne
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

    return df, global_kpi, campagne_kpi, top_val, risques, quality, stats_conf

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

    warn = None
    if "Groupe" not in df.columns:
        warn = "[INFO] Colonne de contrôle 'Groupe' absente -> création automatique (split stable 50/50)."

    try:
        df_out, global_kpi, campagne_kpi, top_val, risques, quality, stats_conf = build_kpis(
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

    # --- Qualité & Fiabilité ---
    rows_q = [
        ("Couverture colonnes clés", fmt_pct(quality["coverage_core"])),
        ("Dates invalides", fmt_pct(quality["date_invalid_rate"])),
        ("Identifiants dupliqués (#)", fmt_num(quality["dup_count"], 0)),
        ("Freq = 0 (impact AOV/parts)", fmt_pct(quality["freq_zero_rate"])),
        ("Outliers dépense totale", fmt_pct(quality["outlier_rate"])),
        ("Score de fiabilité (0-100)", fmt_num(quality["score"], 0) + f"  ({quality['label']})"),
    ]
    print_kv_table("QUALITÉ DES DONNÉES & FIABILITÉ", rows_q, width=width)

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

    # --- Confiance statistique (IC 95%) ---
    conv_trt, (trt_lo, trt_hi) = stats_conf["Conv TRT"]
    conv_ctl, (ctl_lo, ctl_hi) = stats_conf["Conv CTL"]
    uplift, (up_lo, up_hi) = stats_conf["Uplift absolu"]
    z_stat = stats_conf["z (uplift)"]
    pval = stats_conf["p-value (bilatérale)"]

    rows_conf = [
        ("Conv TRT", f"{fmt_pct(conv_trt)}  (IC95% {fmt_pct(trt_lo)} ; {fmt_pct(trt_hi)})"),
        ("Conv CTL", f"{fmt_pct(conv_ctl)}  (IC95% {fmt_pct(ctl_lo)} ; {fmt_pct(ctl_hi)})"),
        ("Uplift absolu", f"{fmt_pct(uplift)}  (IC95% {fmt_pct(up_lo)} ; {fmt_pct(up_hi)})"),
        ("Marge d'erreur uplift (±)", fmt_pct((up_hi - up_lo)/2)),
        ("Test z uplift", f"z = {fmt_num(z_stat, 3)} ; p = {fmt_num(pval, 4)}"),
        ("Significatif (5%)", "Oui" if (not math.isnan(pval) and pval < 0.05) else "Non"),
    ]
    print_kv_table("CONFIANCE STATISTIQUE (IC 95%)", rows_conf, width=width)

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

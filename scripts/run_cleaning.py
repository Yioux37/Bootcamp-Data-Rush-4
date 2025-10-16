#!/usr/bin/env python3
# scripts/run_cleaning.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

# Constantes
POS = {
    "yes","oui","true","t","1","accepted","converted",
    "purchase","purchased","buy","bought","success","y"
}

FR_ORDER = [
    "Identifiant","Année_naissance","Education","Situation_matrimoniale","Revenu",
    "Enfant_charge","Ado_charge","Date_acquisition_client","Nombre_jours_depuis_dernier_achat",
    "Montant_vin","Montant_fruits","Montant_viande","Montant_poisson","Montant_sucreries","Montant_luxe",
    "Nb_achats_promo","Nb_achats_en_ligne","Achats_catalogue","Achats_magasin",
    "Nb_visites_web_mois",
    "Accepte_Campagne_3","Accepte_Campagne_4","Accepte_Campagne_5","Accepte_Campagne_1","Accepte_Campagne_2",
    "Reclamation_client","Z_CostContact","Z_Revenue","Response"
]

# Utils
def bar(title: str, ch: str="═", width: int=70):
    t = f" {title} "
    n = max(0, width - len(t))
    L = n // 2
    R = n - L
    return f"{ch*L}{t}{ch*R}"

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower().replace(" ", "_") for c in out.columns]
    return out

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if "date" in c or c.startswith("dt_"):
            out[c] = pd.to_datetime(out[c], errors="coerce", dayfirst=True)
    return out

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def to_int01(s) -> pd.Series:
    sr = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    return sr.clip(lower=0, upper=1)

# Nettoyage (règles métier)
def clean_business_rules(df: pd.DataFrame):
    log = {}
    df = normalize_cols(df)

    # trim objets
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

    # dates
    df = parse_dates(df)

    # (1) unicité id
    id_col = next((c for c in ["id","client_id","customer_id"] if c in df.columns), None)
    if id_col:
        before = len(df)
        df = df.drop_duplicates(subset=[id_col])
        log["dup_id_removed"] = before - len(df)

    # (2) années aberrantes
    yb_col = next((c for c in ["year_birth","birth_year","annee_naissance","year_of_birth"] if c in df.columns), None)
    if yb_col:
        df[yb_col] = to_num(df[yb_col])
        bad = [1893, 1899, 1900]
        rm = int(df[yb_col].isin(bad).sum())
        df = df.loc[~df[yb_col].isin(bad)].copy()
        log["bad_years_removed"] = rm

    # (3) statut marital (Alone→Single) + drop Absurd/YOLO
    m_col = next((c for c in ["marital_status","status_marital","marital","statut_marital"] if c in df.columns), None)
    if m_col:
        df[m_col] = df[m_col].astype(str).str.strip().str.lower()
        MAP = {
            "alone": "single", "single": "single",
            "married": "married", "together": "together",
            "divorced": "divorced", "widow": "widow", "widowed": "widow",
        }
        BAD = {"absurd","yolo"}
        before = len(df)
        df = df.loc[~df[m_col].isin(BAD)].copy()
        log["marital_bad_rows_removed"] = before - len(df)
        df[m_col] = df[m_col].map(MAP).fillna(df[m_col])
        df[m_col] = df[m_col].str.title()
        log["marital_modalities"] = sorted(df[m_col].dropna().unique().tolist())

    # (4) income: 666666 -> NA -> médiane
    if "income" in df.columns:
        df["income"] = to_num(df["income"])
        outliers = int((df["income"] == 666_666).sum())
        if outliers:
            df.loc[df["income"] == 666_666, "income"] = np.nan
        med = float(df["income"].median(skipna=True))
        df["income"] = df["income"].fillna(med)
        log["income_outliers_666666"] = outliers
        log["income_median_used"] = med

    # (5) conversion large vers numérique (coerce)
    keep_cat = {m_col or "", "education", "response"}
    for c in df.columns:
        if c not in keep_cat:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # (6) imputation simple
    for c in df.columns:
        if is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            mode = df[c].mode(dropna=True)
            df[c] = df[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    # (7) binaires texte -> 0/1
    for c in df.columns:
        if not is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) == 2:
            df[c] = df[c].astype(str).str.lower().map(lambda v: 1 if v in POS else 0)

    # (8) doublons globaux
    df = df.drop_duplicates()

    return df, log

# Schéma FR + zéro NA
def to_french_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "Identifiant":                  to_num(df.get("id")).fillna(0).astype(int),
        "Année_naissance":              to_num(df.get("year_birth")).astype("Int64"),
        "Education":                    df.get("education").astype(str),
        "Situation_matrimoniale":       df.get("marital_status").astype(str),
        "Revenu":                       to_num(df.get("income")).fillna(0).astype(float),
        "Enfant_charge":                to_num(df.get("kidhome")).fillna(0).astype(int),
        "Ado_charge":                   to_num(df.get("teenhome")).fillna(0).astype(int),
        "Date_acquisition_client":      pd.to_datetime(df.get("dt_customer"), errors="coerce"),
        "Nombre_jours_depuis_dernier_achat":
                                        to_num(df.get("recency")).fillna(0).astype(int),
        "Montant_vin":                  to_num(df.get("mntwines")).fillna(0).astype(float),
        "Montant_fruits":               to_num(df.get("mntfruits")).fillna(0).astype(float),
        "Montant_viande":               to_num(df.get("mntmeatproducts")).fillna(0).astype(float),
        "Montant_poisson":              to_num(df.get("mntfishproducts")).fillna(0).astype(float),
        "Montant_sucreries":            to_num(df.get("mntsweetproducts")).fillna(0).astype(float),
        "Montant_luxe":                 to_num(df.get("mntgoldprods")).fillna(0).astype(float),
        "Nb_achats_promo":              to_num(df.get("numdealspurchases")).fillna(0).astype(int),
        "Nb_achats_en_ligne":           to_num(df.get("numwebpurchases")).fillna(0).astype(int),
        # ← ICI : quantités, pas binaire
        "Achats_catalogue":             to_num(df.get("numcatalogpurchases")).fillna(0).astype(int),
        "Achats_magasin":               to_num(df.get("numstorepurchases")).fillna(0).astype(int),
        "Nb_visites_web_mois":          to_num(df.get("numwebvisitsmonth")).fillna(0).astype(int),
        "Accepte_Campagne_3":           to_int01(df.get("acceptedcmp3", 0)),
        "Accepte_Campagne_4":           to_int01(df.get("acceptedcmp4", 0)),
        "Accepte_Campagne_5":           to_int01(df.get("acceptedcmp5", 0)),
        "Accepte_Campagne_1":           to_int01(df.get("acceptedcmp1", 0)),
        "Accepte_Campagne_2":           to_int01(df.get("acceptedcmp2", 0)),
        "Reclamation_client":           to_num(df.get("complain")).fillna(0).astype(int),
        "Z_CostContact":                to_num(df.get("z_costcontact")).fillna(0).astype(float),
        "Z_Revenue":                    to_num(df.get("z_revenue")).fillna(0).astype(float),
        "Response":                     to_int01(df.get("response", 0)),
    })

    # ordre strict
    out = out[FR_ORDER]

    # zéro NA garanti
    for c in out.columns:
        if is_datetime64_any_dtype(out[c]):
            out[c] = out[c].fillna(pd.Timestamp("1970-01-01"))
        elif is_numeric_dtype(out[c]):
            out[c] = out[c].fillna(0)
        else:
            out[c] = out[c].fillna("Unknown")

    assert not out.isna().any().any(), "Il reste des NA dans le schéma FR."
    return out

# Rapport
def quick_report(df: pd.DataFrame, log: dict, src: str, dst: Path):
    print(bar("RAPPORT DE NETTOYAGE", "█"))
    print(f"Source : {src}")
    print(f"Sortie : {dst}\n")

    print(bar("ACTIONS EFFECTUÉES"))
    print(f"• Doublons supprimés sur id           : {log.get('dup_id_removed', 0)}")
    print(f"• Années aberrantes retirées          : {log.get('bad_years_removed', 0)} (1893, 1899, 1900)")
    print(f"• Lignes marital absurdes retirées    : {log.get('marital_bad_rows_removed', 0)} (Absurd/YOLO)")
    imed = log.get("income_median_used", float("nan"))
    print(f"• Income=666666 corrigés              : {log.get('income_outliers_666666', 0)}")
    print(f"• Médiane income utilisée             : {imed:.0f}" if pd.notna(imed) else "• Médiane income utilisée             : —")
    mods = log.get("marital_modalities", [])
    print(f"• Modalités marital finales           : {', '.join(mods) if mods else '—'}")

    print("\n" + bar("VUE D’ENSEMBLE"))
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if not is_numeric_dtype(df[c])]
    print(f"• Lignes x Colonnes                   : {df.shape[0]} x {df.shape[1]}")
    print(f"• Numériques / Catégorielles          : {len(num_cols)} / {len(cat_cols)}")
    print(f"• Zéro NA garanti                     : {not df.isna().any().any()}")

    if "Année_naissance" in df.columns:
        y = df["Année_naissance"].astype("Int64").dropna()
        if not y.empty:
            print("\n" + bar("RÉSUMÉ — Année_naissance"))
            print({"min": int(y.min()), "p50": int(y.median()), "max": int(y.max())})

    if "Revenu" in df.columns:
        s = df["Revenu"].astype(float)
        if not s.empty:
            print("\n" + bar("RÉSUMÉ — Revenu"))
            print({"min": float(s.min()), "p50": float(s.median()), "p95": float(np.percentile(s,95)), "max": float(s.max())})

    if "Nombre_jours_depuis_dernier_achat" in df.columns:
        r = df["Nombre_jours_depuis_dernier_achat"].astype(int)
        if not r.empty:
            print("\n" + bar("RÉSUMÉ — Recency (jours)"))
            print({"min": int(r.min()), "p50": int(np.median(r)), "p95": int(np.percentile(r,95)), "max": int(r.max())})

    print("\n" + bar("FIN DU RAPPORT", "█"))

# CLI
def main():
    ap = argparse.ArgumentParser(description="Nettoyage marketing → CSV (pas de Parquet)")
    ap.add_argument("--input", required=True, help="Chemin du CSV brut (séparateur ';')")
    ap.add_argument("--out", required=True, help="Chemin de sortie CSV propre (sera créé)")
    ap.add_argument("--sep", default=";", help="Séparateur CSV (défaut: ;)")
    ap.add_argument("--encoding", default="utf-8", help="Encodage (défaut: utf-8)")
    args = ap.parse_args()

    # Lecture
    df_raw = pd.read_csv(args.input, sep=args.sep, encoding=args.encoding)

    # Nettoyage + règles métier
    df_clean, log = clean_business_rules(df_raw)

    # Schéma FR + garantie zéro NA
    df_fr = to_french_schema(df_clean)

    # Écriture CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_fr.to_csv(out_path, index=False)

    # Rapport terminal
    quick_report(df_fr, log, args.input, out_path)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

POS = {"yes","oui","true","t","1","accepted","converted","purchase","purchased","buy","bought","success","y"}

# ---------- utils ----------
def bar(title: str, ch: str="═", width: int=70):
    t = f" {title} "; n = max(0, width-len(t)); L=n//2; R=n-L; return f"{ch*L}{t}{ch*R}"

def normalize_cols(df): 
    out=df.copy(); out.columns=[c.strip().lower().replace(" ","_") for c in df.columns]; return out

def parse_dates(df):
    out = df.copy()
    for c in out.columns:
        if "date" in c or c.startswith("dt_"):
            out[c] = pd.to_datetime(out[c], errors="coerce", dayfirst=True, infer_datetime_format=True)
    return out

def to_num(s): return pd.to_numeric(s, errors="coerce")

# ---------- imputation stricte NO-NA ----------
def strict_impute_no_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        s = out[c]
        # Datetime -> médiane temporelle, sinon min date si tout NA
        if np.issubdtype(s.dtype, np.datetime64):
            if s.notna().any():
                ts = s.view("int64")  # ns depuis epoch
                med = np.nanmedian(ts)
                fill = pd.to_datetime(int(med))
            else:
                fill = pd.Timestamp("1970-01-01")
            out[c] = s.fillna(fill)
        # Numérique -> médiane, sinon 0 si tout NA
        elif pd.api.types.is_numeric_dtype(s):
            if s.notna().any():
                fill = float(np.nanmedian(s))
            else:
                fill = 0.0
            out[c] = s.fillna(fill)
            # si colonne était entière, tenter de caster back
            if pd.api.types.is_integer_dtype(s.dropna()):
                out[c] = out[c].round().astype("Int64").astype("float").astype(int) if out[c].notna().all() else out[c]
        # Catégorielle/texte -> mode, sinon "Unknown"
        else:
            mode = s.mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else "Unknown"
            out[c] = s.fillna(fill).astype(str).str.strip()
    return out

# ---------- clean ----------
def clean_business_rules(df: pd.DataFrame):
    log = {}
    df = normalize_cols(df)
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
    df = parse_dates(df)

    # 1) Unicité id
    id_col = next((c for c in ["id","client_id","customer_id"] if c in df.columns), None)
    if id_col:
        before = len(df); df = df.drop_duplicates(subset=[id_col]); log["dup_id_removed"] = before-len(df)

    # 2) Années aberrantes
    yb_col = next((c for c in ["year_birth","birth_year","annee_naissance","year_of_birth"] if c in df.columns), None)
    if yb_col:
        df[yb_col] = to_num(df[yb_col])
        bad = [1893,1899,1900]
        rm = int(df[yb_col].isin(bad).sum())
        df = df.loc[~df[yb_col].isin(bad)].copy()
        log["bad_years_removed"] = rm

    # 3) Statut marital (incl. Alone->Single) + drop absurdes
    m_col = next((c for c in ["marital_status","status_marital","marital","statut_marital"] if c in df.columns), None)
    if m_col:
        df[m_col] = df[m_col].astype(str).str.strip().str.lower()
        MAP = {
            "alone":"single","single":"single","married":"married",
            "together":"together","divorced":"divorced","widow":"widow","widowed":"widow",
        }
        BAD = {"absurd","yolo"}
        before=len(df); df = df.loc[~df[m_col].isin(BAD)].copy()
        log["marital_bad_rows_removed"] = before-len(df)
        df[m_col] = df[m_col].map(MAP).fillna(df[m_col])
        df[m_col] = df[m_col].str.title()
        log["marital_modalities"] = sorted(df[m_col].dropna().unique().tolist())

    # 4) Income: 666666 -> NaN -> médiane
    if "income" in df.columns:
        df["income"] = to_num(df["income"])
        outliers = int((df["income"]==666_666).sum())
        if outliers: df.loc[df["income"]==666_666, "income"] = np.nan
        med = float(df["income"].median(skipna=True))
        df["income"] = df["income"].fillna(med)
        log["income_outliers_666666"] = outliers
        log["income_median_used"] = med

    # 5) Conversion large vers numérique quand possible
    keep_cat = {m_col or "", "education", "response"}
    for c in df.columns:
        if c not in keep_cat:
            df[c] = pd.to_numeric(df[c], errors="ignore")

    # 6) Imputation simple (première passe)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            mode = df[c].mode(dropna=True); fill = mode.iloc[0] if not mode.empty else "Unknown"
            df[c] = df[c].fillna(fill)

    # 7) Binaires texte -> 0/1
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True)==2:
            df[c] = df[c].astype(str).str.lower().map(lambda v: 1 if v in POS else 0)

    # 8) Passe stricte NO-NA (garantie)
    df = strict_impute_no_missing(df)

    # 9) Doublons globaux
    df = df.drop_duplicates()

    # 10) Garde-fou
    if df.isna().any().any():
        # fallback paranoïaque (ne devrait pas s’exécuter)
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]): df[c]=df[c].fillna(0)
            elif np.issubdtype(df[c].dtype, np.datetime64): df[c]=df[c].fillna(pd.Timestamp("1970-01-01"))
            else: df[c]=df[c].fillna("Unknown")
    assert not df.isna().any().any(), "Il reste des NA après imputation stricte."

    return df, log

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Nettoyage marketing (CSV ';') — Sortie SANS NA")
    ap.add_argument("--input", required=True, help="CSV brut")
    ap.add_argument("--out", required=True, help="Sortie .parquet (un .csv sera aussi écrit)")
    ap.add_argument("--sep", default=";", help="Séparateur CSV (défaut: ;) ")
    ap.add_argument("--encoding", default="utf-8", help="Encodage (défaut: utf-8)")
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep=args.sep, encoding=args.encoding)
    out, log = clean_business_rules(df)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_out = out_path.with_suffix(".csv")
    out.to_csv(csv_out, index=False)
    out.to_parquet(out_path, index=False)

    # mini rapport
    print(bar("RAPPORT"), f"\nSource : {args.input}\nSorties: {csv_out} | {out_path}\n")
    print("Modalités marital :", log.get("marital_modalities","—"))
    print("Income médiane    :", f"{log.get('income_median_used', float('nan')):.0f}")
    print("Doublons ID rmvd  :", log.get("dup_id_removed",0))
    print("Années supprimées :", log.get("bad_years_removed",0))
    print("\nAucun NA restant :", out.isna().sum().sum()==0)
    print(bar("FIN"))

if __name__ == "__main__":
    main()
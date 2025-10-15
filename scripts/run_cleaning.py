import argparse
from pathlib import Path
import pandas as pd
import numpy as np

POS = {"yes","oui","true","t","1","accepted","converted","purchase","purchased","buy","bought","success","y"}

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if "date" in c:
            out[c] = pd.to_datetime(out[c], errors="coerce", dayfirst=True, infer_datetime_format=True)
    return out

def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def clean_business_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df)
    df = parse_dates(df)

    # strip espaces sur textes
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

    # 1) Unicité ID si présent
    id_col = next((c for c in ["id","client_id","customer_id"] if c in df.columns), None)
    if id_col:
        before = len(df)
        df = df.drop_duplicates(subset=[id_col])
        after = len(df)
        print(f"[info] doublons supprimés sur {id_col}: {before - after}")

    # 2) Années de naissance aberrantes
    yb_col = next((c for c in ["year_birth","birth_year","annee_naissance","year_of_birth"] if c in df.columns), None)
    if yb_col:
        df[yb_col] = to_numeric_safe(df[yb_col])
        mask_bad = df[yb_col].isin([1893, 1899, 1900])
        removed = int(mask_bad.sum())
        df = df.loc[~mask_bad].copy()
        print(f"[info] années supprimées dans {yb_col}: {removed} (1893, 1899, 1900)")

    # 3) Statut marital – retirer Absurd / YOLO
    m_col = next((c for c in ["marital_status","status_marital","marital","statut_marital"] if c in df.columns), None)
    if m_col:
        bad_vals = {"absurd","yolo"}
        before = len(df)
        df[m_col] = df[m_col].astype(str).str.strip()
        df = df.loc[~df[m_col].str.lower().isin(bad_vals)].copy()
        print(f"[info] lignes statut marital supprimées: {before - len(df)} (Absurd/YOLO)")

    # 4) Income – remplacer 666666 par NA puis médiane
    if "income" in df.columns:
        df["income"] = to_numeric_safe(df["income"])
        outliers = (df["income"] == 666_666).sum()
        if outliers:
            df.loc[df["income"] == 666_666, "income"] = np.nan
            print(f"[info] income=666666 mises à NA: {outliers}")
        median_income = df["income"].median()
        df["income"] = df["income"].fillna(median_income)
        print(f"[info] imputation income médiane: {median_income:.0f}")

    # 5) Imputation générique (num -> médiane, cat -> mode)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            mode = df[c].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else "NA"
            df[c] = df[c].fillna(fill)

    # 6) Binaire texte -> 0/1
    for c in df.columns:
        if df[c].nunique(dropna=True) == 2 and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).str.lower().map(lambda v: 1 if v in POS else 0)

    # 7) Drop doublons globaux
    df = df.drop_duplicates()

    return df

def main():
    p = argparse.ArgumentParser(description="Nettoyage données marketing (règles métier)")
    p.add_argument("--input", required=True, help="CSV brut")
    p.add_argument("--out", required=True, help="Chemin .parquet (un .csv sera aussi créé)")
    args = p.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8")
    out = clean_business_rules(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # export CSV intermédiaire
    csv_out = out_path.with_suffix(".csv")
    out.to_csv(csv_out, index=False)

    # export parquet (si pyarrow installé via requirements)
    out.to_parquet(out_path, index=False)

    # petit rapport de contrôle
    print("\n=== Contrôles rapides ===")
    print(out.dtypes.head(15))
    print("shape:", out.shape)
    print("valeurs manquantes (top 10):")
    print(out.isna().sum().sort_values(ascending=False).head(10))
    print("\nOK ✅ Nettoyage terminé.")
    print("CSV nettoyé :", csv_out)
    print("Parquet     :", out_path)

if __name__ == "__main__":
    main()
import argparse
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Helpers terminal

USE_UNICODE = sys.stdout.encoding and "UTF" in sys.stdout.encoding.upper()

def _hline(width, heavy=False):
    if USE_UNICODE:
        h = "═" if heavy else "─"
        return h * width
    return "-" * width

def _box(title, lines, width=96):
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
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}".replace(",", " ")
    return f"{x:,.{decimals}f}".replace(",", " ")

def fmt_pct(x, decimals=2):
    try:
        return f"{100*float(x):.{decimals}f}%"
    except Exception:
        return "-"

def print_kv_table(title, rows, width=96, key_w=38):
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

# Lecture + Features FR (auto)

def ensure_features(df):
    """
    Assure la présence des variables d’ingénierie demandées.
    Utilise le schéma FR que tu as partagé ; si une feature est absente, on la calcule.
    """
    # Aliases FR (conformes à ton dataset)
    col = {
        "wine": "Montant_vin",
        "fruits": "Montant_fruits",
        "meat": "Montant_viande",
        "fish": "Montant_poisson",
        "sweet": "Montant_sucreries",
        "gold": "Montant_luxe",
        "income": "Revenu",
        "web_p": "Nb_achats_en_ligne",
        "cat_p": "Achats_catalogue",
        "store_p": "Achats_magasin",
        "web_visits": "Nb_visites_web_mois",
        "kid": "Enfant_charge",
        "teen": "Ado_charge",
        "yob": "Année_naissance",
        "acc1": "Accepte_Campagne_1",
        "acc2": "Accepte_Campagne_2",
        "acc3": "Accepte_Campagne_3",
        "acc4": "Accepte_Campagne_4",
        "acc5": "Accepte_Campagne_5",
        "resp": "Response",
    }

    # Conversions sûres
    for k in ["wine","fruits","meat","fish","sweet","gold","income",
              "web_p","cat_p","store_p","web_visits","kid","teen","yob",
              "acc1","acc2","acc3","acc4","acc5","resp"]:
        if col[k] in df.columns:
            df[col[k]] = pd.to_numeric(df[col[k]], errors="coerce")

    # Dépenses totales
    if "Depense_totale" not in df.columns:
        spend_cols = [col["wine"], col["fruits"], col["meat"], col["fish"], col["sweet"], col["gold"]]
        df["Depense_totale"] = df[spend_cols].sum(axis=1)

    # Dépenses plaisir
    if "Depense_plaisir" not in df.columns:
        df["Depense_plaisir"] = df[col["wine"]] + df[col["sweet"]] + df[col["gold"]]

    if "Part_plaisir" not in df.columns:
        denom = df["Depense_totale"].replace(0, np.nan)
        df["Part_plaisir"] = df["Depense_plaisir"] / denom

    if "Taux_depense_sur_revenu" not in df.columns:
        denom = df[col["income"]].replace(0, np.nan)
        df["Taux_depense_sur_revenu"] = df["Depense_totale"] / denom

    # Achats & canaux
    if "Total_achats" not in df.columns:
        df["Total_achats"] = df[col["web_p"]] + df[col["cat_p"]] + df[col["store_p"]]

    if "Part_achats_en_ligne" not in df.columns:
        denom = df["Total_achats"].replace(0, np.nan)
        df["Part_achats_en_ligne"] = df[col["web_p"]] / denom

    if "Part_achats_catalogue" not in df.columns:
        denom = df["Total_achats"].replace(0, np.nan)
        df["Part_achats_catalogue"] = df[col["cat_p"]] / denom

    if "Taux_visite_achat_web" not in df.columns:
        denom = df[col["web_visits"]].replace(0, np.nan)
        df["Taux_visite_achat_web"] = df[col["web_p"]] / denom

    # Enfants, âge
    if "Nb_enfants_total" not in df.columns:
        df["Nb_enfants_total"] = df[col["kid"]] + df[col["teen"]]

    if "Age" not in df.columns:
        # Jeu d’origine daté de 2014
        df["Age"] = 2014 - df[col["yob"]]

    # Engagement marketing
    if "Score_engagement_marketing" not in df.columns:
        acc_cols = [col["acc1"], col["acc2"], col["acc3"], col["acc4"], col["acc5"]]
        df["Score_engagement_marketing"] = df[acc_cols].sum(axis=1)

    if "Taux_acceptation_campagne" not in df.columns:
        df["Taux_acceptation_campagne"] = df["Score_engagement_marketing"] / 5.0

    # Fréquence et AOV utilitaires (peuvent aider le clustering)
    if "Freq" not in df.columns:
        df["Freq"] = df["Total_achats"]
    if "AOV" not in df.columns:
        denom = df["Freq"].replace(0, np.nan)
        df["AOV"] = df["Depense_totale"] / denom

    return df

# Sélection features de clustering 

DEFAULT_FEATURES = [
    # Montants / structure de dépense
    "Depense_totale","Depense_plaisir","Part_plaisir","Taux_depense_sur_revenu",
    # Comportements d’achat & web
    "Total_achats","Part_achats_en_ligne","Part_achats_catalogue","Taux_visite_achat_web",
    # Démographie simple
    "Nb_enfants_total","Age",
    # Engagement marketing
    "Score_engagement_marketing","Taux_acceptation_campagne",
    # Utilitaires
    "AOV"
]

SUMMARY_FEATURES = [
    "Depense_totale","Depense_plaisir","Part_plaisir","Total_achats",
    "Part_achats_en_ligne","Nb_enfants_total","Age","AOV","Taux_depense_sur_revenu",
    "Score_engagement_marketing","Taux_acceptation_campagne","Response"
]

# Règles d’étiquettes “humaines” 

def human_label(row):
    tags = []

    # Dépense
    if row.get("Depense_totale", 0) >= 600:
        tags.append("💰 Dépenses élevées")
    elif row.get("Depense_totale", 0) >= 250:
        tags.append("💵 Dépenses moyennes")
    else:
        tags.append("💳 Dépenses modestes")

    # Plaisir
    pp = row.get("Part_plaisir", np.nan)
    if not np.isnan(pp):
        if pp >= 0.6:
            tags.append("🍷 Plaisir fort")
        elif pp >= 0.35:
            tags.append("🥂 Plaisir moyen")
        else:
            tags.append("🥛 Plaisir faible")

    # Enfants
    kids = row.get("Nb_enfants_total", 0)
    if kids <= 0.5:
        tags.append("👤 Peu d'enfants")
    elif kids <= 1.5:
        tags.append("👨‍👩‍👧 1–2 enfants")
    else:
        tags.append("👨‍👩‍👧‍👦 Famille nombreuse")

    # Âge
    age = row.get("Age", 0)
    if age >= 60:
        tags.append("🧓 Senior")
    elif age >= 40:
        tags.append("🧑‍💼 Mature")
    else:
        tags.append("🧑 Jeune")

    # Canal
    po = row.get("Part_achats_en_ligne", np.nan)
    if not np.isnan(po):
        if po >= 0.6:
            tags.append("🌐 Très digital")
        elif po >= 0.35:
            tags.append("💻 Plutôt digital")
        else:
            tags.append("🏬 Plutôt magasin")

    # Engagement
    acc = row.get("Taux_acceptation_campagne", np.nan)
    if not np.isnan(acc):
        if acc >= 0.4:
            tags.append("📣 Réactif marketing")
        elif acc >= 0.2:
            tags.append("🔔 Sensible marketing")
        else:
            tags.append("🔕 Peu réactif")

    return " ; ".join(tags)

# Clustering pipeline

def run_clustering(df, features, k_min=2, k_max=10, random_state=42, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Nettoyage features -> matrice X
    use_cols = [c for c in features if c in df.columns]
    X = df[use_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    # 2) Coude (inertie) + Silhouette
    ks = list(range(max(2, k_min), max(2, k_max)+1))
    inertias, sils = [], []

    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        sil = silhouette_score(Xs, labels, metric="euclidean")
        sils.append(sil)

    # 3) Choix k — par défaut : k avec silhouette max
    best_k = int(ks[int(np.nanargmax(sils))])

    # 4) Fit final
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(Xs)

    # 5) Projection PCA 2D
    pca = PCA(n_components=2, random_state=random_state)
    Z = pca.fit_transform(Xs)

    # 6) Sauvegardes graphiques
    fig1 = plt.figure(figsize=(6.5, 4.0), dpi=140)
    ax1 = fig1.gca()
    ax1.plot(ks, inertias, marker="o")
    ax1.set_title("Méthode du coude — Inertie vs k")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertie (KMeans)")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig1.tight_layout()
    elbow_path = os.path.join(out_dir, "01_coude_inertie.png")
    fig1.savefig(elbow_path)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(6.5, 4.0), dpi=140)
    ax2 = fig2.gca()
    ax2.plot(ks, sils, marker="o")
    ax2.set_title("Méthode de la silhouette — Score moyen vs k")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Score de silhouette")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.axhline(0.50, linestyle="--", linewidth=1)
    ax2.axhline(0.25, linestyle="--", linewidth=1)
    fig2.tight_layout()
    sil_path = os.path.join(out_dir, "02_silhouette.png")
    fig2.savefig(sil_path)
    plt.close(fig2)

    fig3 = plt.figure(figsize=(6.5, 5.5), dpi=140)
    ax3 = fig3.gca()
    scatter = ax3.scatter(Z[:, 0], Z[:, 1], c=labels, s=16, alpha=0.8)
    ax3.set_title(f"Projection PCA (2D) — KMeans k={best_k}")
    ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
    ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
    # légende simple
    handles, _ = scatter.legend_elements()
    ax3.legend(handles, [f"Cluster {i}" for i in range(best_k)], title="Clusters", loc="best", frameon=True)
    fig3.tight_layout()
    proj_path = os.path.join(out_dir, "03_projection_pca.png")
    fig3.savefig(proj_path)
    plt.close(fig3)

    # 7) Résumés par cluster
    df_out = df.copy()
    df_out["cluster"] = labels

    summaries = []
    for c in range(best_k):
        sub = df_out[df_out["cluster"] == c]
        n = len(sub)
        if n == 0:
            continue

        # Moyennes des features clés
        mean_vals = sub[SUMMARY_FEATURES].mean(numeric_only=True)
        label_text = human_label(mean_vals.to_dict())

        summaries.append({
            "cluster": c,
            "taille": n,
            "Depense_totale_moy": float(mean_vals.get("Depense_totale", np.nan)),
            "AOV_moy": float(mean_vals.get("AOV", np.nan)),
            "Part_plaisir_moy": float(mean_vals.get("Part_plaisir", np.nan)),
            "Total_achats_moy": float(mean_vals.get("Total_achats", np.nan)),
            "Part_online_moy": float(mean_vals.get("Part_achats_en_ligne", np.nan)),
            "Nb_enfants_moy": float(mean_vals.get("Nb_enfants_total", np.nan)),
            "Age_moy": float(mean_vals.get("Age", np.nan)),
            "T_depense_revenu_moy": float(mean_vals.get("Taux_depense_sur_revenu", np.nan)),
            "Resp_taux": float(mean_vals.get("Response", np.nan)),
            "Engagement_mkt_score_moy": float(mean_vals.get("Score_engagement_marketing", np.nan)),
            "Engagement_mkt_taux_moy": float(mean_vals.get("Taux_acceptation_campagne", np.nan)),
            "etiket": label_text
        })

    summary_df = pd.DataFrame(summaries).sort_values("cluster").reset_index(drop=True)
    csv_path = os.path.join(out_dir, "04_resume_clusters.csv")
    summary_df.to_csv(csv_path, index=False, encoding="utf-8")

    # 8) Rendu terminal soigné
    # Header
    print(_box("CLUSTERING CLIENTS – RAPPORT", [
        f"Fichier : {args.file}",
        f"Exécuté : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Paramètres : k∈[{args.k_min},{args.k_max}] | random_state={args.random_state}",
        f"Sorties : {elbow_path}, {sil_path}, {proj_path}, {csv_path}"
    ], width=96))
    print()

    # Interprétation silhouette
    sil_best = max(sils) if len(sils) else float("nan")
    if sil_best >= 0.5:
        interpr = "→ séparation très bonne (≥ 0,50)"
    elif sil_best >= 0.25:
        interpr = "→ séparation acceptable (0,25–0,50)"
    else:
        interpr = "→ séparation faible (< 0,25), clusters possiblement artificiels"

    rows_diag = [
        ("k optimal (silhouette max)", best_k),
        ("Silhouette (k optimal)", fmt_num(sil_best, 3) if not math.isnan(sil_best) else "-"),
        ("Coude / Inertie", "Voir 01_coude_inertie.png"),
        ("Silhouette par k", "Voir 02_silhouette.png"),
        ("Projection 2D", "Voir 03_projection_pca.png"),
        ("Lecture silhouette", interpr),
    ]
    print_kv_table("DIAGNOSTIC CLUSTERING", rows_diag, width=96)

    # Résumé clusters (table)
    lines = []
    header = [
        "Cluster","Taille","Dép.tot","AOV","Part_plaisir","Achats",
        "Online","Enfants","Âge","Dép/Revenu","Resp","Eng.taux","Étiquette"
    ]
    sep = " | "
    lines.append(sep.join(header))
    lines.append("-" * len(lines[0]))
    for _, r in summary_df.iterrows():
        row = [
            int(r.cluster),
            int(r.taille),
            fmt_num(r.Depense_totale_moy, 0),
            fmt_num(r.AOV_moy, 2),
            fmt_pct(r.Part_plaisir_moy, 1),
            fmt_num(r.Total_achats_moy, 1),
            fmt_pct(r.Part_online_moy, 1),
            fmt_num(r.Nb_enfants_moy, 1),
            fmt_num(r.Age_moy, 0),
            fmt_pct(r.T_depense_revenu_moy, 1),
            fmt_pct(r.Resp_taux, 1),
            fmt_pct(r.Engagement_mkt_taux_moy, 1),
            r.etiket
        ]
        lines.append(sep.join([str(x) for x in row]))
    print(_box("RÉSUMÉ PAR CLUSTER", lines, width=96))

    return {
        "best_k": best_k,
        "silhouette_scores": dict(zip(ks, sils)),
        "inertias": dict(zip(ks, inertias)),
        "paths": {
            "elbow": elbow_path,
            "silhouette": sil_path,
            "projection": proj_path,
            "resume_csv": csv_path
        },
        "summary_df": summary_df
    }

# Main CLI 

def parse_args():
    p = argparse.ArgumentParser(description="Clustering clients (FR) — KMeans + coude + silhouette + PCA.")
    p.add_argument("--file", required=True, help="Chemin du CSV nettoyé")
    p.add_argument("--k-min", type=int, default=2, help="k minimum (>=2)")
    p.add_argument("--k-max", type=int, default=10, help="k maximum")
    p.add_argument("--random-state", type=int, default=42, help="Graine aléatoire pour reproductibilité")
    p.add_argument("--out-dir", type=str, default="outputs", help="Dossier de sortie pour graphiques et CSV")
    p.add_argument("--width", type=int, default=96, help="Largeur pour l’affichage terminal")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Lecture
    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        print(f"[ERREUR] Impossible de lire le CSV : {e}")
        sys.exit(1)

    # Features
    df = ensure_features(df)

    # Lancer le clustering
    run_clustering(
        df=df,
        features=DEFAULT_FEATURES,
        k_min=max(2, args.k_min),
        k_max=max(2, args.k_max),
        random_state=args.random_state,
        out_dir=args.out_dir
    )
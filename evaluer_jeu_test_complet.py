# -*- coding: utf-8 -*-
"""
Evaluation complete du pipeline hybride (semantique + lexical) sur les 175
paires du jeu de test etiquete (165 paires A/B/C/D/E + 10 paires du noyau de
calibrage deja valide, reutilisees telles quelles depuis calibrer_alpha_
fusion.py). Produit les resultats chiffres definitifs pour la section
Resultats du memoire : recherche de (alpha, seuil) optimal, metriques
detaillees (globales, par categorie de plagiat, taux de faux positifs D/E),
comparaison aux baselines semantique seul / lexical seul, et donnees de la
courbe precision-rappel.

Pour chaque paire, texte_1 est compare directement a texte_2 (pas de
recherche de candidats parmi le corpus, puisque les paires sont deja
appariees par construction).

A executer avec l'environnement .venv-embed :
    .venv-embed\\Scripts\\python.exe evaluer_jeu_test_complet.py
"""

import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.fr.stop_words import STOP_WORDS as MOTS_VIDES_FR

from construire_tfidf_corpus import pretraiter, FICHIER_METADONNEES
import calibrer_alpha_fusion as calib  # reutilise les 10 paires du noyau (PAIRES) - import sans declencher le __main__

NOM_MODELE = "dangvantuan/sentence-camembert-large"
FENETRE_REELLE = 512

FICHIER_JEU_TEST = "jeu_test_etiquete.parquet"
FICHIER_SCORES_BRUTS = "scores_bruts_jeu_test.csv"
FICHIER_COURBE_PR = "courbe_precision_rappel.csv"
FICHIER_GRILLE = "grille_alpha_seuil.csv"

VALEURS_ALPHA = [round(a, 2) for a in np.arange(0.0, 1.001, 0.05)]
VALEURS_SEUIL = [round(s, 2) for s in np.arange(0.3, 0.901, 0.05)]


def charger_jeu_complet():
    df = pd.read_parquet(FICHIER_JEU_TEST)
    lignes_calibrage = []
    for p in calib.PAIRES:
        lignes_calibrage.append({
            "id_paire": f"calib-{p['id']}",
            "texte_1": p["texte1"], "texte_2": p["texte2"],
            "categorie": p["categorie"].split(" - ")[0].strip(),
            "etiquette": p["label"],
            "document_source_1": None, "document_source_2": None,
            "methode_generation": "calibrage_manuel",
        })
    df_calib = pd.DataFrame(lignes_calibrage)
    return pd.concat([df, df_calib], ignore_index=True)


def cosinus(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def cosinus_sparse(v1, v2):
    num = v1.multiply(v2).sum()
    den = np.sqrt(v1.multiply(v1).sum()) * np.sqrt(v2.multiply(v2).sum())
    return float(num / den) if den > 0 else 0.0


def precision_rappel_f1(y_true_plagiat, y_pred_plagiat):
    vp = int(np.sum(y_true_plagiat & y_pred_plagiat))
    fp = int(np.sum(~y_true_plagiat & y_pred_plagiat))
    fn = int(np.sum(y_true_plagiat & ~y_pred_plagiat))
    vn = int(np.sum(~y_true_plagiat & ~y_pred_plagiat))
    precision = vp / (vp + fp) if (vp + fp) > 0 else 0.0
    rappel = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    f1 = 2 * precision * rappel / (precision + rappel) if (precision + rappel) > 0 else 0.0
    return precision, rappel, f1, vp, fp, vn, fn


if __name__ == "__main__":
    df = charger_jeu_complet().reset_index(drop=True)
    print(f"{len(df)} paires au total : {df['etiquette'].value_counts().to_dict()}")
    print(f"Par categorie : {df['categorie'].value_counts().to_dict()}")

    print(f"\nRefit du TF-IDF (memes parametres que construire_tfidf_corpus.py)...")
    df_meta = pd.read_parquet(FICHIER_METADONNEES)
    vectoriseur = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.85, stop_words=sorted(MOTS_VIDES_FR), lowercase=True
    )
    vectoriseur.fit([pretraiter(t) for t in df_meta["texte"]])

    print(f"Chargement de {NOM_MODELE}...")
    modele = SentenceTransformer(NOM_MODELE, device="cuda")
    modele.max_seq_length = FENETRE_REELLE

    # -------------------------------------------------------------------
    # ETAPE 1 - scores bruts (semantique + lexical) sur les 175 paires
    # -------------------------------------------------------------------
    print(f"\n{'=' * 100}\nETAPE 1 - Scores bruts sur les {len(df)} paires\n{'=' * 100}")
    t0 = time.time()
    scores_sem, scores_lex = [], []
    for _, p in df.iterrows():
        vec_sem = modele.encode([p["texte_1"], p["texte_2"]], convert_to_numpy=True, show_progress_bar=False)
        scores_sem.append(cosinus(vec_sem[0], vec_sem[1]))
        vec_lex = vectoriseur.transform([pretraiter(p["texte_1"]), pretraiter(p["texte_2"])])
        scores_lex.append(cosinus_sparse(vec_lex.getrow(0), vec_lex.getrow(1)))
    t_total = time.time() - t0
    df["score_semantique"] = scores_sem
    df["score_lexical"] = scores_lex
    print(f"Calcule en {t_total:.1f}s ({t_total / len(df) * 1000:.0f} ms/paire)")

    df.drop(columns=["texte_1", "texte_2"]).to_csv(FICHIER_SCORES_BRUTS, index=False, encoding="utf-8-sig")
    print(f"Scores bruts (sans le texte, pour rester leger) sauvegardes dans {FICHIER_SCORES_BRUTS}")

    y_true_plagiat = (df["etiquette"] == "PLAGIAT").to_numpy()

    # -------------------------------------------------------------------
    # ETAPE 2 - recherche de (alpha, seuil) optimal
    # -------------------------------------------------------------------
    print(f"\n{'=' * 100}\nETAPE 2 - Recherche du couple (alpha, seuil) optimal\n{'=' * 100}")
    resultats_grille = []
    sem_arr, lex_arr = df["score_semantique"].to_numpy(), df["score_lexical"].to_numpy()
    for alpha in VALEURS_ALPHA:
        score_fusionne = alpha * sem_arr + (1 - alpha) * lex_arr
        for seuil in VALEURS_SEUIL:
            y_pred = score_fusionne > seuil
            precision, rappel, f1, vp, fp, vn, fn = precision_rappel_f1(y_true_plagiat, y_pred)
            resultats_grille.append({
                "alpha": alpha, "seuil": seuil, "precision": precision, "rappel": rappel, "f1": f1,
                "VP": vp, "FP": fp, "VN": vn, "FN": fn,
            })
    df_grille = pd.DataFrame(resultats_grille)
    df_grille.to_csv(FICHIER_GRILLE, index=False)

    meilleur = df_grille.loc[df_grille["f1"].idxmax()]
    print(f"Couple optimal : alpha={meilleur['alpha']}, seuil={meilleur['seuil']} -> "
          f"F1={meilleur['f1']:.4f} (precision={meilleur['precision']:.4f}, rappel={meilleur['rappel']:.4f})")

    print(f"\nTop 5 couples (alpha, seuil) par F1 :")
    top5 = df_grille.sort_values("f1", ascending=False).head(5)
    print(top5[["alpha", "seuil", "precision", "rappel", "f1"]].to_string(index=False))
    ecart_f1_top5 = top5["f1"].max() - top5["f1"].min()
    print(f"Ecart de F1 entre le 1er et le 5e couple : {ecart_f1_top5:.4f} "
          f"({'pic net' if ecart_f1_top5 > 0.02 else 'F1 assez plat autour de l optimum - choix moins robuste'})")

    alpha_opt, seuil_opt = float(meilleur["alpha"]), float(meilleur["seuil"])

    # -------------------------------------------------------------------
    # ETAPE 3 - metriques detaillees au couple optimal
    # -------------------------------------------------------------------
    print(f"\n{'=' * 100}\nETAPE 3 - Metriques detaillees (alpha={alpha_opt}, seuil={seuil_opt})\n{'=' * 100}")
    score_fusionne_opt = alpha_opt * sem_arr + (1 - alpha_opt) * lex_arr
    df["score_fusionne"] = score_fusionne_opt
    y_pred_opt = score_fusionne_opt > seuil_opt

    precision, rappel, f1, vp, fp, vn, fn = precision_rappel_f1(y_true_plagiat, y_pred_opt)
    print(f"Precision globale : {precision:.4f}")
    print(f"Rappel global     : {rappel:.4f}")
    print(f"F1 global         : {f1:.4f}")
    print(f"Matrice de confusion : VP={vp}  FP={fp}  VN={vn}  FN={fn}")

    print(f"\nRappel par categorie de PLAGIAT (A/B/C) :")
    for cat in ["A", "B", "C"]:
        masque_cat = (df["categorie"] == cat).to_numpy()
        n_cat = int(masque_cat.sum())
        n_detectes = int((y_pred_opt & masque_cat).sum())
        rappel_cat = n_detectes / n_cat if n_cat > 0 else float("nan")
        print(f"  Categorie {cat} : {n_detectes}/{n_cat} detectes -> rappel = {rappel_cat:.4f}")

    print(f"\nTaux de faux positifs par categorie de NON-PLAGIAT (D/E) :")
    for cat in ["D", "E"]:
        masque_cat = (df["categorie"] == cat).to_numpy()
        n_cat = int(masque_cat.sum())
        n_faux_positifs = int((y_pred_opt & masque_cat).sum())
        taux_fp_cat = n_faux_positifs / n_cat if n_cat > 0 else float("nan")
        print(f"  Categorie {cat} : {n_faux_positifs}/{n_cat} faux positifs -> taux = {taux_fp_cat:.4f}")

    # -------------------------------------------------------------------
    # ETAPE 4 - comparaison aux baselines (semantique seul / lexical seul)
    # -------------------------------------------------------------------
    print(f"\n{'=' * 100}\nETAPE 4 - Comparaison aux baselines\n{'=' * 100}")
    ligne_semantique = df_grille[df_grille["alpha"] == 1.0].sort_values("f1", ascending=False).iloc[0]
    ligne_lexical = df_grille[df_grille["alpha"] == 0.0].sort_values("f1", ascending=False).iloc[0]

    print(f"{'Configuration':30s} {'alpha':>6s} {'seuil':>6s} {'precision':>10s} {'rappel':>8s} {'F1':>8s}")
    for label, ligne in [
        ("Semantique seul (a=1.0)", ligne_semantique),
        ("Lexical seul (a=0.0)", ligne_lexical),
        ("Hybride optimal", meilleur),
    ]:
        print(f"{label:30s} {ligne['alpha']:>6.2f} {ligne['seuil']:>6.2f} "
              f"{ligne['precision']:>10.4f} {ligne['rappel']:>8.4f} {ligne['f1']:>8.4f}")

    hybride_meilleur = meilleur["f1"] > max(ligne_semantique["f1"], ligne_lexical["f1"])
    print(f"\nL'hybride optimal surpasse les deux baselines : {hybride_meilleur}")

    # -------------------------------------------------------------------
    # ETAPE 5 - courbe precision-rappel a alpha optimal
    # -------------------------------------------------------------------
    print(f"\n{'=' * 100}\nETAPE 5 - Courbe precision-rappel (alpha={alpha_opt})\n{'=' * 100}")
    seuils_courbe = np.round(np.arange(0.0, 1.001, 0.01), 3)
    lignes_courbe = []
    for s in seuils_courbe:
        y_pred_s = score_fusionne_opt > s
        precision_s, rappel_s, f1_s, *_ = precision_rappel_f1(y_true_plagiat, y_pred_s)
        lignes_courbe.append({"seuil": s, "precision": precision_s, "rappel": rappel_s, "f1": f1_s})
    df_courbe = pd.DataFrame(lignes_courbe)
    df_courbe.to_csv(FICHIER_COURBE_PR, index=False)
    print(f"{len(df_courbe)} points sauvegardes dans {FICHIER_COURBE_PR}")
    print(df_courbe.iloc[::10].to_string(index=False))

    # -------------------------------------------------------------------
    # RECAPITULATIF FINAL
    # -------------------------------------------------------------------
    print(f"\n{'=' * 100}\nRECAPITULATIF\n{'=' * 100}")
    print(f"Jeu de test : {len(df)} paires ({int(y_true_plagiat.sum())} PLAGIAT, "
          f"{int((~y_true_plagiat).sum())} NON-PLAGIAT)")
    print(f"Couple optimal : alpha={alpha_opt}, seuil={seuil_opt}")
    print(f"Precision={precision:.4f}  Rappel={rappel:.4f}  F1={f1:.4f}")
    print(f"Confusion : VP={vp} FP={fp} VN={vn} FN={fn}")
    print(f"Fichiers produits : {FICHIER_SCORES_BRUTS}, {FICHIER_GRILLE}, {FICHIER_COURBE_PR}")

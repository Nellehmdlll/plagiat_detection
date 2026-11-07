# -*- coding: utf-8 -*-
"""
TF-IDF et BM25 sur les 10 paires "verite connue" de comparer_modeles_definitif.py
(5 paraphrases, 5 paires sans rapport) : score moyen par categorie, ecart,
separation et symetrie.

comparer_bm25_tfidf.py ne contient que 3 paires (2 paraphrases et 1 paire a
terme rare partage) : aucune paire sans rapport, donc ni moyenne "sans
rapport" ni ecart. Ce script comble ce manque sur le meme mini-jeu que le
comparatif des modeles d'embedding, pour une lecture directe a cote.

Scores :
  - TF-IDF : cosinus (symetrique par construction), vectoriseur ajuste sur le
    corpus de reference courant ;
  - BM25 : score brut A->B et B->A (asymetrique, non borne), et version
    normalisee dans [0, 1] = moyenne de score(A->B)/score(A->A) et
    score(B->A)/score(B->B), comme dans comparer_lexical_recent.py.

A executer avec l'environnement .venv-embed :
    .venv-embed\\Scripts\\python.exe comparer_lexical_10_paires.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.fr.stop_words import STOP_WORDS as MOTS_VIDES_FR

from comparer_bm25_tfidf import tokeniser, score_bm25_paire, K1, B
from comparer_lexical_recent import normaliser_asymetrique
from comparer_modeles_definitif import PAIRES
from construire_tfidf_corpus import pretraiter, FICHIER_METADONNEES
from evaluer_jeu_test_complet import cosinus_sparse

FICHIER_SORTIE = "comparatif_lexical_10_paires.csv"


if __name__ == "__main__":
    df_meta = pd.read_parquet(FICHIER_METADONNEES)
    print(f"Corpus de reference : {len(df_meta)} chunks, {df_meta['document'].nunique()} documents")

    vectoriseur = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.85, stop_words=sorted(MOTS_VIDES_FR), lowercase=True
    )
    vectoriseur.fit([pretraiter(t) for t in df_meta["texte"]])
    bm25 = BM25Okapi([tokeniser(t) for t in df_meta["texte"]], k1=K1, b=B)

    lignes = []
    for p in PAIRES:
        a, b = p["texte_a"], p["texte_b"]
        va, vb = vectoriseur.transform([pretraiter(a)]), vectoriseur.transform([pretraiter(b)])
        s_ab = score_bm25_paire(bm25, a, b)
        s_ba = score_bm25_paire(bm25, b, a)
        s_aa = score_bm25_paire(bm25, a, a)
        s_bb = score_bm25_paire(bm25, b, b)
        lignes.append({
            "id": p["id"], "categorie": p["categorie"],
            "tfidf_ab": cosinus_sparse(va, vb), "tfidf_ba": cosinus_sparse(vb, va),
            "bm25_ab": s_ab, "bm25_ba": s_ba,
            "bm25_norme": float(normaliser_asymetrique([s_ab], [s_aa], [s_ba], [s_bb])[0]),
        })
    df = pd.DataFrame(lignes)

    print(f"\n{'=' * 100}\nSCORES PAR PAIRE\n{'=' * 100}")
    print(f"{'Paire':>5s} {'categorie':14s} {'TF-IDF A->B':>12s} {'TF-IDF B->A':>12s} "
          f"{'BM25 A->B':>11s} {'BM25 B->A':>11s} {'BM25 norme':>11s}")
    for r in df.itertuples():
        print(f"{r.id:5d} {r.categorie:14s} {r.tfidf_ab:12.4f} {r.tfidf_ba:12.4f} "
              f"{r.bm25_ab:11.4f} {r.bm25_ba:11.4f} {r.bm25_norme:11.4f}")

    para = df[df["categorie"] == "paraphrase"]
    sans = df[df["categorie"] == "sans_rapport"]

    print(f"\n{'=' * 100}\nSYNTHESE ({len(para)} paraphrases, {len(sans)} paires sans rapport)\n{'=' * 100}")
    print(f"{'Mesure':44s} {'TF-IDF':>12s} {'BM25 brut A->B':>16s} {'BM25 brut B->A':>16s} {'BM25 normalise':>16s}")
    colonnes = ["tfidf_ab", "bm25_ab", "bm25_ba", "bm25_norme"]
    moy_p = [para[c].mean() for c in colonnes]
    moy_s = [sans[c].mean() for c in colonnes]
    print(f"{'Score moyen PARAPHRASES':44s} " + " ".join(f"{v:>{w}.4f}" for v, w in zip(moy_p, (12, 16, 16, 16))))
    print(f"{'Score moyen SANS RAPPORT':44s} " + " ".join(f"{v:>{w}.4f}" for v, w in zip(moy_s, (12, 16, 16, 16))))
    print(f"{'ECART (paraphrases - sans rapport)':44s} "
          + " ".join(f"{p - s:>{w}.4f}" for p, s, w in zip(moy_p, moy_s, (12, 16, 16, 16))))
    print(f"{'Rapport moyenne paraphrases / sans rapport':44s} "
          + " ".join(f"{(p / s if s > 0 else float('nan')):>{w}.2f}" for p, s, w in zip(moy_p, moy_s, (12, 16, 16, 16))))
    min_p = [para[c].min() for c in colonnes]
    max_s = [sans[c].max() for c in colonnes]
    print(f"{'Plus faible paraphrase':44s} " + " ".join(f"{v:>{w}.4f}" for v, w in zip(min_p, (12, 16, 16, 16))))
    print(f"{'Plus forte paire sans rapport':44s} " + " ".join(f"{v:>{w}.4f}" for v, w in zip(max_s, (12, 16, 16, 16))))
    print(f"{'Separation parfaite (min paraphrase > max sans rapport)':44s} "
          + " ".join(f"{str(a > b):>{w}s}" for a, b, w in zip(min_p, max_s, (12, 16, 16, 16))))

    print(f"\nSYMETRIE (|score(A->B) - score(B->A)| sur les {len(df)} paires)")
    ecart_tfidf = (df["tfidf_ab"] - df["tfidf_ba"]).abs()
    ecart_bm25 = (df["bm25_ab"] - df["bm25_ba"]).abs()
    rel_bm25 = ecart_bm25 / ((df["bm25_ab"] + df["bm25_ba"]) / 2).replace(0, np.nan)
    print(f"  TF-IDF : ecart moyen = {ecart_tfidf.mean():.6f}, ecart max = {ecart_tfidf.max():.6f}")
    print(f"  BM25   : ecart moyen = {ecart_bm25.mean():.4f}, ecart max = {ecart_bm25.max():.4f} ; "
          f"asymetrie relative moyenne = {rel_bm25.mean():.2%}, max = {rel_bm25.max():.2%}")

    df.to_csv(FICHIER_SORTIE, index=False, encoding="utf-8-sig")
    print(f"\nSauvegarde : {FICHIER_SORTIE}")

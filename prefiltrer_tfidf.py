# -*- coding: utf-8 -*-
"""
Etage 1 - Pre-filtrage rapide par similarite lexicale (TF-IDF).

Pour chaque chunk suspect, reduit l'espace de recherche a ses N candidats
les plus proches (au sens TF-IDF) parmi les 7443 chunks du corpus de
reference, AVANT la comparaison semantique (couteuse) de l'etage 2.

Fonctionne au niveau SEGMENT (chunk), pas document entier, pour ne pas
rater un plagiat localise (methodologie 3.1/3.3). Les chunks
legitime=True restent dans la recherche de candidats ici (decision actee
: le filtrage des zones legitimes se fait a l'etage 2/restitution, pas au
pre-filtrage, pour garder la transparence et la robustesse).

CHOIX D'IMPLEMENTATION : la matrice TF-IDF de scikit-learn est deja
normalisee en norme L2 (comportement par defaut de TfidfVectorizer), donc
le cosinus entre un chunk requete et TOUS les autres chunks du corpus se
calcule en un seul produit matrice-vecteur creux (matrice @ vecteur_
requete.T) plutot qu'en boucle chunk par chunk - c'est deja la operation
que scipy.sparse optimise nativement, sans besoin d'une structure d'index
supplementaire (type ANN/FAISS) qui serait une complexite superflue pour
7443 documents (les index approximatifs se justifient a partir de
volumes bien plus grands, ou le cout d'un produit matriciel complet
deviendrait lui-meme trop eleve).

A executer avec l'environnement .venv-embed :
    .venv-embed\\Scripts\\python.exe prefiltrer_tfidf.py
"""

import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import scipy.sparse as sp

FICHIER_METADONNEES = "metadonnees_corpus.parquet"
FICHIER_MATRICE_TFIDF = "tfidf_corpus.npz"

DOCS_REFERENCE = [
    "SIMPORE Némata Version Finale.pdf",
    "SERE MEMOIRE MASTER IBAM.docxVF corrigée.pdf",
    "08-memoire.pdf",
]

N_CANDIDATS = 50


def top_n_candidats_tfidf(indice_requete, matrice, n=N_CANDIDATS):
    """
    Renvoie (indices, scores) des n chunks les plus proches de
    matrice[indice_requete] au sens du cosinus TF-IDF, tries par score
    decroissant. matrice doit etre L2-normalisee par ligne (defaut
    sklearn) pour que le produit scalaire soit directement un cosinus.
    """
    vecteur_requete = matrice.getrow(indice_requete)
    scores = matrice.dot(vecteur_requete.T).toarray().ravel()

    n_eff = min(n, len(scores))
    indices_top = np.argpartition(-scores, n_eff - 1)[:n_eff]
    ordre = np.argsort(-scores[indices_top])
    indices_top = indices_top[ordre]
    return indices_top, scores[indices_top]


if __name__ == "__main__":
    print(f"Chargement de {FICHIER_METADONNEES} et {FICHIER_MATRICE_TFIDF}...")
    df_meta = pd.read_parquet(FICHIER_METADONNEES)
    matrice = sp.load_npz(FICHIER_MATRICE_TFIDF)
    print(f"{len(df_meta)} chunks, matrice {matrice.shape[0]}x{matrice.shape[1]} "
          f"(alignement ligne-a-ligne avec les metadonnees).")

    # Verification que les lignes sont bien L2-normalisees (condition pour
    # que le produit scalaire soit un cosinus direct).
    normes = np.sqrt(matrice.multiply(matrice).sum(axis=1)).A.ravel()
    normes_non_nulles = normes[normes > 1e-9]
    print(f"Verification normalisation L2 : normes des lignes non nulles dans "
          f"[{normes_non_nulles.min():.6f}, {normes_non_nulles.max():.6f}] (attendu ~1.0)")

    indices_reference = df_meta.index[df_meta["document"].isin(DOCS_REFERENCE)].tolist()
    print(f"\n{len(indices_reference)} chunks dans les 3 documents de reference "
          f"(SIMPORE/SERE/08-memoire) sur {len(df_meta)} au total.")

    # -------------------------------------------------------------------
    # TEST 1 - 5 chunks de test au hasard, top-50 candidats + temps
    # -------------------------------------------------------------------
    print(f"\n{'=' * 100}\nTEST 1 - 5 chunks de test au hasard (3 documents de reference)\n{'=' * 100}")
    rng = np.random.default_rng(42)
    indices_test = rng.choice(indices_reference, size=5, replace=False)

    temps_par_requete = []
    resultats_test = {}
    for idx in indices_test:
        t0 = time.time()
        indices_candidats, scores_candidats = top_n_candidats_tfidf(idx, matrice, N_CANDIDATS)
        t_requete = time.time() - t0
        temps_par_requete.append(t_requete)
        resultats_test[idx] = (indices_candidats, scores_candidats)
        chunk = df_meta.iloc[idx]
        print(f"  Chunk {chunk['chunk_id']!r} : top-{N_CANDIDATS} calcule en {t_requete * 1000:.2f} ms "
              f"(meilleur score = {scores_candidats[0]:.4f})")

    print(f"\nTemps moyen par requete : {np.mean(temps_par_requete) * 1000:.2f} ms")

    # -------------------------------------------------------------------
    # TEST 2 - un chunk cherche contre lui-meme -> rang 1, score 1.0
    # -------------------------------------------------------------------
    print(f"\n{'=' * 100}\nTEST 2 - sanite : un chunk du corpus se retrouve-t-il en rang 1 de sa propre recherche ?\n{'=' * 100}")
    for idx in indices_test:
        indices_candidats, scores_candidats = resultats_test[idx]
        rang1_est_soi_meme = indices_candidats[0] == idx
        score_rang1 = scores_candidats[0]
        ok = rang1_est_soi_meme and abs(score_rang1 - 1.0) < 1e-6
        chunk = df_meta.iloc[idx]
        print(f"  {chunk['chunk_id']!r} : rang1={'lui-meme' if rang1_est_soi_meme else 'AUTRE CHUNK'}, "
              f"score={score_rang1:.6f} -> {'OK' if ok else 'ECHEC'}")

    # -------------------------------------------------------------------
    # TEST 3 - top-10 candidats detailles pour un des 5 chunks
    # -------------------------------------------------------------------
    idx_detail = indices_test[0]
    print(f"\n{'=' * 100}\nTEST 3 - top-10 candidats detailles pour {df_meta.iloc[idx_detail]['chunk_id']!r}\n{'=' * 100}")
    chunk_requete = df_meta.iloc[idx_detail]
    print(f"Chunk requete : document={chunk_requete['document']!r}, "
          f"legitime={chunk_requete['legitime']}, extrait={chunk_requete['texte'][:100]!r}")
    indices_candidats, scores_candidats = resultats_test[idx_detail]
    print(f"\n{'Rang':>5} {'Score':>8} {'Document':45s} {'Legitime':>9} {'chunk_id'}")
    for rang, (i, s) in enumerate(zip(indices_candidats[:10], scores_candidats[:10]), 1):
        c = df_meta.iloc[i]
        print(f"{rang:>5} {s:>8.4f} {c['document'][:45]:45s} {str(c['legitime']):>9} {c['chunk_id']}")

    # -------------------------------------------------------------------
    # TEST 4 - estimation du temps total sur les 7443 chunks
    # -------------------------------------------------------------------
    print(f"\n{'=' * 100}\nTEST 4 - estimation du temps total sur les {len(df_meta)} chunks du corpus\n{'=' * 100}")
    temps_moyen = np.mean(temps_par_requete)
    temps_total_estime = temps_moyen * len(df_meta)
    print(f"Temps moyen mesure par requete : {temps_moyen * 1000:.2f} ms")
    print(f"Estimation pour {len(df_meta)} requetes (chacune contre les {len(df_meta) - 1} autres) : "
          f"{temps_total_estime:.1f}s (~{temps_total_estime / 60:.1f} min)")
    print(
        "Note : cette estimation extrapole lineairement le temps mesure sur 5 requetes ; le cout par "
        "requete est domine par le produit matrice-vecteur (O(nnz de la matrice), quasi constant quel "
        "que soit le chunk requete), donc l'extrapolation lineaire est justifiee ici - non execute sur "
        "les 7443 chunks a ce stade comme demande."
    )

    # -------------------------------------------------------------------
    # TEST 5 - presence de chunks legitime=True dans les candidats
    # -------------------------------------------------------------------
    print(f"\n{'=' * 100}\nTEST 5 - chunks legitime=True parmi les candidats (vocabulaire institutionnel partage) ?\n{'=' * 100}")
    for idx in indices_test:
        indices_candidats, scores_candidats = resultats_test[idx]
        chunk_requete = df_meta.iloc[idx]
        legitimes_dans_candidats = [
            (i, s) for i, s in zip(indices_candidats, scores_candidats)
            if bool(df_meta.iloc[i]["legitime"]) and i != idx
        ]
        print(f"\n  Requete {chunk_requete['chunk_id']!r} (legitime={chunk_requete['legitime']}) : "
              f"{len(legitimes_dans_candidats)}/{N_CANDIDATS} candidats sont legitime=True")
        for i, s in legitimes_dans_candidats[:5]:
            c = df_meta.iloc[i]
            print(f"    score={s:.4f}  {c['document'][:40]:40s}  {c['texte'][:80]!r}")

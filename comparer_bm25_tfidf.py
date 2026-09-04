# -*- coding: utf-8 -*-
"""
Comparatif lexical BM25 vs TF-IDF sur les 7443 chunks du corpus, pour
trancher lequel des deux retenir comme voie lexicale de la fusion hybride.

Reutilise le meme pretraitement deja valide pour TF-IDF (construire_tfidf_
corpus.py : normalisation NFC, suppression des residus "(cid:N)"), et le
teste sur EXACTEMENT les memes paires de sanite, pour une comparaison
directe TF-IDF / BM25 / semantique.

ADAPTATION BM25 POUR UNE COMPARAISON PAIRE-A-PAIRE (voir aussi la
verification technique n°2 affichee par ce script) :
BM25 n'est pas nativement une similarite symetrique paire-a-paire comme
un cosinus : c'est un score de pertinence d'une REQUETE contre un
document, dont la formule utilise deux statistiques ajustees sur
l'ensemble du corpus de reference (l'IDF de chaque terme, et la longueur
moyenne de document avgdl) ainsi que la longueur du document specifique
compare. On construit donc l'index BM25Okapi UNE SEULE FOIS sur les 7443
chunks (cout d'indexation, mesure ci-dessous) pour figer idf et avgdl sur
le corpus de reference - exactement comme le TfidfVectorizer est "fit" une
fois sur le corpus puis reutilise (.transform()) pour scorer des textes
hors corpus. Pour comparer un texte A a un texte B (aucun des deux n'etant
necessairement un chunk du corpus), on calcule ensuite manuellement le
score BM25 standard en reutilisant cet idf et cet avgdl figes, avec la
longueur reelle de B comme longueur de "document" - ce calcul par paire ne
retouche pas a l'index et est quasi instantane (pas de passage sur les
7443 chunks), contrairement a la construction de l'index elle-meme.

A executer avec l'environnement .venv-embed :
    .venv-embed\\Scripts\\python.exe comparer_bm25_tfidf.py
"""

import sys
import io
import re
import time
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from spacy.lang.fr.stop_words import STOP_WORDS as MOTS_VIDES_FR

from construire_tfidf_corpus import pretraiter, FICHIER_METADONNEES

K1 = 1.5
B = 0.75

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def tokeniser(texte):
    """
    Tokenisation BM25 : meme pretraitement (NFC + suppression '(cid:N)')
    et memes mots vides que pour TF-IDF, meme motif de token que le defaut
    scikit-learn. Unigrammes seulement (BM25 "Okapi" canonique et defaut de
    production - Elasticsearch, etc. - est unigramme ; TF-IDF avait en plus
    des bigrammes, difference assumee et documentee ici plutot que passee
    sous silence).
    """
    texte = pretraiter(texte).lower()
    tokens = TOKEN_PATTERN.findall(texte)
    return [t for t in tokens if t not in MOTS_VIDES_FR]


def score_bm25_paire(bm25, texte_requete, texte_document):
    """
    Score BM25 de texte_requete contre texte_document, en reutilisant
    l'idf et l'avgdl figes sur le corpus de reference (bm25), mais la
    longueur reelle de texte_document. Reproduit exactement la formule de
    BM25Okapi.get_scores (idf deja plafonne a l'ajustement, cf. rank_bm25).
    """
    tokens_requete = tokeniser(texte_requete)
    tokens_document = tokeniser(texte_document)
    freq_document = Counter(tokens_document)
    longueur_document = len(tokens_document)

    score = 0.0
    for terme in tokens_requete:
        idf = bm25.idf.get(terme, 0.0)
        if idf == 0.0:
            continue
        f = freq_document.get(terme, 0)
        if f == 0:
            continue
        score += idf * (f * (K1 + 1)) / (f + K1 * (1 - B + B * longueur_document / bm25.avgdl))
    return score


# -----------------------------------------------------------------------------
# Paires de sanite - identiques a construire_tfidf_corpus.py (memes citations
# reelles, verifiees dans le rapport Rahimah)
# -----------------------------------------------------------------------------

PAIRE_1 = {
    "id": "1 (paraphrase)",
    "a": (
        "L'objectif de l'IBAM est de répondre aux besoins du marché de l'emploi "
        "en mettant à sa disposition un potentiel humain de cadres moyens et "
        "supérieurs dans les divers secteurs d'activité."
    ),
    "b": (
        "L'IBAM a pour but de satisfaire les demandes du monde professionnel en "
        "formant des cadres, du niveau intermédiaire au niveau supérieur, dans "
        "différents domaines d'activité."
    ),
    "tfidf": 0.1000, "semantique": 0.7787,
}
PAIRE_3 = {
    "id": "3 (paraphrase)",
    "a": (
        "Ce stage a pour objectif de leur permettre de se familiariser avec le "
        "monde professionnel et d'appliquer leurs connaissances théoriques "
        "acquises au cours de leur formation."
    ),
    "b": (
        "L'objectif de ce stage est de donner aux étudiants une première "
        "expérience du milieu professionnel, tout en mettant en pratique la "
        "théorie apprise pendant leur cursus."
    ),
    "tfidf": 0.1014, "semantique": 0.9013,
}
PAIRE_TERME_RARE = {
    "id": "terme rare partage",
    "a": "L'organisation de l'IBAM est une structure hiérarchico-fonctionnelle.",
    "b": (
        "Contrairement à d'autres établissements, l'IBAM a fait le choix d'une "
        "structure hiérarchico-fonctionnelle pour son organisation interne."
    ),
    "tfidf": 0.6602, "semantique": None,
}

PAIRES = [PAIRE_1, PAIRE_3, PAIRE_TERME_RARE]


def etape_tableau_comparatif(bm25):
    print(f"\n{'=' * 100}\nTABLEAU COMPARATIF - TF-IDF (deja connu) / BM25 (nouveau) / Semantique (deja connu)\n{'=' * 100}")
    print(f"{'Paire':30s} {'TF-IDF':>10s} {'BM25 (A->B)':>13s} {'BM25 (B->A)':>13s} {'Semantique':>12s}")
    for p in PAIRES:
        score_ab = score_bm25_paire(bm25, p["a"], p["b"])
        score_ba = score_bm25_paire(bm25, p["b"], p["a"])
        sem = f"{p['semantique']:.4f}" if p["semantique"] is not None else "n/a"
        tfidf = f"{p['tfidf']:.4f}" if p["tfidf"] is not None else "n/a"
        print(f"{p['id']:30s} {tfidf:>10s} {score_ab:>13.4f} {score_ba:>13.4f} {sem:>12s}")
    print(
        "\nNote lecture : le score BM25 n'est pas borne a [0,1] comme un cosinus (c'est un "
        "score de pertinence, pas une similarite normalisee) - seul l'ORDRE RELATIF entre "
        "paires (paraphrase vs terme rare partage) est directement comparable au sein de la "
        "colonne BM25, pas la valeur absolue face a TF-IDF/semantique. BM25(A->B) et BM25(B->A) "
        "different legerement car chacun utilise la longueur reelle du texte scoré comme "
        "'document', ce qui est le comportement attendu (voir cas 4 ci-dessous)."
    )
    return None


def etape_cas_longueur_differente(df_meta, bm25):
    print(f"\n{'=' * 100}\nCAS 4 - deux chunks reels partageant un terme rare, longueurs tres differentes\n{'=' * 100}")

    print("Recherche d'un terme rare (present dans 2 a 4 chunks du corpus) associe a "
          "deux chunks de longueurs tres differentes...")

    tokens_par_chunk = [tokeniser(t) for t in df_meta["texte"]]
    longueurs = np.array([len(t) for t in tokens_par_chunk])

    freq_doc_terme = Counter()
    chunks_par_terme = {}
    for i, tokens in enumerate(tokens_par_chunk):
        for terme in set(tokens):
            freq_doc_terme[terme] += 1
            chunks_par_terme.setdefault(terme, []).append(i)

    meilleur = None
    for terme, freq in freq_doc_terme.items():
        if freq < 2 or freq > 4:
            continue
        if len(terme) < 6:  # eviter les termes rares "par accident" trop courts/generiques
            continue
        indices = chunks_par_terme[terme]
        i_court = min(indices, key=lambda i: longueurs[i])
        i_long = max(indices, key=lambda i: longueurs[i])
        if i_court == i_long or longueurs[i_court] == 0:
            continue
        ratio = longueurs[i_long] / longueurs[i_court]
        if meilleur is None or ratio > meilleur[0]:
            meilleur = (ratio, terme, i_court, i_long)

    if meilleur is None:
        print("Aucun terme rare avec ecart de longueur exploitable trouve.")
        return

    ratio, terme, i_court, i_long = meilleur
    c_court, c_long = df_meta.iloc[i_court], df_meta.iloc[i_long]
    print(f"Terme retenu : {terme!r} (present dans {freq_doc_terme[terme]} chunks du corpus)")
    print(f"  Chunk court : {c_court['chunk_id']!r} - {longueurs[i_court]} tokens "
          f"({c_court['document']})")
    print(f"    extrait : {c_court['texte'][:150]!r}")
    print(f"  Chunk long  : {c_long['chunk_id']!r} - {longueurs[i_long]} tokens "
          f"({c_long['document']})")
    print(f"    extrait : {c_long['texte'][:150]!r}")
    print(f"  Ratio de longueur : {ratio:.1f}x")

    score_court_vers_long = score_bm25_paire(bm25, c_court["texte"], c_long["texte"])
    score_long_vers_court = score_bm25_paire(bm25, c_long["texte"], c_court["texte"])

    # Comparaison TF-IDF (cosinus) sur la meme paire, en reconstruisant un
    # petit vectoriseur equivalent pour rester autonome de ce script.
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectoriseur_local = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.85, stop_words=sorted(MOTS_VIDES_FR), lowercase=True
    )
    matrice_corpus = vectoriseur_local.fit_transform([pretraiter(t) for t in df_meta["texte"]])
    v_court = matrice_corpus.getrow(i_court)
    v_long = matrice_corpus.getrow(i_long)
    num = v_court.multiply(v_long).sum()
    den = np.sqrt(v_court.multiply(v_court).sum()) * np.sqrt(v_long.multiply(v_long).sum())
    cos_tfidf = float(num / den) if den > 0 else 0.0

    print(f"\n  TF-IDF cosinus (court, long)            : {cos_tfidf:.4f}")
    print(f"  BM25 (requete=court, document=long)     : {score_court_vers_long:.4f}")
    print(f"  BM25 (requete=long, document=court)     : {score_long_vers_court:.4f}")
    print(
        "\n  Lecture : le cosinus TF-IDF est deja normalise en norme L2 (chaque vecteur est "
        "ramene a une longueur 1 avant comparaison), ce qui atténue deja une partie du "
        "biais de longueur brut - BM25 apporte un mecanisme different et ajustable (le "
        "parametre b, ici 0.75) de saturation/normalisation par longueur relative a la "
        "moyenne du corpus (avgdl), avec une courbe de saturation de frequence de terme "
        "(k1) que le cosinus TF-IDF n'a pas. La difference observee ci-dessus (s'il y en a "
        "une) vient de ces deux mecanismes, pas d'un simple defaut de TF-IDF sur la longueur "
        "brute."
    )


if __name__ == "__main__":
    print(f"Chargement de {FICHIER_METADONNEES}...")
    df_meta = pd.read_parquet(FICHIER_METADONNEES)
    print(f"{len(df_meta)} chunks charges.")

    print(f"\n{'=' * 100}\nCONSTRUCTION DE L'INDEX BM25Okapi (k1={K1}, b={B})\n{'=' * 100}")
    t0 = time.time()
    tokens_corpus = [tokeniser(t) for t in df_meta["texte"]]
    t_tokenisation = time.time() - t0

    t0 = time.time()
    bm25 = BM25Okapi(tokens_corpus, k1=K1, b=B)
    t_indexation = time.time() - t0

    print(f"Tokenisation des {len(df_meta)} chunks : {t_tokenisation:.2f}s")
    print(f"Construction de l'index BM25Okapi (idf + avgdl figes sur le corpus) : {t_indexation:.2f}s")
    print(f"avgdl (longueur moyenne de chunk, en tokens) : {bm25.avgdl:.1f}")

    # -------------------------------------------------------------------
    # VERIFICATION 1 - aucun chunk avec 0 token exploitable (equivalent de
    # "ligne nulle" pour TF-IDF : un chunk sans aucun token ne peut jamais
    # etre retrouve ni scorer quoi que ce soit en BM25).
    # -------------------------------------------------------------------
    print(f"\n{'=' * 100}\nVERIFICATIONS\n{'=' * 100}")
    longueurs_tokens = [len(t) for t in tokens_corpus]
    chunks_vides = [i for i, n in enumerate(longueurs_tokens) if n == 0]
    print(f"1. Chunks sans aucun token exploitable apres pretraitement/mots-vides : "
          f"{len(chunks_vides)} / {len(df_meta)}")
    if chunks_vides:
        for i in chunks_vides[:10]:
            print(f"   [{i}] {df_meta.iloc[i]['chunk_id']!r}")
    else:
        print("   Aucun -> OK")

    # -------------------------------------------------------------------
    # VERIFICATION 2 - temps d'indexation vs temps de requete par paire
    # -------------------------------------------------------------------
    t0 = time.time()
    _ = score_bm25_paire(bm25, PAIRE_1["a"], PAIRE_1["b"])
    t_requete_paire = time.time() - t0
    print(f"\n2. Temps de construction de l'index (une fois, sur les {len(df_meta)} chunks) : "
          f"{t_tokenisation + t_indexation:.2f}s")
    print(f"   Temps pour scorer UNE paire (requete vs document, apres index construit) : "
          f"{t_requete_paire * 1000:.3f} ms")
    print(
        "   Explication : contrairement a la matrice TF-IDF (precalculee une fois pour "
        "toutes les paires possibles via un produit scalaire entre lignes), le score BM25 "
        "d'une paire est recalcule a la demande a partir de l'idf/avgdl figes a "
        "l'indexation - d'ou un cout par requete quasi nul ici (2 courts textes, pas un "
        "passage sur les 7443 chunks) mais qui grandirait avec la taille du texte requete "
        "si on scorait une requete contre le corpus entier via get_scores()."
    )

    etape_tableau_comparatif(bm25)
    etape_cas_longueur_differente(df_meta, bm25)

# -*- coding: utf-8 -*-
"""
Assemblage complet - pipeline de detection de bout en bout sur un document.

Combine l'etage 1 (prefiltrer_tfidf.py : top-50 candidats par similarite
TF-IDF) et l'etage 2 (vecteurs semantiques camembert-large + fusion
score = alpha*semantique + (1-alpha)*lexical, alpha=0.6 retenu comme
valeur de travail provisoire - voir calibrer_alpha_fusion.py) en une
fonction unique detecter_plagiat(chemin_document_suspect, ...).

Pour chaque chunk du document suspect, la fonction :
  1. le segmente avec le pipeline deja valide (extraction -> sommaire ->
     hierarchie -> localisation -> decoupage token-based) ;
  2. vectorise le chunk en TF-IDF avec le vectoriseur DEJA FIT sur le
     corpus (memes stats de rarete que construire_tfidf_corpus.py), et en
     semantique avec sentence-camembert-large (max_seq_length=512) ;
  3. calcule le cosinus lexical contre tout le corpus de reference en un
     seul produit matrice-vecteur creux (comme prefiltrer_tfidf.py),
     retient les 50 meilleurs, EN EXCLUANT les chunks dont le champ
     'document' correspond au document suspect lui-meme (cas d'un
     document deja present dans le corpus de reference) ;
  4. calcule le score semantique SEULEMENT sur ces 50 candidats, en
     reutilisant les vecteurs deja encodes du corpus (vecteurs_corpus.npy)
     - pas de reencodage du corpus ;
  5. retient le meilleur candidat (score fusionne le plus eleve) par
     chunk suspect, avec le detail des scores intermediaires et le statut
     legitime des deux cotes (transparence, cf. 3.4).

Le taux global (v1, provisoire - seuil 0.75 a affiner) porte uniquement
sur les chunks NON-legitimes du document suspect : proportion de ces
chunks dont le meilleur score fusionne depasse le seuil. Les chunks
legitimes du suspect restent visibles dans le rapport detaille mais sont
exclus du taux (coherent avec 3.5).

Ne code pas d'interface utilisateur ni de format de rapport definitif
(Word/PDF) - un DataFrame suffit a ce stade.

A executer avec l'environnement .venv-embed :
    .venv-embed\\Scripts\\python.exe detecter_plagiat.py
"""

import sys
import io
import os
import time

if hasattr(sys.stdout, "reconfigure"):  # absent dans un notebook (Colab/Jupyter) - evite aussi le double-enveloppement a l'import
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.fr.stop_words import STOP_WORDS as MOTS_VIDES_FR
from transformers import AutoTokenizer

from inspection import extraire_texte, detecter_sommaire, detecter_lignes_repete
from extraire_titre_sommaire import extraire_titres_sommaire, filtrer_bandeau
from localiser_titres_corps import localiser_titres_dans_corps
from decouper_segments import decouper_en_segments, compteur_taille_tokenizer
from construire_tfidf_corpus import pretraiter, FICHIER_METADONNEES

CORPUS = os.environ.get("CORPUS_DIR", r"C:\Users\1\Documents\MASTER_SOUTENANCE\corpus")
NOM_MODELE_SEMANTIQUE = "dangvantuan/sentence-camembert-large"
FENETRE_REELLE = 512
TAILLE_MAX_CHUNK_TOKENS = 400
CHEVAUCHEMENT_TOKENS = 40

FICHIER_VECTEURS_CORPUS = "vecteurs_corpus.npy"
FICHIER_MATRICE_TFIDF = "tfidf_corpus.npz"

ALPHA = 0.6  # valeur de travail retenue - voir calibrer_alpha_fusion.py
N_CANDIDATS = 50
SEUIL_SIMILARITE = 0.75  # valeur de DEPART, PAS definitive - a affiner


def segmenter_document(chemin_pdf, compteur_taille):
    nom = os.path.basename(chemin_pdf)
    texte_complet = extraire_texte(chemin_pdf)
    present, nb, bloc = detecter_sommaire(texte_complet)
    if not present:
        return []
    bandeau_present, liste_bandeaux, _ = detecter_lignes_repete(texte_complet)
    bloc_f = filtrer_bandeau(bloc, liste_bandeaux) if bandeau_present else bloc
    titres = extraire_titres_sommaire(bloc_f)
    titres = localiser_titres_dans_corps(texte_complet, titres)
    chunks = decouper_en_segments(
        texte_complet, titres,
        taille_max_chunk=TAILLE_MAX_CHUNK_TOKENS,
        chevauchement=CHEVAUCHEMENT_TOKENS,
        compteur_taille=compteur_taille,
        liste_bandeaux=liste_bandeaux,
    )
    compteur_section = 0
    for c in chunks:
        if c["numero_chunk_dans_section"] == 0:
            compteur_section += 1
        c["document"] = nom
        c["chunk_id"] = f"{nom}::section{compteur_section:03d}::{c['numero_chunk_dans_section']}"
    return chunks


def charger_ressources():
    """
    Charge les artefacts du corpus de reference et les modeles necessaires a
    detecter_plagiat(). Factorise hors de __main__ pour etre reutilisee par
    une interface (chargement une seule fois, mis en cache) sans dupliquer
    cette logique.
    """
    df_meta_corpus = pd.read_parquet(FICHIER_METADONNEES)
    matrice_tfidf_corpus = sp.load_npz(FICHIER_MATRICE_TFIDF)
    vecteurs_corpus = np.load(FICHIER_VECTEURS_CORPUS)

    vectoriseur = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.85, stop_words=sorted(MOTS_VIDES_FR), lowercase=True
    )
    vectoriseur.fit([pretraiter(t) for t in df_meta_corpus["texte"]])

    modele = SentenceTransformer(NOM_MODELE_SEMANTIQUE, device="cuda")
    modele.max_seq_length = FENETRE_REELLE

    tokenizer = AutoTokenizer.from_pretrained(NOM_MODELE_SEMANTIQUE)
    compteur_taille = compteur_taille_tokenizer(tokenizer)

    return df_meta_corpus, matrice_tfidf_corpus, vecteurs_corpus, vectoriseur, modele, compteur_taille


def detecter_plagiat(chemin_document_suspect, df_meta_corpus, matrice_tfidf_corpus,
                      vecteurs_corpus, vectoriseur, modele, compteur_taille,
                      alpha=ALPHA, n_candidats=N_CANDIDATS, seuil=SEUIL_SIMILARITE):
    nom_suspect = os.path.basename(chemin_document_suspect)
    print(f"Segmentation de {nom_suspect}...")
    chunks_suspect = segmenter_document(chemin_document_suspect, compteur_taille)
    print(f"  {len(chunks_suspect)} chunks obtenus.")
    if not chunks_suspect:
        raise ValueError(
            f"Aucune structure de sommaire detectee dans {nom_suspect} - "
            f"impossible de segmenter ce document."
        )

    masque_exclusion = (df_meta_corpus["document"] == nom_suspect).to_numpy()
    if masque_exclusion.any():
        print(f"  {nom_suspect} deja present dans le corpus de reference "
              f"({masque_exclusion.sum()} chunks exclus de la recherche de candidats).")

    textes_pretraites = [pretraiter(c["texte"]) for c in chunks_suspect]
    vecteurs_tfidf_suspect = vectoriseur.transform(textes_pretraites)
    vecteurs_sem_suspect = modele.encode(
        [c["texte"] for c in chunks_suspect], convert_to_numpy=True, show_progress_bar=False
    )

    lignes_rapport = []
    for i, chunk in enumerate(chunks_suspect):
        vecteur_requete_tfidf = vecteurs_tfidf_suspect.getrow(i)
        scores_lexical = matrice_tfidf_corpus.dot(vecteur_requete_tfidf.T).toarray().ravel()
        scores_lexical[masque_exclusion] = -1.0

        n_eff = min(n_candidats, int((scores_lexical > -1.0).sum()))
        indices_top = np.argpartition(-scores_lexical, n_eff - 1)[:n_eff]
        indices_top = indices_top[np.argsort(-scores_lexical[indices_top])]

        vecteur_sem_requete = vecteurs_sem_suspect[i]
        vecteurs_sem_candidats = vecteurs_corpus[indices_top]
        normes_candidats = np.linalg.norm(vecteurs_sem_candidats, axis=1)
        norme_requete = np.linalg.norm(vecteur_sem_requete)
        scores_semantique = (vecteurs_sem_candidats @ vecteur_sem_requete) / (normes_candidats * norme_requete)

        scores_lexical_candidats = scores_lexical[indices_top]
        scores_fusionnes = alpha * scores_semantique + (1 - alpha) * scores_lexical_candidats

        meilleur_local = int(np.argmax(scores_fusionnes))
        idx_meilleur = indices_top[meilleur_local]
        candidat = df_meta_corpus.iloc[idx_meilleur]

        lignes_rapport.append({
            "chunk_suspect": chunk["chunk_id"],
            "texte_suspect": chunk["texte"],
            "extrait_suspect": chunk["texte"][:150].replace("\n", " "),
            "legitime_suspect": bool(chunk["legitime"]) if chunk["legitime"] is not None else False,
            "document_candidat": candidat["document"],
            "section_candidat": candidat["chunk_id"],
            "texte_candidat": candidat["texte"],
            "extrait_candidat": candidat["texte"][:150].replace("\n", " "),
            "legitime_candidat": bool(candidat["legitime"]) if candidat["legitime"] is not None else False,
            "score_fusionne": float(scores_fusionnes[meilleur_local]),
            "score_semantique": float(scores_semantique[meilleur_local]),
            "score_lexical": float(scores_lexical_candidats[meilleur_local]),
        })

    df_rapport = pd.DataFrame(lignes_rapport).sort_values(
        "score_fusionne", ascending=False
    ).reset_index(drop=True)

    non_legitimes = df_rapport[~df_rapport["legitime_suspect"]]
    if len(non_legitimes) > 0:
        taux_global = float((non_legitimes["score_fusionne"] > seuil).mean())
    else:
        taux_global = None

    return df_rapport, taux_global


if __name__ == "__main__":
    print(f"Chargement des ressources (corpus de reference + modeles)...")
    df_meta_corpus, matrice_tfidf_corpus, vecteurs_corpus, vectoriseur, modele, compteur_taille = charger_ressources()
    print(f"  {len(df_meta_corpus)} chunks, matrice TF-IDF {matrice_tfidf_corpus.shape}, "
          f"vecteurs semantiques {vecteurs_corpus.shape}")

    # -------------------------------------------------------------------
    # TEST - SIMPORE comme document suspect (deja dans le corpus)
    # -------------------------------------------------------------------
    chemin_simpore = os.path.join(CORPUS, "SIMPORE Némata Version Finale.pdf")

    print(f"\n{'=' * 100}\nDETECTION SUR SIMPORE (document deja dans le corpus de reference)\n{'=' * 100}")
    t0 = time.time()
    df_rapport, taux_global = detecter_plagiat(
        chemin_simpore, df_meta_corpus, matrice_tfidf_corpus, vecteurs_corpus, vectoriseur, modele, compteur_taille
    )
    t_total = time.time() - t0

    print(f"\n{'=' * 100}\nRAPPORT\n{'=' * 100}")
    print(f"Seuil de similarite (provisoire) : {SEUIL_SIMILARITE}")
    print(f"Chunks totaux : {len(df_rapport)} "
          f"(dont {(~df_rapport['legitime_suspect']).sum()} non-legitimes, "
          f"{df_rapport['legitime_suspect'].sum()} legitimes)")
    if taux_global is not None:
        print(f"TAUX GLOBAL (sur chunks non-legitimes uniquement) : {taux_global:.1%}")
    else:
        print("TAUX GLOBAL : n/a (aucun chunk non-legitime)")

    print(f"\nTemps total de bout en bout ({len(df_rapport)} chunks) : {t_total:.1f}s "
          f"({t_total / len(df_rapport) * 1000:.0f} ms/chunk)")

    print(f"\n{'=' * 100}\nTOP 15 CHUNKS LES PLUS SUSPECTS\n{'=' * 100}")
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 220)
    colonnes_affichage = [
        "chunk_suspect", "legitime_suspect", "extrait_suspect",
        "document_candidat", "extrait_candidat", "legitime_candidat",
        "score_fusionne", "score_semantique", "score_lexical",
    ]
    print(df_rapport[colonnes_affichage].head(15).to_string(index=False))

    FICHIER_RAPPORT = "rapport_detection_simpore.csv"
    df_rapport.to_csv(FICHIER_RAPPORT, index=False, encoding="utf-8-sig")
    print(f"\nRapport complet sauvegarde dans {FICHIER_RAPPORT}")

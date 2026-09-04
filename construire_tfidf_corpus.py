# -*- coding: utf-8 -*-
"""
Construction de la voie lexicale (TF-IDF) sur les 7443 chunks du corpus,
en complement de la voie semantique deja construite (vecteurs_corpus.npy).

IDF calcule sur l'ensemble du corpus de reference (les 7443 chunks), pas
document par document, conformement a la methodologie retenue.

PRETRAITEMENT AVANT VECTORISATION (diagnostic prealable sur le corpus reel,
voir le detail dans le rapport affiche par ce script) :
  - Normalisation Unicode NFC : 1/7443 chunks seulement differait de sa
    forme NFC (accents composes vs precomposes) - negligeable mais corrige
    par securite, cout nul.
  - Suppression des residus "(cid:N)" : 57/7443 chunks en contiennent. Ce
    sont des glyphes de polices a encodage personnalise (Identity-H) que
    pdfplumber n'a pas pu mapper vers un caractere reel (essentiellement
    des puces de liste). Ce ne sont pas des mots : les laisser polluerait
    le vocabulaire TF-IDF avec des tokens "cid" sans signification.
  - 2/7443 chunks contiennent le caractere de remplacement U+FFFD (residu
    de symboles mathematiques non extraits dans des formules). Trop rare
    et non recuperable a ce stade sans ameliorer l'extraction PDF en amont
    (hors perimetre de ce script) : laisse tel quel, sklearn l'ignorera
    simplement (ne fait partie d'aucun token alphanumerique).

SORTIES :
  - tfidf_corpus.npz : matrice sparse scipy (scipy.sparse.save_npz),
    7443 lignes dans le MEME ORDRE que metadonnees_corpus.parquet et
    vecteurs_corpus.npy (alignement direct par indice de ligne).
  - vocabulaire_tfidf.txt : un terme par ligne, dans l'ordre des colonnes
    de la matrice (index i du fichier = colonne i de la matrice).
  - correspondance_tfidf.parquet : chunk_id + ligne (0..7442), coherent
    avec l'ordre de metadonnees_corpus.parquet.

Ne construit PAS la fusion ponderee dense+lexical (etape suivante).

A executer avec l'environnement .venv-embed :
    .venv-embed\\Scripts\\python.exe construire_tfidf_corpus.py
"""

import sys
import io
import re
import unicodedata

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.fr.stop_words import STOP_WORDS as MOTS_VIDES_FR

FICHIER_METADONNEES = "metadonnees_corpus.parquet"
FICHIER_MATRICE = "tfidf_corpus.npz"
FICHIER_VOCABULAIRE = "vocabulaire_tfidf.txt"
FICHIER_CORRESPONDANCE = "correspondance_tfidf.parquet"

MOTIF_CID = re.compile(r"\(cid:\d+\)")


def pretraiter(texte):
    texte = unicodedata.normalize("NFC", texte)
    texte = MOTIF_CID.sub(" ", texte)
    return texte


def diagnostiquer_pretraitement(textes):
    print(f"{'=' * 90}\nDIAGNOSTIC PRETRAITEMENT (avant vectorisation)\n{'=' * 90}")
    nb_non_nfc = sum(1 for t in textes if unicodedata.normalize("NFC", t) != t)
    nb_cid = sum(1 for t in textes if MOTIF_CID.search(t))
    nb_fffd = sum(1 for t in textes if "�" in t)
    print(f"Chunks non-NFC (accents composes) : {nb_non_nfc}/{len(textes)} -> normalises")
    print(f"Chunks avec residu '(cid:N)'       : {nb_cid}/{len(textes)} -> supprimes")
    print(f"Chunks avec U+FFFD (non recuperable) : {nb_fffd}/{len(textes)} -> laisses tels quels")


def cosinus_tfidf(matrice, i, j):
    v1 = matrice.getrow(i)
    v2 = matrice.getrow(j)
    num = v1.multiply(v2).sum()
    den = np.sqrt(v1.multiply(v1).sum()) * np.sqrt(v2.multiply(v2).sum())
    return float(num / den) if den > 0 else 0.0


# -----------------------------------------------------------------------------
# Paires de sanite (mêmes citations reelles que comparer_modeles_definitif.py,
# extraites du rapport Rahimah/IBAM/Sira Labs, verifiees dans le PDF source)
# -----------------------------------------------------------------------------

PAIRES_SANITE = [
    {
        "id": "1 (paraphrase, mots differents)",
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
    },
    {
        "id": "3 (paraphrase, mots differents)",
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
    },
    {
        "id": "nouvelle - terme rare partage (\"hiérarchico-fonctionnelle\")",
        "a": "L'organisation de l'IBAM est une structure hiérarchico-fonctionnelle.",
        "b": (
            "Contrairement à d'autres établissements, l'IBAM a fait le choix d'une "
            "structure hiérarchico-fonctionnelle pour son organisation interne."
        ),
    },
]


def etape_sanite(vectoriseur):
    print(f"\n{'=' * 90}\nTEST DE SANITE - similarite cosinus TF-IDF\n{'=' * 90}")
    for p in PAIRES_SANITE:
        vecs = vectoriseur.transform([pretraiter(p["a"]), pretraiter(p["b"])])
        sim = cosinus_tfidf(vecs, 0, 1)
        print(f"  Paire {p['id']:55s} : cos TF-IDF = {sim:.4f}")
    print(
        "\n  Attendu : les paires 1 et 3 (vraies paraphrases, vocabulaire "
        "largement different) devraient scorer bas en TF-IDF - a comparer aux "
        "0.7787 et 0.9013 obtenus en semantique (sentence-camembert-large) sur "
        "ces memes paires dans comparer_modeles_definitif.py. La 3e paire "
        "partage le terme rare et specifique 'hiérarchico-fonctionnelle' malgre "
        "une structure de phrase differente : le TF-IDF devrait la faire "
        "remonter nettement au-dessus des deux premieres, illustrant sa force "
        "sur la reprise de vocabulaire specifique plutot que sur le sens."
    )


if __name__ == "__main__":
    print(f"Chargement de {FICHIER_METADONNEES}...")
    df_meta = pd.read_parquet(FICHIER_METADONNEES)
    print(f"{len(df_meta)} chunks charges (ordre = reference pour l'alignement avec "
          f"vecteurs_corpus.npy).")

    diagnostiquer_pretraitement(df_meta["texte"].tolist())
    textes_pretraites = [pretraiter(t) for t in df_meta["texte"]]

    print(f"\n{'=' * 90}\nVECTORISATION TF-IDF\n{'=' * 90}")
    print(f"Parametres : ngram_range=(1,2), min_df=2, max_df=0.85, "
          f"stop_words=francais ({len(MOTS_VIDES_FR)} mots, spaCy fr)")

    vectoriseur = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        stop_words=sorted(MOTS_VIDES_FR),
        lowercase=True,
    )
    matrice = vectoriseur.fit_transform(textes_pretraites)
    vocabulaire = vectoriseur.get_feature_names_out()

    print(f"Matrice TF-IDF : {matrice.shape[0]} lignes x {matrice.shape[1]} colonnes "
          f"(sparse, {matrice.nnz} valeurs non nulles, densite {matrice.nnz / (matrice.shape[0] * matrice.shape[1]):.4%})")

    # --- Sauvegarde des sorties ---
    sp.save_npz(FICHIER_MATRICE, matrice)
    with open(FICHIER_VOCABULAIRE, "w", encoding="utf-8") as f:
        f.write("\n".join(vocabulaire))
    df_correspondance = pd.DataFrame({
        "chunk_id": df_meta["chunk_id"],
        "ligne": np.arange(len(df_meta)),
    })
    df_correspondance.to_parquet(FICHIER_CORRESPONDANCE, index=False)

    print(f"\nSauvegarde : {FICHIER_MATRICE}, {FICHIER_VOCABULAIRE}, {FICHIER_CORRESPONDANCE}")

    # -------------------------------------------------------------------
    # VERIFICATIONS
    # -------------------------------------------------------------------
    print(f"\n{'=' * 90}\nVERIFICATIONS\n{'=' * 90}")

    # 1. Taille du vocabulaire
    print(f"1. Taille du vocabulaire final : {len(vocabulaire)} termes "
          f"(unigrammes + bigrammes, apres filtres min_df=2/max_df=0.85/mots-vides)")
    nb_unigrammes = sum(1 for t in vocabulaire if " " not in t)
    nb_bigrammes = len(vocabulaire) - nb_unigrammes
    print(f"   dont {nb_unigrammes} unigrammes, {nb_bigrammes} bigrammes")

    # 2. Test de sanite
    etape_sanite(vectoriseur)

    # 3. Lignes nulles
    nb_non_nuls_par_ligne = matrice.getnnz(axis=1)
    lignes_nulles = np.where(nb_non_nuls_par_ligne == 0)[0]
    print(f"\n3. Lignes completement nulles (chunk sans aucun terme retenu) : "
          f"{len(lignes_nulles)} / {matrice.shape[0]}")
    if len(lignes_nulles) > 0:
        print(f"   chunk_id concernes (premiers 10) :")
        for i in lignes_nulles[:10]:
            texte_court = df_meta.iloc[i]["texte"][:80].replace("\n", " ")
            print(f"     [{i}] {df_meta.iloc[i]['chunk_id']!r} : {texte_court!r}...")
    else:
        print("   Aucune -> OK")

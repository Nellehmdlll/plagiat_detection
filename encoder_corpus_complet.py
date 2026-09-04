# -*- coding: utf-8 -*-
"""
Encodage complet des 7443 chunks des 63 documents avec le modele choisi
(sentence-camembert-large, valide face a BGE-M3 dans comparer_modeles_definitif.py).

FORMAT DE STOCKAGE :
  - vecteurs_corpus.npy : tableau numpy 2D (N, 1024) float32, une ligne par
    chunk. Choisi plutot qu'un .npz par chunk_id (trop de petits tableaux,
    lent a l'ecriture/lecture pour 7443 elements) : un tableau 2D contigu se
    charge et se tranche efficacement avec numpy/faiss pour les etapes
    suivantes (pre-filtrage, comparaison fine).
  - metadonnees_corpus.parquet : une ligne par chunk, alignee ligne-a-ligne
    avec vecteurs_corpus.npy (la ligne i du parquet correspond a la ligne i
    du npy - invariant verifie explicitement a la fin du script). Parquet
    plutot que CSV : colonnes typees (booleen pour legitime, entiers pour
    niveau/positions) preservees sans reparsing, pas de risque d'echappement
    avec le texte des chunks (retours a la ligne, guillemets, virgules), et
    lecture partielle par colonne pour les etapes suivantes qui n'auront pas
    toujours besoin du texte complet.

SAUVEGARDE PROGRESSIVE / REPRISE :
  - vecteurs_corpus.npy est un memmap numpy pre-alloue a la taille finale
    (N, 1024), rempli document par document. Un flush() est fait apres
    chaque document encode, donc les donnees sont sur disque au fur et a
    mesure (pas seulement a la toute fin).
  - checkpoint_encodage.json liste les documents deja completement encodes.
    Si le script est interrompu puis relance, les documents deja marques
    termines sont sautes (leurs lignes du memmap sont deja bonnes).

A executer avec l'environnement .venv-embed :
    .venv-embed\\Scripts\\python.exe encoder_corpus_complet.py
"""

import sys
import io
import os
import json
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from inspection import extraire_texte, detecter_sommaire, detecter_lignes_repete
from extraire_titre_sommaire import extraire_titres_sommaire, filtrer_bandeau
from localiser_titres_corps import localiser_titres_dans_corps
from decouper_segments import decouper_en_segments, compteur_taille_tokenizer
from transformers import AutoTokenizer

CORPUS = r"C:\Users\1\Documents\MASTER_SOUTENANCE\corpus"
NOM_MODELE = "dangvantuan/sentence-camembert-large"
FENETRE_REELLE = 512
TAILLE_MAX_CHUNK_TOKENS = 400
CHEVAUCHEMENT_TOKENS = 40
DIMENSION_ATTENDUE = 1024

FICHIER_VECTEURS = "vecteurs_corpus.npy"
FICHIER_METADONNEES = "metadonnees_corpus.parquet"
FICHIER_CHECKPOINT = "checkpoint_encodage.json"

LOG_TOUS_LES_N_CHUNKS = 500


# -----------------------------------------------------------------------------
# ETAPE A - SEGMENTATION DE L'ENSEMBLE DU CORPUS (CPU, pas de GPU necessaire)
# -----------------------------------------------------------------------------

def obtenir_chunks_document(chemin_pdf, compteur_taille):
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
    return chunks


def construire_chunks_corpus_complet(compteur_taille):
    dossier = Path(CORPUS)
    fichiers_pdf = sorted(dossier.glob('**/*.pdf'))

    chunks_par_document = {}
    for chemin_pdf in fichiers_pdf:
        nom = chemin_pdf.name
        chunks = obtenir_chunks_document(chemin_pdf, compteur_taille)
        # cle_unique_section peut valoir None pour plusieurs sections
        # distinctes d'un meme document (aucun titre associe) : ce n'est
        # donc pas un identifiant fiable a lui seul. numero_chunk_dans_section
        # revient a 0 exactement au debut de chaque nouvelle section (voir
        # decouper_segments.decouper_en_segments, qui ajoute les chunks
        # section par section dans l'ordre) : on s'en sert pour compter les
        # sections rencontrees et obtenir un identifiant garanti unique.
        compteur_section = 0
        for c in chunks:
            if c["numero_chunk_dans_section"] == 0:
                compteur_section += 1
            c["document"] = nom
            c["chunk_id"] = f"{nom}::section{compteur_section:03d}::{c['numero_chunk_dans_section']}"
        chunks_par_document[nom] = chunks

    tous_chunks = [c for chunks in chunks_par_document.values() for c in chunks]

    # Verification d'unicite du chunk_id, condition necessaire pour que le
    # lien vecteurs <-> metadonnees soit fiable.
    ids = [c["chunk_id"] for c in tous_chunks]
    doublons = {i for i in ids if ids.count(i) > 1}
    if doublons:
        raise ValueError(f"chunk_id non uniques detectes : {doublons}")

    return chunks_par_document, tous_chunks


# -----------------------------------------------------------------------------
# ETAPE B - ENCODAGE PAR LOTS AVEC SAUVEGARDE PROGRESSIVE
# -----------------------------------------------------------------------------

def charger_checkpoint():
    if os.path.exists(FICHIER_CHECKPOINT):
        with open(FICHIER_CHECKPOINT, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"documents_termines": []}


def sauvegarder_checkpoint(checkpoint):
    with open(FICHIER_CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def encoder_corpus(chunks_par_document, tous_chunks, modele):
    n_total = len(tous_chunks)
    print(f"\nTotal de chunks a encoder : {n_total}")

    # Index global de chaque chunk_id -> position de ligne dans le memmap.
    index_ligne = {c["chunk_id"]: i for i, c in enumerate(tous_chunks)}

    checkpoint = charger_checkpoint()
    reprise = os.path.exists(FICHIER_VECTEURS) and checkpoint["documents_termines"]

    if reprise:
        print(f"Reprise detectee : {len(checkpoint['documents_termines'])} documents deja encodes.")
        vecteurs = np.lib.format.open_memmap(FICHIER_VECTEURS, mode="r+")
        if vecteurs.shape != (n_total, DIMENSION_ATTENDUE):
            raise ValueError(
                f"Le fichier {FICHIER_VECTEURS} existant a la forme {vecteurs.shape}, "
                f"attendu ({n_total}, {DIMENSION_ATTENDUE}). Segmentation modifiee "
                f"depuis le dernier run ? Supprimer le checkpoint pour repartir a zero."
            )
    else:
        vecteurs = np.lib.format.open_memmap(
            FICHIER_VECTEURS, mode="w+", dtype=np.float32, shape=(n_total, DIMENSION_ATTENDUE)
        )

    torch.cuda.reset_peak_memory_stats()
    t_debut = time.time()
    n_encodes_cette_session = 0
    n_deja_faits = 0

    documents = sorted(chunks_par_document.keys())
    for i_doc, nom_doc in enumerate(documents, 1):
        chunks_doc = chunks_par_document[nom_doc]
        if not chunks_doc:
            continue

        if nom_doc in checkpoint["documents_termines"]:
            n_deja_faits += len(chunks_doc)
            continue

        textes = [c["texte"] for c in chunks_doc]
        vecs = modele.encode(
            textes, convert_to_numpy=True, show_progress_bar=False, batch_size=32
        )
        for c, v in zip(chunks_doc, vecs):
            vecteurs[index_ligne[c["chunk_id"]]] = v.astype(np.float32)
        vecteurs.flush()

        checkpoint["documents_termines"].append(nom_doc)
        sauvegarder_checkpoint(checkpoint)

        n_encodes_cette_session += len(chunks_doc)
        n_fait_total = n_deja_faits + n_encodes_cette_session

        print(f"  [{i_doc}/{len(documents)}] {nom_doc} : {len(chunks_doc)} chunks encodes "
              f"({n_fait_total}/{n_total} au total)")

        if n_encodes_cette_session % LOG_TOUS_LES_N_CHUNKS < len(chunks_doc):
            print(f"    -- progression : {n_fait_total}/{n_total} chunks --")

    t_total = time.time() - t_debut
    vram_pic = torch.cuda.max_memory_allocated() / 1024 ** 3

    vecteurs.flush()
    return vecteurs, t_total, vram_pic, n_encodes_cette_session


# -----------------------------------------------------------------------------
# ETAPE C - VERIFICATIONS
# -----------------------------------------------------------------------------

def verifier(tous_chunks, t_total, vram_pic, n_encodes_cette_session, tokenizer):
    print(f"\n{'=' * 90}\nVERIFICATIONS\n{'=' * 90}")

    n_total = len(tous_chunks)

    # Relecture a froid (fichier ferme puis rouvert) pour ne pas juste
    # verifier le memmap encore en memoire de ce process.
    vecteurs_relus = np.load(FICHIER_VECTEURS)

    # 1. Nombre total de vecteurs
    print(f"1. Nombre de vecteurs produits : {vecteurs_relus.shape[0]} "
          f"(attendu {n_total}) -> {'OK' if vecteurs_relus.shape[0] == n_total else 'ECHEC'}")

    # 2. Aucun vecteur NaN ou nul
    normes = np.linalg.norm(vecteurs_relus, axis=1)
    nb_nan = int(np.isnan(vecteurs_relus).any(axis=1).sum())
    nb_nuls = int((normes < 1e-6).sum())
    print(f"2. Vecteurs NaN : {nb_nan} -> {'OK' if nb_nan == 0 else 'ECHEC'}")
    print(f"   Vecteurs nuls (norme ~0) : {nb_nuls} -> {'OK' if nb_nuls == 0 else 'ECHEC'}")

    # 3. Dimension coherente
    dims_ok = vecteurs_relus.shape[1] == DIMENSION_ATTENDUE
    print(f"3. Dimension : {vecteurs_relus.shape[1]} (attendu {DIMENSION_ATTENDUE}) "
          f"-> {'OK' if dims_ok else 'ECHEC'}")

    # 4. Temps et VRAM
    estimation_min = 5.7
    print(f"4. Temps d'encodage (chunks encodes cette session : {n_encodes_cette_session}) : "
          f"{t_total:.1f}s ({t_total / 60:.1f} min)")
    print(f"   Estimation faite lors du comparatif (extrapolation sur 7443 chunks) : "
          f"~{estimation_min} min")
    print(f"   VRAM pic observee : {vram_pic:.2f} Go")

    # 5. Echantillon de verification stockage/relecture
    print(f"\n5. Echantillon de verification (relecture depuis disque) :")
    rng = np.random.default_rng(42)
    indices_echantillon = rng.choice(n_total, size=3, replace=False)

    # metadonnees pour retrouver le texte des chunks choisis + un chunk "different"
    df_meta = pd.read_parquet(FICHIER_METADONNEES)

    def cosinus(u, v):
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

    for idx in indices_echantillon:
        chunk_id = df_meta.iloc[idx]["chunk_id"]
        vec = vecteurs_relus[idx]
        sim_soi_meme = cosinus(vec, vec)

        # chunk "clairement different" : un autre document, loin dans le tableau
        idx_different = (idx + n_total // 2) % n_total
        doc_a = df_meta.iloc[idx]["document"]
        doc_b = df_meta.iloc[idx_different]["document"]
        vec_different = vecteurs_relus[idx_different]
        sim_different = cosinus(vec, vec_different)

        print(f"   {chunk_id!r}")
        print(f"     similarite avec lui-meme (relu du disque) : {sim_soi_meme:.6f} "
              f"-> {'OK' if abs(sim_soi_meme - 1.0) < 1e-4 else 'ECHEC'}")
        print(f"     similarite avec un chunk different ({doc_a} vs {doc_b}) : "
              f"{sim_different:.4f} -> {'OK (nettement < 1)' if sim_different < 0.9 else 'A VERIFIER'}")


if __name__ == "__main__":
    print("Chargement du tokenizer et segmentation des 63 documents...")
    tokenizer = AutoTokenizer.from_pretrained(NOM_MODELE)
    compteur = compteur_taille_tokenizer(tokenizer)
    chunks_par_document, tous_chunks = construire_chunks_corpus_complet(compteur)
    print(f"{len(tous_chunks)} chunks au total sur {len(chunks_par_document)} documents.")

    if not os.path.exists(FICHIER_METADONNEES):
        colonnes = ["chunk_id", "document", "cle_unique_section", "niveau", "legitime",
                    "chapitre_parent", "position_debut", "position_fin",
                    "numero_chunk_dans_section", "texte"]
        df_meta = pd.DataFrame(tous_chunks)[colonnes]
        df_meta.to_parquet(FICHIER_METADONNEES, index=False)
        print(f"Metadonnees sauvegardees dans {FICHIER_METADONNEES}")
    else:
        print(f"{FICHIER_METADONNEES} existe deja, non recree (reprise).")

    print(f"\nChargement du modele {NOM_MODELE}...")
    modele = SentenceTransformer(NOM_MODELE, device="cuda")
    modele.max_seq_length = FENETRE_REELLE

    vecteurs, t_total, vram_pic, n_encodes_cette_session = encoder_corpus(
        chunks_par_document, tous_chunks, modele
    )

    verifier(tous_chunks, t_total, vram_pic, n_encodes_cette_session, tokenizer)

# -*- coding: utf-8 -*-
"""
Etape 3 - Encodage des chunks reels des 3 documents de reference (SIMPORE,
SERE, 08-memoire) avec sentence-camembert-large : analyse de troncature,
temps/VRAM d'encodage, sauvegarde des vecteurs pour le test de sanite
semantique.

Correctif important : le modele/tokenizer annonce une fenetre de 514
tokens, mais c'est une valeur buggee pour ce type d'architecture
(CamemBERT/RoBERTa) - les identifiants de position demarrent a
padding_idx+1=2, pas 0, donc la fenetre reellement exploitable est
max_position_embeddings - 2 = 512 tokens. Un chunk qui atteint exactement
514 tokens fait planter l'encodage (CUDA device-side assert / "index 514
is out of bounds for dimension 1 with size 514" sur CPU). D'ou le
FENETRE_REELLE = 512 force ci-dessous plutot que de faire confiance a
modele.get_max_seq_length().

A executer avec l'environnement .venv-embed :
    .venv-embed\\Scripts\\python.exe tester_encodage_chunks.py
"""

import sys
import io
import time
import pickle

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from inspection import extraire_texte, detecter_sommaire, detecter_lignes_repete
from extraire_titre_sommaire import extraire_titres_sommaire, filtrer_bandeau
from localiser_titres_corps import localiser_titres_dans_corps
from decouper_segments import decouper_en_segments

CORPUS = r"C:\Users\1\Documents\MASTER_SOUTENANCE\corpus"
DOCS = [
    "SIMPORE Némata Version Finale.pdf",
    "SERE MEMOIRE MASTER IBAM.docxVF corrigée.pdf",
    "08-memoire.pdf",
]

FENETRE_REELLE = 512
FICHIER_VECTEURS = "vecteurs_ref.npy"
FICHIER_CHUNKS = "chunks_ref.pkl"


def obtenir_chunks(nom):
    chemin = f"{CORPUS}\\{nom}"
    texte_complet = extraire_texte(chemin)
    present, nb, bloc = detecter_sommaire(texte_complet)
    bandeau_present, liste_bandeaux, _ = detecter_lignes_repete(texte_complet)
    bloc_f = filtrer_bandeau(bloc, liste_bandeaux) if bandeau_present else bloc
    titres = extraire_titres_sommaire(bloc_f)
    titres = localiser_titres_dans_corps(texte_complet, titres)
    chunks = decouper_en_segments(texte_complet, titres, liste_bandeaux=liste_bandeaux)
    for c in chunks:
        c["_document"] = nom
    return chunks


if __name__ == "__main__":
    print("Chargement des chunks des 3 documents de reference...")
    tous_chunks = []
    for nom in DOCS:
        chunks = obtenir_chunks(nom)
        print(f"  {nom} : {len(chunks)} chunks")
        tous_chunks.extend(chunks)
    print(f"Total : {len(tous_chunks)} chunks")

    print("\nChargement du modele sentence-camembert-large...")
    modele = SentenceTransformer("dangvantuan/sentence-camembert-large", device="cuda")
    modele.max_seq_length = FENETRE_REELLE
    print(f"Fenetre annoncee par le modele/tokenizer : 514 tokens (buguee pour cette architecture)")
    print(f"Fenetre reellement utilisee (corrigee) : {FENETRE_REELLE} tokens")

    print("\nAnalyse de troncature...")
    tokenizer = modele.tokenizer
    details_troncature = []
    for c in tous_chunks:
        nb_tokens = len(tokenizer.encode(c["texte"], add_special_tokens=True))
        if nb_tokens > FENETRE_REELLE:
            details_troncature.append((c["_document"], c["position_debut"],
                                        c["numero_chunk_dans_section"],
                                        len(c["texte"].split()), nb_tokens))

    print(f"Chunks depassant la fenetre reelle ({FENETRE_REELLE} tokens) : "
          f"{len(details_troncature)} / {len(tous_chunks)} "
          f"({len(details_troncature) / len(tous_chunks):.1%})")
    print("\nDetail des 15 plus tronques :")
    for d in sorted(details_troncature, key=lambda x: -x[4])[:15]:
        print(f"  {d}")

    print(f"\nEncodage de {len(tous_chunks)} chunks (fenetre corrigee)...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    textes = [c["texte"] for c in tous_chunks]
    vecteurs = modele.encode(textes, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
    t_total = time.time() - t0

    print(f"Temps total : {t_total:.1f} s ({t_total / len(tous_chunks) * 1000:.1f} ms/chunk)")
    print(f"Dimension : {vecteurs.shape}")
    print(f"VRAM pic : {torch.cuda.max_memory_allocated() / 1024**3:.2f} Go")

    np.save(FICHIER_VECTEURS, vecteurs)
    with open(FICHIER_CHUNKS, "wb") as f:
        pickle.dump(tous_chunks, f)

    print(f"\nOK - vecteurs sauvegardes dans {FICHIER_VECTEURS}, chunks dans {FICHIER_CHUNKS}")

# -*- coding: utf-8 -*-
"""
Comparatif de sentence-camembert-large avec trois autres modeles d'embedding,
sur les memes donnees et avec le meme tableau que comparer_modeles_definitif.py :

  - dangvantuan/sentence-camembert-large                         (modele retenu)
  - sentence-transformers/paraphrase-multilingual-mpnet-base-v2
  - intfloat/multilingual-e5-large
  - dangvantuan/sentence-camembert-base

Etapes (reprises de comparer_modeles_definitif.py, dont on importe le mini-jeu
et les fonctions) :
  1. mini-jeu "verite connue" : 10 paires (5 paraphrases, 5 sans rapport),
     moyennes, ecart, paire 9 ambigue ;
  2. normes de vecteurs sur les chunks a forte densite de citations ;
  3. vitesse et VRAM sur les chunks reels des 3 documents de reference.

Choix de mesure, appliques de la meme facon aux 4 modeles :
  - un seul modele charge a la fois (la VRAM mesuree n'inclut donc pas les
    poids des autres modeles) ;
  - max_seq_length = 512 pour tous (le defaut de paraphrase-multilingual-mpnet
    est 128, qui tronquerait des chunks pouvant atteindre 400 tokens) ;
  - multilingual-e5-large demande un prefixe : "query: " est ajoute a tous les
    textes (usage recommande pour une similarite symetrique) ;
  - un petit passage d'echauffement GPU avant la mesure du temps ;
  - les temps sont extrapoles au nombre de chunks du corpus courant, lu dans
    metadonnees_corpus.parquet ;
  - la colonne "chunks tronques" compte, avec le tokenizer de chaque modele,
    les chunks depassant max_seq_length (information absente du tableau
    d'origine).

A executer avec l'environnement .venv-embed :
    .venv-embed\\Scripts\\python.exe comparer_modeles_supplementaires.py
"""

import gc
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from comparer_modeles_definitif import (
    CAMEMBERT_TEMPS_298, CAMEMBERT_VRAM_PIC_298, NB_CHUNKS_CORPUS_COMPLET, NOM_CAMEMBERT,
    etape1_verite_connue, obtenir_chunks_reference, selectionner_chunks_difficiles,
)
from construire_tfidf_corpus import FICHIER_METADONNEES
from decouper_segments import compteur_taille_tokenizer

LONGUEUR_MAX = 512
FICHIER_SORTIE = "comparatif_modeles_supplementaires.csv"

MODELES = [
    ("sentence-camembert-large", "dangvantuan/sentence-camembert-large", ""),
    ("paraphrase-multilingual-mpnet-base-v2", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", ""),
    ("multilingual-e5-large", "intfloat/multilingual-e5-large", "query: "),
    ("sentence-camembert-base", "dangvantuan/sentence-camembert-base", ""),
]


class AvecPrefixe:
    """Ajoute un prefixe a tous les textes encodes (necessaire pour E5)."""

    def __init__(self, modele, prefixe):
        self.modele, self.prefixe = modele, prefixe

    def encode(self, textes, **kwargs):
        return self.modele.encode([self.prefixe + t for t in textes], **kwargs)


def etape2_normes(encodeur, chunks_difficiles):
    vecteurs = encodeur.encode([c["texte"] for c in chunks_difficiles],
                               convert_to_numpy=True, show_progress_bar=False)
    normes = np.linalg.norm(vecteurs, axis=1)
    anomalies = int(np.sum((normes < 1e-3) | np.isnan(normes)))
    return {"norme_min": float(normes.min()), "norme_max": float(normes.max()),
            "norme_ecart_type": float(normes.std()), "anomalies": anomalies}


def etape3_vitesse_vram(encodeur, textes, n_passes=3):
    """Temps = mediane de n_passes encodages complets (apres echauffement) ; VRAM = pic de memoire allouee."""
    encodeur.encode(textes[:16], convert_to_numpy=True, show_progress_bar=False, batch_size=16)  # echauffement
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    durees = []
    for _ in range(n_passes):
        t0 = time.time()
        encodeur.encode(textes, convert_to_numpy=True, show_progress_bar=False, batch_size=16)
        torch.cuda.synchronize()
        durees.append(time.time() - t0)
    return float(np.median(durees)), torch.cuda.max_memory_allocated() / 1024 ** 3, durees


def chunks_tronques(tokenizer, textes, prefixe):
    n = sum(1 for t in textes if len(tokenizer(prefixe + t, add_special_tokens=True)["input_ids"]) > LONGUEUR_MAX)
    return n


if __name__ == "__main__":
    nb_chunks_corpus = len(pd.read_parquet(FICHIER_METADONNEES))
    print(f"Corpus courant : {nb_chunks_corpus} chunks (extrapolation des temps)")

    print("\nChargement des chunks reels (3 documents de reference, decoupage token-based)...")
    compteur = compteur_taille_tokenizer(AutoTokenizer.from_pretrained(NOM_CAMEMBERT))
    tous_chunks = obtenir_chunks_reference(compteur)
    chunks_difficiles = selectionner_chunks_difficiles(tous_chunks)
    textes_tous = [c["texte"] for c in tous_chunks]
    print(f"  {len(tous_chunks)} chunks, dont {len(chunks_difficiles)} chunks a forte densite de citations")

    resultats = []
    for label, nom_hf, prefixe in MODELES:
        print(f"\n{'#' * 90}\nMODELE : {label} ({nom_hf})\n{'#' * 90}")
        gc.collect()
        torch.cuda.empty_cache()
        modele = SentenceTransformer(nom_hf, device="cuda")
        modele.max_seq_length = LONGUEUR_MAX
        vram_chargement = torch.cuda.memory_allocated() / 1024 ** 3
        print(f"  Memoire GPU allouee apres chargement (poids seuls, aucun autre modele residuel) : "
              f"{vram_chargement:.2f} Go")
        encodeur = AvecPrefixe(modele, prefixe) if prefixe else modele

        r1 = etape1_verite_connue(encodeur, label)
        r2 = etape2_normes(encodeur, chunks_difficiles)
        t, vram, durees = etape3_vitesse_vram(encodeur, textes_tous)
        tronques = chunks_tronques(modele.tokenizer, textes_tous, prefixe)
        print(f"\n  Normes (zones difficiles) : min={r2['norme_min']:.4f}, max={r2['norme_max']:.4f}, "
              f"ecart-type={r2['norme_ecart_type']:.4f}, anomalies={r2['anomalies']}")
        print(f"  Vitesse : mediane {t:.1f}s pour {len(tous_chunks)} chunks ({t / len(tous_chunks) * 1000:.1f} ms/chunk) "
              f"sur 3 passes ({', '.join(f'{d:.1f}' for d in durees)}), VRAM pic = {vram:.2f} Go ; "
              f"chunks tronques (> {LONGUEUR_MAX} tokens) : {tronques}")

        resultats.append({
            "modele": label, "dimension": modele.get_sentence_embedding_dimension(),
            "moy_paraphrase": r1["moy_paraphrase"], "moy_sans_rapport": r1["moy_sans_rapport"],
            "ecart": r1["ecart"], "paire9": r1["paire9"],
            "similarites_par_paire": ";".join(f"{x['id']}:{x['similarite']:.4f}" for x in r1["resultats"]),
            "anomalies_normes": r2["anomalies"], "norme_min": r2["norme_min"], "norme_max": r2["norme_max"],
            "temps_s": t, "ms_par_chunk": t / len(tous_chunks) * 1000,
            "temps_extrapole_corpus_min": t / len(tous_chunks) * nb_chunks_corpus / 60,
            "vram_go": vram, "vram_chargement_go": vram_chargement, "chunks_tronques": tronques,
        })
        del modele, encodeur
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'=' * 130}\nTABLEAU DE SYNTHESE (meme structure que comparer_modeles_definitif.py)\n{'=' * 130}")
    noms = [r["modele"] for r in resultats]
    print(f"{'Dimension':40s} " + " ".join(f"{n[:30]:>30s}" for n in noms))

    def ligne(titre, cle, fmt):
        print(f"{titre:40s} " + " ".join(f"{fmt.format(r[cle]):>30s}" for r in resultats))

    ligne("Dimension des vecteurs", "dimension", "{:d}")
    ligne("Moyenne PARAPHRASE", "moy_paraphrase", "{:.4f}")
    ligne("Moyenne SANS RAPPORT", "moy_sans_rapport", "{:.4f}")
    ligne("Ecart paraphrase/sans-rapport", "ecart", "{:.4f}")
    ligne("Similarite paire 9 (ambigu)", "paire9", "{:.4f}")
    ligne("Anomalies normes (zones difficiles)", "anomalies_normes", "{:d}")
    ligne(f"Temps ({len(tous_chunks)} chunks, s)", "temps_s", "{:.1f}")
    ligne("ms par chunk", "ms_par_chunk", "{:.1f}")
    ligne(f"Temps extrapole {nb_chunks_corpus} chunks (min)", "temps_extrapole_corpus_min", "{:.1f}")
    ligne("VRAM apres chargement (poids, Go)", "vram_chargement_go", "{:.2f}")
    ligne(f"VRAM pic ({len(tous_chunks)} chunks, Go)", "vram_go", "{:.2f}")
    ligne(f"Chunks tronques (> {LONGUEUR_MAX} tokens)", "chunks_tronques", "{:d}")

    print(f"\nRappel des valeurs mesurees autrefois pour sentence-camembert-large "
          f"(comparer_modeles_definitif.py) : {CAMEMBERT_TEMPS_298:.1f}s pour 298 chunks, "
          f"VRAM pic {CAMEMBERT_VRAM_PIC_298:.2f} Go, extrapolation a {NB_CHUNKS_CORPUS_COMPLET} chunks.")

    pd.DataFrame(resultats).to_csv(FICHIER_SORTIE, index=False, encoding="utf-8-sig")
    print(f"\nSauvegarde : {FICHIER_SORTIE}")

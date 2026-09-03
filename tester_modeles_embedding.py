# -*- coding: utf-8 -*-
"""
Etape 2 - Comparatif de deux modeles d'embedding candidats sur un petit
lot de phrases de test (pas encore les vrais chunks) : temps de chargement,
VRAM utilisee, temps d'encodage, dimension des vecteurs produits.

A executer avec l'environnement .venv-embed (Python 3.11 + torch CUDA +
sentence-transformers) :
    .venv-embed\\Scripts\\python.exe tester_modeles_embedding.py
"""

import time
import torch
from sentence_transformers import SentenceTransformer

PHRASES_TEST = [
    "Le système d'information comptable améliore la performance financière des entreprises.",
    "Les entreprises familiales font face à des défis spécifiques de gouvernance.",
    "Il pleut beaucoup à Ouagadougou pendant la saison des pluies.",
]


def vram_utilisee_go():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**3


def tester_modele(nom_hf, label):
    print("=" * 90)
    print(f"MODELE : {label} ({nom_hf})")
    print("=" * 90)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    modele = SentenceTransformer(nom_hf, device="cuda")
    t_chargement = time.time() - t0

    vram_apres_chargement = vram_utilisee_go()
    print(f"Temps de chargement : {t_chargement:.1f} s")
    print(f"VRAM allouee apres chargement : {vram_apres_chargement:.2f} Go")
    print(f"VRAM libre restante (mem_get_info) : {torch.cuda.mem_get_info()[0] / 1024**3:.2f} Go")

    t0 = time.time()
    vecteurs = modele.encode(PHRASES_TEST, convert_to_numpy=True, show_progress_bar=False)
    t_encodage = time.time() - t0

    print(f"Temps d'encodage ({len(PHRASES_TEST)} phrases) : {t_encodage:.3f} s")
    print(f"Dimension des vecteurs : {vecteurs.shape}")
    print(f"VRAM pic (max_memory_allocated) : {torch.cuda.max_memory_allocated() / 1024**3:.2f} Go")

    del modele
    torch.cuda.empty_cache()

    return {
        "label": label,
        "t_chargement": t_chargement,
        "vram_apres_chargement_go": vram_apres_chargement,
        "t_encodage": t_encodage,
        "dimension": vecteurs.shape[1],
    }


if __name__ == "__main__":
    resultats = [
        tester_modele("dangvantuan/sentence-camembert-large", "sentence-camembert-large"),
        tester_modele("BAAI/bge-m3", "BGE-M3"),
    ]

    print("\n" + "=" * 90)
    print("COMPARATIF")
    print("=" * 90)
    print(f"{'Modele':30s} {'Chargement':>12s} {'VRAM (Go)':>12s} {'Encodage 3 phr.':>18s} {'Dimension':>10s}")
    for r in resultats:
        print(f"{r['label']:30s} {r['t_chargement']:>10.1f}s {r['vram_apres_chargement_go']:>10.2f}Go "
              f"{r['t_encodage']:>16.3f}s {r['dimension']:>10d}")

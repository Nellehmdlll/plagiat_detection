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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from comparer_lexical_recent import (
    GRAINE, N_FOLDS, N_REPETITIONS, bootstrap_auc, evaluer_fusion, evaluer_signal_seul, scores_tfidf,
)
from construire_tfidf_corpus import FICHIER_METADONNEES
from decouper_segments import compteur_taille_tokenizer
from evaluer_jeu_test_complet import VALEURS_ALPHA, charger_jeu_complet

FICHIER_SORTIE_175 = "comparatif_modeles_175_paires.csv"
SEUILS_FUSION = [round(s, 2) for s in np.arange(0.0, 1.0001, 0.01)]

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

    df175 = charger_jeu_complet().reset_index(drop=True)
    y175 = (df175["etiquette"] == "PLAGIAT").to_numpy()
    categories175 = df175["categorie"].to_numpy()
    ids175 = df175["id_paire"].astype(str).to_numpy()
    textes175 = df175["texte_1"].tolist() + df175["texte_2"].tolist()
    df_meta = pd.read_parquet(FICHIER_METADONNEES)
    _, tfidf175 = scores_tfidf(df175, df_meta)
    decoupages = list(RepeatedStratifiedKFold(
        n_splits=N_FOLDS, n_repeats=N_REPETITIONS, random_state=GRAINE).split(np.zeros(len(y175)), y175))
    scores175, tronques175 = {}, {}
    print(f"  Jeu de test : {len(df175)} paires ({int(y175.sum())} PLAGIAT, {int((~y175).sum())} NON-PLAGIAT)")

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
        v1 = encodeur.encode(df175["texte_1"].tolist(), convert_to_numpy=True, show_progress_bar=False, batch_size=16)
        v2 = encodeur.encode(df175["texte_2"].tolist(), convert_to_numpy=True, show_progress_bar=False, batch_size=16)
        scores175[label] = (v1 * v2).sum(axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
        tronques175[label] = chunks_tronques(modele.tokenizer, textes175, prefixe)
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

    # -------------------------------------------------------------------
    # EVALUATION SUR LES 175 PAIRES ETIQUETEES (meme critere que le comparatif lexical)
    # -------------------------------------------------------------------
    print(f"\n{'#' * 120}\nEVALUATION SUR LES {len(df175)} PAIRES ETIQUETEES\n{'#' * 120}")
    print("Critere de decision : F1 en validation croisee de la FUSION avec TF-IDF (c'est la configuration du "
          "pipeline),\nconfirme par l'AUC et le rappel ; si un modele est a moins d'un ecart-type du meilleur "
          "F1 CV, on garde le plus simple.")
    print(f"Fusion : alpha x cosinus du modele + (1 - alpha) x cosinus TF-IDF ; grille alpha {VALEURS_ALPHA[0]} a "
          f"{VALEURS_ALPHA[-1]} (pas 0.05) x seuil {SEUILS_FUSION[0]} a {SEUILS_FUSION[-1]} (pas 0.01), "
          f"la meme pour les 4 modeles (les echelles de cosinus different : E5 est tres comprime).")
    print(f"Validation croisee : {N_FOLDS} plis x {N_REPETITIONS} repetitions, graine {GRAINE}")

    _, auc175 = bootstrap_auc(y175, scores175)
    auc_ref = roc_auc_score(y175, scores175["sentence-camembert-large"])

    print(f"\n{'=' * 120}\nA. SIGNAL SEMANTIQUE SEUL\n{'=' * 120}")
    print(f"{'Modele':40s} {'AUC':>7s} {'IC 95% AUC':>20s} {'AP':>7s} {'F1 opt.':>8s} {'rappel':>7s} "
          f"{'F1 CV':>14s} {'textes tronques':>16s}")
    lignes175 = []
    for label, s in scores175.items():
        r = evaluer_signal_seul(y175, s, decoupages)
        ic = f"[{np.percentile(auc175[label], 2.5):.4f} ; {np.percentile(auc175[label], 97.5):.4f}]"
        print(f"{label:40s} {r['auc']:7.4f} {ic:>20s} {r['ap']:7.4f} {r['f1_optimiste']:8.4f} {r['rappel']:7.4f} "
              f"{r['f1_cv']:7.4f}+-{r['f1_cv_ecart_type']:.3f} {tronques175[label]:>10d} / {len(textes175)}")
        lignes175.append({"evaluation": "semantique_seul", "modele": label,
                          "auc_ic_bas": float(np.percentile(auc175[label], 2.5)),
                          "auc_ic_haut": float(np.percentile(auc175[label], 97.5)),
                          "textes_tronques": tronques175[label], **r})

    print(f"\n{'=' * 120}\nB. FUSION AVEC TF-IDF (configuration du pipeline)\n{'=' * 120}")
    print(f"{'Modele':40s} {'alpha':>6s} {'seuil':>6s} {'F1 opt.':>8s} {'prec.':>7s} {'rappel':>7s} "
          f"{'F1 CV':>14s}  {'VP/FP/VN/FN':>14s}")
    fusions175 = {}
    for label, s in scores175.items():
        r = evaluer_fusion(y175, s, tfidf175, categories175, ids175, decoupages,
                           alphas=VALEURS_ALPHA, seuils=SEUILS_FUSION)
        fusions175[label] = r
        print(f"{label:40s} {r['alpha']:6.2f} {r['seuil']:6.2f} {r['f1_optimiste']:8.4f} {r['precision']:7.4f} "
              f"{r['rappel']:7.4f} {r['f1_cv']:7.4f}+-{r['f1_cv_ecart_type']:.3f}  "
              f"{r['VP']:>3d}/{r['FP']:>2d}/{r['VN']:>3d}/{r['FN']:>2d}")
        lignes175.append({"evaluation": "fusion_avec_tfidf", "modele": label, **r})

    print(f"\nPaires mal classees au couple optimal de chaque fusion :")
    for label, r in fusions175.items():
        print(f"  {label:40s} faux positifs : {r['faux_positifs'] or '-'} | faux negatifs : {r['faux_negatifs'] or '-'}")
    print(f"\nDetail par categorie (rappel A/B/C ; faux positifs D/E)")
    print(f"{'Modele':40s} {'rappel A':>9s} {'rappel B':>9s} {'rappel C':>9s} {'FP D':>7s} {'FP E':>7s}")
    for label, r in fusions175.items():
        print(f"{label:40s} {r['rappel_A']:9.4f} {r['rappel_B']:9.4f} {r['rappel_C']:9.4f} "
              f"{r['fp_D']:7.4f} {r['fp_E']:7.4f}")

    print(f"\n{'=' * 120}\nC. DIFFERENCE D'AUC AVEC sentence-camembert-large (modele retenu), IC 95%\n{'=' * 120}")
    for label, s in scores175.items():
        if label == "sentence-camembert-large":
            continue
        d = auc175[label] - auc175["sentence-camembert-large"]
        lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
        print(f"  {label:40s} {roc_auc_score(y175, s) - auc_ref:+.4f} [{lo:+.4f} ; {hi:+.4f}] "
              + ("contient 0" if lo <= 0 <= hi else "NE contient PAS 0"))

    print(f"\n{'=' * 120}\nD. APPLICATION DE LA REGLE DE DECISION\n{'=' * 120}")
    meilleur = max(fusions175, key=lambda m: fusions175[m]["f1_cv"])
    f1_best, std_best = fusions175[meilleur]["f1_cv"], fusions175[meilleur]["f1_cv_ecart_type"]
    print(f"Meilleur F1 CV en fusion : {meilleur} = {f1_best:.4f} (ecart-type {std_best:.4f})")
    print(f"Modeles a moins d'un ecart-type du meilleur (F1 CV >= {f1_best - std_best:.4f}) :")
    for label, r in fusions175.items():
        ecart = f1_best - r["f1_cv"]
        print(f"  {label:40s} F1 CV = {r['f1_cv']:.4f} (ecart au meilleur : {ecart:.4f}) -> "
              + ("DANS un ecart-type" if ecart <= std_best else "hors ecart-type"))
    print("Cout (pour departager a egalite) : voir la table de synthese ci-dessus (VRAM, temps, dimension).")

    pd.DataFrame(lignes175).to_csv(FICHIER_SORTIE_175, index=False, encoding="utf-8-sig")
    print(f"\nSauvegarde : {FICHIER_SORTIE_175}")

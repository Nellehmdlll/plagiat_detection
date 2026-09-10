# -*- coding: utf-8 -*-
"""
Remplace, parmi les sources B/C restant a paraphraser, celles detectees
comme tabulaires ou mal extraites (tableaux de donnees, mots colles par
l'extraction PDF en colonnes) par de nouvelles sources propres, de meme
categorie cible, prises ailleurs dans le corpus (documents non deja
utilises dans le jeu de test).
"""
import sys
import io
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd

RNG = np.random.default_rng(99)


def est_suspect(texte):
    if re.search(r"TABLE\s*\d|Tableau\s*[IVX\d]", texte):
        return True
    mots = texte.split()
    if not mots:
        return True
    longueur_moy = sum(len(m) for m in mots) / len(mots)
    if longueur_moy > 9:
        return True
    if max(len(m) for m in mots) > 18:
        return True  # un seul "mot" tres long trahit un texte colle par l'extraction
    if texte.count("/_/") >= 2:
        return True  # questionnaire/formulaire a cocher
    if len(re.findall(r"\bScore\b", texte)) >= 2:
        return True  # grille de score/questionnaire
    return False


if __name__ == "__main__":
    df_meta = pd.read_parquet("metadonnees_corpus.parquet")
    sources_bc = pd.read_parquet("sources_bc_a_paraphraser.parquet")
    jeu_test = pd.read_parquet("jeu_test_etiquete.parquet")

    # Ne jamais toucher aux sources deja validees (texte_2 deja rempli) -
    # sinon on desynchronise texte_1 (nouvelle source) et texte_2 (ancienne
    # reformulation, ecrite pour l'ancienne source).
    textes_valides = set(jeu_test.loc[jeu_test["texte_2"].notna(), "texte_1"])

    sources_bc["suspect"] = sources_bc["texte"].apply(est_suspect) & ~sources_bc["texte"].isin(textes_valides)
    suspects = sources_bc[sources_bc["suspect"]]
    print(f"{len(suspects)} sources suspectes a remplacer :")
    for _, r in suspects.iterrows():
        print(f"  {r['categorie_cible']} | {r['chunk_id']}")

    # Seule contrainte stricte : ne pas reprendre un chunk deja utilise comme
    # SOURCE ailleurs dans le jeu de test (le nombre de documents distincts
    # du corpus, 63, est trop faible pour garantir "1 chunk par document" une
    # fois les 165 sources A/B/C/D/E deja attribuees).
    chunks_deja_sources = set(jeu_test["texte_1"]).union(sources_bc["chunk_id"])
    textes_deja_sources = set(jeu_test["texte_1"])

    non_legitimes = df_meta[~df_meta["legitime"]].copy()
    non_legitimes = non_legitimes[non_legitimes["texte"].str.split().str.len() >= 40]
    non_legitimes = non_legitimes[~non_legitimes["texte"].apply(est_suspect)]
    non_legitimes = non_legitimes[~non_legitimes["chunk_id"].isin(sources_bc["chunk_id"])]
    non_legitimes = non_legitimes[~non_legitimes["texte"].isin(textes_deja_sources)]

    candidats = non_legitimes.sample(frac=1, random_state=99)

    remplacements = {}
    candidats_iter = iter(candidats.iterrows())
    for _, ligne_suspecte in suspects.iterrows():
        _, nouveau = next(candidats_iter)
        remplacements[ligne_suspecte["chunk_id"]] = nouveau
        # mettre a jour sources_bc
        masque = sources_bc["chunk_id"] == ligne_suspecte["chunk_id"]
        sources_bc.loc[masque, "chunk_id"] = nouveau["chunk_id"]
        sources_bc.loc[masque, "document"] = nouveau["document"]
        sources_bc.loc[masque, "texte"] = nouveau["texte"]

        # mettre a jour jeu_test (meme texte source = meme ligne)
        masque_jt = jeu_test["texte_1"] == ligne_suspecte["texte"]
        jeu_test.loc[masque_jt, "texte_1"] = nouveau["texte"]
        jeu_test.loc[masque_jt, "document_source_1"] = nouveau["document"]
        jeu_test.loc[masque_jt, "document_source_2"] = nouveau["document"]

        print(f"  remplace {ligne_suspecte['chunk_id']!r} -> {nouveau['chunk_id']!r}")

    sources_bc = sources_bc.drop(columns=["suspect"])
    sources_bc.to_parquet("sources_bc_a_paraphraser.parquet", index=False)
    jeu_test.to_parquet("jeu_test_etiquete.parquet", index=False)
    print(f"\n{len(remplacements)} sources remplacees. Fichiers mis a jour.")

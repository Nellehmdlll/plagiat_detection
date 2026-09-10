# -*- coding: utf-8 -*-
"""
Regenere UNIQUEMENT la categorie E (30 paires, id_paire 41-70) du jeu de
test, suite a la relecture qualitative du 2026-09-10 qui a releve :
  1. Un manque de diversite de nature (23/30 paires "figures_tableaux",
     0 institution/remerciements/sigles/sommaire) du a l'absence de
     plafond par nature dans la construction initiale.
  2. Un faux positif de contenu (paire 41) : une page de sommaire qui
     LISTE "DEDICACE" comme titre de section, appariee a tort avec une
     vraie section de dedicace - la fonction de detection de nature ne
     verifiait pas "sommaire" en priorite.

Corrige : "sommaire" verifie en premier dans deviner_nature(), et un
plafond de 6 paires par nature pour forcer la diversite. Ne touche PAS
aux categories A, B, C, D deja validees - ne modifie que les 30 lignes
de la categorie E, en conservant leurs id_paire (41-70).
"""
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import re

import numpy as np
import pandas as pd

FICHIER_JEU_TEST = "jeu_test_etiquete.parquet"
FICHIER_METADONNEES = "metadonnees_corpus.parquet"

RNG = np.random.default_rng(2028)  # nouvelle graine (2027 a re-tire la meme paire fautive)

MOTS_CLES_NATURE = {
    # Regex tolerante a "table DE matieres" vs "table DES matieres" (variante
    # orthographique reelle du corpus, ex. 05-memoire.pdf) - sinon une page de
    # sommaire listant "DEDICACE" comme titre est confondue avec une vraie
    # dedicace (defaut releve puis reproduit a la relecture du 2026-09-10).
    "sommaire": [r"table\s+des?\s+mati[eè]res", r"\bsommaire\b"],
    "dedicace": [re.escape("dédicace")], "remerciements": [re.escape("remerciement")],
    "sigles": [re.escape("sigle"), re.escape("abréviation")],
    "figures_tableaux": [re.escape("liste des figures"), re.escape("liste des tableaux")],
    # mots limites (\b) pour ne pas matcher l'anglais "institution(al)" via
    # la sous-chaine "institut" (defaut releve a la relecture du 2026-09-10 :
    # deux resumes en anglais s'etaient retrouves etiquetes "institution").
    "institution": [r"\binstitut\b", r"\binstitution\b", re.escape("présentation")],
}
MAX_PAIRES_PAR_NATURE = 6
N_E_VISEES = 30


def deviner_nature(texte):
    t = texte[:200].lower()
    for nature, motifs in MOTS_CLES_NATURE.items():
        if any(re.search(motif, t) for motif in motifs):
            return nature
    return "autre"


if __name__ == "__main__":
    df_meta = pd.read_parquet(FICHIER_METADONNEES)
    df_jeu = pd.read_parquet(FICHIER_JEU_TEST)

    e_actuelles = df_jeu[df_jeu["categorie"] == "E"]
    assert len(e_actuelles) == N_E_VISEES, f"attendu {N_E_VISEES} paires E, trouve {len(e_actuelles)}"
    id_paires_e = sorted(e_actuelles["id_paire"].tolist())
    print(f"Categorie E actuelle : id_paire {id_paires_e[0]}-{id_paires_e[-1]}, {len(e_actuelles)} paires.")

    legitimes = df_meta[df_meta["legitime"]].copy()
    legitimes["nature"] = legitimes["texte"].apply(deviner_nature)
    print("\nRepartition des chunks legitime=True par nature (avant appariement) :")
    print(legitimes["nature"].value_counts())

    paires_e_indices = []
    nb_appariees_meme_nature = 0
    repartition_retenue = {}

    for nature, groupe in legitimes.groupby("nature"):
        if nature == "autre" or len(groupe) < 2:
            continue
        indices_groupe = groupe.index.to_numpy().copy()
        RNG.shuffle(indices_groupe)
        k = 0
        nb_pour_cette_nature = 0
        while (k + 1 < len(indices_groupe) and len(paires_e_indices) < N_E_VISEES
               and nb_pour_cette_nature < MAX_PAIRES_PAR_NATURE):
            i, j = indices_groupe[k], indices_groupe[k + 1]
            if df_meta.loc[i, "document"] != df_meta.loc[j, "document"]:
                paires_e_indices.append((i, j, nature))
                nb_appariees_meme_nature += 1
                nb_pour_cette_nature += 1
            k += 2
        if nb_pour_cette_nature:
            repartition_retenue[nature] = nb_pour_cette_nature

    indices_legitimes = legitimes.index.to_numpy()
    nb_tentatives_e = 0
    while len(paires_e_indices) < N_E_VISEES and nb_tentatives_e < N_E_VISEES * 50:
        nb_tentatives_e += 1
        i, j = RNG.choice(indices_legitimes, size=2, replace=False)
        if df_meta.loc[i, "document"] == df_meta.loc[j, "document"]:
            continue
        if any(i == pi and j == pj for pi, pj, _ in paires_e_indices) or \
           any(i == pj and j == pi for pi, pj, _ in paires_e_indices):
            continue
        paires_e_indices.append((i, j, "aleatoire"))

    print(f"\n{len(paires_e_indices)}/{N_E_VISEES} paires E constituees "
          f"({nb_appariees_meme_nature} appariees par nature identique, "
          f"{len(paires_e_indices) - nb_appariees_meme_nature} appariees aleatoirement).")
    print(f"Repartition retenue par nature : {repartition_retenue}")

    # Reconstruction des 30 lignes E, en reutilisant les id_paire existants
    # (41-70) pour ne pas perturber la numerotation globale du fichier.
    nouvelles_lignes_e = []
    for (i, j, nature), id_paire in zip(paires_e_indices, id_paires_e):
        nouvelles_lignes_e.append({
            "id_paire": id_paire,
            "texte_1": df_meta.loc[i, "texte"], "texte_2": df_meta.loc[j, "texte"],
            "categorie": "E", "etiquette": "NON_PLAGIAT",
            "document_source_1": df_meta.loc[i, "document"], "document_source_2": df_meta.loc[j, "document"],
            "methode_generation": "echantillonnage",
        })
    df_e_nouvelle = pd.DataFrame(nouvelles_lignes_e)

    df_jeu_sans_e = df_jeu[df_jeu["categorie"] != "E"]
    df_jeu_final = pd.concat([df_jeu_sans_e, df_e_nouvelle], ignore_index=True).sort_values("id_paire").reset_index(drop=True)

    assert len(df_jeu_final) == len(df_jeu), "le nombre total de paires a change - anomalie"
    assert (df_jeu_final[df_jeu_final["categorie"] != "E"].reset_index(drop=True)
            .equals(df_jeu[df_jeu["categorie"] != "E"].sort_values("id_paire").reset_index(drop=True))), \
        "des lignes hors categorie E ont ete modifiees - anomalie"

    df_jeu_final.to_parquet(FICHIER_JEU_TEST, index=False)
    print(f"\n{FICHIER_JEU_TEST} mis a jour : categorie E regeneree "
          f"({len(df_jeu_final)} paires au total, inchange), A/B/C/D non modifiees.")

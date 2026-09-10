# -*- coding: utf-8 -*-
"""
Relecture qualitative d'un echantillon des categories D (sujets differents)
et E (reutilisation legitime) du jeu de test, avant commit.
"""
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import scipy.sparse as sp

RNG = np.random.default_rng(7)

MOTS_CLES_NATURE = {
    "dedicace": ["dédicace"], "remerciements": ["remerciement"],
    "sigles": ["sigle", "abréviation"], "figures_tableaux": ["liste des figures", "liste des tableaux"],
    "sommaire": ["table des matières", "sommaire"], "institution": ["institut", "présentation"],
}


def deviner_nature(texte):
    t = texte[:200].lower()
    for nature, mots in MOTS_CLES_NATURE.items():
        if any(m in t for m in mots):
            return nature
    return "autre"


if __name__ == "__main__":
    df_jeu = pd.read_parquet("jeu_test_etiquete.parquet")
    df_meta = pd.read_parquet("metadonnees_corpus.parquet")
    matrice = sp.load_npz("tfidf_corpus.npz")

    index_par_texte = pd.Series(df_meta.index, index=df_meta["texte"]).to_dict()

    def cosinus_corpus(t1, t2):
        i, j = index_par_texte.get(t1), index_par_texte.get(t2)
        if i is None or j is None:
            return None
        return float(matrice.getrow(i).multiply(matrice.getrow(j)).sum())

    print(f"{'=' * 100}\nCATEGORIE D - 10 paires au hasard (sujets differents attendus)\n{'=' * 100}")
    d = df_jeu[df_jeu["categorie"] == "D"]
    echantillon_d = d.sample(n=10, random_state=int(RNG.integers(0, 1_000_000)))
    for _, p in echantillon_d.iterrows():
        score = cosinus_corpus(p["texte_1"], p["texte_2"])
        print(f"\n--- Paire {p['id_paire']} ---")
        print(f"  doc1: {p['document_source_1']}")
        print(f"  texte_1: {p['texte_1'][:220].replace(chr(10), ' ')!r}")
        print(f"  doc2: {p['document_source_2']}")
        print(f"  texte_2: {p['texte_2'][:220].replace(chr(10), ' ')!r}")
        print(f"  score TF-IDF: {score:.4f}" if score is not None else "  score TF-IDF: n/a (texte non retrouve)")

    print(f"\n{'=' * 100}\nCATEGORIE E - 10 paires au hasard (meme nature attendue)\n{'=' * 100}")
    e = df_jeu[df_jeu["categorie"] == "E"]
    echantillon_e = e.sample(n=10, random_state=int(RNG.integers(0, 1_000_000)))
    for _, p in echantillon_e.iterrows():
        nature1 = deviner_nature(p["texte_1"])
        nature2 = deviner_nature(p["texte_2"])
        coherent = "OK (meme nature)" if nature1 == nature2 and nature1 != "autre" else \
            ("A VERIFIER" if nature1 != nature2 else "nature=autre des deux cotes, a verifier a l'oeil")
        print(f"\n--- Paire {p['id_paire']} ---")
        print(f"  doc1: {p['document_source_1']} | nature devinee: {nature1}")
        print(f"  texte_1: {p['texte_1'][:220].replace(chr(10), ' ')!r}")
        print(f"  doc2: {p['document_source_2']} | nature devinee: {nature2}")
        print(f"  texte_2: {p['texte_2'][:220].replace(chr(10), ' ')!r}")
        print(f"  coherence de nature: {coherent}")

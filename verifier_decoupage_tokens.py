# -*- coding: utf-8 -*-
"""
Verifie le decoupage en chunks base sur un vrai compte de tokens (au lieu
d'un compte de mots) sur les 3 documents de reference, et compare aux
resultats de l'ancien decoupage (mots).

A executer avec l'environnement .venv-embed (transformers y est installe) :
    .venv-embed\\Scripts\\python.exe verifier_decoupage_tokens.py
"""

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from transformers import AutoTokenizer

from inspection import extraire_texte, detecter_sommaire, detecter_lignes_repete
from extraire_titre_sommaire import extraire_titres_sommaire, filtrer_bandeau
from localiser_titres_corps import localiser_titres_dans_corps
from decouper_segments import decouper_en_segments, compteur_taille_tokenizer

CORPUS = r"C:\Users\1\Documents\MASTER_SOUTENANCE\corpus"
DOCS = [
    "SIMPORE Némata Version Finale.pdf",
    "SERE MEMOIRE MASTER IBAM.docxVF corrigée.pdf",
    "08-memoire.pdf",
]

NOM_MODELE = "dangvantuan/sentence-camembert-large"
FENETRE_REELLE_MODELE = 512  # voir tester_encodage_chunks.py pour le diagnostic
TAILLE_MAX_CHUNK_TOKENS = 400
CHEVAUCHEMENT_TOKENS = 40


def obtenir_titres_localises(nom):
    chemin = f"{CORPUS}\\{nom}"
    texte_complet = extraire_texte(chemin)
    present, nb, bloc = detecter_sommaire(texte_complet)
    bandeau_present, liste_bandeaux, _ = detecter_lignes_repete(texte_complet)
    bloc_f = filtrer_bandeau(bloc, liste_bandeaux) if bandeau_present else bloc
    titres = extraire_titres_sommaire(bloc_f)
    titres = localiser_titres_dans_corps(texte_complet, titres)
    return texte_complet, titres, liste_bandeaux


def stats(chunks, tokenizer):
    sections = {}
    for c in chunks:
        sections.setdefault(c["position_debut"], []).append(c)
    tailles_mots = [len(c["texte"].split()) for c in chunks]
    tailles_tokens = [len(tokenizer.encode(c["texte"], add_special_tokens=True)) for c in chunks]
    depassements = [c for c, t in zip(chunks, tailles_tokens) if t > FENETRE_REELLE_MODELE]
    depassements_non_legitimes = [c for c in depassements if not c["legitime"]]
    suspects = [c for c in chunks if len(c["texte"].split()) < 10]
    return {
        "nb_sections": len(sections),
        "nb_chunks": len(chunks),
        "nb_multi_chunks": sum(1 for cs in sections.values() if len(cs) > 1),
        "taille_mots_min": min(tailles_mots) if tailles_mots else 0,
        "taille_mots_max": max(tailles_mots) if tailles_mots else 0,
        "taille_tokens_min": min(tailles_tokens) if tailles_tokens else 0,
        "taille_tokens_max": max(tailles_tokens) if tailles_tokens else 0,
        "nb_depassements": len(depassements),
        "nb_depassements_non_legitimes": len(depassements_non_legitimes),
        "nb_suspects": len(suspects),
    }


if __name__ == "__main__":
    print(f"Chargement du tokenizer ({NOM_MODELE})...")
    tokenizer = AutoTokenizer.from_pretrained(NOM_MODELE)
    compteur = compteur_taille_tokenizer(tokenizer)

    resultats = {}
    for nom in DOCS:
        print(f"\n{'=' * 90}\n{nom}\n{'=' * 90}")
        texte_complet, titres, liste_bandeaux = obtenir_titres_localises(nom)

        chunks_avant = decouper_en_segments(texte_complet, titres, liste_bandeaux=liste_bandeaux)
        chunks_apres = decouper_en_segments(
            texte_complet, titres,
            taille_max_chunk=TAILLE_MAX_CHUNK_TOKENS,
            chevauchement=CHEVAUCHEMENT_TOKENS,
            compteur_taille=compteur,
            liste_bandeaux=liste_bandeaux,
        )

        s_avant = stats(chunks_avant, tokenizer)
        s_apres = stats(chunks_apres, tokenizer)
        resultats[nom] = (s_avant, s_apres, chunks_apres)

        print(f"{'':30s} {'AVANT (mots)':>15s} {'APRES (tokens)':>16s}")
        for cle, label in [
            ("nb_sections", "Sections"),
            ("nb_chunks", "Chunks"),
            ("nb_multi_chunks", "Sections multi-chunks"),
            ("taille_tokens_max", "Taille max (tokens)"),
            ("nb_depassements", "Chunks > 512 tokens"),
            ("nb_depassements_non_legitimes", "  dont legitime=False"),
            ("nb_suspects", "Chunks suspects (<10 mots)"),
        ]:
            print(f"{label:30s} {s_avant[cle]:>15} {s_apres[cle]:>16}")

    # ------------------------------------------------------------------
    # Verification du chevauchement (phrase complete, pas de troncature)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 90}\nEXEMPLE DE CHEVAUCHEMENT (nouvelle logique en tokens)\n{'=' * 90}")
    for nom, (s_avant, s_apres, chunks_apres) in resultats.items():
        sections = {}
        for c in chunks_apres:
            sections.setdefault(c["position_debut"], []).append(c)
        multi = {k: v for k, v in sections.items() if len(v) > 1}
        if not multi:
            continue
        cle = max(multi, key=lambda k: len(multi[k]))
        cs = multi[cle]
        print(f"\n--- {nom} : section g{cle}, {len(cs)} chunks ---")
        print("Fin du chunk#0 (200 derniers caracteres) :")
        print(" ", repr(cs[0]["texte"][-200:]))
        print("Debut du chunk#1 (200 premiers caracteres) :")
        print(" ", repr(cs[1]["texte"][:200]))
        break

    # ------------------------------------------------------------------
    # Detail des depassements restants (s'il y en a), legitime=False
    # ------------------------------------------------------------------
    print(f"\n{'=' * 90}\nCHUNKS ENCORE AU-DESSUS DE {FENETRE_REELLE_MODELE} TOKENS, legitime=False\n{'=' * 90}")
    trouve = False
    for nom, (s_avant, s_apres, chunks_apres) in resultats.items():
        for c in chunks_apres:
            n = len(tokenizer.encode(c["texte"], add_special_tokens=True))
            if n > FENETRE_REELLE_MODELE and not c["legitime"]:
                trouve = True
                print(f"{nom} | section={c['cle_unique_section']!r} | {n} tokens")
                print("  extrait:", repr(c["texte"][:200]))
    if not trouve:
        print("Aucun.")

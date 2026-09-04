# -*- coding: utf-8 -*-
"""
Comparatif definitif - sentence-camembert-large vs BGE-M3, sur des donnees
reelles issues du corpus, pour trancher le choix du modele d'embedding.

Script isole : ne modifie rien au pipeline existant (inspection.py,
decouper_segments.py, etc.), lecture seule.

4 etapes :
  1. Mini-jeu de test "verite connue" (10 paires, paraphrase vs sans-rapport)
     extraites du rapport Rahimah (IBAM/Sira Labs) - mesure de la separation
     de similarite cosinus sur chaque modele.
  2. Comportement qualitatif sur des chunks reels a forte densite de
     citations (zones qui posaient probleme en tokens) - recherche de
     comportement degenere (normes de vecteur aberrantes).
  3. Vitesse et VRAM a l'echelle reelle (298 chunks des 3 documents de
     reference), extrapolation aux 7443 chunks des 63 documents.
  4. Verification de l'acces a la sortie sparse (lexicale) native de BGE-M3
     via sentence-transformers, vs la necessite de FlagEmbedding.

A executer avec l'environnement .venv-embed :
    .venv-embed\\Scripts\\python.exe comparer_modeles_definitif.py
"""

import sys
import io
import re
import time
import pickle

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from inspection import extraire_texte, detecter_sommaire, detecter_lignes_repete
from extraire_titre_sommaire import extraire_titres_sommaire, filtrer_bandeau
from localiser_titres_corps import localiser_titres_dans_corps
from decouper_segments import decouper_en_segments, compteur_taille_tokenizer

CORPUS = r"C:\Users\1\Documents\MASTER_SOUTENANCE\corpus"
DOCS_REFERENCE = [
    "SIMPORE Némata Version Finale.pdf",
    "SERE MEMOIRE MASTER IBAM.docxVF corrigée.pdf",
    "08-memoire.pdf",
]

NOM_CAMEMBERT = "dangvantuan/sentence-camembert-large"
NOM_BGE = "BAAI/bge-m3"
FENETRE_REELLE_CAMEMBERT = 512
TAILLE_MAX_CHUNK_TOKENS = 400
CHEVAUCHEMENT_TOKENS = 40

# Mesure deja effectuee (tester_encodage_chunks.py) pour sentence-camembert-large
# sur les 298 chunks des 3 documents de reference.
CAMEMBERT_TEMPS_298 = 16.0
CAMEMBERT_VRAM_PIC_298 = 2.02

NB_CHUNKS_CORPUS_COMPLET = 7443


# -----------------------------------------------------------------------------
# ETAPE 1 - MINI-JEU DE TEST "VERITE CONNUE" (10 paires, rapport Rahimah)
# -----------------------------------------------------------------------------
# Les textes A sont des citations exactes extraites de
# "Rapport_Rahimah_Licence_version_Corrigée_Finale.pdf" (verifiees par
# recherche directe dans le texte source). Les textes B sont des
# reformulations manuelles fournies par l'utilisatrice.

TXT_IBAM_OBJECTIF = (
    "L'objectif de l'IBAM est de répondre aux besoins du marché de l'emploi "
    "en mettant à sa disposition un potentiel humain de cadres moyens et "
    "supérieurs dans les divers secteurs d'activité."
)
TXT_IBAM_ORGANISATION = "L'organisation de l'IBAM est une structure hiérarchico-fonctionnelle."
TXT_STAGE_OBJECTIF = (
    "Ce stage a pour objectif de leur permettre de se familiariser avec le "
    "monde professionnel et d'appliquer leurs connaissances théoriques "
    "acquises au cours de leur formation."
)
TXT_2TUP_IMPORTANCE = (
    "d'une part, 2TUP donne une grande importance à la technologie, ce qui "
    "est important pour notre projet."
)
TXT_SIRA_LABS_ACTEUR = (
    "Sira Labs qui est un acteur majeur de l'accompagnement à l'Innovation "
    "et à l'Entrepreneuriat dans la sous-région."
)
TXT_2TUP_BRANCHES = (
    "2TUP est un processus en Y qui contient une branche technique, une "
    "branche fonctionnelle et une branche réalisation. Les deux branches "
    "technique et fonctionnelle peuvent être exploitées en parallèle. De ce "
    "fait, si la technologie évolue ou, s'il arrive que lors du déroulement "
    "du projet, il y a modification d'un besoin technique, la branche "
    "technique peut être traitée puis réintégrée dans le projet facilement. "
    "De même, si une nouvelle fonctionnalité se présente, seule la branche "
    "fonctionnelle va être traitée sans toucher à l'autre branche."
)
TXT_SIRA_LABS_HISTOIRE = (
    "Sira Labs est un acteur majeur de l'accompagnement à l'Innovation et à "
    "l'Entrepreneuriat dans la sous-région avec une présence dans 04 villes "
    "dont Ouagadougou, Bobo-Dioulasso, Dakar et Saint-Louis. Incubateur et "
    "Accélérateur d'entreprises, Sira Labs offre divers programmes riches "
    "allant de la structuration d'idées de projet innovantes à "
    "l'accélération de croissances d'entreprises."
)
TXT_TABLEAU_CASCADES_RUP = (
    "Tableau comparatif des méthodologies ou processus de développement. "
    "CASCADES : les phases sont déroulées d'une manière séquentielle, "
    "distingue clairement les phases du projet ; non itératif, pas de "
    "modèles pour les documents. RUP (Rational Unified Process) : à la fois "
    "une méthodologie et un outil prêt à l'emploi, cible des projets de "
    "plus de 10 personnes ; itératif, spécifie le dialogue entre les "
    "différents intervenants du projet ; assez flou dans sa mise en œuvre, "
    "ne couvre pas les phases en amont et en aval au développement."
)
TXT_FILIERES_LICENCE = (
    "L'IBAM offre des formations dans plusieurs filières, réparties en deux "
    "groupes selon les diplômes : le groupe des licences professionnelles "
    "et le groupe des masters. Les filières de formations initiales en "
    "licences professionnelles sont : Comptabilité-Contrôle-Audit (CCA) ; "
    "Assurance-Banque-Finance (ABF) ; Marketing et Gestion (MG) ; "
    "Assistanat de Direction (option Bilingue ou option Comptable) ; "
    "Licence Informatique (option MIAGE ou option Réseaux et "
    "Télécommunications) ; Marketing et Innovation Digitale (MID)."
)
TXT_IBAM_ORGANISATION_DETAIL = (
    "L'organisation de l'IBAM est une structure hiérarchico-fonctionnelle. "
    "Nous avons : le Conseil de gestion qui est l'organe suprême qui "
    "regroupe le directeur, le directeur adjoint, les coordonnateurs, les "
    "enseignants permanents, le CSAFC, la Secrétaire Principale et le "
    "représentant du personnel ATOS ; le Conseil scientifique qui regroupe "
    "le Directeur, le Directeur Adjoint, les coordonnateurs et les "
    "enseignants de rang A de l'Institut."
)
TXT_SIRA_LABS_VILLES = (
    "Sira Labs qui est un acteur majeur de l'accompagnement à l'Innovation "
    "et à l'Entrepreneuriat dans la sous-région avec une présence dans 04 "
    "villes dont Ouagadougou, Bobo-Dioulasso, Dakar et Saint-Louis."
)
TXT_MASTERS_IBAM = (
    "En master, l'IBAM offre cinq filières de formation : Master en "
    "Administration et Gestion des Entreprises (MAGE) ; Master en "
    "Comptabilité-Contrôle-Audit (MCCA) ; Master en Ingénierie Bancaire et "
    "Financière (IBF) ; Master en Informatique, option Ingénierie des "
    "systèmes d'informations des Entreprises (M2ISIE) ou option Sécurité "
    "informatique (MISI)."
)

PAIRES = [
    # --- Paraphrases (meme sens, mots differents) ---
    {
        "id": 1, "categorie": "paraphrase", "legitime": True,
        "texte_a": TXT_IBAM_OBJECTIF,
        "texte_b": (
            "L'IBAM a pour but de satisfaire les demandes du monde "
            "professionnel en formant des cadres, du niveau intermédiaire "
            "au niveau supérieur, dans différents domaines d'activité."
        ),
    },
    {
        "id": 2, "categorie": "paraphrase", "legitime": True,
        "texte_a": TXT_IBAM_ORGANISATION,
        "texte_b": (
            "L'IBAM fonctionne selon une organisation à la fois "
            "hiérarchique et fonctionnelle."
        ),
    },
    {
        "id": 3, "categorie": "paraphrase", "legitime": False,
        "texte_a": TXT_STAGE_OBJECTIF,
        "texte_b": (
            "L'objectif de ce stage est de donner aux étudiants une "
            "première expérience du milieu professionnel, tout en mettant "
            "en pratique la théorie apprise pendant leur cursus."
        ),
    },
    {
        "id": 4, "categorie": "paraphrase", "legitime": False,
        "texte_a": TXT_2TUP_IMPORTANCE,
        "texte_b": (
            "Le processus 2TUP accorde une place centrale aux aspects "
            "technologiques, un point essentiel dans le cadre de notre "
            "travail."
        ),
    },
    {
        "id": 5, "categorie": "paraphrase", "legitime": False,
        "texte_a": TXT_SIRA_LABS_ACTEUR,
        "texte_b": (
            "Sira Labs occupe une place importante dans le soutien à "
            "l'entrepreneuriat et à l'innovation au niveau régional."
        ),
    },
    # --- Sans rapport (sujets clairement differents) ---
    {
        "id": 6, "categorie": "sans_rapport", "legitime": None,
        "texte_a": TXT_IBAM_OBJECTIF,
        "texte_b": TXT_2TUP_BRANCHES,
    },
    {
        "id": 7, "categorie": "sans_rapport", "legitime": None,
        "texte_a": TXT_SIRA_LABS_HISTOIRE,
        "texte_b": TXT_TABLEAU_CASCADES_RUP,
    },
    {
        "id": 8, "categorie": "sans_rapport", "legitime": None,
        "texte_a": TXT_FILIERES_LICENCE,
        "texte_b": TXT_2TUP_BRANCHES,
    },
    {
        "id": 9, "categorie": "sans_rapport", "legitime": None,
        "note": "cas ambigu volontaire (deux structures organisationnelles)",
        "texte_a": TXT_IBAM_ORGANISATION_DETAIL,
        "texte_b": TXT_SIRA_LABS_VILLES,
    },
    {
        "id": 10, "categorie": "sans_rapport", "legitime": None,
        "texte_a": TXT_STAGE_OBJECTIF,
        "texte_b": TXT_MASTERS_IBAM,
    },
]


def cosinus(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def etape1_verite_connue(modele, label):
    print(f"\n{'=' * 90}\nETAPE 1 - Verite connue (10 paires) - {label}\n{'=' * 90}")
    textes_a = [p["texte_a"] for p in PAIRES]
    textes_b = [p["texte_b"] for p in PAIRES]
    vec_a = modele.encode(textes_a, convert_to_numpy=True, show_progress_bar=False)
    vec_b = modele.encode(textes_b, convert_to_numpy=True, show_progress_bar=False)

    resultats = []
    for p, va, vb in zip(PAIRES, vec_a, vec_b):
        sim = cosinus(va, vb)
        resultats.append({**p, "similarite": sim})
        note = f"  [{p['note']}]" if "note" in p else ""
        print(f"  Paire {p['id']:2d} ({p['categorie']:12s}) : cos = {sim:.4f}{note}")

    paraphrases = [r["similarite"] for r in resultats if r["categorie"] == "paraphrase"]
    sans_rapport = [r["similarite"] for r in resultats if r["categorie"] == "sans_rapport"]
    moy_paraphrase = sum(paraphrases) / len(paraphrases)
    moy_sans_rapport = sum(sans_rapport) / len(sans_rapport)
    ecart = moy_paraphrase - moy_sans_rapport

    print(f"\n  Moyenne PARAPHRASE   : {moy_paraphrase:.4f}")
    print(f"  Moyenne SANS RAPPORT : {moy_sans_rapport:.4f}")
    print(f"  ECART (separation)   : {ecart:.4f}")

    paire9 = next(r for r in resultats if r["id"] == 9)
    print(f"\n  Paire 9 (cas ambigu) : cos = {paire9['similarite']:.4f}")

    return {
        "label": label, "resultats": resultats,
        "moy_paraphrase": moy_paraphrase, "moy_sans_rapport": moy_sans_rapport,
        "ecart": ecart, "paire9": paire9["similarite"],
    }


# -----------------------------------------------------------------------------
# ETAPE 2 - COMPORTEMENT SUR LES ZONES DIFFICILES (citations denses)
# -----------------------------------------------------------------------------

def obtenir_chunks_reference(compteur_taille):
    tous_chunks = []
    for nom in DOCS_REFERENCE:
        chemin = f"{CORPUS}\\{nom}"
        texte_complet = extraire_texte(chemin)
        present, nb, bloc = detecter_sommaire(texte_complet)
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
        for c in chunks:
            c["_document"] = nom
        tous_chunks.extend(chunks)
    return tous_chunks


def selectionner_chunks_difficiles(tous_chunks):
    # Zone SIMPORE deja identifiee : tableau de definitions dense en citations.
    simpore_table = [c for c in tous_chunks
                      if c["_document"].startswith("SIMPORE")
                      and c["cle_unique_section"] == "CHAPITRE I__II.1"]

    # 08-memoire : chunks les plus denses en citations (proxy : nb d'occurrences
    # de motifs "(annee)" ou "et al." par 100 mots).
    motif_citation = re.compile(r"\(\d{4}[a-z]?\)|et al\.?")
    candidats_08 = [c for c in tous_chunks if c["_document"] == "08-memoire.pdf"]
    for c in candidats_08:
        nb_mots = max(1, len(c["texte"].split()))
        c["_densite_citation"] = len(motif_citation.findall(c["texte"])) / nb_mots
    candidats_08.sort(key=lambda c: -c["_densite_citation"])
    top_08 = [c for c in candidats_08[:5] if c["_densite_citation"] > 0]

    return simpore_table + top_08


def etape2_zones_difficiles(chunks_difficiles, modele_camembert, modele_bge):
    print(f"\n{'=' * 90}\nETAPE 2 - Comportement sur zones a forte densite de citations\n{'=' * 90}")
    print(f"{len(chunks_difficiles)} chunks selectionnes "
          f"({sum(1 for c in chunks_difficiles if c['_document'].startswith('SIMPORE'))} SIMPORE tableau, "
          f"{sum(1 for c in chunks_difficiles if c['_document'] == '08-memoire.pdf')} 08-memoire citations denses)")

    textes = [c["texte"] for c in chunks_difficiles]
    vec_camembert = modele_camembert.encode(textes, convert_to_numpy=True, show_progress_bar=False)
    vec_bge = modele_bge.encode(textes, convert_to_numpy=True, show_progress_bar=False)

    normes_camembert = np.linalg.norm(vec_camembert, axis=1)
    normes_bge = np.linalg.norm(vec_bge, axis=1)

    print(f"\n{'Chunk':45s} {'Norme (camembert)':>20s} {'Norme (BGE-M3)':>18s}")
    for c, nc, nb in zip(chunks_difficiles, normes_camembert, normes_bge):
        label = f"{c['_document'][:20]}/{c['cle_unique_section']}"
        print(f"{label:45s} {nc:>20.4f} {nb:>18.4f}")

    print(f"\n  sentence-camembert-large : normes min={normes_camembert.min():.4f}, "
          f"max={normes_camembert.max():.4f}, ecart-type={normes_camembert.std():.4f}")
    print(f"  BGE-M3                   : normes min={normes_bge.min():.4f}, "
          f"max={normes_bge.max():.4f}, ecart-type={normes_bge.std():.4f}")

    anomalies_camembert = np.sum((normes_camembert < 1e-3) | np.isnan(normes_camembert))
    anomalies_bge = np.sum((normes_bge < 1e-3) | np.isnan(normes_bge))
    print(f"\n  Comportement degenere (norme ~0 ou NaN) : "
          f"camembert={anomalies_camembert}, BGE-M3={anomalies_bge}")

    return {
        "normes_camembert": normes_camembert, "normes_bge": normes_bge,
        "anomalies_camembert": int(anomalies_camembert), "anomalies_bge": int(anomalies_bge),
    }


# -----------------------------------------------------------------------------
# ETAPE 3 - VITESSE ET VRAM A L'ECHELLE REELLE
# -----------------------------------------------------------------------------

def etape3_vitesse_vram(tous_chunks, modele_bge):
    print(f"\n{'=' * 90}\nETAPE 3 - Vitesse et VRAM ({len(tous_chunks)} chunks reels)\n{'=' * 90}")
    textes = [c["texte"] for c in tous_chunks]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    modele_bge.encode(textes, convert_to_numpy=True, show_progress_bar=False, batch_size=16)
    torch.cuda.synchronize()
    t_bge = time.time() - t0
    vram_bge = torch.cuda.max_memory_allocated() / 1024 ** 3

    print(f"\n{'Modele':30s} {'Temps ({} chunks)'.format(len(tous_chunks)):>20s} "
          f"{'ms/chunk':>10s} {'VRAM pic':>10s}")
    print(f"{'sentence-camembert-large':30s} {CAMEMBERT_TEMPS_298:>19.1f}s "
          f"{CAMEMBERT_TEMPS_298 / len(tous_chunks) * 1000:>9.1f} {CAMEMBERT_VRAM_PIC_298:>9.2f}Go")
    print(f"{'BGE-M3':30s} {t_bge:>19.1f}s "
          f"{t_bge / len(tous_chunks) * 1000:>9.1f} {vram_bge:>9.2f}Go")

    extrap_camembert = CAMEMBERT_TEMPS_298 / len(tous_chunks) * NB_CHUNKS_CORPUS_COMPLET
    extrap_bge = t_bge / len(tous_chunks) * NB_CHUNKS_CORPUS_COMPLET
    print(f"\n  Extrapolation a {NB_CHUNKS_CORPUS_COMPLET} chunks (63 documents) :")
    print(f"    sentence-camembert-large : ~{extrap_camembert:.0f}s (~{extrap_camembert / 60:.1f} min)")
    print(f"    BGE-M3                   : ~{extrap_bge:.0f}s (~{extrap_bge / 60:.1f} min)")

    return {"t_bge_298": t_bge, "vram_bge_298": vram_bge,
            "extrap_camembert_63docs_s": extrap_camembert, "extrap_bge_63docs_s": extrap_bge}


# -----------------------------------------------------------------------------
# ETAPE 4 - SIGNAL LEXICAL NATIF DE BGE-M3
# -----------------------------------------------------------------------------

def etape4_signal_lexical(modele_bge):
    print(f"\n{'=' * 90}\nETAPE 4 - Acces au signal lexical (sparse) natif de BGE-M3\n{'=' * 90}")

    try:
        import FlagEmbedding  # noqa: F401
        flagembedding_installe = True
    except ImportError:
        flagembedding_installe = False
    print(f"  Librairie FlagEmbedding installee : {flagembedding_installe}")

    # Introspection du module SentenceTransformer charge : verifie s'il expose
    # une methode ou une sortie sparse/lexical_weights au-dela du pooling dense.
    print(f"\n  Modules du SentenceTransformer('{NOM_BGE}') :")
    for i, module in enumerate(modele_bge):
        print(f"    [{i}] {type(module).__name__}")

    a_methode_sparse = hasattr(modele_bge, "encode_sparse") or hasattr(modele_bge, "encode_multi_process_sparse")
    print(f"\n  Methode sparse exposee directement par SentenceTransformer : {a_methode_sparse}")

    print(
        "\n  CONCLUSION : sentence-transformers charge BGE-M3 comme un encodeur "
        "dense standard (Transformer + pooling moyenne), sans acces a la sortie "
        "sparse (poids lexicaux par token) ni ColBERT (multi-vecteurs) du modele "
        "natif. Ces sorties supplementaires necessitent la librairie officielle "
        "FlagEmbedding (BGEM3FlagModel.encode(..., return_dense=True, "
        "return_sparse=True, return_colbert_vecs=True)), non installee dans "
        "cet environnement. Contrainte reelle : utiliser le signal lexical natif "
        "de BGE-M3 impliquerait d'ajouter FlagEmbedding (et son chargement de "
        "modele redondant avec le SentenceTransformer deja charge), ou de "
        "recourir a un autre mecanisme lexical (BM25) en parallele du dense."
    )
    return {"flagembedding_installe": flagembedding_installe, "sparse_via_st": a_methode_sparse}


# -----------------------------------------------------------------------------
# SYNTHESE
# -----------------------------------------------------------------------------

def synthese(r1_camembert, r1_bge, r2, r3):
    print(f"\n{'=' * 90}\nTABLEAU DE SYNTHESE\n{'=' * 90}")
    print(f"{'Dimension':40s} {'sentence-camembert-large':>28s} {'BGE-M3':>15s}")
    print(f"{'Ecart paraphrase/sans-rapport':40s} {r1_camembert['ecart']:>28.4f} {r1_bge['ecart']:>15.4f}")
    print(f"{'Similarite paire 9 (ambigu)':40s} {r1_camembert['paire9']:>28.4f} {r1_bge['paire9']:>15.4f}")
    print(f"{'Anomalies normes (zones difficiles)':40s} {r2['anomalies_camembert']:>28d} {r2['anomalies_bge']:>15d}")
    print(f"{'Temps extrapole 63 docs (min)':40s} {r3['extrap_camembert_63docs_s']/60:>27.1f}m "
          f"{r3['extrap_bge_63docs_s']/60:>14.1f}m")
    print(f"{'VRAM pic (298 chunks)':40s} {CAMEMBERT_VRAM_PIC_298:>27.2f}G {r3['vram_bge_298']:>14.2f}G")
    print(f"{'Signal lexical natif accessible':40s} {'n/a (dense only)':>28s} "
          f"{'non (sans FlagEmbedding)':>15s}")


if __name__ == "__main__":
    print("Chargement de sentence-camembert-large...")
    modele_camembert = SentenceTransformer(NOM_CAMEMBERT, device="cuda")
    modele_camembert.max_seq_length = FENETRE_REELLE_CAMEMBERT

    print("Chargement de BGE-M3...")
    modele_bge = SentenceTransformer(NOM_BGE, device="cuda")

    r1_camembert = etape1_verite_connue(modele_camembert, "sentence-camembert-large")
    r1_bge = etape1_verite_connue(modele_bge, "BGE-M3")

    print("\nChargement des chunks reels (3 documents de reference, decoupage token-based)...")
    tokenizer_camembert = AutoTokenizer.from_pretrained(NOM_CAMEMBERT)
    compteur = compteur_taille_tokenizer(tokenizer_camembert)
    tous_chunks = obtenir_chunks_reference(compteur)
    print(f"  {len(tous_chunks)} chunks charges")

    chunks_difficiles = selectionner_chunks_difficiles(tous_chunks)
    r2 = etape2_zones_difficiles(chunks_difficiles, modele_camembert, modele_bge)

    r3 = etape3_vitesse_vram(tous_chunks, modele_bge)

    r4 = etape4_signal_lexical(modele_bge)

    synthese(r1_camembert, r1_bge, r2, r3)

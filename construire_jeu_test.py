# -*- coding: utf-8 -*-
"""
Construction du jeu de test etiquete (~180 paires) pour le calcul de
precision/rappel/F1, en complement du mini-jeu de calibrage (10 paires,
deja valide - voir calibrer_alpha_fusion.py) qui reste inchange et fait
partie du jeu de test complet comme noyau annote manuellement.

REPARTITION VISEE :
  A (25) - copie litterale quasi identique (1-2 mots changes)  -> PLAGIAT
  B (30) - modification legere (modele de langage)             -> PLAGIAT
  C (40) - paraphrase profonde (modele de langage)              -> PLAGIAT
  D (40) - sujets differents (echantillonnage + verif TF-IDF)  -> NON-PLAGIAT
  E (30) - reutilisation legitime (chunks legitime=True)       -> NON-PLAGIAT
  + 10 paires du mini-jeu de calibrage existant (noyau manuel), reutilisees
    telles quelles depuis calibrer_alpha_fusion.py.
  Total vise : 175 (l'instruction parle de "15 paires" pour le noyau
  existant, mais celui-ci en compte reellement 10 - voir la note dans le
  rapport de generation plutot que d'en fabriquer 5 de plus a la main,
  ce qui contredirait consigne "deja fait, ne pas refaire").

GENERATION DES CATEGORIES B/C : necessite un modele de langage (etape non
scriptable automatiquement de bout en bout). Ce script selectionne les 70
passages source (30 pour B, 40 pour C) de facon deterministe et les
sauvegarde ; les reformulations elles-memes sont produites par un modele
de langage (Claude Sonnet 5, Anthropic) suivant EXACTEMENT les consignes
documentees ci-dessous, en deux temps : un echantillon de validation (5 B
+ 5 C) d'abord, le reste seulement apres validation de la qualite par
l'utilisatrice (voir rapport de generation affiche par ce script).

A executer avec l'environnement .venv-embed :
    .venv-embed\\Scripts\\python.exe construire_jeu_test.py
"""

import sys
import io
import re
import json
from datetime import date

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import scipy.sparse as sp

FICHIER_METADONNEES = "metadonnees_corpus.parquet"
FICHIER_MATRICE_TFIDF = "tfidf_corpus.npz"
FICHIER_JEU_TEST = "jeu_test_etiquete.parquet"
FICHIER_RAPPORT_GENERATION = "rapport_generation_jeu_test.json"

DATE_GENERATION = "2026-09-10"
MODELE_GENERATION = "Claude Sonnet 5 (Anthropic), via Claude Code"

PROMPT_CATEGORIE_B = (
    "Reformule ce texte avec un changement modéré : remplace 30 à 50% des mots "
    "par des synonymes ou tournures différentes, réordonne au moins une "
    "proposition ou change la structure d'au moins une phrase, mais reste plus "
    "proche de l'original que ne le serait une paraphrase complète (catégorie "
    "C). Le texte doit rester clairement reconnaissable comme dérivé de "
    "l'original par sa structure générale, tout en montrant un effort de "
    "reformulation net, pas de simples synonymes isolés."
)  # v2, corrige apres relecture du 2026-09-10 : le v1 ("reformule legerement...
   # garde la structure tres proche") produisait des variantes trop proches de
   # la categorie A (synonymes isoles, structure identique a 95%+), pas assez
   # distinctes pour etre informatives dans le jeu de test.
PROMPT_CATEGORIE_C = (
    "reformule entièrement ce texte en gardant exactement le même sens, "
    "change le vocabulaire et la structure des phrases autant que possible"
)

SEUIL_TFIDF_NON_APPARENTE = 0.05  # categorie D : rejeter les paires au-dessus

RNG = np.random.default_rng(2026)

# -----------------------------------------------------------------------------
# Heuristique de discipline (approximative, indicative uniquement - le
# pipeline ne porte pas d'etiquette de discipline officielle) : mots-cles
# recherches dans le nom de fichier + un extrait du texte.
# -----------------------------------------------------------------------------

MOTS_CLES_DISCIPLINE = {
    "sante": ["patient", "santé", "médical", "infirmi", "soin", "hôpital", "chup",
              "maladie", "clinique", "pharma", "grossesse", "sage-femme", "accouch",
              "paludisme", "vih", "sida", "contracepti"],
    "comptabilite_audit": ["comptab", "audit", "contrôle interne", "coso", "finance",
                           "ibam", "bilan", "mcca", "trésorerie", "fiscal"],
    "informatique": ["algorithme", "informatique", "réseau de neurone", "base de données",
                     "logiciel", "système d'information", "programmation", "2tup",
                     "uml", "cnn", "apprentissage automatique", "tokenisation"],
}


def deviner_discipline(nom_document, texte_extrait):
    contexte = (nom_document + " " + texte_extrait[:500]).lower()
    for discipline, mots_cles in MOTS_CLES_DISCIPLINE.items():
        if any(mc in contexte for mc in mots_cles):
            return discipline
    return "autre"


# -----------------------------------------------------------------------------
# Categorie A - copie quasi identique (1-2 mots changes)
# -----------------------------------------------------------------------------

SYNONYMES_A = {
    "cependant": "toutefois", "important": "essentiel", "étude": "analyse",
    "résultats": "conclusions", "montre": "révèle", "permet": "autorise",
    "notamment": "en particulier", "ainsi": "de ce fait", "objectif": "but",
    "principal": "premier", "différent": "distinct", "nombreux": "multiples",
    "actuellement": "aujourd'hui", "plusieurs": "divers", "également": "aussi",
    "grande": "importante", "grand": "important", "utilise": "emploie",
    "concerne": "porte sur", "constitue": "représente", "présente": "expose",
}


def creer_variante_quasi_identique(texte, max_remplacements=2):
    """
    Applique 1 a 2 remplacements ponctuels (dictionnaire de synonymes
    documente ci-dessus) pour produire une variante "quasi identique" au
    sens de la categorie A. Deterministe et reproductible (meme texte en
    entree -> meme sortie).
    """
    texte_modifie = texte
    nb_remplacements = 0
    mots_remplaces = []
    for mot, synonyme in SYNONYMES_A.items():
        if nb_remplacements >= max_remplacements:
            break
        motif = re.compile(r"\b" + re.escape(mot) + r"\b", re.IGNORECASE)
        if motif.search(texte_modifie):
            texte_modifie = motif.sub(synonyme, texte_modifie, count=1)
            mots_remplaces.append((mot, synonyme))
            nb_remplacements += 1

    if nb_remplacements == 0:
        # repli : aucun mot du dictionnaire trouve - remplacement universel
        # " et " -> " ainsi que " (une occurrence), quasiment toujours present
        texte_modifie2 = re.sub(r"\bet\b", "ainsi que", texte, count=1)
        if texte_modifie2 != texte:
            texte_modifie = texte_modifie2
            mots_remplaces.append(("et", "ainsi que (repli)"))

    if texte_modifie == texte:
        # dernier repli : texte source tres degrade (mots colles par une
        # extraction PDF defaillante), ni le dictionnaire ni "et" isole ne
        # s'appliquent - on modifie la ponctuation finale du premier point
        # rencontre pour garantir que la paire n'est jamais texte_1==texte_2.
        texte_modifie = texte.replace(".", " .", 1)
        mots_remplaces.append(("(ponctuation)", "espace insere (dernier repli)"))
        if texte_modifie == texte:
            raise ValueError(
                "creer_variante_quasi_identique n'a trouve aucune modification "
                "possible, meme en dernier repli - texte source a inspecter."
            )

    return texte_modifie, mots_remplaces


# -----------------------------------------------------------------------------
# Chargement des donnees
# -----------------------------------------------------------------------------

print(f"Chargement de {FICHIER_METADONNEES} et {FICHIER_MATRICE_TFIDF}...")
df_meta = pd.read_parquet(FICHIER_METADONNEES)
matrice_tfidf = sp.load_npz(FICHIER_MATRICE_TFIDF)
print(f"{len(df_meta)} chunks ({(~df_meta['legitime']).sum()} non-legitimes, "
      f"{df_meta['legitime'].sum()} legitimes) sur {df_meta['document'].nunique()} documents.")

df_meta["discipline"] = [
    deviner_discipline(d, t) for d, t in zip(df_meta["document"], df_meta["texte"])
]
print("\nRepartition (approximative) par discipline detectee :")
print(df_meta["discipline"].value_counts())

non_legitimes = df_meta[~df_meta["legitime"]].copy()
legitimes = df_meta[df_meta["legitime"]].copy()

# Filtre de taille minimale (eviter les chunks trop courts, peu informatifs,
# pour des paires de test qui doivent avoir un contenu substantiel).
non_legitimes = non_legitimes[non_legitimes["texte"].str.split().str.len() >= 40]

# Filtre anti-tableau (densite de chiffres) : un tableau de budget/planning
# extrait en texte brut n'est pas un contenu paraphrasable au sens de B/C -
# ecarte des sources potentielles les chunks trop denses en chiffres bruts.
def _densite_chiffres(texte):
    return sum(ch.isdigit() for ch in texte) / max(1, len(texte))


non_legitimes_prose = non_legitimes[
    non_legitimes["texte"].apply(_densite_chiffres) <= 0.06
]

paires = []
id_paire = 0


def ajouter_paire(texte_1, texte_2, categorie, etiquette, doc_1, doc_2, methode):
    global id_paire
    id_paire += 1
    paires.append({
        "id_paire": id_paire, "texte_1": texte_1, "texte_2": texte_2,
        "categorie": categorie, "etiquette": etiquette,
        "document_source_1": doc_1, "document_source_2": doc_2,
        "methode_generation": methode,
    })


# -----------------------------------------------------------------------------
# CATEGORIE D (40) - sujets differents, verifies non-apparentes par TF-IDF
# -----------------------------------------------------------------------------

print(f"\n{'=' * 90}\nCATEGORIE D - sujets differents\n{'=' * 90}")
indices_non_legitimes = non_legitimes.index.to_numpy()
n_d_visees = 40
paires_d_indices = []
nb_rejetees_tfidf = 0
nb_tentatives = 0
docs_deja_pairees_d = set()

while len(paires_d_indices) < n_d_visees and nb_tentatives < n_d_visees * 50:
    nb_tentatives += 1
    i, j = RNG.choice(indices_non_legitimes, size=2, replace=False)
    doc_i, doc_j = df_meta.loc[i, "document"], df_meta.loc[j, "document"]
    if doc_i == doc_j:
        continue
    cle_paire_docs = tuple(sorted([doc_i, doc_j]))
    # limite la repetition de la meme paire de documents pour la diversite
    if list(docs_deja_pairees_d).count(cle_paire_docs) >= 2:
        continue
    cos = matrice_tfidf.getrow(i).multiply(matrice_tfidf.getrow(j)).sum()
    if cos > SEUIL_TFIDF_NON_APPARENTE:
        nb_rejetees_tfidf += 1
        continue
    paires_d_indices.append((i, j))
    docs_deja_pairees_d.add(cle_paire_docs)

print(f"{len(paires_d_indices)}/{n_d_visees} paires D constituees en {nb_tentatives} tirages "
      f"({nb_rejetees_tfidf} rejetees pour similarite TF-IDF > {SEUIL_TFIDF_NON_APPARENTE}).")

for i, j in paires_d_indices:
    ajouter_paire(
        df_meta.loc[i, "texte"], df_meta.loc[j, "texte"], "D", "NON_PLAGIAT",
        df_meta.loc[i, "document"], df_meta.loc[j, "document"], "echantillonnage",
    )

# -----------------------------------------------------------------------------
# CATEGORIE E (30) - reutilisation legitime (legitime=True, documents differents)
# -----------------------------------------------------------------------------

print(f"\n{'=' * 90}\nCATEGORIE E - reutilisation legitime\n{'=' * 90}")

MOTS_CLES_NATURE = {
    # "sommaire" verifie en PREMIER : une page de table des matieres qui se
    # contente de LISTER "dedicace"/"remerciements" comme titres de section
    # (ex. "DEDICACE....I") ne doit pas etre confondue avec le contenu reel
    # de ces sections - sinon on apparie a tort un sommaire avec une vraie
    # dedicace (defaut releve a la relecture du 2026-09-10). Regex tolerante
    # a "table DE matieres" vs "table DES matieres" (variante orthographique
    # reellement rencontree dans le corpus, ex. 05-memoire.pdf).
    "sommaire": [r"table\s+des?\s+mati[eè]res", r"\bsommaire\b"],
    "dedicace": [re.escape("dédicace")], "remerciements": [re.escape("remerciement")],
    "sigles": [re.escape("sigle"), re.escape("abréviation")],
    "figures_tableaux": [re.escape("liste des figures"), re.escape("liste des tableaux")],
    # mots limites (\b) pour ne pas matcher l'anglais "institution(al)" via
    # la sous-chaine "institut" (defaut releve a la relecture du 2026-09-10 :
    # deux resumes en anglais s'etaient retrouves etiquetes "institution").
    "institution": [r"\binstitut\b", r"\binstitution\b", re.escape("présentation")],
}
MAX_PAIRES_PAR_NATURE = 6  # evite qu'une nature volumineuse (ex. listes de
# figures/tableaux, presente dans quasi tous les documents) monopolise le
# quota au detriment de la diversite de natures visee par la categorie E.


def deviner_nature(texte):
    t = texte[:200].lower()
    for nature, motifs in MOTS_CLES_NATURE.items():
        if any(re.search(motif, t) for motif in motifs):
            return nature
    return "autre"


legitimes["nature"] = legitimes["texte"].apply(deviner_nature)
n_e_visees = 30
paires_e_indices = []
nb_appariees_meme_nature = 0

# Priorite : apparier par nature identique quand possible, plafonne par
# nature pour garantir une diversite de natures dans les 30 paires.
for nature, groupe in legitimes.groupby("nature"):
    if nature == "autre" or len(groupe) < 2:
        continue
    indices_groupe = groupe.index.to_numpy().copy()
    RNG.shuffle(indices_groupe)
    k = 0
    nb_pour_cette_nature = 0
    while (k + 1 < len(indices_groupe) and len(paires_e_indices) < n_e_visees
           and nb_pour_cette_nature < MAX_PAIRES_PAR_NATURE):
        i, j = indices_groupe[k], indices_groupe[k + 1]
        if df_meta.loc[i, "document"] != df_meta.loc[j, "document"]:
            paires_e_indices.append((i, j))
            nb_appariees_meme_nature += 1
            nb_pour_cette_nature += 1
        k += 2

# Complement aleatoire (documents differents) si le quota n'est pas atteint
indices_legitimes = legitimes.index.to_numpy()
nb_tentatives_e = 0
while len(paires_e_indices) < n_e_visees and nb_tentatives_e < n_e_visees * 50:
    nb_tentatives_e += 1
    i, j = RNG.choice(indices_legitimes, size=2, replace=False)
    if df_meta.loc[i, "document"] == df_meta.loc[j, "document"]:
        continue
    if (i, j) in paires_e_indices or (j, i) in paires_e_indices:
        continue
    paires_e_indices.append((i, j))

print(f"{len(paires_e_indices)}/{n_e_visees} paires E constituees "
      f"({nb_appariees_meme_nature} appariees par nature identique, "
      f"{len(paires_e_indices) - nb_appariees_meme_nature} appariees aleatoirement).")

for i, j in paires_e_indices:
    ajouter_paire(
        df_meta.loc[i, "texte"], df_meta.loc[j, "texte"], "E", "NON_PLAGIAT",
        df_meta.loc[i, "document"], df_meta.loc[j, "document"], "echantillonnage",
    )

# -----------------------------------------------------------------------------
# CATEGORIE A (25) - copie quasi identique
# -----------------------------------------------------------------------------

print(f"\n{'=' * 90}\nCATEGORIE A - copie litterale quasi identique\n{'=' * 90}")
n_a_visees = 25
# diversite : au plus 1 chunk par document pour repartir sur des documents varies
non_legitimes_melanges = non_legitimes.sample(frac=1, random_state=7)
sources_a = non_legitimes_melanges.drop_duplicates(subset="document").head(n_a_visees)
print(f"{len(sources_a)}/{n_a_visees} sources A selectionnees (1 par document, documents varies).")

indices_utilises_a = set(sources_a.index)
nb_avec_repli = 0
for idx, ligne in sources_a.iterrows():
    variante, mots_remplaces = creer_variante_quasi_identique(ligne["texte"])
    if any("repli" in str(m[1]) for m in mots_remplaces):
        nb_avec_repli += 1
    ajouter_paire(
        ligne["texte"], variante, "A", "PLAGIAT",
        ligne["document"], ligne["document"], "manuel_scripte",
    )
print(f"  {nb_avec_repli}/{len(sources_a)} ont utilise le remplacement de repli "
      f"(aucun mot du dictionnaire de synonymes trouve).")

# -----------------------------------------------------------------------------
# CATEGORIES B (30) et C (40) - sources selectionnees, reformulation LLM
# -----------------------------------------------------------------------------

print(f"\n{'=' * 90}\nCATEGORIES B/C - selection des 70 passages source\n{'=' * 90}")
n_b_visees, n_c_visees = 30, 40
non_legitimes_restants = non_legitimes_prose.drop(
    index=[i for i in indices_utilises_a if i in non_legitimes_prose.index]
).sample(frac=1, random_state=13)
sources_bc = non_legitimes_restants.drop_duplicates(subset="document").head(n_b_visees + n_c_visees)
if len(sources_bc) < n_b_visees + n_c_visees:
    # complete sans la contrainte "1 par document" si le corpus ne fournit pas
    # assez de documents distincts restants
    complement = non_legitimes_restants.drop(index=sources_bc.index).head(
        n_b_visees + n_c_visees - len(sources_bc)
    )
    sources_bc = pd.concat([sources_bc, complement])

sources_b = sources_bc.iloc[:n_b_visees]
sources_c = sources_bc.iloc[n_b_visees:n_b_visees + n_c_visees]
print(f"{len(sources_b)} sources B, {len(sources_c)} sources C selectionnees "
      f"(documents distincts des sources A, {sources_bc['document'].nunique()} documents distincts au total).")

sources_bc_a_paraphraser = pd.concat([
    sources_b.assign(categorie_cible="B"),
    sources_c.assign(categorie_cible="C"),
])[["document", "categorie_cible", "chunk_id", "texte"]]
sources_bc_a_paraphraser.to_parquet("sources_bc_a_paraphraser.parquet", index=False)
print("Sources B/C sauvegardees dans sources_bc_a_paraphraser.parquet "
      "(pour generation par modele de langage, hors de ce script).")

# -----------------------------------------------------------------------------
# Assemblage : A, D, E deja complets ; B/C places en attente (aucune
# reformulation encore generee a ce stade - voir rapport de generation).
# -----------------------------------------------------------------------------

for _, ligne in sources_b.iterrows():
    ajouter_paire(ligne["texte"], None, "B", "PLAGIAT", ligne["document"], ligne["document"],
                  "modele_langage (en attente de generation/validation)")
for _, ligne in sources_c.iterrows():
    ajouter_paire(ligne["texte"], None, "C", "PLAGIAT", ligne["document"], ligne["document"],
                  "modele_langage (en attente de generation/validation)")

df_jeu_test = pd.DataFrame(paires)
df_jeu_test.to_parquet(FICHIER_JEU_TEST, index=False)

# -----------------------------------------------------------------------------
# Rapport de generation
# -----------------------------------------------------------------------------

rapport = {
    "date_generation": DATE_GENERATION,
    "modele_generation_bc": MODELE_GENERATION,
    "prompt_categorie_b": PROMPT_CATEGORIE_B,
    "prompt_categorie_c": PROMPT_CATEGORIE_C,
    "seuil_tfidf_categorie_d": SEUIL_TFIDF_NON_APPARENTE,
    "note_noyau_calibrage": (
        "L'instruction mentionne 15 paires pour le noyau de calibrage existant, "
        "mais calibrer_alpha_fusion.py en contient reellement 10 (documente et deja "
        "valide) - reutilisees telles quelles, aucune paire manuelle supplementaire "
        "fabriquee pour ne pas contredire la consigne 'deja fait, ne pas refaire'."
    ),
    "comptes_par_categorie": df_jeu_test["categorie"].value_counts().to_dict(),
    "total_paires_construites_ici": len(df_jeu_test),
    "total_avec_noyau_calibrage": len(df_jeu_test) + 10,
    "paires_bc_en_attente_generation": int(
        df_jeu_test["texte_2"].isna().sum()
    ),
}
with open(FICHIER_RAPPORT_GENERATION, "w", encoding="utf-8") as f:
    json.dump(rapport, f, ensure_ascii=False, indent=2)

print(f"\n{'=' * 90}\nRAPPORT DE GENERATION\n{'=' * 90}")
for cle, valeur in rapport.items():
    print(f"  {cle} : {valeur}")

print(f"\nJeu de test (partiel, B/C en attente) sauvegarde dans {FICHIER_JEU_TEST}")
print(f"Rapport sauvegarde dans {FICHIER_RAPPORT_GENERATION}")

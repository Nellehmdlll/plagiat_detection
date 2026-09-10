# -*- coding: utf-8 -*-
"""
Application 1 - Gestion du corpus de reference.

Upload de documents de reference, construction/reconstruction du corpus
(segmentation + encodage semantique + index lexical), etat du corpus.
Reutilise sans les dupliquer les fonctions existantes de encoder_corpus_
complet.py et construire_tfidf_corpus.py - aucune logique metier reecrite
ici, uniquement l'interface et l'orchestration des etapes.

Important (limite assumee, documentee dans le README) : la reconstruction
est TOUJOURS complete (tous les documents du dossier de reference sont
retraites), pas incrementale. Les scripts sous-jacents n'ont pas ete concus
pour ajouter quelques documents a un corpus deja construit sans tout
retraiter ; une reconstruction complete reste rapide a l'echelle actuelle
du corpus (quelques minutes) et evite tout risque d'incoherence entre
vecteurs, metadonnees et index lexical.

A lancer avec l'environnement partage .venv-embed :
    .venv-embed\\Scripts\\python.exe -m streamlit run app_corpus.py
"""

import os
import json
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from encoder_corpus_complet import (
    CORPUS, NOM_MODELE, FENETRE_REELLE, FICHIER_VECTEURS, FICHIER_METADONNEES,
    FICHIER_CHECKPOINT, construire_chunks_corpus_complet, encoder_corpus,
)
from decouper_segments import compteur_taille_tokenizer
from construire_tfidf_corpus import (
    construire_index_tfidf, FICHIER_MATRICE, FICHIER_VOCABULAIRE, FICHIER_CORRESPONDANCE,
)

TAILLE_MAX_MO = 50
FICHIER_STATUTS = "statuts_corpus.json"

st.set_page_config(page_title="Gestion du corpus", page_icon="📚", layout="wide")

st.title("Gestion du corpus de référence")
st.write(
    "Cette application permet d'ajouter des mémoires ou thèses de référence et de préparer "
    "l'index utilisé pour la comparaison. Elle n'a besoin d'être relancée qu'à l'ajout de "
    "nouveaux documents de référence — pas à chaque vérification d'un document (voir "
    "l'application de vérification pour cela)."
)


def charger_statuts():
    if os.path.exists(FICHIER_STATUTS):
        with open(FICHIER_STATUTS, encoding="utf-8") as f:
            return json.load(f)
    return {}


def sauvegarder_statuts(statuts):
    with open(FICHIER_STATUTS, "w", encoding="utf-8") as f:
        json.dump(statuts, f, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# ETAT ACTUEL DU CORPUS
# -----------------------------------------------------------------------------

st.divider()
st.subheader("État actuel du corpus")

if os.path.exists(FICHIER_METADONNEES):
    df_meta_apercu = pd.read_parquet(FICHIER_METADONNEES, columns=["document"])
    col1, col2 = st.columns(2)
    col1.metric("Documents indexés", df_meta_apercu["document"].nunique())
    col2.metric("Segments (chunks) indexés", len(df_meta_apercu))
else:
    st.info("Aucun corpus n'a encore été construit. Déposez des documents ci-dessous puis lancez la construction.")

statuts = charger_statuts()
if statuts:
    ok = sorted(n for n, s in statuts.items() if s["statut"] == "ok")
    limitation = sorted(n for n, s in statuts.items() if s["statut"] == "limitation")
    echec = sorted(n for n, s in statuts.items() if s["statut"] == "echec")

    st.caption(
        f"✅ {len(ok)} traité(s) normalement · ⚠️ {len(limitation)} avec limitation · "
        f"❌ {len(echec)} en échec"
    )

    for nom in limitation:
        with st.expander(f"⚠️ {nom} — traité avec une limitation"):
            st.write(
                "La structure de ce document (sommaire ou table des matières) n'a pas pu être "
                "reconnue automatiquement. Ce document n'a donc pas pu être intégré au corpus "
                "de comparaison."
            )
            st.caption("Détail technique :")
            st.code(statuts[nom]["detail"])

    for nom in echec:
        with st.expander(f"❌ {nom} — échec du traitement"):
            st.write(
                "Ce document n'a pas pu être traité, probablement parce que le fichier PDF est "
                "corrompu, protégé par un mot de passe, ou dans un format inattendu."
            )
            st.caption("Détail technique :")
            st.code(statuts[nom]["detail"])

    if ok:
        with st.expander(f"✅ {len(ok)} document(s) traité(s) normalement"):
            st.dataframe(
                pd.DataFrame([{"Document": n, "Segments": statuts[n]["nb_chunks"]} for n in ok]),
                hide_index=True, use_container_width=True,
            )

# -----------------------------------------------------------------------------
# AJOUT DE DOCUMENTS
# -----------------------------------------------------------------------------

st.divider()
st.subheader("Ajouter des documents de référence")

dossier_corpus = Path(CORPUS)
dossier_corpus.mkdir(parents=True, exist_ok=True)
nb_fichiers_disque = len(list(dossier_corpus.glob("**/*.pdf")))
st.caption(f"Le dossier de référence contient actuellement {nb_fichiers_disque} fichier(s) PDF.")

fichiers = st.file_uploader(
    "Déposer un ou plusieurs PDF de mémoires/thèses de référence",
    type=["pdf"], accept_multiple_files=True,
)

if fichiers:
    ajoutes, deja_presents, refuses = [], [], []
    for f in fichiers:
        taille_mo = f.size / (1024 * 1024)
        if taille_mo > TAILLE_MAX_MO:
            refuses.append(f"{f.name} ({taille_mo:.1f} Mo)")
            continue
        chemin_dest = dossier_corpus / f.name
        if chemin_dest.exists():
            deja_presents.append(f.name)
            continue
        with open(chemin_dest, "wb") as out:
            out.write(f.getbuffer())
        ajoutes.append(f.name)

    if ajoutes:
        st.success(f"{len(ajoutes)} document(s) ajouté(s) : {', '.join(ajoutes)}")
        st.info("Cliquez sur « Reconstruire les index » ci-dessous pour les intégrer au corpus de comparaison.")
    if deja_presents:
        st.warning(
            f"{len(deja_presents)} fichier(s) déjà présents, non remplacés : "
            f"{', '.join(deja_presents)}. Renommez le fichier pour ajouter une nouvelle version."
        )
    if refuses:
        st.error(
            f"{len(refuses)} fichier(s) dépassent la limite de {TAILLE_MAX_MO} Mo et n'ont pas "
            f"été ajoutés : {', '.join(refuses)}."
        )

# -----------------------------------------------------------------------------
# (RE)CONSTRUCTION DU CORPUS
# -----------------------------------------------------------------------------

st.divider()
st.subheader("Construire / reconstruire l'index de comparaison")
st.caption(
    "Cette opération retraite l'ensemble des documents du dossier de référence (pas seulement "
    "les nouveaux) : extraction, détection du sommaire, segmentation, puis encodage sémantique "
    "et construction de l'index lexical. Prend plusieurs minutes selon le nombre de documents. "
    "Nécessaire après tout ajout de nouveaux documents."
)

if st.button("🔄 Reconstruire les index", type="primary"):
    barre = st.progress(0.0, text="Préparation...")
    try:
        for chemin in [FICHIER_METADONNEES, FICHIER_VECTEURS, FICHIER_CHECKPOINT,
                       FICHIER_MATRICE, FICHIER_VOCABULAIRE, FICHIER_CORRESPONDANCE, FICHIER_STATUTS]:
            if os.path.exists(chemin):
                os.remove(chemin)

        tokenizer = AutoTokenizer.from_pretrained(NOM_MODELE)
        compteur_taille = compteur_taille_tokenizer(tokenizer)

        def maj_segmentation(i, total, nom):
            barre.progress(i / total * 0.4, text=f"Étape 1/3 — Analyse structurelle : {nom} ({i}/{total})")

        chunks_par_document, tous_chunks, statuts_documents = construire_chunks_corpus_complet(
            compteur_taille, rappel_progression=maj_segmentation
        )
        sauvegarder_statuts(statuts_documents)

        if not tous_chunks:
            barre.empty()
            st.error(
                "Aucun document du dossier de référence n'a pu être segmenté. Vérifiez le détail "
                "par document ci-dessus (section « État actuel du corpus » après rafraîchissement)."
            )
            st.rerun()

        colonnes = ["chunk_id", "document", "cle_unique_section", "niveau", "legitime",
                    "chapitre_parent", "position_debut", "position_fin",
                    "numero_chunk_dans_section", "texte"]
        df_meta = pd.DataFrame(tous_chunks)[colonnes]
        df_meta.to_parquet(FICHIER_METADONNEES, index=False)

        barre.progress(0.4, text="Étape 2/3 — Chargement du modèle sémantique...")
        modele = SentenceTransformer(NOM_MODELE, device="cuda")
        modele.max_seq_length = FENETRE_REELLE

        def maj_encodage(i, total, nom):
            barre.progress(0.4 + i / total * 0.5, text=f"Étape 2/3 — Encodage sémantique : {nom} ({i}/{total})")

        encoder_corpus(chunks_par_document, tous_chunks, modele, rappel_progression=maj_encodage)

        barre.progress(0.9, text="Étape 3/3 — Construction de l'index lexical (TF-IDF)...")
        construire_index_tfidf(df_meta)

        barre.progress(1.0, text="Terminé.")
        st.success(
            f"Corpus reconstruit avec succès : {df_meta['document'].nunique()} documents, "
            f"{len(df_meta)} segments."
        )
        st.rerun()

    except Exception:
        barre.empty()
        st.error(
            "Une erreur inattendue est survenue pendant la construction du corpus. Le corpus "
            "précédent (s'il existait) a été effacé — relancez la reconstruction après avoir "
            "consulté le détail technique."
        )
        with st.expander("Détail technique"):
            st.code(traceback.format_exc())

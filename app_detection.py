# -*- coding: utf-8 -*-
"""
Application 2 - Test de detection de similarites sur un document.

Version minimale : upload d'un PDF, lancement du pipeline de detection deja
valide, affichage du rapport. Reutilise sans les dupliquer les fonctions
existantes de detecter_plagiat.py (charger_ressources, detecter_plagiat) -
aucune logique metier reecrite ici, uniquement l'interface.

A lancer avec l'environnement partage .venv-embed :
    .venv-embed\\Scripts\\python.exe -m streamlit run app_detection.py
"""

import os
import re
import tempfile
import traceback

import streamlit as st

from detecter_plagiat import charger_ressources, detecter_plagiat

TAILLE_MAX_MO = 50
# Le seuil par defaut de detecter_plagiat.py (0.75) est une valeur de depart
# documentee comme provisoire ; l'evaluation complete (evaluer_jeu_test_
# complet.py) a trouve un optimum autour de 0.60-0.65 - retenu ici par
# defaut, ajustable dans l'interface.
SEUIL_PAR_DEFAUT = 0.65
NB_PASSAGES_AFFICHES = 30

st.set_page_config(page_title="Verification de similarites", page_icon="🔍", layout="wide")

st.title("Vérification de similarités entre documents")
st.warning(
    "⚠️ Cet outil signale des **similarités** entre le document soumis et un corpus de "
    "référence de mémoires/thèses. Il ne prononce **aucun verdict de plagiat** : "
    "l'appréciation finale revient à l'enseignant."
)


@st.cache_resource(show_spinner="Chargement du corpus de référence et des modèles (une seule fois)...")
def obtenir_ressources():
    return charger_ressources()


def expliquer_section_legitime(texte):
    """
    Explication courte et non technique du type de section legitime,
    purement pour l'affichage (n'affecte pas le score ni la classification).
    """
    t = texte[:200].lower()
    if "dédicace" in t:
        return "dédicace"
    if "remerciement" in t:
        return "section de remerciements"
    if "sigle" in t or "abréviation" in t:
        return "liste de sigles ou d'abréviations"
    if "liste des figures" in t or "liste des tableaux" in t:
        return "liste de figures ou de tableaux"
    if re.search(r"table\s+des?\s+mati[eè]res", t) or "sommaire" in t:
        return "sommaire / table des matières"
    if "institut" in t or "présentation" in t:
        return "présentation d'institution"
    return "section généralement partagée entre mémoires (structure standard)"


with st.sidebar:
    st.header("Paramètres")
    seuil = st.slider(
        "Seuil de sensibilité", min_value=0.30, max_value=0.90, value=SEUIL_PAR_DEFAUT, step=0.05,
        help=(
            "Un seuil plus bas signale davantage de passages (au risque de plus de fausses "
            "alertes) ; un seuil plus haut n'affiche que les similarités les plus fortes."
        ),
    )

fichier = st.file_uploader("Déposer le document à vérifier (PDF)", type=["pdf"])

if fichier is not None:
    taille_mo = fichier.size / (1024 * 1024)
    if taille_mo > TAILLE_MAX_MO:
        st.error(
            f"Ce fichier fait {taille_mo:.1f} Mo, ce qui dépasse la limite fixée à "
            f"{TAILLE_MAX_MO} Mo. Merci de déposer un document plus léger."
        )
    elif st.button("Lancer la vérification", type="primary"):
        with tempfile.TemporaryDirectory() as dossier_temp:
            chemin_temp = os.path.join(dossier_temp, fichier.name)
            with open(chemin_temp, "wb") as f:
                f.write(fichier.getbuffer())

            try:
                ressources = obtenir_ressources()
                with st.spinner("Analyse du document en cours (segmentation, comparaison)..."):
                    df_meta, matrice_tfidf, vecteurs, vectoriseur, modele, compteur_taille = ressources
                    df_rapport, taux_global = detecter_plagiat(
                        chemin_temp, df_meta, matrice_tfidf, vecteurs, vectoriseur, modele,
                        compteur_taille, seuil=seuil,
                    )
            except ValueError as e:
                st.error(
                    f"Ce document n'a pas pu être analysé : sa structure (table des matières, "
                    f"sommaire) n'a pas été reconnue. Vérifiez qu'il s'agit bien d'un mémoire ou "
                    f"d'une thèse avec un sommaire clairement identifiable."
                )
                with st.expander("Détail technique"):
                    st.code(str(e))
            except Exception:
                st.error(
                    "Une erreur est survenue pendant l'analyse de ce fichier. Il est peut-être "
                    "corrompu, protégé par un mot de passe, ou dans un format inattendu."
                )
                with st.expander("Détail technique"):
                    st.code(traceback.format_exc())
            else:
                st.session_state["df_rapport"] = df_rapport
                st.session_state["taux_global"] = taux_global
                st.session_state["seuil_utilise"] = seuil
                st.session_state["nom_fichier"] = fichier.name

if "df_rapport" in st.session_state:
    df_rapport = st.session_state["df_rapport"]
    taux_global = st.session_state["taux_global"]
    seuil_utilise = st.session_state["seuil_utilise"]

    st.divider()
    st.subheader(f"Résultat — {st.session_state['nom_fichier']}")

    if taux_global is None:
        st.info(
            "Aucun passage de contenu académique (hors sections légitimes) n'a pu être évalué "
            "dans ce document."
        )
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Taux de similarité détecté", f"{taux_global:.0%}")
        col2.metric("Passages analysés", len(df_rapport))
        col3.metric("Seuil utilisé", f"{seuil_utilise:.2f}")

    st.markdown("#### Passages les plus suspects")
    df_tri = df_rapport.sort_values("score_fusionne", ascending=False)
    st.caption(
        f"{len(df_tri)} passages au total, {min(NB_PASSAGES_AFFICHES, len(df_tri))} affichés "
        f"ci-dessous par score décroissant."
    )

    for _, ligne in df_tri.head(NB_PASSAGES_AFFICHES).iterrows():
        badge = " · 🟡 section légitime" if (ligne["legitime_suspect"] or ligne["legitime_candidat"]) else ""
        titre = f"Score {ligne['score_fusionne']:.0%} — source : {ligne['document_candidat']}{badge}"
        with st.expander(titre):
            if ligne["legitime_suspect"] or ligne["legitime_candidat"]:
                texte_ref = ligne["texte_candidat"] if ligne["legitime_candidat"] else ligne["texte_suspect"]
                explication = expliquer_section_legitime(texte_ref)
                st.caption(f"ℹ️ Ce passage provient d'une **{explication}**, généralement partagée entre mémoires.")

            col_gauche, col_droite = st.columns(2)
            with col_gauche:
                st.markdown("**Document vérifié**")
                st.write(ligne["texte_suspect"])
            with col_droite:
                st.markdown(f"**Source : {ligne['document_candidat']}**")
                st.write(ligne["texte_candidat"])

            st.caption(
                f"Score sémantique : {ligne['score_semantique']:.2f} · "
                f"Score lexical : {ligne['score_lexical']:.2f}"
            )

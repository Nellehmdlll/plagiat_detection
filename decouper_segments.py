"""
Découpage du texte de chaque section en chunks prêts pour l'encodage sémantique.

localiser_titres_corps.py donne, pour chaque titre, sa position_debut et sa
position_fin dans le corps (idx_global). Ce module en extrait le texte réel,
le nettoie, puis le découpe en chunks de taille contrôlée en respectant en
priorité les frontières de paragraphe (à défaut, de phrase), avec un léger
chevauchement entre chunks consécutifs d'une même section.

Auteur : Sawadogo Rimalguedo Rahimata
Date : 02/09/2026
"""

import re

from inspection import detecter_lignes_repete, detecter_sommaire
from extraire_titre_sommaire import normaliser_texte, correspond_au_bandeau
from localiser_titres_corps import _aplatir

# -----------------------------------------------------------------------------
# PARAMÈTRES PAR DÉFAUT
# -----------------------------------------------------------------------------

# Valeur par défaut en nombre de MOTS, utilisée quand aucun `compteur_taille`
# n'est fourni à decouper_en_segments() (voir plus bas). Un simple compte de
# mots n'est qu'une approximation grossière de la fenêtre d'un modèle
# d'embedding : mesuré sur le corpus de référence, le ratio réel varie de
# 1,3 à 2 tokens par mot selon le vocabulaire du document (nettement plus
# haut sur un texte à vocabulaire technique/médical dense que sur un texte
# courant). Pour un découpage fiable vis-à-vis de la fenêtre RÉELLE d'un
# modèle donné, passer un `compteur_taille` basé sur son tokenizer (voir
# `compteur_taille_tokenizer` ci-dessous) plutôt que de se fier à cette
# valeur par défaut.
TAILLE_MAX_CHUNK_DEFAUT = 350

# Chevauchement par défaut entre deux chunks consécutifs d'une même section,
# dans la même unité que taille_max_chunk (mots par défaut, tokens si un
# compteur_taille basé sur un tokenizer est utilisé).
CHEVAUCHEMENT_DEFAUT = 40

# En dessous de ce nombre de mots, une section qui ne produit qu'UN seul
# chunk est considérée "titre seul" (un titre immédiatement suivi d'un
# sous-titre, sans texte de section propre) : elle est fusionnée en tête du
# premier chunk de la section suivante plutôt que publiée comme chunk à
# part entière quasi vide.
SEUIL_TITRE_SEUL_MOTS = 10


def compteur_taille_mots(texte):
    """Compteur par défaut : nombre de mots (`str.split()`)."""
    return len(texte.split())


def compteur_taille_tokenizer(tokenizer):
    """
    Construit un compteur de taille basé sur le tokenizer d'un modèle
    d'embedding donné (ex: `SentenceTransformer(...).tokenizer` pour
    sentence-camembert-large ou BGE-M3), à passer en `compteur_taille` à
    decouper_en_segments() pour que le découpage respecte la fenêtre RÉELLE
    du modèle cible plutôt qu'une approximation en nombre de mots.

    Changer de modèle d'embedding plus tard ne demande donc que de
    reconstruire ce compteur avec le nouveau tokenizer — la logique de
    découpage elle-même (decouper_en_segments et les fonctions internes de
    ce module) n'a pas besoin d'être modifiée.
    """
    def compter(texte):
        return len(tokenizer.encode(texte, add_special_tokens=False))
    return compter


# -----------------------------------------------------------------------------
# ÉTAPE B — NETTOYAGE
# -----------------------------------------------------------------------------

def _est_ligne_bruit(ligne_strip):
    """Même définition que calculer_taux_bruit() dans inspection.py."""
    return bool(
        len(ligne_strip) == 0
        or re.match(r'^[_\-\=\.]{3,}$', ligne_strip)
        or len(ligne_strip) <= 3
    )


def _est_numero_page_isole(ligne_strip):
    """Une ligne qui n'est RIEN d'autre qu'un numéro de page (arabe ou romain)."""
    return bool(
        re.match(r'^\d{1,4}$', ligne_strip)
        or re.match(r'^[ivxlcdm]{1,6}$', ligne_strip, re.IGNORECASE)
    )


def _est_bandeau(ligne_norm, bandeaux_normalises):
    return any(correspond_au_bandeau(ligne_norm, b) for b in bandeaux_normalises)


def _extraire_paragraphes_nettoyes(lignes, bandeaux_normalises):
    """
    Regroupe les lignes brutes d'une section en paragraphes (une ligne vide
    marque une frontière de paragraphe, tant que ce signal existe encore —
    d'où le regroupement AVANT le nettoyage, qui retire justement les lignes
    vides), puis nettoie chaque paragraphe : bandeaux répétés, bruit pur,
    numéros de page isolés. Retourne une liste de chaînes (un paragraphe =
    une chaîne, ses lignes internes jointes par un espace).
    """
    paragraphes_bruts = []
    courant = []
    for ligne in lignes:
        if ligne.strip() == '':
            if courant:
                paragraphes_bruts.append(courant)
                courant = []
        else:
            courant.append(ligne)
    if courant:
        paragraphes_bruts.append(courant)

    paragraphes_nettoyes = []
    for para_lignes in paragraphes_bruts:
        lignes_gardees = []
        for ligne in para_lignes:
            ls = ligne.strip()
            if _est_ligne_bruit(ls):
                continue
            if _est_numero_page_isole(ls):
                continue
            if _est_bandeau(normaliser_texte(ls), bandeaux_normalises):
                continue
            lignes_gardees.append(ls)
        if lignes_gardees:
            paragraphes_nettoyes.append(' '.join(lignes_gardees))

    return paragraphes_nettoyes


# -----------------------------------------------------------------------------
# ÉTAPE C — DÉCOUPAGE EN CHUNKS
# -----------------------------------------------------------------------------

# Frontière de phrase : après un ., ! ou ?, suivi d'espace(s), suivi d'une
# majuscule (limite volontairement simple — n'essaie pas de distinguer une
# vraie fin de phrase d'une abréviation comme "M." ; acceptable en dernier
# recours, uniquement quand un paragraphe dépasse à lui seul la taille max).
_MOTIF_FIN_PHRASE = re.compile(
    r'(?<=[.!?])\s+(?=[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ])'
)


def _decouper_en_phrases(texte):
    phrases = _MOTIF_FIN_PHRASE.split(texte)
    return [p.strip() for p in phrases if p.strip()]


def _decouper_paragraphe_en_sous_unites(paragraphe, taille_max_chunk, compteur_taille):
    """
    Si un paragraphe à lui seul dépasse taille_max_chunk (mesuré par
    compteur_taille), le redécoupe par phrases. Si une "phrase" individuelle
    dépasse elle-même la limite (texte sans ponctuation de fin de phrase
    régulière sur une longue portion — ex: une transcription d'entretien
    qualitatif, un tableau retranscrit en texte brut), dernier recours :
    coupe brute mot par mot pour CETTE phrase uniquement (la taille de
    chaque morceau est ajustée via compteur_taille, pas juste comptée en
    mots), le reste du paragraphe continuant d'être découpé par phrases.
    """
    if compteur_taille(paragraphe) <= taille_max_chunk:
        return [paragraphe]

    phrases = _decouper_en_phrases(paragraphe)
    if len(phrases) <= 1:
        phrases = [paragraphe]

    sous_unites = []
    courant = []
    taille_courante = 0
    for phrase in phrases:
        n = compteur_taille(phrase)
        if n > taille_max_chunk:
            if courant:
                sous_unites.append(' '.join(courant))
                courant, taille_courante = [], 0
            mots = phrase.split()
            debut = 0
            while debut < len(mots):
                fin = len(mots)
                while fin > debut + 1 and compteur_taille(' '.join(mots[debut:fin])) > taille_max_chunk:
                    fin -= 1
                sous_unites.append(' '.join(mots[debut:fin]))
                debut = fin
            continue
        if courant and taille_courante + n > taille_max_chunk:
            sous_unites.append(' '.join(courant))
            courant, taille_courante = [], 0
        courant.append(phrase)
        taille_courante += n
    if courant:
        sous_unites.append(' '.join(courant))
    return sous_unites


def _regrouper_paragraphes_en_chunks(paragraphes, taille_max_chunk, compteur_taille):
    """
    Assemble les paragraphes (ou leurs sous-unités si un paragraphe dépasse
    à lui seul la limite) en chunks, en coupant à la frontière de paragraphe
    (ou de phrase, en dernier recours) la plus proche de taille_max_chunk,
    telle que mesurée par compteur_taille.
    """
    unites = []
    for p in paragraphes:
        unites.extend(_decouper_paragraphe_en_sous_unites(p, taille_max_chunk, compteur_taille))

    # Taille plancher en dessous de laquelle un chunk en cours de
    # constitution n'est jamais isolé seul (ex: un titre de sous-section
    # suivi d'un paragraphe déjà proche du budget) : on absorbe l'unité
    # suivante quitte à dépasser légèrement taille_max_chunk, plutôt que de
    # publier un fragment quasi vide comme chunk à part entière. Comme
    # chaque unité est déjà bornée à taille_max_chunk par
    # _decouper_paragraphe_en_sous_unites, le dépassement induit reste
    # limité (au plus une unité de plus que le budget).
    taille_plancher = max(1, taille_max_chunk // 10)

    chunks = []
    courant = []
    taille_courante = 0
    for u in unites:
        n = compteur_taille(u)
        if courant and taille_courante + n > taille_max_chunk and taille_courante >= taille_plancher:
            chunks.append('\n\n'.join(courant))
            courant, taille_courante = [], 0
        courant.append(u)
        taille_courante += n
    if courant:
        chunks.append('\n\n'.join(courant))
    return chunks


def _appliquer_chevauchement(chunks, chevauchement, compteur_taille):
    """
    Préfixe chaque chunk (sauf le premier) avec un rappel du chunk
    précédent, pour ne pas perdre le contexte à la frontière de coupe. Le
    rappel démarre toujours à une frontière de phrase (jamais une phrase
    amputée de son début), quitte à dépasser légèrement chevauchement
    (mesuré par compteur_taille).
    """
    if len(chunks) <= 1 or chevauchement <= 0:
        return chunks

    resultat = [chunks[0]]
    for i in range(1, len(chunks)):
        rappel = _rappel_par_phrases(chunks[i - 1], chevauchement, compteur_taille)
        resultat.append(rappel + ' ' + chunks[i])
    return resultat


def _rappel_par_phrases(texte, chevauchement, compteur_taille):
    """
    Construit le rappel de chevauchement à partir des dernières phrases
    COMPLÈTES du texte précédent, en cumulant depuis la fin jusqu'à
    atteindre au moins chevauchement (selon compteur_taille) — on dépasse
    légèrement la cible plutôt que de couper une phrase en deux. Si aucune
    frontière de phrase n'est détectable (ex: liste de références sans
    ponctuation forte régulière), on retombe sur une fenêtre brute de mots.
    """
    phrases = _decouper_en_phrases(texte)
    if len(phrases) <= 1:
        mots = texte.split()
        if compteur_taille(texte) <= chevauchement:
            return texte
        fin = len(mots)
        while fin > 1 and compteur_taille(' '.join(mots[-fin:])) > chevauchement:
            fin -= 1
        return ' '.join(mots[-fin:])

    choisies = []
    taille = 0
    for phrase in reversed(phrases):
        choisies.insert(0, phrase)
        taille += compteur_taille(phrase)
        if taille >= chevauchement:
            break
    rappel = ' '.join(choisies)

    # Cas limite : certains contenus (tableaux de citations, définitions
    # denses) contiennent des phrases individuelles très longues. Si la
    # dernière phrase à elle seule fait déjà dépasser largement la cible
    # (plus du double), l'accumulation par phrases entières produirait un
    # chevauchement disproportionné. On retombe alors sur une fenêtre de
    # mots bornée à la cible, au prix d'un début de phrase éventuellement
    # incomplet pour ce cas précis seulement.
    if compteur_taille(rappel) > chevauchement * 2:
        mots = texte.split()
        fin = len(mots)
        while fin > 1 and compteur_taille(' '.join(mots[-fin:])) > chevauchement:
            fin -= 1
        return ' '.join(mots[-fin:])

    return rappel


# -----------------------------------------------------------------------------
# FONCTION PRINCIPALE
# -----------------------------------------------------------------------------

def decouper_en_segments(
    texte_complet,
    titres_localises,
    taille_max_chunk=TAILLE_MAX_CHUNK_DEFAUT,
    chevauchement=CHEVAUCHEMENT_DEFAUT,
    compteur_taille=None,
    liste_bandeaux=None,
    debug=False,
):
    """
    Découpe le corps du document en chunks, section par section.

    texte_complet : sortie de inspection.extraire_texte().
    titres_localises : sortie de localiser_titres_corps.localiser_titres_dans_corps()
        (liste de titres enrichis de position_debut/position_fin/localise).
    taille_max_chunk : taille cible maximale d'un chunk, dans l'unité de
        compteur_taille (mots par défaut — voir TAILLE_MAX_CHUNK_DEFAUT ;
        tokens si compteur_taille est basé sur un tokenizer).
    chevauchement : recouvrement entre deux chunks consécutifs d'une même
        section, dans la même unité que taille_max_chunk.
    compteur_taille : fonction texte -> taille utilisée pour toutes les
        décisions de découpage. Par défaut (None), compte les mots
        (compteur_taille_mots) — une approximation grossière. Pour un
        découpage fiable vis-à-vis de la fenêtre réelle d'un modèle
        d'embedding donné, passer compteur_taille_tokenizer(tokenizer) avec
        le tokenizer de ce modèle (voir ce module). Changer de modèle ne
        demande alors que de reconstruire ce compteur, sans toucher au
        reste de la logique de découpage.
    liste_bandeaux : liste de bandeaux déjà détectés (detecter_lignes_repete)
        à réutiliser ; si None, recalculée ici.

    Les titres non localisés sont ignorés : leur contenu reste absorbé dans
    la section précédente (déjà pris en compte par le calcul de position_fin
    de localiser_titres_dans_corps, qui saute par-dessus les titres non
    localisés pour pointer vers le prochain titre RÉELLEMENT localisé).

    Une section "titre seul" (un seul chunk produit, sous SEUIL_TITRE_SEUL_MOTS
    mots — typiquement un titre immédiatement suivi d'un sous-titre, sans
    texte de section propre) n'est jamais publiée seule : elle est fusionnée
    en tête du premier chunk de la prochaine section non vide (en cascade si
    plusieurs titres consécutifs sont dans ce cas), pour éviter un chunk
    quasi vide à encoder.

    Retourne une liste de dicts :
        {texte, cle_unique_section, niveau, legitime, chapitre_parent,
         position_debut, position_fin, numero_chunk_dans_section}
    """
    if compteur_taille is None:
        compteur_taille = compteur_taille_mots

    toutes_lignes = _aplatir(texte_complet)
    n_lignes = len(toutes_lignes)

    if liste_bandeaux is None:
        _, liste_bandeaux, _ = detecter_lignes_repete(texte_complet)
    bandeaux_normalises = [normaliser_texte(b) for b in liste_bandeaux]

    # La plage [position_debut, position_fin) d'une section peut, dans de
    # rares cas, chevaucher physiquement le bloc sommaire lui-même (ex: un
    # titre de résumé/abstract situé juste avant le sommaire, dont le
    # prochain titre localisé se trouve seulement après lui) : sans cette
    # exclusion, tout le sommaire (dédicace, table des matières...) se
    # retrouverait aspiré comme "texte de section". On exclut donc la même
    # zone que localiser_titres_dans_corps().
    resultat_bornes = detecter_sommaire(texte_complet, retourner_bornes=True)
    sommaire_present = resultat_bornes[0]
    zone_exclue = resultat_bornes[3] if sommaire_present else None

    titres_valides = [t for t in titres_localises if t.get('localise')]

    # -----------------------------------------------------------------
    # Étape 1 : extraire les paragraphes nettoyés de chaque section, SANS
    # encore les découper en chunks — une section "titre seul" doit pouvoir
    # voir ses paragraphes rejoints à ceux de la section suivante AVANT le
    # découpage en chunks, pour que taille_max_chunk reste toujours respecté
    # (le concaténer après coup, une fois les chunks déjà formés, pourrait
    # produire un chunk sans plus aucune limite de taille).
    # -----------------------------------------------------------------
    sections = []
    for titre in titres_valides:
        debut = titre['position_debut']['idx_global']
        fin = titre['position_fin']
        fin_bornee = min(fin, n_lignes) if fin is not None else n_lignes

        if zone_exclue is not None:
            lignes_section = [
                toutes_lignes[i][3] for i in range(debut, fin_bornee)
                if i < zone_exclue[0] or i > zone_exclue[1]
            ]
        else:
            lignes_section = [toutes_lignes[i][3] for i in range(debut, fin_bornee)]
        paragraphes = _extraire_paragraphes_nettoyes(lignes_section, bandeaux_normalises)

        if not paragraphes:
            if debug:
                print(f"[VIDE] section '{titre.get('cle_unique')}' "
                      f"(g{debut}-g{fin_bornee}) : rien après nettoyage, ignorée")
            continue

        sections.append({
            'titre': titre,
            'debut': debut,
            'fin': fin,
            'paragraphes': paragraphes,
        })

    # -----------------------------------------------------------------
    # Étape 2 : découper en chunks section par section, en fusionnant en
    # cascade les paragraphes des sections "titre seul" (un seul chunk
    # produit, sous SEUIL_TITRE_SEUL_MOTS mots — typiquement un titre
    # immédiatement suivi d'un sous-titre, sans texte propre) à ceux de la
    # prochaine section non vide AVANT découpage, plutôt que de publier un
    # chunk quasi vide à part entière.
    # -----------------------------------------------------------------
    chunks_resultat = []
    paragraphes_en_attente = []
    debut_fusion_en_attente = None

    for section in sections:
        paragraphes = section['paragraphes']

        # "Titre seul" ou non : décidé sur les paragraphes PROPRES de la
        # section (sans le préfixe en attente), pour que ce ne soit pas
        # l'accumulation de titres précédents qui la fasse artificiellement
        # sortir du lot.
        chunks_propres = _regrouper_paragraphes_en_chunks(paragraphes, taille_max_chunk, compteur_taille)
        est_titre_seul = (
            len(chunks_propres) == 1
            and len(chunks_propres[0].split()) < SEUIL_TITRE_SEUL_MOTS
        )

        if est_titre_seul:
            if debut_fusion_en_attente is None:
                debut_fusion_en_attente = section['debut']
            paragraphes_en_attente.extend(paragraphes)
            if debug:
                print(f"[FUSION] '{section['titre'].get('cle_unique')}' "
                      f"({len(chunks_propres[0].split())} mots) mis en attente")
            continue

        if paragraphes_en_attente:
            paragraphes_effectifs = paragraphes_en_attente + paragraphes
            debut_effectif = debut_fusion_en_attente
            paragraphes_en_attente = []
            debut_fusion_en_attente = None
            chunks_texte = _regrouper_paragraphes_en_chunks(paragraphes_effectifs, taille_max_chunk, compteur_taille)
        else:
            debut_effectif = section['debut']
            chunks_texte = chunks_propres

        chunks_texte = _appliquer_chevauchement(chunks_texte, chevauchement, compteur_taille)

        if debug:
            print(f"[OK] section '{section['titre'].get('cle_unique')}' : "
                  f"{len(paragraphes)} paragraphe(s) -> {len(chunks_texte)} chunk(s)")

        titre = section['titre']
        for idx, texte_chunk in enumerate(chunks_texte):
            chunks_resultat.append({
                'texte': texte_chunk,
                'cle_unique_section': titre.get('cle_unique'),
                'niveau': titre.get('niveau'),
                'legitime': titre.get('legitime'),
                'chapitre_parent': titre.get('chapitre_parent'),
                # Le premier chunk peut couvrir aussi le(s) titre(s) seul(s)
                # fusionnés en amont : sa position_debut reflète le début
                # réel du texte qu'il contient, pas seulement celui de sa
                # propre section.
                'position_debut': debut_effectif if idx == 0 else section['debut'],
                'position_fin': section['fin'],
                'numero_chunk_dans_section': idx,
            })

    # Cas limite : si le document se termine sur un ou plusieurs titres
    # seuls sans aucune section non vide après eux pour les absorber, on les
    # publie tels quels (découpés/bornés normalement) plutôt que de perdre
    # l'information.
    if paragraphes_en_attente:
        section_finale = sections[-1] if sections else None
        titre = section_finale['titre'] if section_finale else {}
        chunks_texte = _regrouper_paragraphes_en_chunks(paragraphes_en_attente, taille_max_chunk, compteur_taille)
        chunks_texte = _appliquer_chevauchement(chunks_texte, chevauchement, compteur_taille)
        for idx, texte_chunk in enumerate(chunks_texte):
            chunks_resultat.append({
                'texte': texte_chunk,
                'cle_unique_section': titre.get('cle_unique'),
                'niveau': titre.get('niveau'),
                'legitime': titre.get('legitime'),
                'chapitre_parent': titre.get('chapitre_parent'),
                'position_debut': debut_fusion_en_attente if idx == 0 else None,
                'position_fin': section_finale['fin'] if section_finale else None,
                'numero_chunk_dans_section': idx,
            })

    return chunks_resultat

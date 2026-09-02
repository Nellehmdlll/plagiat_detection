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

# Valeur INDICATIVE en nombre de MOTS (pas de tokens : aucun tokenizer n'est
# encore branché à ce stade du pipeline). À recaler une fois le modèle
# d'embedding choisi (ex. sentence-camembert-large) et son tokenizer connus —
# un mot français fait grossièrement 1,3 à 1,8 token selon le modèle, donc
# 350 mots correspond très approximativement à 500-600 tokens.
TAILLE_MAX_CHUNK_DEFAUT = 350

# Chevauchement par défaut entre deux chunks consécutifs d'une même section,
# en nombre de mots pris en fin du chunk précédent.
CHEVAUCHEMENT_MOTS_DEFAUT = 40

# En dessous de ce nombre de mots, une section qui ne produit qu'UN seul
# chunk est considérée "titre seul" (un titre immédiatement suivi d'un
# sous-titre, sans texte de section propre) : elle est fusionnée en tête du
# premier chunk de la section suivante plutôt que publiée comme chunk à
# part entière quasi vide.
SEUIL_TITRE_SEUL_MOTS = 10


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


def _decouper_paragraphe_en_sous_unites(paragraphe, taille_max_chunk):
    """
    Si un paragraphe à lui seul dépasse taille_max_chunk, le redécoupe par
    phrases. Si une "phrase" individuelle dépasse elle-même la limite (texte
    sans ponctuation de fin de phrase régulière sur une longue portion —
    ex: une transcription d'entretien qualitatif, un tableau retranscrit en
    texte brut), dernier recours : coupe brute par mots pour CETTE phrase
    uniquement, le reste du paragraphe continuant d'être découpé par phrases.
    """
    if len(paragraphe.split()) <= taille_max_chunk:
        return [paragraphe]

    phrases = _decouper_en_phrases(paragraphe)
    if len(phrases) <= 1:
        phrases = [paragraphe]

    sous_unites = []
    courant = []
    n_mots_courant = 0
    for phrase in phrases:
        n = len(phrase.split())
        if n > taille_max_chunk:
            if courant:
                sous_unites.append(' '.join(courant))
                courant, n_mots_courant = [], 0
            mots = phrase.split()
            sous_unites.extend(
                ' '.join(mots[i:i + taille_max_chunk])
                for i in range(0, len(mots), taille_max_chunk)
            )
            continue
        if courant and n_mots_courant + n > taille_max_chunk:
            sous_unites.append(' '.join(courant))
            courant, n_mots_courant = [], 0
        courant.append(phrase)
        n_mots_courant += n
    if courant:
        sous_unites.append(' '.join(courant))
    return sous_unites


def _regrouper_paragraphes_en_chunks(paragraphes, taille_max_chunk):
    """
    Assemble les paragraphes (ou leurs sous-unités si un paragraphe dépasse
    à lui seul la limite) en chunks, en coupant à la frontière de paragraphe
    (ou de phrase, en dernier recours) la plus proche de taille_max_chunk.
    """
    unites = []
    for p in paragraphes:
        unites.extend(_decouper_paragraphe_en_sous_unites(p, taille_max_chunk))

    chunks = []
    courant = []
    n_mots_courant = 0
    for u in unites:
        n = len(u.split())
        if courant and n_mots_courant + n > taille_max_chunk:
            chunks.append('\n\n'.join(courant))
            courant, n_mots_courant = [], 0
        courant.append(u)
        n_mots_courant += n
    if courant:
        chunks.append('\n\n'.join(courant))
    return chunks


def _appliquer_chevauchement(chunks, chevauchement_mots):
    """
    Préfixe chaque chunk (sauf le premier) avec un rappel du chunk
    précédent, pour ne pas perdre le contexte à la frontière de coupe. Le
    rappel démarre toujours à une frontière de phrase (jamais une phrase
    amputée de son début), quitte à dépasser légèrement chevauchement_mots.
    """
    if len(chunks) <= 1 or chevauchement_mots <= 0:
        return chunks

    resultat = [chunks[0]]
    for i in range(1, len(chunks)):
        rappel = _rappel_par_phrases(chunks[i - 1], chevauchement_mots)
        resultat.append(rappel + ' ' + chunks[i])
    return resultat


def _rappel_par_phrases(texte, chevauchement_mots):
    """
    Construit le rappel de chevauchement à partir des dernières phrases
    COMPLÈTES du texte précédent, en cumulant depuis la fin jusqu'à
    atteindre au moins chevauchement_mots mots (on dépasse légèrement la
    cible plutôt que de couper une phrase en deux). Si aucune frontière de
    phrase n'est détectable (ex: liste de références sans ponctuation
    forte régulière), on retombe sur une fenêtre brute de mots.
    """
    phrases = _decouper_en_phrases(texte)
    if len(phrases) <= 1:
        mots = texte.split()
        if len(mots) <= chevauchement_mots:
            return texte
        return ' '.join(mots[-chevauchement_mots:])

    choisies = []
    n_mots = 0
    for phrase in reversed(phrases):
        choisies.insert(0, phrase)
        n_mots += len(phrase.split())
        if n_mots >= chevauchement_mots:
            break
    return ' '.join(choisies)


# -----------------------------------------------------------------------------
# FONCTION PRINCIPALE
# -----------------------------------------------------------------------------

def decouper_en_segments(
    texte_complet,
    titres_localises,
    taille_max_chunk=TAILLE_MAX_CHUNK_DEFAUT,
    chevauchement_mots=CHEVAUCHEMENT_MOTS_DEFAUT,
    liste_bandeaux=None,
    debug=False,
):
    """
    Découpe le corps du document en chunks, section par section.

    texte_complet : sortie de inspection.extraire_texte().
    titres_localises : sortie de localiser_titres_corps.localiser_titres_dans_corps()
        (liste de titres enrichis de position_debut/position_fin/localise).
    taille_max_chunk : taille cible maximale d'un chunk, en nombre de MOTS
        (voir TAILLE_MAX_CHUNK_DEFAUT — valeur indicative, à recaler une fois
        le tokenizer du modèle d'embedding connu).
    chevauchement_mots : nombre de mots de recouvrement entre deux chunks
        consécutifs d'une même section.
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
    # Étape 1 : produire les chunks de chaque section, indépendamment.
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

        chunks_texte = _regrouper_paragraphes_en_chunks(paragraphes, taille_max_chunk)
        chunks_texte = _appliquer_chevauchement(chunks_texte, chevauchement_mots)

        sections.append({
            'titre': titre,
            'debut': debut,
            'fin': fin,
            'chunks_texte': chunks_texte,
        })

        if debug:
            print(f"[OK] section '{titre.get('cle_unique')}' (g{debut}-g{fin_bornee}) : "
                  f"{len(paragraphes)} paragraphe(s) -> {len(chunks_texte)} chunk(s)")

    # -----------------------------------------------------------------
    # Étape 2 : fusionner en cascade les sections "titre seul" en tête de
    # la prochaine section non vide.
    # -----------------------------------------------------------------
    chunks_resultat = []
    textes_titres_seuls_en_attente = []
    debut_fusion_en_attente = None

    for section in sections:
        chunks_texte = list(section['chunks_texte'])
        est_titre_seul = (
            len(chunks_texte) == 1
            and len(chunks_texte[0].split()) < SEUIL_TITRE_SEUL_MOTS
        )

        if est_titre_seul:
            if debut_fusion_en_attente is None:
                debut_fusion_en_attente = section['debut']
            textes_titres_seuls_en_attente.append(chunks_texte[0])
            if debug:
                print(f"[FUSION] '{section['titre'].get('cle_unique')}' "
                      f"({len(chunks_texte[0].split())} mots) mis en attente")
            continue

        if textes_titres_seuls_en_attente:
            prefixe = '\n\n'.join(textes_titres_seuls_en_attente)
            chunks_texte[0] = prefixe + '\n\n' + chunks_texte[0]
            debut_effectif = debut_fusion_en_attente
            textes_titres_seuls_en_attente = []
            debut_fusion_en_attente = None
        else:
            debut_effectif = section['debut']

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
    # publie tels quels plutôt que de perdre l'information.
    if textes_titres_seuls_en_attente:
        section_finale = sections[-1] if sections else None
        titre = section_finale['titre'] if section_finale else {}
        chunks_resultat.append({
            'texte': '\n\n'.join(textes_titres_seuls_en_attente),
            'cle_unique_section': titre.get('cle_unique'),
            'niveau': titre.get('niveau'),
            'legitime': titre.get('legitime'),
            'chapitre_parent': titre.get('chapitre_parent'),
            'position_debut': debut_fusion_en_attente,
            'position_fin': section_finale['fin'] if section_finale else None,
            'numero_chunk_dans_section': 0,
        })

    return chunks_resultat

"""
Localisation des titres du sommaire dans le corps du document.
Le sommaire propose la liste et l'ordre attendu des titres ; le corps donne
leur position réelle. Recherche séquentielle : le titre N est cherché à
partir de la position où le titre N-1 a été trouvé.

Auteur : Sawadogo Rimalguedo Rahimata
Date : 02/09/2026
"""

import difflib
import re

from inspection import detecter_sommaire
from extraire_titre_sommaire import (
    MOTIF_CHAPITRE_ROMAIN,
    MOTIF_CHAPITRE_ARABE,
    MOTIF_NUMERO_STRUCTURE,
    normaliser_texte,
)

# Au-delà de cette distance (en lignes) depuis la position de recherche
# courante, le repli approximatif (tier 'approx') n'est plus tenté : un
# accord de mots à cette distance a trop de chances d'être une coïncidence
# plutôt qu'une vraie correspondance, et accepter un tel match ferait
# dérailler la recherche séquentielle des titres suivants.
FENETRE_APPROX = 800


# -----------------------------------------------------------------------------
# EXTRACTION D'UN NUMÉRO EN TÊTE DE LIGNE (même logique que dans le sommaire)
# -----------------------------------------------------------------------------

def _extraire_numero_ligne(ligne):
    """
    Retourne (numero_brut, texte_restant) si la ligne commence par un motif
    de titre (CHAPITRE X, romain simple, décimal...), sinon (None, ligne).
    Mêmes motifs que ceux utilisés pour parser le sommaire : un numéro au
    milieu d'une phrase n'est pas un titre, seul un numéro en tête de ligne
    l'est.
    """
    ligne = ligne.strip()

    m = MOTIF_CHAPITRE_ROMAIN.search(ligne)
    if m:
        reste = ligne[m.end():].strip().lstrip(':').strip()
        return m.group(1), reste

    m = MOTIF_CHAPITRE_ARABE.search(ligne)
    if m:
        reste = ligne[m.end():].strip().lstrip(':').strip()
        return m.group(1), reste

    m = MOTIF_NUMERO_STRUCTURE.search(ligne)
    if m:
        reste = ligne[m.end():].strip()
        return m.group(0).strip(), reste

    return None, ligne


def _normaliser_numero(numero):
    """
    Normalise un numéro de titre pour comparaison : tiret et point traités
    comme équivalents (I-1-1 ~ I.1.1), casse et espaces uniformisés.
    """
    if not numero:
        return None
    n = numero.strip().rstrip('.:').replace('-', '.')
    n = re.sub(r'\s*\.\s*', '.', n)
    n = re.sub(r'\s+', ' ', n).strip()
    return n.lower()


def _mots_pour_comparaison(texte):
    """
    Normalise puis découpe en mots, en retirant la ponctuation résiduelle
    (typiquement un tiret) qui reste collée au premier mot quand un titre
    est écrit "CHAPITRE X -Titre" (le "-" n'est pas retiré par l'extraction
    du numéro, seulement les deux-points le sont) : sans ce nettoyage,
    "-cadre" et "cadre" ne sont jamais reconnus comme le même mot.
    """
    norm = normaliser_texte(texte or '')
    norm = re.sub(r'^[^a-z0-9]+', '', norm)
    mots = norm.split()
    # Un numéro de page reste parfois collé au texte du titre (ex: "Approche
    # méthodologique 15") quand le nettoyage des pointillés échoue faute
    # d'un séparateur assez long avant le chiffre (un seul espace au lieu
    # de 4+ points) : ce résidu fausserait toute comparaison de texte.
    if mots and re.match(r'^\d{1,4}$', mots[-1]):
        mots = mots[:-1]
    return mots


# -----------------------------------------------------------------------------
# APLATISSEMENT DU DOCUMENT
# -----------------------------------------------------------------------------

def _aplatir(texte_complet):
    toutes_lignes = []
    compteur = 0
    for num_page, lignes in enumerate(texte_complet):
        for idx_ligne, ligne in enumerate(lignes):
            toutes_lignes.append(
                (num_page, idx_ligne, compteur, ligne.strip().strip('\xa0').strip())
            )
            compteur += 1
    return toutes_lignes


def _hors_zone_sommaire(i, zone_exclue):
    debut, fin = zone_exclue
    return i < debut or i > fin


# -----------------------------------------------------------------------------
# RECHERCHE D'UN TITRE SANS NUMÉRO (marqueurs autonomes : Introduction, etc.)
# -----------------------------------------------------------------------------

def _chercher_marqueur_sans_numero(texte_norm, toutes_lignes, start_idx, n_lignes, zone_exclue):
    if not texte_norm:
        return None, None

    # Un repère de page résiduel ("resume . vi", "anexes . xi") reste parfois
    # collé au texte quand le nettoyage des pointillés en amont échoue faute
    # d'un séparateur assez long (cf. _mots_pour_comparaison) : on le retire
    # ici aussi, sur la forme texte complète cette fois.
    texte_norm = re.sub(r'\s*\.?\s+[ivxlcdm]{1,5}$', '', texte_norm)
    if not texte_norm:
        return None, None

    # Passe stricte : la ligne EST le marqueur (courte, sans numéro en tête).
    for i in range(start_idx, n_lignes):
        if not _hors_zone_sommaire(i, zone_exclue):
            continue
        ligne = toutes_lignes[i][3]
        if not ligne or len(ligne) > 80:
            continue
        num, _ = _extraire_numero_ligne(ligne)
        if num is not None:
            continue
        if normaliser_texte(ligne) == texte_norm:
            return i, 'texte_exact'

    # Passe tolérante : la ligne commence par (ou contient) le texte cible,
    # fenêtre limitée pour éviter un faux positif lointain. Un marqueur sans
    # numéro n'a aucun signal fort pour s'ancrer (contrairement à un titre
    # numéroté) : sous une quinzaine de caractères, un fragment générique
    # (souvent un reste de ligne coupée dans le sommaire, pas un vrai titre)
    # risquerait de matcher n'importe où dans le document et de faire
    # dérailler toute la recherche séquentielle des titres suivants.
    if len(texte_norm) < 15:
        return None, None
    limite = min(start_idx + FENETRE_APPROX, n_lignes)
    for i in range(start_idx, limite):
        if not _hors_zone_sommaire(i, zone_exclue):
            continue
        ligne = toutes_lignes[i][3]
        if not ligne or len(ligne) > 120:
            continue
        num, _ = _extraire_numero_ligne(ligne)
        if num is not None:
            continue
        ln = normaliser_texte(ligne)
        if ln.startswith(texte_norm) or texte_norm.startswith(ln):
            return i, 'texte_approx'

    return None, None


# -----------------------------------------------------------------------------
# RECHERCHE D'UN TITRE NUMÉROTÉ
# -----------------------------------------------------------------------------

def _chercher_titre_numerote(numeros_acceptables, mots_titre, toutes_lignes, start_idx, n_lignes, zone_exclue):
    # Une seule passe : on collecte tous les candidats dont le numéro en
    # tête de ligne correspond exactement (normalisé) à l'une des formes
    # acceptables du numéro cherché.
    candidats = []
    for i in range(start_idx, n_lignes):
        if not _hors_zone_sommaire(i, zone_exclue):
            continue
        ligne = toutes_lignes[i][3]
        if not ligne:
            continue
        num, reste = _extraire_numero_ligne(ligne)
        if num is None:
            continue
        if _normaliser_numero(num) not in numeros_acceptables:
            continue

        mots_ligne = _mots_pour_comparaison(reste)
        # Un titre long est parfois coupé sur 2-3 lignes physiques dans le
        # corps (ex: "CHAPITRE II -CADRE THEORIQUE...LA" / "METHODOLOGIE DE
        # LA RECHERCHE..."). Si le reste de la ligne est court, on complète
        # avec les lignes suivantes tant qu'elles ne sont pas elles-mêmes un
        # nouveau titre numéroté.
        j = i + 1
        lignes_ajoutees = 0
        while len(mots_ligne) < 10 and lignes_ajoutees < 2 and j < n_lignes:
            ligne_suivante = toutes_lignes[j][3]
            if not ligne_suivante:
                break
            num_suivant, _ = _extraire_numero_ligne(ligne_suivante)
            if num_suivant is not None:
                break
            mots_ligne = mots_ligne + _mots_pour_comparaison(ligne_suivante)
            j += 1
            lignes_ajoutees += 1

        candidats.append((i, mots_ligne))

    if not candidats:
        return None, None

    if not mots_titre:
        return candidats[0][0], 'numero_seul'

    # Tier strict : les 5 premiers mots (ou moins si le titre est plus court)
    # correspondent exactement, dans l'ordre.
    n = min(5, len(mots_titre))
    for i, mots_ligne in candidats:
        if mots_titre[:n] == mots_ligne[:n]:
            return i, 'strict'

    # Tier relâché : seulement les 2 premiers mots.
    n2 = min(2, len(mots_titre))
    for i, mots_ligne in candidats:
        if mots_titre[:n2] == mots_ligne[:n2]:
            return i, 'relax'

    # Tier sans-espaces : certains documents subissent un artefact
    # d'extraction qui colle les mots du sommaire entre eux ("Conceptset
    # définitions" au lieu de "Concepts et définitions"), rendant toute
    # comparaison mot à mot impossible. On compare alors les deux côtés en
    # ignorant complètement les espaces.
    titre_compact = ''.join(mots_titre)
    if len(titre_compact) >= 6:
        for i, mots_ligne in candidats:
            ligne_compact = ''.join(mots_ligne)
            if not ligne_compact:
                continue
            if titre_compact.startswith(ligne_compact) or ligne_compact.startswith(titre_compact):
                return i, 'sans_espaces'

    # Tier approximatif : chevauchement de mots >= 50%, fenêtre limitée.
    mots_titre_set = set(mots_titre)
    for i, mots_ligne in candidats:
        if i > start_idx + FENETRE_APPROX:
            break
        intersection = set(mots_ligne[:len(mots_titre) + 5]) & mots_titre_set
        if len(intersection) / len(mots_titre_set) >= 0.5:
            return i, 'approx'

    # Tier flou (dernier repli) : similarité de caractères sur les formes
    # compactes, pour tolérer de petites coquilles ou artefacts d'extraction
    # (ex: une correction de lettres doublées trop agressive sur une page
    # particulière : "classification" -> "clasification").
    if len(titre_compact) >= 8:
        for i, mots_ligne in candidats:
            if i > start_idx + FENETRE_APPROX:
                break
            ligne_compact = ''.join(mots_ligne)
            if not ligne_compact:
                continue
            longueur = min(len(titre_compact), len(ligne_compact))
            ratio = difflib.SequenceMatcher(
                None, titre_compact[:longueur], ligne_compact[:longueur]
            ).ratio()
            if ratio >= 0.85:
                return i, 'fuzzy'

    return None, None


def _chercher_titre(titre, toutes_lignes, start_idx, n_lignes, zone_exclue):
    numero_norm = _normaliser_numero(titre.get('numero'))

    if numero_norm is None:
        texte_norm = normaliser_texte(titre.get('texte') or '')
        return _chercher_marqueur_sans_numero(
            texte_norm, toutes_lignes, start_idx, n_lignes, zone_exclue
        )

    # Un titre de niveau 1 numéroté "1", "2"... dans un sommaire sans mot-clé
    # CHAPITRE peut très bien correspondre, dans le corps, à une vraie
    # ligne "Chapitre 1", "Chapitre 2"... (le sommaire et le corps du même
    # document n'utilisent pas toujours la même convention). On accepte les
    # deux formes pour ce cas précis.
    numeros_acceptables = {numero_norm}
    if titre.get('niveau') == 1 and not numero_norm.startswith('chapitre'):
        numeros_acceptables.add(f'chapitre {numero_norm}')

    mots_titre = _mots_pour_comparaison(titre.get('texte') or '')
    return _chercher_titre_numerote(
        numeros_acceptables, mots_titre, toutes_lignes, start_idx, n_lignes, zone_exclue
    )


# -----------------------------------------------------------------------------
# FONCTION PRINCIPALE
# -----------------------------------------------------------------------------

def localiser_titres_dans_corps(texte_complet, titres, debug=False):
    """
    Pour chaque titre de `titres` (dans l'ordre du sommaire), localise la
    ligne du CORPS du document où il apparaît réellement (pas dans le
    sommaire lui-même). Recherche séquentielle : le titre N est cherché à
    partir de la position où le titre N-1 a été trouvé, ce qui élimine le
    risque de confondre deux occurrences du même numéro sous des chapitres
    différents.

    La recherche parcourt le document dans l'ordre naturel des pages en
    ignorant simplement la plage occupée par le bloc sommaire lui-même —
    et non "tout ce qui précède le sommaire" : sur certains documents (ex.
    SIMPORE, SERE), la table des matières détaillée est placée en fin de
    document plutôt qu'en tête, et le vrai corps la précède entièrement.

    Enrichit chaque dict de `titres` avec :
      - 'localise' (bool)
      - 'methode_localisation' (str ou None) : 'strict' / 'relax' / 'approx' /
        'numero_seul' / 'texte_exact' / 'texte_approx' / None
      - 'position_debut' : {'page', 'idx_ligne', 'idx_global'} ou None
      - 'position_fin' : idx_global du titre suivant localisé, ou fin de
        document si dernier titre localisé (ou tous les suivants non
        localisés) ; None si le titre lui-même n'est pas localisé.

    Retourne la même liste `titres` (modifiée en place) pour chaînage facile.
    """
    resultat_bornes = detecter_sommaire(texte_complet, retourner_bornes=True)
    sommaire_present = resultat_bornes[0]
    zone_exclue = resultat_bornes[3] if sommaire_present else (-1, -1)

    toutes_lignes = _aplatir(texte_complet)
    n_lignes = len(toutes_lignes)

    position_courante = 0

    for titre in titres:
        idx_trouve, methode = _chercher_titre(
            titre, toutes_lignes, position_courante, n_lignes, zone_exclue
        )

        if idx_trouve is not None:
            page, idx_ligne, idx_global, _ = toutes_lignes[idx_trouve]
            titre['localise'] = True
            titre['methode_localisation'] = methode
            titre['position_debut'] = {
                'page': page, 'idx_ligne': idx_ligne, 'idx_global': idx_global
            }
            position_courante = idx_global + 1
            if debug:
                print(f"[OK]  {methode:14s} | numero={titre.get('numero')!r} "
                      f"texte={titre.get('texte', '')[:40]!r} -> page {page}, global {idx_global}")
        else:
            titre['localise'] = False
            titre['methode_localisation'] = None
            titre['position_debut'] = None
            if debug:
                print(f"[??]  NON LOCALISE  | numero={titre.get('numero')!r} "
                      f"texte={titre.get('texte', '')[:40]!r} (recherche depuis global {position_courante})")

    # position_fin = idx_global du prochain titre LOCALISÉ, ou fin de document
    # pour le dernier titre localisé (ou si tous les suivants sont manqués).
    idx_fin_doc = (toutes_lignes[-1][2] + 1) if toutes_lignes else 0
    prochaine_position_connue = idx_fin_doc
    for titre in reversed(titres):
        if titre['localise']:
            titre['position_fin'] = prochaine_position_connue
            prochaine_position_connue = titre['position_debut']['idx_global']
        else:
            titre['position_fin'] = None

    return titres


"""
Script d'inspection du corpus de mémoires/thèses burkinabè
Objectif : produire une grille d'observation automatique pour chaque PDF,
mesurer les caractéristiques "Type 1" (objectives), et signaler les documents atypiques.

Auteur : Sawadogo Rimalguedo Rahimata
Date : 17/08/2026
Extracteur principal : pdfplumber
"""

import os
import re
import pandas as pd
import pdfplumber
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

DOSSIER_CORPUS = r"C:\Users\1\Documents\MASTER_SOUTENANCE\corpus"
FICHIER_SORTIE = "grille_observation_corpus.xlsx"

# Motifs regex pour la détection des éléments structurels
MOTIF_SOMMAIRE = re.compile(
    r'[\.\-_\u2026\u2015\s]{4,}\s*[\divxlcdm]+\s*$',
    re.IGNORECASE
)
MOTIF_CHAPITRE = re.compile(r'\bCHAPITRE\s+[IVXLCDM]+\b', re.IGNORECASE)
MOTIF_NUMERO_DECIMAL = re.compile(r'^\s*\d+\.\d+(?:\s|\.|\))', re.IGNORECASE)
MOTIF_NUMERO_ROMAIN = re.compile(r'^\s*[IVXLCDM]+\.\d+(?:\s|\.|\))', re.IGNORECASE)
MOTIF_NUMERO_SIMPLE = re.compile(r'^\s*\d+(?:\s|\.|\))', re.IGNORECASE)
MOTIF_NUMERO_ROMAIN_SIMPLE = re.compile(r'^\s*[IVXLCDM]+(?:\s|\.|\))', re.IGNORECASE)

# Seuils de signalement
SEUIL_RATIO_BAS = 100
SEUIL_BRUIT_HAUT = 0.5

# Motifs pour la détection des chapitres sans points de suite
# Le "(?:\d+\s+)?" optionnel tolère un numéro de page mal recollé en tête de
# ligne (artefact d'extraction fréquent avec les longs pointillés).
MOTIF_CHAPITRE_ROMAIN = re.compile(
    r'^\s*(?:\d+\s+)?(CHAPITRE\s+[IVXLCDM]+)\b',
    re.IGNORECASE
)
MOTIF_CHAPITRE_ARABE = re.compile(
    r'^\s*(?:\d+\s+)?(CHAPITRE\s+\d+)\b',
    re.IGNORECASE
)


# -----------------------------------------------------------------------------
# FONCTIONS AUXILIAIRES
# -----------------------------------------------------------------------------

def extraire_metadonnees(chemin_pdf):
    """
    Extrait les métadonnées de base du PDF : producteur, nombre de pages.
    Retourne un tuple (producteur, nb_pages).
    """
    with pdfplumber.open(chemin_pdf) as pdf:
        producteur = pdf.metadata.get('Producer', 'inconnu') if pdf.metadata else 'inconnu'
        nb_pages = len(pdf.pages)
    return producteur, nb_pages


def corriger_texte_double(texte):
    """
    Corrige un texte qui semble avoir chaque caractère doublé (ex: CCHHAAPPIITTRREE).
    """
    if not texte:
        return texte
    mots = texte.split()
    if not mots:
        return texte
    echantillon = mots[:100]
    nb_doubles = sum(1 for m in echantillon if re.search(r'(.)\1', m))
    ratio = nb_doubles / len(echantillon)
    if ratio > 0.5:
        return re.sub(r'(.)\1+', r'\1', texte)
    return texte


def extraire_texte(chemin_pdf):
    """
    Extrait le texte page par page avec pdfplumber.
    Retourne une liste de listes de lignes : texte[page][ligne].
    """
    texte_complet = []
    with pdfplumber.open(chemin_pdf) as pdf:
        for page in pdf.pages:
            texte_page = page.extract_text()
            if texte_page:
                lignes = texte_page.split('\n')
                lignes = [corriger_texte_double(l) for l in lignes]
            else:
                lignes = []
            texte_complet.append(lignes)
    return texte_complet


# -----------------------------------------------------------------------------
# DÉTECTION DU SOMMAIRE
# -----------------------------------------------------------------------------

def detecter_blocs_par_points_de_suite(texte_complet, debug=False):
    """
    Détecte les blocs denses de lignes avec points de suite + numéro de page.
    Retourne une liste de blocs, chaque bloc étant une liste de tuples
    (num_page, idx_ligne, global_idx, ligne).
    """
    lignes_sommaire = []
    compteur_global = 0
    for num_page, lignes in enumerate(texte_complet):
        for idx_ligne, ligne in enumerate(lignes):
            ligne_nettoyee = ligne.strip().strip('\xa0').strip()
            if MOTIF_SOMMAIRE.search(ligne_nettoyee):
                lignes_sommaire.append((num_page, idx_ligne, compteur_global, ligne_nettoyee))
            compteur_global += 1

    SAUT_MAX_LIGNES = 20
    blocs = []
    bloc_courant = []
    for num_page, idx_ligne, global_idx, ligne in lignes_sommaire:
        if bloc_courant:
            _, _, dernier_global, _ = bloc_courant[-1]
            if (global_idx - dernier_global) > SAUT_MAX_LIGNES:
                blocs.append(bloc_courant)
                bloc_courant = []
        bloc_courant.append((num_page, idx_ligne, global_idx, ligne))
    if bloc_courant:
        blocs.append(bloc_courant)

    if debug:
        print(f"[DEBUG] Blocs par points de suite : {len(blocs)}")
        for i, bloc in enumerate(blocs):
            pages_bloc = sorted(set(p for p, _, _, _ in bloc))
            print(f"   Bloc {i} : {len(bloc)} entrées, pages {pages_bloc}")

    return blocs


def detecter_blocs_par_chapitres(texte_complet, debug=False):
    """
    Détecte les zones où plusieurs lignes commencent par CHAPITRE (romain ou arabe),
    sans exiger de points de suite.
    """
    lignes_chapitre = []
    compteur_global = 0
    for num_page, lignes in enumerate(texte_complet):
        for idx_ligne, ligne in enumerate(lignes):
            ligne_nettoyee = ligne.strip().strip('\xa0').strip()
            if MOTIF_CHAPITRE_ROMAIN.search(ligne_nettoyee) or MOTIF_CHAPITRE_ARABE.search(ligne_nettoyee):
                lignes_chapitre.append((num_page, idx_ligne, compteur_global, ligne_nettoyee))
            compteur_global += 1

    SAUT_MAX_LIGNES = 80
    blocs = []
    bloc_courant = []
    for num_page, idx_ligne, global_idx, ligne in lignes_chapitre:
        if bloc_courant:
            _, _, dernier_global, _ = bloc_courant[-1]
            if (global_idx - dernier_global) > SAUT_MAX_LIGNES:
                blocs.append(bloc_courant)
                bloc_courant = []
        bloc_courant.append((num_page, idx_ligne, global_idx, ligne))
    if bloc_courant:
        blocs.append(bloc_courant)

    if debug:
        print(f"[DEBUG] Blocs par chapitres : {len(blocs)}")
        for i, bloc in enumerate(blocs):
            pages_bloc = sorted(set(p for p, _, _, _ in bloc))
            print(f"   Bloc {i} : {len(bloc)} lignes CHAPITRE, pages {pages_bloc}")

    return blocs


def detecter_blocs_par_structure_sans_points(texte_complet, debug=False):
    """
    Détecte les blocs de lignes qui ressemblent à des titres numérotés
    (romain simple, décimal), SANS exiger de points de suite.
    """
    motif_structure = re.compile(
        r'^\s*([IVXLCDM]+|\d+)([.\-]\d+)*[.:]?\s+[^\d\s]{2,}',
        re.IGNORECASE
    )
    lignes_struct = []
    compteur = 0
    for num_page, lignes in enumerate(texte_complet):
        for idx_ligne, ligne in enumerate(lignes):
            ligne_nettoyee = ligne.strip().strip('\xa0').strip()
            if motif_structure.search(ligne_nettoyee) and len(ligne_nettoyee) < 120:
                lignes_struct.append((num_page, idx_ligne, compteur, ligne_nettoyee))
            compteur += 1

    SAUT_MAX_LIGNES = 30
    blocs = []
    bloc_courant = []
    for num_page, idx_ligne, global_idx, ligne in lignes_struct:
        if bloc_courant:
            _, _, dernier_global, _ = bloc_courant[-1]
            if (global_idx - dernier_global) > SAUT_MAX_LIGNES:
                blocs.append(bloc_courant)
                bloc_courant = []
        bloc_courant.append((num_page, idx_ligne, global_idx, ligne))
    if bloc_courant:
        blocs.append(bloc_courant)

    blocs_filtres = [b for b in blocs if 3 <= len(b) <= 200]

    if debug:
        print(f"[DEBUG] Blocs par structure sans points : {len(blocs_filtres)}")
        for i, bloc in enumerate(blocs_filtres):
            pages = sorted(set(p for p, _, _, _ in bloc))
            print(f"   Bloc {i} : {len(bloc)} lignes, pages {pages}")

    return blocs_filtres


def detecter_sommaire(texte_complet, debug=False, retourner_bornes=False):
    # -----------------------------------------------------------------
    # 1. Détection des blocs par les trois stratégies
    # -----------------------------------------------------------------
    blocs_points = detecter_blocs_par_points_de_suite(texte_complet, debug=debug)
    blocs_chapitres = detecter_blocs_par_chapitres(texte_complet, debug=debug)
    blocs_structure = detecter_blocs_par_structure_sans_points(texte_complet, debug=debug)

    tous_les_blocs = []
    for bloc in blocs_points:
        if len(bloc) >= 5:
            tous_les_blocs.append(('points', bloc))
    for bloc in blocs_chapitres:
        if len(bloc) >= 3:
            tous_les_blocs.append(('chapitres', bloc))
    for bloc in blocs_structure:
        if len(bloc) >= 3:
            tous_les_blocs.append(('structure', bloc))

    if not tous_les_blocs:
        if retourner_bornes:
            return False, 0, [], (0, 0)
        return False, 0, []

    # -----------------------------------------------------------------
    # 2. Score de chaque bloc
    # -----------------------------------------------------------------
    def score_bloc(bloc):
        score = len(bloc)
        contient_chapitre = False
        for _, _, _, ligne in bloc:
            ligne_up = ligne.upper()
            if re.search(r'\b(CHAPITRE|INTRODUCTION|CONCLUSION|SOMMAIRE|TABLE\s+DES\s+MATI[EÈ]RES)\b', ligne_up):
                score += 30
                contient_chapitre = True
            if re.search(r'\b(LISTE\s+DES\s+TABLEAUX|LISTE\s+DES\s+FIGURES|LISTE\s+DES\s+IMAGES|TABLEAU\s+[IVXLC\d]|FIGURE\s+\d|IMAGE\s+\d|ANNEXE\s+\d)\b', ligne_up):
                score -= 10
        if contient_chapitre:
            score += 50
        return score

    # -----------------------------------------------------------------
    # 3. Choisir le meilleur bloc
    # -----------------------------------------------------------------
    meilleur_points = None
    meilleur_score_points = -9999
    for type_bloc, bloc in tous_les_blocs:
        if type_bloc == 'points':
            sc = score_bloc(bloc)
            if sc > meilleur_score_points:
                meilleur_score_points = sc
                meilleur_points = bloc

    # Chercher le meilleur bloc non-points
    meilleur_autre = None
    meilleur_score_autre = -9999
    for type_bloc, bloc in tous_les_blocs:
        if type_bloc != 'points':
            sc = score_bloc(bloc)
            if sc > meilleur_score_autre:
                meilleur_score_autre = sc
                meilleur_autre = bloc

    # Règle : si un bloc points de suite a au moins 20 lignes et un score décent,
    # on le préfère systématiquement.
    if meilleur_points is not None and len(meilleur_points) >= 20 and meilleur_score_points > 0:
        meilleur_bloc = meilleur_points
        meilleur_score = meilleur_score_points
    elif meilleur_points is not None and meilleur_score_points >= meilleur_score_autre:
        meilleur_bloc = meilleur_points
        meilleur_score = meilleur_score_points
    else:
        meilleur_bloc = meilleur_autre
        meilleur_score = meilleur_score_autre
        
    pages_meilleur = set(p for p, _, _, _ in meilleur_bloc)
    min_page_meilleur = min(pages_meilleur)
    max_page_meilleur = max(pages_meilleur)

        # -----------------------------------------------------------------
    # 4. Union des bornes : inclure uniquement les blocs pertinents
    # -----------------------------------------------------------------
    def bloc_est_pertinent(bloc):
        motif_structure = re.compile(
            r'^\s*([IVXLCDM]+|\d+)([.\-]\d+)*[.:]?\s+[^\d\s]{2,}',
            re.IGNORECASE
        )
        for _, _, _, ligne in bloc:
            ligne_up = ligne.upper()
            if re.search(r'\b(CHAPITRE|INTRODUCTION|CONCLUSION|SOMMAIRE|TABLE\s+DES\s+MATI[EÈ]RES)\b', ligne_up):
                return True
            if MOTIF_SOMMAIRE.search(ligne):
                return True
            if motif_structure.search(ligne):
                return True
        return False

    bornes = []
    for type_bloc, bloc in tous_les_blocs:
        if not bloc_est_pertinent(bloc):
            continue
        pages_bloc = set(p for p, _, _, _ in bloc)
        min_page_bloc = min(pages_bloc)
        max_page_bloc = max(pages_bloc)
        if min_page_bloc <= max_page_meilleur and max_page_bloc >= min_page_meilleur:
            # On ne prend que les lignes du bloc réellement situées dans la
            # plage de pages du bloc gagnant : un bloc peut chevaucher cette
            # plage sur quelques pages seulement tout en débordant, sur ses
            # autres pages, sur un contenu sans rapport (ex: une annexe
            # adjacente à la table des matières).
            lignes_dans_zone = [
                t for t in bloc if min_page_meilleur <= t[0] <= max_page_meilleur
            ]
            if lignes_dans_zone:
                bornes.append((lignes_dans_zone[0][2], lignes_dans_zone[-1][2]))

    if not bornes:
        bornes = [(meilleur_bloc[0][2], meilleur_bloc[-1][2])]

    premier_global = min(b[0] for b in bornes)
    dernier_global = max(b[1] for b in bornes)

    # -----------------------------------------------------------------
    # 5. Élargir et retourner
    # -----------------------------------------------------------------
    toutes_lignes = []
    compteur = 0
    for num_page, lignes in enumerate(texte_complet):
        for idx_ligne, ligne in enumerate(lignes):
            toutes_lignes.append((num_page, idx_ligne, compteur, ligne.strip().strip('\xa0').strip()))
            compteur += 1

    bloc_elargi = [
        ligne for _, _, global_idx, ligne in toutes_lignes
        if premier_global <= global_idx <= dernier_global
    ]

    nb_entrees = len(bloc_elargi)

    if debug:
        print(f"[DEBUG] Meilleur bloc : score={meilleur_score}, taille={len(meilleur_bloc)}")
        pages = sorted(pages_meilleur)
        print(f"[DEBUG] Pages du meilleur bloc : {pages}")
        print(f"[DEBUG] Union des bornes : {len(bornes)} blocs fusionnés")
        print(f"[DEBUG] Bloc élargi : {nb_entrees} lignes (index {premier_global} à {dernier_global})")

    if retourner_bornes:
        return True, nb_entrees, bloc_elargi, (premier_global, dernier_global)
    return True, nb_entrees, bloc_elargi


# -----------------------------------------------------------------------------
# LIGNES RÉPÉTÉES
# -----------------------------------------------------------------------------

def detecter_lignes_repete(texte_complet, seuil_ratio=0.4, debug=False):
    """
    Détecte TOUTES les lignes fréquentes (en-têtes, pieds de page, fragments).
    Retourne (True, liste_de_lignes, nb_pages_max) ou (False, [], 0).
    """
    nb_pages = len(texte_complet)
    if nb_pages == 0:
        return False, [], 0

    occurrences = {}
    for lignes_page in texte_complet:
        lignes_uniques_page = set(l.strip().strip('\xa0').strip() for l in lignes_page)
        for ligne in lignes_uniques_page:
            if len(ligne) > 30:
                occurrences[ligne] = occurrences.get(ligne, 0) + 1

    lignes_frequentes = []
    for ligne, count in occurrences.items():
        ratio = count / nb_pages
        if ratio >= seuil_ratio:
            lignes_frequentes.append((ligne, count, ratio))

    lignes_frequentes.sort(key=lambda x: x[1], reverse=True)

    if debug:
        print(f"\n[DEBUG] Lignes fréquentes (seuil {seuil_ratio:.0%}) :")
        for ligne, count, ratio in lignes_frequentes:
            print(f"  {count:3d}/{nb_pages} pages ({ratio:.0%}) | {ligne[:80]}")

    if lignes_frequentes:
        return True, [l for l, _, _ in lignes_frequentes], lignes_frequentes[0][1]

    return False, [], 0


# -----------------------------------------------------------------------------
# FONCTION PRINCIPALE D'INSPECTION
# -----------------------------------------------------------------------------

def inspecter_pdf(chemin_pdf):
    """
    Inspecte un PDF et retourne un dictionnaire avec toutes les mesures.
    """
    resultat = {
        'fichier': os.path.basename(chemin_pdf),
        'chemin': chemin_pdf,
        'erreur': None,
    }

    try:
        producteur, nb_pages = extraire_metadonnees(chemin_pdf)
        resultat['producteur'] = producteur
        resultat['nb_pages'] = nb_pages

        texte_complet = extraire_texte(chemin_pdf)

        nb_caracteres = sum(len(l) for lignes_page in texte_complet for l in lignes_page)
        ratio_caracteres_par_page = nb_caracteres / nb_pages if nb_pages > 0 else 0
        resultat['nb_caracteres'] = nb_caracteres
        resultat['ratio_caracteres_par_page'] = round(ratio_caracteres_par_page, 1)
        resultat['extractible'] = nb_caracteres > 0
        resultat['signal_ratio_bas'] = ratio_caracteres_par_page < SEUIL_RATIO_BAS

        sommaire_present, nb_lignes_sommaire, _ = detecter_sommaire(texte_complet)
        resultat['sommaire_present'] = sommaire_present
        resultat['nb_lignes_sommaire'] = nb_lignes_sommaire

        schemas = detecter_schemas_numerotation(texte_complet)
        resultat['schemas_numerotation'] = schemas
        resultat['nb_schemas_differents'] = len(schemas)

        taux_bruit, nb_lignes_bruit, total_lignes = calculer_taux_bruit(texte_complet)
        resultat['taux_bruit'] = round(taux_bruit, 3)
        resultat['nb_lignes_bruit'] = nb_lignes_bruit
        resultat['total_lignes'] = total_lignes
        resultat['signal_bruit_haut'] = taux_bruit > SEUIL_BRUIT_HAUT

        bandeau_present, liste_lignes_repete, nb_pages_max = detecter_lignes_repete(texte_complet)
        resultat['ligne_repete_present'] = bandeau_present
        if bandeau_present:
            resultat['exemple_ligne_repete'] = liste_lignes_repete[0][:100] if liste_lignes_repete else None
            resultat['nb_lignes_repete'] = len(liste_lignes_repete)
            resultat['nb_pages_ligne_repete'] = nb_pages_max
        else:
            resultat['exemple_ligne_repete'] = None
            resultat['nb_lignes_repete'] = 0
            resultat['nb_pages_ligne_repete'] = 0

    except Exception as e:
        resultat['erreur'] = str(e)
        for champ in ['producteur', 'nb_pages', 'nb_caracteres', 'ratio_caracteres_par_page',
                      'extractible', 'signal_ratio_bas', 'sommaire_present', 'nb_lignes_sommaire',
                      'schemas_numerotation', 'nb_schemas_differents', 'taux_bruit',
                      'nb_lignes_bruit', 'total_lignes', 'signal_bruit_haut',
                      'ligne_repete_present', 'exemple_ligne_repete', 'nb_lignes_repete',
                      'nb_pages_ligne_repete']:
            resultat.setdefault(champ, None)

    return resultat


def detecter_schemas_numerotation(texte_complet):
    schemas = {
        'CHAPITRE_romain': 0,
        'decimal.point': 0,
        'romain.point': 0,
        'decimal_simple': 0,
        'romain_simple': 0,
    }
    for lignes_page in texte_complet:
        for ligne in lignes_page:
            ligne_strip = ligne.strip()
            if MOTIF_CHAPITRE.search(ligne_strip):
                schemas['CHAPITRE_romain'] += 1
            if MOTIF_NUMERO_DECIMAL.search(ligne_strip):
                schemas['decimal.point'] += 1
            if MOTIF_NUMERO_ROMAIN.search(ligne_strip):
                schemas['romain.point'] += 1
            if MOTIF_NUMERO_SIMPLE.search(ligne_strip):
                schemas['decimal_simple'] += 1
            if MOTIF_NUMERO_ROMAIN_SIMPLE.search(ligne_strip):
                schemas['romain_simple'] += 1
    return {k: v for k, v in schemas.items() if v > 0}


def calculer_taux_bruit(texte_complet):
    total_lignes = 0
    lignes_bruit = 0
    for lignes_page in texte_complet:
        for ligne in lignes_page:
            total_lignes += 1
            ligne_strip = ligne.strip()
            if (len(ligne_strip) == 0 or
                re.match(r'^[_\-\=\.]{3,}$', ligne_strip) or
                len(ligne_strip) <= 3):
                lignes_bruit += 1
    if total_lignes == 0:
        return 0.0, 0, 0
    taux = lignes_bruit / total_lignes
    return taux, lignes_bruit, total_lignes


# -----------------------------------------------------------------------------
# EXÉCUTION PRINCIPALE
# -----------------------------------------------------------------------------

def main():
    dossier = Path(DOSSIER_CORPUS)
    fichiers_pdf = sorted(dossier.glob('**/*.pdf'))

    print(f"📄 {len(fichiers_pdf)} PDF trouvés dans {DOSSIER_CORPUS}")
    print("=" * 60)

    resultats = []
    for i, chemin_pdf in enumerate(fichiers_pdf, 1):
        print(f"[{i}/{len(fichiers_pdf)}] Inspection de {chemin_pdf.name}...")
        resultat = inspecter_pdf(chemin_pdf)
        resultats.append(resultat)

        if resultat['erreur']:
            print(f"   ⚠️  ERREUR : {resultat['erreur'][:80]}")
        else:
            signaux = []
            if resultat['signal_ratio_bas']:
                signaux.append(f"ratio bas ({resultat['ratio_caracteres_par_page']} car/p)")
            if resultat['signal_bruit_haut']:
                signaux.append(f"bruit haut ({resultat['taux_bruit']})")
            if not resultat['sommaire_present']:
                signaux.append("pas de sommaire détecté")
            if resultat['ligne_repete_present']:
                signaux.append("ligne répétée sur chaque page")
            if signaux:
                print(f"   🔍 Signaux : {', '.join(signaux)}")
            else:
                print(f"   ✅ Rien à signaler")

    df = pd.DataFrame(resultats)
    colonnes_ordre = [
        'fichier', 'nb_pages', 'producteur', 'extractible', 'ratio_caracteres_par_page',
        'signal_ratio_bas', 'sommaire_present', 'nb_lignes_sommaire',
        'nb_schemas_differents', 'schemas_numerotation',
        'taux_bruit', 'signal_bruit_haut',
        'ligne_repete_present', 'nb_lignes_repete', 'nb_pages_ligne_repete', 'erreur'
    ]
    df = df[[c for c in colonnes_ordre if c in df.columns]]
    df.to_excel(FICHIER_SORTIE, index=False, sheet_name='Grille_observation')
    print(f"📊 Grille exportée dans : {FICHIER_SORTIE}")
    return df


if __name__ == '__main__':
    main()
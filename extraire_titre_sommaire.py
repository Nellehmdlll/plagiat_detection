"""
Module d'extraction des titres du sommaire.
Parse le bloc de lignes du sommaire (fourni par inspection.detecter_sommaire)
et extrait les titres structurés avec leur niveau hiérarchique et leur contexte.

Auteur : Sawadogo Rimalguedo Rahimata
Date : 29/08/2026
"""

import re
import os

# -----------------------------------------------------------------------------
# MOTIFS DE DÉTECTION DES TITRES
# -----------------------------------------------------------------------------

# Niveau 1 : CHAPITRE suivi d'un chiffre romain (CHAPITRE I, CHAPITRE IV)
# Le "(?:\d+\s+)?" optionnel tolère un numéro de page mal recollé en tête de
# ligne (artefact d'extraction fréquent avec les longs pointillés qui font
# déborder le numéro de page de la ligne précédente sur la suivante).
MOTIF_CHAPITRE_ROMAIN = re.compile(
    r'^\s*(?:\d+\s+)?(CHAPITRE\s+[IVXLCDM]+)\b',
    re.IGNORECASE
)

# Niveau 1 : CHAPITRE suivi d'un chiffre arabe (CHAPITRE 1, CHAPITRE 2)
MOTIF_CHAPITRE_ARABE = re.compile(
    r'^\s*(?:\d+\s+)?(CHAPITRE\s+\d+)\b',
    re.IGNORECASE
)



# Niveaux 2/3 : numéro en tête de ligne (I.1, 4.2.1, 2.3)
MOTIF_NUMERO_STRUCTURE = re.compile(
    r'^\s*([IVXLCDM]+|\d+)([.\-]\d+)*[.:]?\s+',
    re.IGNORECASE
)
# Motif pour nettoyer les points de suite + numéro de page en fin de ligne
MOTIF_NETTOYAGE = re.compile(
    r'[\.\-_\u2026\u2015\s]{4,}\s*[\divxlcdm]+\s*$',
    re.IGNORECASE
)

# Mots-clés de sections légitimes (non-contenu académique)
MOTS_CLES_LEGITIMES = [
    'REMERCIEMENTS', 'DEDICACE', 'DÉDICACE', 'RESUME', 'RÉSUMÉ',
    'ABSTRACT', 'SUMMARY', 'REFERENCES', 'RÉFÉRENCES', 'BIBLIOGRAPHIE',
    'ANNEXES', 'ANNEXE', 'LISTE DES TABLEAUX', 'LISTE DES FIGURES',
    'LISTE DES ANNEXES', 'LISTE DES SIGLES', 'LISTE DES ABREVIATIONS',
    'LISTE DES ACRONYMES', 'TABLE DES MATIERES', 'TABLE DE MATIÈRES',
    'SOMMAIRE', 'AVANT-PROPOS', 'PREFACE',
]

# Titres de sections sans numéro qui ne doivent JAMAIS être rattachés
# à un titre précédent (ce sont des sections autonomes)
TITRES_SECTIONS_SANS_NUMERO = [
    'INTRODUCTION', 'INTRODUCTION GÉNÉRALE', 'CONCLUSION', 'CONCLUSION GÉNÉRALE',
    'BIBLIOGRAPHIE', 'RÉFÉRENCES', 'RÉFÉRENCES BIBLIOGRAPHIQUES',
    'ANNEXES', 'TABLE DES MATIERES', 'TABLE DE MATIÈRES',
    'REMERCIEMENTS', 'DEDICACE', 'DÉDICACE', 'RESUME', 'RÉSUMÉ',
    'ABSTRACT', 'SUMMARY', 'LISTE DES TABLEAUX', 'LISTE DES FIGURES',
    'LISTE DES ANNEXES', 'LISTE DES SIGLES', 'LISTE DES ABREVIATIONS',
    'LISTE DES ACRONYMES', 'SOMMAIRE', 'AVANT-PROPOS', 'PREFACE',
]

# Titres qui marquent la fin d'une hiérarchie de chapitre (listes de figures/
# tableaux, bibliographie, annexes...). Contrairement à TITRES_SECTIONS_SANS_
# NUMERO (qui inclut Introduction/Conclusion, internes à un chapitre), ceux-ci
# signalent qu'on quitte le contenu numéroté du chapitre : les numéros nus qui
# suivent (légendes de figures, entrées de bibliographie...) ne doivent plus
# être rattachés au chapitre/à la section romaine en cours.
TITRES_RESET_HIERARCHIE = [
    'BIBLIOGRAPHIE', 'RÉFÉRENCES BIBLIOGRAPHIQUES', 'RÉFÉRENCES', 'WEBOGRAPHIE',
    'ANNEXES', 'ANNEXE',
    'TABLE DES FIGURES', 'LISTE DES FIGURES',
    'TABLE DES TABLEAUX', 'LISTE DES TABLEAUX',
    'LISTE DES ANNEXES',
    'CONCLUSION', 'CONCLUSION GÉNÉRALE', 'CONCLUSION GENERALE',
    'TABLE DES MATIERES', 'TABLE DES MATIÈRES', 'TABLE DE MATIÈRES',
]

# Sous-ensemble de TITRES_RESET_HIERARCHIE qui marque la fin définitive du
# corps numéroté (par opposition à un simple "CONCLUSION" de fin de chapitre,
# après lequel un nouveau chapitre implicite peut encore apparaître). Après
# l'un de ces titres, plus aucun numéro nu ne doit être promu chapitre.
# "TABLE DES MATIERES" en est exclu : dans un sommaire bref qui renvoie vers
# une table des matières détaillée plus loin dans le document (ex: "... vii"
# en fin de ligne), cette mention apparaît AU MILIEU du bloc — pas à la vraie
# fin du corps numéroté — et verrouillerait à tort toute détection de chapitre
# pour le reste du document.
TITRES_FIN_DEFINITIVE = [
    m for m in TITRES_RESET_HIERARCHIE
    if m not in ('CONCLUSION', 'TABLE DES MATIERES', 'TABLE DES MATIÈRES', 'TABLE DE MATIÈRES')
]


# -----------------------------------------------------------------------------
# FONCTION PRINCIPALE
# -----------------------------------------------------------------------------

def extraire_titres_sommaire(bloc_sommaire, debug=False):
    """
    Parse un bloc de sommaire et extrait les titres structurés.
    Gère :
    - Les vrais chapitres (CHAPITRE X)
    - Les parties (romains simples en majuscules sans mot CHAPITRE)
    - Les sections/sous-sections
    - Le rattachement des titres cassés
    - Les clés uniques par contexte de chapitre/partie
    """
    titres = []
    chapitre_courant = None
    section_romaine_vue = False
    deja_vu_mot_chapitre = False  # True dès qu'on a rencontré CHAPITRE
    # True dès qu'un vrai titre niveau 1 (chapitre) a été posé. Sert à ne pas
    # confondre un marqueur comme "Liste des tableaux" ou "Bibliographie" cité
    # en avant-propos (dans le sommaire bref, avant tout chapitre réel) avec
    # le même marqueur employé plus loin comme vraie section de fin de document.
    au_moins_un_chapitre_vu = False
    # True quand chapitre_courant vient d'un numéro nu (1, 2, 3...) utilisé
    # comme chapitre implicite, dans un schéma décimal continu (1, 1.1, 3.1.1) :
    # la profondeur des enfants se déduit alors directement de nb_separateurs,
    # sans le palier supplémentaire qu'introduit un vrai mot-clé CHAPITRE/PARTIE.
    chapitre_numerotation_continue = False
    # True dès qu'on a franchi un marqueur de fin de hiérarchie (bibliographie,
    # liste de figures/tableaux...). Un numéro nu rencontré après ce point
    # (ex: la légende "1." d'une figure) ne doit plus jamais être promu
    # chapitre implicite : ce n'est pas parce qu'il "repart à 1" que c'est un
    # nouveau chapitre — le vrai corps numéroté du document est terminé.
    hors_chapitres = False
    # True entre un marqueur "Introduction" et le "Conclusion" qui le referme.
    # Certains sommaires mal formatés font recommencer leur numérotation à 1
    # à l'intérieur de CHAQUE chapitre (ex: chapitre "2 État de l'art" suivi
    # de sous-parties nues "1 Techniques...", "2 Applications...") au lieu
    # d'utiliser "2.1", "2.2" — ce numéro nu est alors indiscernable d'un
    # nouveau chapitre par sa seule forme. Le bracket Introduction/Conclusion
    # sert de garde-fou : tant qu'on y est, un numéro nu reste une sous-partie
    # du chapitre en cours, jamais un nouveau chapitre implicite.
    dans_corps_chapitre = False

    # Pour le rattachement des titres cassés
    titre_precedent_avait_numero = False
    titre_precedent_etait_legitime = False

    for idx, ligne in enumerate(bloc_sommaire):
        ligne_nettoyee = ligne.strip().strip('\xa0').strip()
        if not ligne_nettoyee:
            continue

        ligne_sans_fin = MOTIF_NETTOYAGE.sub('', ligne_nettoyee).strip()

        if debug:
            print(f"\n[DEBUG] Ligne {idx} : '{ligne_sans_fin[:80]}'")
            print(f"  → Contexte : chapitre={chapitre_courant}, romain_vu={section_romaine_vue}, deja_vu_mot_chapitre={deja_vu_mot_chapitre}")

        if len(ligne_sans_fin) < 3:
            if debug:
                print(f"  → Trop courte ({len(ligne_sans_fin)} chars), ignorée")
            continue

        if re.match(r'^[IVXLCDM]+$', ligne_sans_fin, re.IGNORECASE):
            if debug:
                print(f"  → Numéro seul, ignorée")
            continue

        if re.match(r'^\d+$', ligne_sans_fin):
            if debug:
                print(f"  → Chiffre seul, ignorée")
            continue

        ligne_upper_stricte = ligne_sans_fin.strip().upper()
        if ligne_upper_stricte == 'INTRODUCTION':
            # Un "Introduction" isolé avant tout chapitre établi est
            # l'introduction générale du document, pas le marqueur de corps
            # d'un chapitre déjà commencé : il ne doit rien bloquer.
            if chapitre_courant is not None:
                dans_corps_chapitre = True
        elif ligne_upper_stricte == 'CONCLUSION':
            dans_corps_chapitre = False

        numero = None
        niveau = 0
        texte_restant = ligne_sans_fin

        match_chapitre_romain = MOTIF_CHAPITRE_ROMAIN.search(ligne_sans_fin)
        match_chapitre_arabe = MOTIF_CHAPITRE_ARABE.search(ligne_sans_fin)

        if match_chapitre_romain:
            # VRAI CHAPITRE (niveau 1)
            numero = match_chapitre_romain.group(1)
            niveau = 1
            texte_restant = ligne_sans_fin[match_chapitre_romain.end():].strip()
            if texte_restant.startswith(':'):
                texte_restant = texte_restant[1:].strip()

            chapitre_courant = numero
            section_romaine_vue = False
            deja_vu_mot_chapitre = True
            chapitre_numerotation_continue = False
            au_moins_un_chapitre_vu = True
            dans_corps_chapitre = False
            if debug:
                print(f"  → NOUVEAU CHAPITRE : {chapitre_courant}")

        elif match_chapitre_arabe:
            # VRAI CHAPITRE (niveau 1)
            numero = match_chapitre_arabe.group(1)
            niveau = 1
            texte_restant = ligne_sans_fin[match_chapitre_arabe.end():].strip()
            if texte_restant.startswith(':'):
                texte_restant = texte_restant[1:].strip()

            chapitre_courant = numero
            section_romaine_vue = False
            deja_vu_mot_chapitre = True
            chapitre_numerotation_continue = False
            au_moins_un_chapitre_vu = True
            dans_corps_chapitre = False
            if debug:
                print(f"  → NOUVEAU CHAPITRE : {chapitre_courant}")

        else:
            match_numero = MOTIF_NUMERO_STRUCTURE.search(ligne_sans_fin)
            if match_numero:
                numero_brut = match_numero.group(0).strip()
                # Supprimer le terminateur (point ou deux-points) en fin
                numero_clean = numero_brut.rstrip('.:')
                # Compter les séparateurs internes (point OU tiret)
                nb_separateurs = numero_clean.count('.') + numero_clean.count('-')

                # Détecter si c'est un romain simple (ex: I., II:, III)
                est_romain_simple = re.match(r'^[IVXLCDM]+$', numero_clean, re.IGNORECASE)

                if est_romain_simple:
                    texte_majuscule = texte_restant.upper() == texte_restant

                    if texte_majuscule and not deja_vu_mot_chapitre:
                        # Aucun mot CHAPITRE rencontré : c'est une PARTIE
                        niveau = 1
                        chapitre_courant = numero_brut
                        section_romaine_vue = False
                        chapitre_numerotation_continue = False
                        au_moins_un_chapitre_vu = True
                        dans_corps_chapitre = False
                        if debug:
                            print(f"  → PARTIE (romain simple) : {chapitre_courant} → niveau 1")
                    elif chapitre_courant is not None:
                        niveau = 2
                        section_romaine_vue = True
                        # Ce romain introduit une sous-numérotation propre :
                        # les numéros nus qui suivent (1., 2., 3.) sont ses
                        # enfants, pas de nouveaux chapitres implicites.
                        chapitre_numerotation_continue = False
                        if debug:
                            print(f"  → Romain simple sous {chapitre_courant} → niveau 2")
                    else:
                        niveau = 2
                        section_romaine_vue = True
                        if debug:
                            print(f"  → Romain simple hors chapitre → niveau 2")
                else:
                    # Numéro arabe simple ou décimal
                    if (chapitre_numerotation_continue and nb_separateurs == 0
                            and chapitre_courant is not None and dans_corps_chapitre):
                        # On est entre le marqueur "Introduction" et le
                        # "Conclusion" du chapitre courant : ce numéro nu
                        # recommence une sous-numérotation locale (le sommaire
                        # est mal formaté par l'auteur), ce n'est pas un
                        # nouveau chapitre.
                        niveau = 2
                        if debug:
                            print(f"  → Sous-partie locale (corps du chapitre {chapitre_courant}) → niveau 2")
                    elif (chapitre_numerotation_continue and nb_separateurs == 0
                            and chapitre_courant is not None):
                        # Nouveau numéro nu dans un schéma décimal continu déjà
                        # entamé (ex: "2" après "1.3") : c'est un nouveau
                        # chapitre implicite, pas un enfant du précédent.
                        niveau = 1
                        chapitre_courant = numero_brut
                        au_moins_un_chapitre_vu = True
                        dans_corps_chapitre = False
                        if debug:
                            print(f"  → Chapitre implicite (numérotation continue) : {chapitre_courant} → niveau 1")
                    elif chapitre_courant is not None:
                        # Un numéro composé qui répète déjà le préfixe romain
                        # (ex: "I.1", "I.2.1") encode sa profondeur complète à
                        # lui seul ; un numéro nu (ex: "1.", "2.") qui repart
                        # de zéro sous chaque romain (I., II., III.) a besoin
                        # d'un palier de profondeur supplémentaire.
                        numero_repete_romain = re.match(
                            r'^[IVXLCDM]+[.\-]', numero_clean, re.IGNORECASE
                        )
                        if chapitre_numerotation_continue:
                            # Le chapitre lui-même fait partie du même schéma
                            # décimal (1, 1.1, 3.1.1...) : nb_separateurs seul
                            # donne déjà la profondeur.
                            niveau = nb_separateurs + 1
                        elif section_romaine_vue and not numero_repete_romain:
                            # Numéro nu recommençant sous un romain simple
                            # (I., II.) déjà rencontré dans ce chapitre : un
                            # niveau plus profond.
                            niveau = nb_separateurs + 3
                        else:
                            # Sous un chapitre/une partie "mot-clé", ou numéro
                            # composé encodant déjà sa profondeur : niveau minimum 2
                            niveau = nb_separateurs + 2
                    else:
                        if nb_separateurs == 0:
                            if not deja_vu_mot_chapitre and not hors_chapitres and not dans_corps_chapitre:
                                niveau = 1
                                chapitre_courant = numero_brut
                                chapitre_numerotation_continue = True
                                section_romaine_vue = False
                                au_moins_un_chapitre_vu = True
                            else:
                                # dans_corps_chapitre=True ici signifie qu'un
                                # marqueur "Introduction" sans "Conclusion" est
                                # toujours actif : chapitre_courant a pu être
                                # remis à None entre-temps par une ligne parasite
                                # (ex: un en-tête "TABLE DES MATIÈRES viii" qui a
                                # débordé sur sa propre ligne), mais on est encore
                                # dans le corps du chapitre — ne pas re-promouvoir.
                                niveau = 2
                        else:
                            niveau = nb_separateurs + 1
                    if debug:
                        print(f"  → Numéro {numero_brut} → nb_separateurs={nb_separateurs}, niveau={niveau}")

                numero = numero_brut
                texte_restant = ligne_sans_fin[match_numero.end():].strip()

        # Construction de la clé unique
        if numero is not None and chapitre_courant is not None:
            if not (match_chapitre_romain or match_chapitre_arabe):
                cle_unique = f"{chapitre_courant}__{numero}"
            else:
                cle_unique = numero
        elif numero is not None:
            cle_unique = numero
        else:
            cle_unique = None

        if numero is None and any(
            mot_cle in ligne_sans_fin.upper() for mot_cle in TITRES_RESET_HIERARCHIE
        ):
            # Liste de figures/tableaux, bibliographie, annexes... : on sort
            # du contenu numéroté du chapitre en cours. Les numéros nus qui
            # suivent (légendes, entrées de bibliographie) ne doivent pas
            # être rattachés au chapitre ni à la section romaine précédente.
            chapitre_courant = None
            section_romaine_vue = False
            chapitre_numerotation_continue = False
            if au_moins_un_chapitre_vu and any(
                mot_cle in ligne_sans_fin.upper() for mot_cle in TITRES_FIN_DEFINITIVE
            ):
                hors_chapitres = True
            if debug:
                print(f"  → Marqueur de fin de hiérarchie : contexte de chapitre réinitialisé")

        texte_upper = texte_restant.upper()
        est_legitime = any(
            mot_cle in texte_upper
            for mot_cle in MOTS_CLES_LEGITIMES
        )

        # Gestion des titres cassés
        ligne_est_titre_section = any(
            mot_cle in ligne_sans_fin.upper()
            for mot_cle in TITRES_SECTIONS_SANS_NUMERO
        )

        if (numero is None and
            titre_precedent_avait_numero and
            not titre_precedent_etait_legitime and
            len(ligne_sans_fin) <= 40 and
            not ligne_est_titre_section and
            titres):

            titres[-1]['texte'] = titres[-1]['texte'] + ' ' + ligne_sans_fin
            if debug:
                print(f"  → Rattachée à : '{titres[-1]['texte'][-60:]}'")
            titre_precedent_avait_numero = False
            continue

        titre = {
            'numero': numero,
            'cle_unique': cle_unique,
            'texte': texte_restant if texte_restant else ligne_sans_fin,
            'niveau': niveau,
            'legitime': est_legitime,
            'chapitre_parent': chapitre_courant,
        }
        titres.append(titre)

        if debug:
            print(f"  → Titre ajouté : niveau={niveau}, numero={numero}, texte={texte_restant[:50]}")

        titre_precedent_avait_numero = (numero is not None)
        titre_precedent_etait_legitime = est_legitime

    return titres
# -----------------------------------------------------------------------------
# CHAÎNE COMPLÈTE : extraction → détection → parsing
# -----------------------------------------------------------------------------
def extraire_et_parser_sommaire(chemin_pdf, debug=False):
    """
    Chaîne complète : extrait le texte, détecte le sommaire,
    filtre les bandeaux, puis parse les titres.
    """
    from inspection import extraire_texte, detecter_sommaire, detecter_lignes_repete
    
    # 1. Extraction
    texte_complet = extraire_texte(chemin_pdf)
    
    # 2. Détection du sommaire
    sommaire_present, nb_entrees, bloc_sommaire = detecter_sommaire(texte_complet, debug=debug)
    
    if not sommaire_present:
        print(f"[INFO] Pas de sommaire détecté dans {os.path.basename(chemin_pdf)}")
        return []
    
    # 3. Détection des lignes fréquentes
    bandeau_present, liste_bandeaux, nb_pages_max = detecter_lignes_repete(
        texte_complet, debug=debug
    )
    
    if debug:
        print(f"\n[DEBUG] Bandeaux détectés : {len(liste_bandeaux)}")
        for b in liste_bandeaux[:5]:
            print(f"  → {b[:80]}")
    
    # 4. Filtrer le bloc
    if bandeau_present and liste_bandeaux:
        bloc_sommaire = filtrer_bandeau(bloc_sommaire, liste_bandeaux, debug=debug)
        if debug:
            print(f"[DEBUG] Bloc après filtrage : {len(bloc_sommaire)} lignes")
    
    # 5. Parsing
    titres = extraire_titres_sommaire(bloc_sommaire, debug=debug)
    
    return titres
# -----------------------------------------------------------------------------
# FILTRAGE DU BANDEAU RÉPÉTÉ
# -----------------------------------------------------------------------------

def normaliser_texte(texte):
    """
    Normalise un texte pour comparaison : minuscules, sans accents,
    espaces multiples réduits à un seul, sans caractères spéciaux.
    """
    import unicodedata
    texte = texte.lower()
    texte = unicodedata.normalize('NFD', texte)
    texte = ''.join(c for c in texte if unicodedata.category(c) != 'Mn')
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte

def correspond_au_bandeau(ligne_normalisee, bandeau_normalise):
    """
    Vérifie si une ligne est le bandeau ou un fragment du bandeau.
    """
    if not bandeau_normalise or len(bandeau_normalise) < 10:
        return False
    
    # Cas 1 : la ligne EST le bandeau
    if ligne_normalisee == bandeau_normalise:
        return True
    
    # Cas 2 : la ligne est entièrement contenue dans le bandeau
    if ligne_normalisee in bandeau_normalise:
        return True
    
    # Cas 3 : le bandeau est contenu dans la ligne
    if bandeau_normalise in ligne_normalisee:
        return True
    
    # Cas 4 : les 5 premiers mots de la ligne sont dans le bandeau
    mots_ligne = ligne_normalisee.split()
    if len(mots_ligne) >= 5:
        debut_ligne = ' '.join(mots_ligne[:5])
        if debut_ligne in bandeau_normalise:
            return True
    
    # Cas 5 : les 5 derniers mots de la ligne sont dans le bandeau
    if len(mots_ligne) >= 5:
        fin_ligne = ' '.join(mots_ligne[-5:])
        if fin_ligne in bandeau_normalise:
            return True
    
    # Cas 6 : fort chevauchement de mots (au moins 60% des mots de la ligne
    # sont dans le bandeau)
    mots_bandeau = set(bandeau_normalise.split())
    mots_ligne_set = set(mots_ligne)
    intersection = mots_ligne_set & mots_bandeau
    if len(mots_ligne_set) > 0:
        taux = len(intersection) / len(mots_ligne_set)
        if taux >= 0.6:
            return True
    
    return False

def filtrer_bandeau(bloc_sommaire, liste_bandeaux, debug=False):
    """
    Retire du bloc sommaire les lignes qui correspondent à l'un des bandeaux.
    Ne retire JAMAIS les lignes qui ont un numéro de section.
    """
    if not liste_bandeaux:
        return bloc_sommaire
    
    lignes_filtrees = []
    lignes_retirees = []
    
    for ligne in bloc_sommaire:
        # Protéger les lignes avec numéro de section
        a_numero = bool(re.match(r'^\s*(CHAPITRE|[IVXLCDM]+|\d+)(\.\d+)*[\.:]?\s+', ligne, re.IGNORECASE))
        
        if a_numero:
            lignes_filtrees.append(ligne)
            continue
        
        ligne_normalisee = normaliser_texte(ligne)
        est_bandeau = False
        
        for bandeau in liste_bandeaux:
            bandeau_normalise = normaliser_texte(bandeau)
            if correspond_au_bandeau(ligne_normalisee, bandeau_normalise):
                est_bandeau = True
                break
        
        if est_bandeau:
            lignes_retirees.append(ligne)
        else:
            lignes_filtrees.append(ligne)
    
    if debug:
        print(f"[DEBUG] Lignes retirées par filtre : {len(lignes_retirees)}")
        for ligne in lignes_retirees[:10]:
            print(f"  → {ligne[:80]}")
    
    return lignes_filtrees
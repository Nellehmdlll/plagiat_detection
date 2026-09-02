"""
Script de test sur l'ensemble du corpus.
Extrait les titres du sommaire de chaque PDF et produit des statistiques de structure.

Auteur : Sawadogo Rimalguedo Rahimata
Date : 30/08/2026
"""

import os
import re
import pandas as pd
from pathlib import Path

from inspection import extraire_texte, detecter_sommaire, detecter_lignes_repete
from extraire_titre_sommaire import extraire_titres_sommaire, filtrer_bandeau, normaliser_texte, correspond_au_bandeau
from localiser_titres_corps import localiser_titres_dans_corps
from decouper_segments import decouper_en_segments

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

DOSSIER_CORPUS = r"C:\Users\1\Documents\MASTER_SOUTENANCE\corpus"
FICHIER_SORTIE = "statistiques_structure_corpus.xlsx"
DOSSIER_CORPUS2 = r"C:\Users\1\Documents\MASTER_SOUTENANCE\corpus\corpus2"


# -----------------------------------------------------------------------------
# FONCTION D'ANALYSE D'UN DOCUMENT
# -----------------------------------------------------------------------------

def analyser_document(chemin_pdf, debug=False):
    """
    Analyse un document : extrait le sommaire, filtre les bandeaux,
    parse les titres, et retourne un dictionnaire de statistiques.
    """
    resultat = {
        'fichier': os.path.basename(chemin_pdf),
        'erreur': None,
    }

    try:
        # 1. Extraction du texte
        texte_complet = extraire_texte(chemin_pdf)

        # 2. Détection du sommaire
        sommaire_present, nb_entrees, bloc_sommaire = detecter_sommaire(texte_complet)
        resultat['sommaire_present'] = sommaire_present
        resultat['nb_lignes_bloc_sommaire'] = nb_entrees

        if not sommaire_present:
            resultat['nb_titres'] = 0
            resultat['nb_chapitres'] = 0
            resultat['nb_sections_niveau2'] = 0
            resultat['nb_sections_niveau3'] = 0
            resultat['nb_sections_niveau4'] = 0
            resultat['nb_sections_legitimes'] = 0
            resultat['profondeur_max'] = 0
            resultat['nb_titres_sans_numero'] = 0
            resultat['nb_titres_casses'] = 0
            resultat['nb_bandeaux_filtres'] = 0
            resultat['nb_titres_localises'] = 0
            resultat['taux_localisation'] = 0.0
            resultat['nb_sections_chunks'] = 0
            resultat['nb_chunks'] = 0
            resultat['nb_sections_multi_chunks'] = 0
            resultat['nb_chunks_suspects'] = 0
            resultat['nb_chunks_vides'] = 0
            resultat['taille_chunk_min'] = 0
            resultat['taille_chunk_max'] = 0
            resultat['taille_chunk_moyenne'] = 0.0
            return resultat

        # 3. Détection des lignes fréquentes
        bandeau_present, liste_bandeaux, nb_pages_max = detecter_lignes_repete(texte_complet)
        resultat['bandeau_present'] = bandeau_present
        resultat['nb_bandeaux_frequents'] = len(liste_bandeaux)

        # 4. Filtrer les bandeaux
        nb_lignes_avant = len(bloc_sommaire)
        if bandeau_present and liste_bandeaux:
            bloc_sommaire = filtrer_bandeau(bloc_sommaire, liste_bandeaux)
        nb_lignes_apres = len(bloc_sommaire)
        resultat['nb_bandeaux_filtres'] = nb_lignes_avant - nb_lignes_apres

        # 5. Parsing des titres
        titres = extraire_titres_sommaire(bloc_sommaire)
        resultat['nb_titres'] = len(titres)

        # 6. Statistiques sur les titres
        nb_chapitres = sum(1 for t in titres if t['niveau'] == 1)
        nb_niveau2 = sum(1 for t in titres if t['niveau'] == 2)
        nb_niveau3 = sum(1 for t in titres if t['niveau'] == 3)
        nb_niveau4 = sum(1 for t in titres if t['niveau'] == 4)
        nb_legitimes = sum(1 for t in titres if t['legitime'])
        nb_sans_numero = sum(1 for t in titres if t['numero'] is None)
        nb_titres_casses = sum(1 for t in titres if t['texte'].count('  ') > 0 or len(t['texte']) > 80)

        niveaux_presents = set(t['niveau'] for t in titres if t['niveau'] > 0)
        profondeur_max = max(niveaux_presents) if niveaux_presents else 0

        resultat['nb_chapitres'] = nb_chapitres
        resultat['nb_sections_niveau2'] = nb_niveau2
        resultat['nb_sections_niveau3'] = nb_niveau3
        resultat['nb_sections_niveau4'] = nb_niveau4
        resultat['nb_sections_legitimes'] = nb_legitimes
        resultat['nb_titres_sans_numero'] = nb_sans_numero
        resultat['nb_titres_casses'] = nb_titres_casses
        resultat['profondeur_max'] = profondeur_max

        # Schémas de numérotation détectés
        schemas = set()
        for t in titres:
            if t['numero']:
                if 'CHAPITRE' in t['numero'].upper():
                    schemas.add('CHAPITRE')
                elif re.match(r'^[IVXLCDM]+\.?$', t['numero'], re.IGNORECASE):
                    schemas.add('romain_simple')
                elif re.match(r'^[IVXLCDM]+\.\d+', t['numero'], re.IGNORECASE):
                    schemas.add('romain_decimal')
                elif re.match(r'^\d+\.\d+', t['numero']):
                    schemas.add('decimal')
                elif re.match(r'^\d+$', t['numero']):
                    schemas.add('arabe_simple')
        resultat['schemas_titres'] = ', '.join(sorted(schemas))

        # 7. Localisation des titres dans le corps du document
        titres = localiser_titres_dans_corps(texte_complet, titres)
        nb_localises = sum(1 for t in titres if t['localise'])
        resultat['nb_titres_localises'] = nb_localises
        resultat['taux_localisation'] = round(nb_localises / len(titres), 3) if titres else 0.0

        # 8. Découpage en chunks
        chunks = decouper_en_segments(texte_complet, titres, liste_bandeaux=liste_bandeaux)
        sections_chunks = {}
        for c in chunks:
            sections_chunks.setdefault(c['position_debut'], []).append(c)
        tailles_mots = [len(c['texte'].split()) for c in chunks]

        resultat['nb_sections_chunks'] = len(sections_chunks)
        resultat['nb_chunks'] = len(chunks)
        resultat['nb_sections_multi_chunks'] = sum(1 for cs in sections_chunks.values() if len(cs) > 1)
        resultat['nb_chunks_suspects'] = sum(1 for n in tailles_mots if n < 10)
        resultat['nb_chunks_vides'] = sum(1 for n in tailles_mots if n == 0)
        resultat['taille_chunk_min'] = min(tailles_mots) if tailles_mots else 0
        resultat['taille_chunk_max'] = max(tailles_mots) if tailles_mots else 0
        resultat['taille_chunk_moyenne'] = round(sum(tailles_mots) / len(tailles_mots), 1) if tailles_mots else 0.0

    except Exception as e:
        resultat['erreur'] = str(e)
        for champ in ['sommaire_present', 'nb_lignes_bloc_sommaire', 'nb_titres',
                      'nb_chapitres', 'nb_sections_niveau2', 'nb_sections_niveau3',
                      'nb_sections_niveau4', 'nb_sections_legitimes',
                      'nb_titres_sans_numero', 'nb_titres_casses',
                      'profondeur_max', 'schemas_titres',
                      'bandeau_present', 'nb_bandeaux_frequents', 'nb_bandeaux_filtres',
                      'nb_titres_localises', 'taux_localisation',
                      'nb_sections_chunks', 'nb_chunks', 'nb_sections_multi_chunks',
                      'nb_chunks_suspects', 'nb_chunks_vides',
                      'taille_chunk_min', 'taille_chunk_max', 'taille_chunk_moyenne']:
            resultat.setdefault(champ, None)

    return resultat


# -----------------------------------------------------------------------------
# EXÉCUTION PRINCIPALE
# -----------------------------------------------------------------------------

def main():
    dossier = Path(DOSSIER_CORPUS)
    fichiers_pdf = sorted(dossier.glob('**/*.pdf'))

    print(f"📄 {len(fichiers_pdf)} PDF trouvés")
    print("=" * 60)

    resultats = []
    for i, chemin_pdf in enumerate(fichiers_pdf, 1):
        print(f"[{i}/{len(fichiers_pdf)}] Analyse de {chemin_pdf.name}...")
        resultat = analyser_document(chemin_pdf)
        resultats.append(resultat)

        if resultat['erreur']:
            print(f"   ⚠️  ERREUR : {resultat['erreur'][:80]}")
        else:
            taux = resultat.get('taux_localisation')
            taux_str = f"{taux:.0%}" if taux is not None else "n/a"
            print(f"   📊 {resultat['nb_titres']} titres, {resultat['nb_chapitres']} chapitres, "
                  f"profondeur max {resultat['profondeur_max']}, localisation {taux_str}, "
                  f"{resultat.get('nb_chunks')} chunks ({resultat.get('nb_sections_multi_chunks')} redécoupés, "
                  f"{resultat.get('nb_chunks_suspects')} suspects)")

    print("=" * 60)

    # Construction du DataFrame
    df = pd.DataFrame(resultats)

    # Réorganisation des colonnes
    colonnes_ordre = [
        'fichier', 'nb_titres', 'nb_chapitres', 'nb_sections_niveau2',
        'nb_sections_niveau3', 'nb_sections_niveau4', 'profondeur_max',
        'nb_sections_legitimes', 'nb_titres_sans_numero', 'nb_titres_casses',
        'nb_lignes_bloc_sommaire', 'nb_bandeaux_frequents', 'nb_bandeaux_filtres',
        'nb_titres_localises', 'taux_localisation',
        'nb_sections_chunks', 'nb_chunks', 'nb_sections_multi_chunks',
        'nb_chunks_suspects', 'nb_chunks_vides',
        'taille_chunk_min', 'taille_chunk_max', 'taille_chunk_moyenne',
        'schemas_titres', 'sommaire_present', 'bandeau_present', 'erreur'
    ]
    df = df[[c for c in colonnes_ordre if c in df.columns]]

    # Export Excel
    df.to_excel(FICHIER_SORTIE, index=False, sheet_name='Statistiques_structure')
    print(f"\n📊 Statistiques exportées dans : {FICHIER_SORTIE}")

    # -----------------------------------------------------------------
    # STATISTIQUES D'ENSEMBLE
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STATISTIQUES DE STRUCTURE DU CORPUS")
    print("=" * 60)

    print(f"\nDocuments analysés : {len(df)}")
    print(f"Documents avec erreur : {df['erreur'].notna().sum()}")
    print(f"Documents avec sommaire détecté : {df['sommaire_present'].sum()}/{len(df)}")

    print(f"\n--- PROFONDEUR HIÉRARCHIQUE ---")
    print(df['profondeur_max'].value_counts(dropna=False).sort_index())

    print(f"\n--- NOMBRE DE TITRES PAR DOCUMENT ---")
    print(f"Minimum : {df['nb_titres'].min()}")
    print(f"Maximum : {df['nb_titres'].max()}")
    print(f"Moyenne : {df['nb_titres'].mean():.1f}")

    print(f"\n--- NOMBRE DE CHAPITRES PAR DOCUMENT ---")
    print(f"Minimum : {df['nb_chapitres'].min()}")
    print(f"Maximum : {df['nb_chapitres'].max()}")
    print(f"Moyenne : {df['nb_chapitres'].mean():.1f}")

    print(f"\n--- SCHÉMAS DE NUMÉROTATION ---")
    schema_counts = {}
    for schemas in df['schemas_titres'].dropna():
        for schema in schemas.split(', '):
            if schema:
                schema_counts[schema] = schema_counts.get(schema, 0) + 1
    for schema, count in sorted(schema_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {schema} : {count} documents")

    print(f"\n--- SECTIONS LÉGITIMES ---")
    print(f"Minimum : {df['nb_sections_legitimes'].min()}")
    print(f"Maximum : {df['nb_sections_legitimes'].max()}")
    print(f"Moyenne : {df['nb_sections_legitimes'].mean():.1f}")

    print(f"\n--- BANDEAUX FILTRÉS ---")
    print(f"Documents avec bandeau : {df['bandeau_present'].sum()}/{len(df)}")
    print(f"Moyenne de fragments filtrés : {df['nb_bandeaux_filtres'].mean():.1f}")

    print(f"\n--- TITRES CASSÉS ---")
    print(f"Documents avec titres cassés : {(df['nb_titres_casses'] > 0).sum()}/{len(df)}")

    print(f"\n--- LOCALISATION DES TITRES DANS LE CORPS ---")
    print(f"Taux moyen : {df['taux_localisation'].mean():.1%}")
    print(f"Taux minimum : {df['taux_localisation'].min():.1%}")
    print(f"Documents sous 80% de localisation : {(df['taux_localisation'] < 0.80).sum()}/{len(df)}")
    print(f"Documents sous 70% de localisation : {(df['taux_localisation'] < 0.70).sum()}/{len(df)}")

    print(f"\n--- DÉCOUPAGE EN CHUNKS ---")
    print(f"Sections extraites : min={df['nb_sections_chunks'].min()}, max={df['nb_sections_chunks'].max()}, "
          f"moyenne={df['nb_sections_chunks'].mean():.1f}")
    print(f"Chunks produits : min={df['nb_chunks'].min()}, max={df['nb_chunks'].max()}, "
          f"moyenne={df['nb_chunks'].mean():.1f}, total={df['nb_chunks'].sum()}")
    print(f"Sections redécoupées (> taille_max_chunk) : {df['nb_sections_multi_chunks'].sum()} au total")
    print(f"Chunks vides : {df['nb_chunks_vides'].sum()} au total")
    print(f"Chunks suspects (< 10 mots) : {df['nb_chunks_suspects'].sum()} au total, "
          f"sur {(df['nb_chunks_suspects'] > 0).sum()}/{len(df)} documents")
    print(f"Taille de chunk (mots) : min={df['taille_chunk_min'].min()}, max={df['taille_chunk_max'].max()}, "
          f"moyenne des moyennes={df['taille_chunk_moyenne'].mean():.1f}")

    documents_atypiques = df[
        (df['profondeur_max'] < 3) |
        (df['nb_titres'] < 20) |
        (df['nb_titres'] > 120) |
        (df['nb_chapitres'] == 0) |
        (df['taux_localisation'] < 0.70) |
        (df['nb_chunks_vides'] > 0) |
        (df['nb_chunks_suspects'] > 5) |
        (df['erreur'].notna())
    ]

    print(f"\n⚠️  Documents atypiques : {len(documents_atypiques)}")
    if len(documents_atypiques) > 0:
        for _, row in documents_atypiques.iterrows():
            raisons = []
            if row.get('profondeur_max') and row['profondeur_max'] < 3:
                raisons.append(f"profondeur faible ({row['profondeur_max']})")
            if row.get('nb_titres') and row['nb_titres'] < 20:
                raisons.append(f"peu de titres ({row['nb_titres']})")
            if row.get('nb_titres') and row['nb_titres'] > 120:
                raisons.append(f"beaucoup de titres ({row['nb_titres']})")
            if row.get('nb_chapitres') == 0:
                raisons.append("aucun chapitre détecté")
            if row.get('taux_localisation') is not None and row['taux_localisation'] < 0.70:
                raisons.append(f"localisation faible ({row['taux_localisation']:.0%})")
            if row.get('nb_chunks_vides'):
                raisons.append(f"{row['nb_chunks_vides']} chunk(s) vide(s)")
            if row.get('nb_chunks_suspects') and row['nb_chunks_suspects'] > 5:
                raisons.append(f"{row['nb_chunks_suspects']} chunks suspects (<10 mots)")
            if row.get('erreur'):
                raisons.append(f"erreur : {row['erreur'][:50]}")
            print(f"   • {row['fichier']} : {', '.join(raisons)}")

    return df


if __name__ == '__main__':
    df_resultat = main()
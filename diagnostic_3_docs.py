"""
Diagnostic ciblé sur les documents 07, 15, 17 pour voir la forme
exacte de leur sommaire (titres, numérotation, etc.)
"""

import os
from inspection import extraire_texte, detecter_sommaire

DOSSIER = r"C:\Users\1\Documents\MASTER_SOUTENANCE\corpus"
DOCS = [
    "07-memoire.pdf",
    "15-memoire.pdf",
    "17-memoire.pdf",
]

def diagnostiquer_un_document(chemin_pdf, nb_lignes=60):
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC : {os.path.basename(chemin_pdf)}")
    print('='*80)

    texte_complet = extraire_texte(chemin_pdf)
    sommaire_present, nb_entrees, bloc_sommaire = detecter_sommaire(texte_complet)

    print(f"Sommaire détecté : {sommaire_present}, lignes dans le bloc : {nb_entrees}")

    if not sommaire_present:
        print("Aucun sommaire détecté.")
        return

    print(f"\n--- {min(nb_lignes, len(bloc_sommaire))} premières lignes du bloc sommaire (avec repr) ---")
    for i, ligne in enumerate(bloc_sommaire[:nb_lignes], 1):
        print(f"Ligne {i:3d} : {repr(ligne[:150])}")

    # Afficher aussi les motifs de numérotation uniques détectés grossièrement
    print("\n--- Échantillon de débuts de lignes (15 premiers caractères nettoyés) ---")
    for i, ligne in enumerate(bloc_sommaire[:nb_lignes], 1):
        ligne_nettoyee = ligne.strip().strip('\xa0').strip()
        print(f"Ligne {i:3d} : {ligne_nettoyee[:60]}")

def main():
    for doc in DOCS:
        chemin = os.path.join(DOSSIER, doc)
        if os.path.exists(chemin):
            diagnostiquer_un_document(chemin)
        else:
            print(f"Fichier introuvable : {chemin}")

if __name__ == "__main__":
    main()
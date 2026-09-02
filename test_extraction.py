

"""
Script de test pour l'extraction des titres du sommaire.
Utilise la chaîne complète extraire_et_parser_sommaire.
"""

import os
from extraire_titre_sommaire import extraire_et_parser_sommaire

CHEMIN_PDF = r"C:\Users\1\Documents\MASTER_SOUTENANCE\corpus\SIMPORE Némata Version Finale.pdf"

def main():
    print(f"Test sur : {os.path.basename(CHEMIN_PDF)}")
    print("=" * 60)
    
    # Utiliser la chaîne complète (extraction → détection → filtrage → parsing)
    titres = extraire_et_parser_sommaire(CHEMIN_PDF, debug=False)
    
    print(f"\n{len(titres)} titres extraits du sommaire\n")
    
    for i, titre in enumerate(titres, 1):
        niveau_str = str(titre['niveau']) if titre['niveau'] > 0 else '?'
        numero_str = titre['numero'] if titre['numero'] else '(sans numéro)'
        legitime_str = ' [LÉGITIME]' if titre['legitime'] else ''
        print(f"{i:3d}. Niv {niveau_str} | {numero_str:20s} | {titre['texte'][:60]}{legitime_str}")

if __name__ == '__main__':
    main()
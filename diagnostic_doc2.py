import re
import pdfplumber
from pathlib import Path

# Le motif sommaire actuel
MOTIF_SOMMAIRE = re.compile(r'[._\s]{4,}\s*[\divxlcdm]+\s*$', re.IGNORECASE)

# Le motif titre actuel
MOTIF_TITRE_TDM = re.compile(
    r'\b(TABLE\s+DES\s+MATIÈRES|TABLE\s+DES\s+MATIERES|SOMMAIRE)\b',
    re.IGNORECASE
)

# Charger le document 02
chemin = Path("C:/Users/1/Documents/MASTER_SOUTENANCE/corpus/02-memoire.pdf")
# DOSSIER_CORPUS = r"C:\Users\1\Documents\MASTER_SOUTENANCE\corpus"


print(f"Analyse de {chemin.name}")
print("=" * 60)

with pdfplumber.open(chemin) as pdf:
    nb_pages = len(pdf.pages)
    print(f"Nombre de pages : {nb_pages}")
    print()
    
    # Compteurs
    total_lignes_sommaire = 0
    lignes_avec_titre = []
    premieres_lignes_sommaire = []
    
    for num_page, page in enumerate(pdf.pages):
        texte = page.extract_text()
        if not texte:
            continue
        lignes = texte.split('\n')
        
        for ligne in lignes:
            ligne_nettoyee = ligne.strip().strip('\xa0').strip()
            
            # Teste si la ligne matche le motif sommaire
            if MOTIF_SOMMAIRE.search(ligne_nettoyee):
                total_lignes_sommaire += 1
                if len(premieres_lignes_sommaire) < 10:
                    premieres_lignes_sommaire.append((num_page, ligne_nettoyee))
            
            # Teste si la ligne contient un mot-titre
            if MOTIF_TITRE_TDM.search(ligne_nettoyee):
                lignes_avec_titre.append((num_page, ligne_nettoyee))
    
    print(f"Nombre total de lignes qui matchent MOTIF_SOMMAIRE : {total_lignes_sommaire}")
    print()
    
    if premieres_lignes_sommaire:
        print("10 premières lignes qui matchent MOTIF_SOMMAIRE :")
        for num_page, ligne in premieres_lignes_sommaire:
            print(f"  Page {num_page} : {repr(ligne[:100])}")
    else:
        print("Aucune ligne ne matche MOTIF_SOMMAIRE dans tout le document.")
    
    print()
    print(f"Lignes contenant un mot-titre (SOMMAIRE/TABLE DES MATIERES) : {len(lignes_avec_titre)}")
    for num_page, ligne in lignes_avec_titre[:5]:
        print(f"  Page {num_page} : {repr(ligne[:100])}")
        
        

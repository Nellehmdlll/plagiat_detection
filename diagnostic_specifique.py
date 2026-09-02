import os
from inspection import extraire_texte, detecter_sommaire, detecter_lignes_repete
from extraire_titre_sommaire import extraire_titres_sommaire, filtrer_bandeau

DOSSIER = r"C:\Users\1\Documents\MASTER_SOUTENANCE\corpus"

def diagnostiquer(chemin_pdf):
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC : {os.path.basename(chemin_pdf)}")
    print('='*80)

    texte_complet = extraire_texte(chemin_pdf)

    # Détection sommaire
    sommaire_present, nb_entrees, bloc_sommaire = detecter_sommaire(texte_complet, debug=True)
    print(f"Sommaire présent : {sommaire_present}, lignes bloc élargi : {nb_entrees}")

    if not sommaire_present:
        print("Pas de sommaire détecté.")
        return

    # Afficher les 40 premières lignes du bloc
    print(f"\n--- 40 premières lignes du bloc sommaire (repr) ---")
    for i, ligne in enumerate(bloc_sommaire[:40], 1):
        print(f"Ligne {i:3d} : {repr(ligne[:150])}")

    # Détection bandeau
    bandeau_present, liste_bandeaux, _ = detecter_lignes_repete(texte_complet)
    print(f"\nBandeau présent : {bandeau_present}, nb fragments : {len(liste_bandeaux)}")
    if liste_bandeaux:
        for frag in liste_bandeaux[:5]:
            print(f"  Fragment : {frag[:80]}")

    # Filtrer bandeau
    bloc_filtre = filtrer_bandeau(bloc_sommaire, liste_bandeaux) if bandeau_present else bloc_sommaire
    print(f"\nBloc après filtrage : {len(bloc_filtre)} lignes")

    # Extraire titres et afficher les niveaux
    titres = extraire_titres_sommaire(bloc_filtre)
    print(f"\nTitres extraits : {len(titres)}")
    print("\n--- 50 premiers titres (numéro, niveau, texte) ---")
    for i, t in enumerate(titres[:50], 1):
        print(f"{i:3d}. Niv {t['niveau']} | {str(t['numero']):20s} | {t['texte'][:60]}")

def main():
    docs = ["05-memoire.pdf", "17-memoire.pdf", "10-memoire.pdf"]
    for doc in docs:
        chemin = os.path.join(DOSSIER, doc)
        if os.path.exists(chemin):
            diagnostiquer(chemin)
        else:
            print(f"Fichier introuvable : {chemin}")

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Regenere l'echantillon de validation de la categorie B (5 paires) avec la
consigne resserree (v2, voir PROMPT_CATEGORIE_B dans construire_jeu_test.py) :
la v1 ("reformule legerement... garde la structure tres proche") produisait
des variantes trop proches de la categorie A (synonymes isoles, structure
identique a 95%+), releve par l'utilisatrice le 2026-09-10 apres relecture.

Remplace, pour les 5 memes sources B deja choisies, la reformulation v1 par
une reformulation v2 : 30-50% des mots changes, au moins une proposition
reordonnee ou une phrase restructuree, tout en restant plus proche de
l'original qu'une paraphrase complete (categorie C, inchangee).
"""
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd

FICHIER_JEU_TEST = "jeu_test_etiquete.parquet"
METHODE = "modele_langage (Claude Sonnet 5, 2026-09-10, consigne B v2) - echantillon valide"

REFORMULATIONS_B_V2 = {
    "Rapport_Zonabo-Ouedraogo.pdf::section029::0": (
        "3.2.2 Apache Jena Fuseki\n"
        "La gestion directe des triples n'étant pas réalisable avec Protégé seul, nous avons complété "
        "cet outil par le serveur Apache Jena Fuseki, mieux adapté à cette tâche. Ce dernier constitue "
        "un serveur SPARQL polyvalent : il peut s'exécuter comme service de structure actif, comme "
        "application web Java packagée en fichier WAR, ou encore de manière autonome. Sa vocation "
        "première est de permettre le développement d'applications Web sémantiques exploitant des "
        "données liées [30]. C'est pour ses avantages en termes de facilité d'utilisation, et parce "
        "qu'il dépasse les limites rencontrées avec Protégé, que nous l'avons retenu pour l'exécution "
        "de nos requêtes SPARQL."
    ),
    "Rapport_Final_Goya (1).pdf::section010::1": (
        "Certains cookies subsistent indéfiniment sur votre disque dur, même si la réglementation "
        "impose normalement leur suppression au bout d'un certain délai. Leur rôle consiste souvent à "
        "conserver des éléments comme vos identifiants de connexion, vos coordonnées ou vos numéros de "
        "compte, évitant ainsi une nouvelle saisie à chaque visite d'un site[2]. On distingue deux "
        "grandes familles : d'une part, les cookies internes, créés directement par le site visité afin "
        "de mémoriser vos préférences et vos données de navigation[2] ; d'autre part, les cookies "
        "tiers, générés non pas par le site consulté mais par un service externe — par exemple, un "
        "bouton « J'aime » Facebook intégré à une page de TF1 déclenchera un cookie exploitable par "
        "Facebook. Ces cookies tiers servent en général à collecter des informations sur votre "
        "comportement et votre profil démographique, notamment à des fins publicitaires ciblées[2]."
    ),
    "rapport Moumoula.pdf::section007::1": (
        "Sur deux années académiques, le cursus de Technicien Supérieur accueille les bacheliers issus "
        "des séries techniques et scientifiques C, D, E, F2, F3 et H. Les détenteurs d'un BAC + 2 (BTS, "
        "DUT ou diplôme équivalent) en génie électrique peuvent quant à eux intégrer directement la "
        "formation d'ingénieurs de travaux, sur une année. Des cours du soir permettent en parallèle de "
        "suivre une formation continue diplômante. Quant au cycle d'Ingénieurs de Conception, il "
        "s'étend sur trois ans et reprend le modèle en vigueur dans les grandes écoles d'ingénieurs à "
        "l'échelle internationale. Plusieurs partenariats ont été noués par l'ISGE avec des "
        "établissements aux objectifs similaires, dans le but d'ajuster l'offre pédagogique aux "
        "attentes des entreprises : on peut citer l'Université Polytechnique de Bobo Dioulasso, "
        "l'ESMT-Dakar (Ecole Supérieure Multinationale des Télécommunications) au Sénégal, l'ENI de "
        "Bamako au Mali, ainsi que la Haute Ecole de Mont, en Belgique. Une phase théorique, centrée "
        "sur l'acquisition des connaissances, précède une phase pratique organisée autour d'un stage "
        "obligatoire pour valider le Diplôme d'Ingénieur de Conception (DIC) — c'est dans ce cadre que "
        "s'est déroulé notre stage de six mois, du 1er avril au 30 septembre 2022, au Centre "
        "d'Excellence Interdisciplinaire en Intelligence Artificielle pour le Développement (CITADEL)."
    ),
    "Rapport_SANOU_Bakary.pdf::section097::0": (
        "CONCLUSION GÉNÉRALE Les travaux menés autour de l'optimisation des systèmes de surveillance "
        "multi-caméras dédiés au suivi en temps réel d'objets d'intérêt aboutissent à des résultats en "
        "pleine adéquation avec les objectifs fixés par la DGTI. En combinant une revue approfondie de "
        "l'état de l'art, des essais rigoureux et l'intégration d'algorithmes avancés, notre démarche a "
        "permis des gains sensibles en précision, en rapidité et en efficacité du dispositif de "
        "surveillance. Grâce à une exploitation efficace des architectures matérielles disponibles, "
        "l'optimisation des ressources de calcul a porté ses fruits. De son côté, la combinaison "
        "d'algorithmes performants et d'une gestion intelligente des flux vidéo a permis au système de "
        "fonctionner de manière fiable en temps réel, sa robustesse ayant été confirmée face à des "
        "environnements complexes, à des éclairages variables et à des mouvements rapides. Un bon "
        "équilibre entre précision et vitesse d'exécution découle directement des choix technologiques "
        "opérés : les modèles d'apprentissage profond optimisés améliorent la reconnaissance des objets "
        "tout en limitant les fausses alertes, pendant que l'allocation dynamique des ressources rend "
        "le système plus facilement extensible à une surveillance de plus grande ampleur. Ces avancées "
        "n'excluent cependant pas des marges de progression : des modèles d'IA auto-adaptatifs, "
        "capables d'apprentissage continu, pourraient encore accroître l'efficacité du système, tout "
        "comme une meilleure gestion des flux vidéo grâce à des techniques de compression adaptative et "
        "de transmission intelligente améliorerait le confort d'utilisation. Combiner traitement local "
        "et calcul en périphérie (edge computing) constitue enfin une piste prometteuse pour un "
        "traitement à la fois plus rapide et plus sécurisé."
    ),
    "01-memoire.pdf::section011::0": (
        "1.2.2. Concept et approche d'appréciation de la charge de travail  Le concept de la charge de "
        "travail Phénomène complexe, la charge de travail se prête à des interprétations variables "
        "selon les acteurs concernés. Parmi les grilles de lecture disponibles, c'est l'approche "
        "ergonomique qui a retenu notre attention, en raison de son opérationnalité supérieure : en "
        "intégrant plusieurs dimensions de la charge, elle permet d'articuler les points de vue de "
        "l'organisation, du management et des salariés, sans pour autant exclure un enrichissement par "
        "d'autres modèles — on pense notamment à la psychodynamique du travail, à la théorie et à la "
        "clinique de l'activité, ou encore aux modèles de Karasek et Siegrist. Trois dimensions "
        "structurent la charge de travail selon le modèle ergonomique (voir figure 1) : la charge "
        "prescrite, la charge réelle et la charge vécue (Figure I, d'après les travaux de P. Falzon et "
        "C. Sauvagnac). La première, la charge prescrite, renvoie aux outils mobilisés pour exécuter le "
        "travail, aux performances attendues et aux procédures imposées ; elle constitue une contrainte "
        "dictée par l'organisation, incluant à la fois des objectifs quantitatifs (quantité, durée, "
        "ampleur) et qualitatifs (qualité, délai, satisfaction), ainsi que les procédures propres à "
        "chaque poste. Loin d'être figée, cette charge prescrite évolue avec le temps et se transforme "
        "au gré de la dynamique quotidienne du travail (Falzon, 2004), influençant ainsi l'ensemble des "
        "conditions dans lesquelles le travail est réalisé — elle ne se réduit donc pas à une simple "
        "description de la tâche ou du poste."
    ),
}

if __name__ == "__main__":
    df = pd.read_parquet(FICHIER_JEU_TEST)
    sources = pd.read_parquet("sources_bc_a_paraphraser.parquet").set_index("chunk_id")["texte"]

    nb_remplaces = 0
    for chunk_id, reformulation in REFORMULATIONS_B_V2.items():
        texte_source = sources.loc[chunk_id]
        masque = df["texte_1"] == texte_source
        if masque.sum() != 1:
            print(f"ATTENTION : {masque.sum()} lignes correspondantes pour {chunk_id!r} (attendu 1)")
            continue
        df.loc[masque, "texte_2"] = reformulation
        df.loc[masque, "methode_generation"] = METHODE
        nb_remplaces += 1

    print(f"{nb_remplaces}/5 paires B remplacees par la version v2.")
    df.to_parquet(FICHIER_JEU_TEST, index=False)
    print(f"Sauvegarde dans {FICHIER_JEU_TEST}")

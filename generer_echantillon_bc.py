# -*- coding: utf-8 -*-
"""
Remplit les 5 paires B + 5 paires C de l'echantillon de validation dans
jeu_test_etiquete.parquet, avec des reformulations produites par Claude
Sonnet 5 en suivant exactement les prompts documentes dans
rapport_generation_jeu_test.json. Script d'appoint, non destine a rester
dans le pipeline final (integre au besoin dans construire_jeu_test.py
une fois la qualite validee).
"""
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd

FICHIER_JEU_TEST = "jeu_test_etiquete.parquet"
METHODE = "modele_langage (Claude Sonnet 5, 2026-09-10) - echantillon valide"

REFORMULATIONS_B = {
    "Rapport_Zonabo-Ouedraogo.pdf::section029::0": (
        "3.2.2 Apache Jena Fuseki\n"
        "Afin de gérer plus efficacement les triplets, le serveur Apache Jena Fuseki est employé "
        "conjointement avec Protégé. Ce serveur permet la gestion directe des triplets, une tâche "
        "qu'il est impossible d'accomplir uniquement avec l'outil Protégé. Apache Jena Fuseki est un "
        "serveur SPARQL apte à fonctionner comme service de structure actif, comme application web "
        "Java (fichier WAR) et comme serveur autonome. Il est principalement employé pour la création "
        "d'applications Web sémantiques et de données liées [30]. Nous avons utilisé cet outil pour "
        "effectuer nos requêtes SPARQL, car il est facile d'utilisation et pallie les limites de "
        "Protégé."
    ),
    "Rapport_Final_Goya (1).pdf::section010::1": (
        "Bien que la loi impose que ces cookies soient effacés après un certain délai, certains "
        "cookies peuvent demeurer sur votre disque indéfiniment. Ces cookies peuvent conserver des "
        "données telles que vos informations de connexion, vos coordonnées et vos numéros de compte "
        "afin que vous n'ayez pas à les saisir de nouveau chaque fois que vous utilisez un site[2]. "
        "— Les cookies internes : Ces cookies sont générés par le site auquel vous avez accédé. Ils "
        "servent généralement à permettre au site de se rappeler de vos données et de vos "
        "préférences[2]. — Les cookies tiers : Les cookies tiers sont générés par un site autre que "
        "celui auquel vous avez accédé. Par exemple, le site de TF1 pourrait avoir un bouton « J'aime » "
        "de Facebook sur ses pages. Ce bouton « J'aime » déclencherait un cookie pouvant être lu par "
        "Facebook. L'objectif des cookies tiers est souvent de collecter certaines informations "
        "destinées à se renseigner sur votre comportement et vos données démographiques à des fins de "
        "marketing ciblé, etc[2]."
    ),
    "rapport Moumoula.pdf::section007::1": (
        "La formation de Technicien Supérieur s'adresse aux bacheliers des filières techniques et "
        "scientifiques C, D, E, F2, F3, H et se déroule sur deux années académiques. Peuvent accéder à "
        "la formation d'ingénieurs de travaux les titulaires d'un diplôme de niveau BAC + 2 (BTS, DUT "
        "et équivalent) dans le domaine du génie électrique pour une année académique. Parallèlement à "
        "la formation initiale, la formation continue diplômante est proposée en cours du soir. "
        "MOUMOULA Micheline Bénédicte x La formation d'Ingénieurs de Conception est un cycle de trois "
        "ans bâti sur le modèle des grandes écoles d'ingénieurs à l'international. Pour une formation "
        "pédagogique adaptée aux besoins des entreprises, l'ISGE a établi des partenariats avec des "
        "institutions poursuivant les mêmes objectifs ; il s'agit notamment de : • L'Université "
        "Polytechnique de Bobo Dioulasso • L'Ecole Supérieure Multinationale des Télécommunications de "
        "Dakar au Sénégal (ESMT-Dakar) • L'ENI (Ecole Nationale d'Ingénieurs) de Bamako au Mali • La "
        "Haute Ecole de Mont en Belgique. Le déroulement de la formation comporte une phase théorique "
        "et une phase pratique. La phase théorique porte sur le transfert des connaissances "
        "nécessaires et la phase pratique sur la réalisation d'un stage. Le stage constitue donc un "
        "passage obligatoire pour l'obtention du Diplôme d'Ingénieur de Conception (DIC). C'est dans "
        "cette optique que nous avons effectué un stage de six mois (01 avril au 30 septembre 2022) au "
        "sein du Centre d'Excellence Interdisciplinaire en Intelligence Artificielle pour le "
        "Développement (CITADEL). MOUMOULA Micheline Bénédicte xi"
    ),
    "Rapport_SANOU_Bakary.pdf::section097::0": (
        "CONCLUSION GÉNÉRALE Au terme de nos travaux sur l'optimisation des systèmes de surveillance "
        "multi-caméras pour le suivi en temps réel d'objets d'intérêt, nous pouvons affirmer que les "
        "résultats obtenus répondent pleinement aux objectifs spécifiques fixés par la DGTI. Notre "
        "démarche méthodique, associant une analyse approfondie de l'état de l'art, des "
        "expérimentations rigoureuses et l'intégration d'algorithmes avancés, a permis d'apporter des "
        "améliorations notables en matière de précision, de rapidité et d'efficacité du système de "
        "surveillance. L'optimisation des ressources de calcul a permis d'exploiter efficacement les "
        "architectures matérielles existantes. La mise en œuvre d'algorithmes performants, associée à "
        "une gestion intelligente des flux vidéo, a renforcé la capacité du système à fonctionner en "
        "temps réel avec une fiabilité élevée. Les résultats expérimentaux ont démontré sa robustesse "
        "face à la complexité des environnements, aux variations d'éclairage et aux mouvements "
        "rapides. Les choix technologiques retenus assurent un équilibre optimal entre précision et "
        "rapidité d'exécution. L'optimisation des modèles d'apprentissage profond a amélioré la "
        "reconnaissance des objets et diminué les fausses alertes, tandis que l'allocation dynamique "
        "des ressources a favorisé la scalabilité du système pour une surveillance à grande échelle. "
        "Toutefois, malgré ces progrès notables, certaines pistes d'amélioration méritent d'être "
        "explorées. L'intégration de modèles d'intelligence artificielle plus avancés, notamment des "
        "modèles auto-adaptatifs capables d'apprendre en continu à partir de nouvelles données, "
        "pourrait encore renforcer l'efficacité du système. De même, l'optimisation de la gestion des "
        "flux vidéo, en exploitant des techniques de compression adaptative et de transmission "
        "intelligente des données, permettrait d'améliorer l'expérience utilisateur. Enfin, l'adoption "
        "de solutions hybrides combinant traitement en local et calcul en périphérie (edge computing) "
        "pourrait offrir une alternative efficace pour garantir un traitement toujours plus rapide et "
        "sécurisé."
    ),
    "01-memoire.pdf::section011::0": (
        "1.2.2. Concept et approche d'appréciation de la charge de travail  Le concept de la charge de "
        "travail La charge de travail est un phénomène complexe pour lequel chaque acteur possède sa "
        "propre compréhension. Parmi les cadres de référence envisageables, l'approche ergonomique a "
        "plus particulièrement retenu notre attention en raison de sa plus forte opérationnalité. "
        "Celle-ci, par l'intégration de plusieurs dimensions de la charge, permet d'articuler des "
        "visions différentes selon que l'on se place du point de vue de l'organisation, du management "
        "ou des salariés. Elle n'exclut pas de s'appuyer sur d'autres modèles pour l'enrichir. Nous "
        "citerons, pour mémoire : la psychodynamique du travail, la théorie de l'activité, la clinique "
        "de l'activité, les modèles de Karasek et Siegrist. Le modèle ergonomique caractérise la "
        "charge de travail au travers de trois dimensions (voir figure 1) que sont : • la charge "
        "prescrite • la charge réelle • la charge vécue Figure I: Modèle de la charge de travail "
        "(Issus des travaux de P. Falzon et C. Sauvagnac) La charge prescrite désigne les outils pour "
        "effectuer le travail et ses réactions, les performances exigées et les procédures prescrites. "
        "Elle est considérée comme une contrainte et correspond à ce qui est demandé par "
        "l'organisation. Ces contraintes englobent les objectifs de performance à atteindre tant au "
        "plan quantitatif (ex.: quantité, durée, ampleur) que qualitatif (ex.: qualité, délai, "
        "satisfaction), mais également les diverses procédures prévues pour un poste donné. La charge "
        "de travail prescrite n'est donc pas déterminée définitivement et elle ne correspond pas à la "
        "description de la tâche ou du poste de travail, elle évolue dans le temps et se transforme "
        "avec la dynamique quotidienne du travail (Falzon, 2004) affectant l'ensemble des conditions "
        "de réalisation du travail."
    ),
}

REFORMULATIONS_C = {
    "Rapport_Lorenza- VFinale.pdf::section055::15": (
        "Par ailleurs, on observe un décalage temporel dans l'apparition des pics de puissance, ce qui "
        "tend à montrer que l'algorithme neuronal réagit plus rapidement lorsque l'irradiance ou la "
        "température changent brusquement.\n"
        "Sur le plan de l'analyse d'ensemble, plusieurs constats se dégagent. D'abord, en matière de "
        "justesse et de constance, la commande MPPT fondée sur un réseau de neurones parvient à "
        "maintenir le système plus proche du point de puissance maximale, avec des dépassements et des "
        "sous-régimes limités, ce qui traduit une puissance délivrée plus stable dans le temps. "
        "Ensuite, du point de vue de la réactivité, la méthode Perturb & Observe (P&O) a tendance à "
        "osciller de façon marquée, ce qui nuit à la précision du suivi du point optimal de puissance. "
        "Enfin, si l'on considère l'efficacité dans son ensemble, l'approche par réseau de neurones "
        "parvient à conserver une puissance moyenne et médiane supérieure, tout en produisant des pics "
        "plus élevés et une réponse globalement plus régulière — ce qui en fait, au final, l'option la "
        "plus performante."
    ),
    "21-memoire.pdf::section035::1": (
        "négatif et continue de se dégrader. De ce fait, le Burkina Faso ne réunit pas actuellement les "
        "conditions démographiques nécessaires pour bénéficier du dividende démographique évoqué par "
        "le PNDES-BF (2016). S'agissant du secteur éducatif, la scolarisation au primaire progresse : "
        "les effectifs sont passés de 2,2 millions d'élèves en 2010 à 2,7 millions en 2014, et le taux "
        "brut de scolarisation dans ce cycle a fortement augmenté, de 48,7 % en 2003 à 83 % en 2014 "
        "(PNUD, 2016). Ce progrès quantitatif masque cependant un niveau d'instruction qui demeure "
        "faible : 2,71 % des Burkinabè âgés de 6 ans et plus n'ont suivi aucune scolarité, et 67 % des "
        "hommes comme 80 % des femmes de 10 ans et plus sont analphabètes (Unicef, 2012). Cette "
        "situation limite fortement l'accès à une information adéquate sur la santé sexuelle et "
        "reproductive. Par ailleurs, les femmes burkinabè disposent rarement d'un pouvoir de décision "
        "sur leur propre vie sexuelle et reproductive, ce qui contribue à retarder le recours aux "
        "structures de santé lorsqu'un problème survient. Des pratiques préjudiciables comme "
        "l'excision (76 % des femmes concernées en 2010) et les violences conjugales (20 % des femmes "
        "ayant subi des violences physiques depuis l'âge de 15 ans en 2010) restent par ailleurs "
        "largement répandues (Unicef, 2012). Sur le plan de l'accès géographique, le développement de "
        "l'offre scolaire a permis à 87 enfants sur 100 de se trouver désormais à moins de 3 kilomètres "
        "d'une école. En revanche, la qualité de l'enseignement reste problématique : en dépit du "
        "recrutement continu d'enseignants du primaire depuis plus de dix ans, le ratio élève-maître "
        "n'a pratiquement pas évolué (51,3 entre 2003 et 2013), et si le taux d'achèvement du cycle "
        "primaire s'améliore peu à peu, une part importante des enfants n'arrive toujours pas à le "
        "terminer ; il atteignait 57,7 % en 2014."
    ),
    "16-memoire.pdf::section017::0": (
        "3.6. La rigueur scientifique Toute démarche qualitative doit se prémunir contre les biais en "
        "respectant certains critères de rigueur. Giorgi (1997) rattache la rigueur d'une recherche "
        "phénoménologique à deux exigences : la crédibilité et l'authenticité des résultats produits. "
        "S'agissant de la crédibilité, il s'agit de garantir que ce qui est décrit correspond bien à la "
        "réalité du phénomène observé. Plusieurs précautions ont été prises dans cette optique : "
        "varier le profil des participants, poursuivre les entretiens jusqu'à ce qu'aucune information "
        "ni thématique nouvelle n'émerge (saturation), et laisser place à une interprétation libre et "
        "imaginative des données recueillies. Nous avons également croisé nos sources — entretiens, "
        "observations et documents — par un travail de triangulation, et fait valider nos conclusions "
        "à la fois par les participants eux-mêmes (afin qu'ils reconnaissent leurs propos) et par des "
        "experts du sujet étudié. Quant à l'authenticité, elle renvoie à la fidélité des résultats par "
        "rapport au vécu réel des participants. Nous nous sommes efforcés d'appliquer la démarche de "
        "réduction phénoménologique décrite par Giorgi (1997), en procédant à un exercice de mise entre "
        "parenthèses (« bracketing ») : nos croyances, a priori, connaissances et expériences "
        "personnelles vis-à-vis du phénomène ont été explicitées avant les entretiens et l'analyse, "
        "afin d'en rester conscients tout au long du processus. Cette démarche nous a aidés à séparer "
        "notre propre regard de celui des participants et à restituer aussi fidèlement que possible la "
        "perception des infirmiers. Les verbatims ont par ailleurs été relus à plusieurs reprises pour "
        "en saisir précisément le sens. Cette exigence d'authenticité nous a conduits à adopter une "
        "posture réflexive tout au long de la recherche, dans la mesure où la subjectivité et la "
        "complexité des relations humaines imposent un questionnement constant sur notre propre "
        "positionnement."
    ),
    "Rapport de stage final_K Franck.pdf::section049::0": (
        "Annexes\n\nAnnexe 1 : Généralités sur les images\n\n"
        "Qu'est-ce qu'une image ? Une image constitue la représentation, sur un support numérique ou "
        "physique, d'une réalité, d'une scène observée ou même d'une idée abstraite. Lorsqu'elle est "
        "numérique, elle correspond à la projection plane d'une scène ou d'un objet, en deux ou trois "
        "dimensions, obtenue lorsque la lumière émise ou réfléchie par les éléments de la scène atteint "
        "un capteur tel qu'une caméra ou un scanner. Cette technique de représentation peut donner lieu "
        "à différents types de rendus. On distingue ainsi deux grandes catégories : les images dites "
        "« en niveaux de gris » (ou noir et blanc) et les images en couleur. Formellement, une image se "
        "définit comme une fonction de deux variables f(x, y), où (x, y) désigne les coordonnées d'un "
        "pixel — c'est-à-dire d'un point élémentaire de l'image. Cette fonction s'écrit : f : R2 ➔ "
        "[0,255] (x, y) ➔ f (x, y) Dans le cas d'images numériques, le couple de coordonnées (x, y) "
        "appartient à l'ensemble N2 des entiers naturels, et la valeur f(x, y) correspond à l'intensité "
        "lumineuse mesurée en ce pixel. Lorsque l'image est en couleur, l'information se répartit sur "
        "trois canaux — rouge (R), vert (G) et bleu (B) — et l'image peut alors être modélisée comme "
        "une fonction à valeurs vectorielles : f : R2 → [0, 255]3 (x, y) → f(x, y) = (R(x, y), G(x, y), "
        "B(x, y))"
    ),
    "05-memoire.pdf::section018::5": (
        "La possibilité de transposer les résultats de cette étude à d'autres contextes, comme c'est le "
        "cas pour toute recherche qualitative, suppose des milieux comparables (Olivier de Sardan, "
        "1995 ; Malterud, 2001). On pourrait ainsi envisager une telle transposition vers d'autres "
        "districts sanitaires du Burkina Faso appliquant le FBR, dans la mesure où ces zones partagent "
        "globalement des caractéristiques géographiques, socioprofessionnelles, sanitaires et "
        "économiques comparables (Ministère de la Santé, 2013). Les enseignements tirés de ce travail "
        "pourraient ainsi orienter une éventuelle généralisation de la stratégie à l'ensemble du "
        "territoire national, voire inspirer d'autres pays souhaitant mettre en place le FBR ou un "
        "dispositif comparable de fidélisation du personnel de santé en zone rurale.\n"
        "4.9 Difficultés rencontrées\n"
        "Cette recherche n'a pas été exempte d'obstacles. N'ayant reçu aucun financement extérieur, "
        "elle a reposé uniquement sur les ressources personnelles, limitées, de l'étudiant. Sur le "
        "terrain, malgré l'emploi du temps chargé des sages-femmes, celles-ci ont généralement accepté "
        "de lui consacrer un moment pour l'entretien, dans une ambiance chaleureuse — ce qui explique "
        "que certains entretiens aient eu lieu à des horaires inhabituels. Quelques difficultés "
        "ponctuelles sont néanmoins survenues (problèmes de santé d'une sage-femme, urgences "
        "obstétricales pendant un entretien, déplacements imprévus pour des motifs personnels), mais "
        "elles ont chaque fois été résolues sur le moment, ce qui n'a pas compromis le bon déroulement "
        "des travaux. Faute de logiciel dédié à l'analyse de données qualitatives, l'étudiant a par "
        "ailleurs procédé à une analyse manuelle des données."
    ),
}

if __name__ == "__main__":
    df = pd.read_parquet(FICHIER_JEU_TEST)
    sources = pd.read_parquet("sources_bc_a_paraphraser.parquet").set_index("chunk_id")["texte"]

    nb_remplis = 0
    for chunk_id, reformulation in {**REFORMULATIONS_B, **REFORMULATIONS_C}.items():
        texte_source = sources.loc[chunk_id]
        masque = (df["texte_1"] == texte_source) & (df["texte_2"].isna())
        if masque.sum() != 1:
            print(f"ATTENTION : {masque.sum()} lignes correspondantes pour {chunk_id!r} (attendu 1)")
            continue
        df.loc[masque, "texte_2"] = reformulation
        df.loc[masque, "methode_generation"] = METHODE
        nb_remplis += 1

    print(f"{nb_remplis}/10 paires de l'echantillon de validation remplies.")
    df.to_parquet(FICHIER_JEU_TEST, index=False)
    print(f"Sauvegarde dans {FICHIER_JEU_TEST}")
    print(f"Paires B/C encore en attente : {df['texte_2'].isna().sum()}")

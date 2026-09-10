# -*- coding: utf-8 -*-
"""
Calibrage du poids alpha de la fusion hybride :
    score_final = alpha * score_semantique + (1 - alpha) * score_lexical

Sur un mini-jeu de calibrage de 10 paires construites a la main sur des
passages reels (copie litterale, modification legere, paraphrase profonde,
sujets differents, reutilisation legitime), en l'absence encore du jeu de
test complet de la section 3.6 du memoire.

Score semantique : cosinus entre les encodages sentence-camembert-large
(max_seq_length=512, correctif deja valide) des deux textes de la paire.

Score lexical : cosinus TF-IDF, avec le VOCABULAIRE ET LES STATISTIQUES
D'IDF DU CORPUS DE REFERENCE (7443 chunks), pas un TF-IDF recalcule sur
les 10 paires seules - sinon les statistiques de rarete n'auraient aucun
sens (une paire de 20 textes ne peut pas definir ce qui est "rare"). Le
vectoriseur est refit ici avec exactement les memes parametres et le meme
pretraitement que construire_tfidf_corpus.py (memes textes source, memes
options) plutot que rechargé depuis un fichier pickle : cela reproduit
exactement le meme etat (vocabulaire, idf) de facon autonome et
verifiable, sans dependance a un artefact binaire supplementaire.

Ne code pas le pre-filtrage. S'arrete a la determination d'un alpha de
travail.

A executer avec l'environnement .venv-embed :
    .venv-embed\\Scripts\\python.exe calibrer_alpha_fusion.py
"""

import sys
import io

if __name__ == "__main__":
    # ne reconfigure stdout que lors d'une execution directe - sinon un
    # import de ce module (ex. pour reutiliser PAIRES) re-enveloppe un
    # stdout deja enveloppe et ferme le flux du process appelant.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.fr.stop_words import STOP_WORDS as MOTS_VIDES_FR

from construire_tfidf_corpus import pretraiter, FICHIER_METADONNEES

NOM_MODELE_SEMANTIQUE = "dangvantuan/sentence-camembert-large"
FENETRE_REELLE = 512

ALPHA_REFERENCE_MUTSADDI = 0.6  # Mutsaddi & Choudhary, 2025

# -----------------------------------------------------------------------------
# MINI-JEU DE CALIBRAGE (10 paires, texte fourni par l'utilisatrice)
# -----------------------------------------------------------------------------

PAIRES = [
    {
        "id": 1, "categorie": "A - copie litterale quasi identique", "label": "PLAGIAT",
        "texte1": (
            "Des progrès ont été réalisés dans les domaines variés afin de permettre aux populations "
            "d’être dans un état de complet bien- être physique, mental et social. On peut citer entre "
            "autres, la vaccination, le développement de moyens diagnostiques et thérapeutiques, "
            "l’utilisation des TIC dans la médecine etc.\n"
            "Pourtant, la santé dans le monde ne s'en porte pas mieux dans plusieurs pays de l'Afrique "
            "subsaharienne (OMS, 2000). Malgré des progrès enregistrés ces dernières années dans le "
            "domaine économique à la faveur de l’augmentation des coûts mondiaux des matières "
            "premières, l’Afrique subsaharienne cumule encore et toujours les indicateurs de "
            "développement parmi les plus faibles. Le rapport 2016 du Programme des Nations Unies pour "
            "le Développement (PNUD) classe 18 pays africains au dernier rang mondial selon l’indice de "
            "développement humain. L’extrême pauvreté en 2015 a été estimée à 35,2% de la population "
            "de l’Afrique subsaharienne (Banque Mondiale, 2015).\n"
            "Les systèmes de santé dans cette région continuent de faire face à des défis majeurs. Les "
            "pays d’Afrique subsaharienne sont ceux qui supportent la plus lourde charge de morbidité et "
            "de mortalité. Entre 1990 et 2011, l’Afrique a dû faire face à l’épidémie la plus dévastatrice de "
            "l’histoire de l’humanité, l’épidémie due au Virus de l’Immunodéficience Humaine (VIH) "
            "(OMS, 2015). Le syndrome d’immunodéficience acquise (SIDA) continue à dévaster "
            "l’Afrique qui compte 11 % de la population mondiale mais 60 % des personnes vivant avec le "
            "VIH. En outre , le paludisme continue de ravager les populations en Afrique subsaharienne ; "
            "plus de 90 % des 300 à 500 millions des cas de paludisme qui surviennent chaque année dans "
            "le monde selon les estimations touchent des africains, principalement des enfants de moins de "
            "5 ans (OMS, 2006). A cela s’ajoutent les maladies chroniques (diabète, cancer, maladies "
            "cardio-vasculaires) qui ne cessent de progresser."
        ),
        "texte2": (
            "Des progrès ont été réalisés dans les domaines variés afin de permettre aux populations "
            "d’être dans un état de complet bien- être physique, mental et social. On peut citer entre "
            "autres, la vaccination, le développement de moyens diagnostiques et thérapeutiques, "
            "l’utilisation des TIC dans la médecine etc.\n"
            "Pourtant, la santé dans le monde ne s'en porte pas mieux dans plusieurs pays de l'Afrique "
            "subsaharienne (OMS, 2000). Malgré des progrès enregistrés ces dernières années dans le "
            "domaine économique à la faveur de l’augmentation des coûts mondiaux des matières "
            "premières, l’Afrique subsaharienne cumule encore et toujours les indicateurs de "
            "développement parmi les plus faibles. Le rapport 2016 du Programme des Nations Unies pour "
            "le Développement (PNUD) classe 18 pays africains au dernier rang mondial selon l’indice de "
            "développement humain. L’extrême pauvreté en 2015 a été estimée à 35,2% de la population "
            "de l’Afrique subsaharienne (Banque Mondiale, 2015).\n"
            "Les systèmes de santé dans cette région continuent de faire face à des défis majeurs. Les "
            "pays d’Afrique subsaharienne sont ceux qui supportent la plus lourde charge de morbidité et "
            "de mortalité. Entre 1990 et 2011, l’Afrique a dû faire face à l’épidémie la plus dévastatrice de "
            "l’histoire de l’humanité, l’épidémie due au Virus de l’Immunodéficience Humaine (VIH) "
            "(OMS, 2015). Le syndrome d’immunodéficience acquise (SIDA) continue à dévaster "
            "l’Afrique qui compte 11 % de la population mondiale mais 60 % des personnes vivant avec le "
            "VIH. En outre , le paludisme continue de ravager les populations en Afrique subsaharienne ; "
            "plus de 90 % des 300 à 500 millions des cas de paludisme qui surviennent chaque année dans "
            "le monde selon les estimations touchent des africains, principalement des enfants de moins de "
            "5 ans (OMS, 2006). A cela s’ajoutent les maladies chroniques (diabète, cancer, maladies "
            "cardio-vasculaires) qui ne cessent de progresser."
        ),
    },
    {
        "id": 2, "categorie": "B - modification legere, vocabulaire majoritairement partage", "label": "PLAGIAT",
        "texte1": (
            "Les revues systématiques (Dambisya, 2007; Lehmann, Dieleman, & Martineau, 2008; Wilson et al., 2009; "
            "Dolea et al., 2010), les études qualitatives (Lori, Rominski, Gyakobo, et al., 2012; Huicho et al., 2015; "
            "Bertone, Lagarde, & Witter, 2016) et ou quantitatives (Rockers et al., 2012; Khim, 2016; Takemura, "
            "Kielmann, & Blaauw, 2016) de par le monde confirment que les incitations financières sont essentielles "
            "mais pas suffisantes à elles seules pour fidéliser les agents de santé dans les zones rurales ou "
            "éloignées. Elles gagnent en efficacité que lorsqu’elles sont réellement compensatoires et accompagnées "
            "par des incitations non financières telles l’amélioration des conditions de vie et de travail, la "
            "souplesse dans l’organisation du travail, l’attribution de logement d’astreinte et de moyen de "
            "locomotion, un système de reconnaissance et de valorisation tel le droit à l’avancement ou la "
            "promotion après l’exercice en zone rurale (Dolea et al., 2010).\n"
            "Il en est de même pour les autres stratégies de rétention telles que le service obligatoire en milieu "
            "rural des travailleurs (Frehywot, Mullan, Payne, & Ross, 2010; Mollahaliloğlu S., et al., 2012; Kouanda "
            "et al., 2014), le recrutement des étudiants issus d’un milieu rural et l’ouverture d’écoles de formation "
            "dans les régions rurales (Rabinowitz et al., 1999), le soutien personnel et professionnel (Codjia, "
            "Jabot, Dubois, WHO, et al., 2010; Zurn, Codjia, Sall, & Braichet, 2010) qui, implémentées de façon "
            "isolée ne peuvent être viables quelques soient les contextes car elles sont aussi fortement influencées "
            "par des facteurs d’autant plus complexes qu’ils sont liés à des caractéristiques personnelles, au "
            "système de santé et à tout l’environnement sociopolitique, socioéconomique et culturel. Ainsi ces "
            "facteurs interagissent et sont fortement liés à des motivations sous-jacentes, qu’elles soient "
            "économiques, sociales, culturelles ou religieuses (WHO, 2010)."
        ),
        "texte2": (
            "Les revues systématiques (Dambisya, 2007; Lehmann, Dieleman, & Martineau, 2008; Wilson et al., 2009; "
            "Dolea et al., 2010), les études qualitatives (Lori, Rominski, Gyakobo, et al., 2012; Huicho et al., 2015; "
            "Bertone, Lagarde, & Witter, 2016) et ou quantitatives (Rockers et al., 2012; Khim, 2016; Takemura, "
            "Kielmann, & Blaauw, 2016) de par le monde confirment que les incitations financières sont très "
            "importantes mais pas suffisantes à elles seules pour fidéliser les agents de santé dans les zones "
            "rurales. Elles obtiennent en efficacité que lorsqu’elles sont vraiment compensatoires et accompagnées "
            "par des incitations non financières telles l’amélioration des conditions de vie et de travail, la "
            "souplesse dans l’organisation du travail, l’attribution de logement d’astreinte et de moyen de "
            "locomotion, un système de reconnaissance et de valorisation tel le droit à l’avancement ou la "
            "promotion après l’exercice en zone rurale (Dolea et al., 2010).\n"
            "Il en est de même pour les autres stratégies de rétention telles que le service obligatoire en milieu "
            "rural des travailleurs (Frehywot, Mullan, Payne, & Ross, 2010; Mollahaliloğlu S., et al., 2012; Kouanda "
            "et al., 2014), le recrutement des étudiants issus d’un milieu rural et l’ouverture d’écoles de formation "
            "dans les régions rurales (Rabinowitz et al., 1999), le soutien personnel et professionnel (Codjia, "
            "Jabot, Dubois, WHO, et al., 2010; Zurn, Codjia, Sall, & Braichet, 2010) qui, implémentées de façon "
            "isolée ne peuvent être viables quelques soient les contextes car elles sont aussi fortement influencées "
            "par des facteurs d’autant plus complexes qu’ils sont liés à des caractéristiques personnelles, au "
            "système de santé et à tout l’environnement sociopolitique, socioéconomique et culturel. Ainsi ces "
            "facteurs interagissent et sont fortement liés à des motivations sous-jacentes, qu’elles soient "
            "économiques, sociales, culturelles ou religieuses (WHO, 2010)."
        ),
    },
    {
        "id": 3, "categorie": "C - paraphrase profonde, restructuration complete", "label": "PLAGIAT",
        "texte1": (
            "Pour mobiliser et motiver les équipes de soins en vue de la réalisation des activités de l’unité de "
            "soins, le SUS utilise diverses stratégies parmi lesquelles l’animation de groupe, la communication, "
            "l’écoute active des membres des équipes et la réceptivité à leurs attentes, la reconnaissance du "
            "travail bien fait, les encouragements et les félicitations des équipes pour un travail bien exécuté, "
            "et la création d’une bonne ambiance de travail. En outre, sa présence parmi les agents, sa "
            "participation aux évènements sociaux des membres des équipes et l’initiation d’activités récréatives "
            "en fin d’année participent à la cohésion du groupe. En attestent les propos du SUS7 (M14) :\n"
            "Bon, les astuces c’est, surtout l’animation du groupe, la communication, la présence aussi, "
            "c’est-à-dire il faut être permanemment dans le groupe, sur le terrain avec le personnel ! Les autres "
            "sources de motivation c’est, surtout les actions sociales. Quand y a un évènement malheureux ou bien "
            "heureux chez un agent, il faut tout de suite mobiliser les autres pour qu’on puisse en tout cas faire "
            "des visites à domicile, c’est très important ! Bien vrai tout le monde ne peut pas partir mais le SUS "
            "avec certains éléments font le déplacement, et au nom de l’unité, voilà ! Ça contribue en tout cas à "
            "motiver le personnel […]. (Entrevue SUS7 (M14) CHUP-CDG PAFADNAM [32977:33741]).\n"
            "Cela permet au SUS de créer une relation de confiance avec le personnel, de maintenir la cohésion du "
            "groupe et d’instaurer un climat de travail favorable à l’amélioration de la qualité des soins dans "
            "l’unité de soins. C’est aussi et en partie dans ce sens que s’orientent les propos du SUS3 (F11) :\n"
            "C’est, c’est peut-être l’animation des équipes, on essaie de les encourager, voilà, avec aussi les "
            "réunions et les trucs de fin d’année. Bon ! On essaie d’organiser des genres retrouvailles, en fin "
            "d’année, pour essayer d’encourager. En tout cas, ça permet de tisser les relations entre les agents. "
            "(Entrevue SUS3 (F11) CHUP-CDG PAFADNAM [36802:37136]).\n"
            "En somme, par l’exploitation de plusieurs éléments appartenant à leur leadership, les SUS mobilisent "
            "et motivent les équipes de soins en vue de la réalisation des activités dans les unités de soins. La "
            "mise en oeuvre de ces stratégies crée une relation de confiance entre SUS et personnel, ce qui favorise "
            "la cohésion des équipes de soins. Cela leur permet de contribuer à l’amélioration de la qualité des "
            "soins dans les unités des services cliniques."
        ),
        "texte2": (
            "Pour stimuler et motiver les équipes soignantes dans l’accomplissement de leurs tâches au sein de "
            "l’unité, le surveillant d’unité de soins (SUS) mobilise plusieurs leviers : l’animation des équipes, "
            "la communication, l’écoute attentive des agents et la prise en compte de leurs préoccupations, la "
            "valorisation du travail accompli, les encouragements et les félicitations adressées aux équipes après "
            "un travail bien fait, ainsi que l’instauration d’un climat de travail agréable. Sa proximité avec le "
            "personnel, sa participation aux moments importants de la vie sociale des membres de l’équipe et "
            "l’organisation d’activités de fin d’année contribuent également à renforcer l’unité du groupe. C’est "
            "ce que confirme le témoignage du SUS7 (M14) :\n"
            "Les principales stratégies consistent à animer l’équipe, à communiquer et surtout à être présent, "
            "c’est-à-dire à rester en permanence avec le personnel sur le terrain. La motivation passe aussi "
            "beaucoup par les actions sociales. Lorsqu’un événement, heureux ou malheureux, survient dans la vie "
            "d’un agent, il faut rapidement mobiliser les collègues pour organiser une visite à domicile ; c’est "
            "essentiel. Tout le monde ne peut pas se déplacer, mais le SUS et quelques membres de l’équipe le font "
            "au nom de l’unité, ce qui contribue fortement à motiver le personnel. (Entretien SUS7 (M14) CHUP-CDG "
            "PAFADNAM [32977:33741]).\n"
            "Ces pratiques permettent au SUS d’établir une relation de confiance avec les agents, de préserver la "
            "cohésion de l’équipe et d’instaurer un environnement de travail favorable à la qualité des soins dans "
            "l’unité. C’est également dans ce sens que vont les propos du SUS3 (F11) :\n"
            "Il s’agit surtout d’animer les équipes, de les encourager, notamment à travers les réunions et les "
            "activités de fin d’année. On essaie d’organiser des moments de retrouvailles en fin d’année pour "
            "encourager les agents ; cela permet de renforcer les liens entre eux. (Entretien SUS3 (F11) CHUP-CDG "
            "PAFADNAM [36802:37136]).\n"
            "En définitive, en s’appuyant sur différentes dimensions de leur leadership, les SUS parviennent à "
            "mobiliser et motiver les équipes soignantes dans la réalisation des activités des unités de soins. La "
            "mise en œuvre de ces stratégies instaure une relation de confiance entre le SUS et le personnel, "
            "renforce la cohésion des équipes et contribue ainsi à l’amélioration de la qualité des soins dans les "
            "services cliniques."
        ),
    },
    {
        "id": 4, "categorie": "D - sujets totalement differents", "label": "NON_PLAGIAT",
        "texte1": (
            "4.2 Population et échantillonnage/ échantillon\n"
            "4.2.1 Population d’étude\n"
            "La population d’étude est constituée des infirmiers/ères exerçant dans le domaine des soins au "
            "CHUP-CDG.\n"
            "4.2.2 Echantillonnage/échantillon\n"
            "L’échantillonnage a été fait par choix raisonné des infirmiers eu regard au type d’étude et la "
            "question de recherche, qui veulent que l’échantillon soit intentionnel et évolutif. Selon Gillis & "
            "Jackson (2002), ce type d'échantillonnage nécessite un engagement à entendre en entrevue des sujets "
            "qui possèdent l'expertise liée au phénomène étudié, afin de pouvoir obtenir le plus d'information sur "
            "le sujet à l'étude (Dias, 2009). Il s’est agi pour nous de mener des entrevues au fur et à mesure "
            "avec les sujets choisis parmi ceux qui ont un rapport avec le phénomène étudié, et ce jusqu’à ce "
            "qu’on ait constaté que les entrevues additionnelles n’apportaient plus d’information nouvelle à la "
            "compréhension du phénomène. Nous avons procédé à une diversification interne et externe de notre "
            "échantillon. Elle a été assurée de manière interne en tenant compte de l’occupation au sein du "
            "service, de l’expérience dans la pratique infirmière, de l’ancienneté dans le service et de "
            "l’inscription au tableau de l’ordre des infirmiers/ères du Burkina Faso, puis de manière externe en "
            "prenant en compte les infirmiers de deux sexes et les services d’exercice. Notre échantillon définitif "
            "a été constitué de dix infirmiers.\n"
            "Critères de sélection\n"
            "Les critères d’inclusion: A fait partie de notre échantillon, les infirmiers qui avaient une "
            "ancienneté d’au moins deux (2) ans dans le corps des infirmiers, et d’au moins un (1) an au CHUP-CDG. "
            "De même ils devraient être en contact directe avec les patients, être inscrits au tableau de l’ordre "
            "des infirmiers et accepter de prendre part à l’étude\n"
            "Les critères d’exclusion: Ont été exclues de l’échantillon les infirmiers/ères qui occupaient un "
            "poste ne permettant pas d’être en contact direct avec les patients parce que appelés à exercer des "
            "tâches différentes de celles réalisées par les infirmiers soignants. De même, que ceux qui n’avaient "
            "pas marqué leur adhésion de participer à l’étude."
        ),
        "texte2": (
            "Le paludisme continue de ravager les populations en Afrique subsaharienne ; plus de 90% des 300 à "
            "500 millions des cas de paludisme qui surviennent chaque année dans le monde selon les estimations "
            "touchent des africains, principalement des enfants de moins de 5 ans (OMS, 2006).\n"
            "La planification familiale se définit comme la possibilité pour les individus et les couples de "
            "choisir librement le nombre de leurs enfants, et le moment de les avoir (OMS, 2011). Les méthodes "
            "contraceptives incluent les méthodes traditionnelles, naturelles, hormonales et de barrières "
            "mécaniques et chimiques"
        ),
    },
    {
        "id": 5, "categorie": "A - copie litterale identique", "label": "PLAGIAT",
        "texte1": (
            "À la suite de l’étude et à la comparaison des processus de développement et aux fins de contrôler "
            "les risques et de mener à bon terme notre projet, nous avons opté pour le processus 2TUP pour les "
            "raisons suivantes :\n"
            "• d’une part, 2TUP donne une grande importance à la technologie, ce qui est important pour notre "
            "projet ;\n"
            "• d’autre part, 2TUP est un processus en Y qui contient une branche technique, une branche "
            "fonctionnelle et une branche réalisation. Les deux branches technique et fonctionnelle peuvent être "
            "exploitées en parallèle. De ce fait, si la technologie évolue ou, s’il arrive que lors du déroulement "
            "du projet, il y a modification d’un besoin technique, la branche technique peut être traitée puis "
            "réintégrée dans le projet facilement. De même, si une nouvelle fonctionnalité se présente, seule la "
            "branche fonctionnelle va être traitée sans toucher à l’autre branche.\n"
            "Le processus 2TUP comporte 2 branches :\n"
            "❖ une branche fonctionnelle qui se présente comme suit :\n"
            "• la capture des besoins fonctionnels, Cette étape vise à identifier les besoins spécifiques des "
            "utilisateurs en se concentrant sur leur métier ;\n"
            "• l’analyse qui consiste à étudier les spécifications fonctionnelles dans le but de connaitre le "
            "métier qui sera effectué par le système ;\n"
            "❖ une branche technique qui se présente comme suit :\n"
            "• la capture des besoins techniques qui consiste à énumérer les outils, matériels et technologies "
            "dont on aura besoin.\n"
            "• la conception générique qui consiste à indiquer les composants essentiels à l’élaboration de "
            "l’architecture technique.\n"
            "Ces deux branches se fusionnent pour donner la branche du milieu. Cette branche est constituée de :\n"
            "• la conception préliminaire qui doit intégrer le modèle d’analyse fonctionnelle dans l’architecture "
            "technique afin d’indiquer les composants du système à développer ;\n"
            "• la conception détaillée qui permet d’exécuter chaque composant du système ;\n"
            "• le codage qui consiste à implémenter les fonctionnalités du système avec des tests effectués au "
            "fur et à mesure ;"
        ),
        "texte2": (
            "À la suite de l’étude et à la comparaison des processus de développement et aux fins de contrôler "
            "les risques et de mener à bon terme notre projet, nous avons opté pour le processus 2TUP pour les "
            "raisons suivantes :\n"
            "• d’une part, 2TUP donne une grande importance à la technologie, ce qui est important pour notre "
            "projet ;\n"
            "• d’autre part, 2TUP est un processus en Y qui contient une branche technique, une branche "
            "fonctionnelle et une branche réalisation. Les deux branches technique et fonctionnelle peuvent être "
            "exploitées en parallèle. De ce fait, si la technologie évolue ou, s’il arrive que lors du déroulement "
            "du projet, il y a modification d’un besoin technique, la branche technique peut être traitée puis "
            "réintégrée dans le projet facilement. De même, si une nouvelle fonctionnalité se présente, seule la "
            "branche fonctionnelle va être traitée sans toucher à l’autre branche.\n"
            "Le processus 2TUP comporte 2 branches :\n"
            "❖ une branche fonctionnelle qui se présente comme suit :\n"
            "• la capture des besoins fonctionnels, Cette étape vise à identifier les besoins spécifiques des "
            "utilisateurs en se concentrant sur leur métier ;\n"
            "• l’analyse qui consiste à étudier les spécifications fonctionnelles dans le but de connaitre le "
            "métier qui sera effectué par le système ;\n"
            "❖ une branche technique qui se présente comme suit :\n"
            "• la capture des besoins techniques qui consiste à énumérer les outils, matériels et technologies "
            "dont on aura besoin.\n"
            "• la conception générique qui consiste à indiquer les composants essentiels à l’élaboration de "
            "l’architecture technique.\n"
            "Ces deux branches se fusionnent pour donner la branche du milieu. Cette branche est constituée de :\n"
            "• la conception préliminaire qui doit intégrer le modèle d’analyse fonctionnelle dans l’architecture "
            "technique afin d’indiquer les composants du système à développer ;\n"
            "• la conception détaillée qui permet d’exécuter chaque composant du système ;\n"
            "• le codage qui consiste à implémenter les fonctionnalités du système avec des tests effectués au "
            "fur et à mesure ;"
        ),
    },
    {
        "id": 6, "categorie": "B - modification tres legere (quasi-synonymes)", "label": "PLAGIAT",
        "texte1": (
            "1.2.3.1 Réseaux de neurone artificiel\n"
            "Les réseaux de neurones [21], également appelés réseaux de neurones artificiels ou Artificial "
            "Neurals Network (ANNs) sont un sous-ensemble de l’apprentissage automatique et sont au coeur des "
            "algorithmes d’apprentissage profond. Leur nom et leur structure sont inspirés du cerveau humain, "
            "imitant la façon dont les neurones biologiques communiquent entre eux. Chaque neurone effectue des "
            "opérations mathématiques sur ses entrées et produit une sortie. Les couches successives de neurones "
            "permettent de capturer des informations de plus en plus abstraites et complexes.\n"
            "Les ANNs sont composés de couches de noeuds incluant une entrée, une ou plusieurs cachées et une de "
            "sortie. Chaque noeud ou neurone artificiel se relie aux autres avec des poids et des seuils. Si la "
            "sortie d’un noeud dépasse son seuil, il s’active et envoie des données à la couche suivante. Les "
            "ANNs utilisent des données d’entraînement pour apprendre et gagner en précision. Une fois optimisés, "
            "ces algorithmes deviennent puissants en informatique et en intelligence artificielle, accomplissant "
            "rapidement des tâches comme la reconnaissance vocale ou d’images et aussi la traduction automatique "
            "bien plus efficacement que les méthodes traditionnelles figure 1.3.\n"
            "1.2.3.2 Encodeur\n"
            "Les modèles de traduction automatique n’utilisent pas de texte. Le texte doit être découpé en une "
            "suite de caractères, de mots ou de sous mots. Cette phase constitue la tokenisation et les éléments "
            "obtenus sont appelés des tokens. Chaque token est associé à un nombre qui représente son indice. Une "
            "phrase est donc représentée"
        ),
        "texte2": (
            "1.2.3.1 Réseaux de neurone artificiel\n"
            "Les réseaux de neurones [21], aussi appelés réseaux de neurones artificiels ou Artificial Neurals "
            "Network (ANNs) sont un sous-ensemble de l’apprentissage automatique et sont en plein milieu des "
            "algorithmes d’apprentissage profond. Leur nom et leur structure sont inspirés du cerveau humain, "
            "imitant la façon dont les neurones biologiques communiquent entre eux. Chaque neurone effectue des "
            "opérations mathématiques sur ses entrées et produit une sortie. Les couches successives de neurones "
            "permettent de capter des informations de plus en plus abstraites et complexes.\n"
            "Les ANNs sont composés de couches de noeuds incluant une entrée, une ou plusieurs cachées et une de "
            "sortie. Chaque noeud ou neurone artificiel se lie aux autres avec des poids et des seuils. Si la "
            "sortie d’un noeud dépasse son seuil, il s’active et envoie des données à la couche suivante. Les "
            "ANNs utilisent des données d’entraînement pour apprendre et gagner en précision. Une fois optimisés, "
            "ces algorithmes deviennent très puissants en informatique et en intelligence artificielle, faisant "
            "rapidement des tâches comme la reconnaissance vocale ou d’images et aussi la traduction automatique "
            "bien plus efficacement que les méthodes traditionnelles figure 1.3.\n"
            "1.2.3.2 Encodeur\n"
            "Les modèles de traduction automatique n’utilisent pas de texte. Le texte doit être découpé en une "
            "suite de caractères, de mots ou de sous mots. Cette phase constitue la tokenisation et les éléments "
            "obtenus sont appelés des tokens. Chaque token est associé à un nombre qui représente son indice. Une "
            "phrase est donc représentée"
        ),
    },
    {
        "id": 7, "categorie": "E - reutilisation legitime, contenu institutionnel reformule independamment",
        "label": "NON_PLAGIAT",
        "texte1": (
            "L'Institut Supérieur de Génie Électrique du Burkina Faso (ISGE-BF) a été fondé à l'initiative de "
            "dix-huit (18) entreprises du secteur de l'énergie et des télécommunications, réunies autour de la "
            "chambre de commerce et d'industrie du Burkina Faso (CCI-BF).\n"
            "Depuis 2003, cet institut a pour vocation la formation de cadres de haut niveau en génie électrique, "
            "répondant ainsi aux besoins des entreprises. L'ISGE-BF est situé à la ZAD II à Ouagadougou sur la "
            "circulaire du SIAO (Boulevard des Tengsoba n°2072).\n"
            "L’ISGE-BF est donc un établissement d'enseignement supérieur à caractère professionnel et "
            "scientifique qui figure parmi les écoles les plus dynamiques du Burkina. Il jouit d'une personnalité "
            "morale et d'une autonomie pédagogique et constitue un carrefour interdisciplinaire sans pareil. C'est "
            "également un institut de formation professionnalisant d'une richesse exceptionnelle où de grands "
            "secteurs d'avenirs sont représentés. Il y est délivré trois types de diplômes :\n"
            "Le Diplôme d'Ingénieur de Travaux (DIT) ;\n"
            "Le Diplôme d'Ingénieur de Conception (DIC).\n"
            "Trois options existent pour le Diplôme de Technicien Supérieur ; il s’agit de :\n"
            "L’Électricité Industrielle (EI) ;\n"
            "La Maintenance Industrielle (MI) ;\n"
            "Les Réseaux Informatique et Télécommunication (RIT).\n"
            "Trois options existent également pour le cycle d'ingénieur de Travaux :\n"
            "L’Ingénierie des Systèmes Électriques (ISE);\n"
            "Les Réseaux et Systèmes de Télécommunications (RST);\n"
            "La Maintenance des systèmes Industriels (MI).\n"
            "Quant au cycle d’Ingénieur de Conception, deux options existent :\n"
            "Le Génie des Systèmes Electriques (GSE);\n"
            "Le Génie des Systèmes Numériques (GSN).\n"
            "L'Institut constitue, grâce à son consortium d'entreprises membres fondateurs et son réseau d'écoles "
            "partenaires (Chambre de Commerce et d'Industrie du Burkina Faso ; Conseil National du Patronat"
        ),
        "texte2": (
            "L’Institut Supérieur de Génie Électrique du Burkina Faso (ISGE-BF) a vu le jour grâce à l’initiative "
            "de dix-huit (18) entreprises actives dans les domaines de l’énergie et des télécommunications, "
            "fédérées autour de la Chambre de Commerce et d’Industrie du Burkina Faso (CCI-BF). Depuis sa "
            "création en 2003, l’institut s’est donné pour mission de former des cadres supérieurs en génie "
            "électrique afin de répondre aux besoins exprimés par les entreprises du secteur. Il est implanté à "
            "Ouagadougou, dans la Zone d’Activités Diverses II (ZAD II), sur la circulaire du SIAO, au Boulevard "
            "des Tengsoba n°2072.\n"
            "L’ISGE-BF se présente comme un établissement d’enseignement supérieur à la fois scientifique et "
            "professionnel, considéré parmi les institutions les plus dynamiques du pays. Doté d’une personnalité "
            "juridique propre et d’une autonomie pédagogique, il constitue un pôle interdisciplinaire remarquable. "
            "C’est également un institut de formation professionnalisante qui couvre plusieurs secteurs porteurs. "
            "Trois types de diplômes y sont préparés :\n"
            "le Diplôme de Technicien Supérieur (DTS) ;\n"
            "le Diplôme d’Ingénieur de Travaux (DIT) ;\n"
            "le Diplôme d’Ingénieur de Conception (DIC).\n"
            "Pour le Diplôme de Technicien Supérieur, l’offre de formation s’articule autour de trois spécialités "
            ":\n"
            "l’Électricité Industrielle (EI) ;\n"
            "la Maintenance Industrielle (MI) ;\n"
            "les Réseaux Informatiques et Télécommunications (RIT).\n"
            "Le cycle d’Ingénieur de Travaux propose également trois options :\n"
            "l’Ingénierie des Systèmes Électriques (ISE) ;\n"
            "les Réseaux et Systèmes de Télécommunications (RST) ;\n"
            "la Maintenance des Systèmes Industriels (MI).\n"
            "Quant au cycle d’Ingénieur de Conception, deux parcours sont offerts :\n"
            "le Génie des Systèmes Électriques (GSE) ;\n"
            "le Génie des Systèmes Numériques (GSN).\n"
            "Grâce au regroupement de ses entreprises fondatrices et à son réseau d’écoles partenaires — parmi "
            "lesquelles la Chambre de Commerce et d’Industrie du Burkina Faso et le Conseil National du Patronat "
            "— l’institut dispose d’un ancrage professionnel solide qui favorise l’insertion de ses diplômés."
        ),
    },
    {
        "id": 8, "categorie": "E - reutilisation legitime, meme bibliographie reformulee/traduite independamment",
        "label": "NON_PLAGIAT",
        "texte1": (
            "Bibliographies\n"
            "[1] Caldwell-Harris, C. L., Lancaster, A., Ladd, D. R., Dediu, D., & Christiansen, M. H. (2015). "
            "Factors influencing sensitivity to lexical tone in an artificial language: Implications for second "
            "language learning. Studies in Second Language Acquisition, 37(2), 335-357.\n"
            "[2] Compaoré, L. (2016). Acoustic cues signaling prosodic units in Moore: a comparison of journalist "
            "and non-journalist realizations. Speech Prosody 2016, 659-662.\n"
            "[3] Pengwendé ZONGO (2022). Construction d’un système tutoriel intelligent pour l’apprentissage de "
            "la langue mooré [4] Abdelrahman, G., Wang, Q., & Nunes, B. P. (2022). Knowledge Tracing: A Survey. "
            "arXiv preprint arXiv:2201.06953\n"
            "[5] S´ebastien Gavois. IA : des nano-neurones pour « repenser l’architecture interne de "
            "l’électronique ». 25 sept. 2017. Url : "
            "https://www.nextinpact.com/article/27283/105231-ia-nano-neurones-pour-repenser-larchitecture-interne-"
            "electronique (visité le 10/12/2022).\n"
            "[6] Saad Albawi, Tareq Abed Mohammed et Saad Al-Zawi. “Understanding of a convolutional neural "
            "network”. In: (2017), p. 1-6.\n"
            "[7] Laith Alzubaidi et al. “Review of deep learning: concepts, CNN architectures, challenges, "
            "applications, future directions”. In: Journal of Big Data 8.1 (31 mars 2021), p. 53. ISSN: "
            "2196-1115. DOI: 10.1186/s40537-021-00444-8.\n"
            "[8] Convolutional Networks. Url : https://www.deeplearningbook.org/contents/convnets.html (visité "
            "le 13/12/2022).\n"
            "[9] What are Convolutional Neural Networks? 6 Jan. 2021. URL: "
            "https://www.ibm.com/cloud/learn/convolutional-neural-networks (visité le 14/12/2022).\n"
            "[10] Keiron O’Shea et Ryan Nash. “An introduction to convolutional neural networks”. In: arXiv "
            "preprint arXiv :1511.08458 (2015).\n"
            "[11] WINSTON, P.H., Artificial Intelligence, Addison Winsley Publishing Company, 1992. 737 p.\n"
            "[12] WIPO, ‘’Technology Trends: Artificial Intelligence’’, 2019. 154 p."
        ),
        "texte2": (
            "Références bibliographiques\n"
            "[1] Caldwell-Harris, C. L., Lancaster, A., Ladd, D. R., Dediu, D., & Christiansen, M. H. (2015). "
            "Sensitivity to lexical tone in an artificial language: influencing factors and implications for "
            "second language acquisition. Studies in Second Language Acquisition, 37(2), 335-357.\n"
            "[2] Compaoré, L. (2016). Prosodic unit signaling in Moore through acoustic cues: a comparison "
            "between journalist and non-journalist speech. Actes de Speech Prosody 2016, 659-662.\n"
            "[3] Zongo, P. (2022). Conception d’un système tutoriel intelligent dédié à l’apprentissage de la "
            "langue mooré.\n"
            "[4] Abdelrahman, G., Wang, Q., & Nunes, B. P. (2022). A survey on knowledge tracing. arXiv preprint, "
            "arXiv:2201.06953.\n"
            "[5] Gavois, S. (2017, 25 septembre). Intelligence artificielle : des nano-neurones pour repenser "
            "l’architecture interne des composants électroniques. Consulté le 10 décembre 2022 sur "
            "https://www.nextinpact.com/article/27283/105231-ia-nano-neurones-pour-repenser-larchitecture-interne-"
            "electronique.\n"
            "[6] Albawi, S., Abed Mohammed, T., & Al-Zawi, S. (2017). Comprendre les réseaux de neurones "
            "convolutifs. p. 1-6.\n"
            "[7] Alzubaidi, L., et al. (2021, 31 mars). Deep learning : concepts, architectures CNN, défis, "
            "applications et perspectives. Journal of Big Data, 8(1), 53. DOI : 10.1186/s40537-021-00444-8.\n"
            "[8] Réseaux convolutifs. Consulté le 13 décembre 2022 sur "
            "https://www.deeplearningbook.org/contents/convnets.html.\n"
            "[9] Que sont les réseaux de neurones convolutifs ? (2021, 6 janvier). Consulté le 14 décembre 2022 "
            "sur https://www.ibm.com/cloud/learn/convolutional-neural-networks.\n"
            "[10] O’Shea, K., & Nash, R. (2015). Introduction aux réseaux de neurones convolutifs. arXiv "
            "preprint, arXiv:1511.08458.\n"
            "[11] Winston, P. H. (1992). Artificial Intelligence. Addison-Wesley Publishing Company, 737 p.\n"
            "[12] Organisation Mondiale de la Propriété Intellectuelle (OMPI). (2019). Tendances technologiques : "
            "l’intelligence artificielle. 154 p."
        ),
    },
    {
        "id": 9, "categorie": "C - paraphrase profonde sur definitions academiques", "label": "PLAGIAT",
        "texte1": (
            "2.1.4 Planification familiale\n"
            "La planification familiale se définit comme « la possibilité pour les individus et les couples de "
            "choisir librement le nombre de leurs enfants, et le moment de les avoir » (OMS, 2011, p. 5). Pour le "
            "Ministère de la santé (2010), la planification familiale se réfère à l’ensemble des moyens "
            "techniques, psychologiques, éducatifs qui permettent aux couples et aux individus d’atteindre "
            "certains buts en fonction de leurs possibilités et de leurs désirs, d’éviter les grossesses non "
            "désirées, de favoriser les naissances désirées, d’espacer les naissances et de déterminer le nombre "
            "d’enfants qu’ils souhaitent. Les méthodes contraceptives sont :\n"
            "(i) les méthodes traditionnelles (séparation du couple, abstinence totale, etc.) ;\n"
            "(ii) les méthodes naturelles (calendrier, température, glaire cervicale, MAMA exclusif, collier, "
            "etc.) ;\n"
            "(iii) les méthodes hormonales : pilules (comprimés oraux progestatifs ; comprimés oraux combinés), "
            "les injectables, les implants\n"
            "(iv) les méthodes de barrières mécaniques et chimiques (condom masculin, condom féminin, "
            "spermicides ; DIU ; vasectomie, ligature des trompes).\n"
            "Somme toute, cette définition souligne trois aspects qui se prêtent à notre compréhension, à savoir "
            "la liberté de choix du moment d’avoir une grossesse, l'espacement et la limitation des naissances.\n"
            "2.1.5 Pratique de la contraception moderne\n"
            "L’OMS (2017) définit la pratique de la contraception moderne comme étant l’utilisation d’un ensemble "
            "de procédés (chimiques et mécaniques pour la plupart) à caractère permanent ou non, destinés à "
            "prévenir la grossesse ou l’empêcher, de façon réversible, temporaire ou définitive. Ces méthodes "
            "modernes de contraception correspondent aux méthodes hormonales : pilule progestative, pilule "
            "combinée, les injectables, les implants sous cutanés et méthodes de barrières mécaniques et "
            "chimiques : condom féminin, spermicides ; DIU ; ligature des trompes (Ministère de la Santé, 2017). "
            "De ce qui précède, la pratique de la contraception moderne des femmes correspond dans notre étude à "
            "l’utilisation d’une des méthodes modernes de contraception par une femme pendant la période de la "
            "collecte des données.\n"
            "2.1.6 Dividende Démographique : Concept, enjeux et défis\n"
            "Le dividende démographique est l’accélération de la croissance économique qui peut résulter d’une "
            "évolution de la structure par âge de la population, accompagnée d’investissements stratégiques dans "
            "la santé, l’éducation, la politique économique et la gouvernance. Il est un booste que le changement "
            "dans la structure d’âge d’une population peut apporter au développement économique (OMS, 2015)."
        ),
        "texte2": (
            "2.1.4 Planification familiale\n"
            "D’après l’OMS (2011, p. 5), la planification familiale renvoie à la capacité des personnes et des "
            "couples de décider librement du nombre d’enfants qu’ils souhaitent avoir et du moment où ils les "
            "auront. Pour le Ministère de la Santé (2010), elle rassemble l’ensemble des dispositifs techniques, "
            "psychologiques et éducatifs qui aident les couples et les individus à atteindre leurs objectifs en "
            "matière de procréation selon leurs désirs et leurs possibilités, à prévenir les grossesses non "
            "souhaitées, à favoriser les naissances voulues, à espacer les naissances et à choisir la taille de "
            "leur famille. On distingue plusieurs catégories de méthodes contraceptives :\n"
            "les méthodes dites traditionnelles, comme la séparation des conjoints ou l’abstinence totale ;\n"
            "les méthodes naturelles, fondées par exemple sur le calendrier, la température, l’observation de la "
            "glaire cervicale, l’allaitement maternel exclusif (MAMA) ou le collier ;\n"
            "les méthodes hormonales, notamment les comprimés oraux (progestatifs ou combinés), les "
            "contraceptifs injectables et les implants ;\n"
            "les méthodes de barrière mécanique ou chimique, telles que le préservatif masculin ou féminin, les "
            "spermicides, le dispositif intra-utérin (DIU), ainsi que la vasectomie et la ligature des trompes.\n"
            "En résumé, cette définition met en avant trois dimensions essentielles : la liberté de choisir le "
            "moment d’une grossesse, l’espacement des naissances et la limitation du nombre d’enfants.\n"
            "2.1.5 Pratique de la contraception moderne\n"
            "Selon l’OMS (2017), la pratique de la contraception moderne correspond à l’emploi de différents "
            "procédés, principalement chimiques ou mécaniques, permanents ou non, qui visent à éviter ou prévenir "
            "une grossesse de manière réversible, temporaire ou définitive. Les méthodes modernes incluent, "
            "d’après le Ministère de la Santé (2017), les contraceptifs hormonaux — pilules progestatives ou "
            "combinées, injectables, implants sous-cutanés — ainsi que les méthodes de barrière mécanique et "
            "chimique : préservatif féminin, spermicides, dispositif intra-utérin et ligature des trompes. Dans "
            "le cadre de cette étude, la pratique contraceptive moderne chez les femmes désigne le recours à "
            "l’une de ces méthodes pendant la période de collecte des données.\n"
            "2.1.6 Dividende démographique : concept, enjeux et défis\n"
            "Le dividende démographique correspond à l’accélération de la croissance économique rendue possible "
            "par une modification de la structure par âge d’une population, à condition qu’elle s’accompagne "
            "d’investissements stratégiques dans la santé, l’éducation, les politiques économiques et la "
            "gouvernance. Autrement dit, il s’agit d’un effet d’entraînement que le changement démographique peut "
            "exercer sur le développement économique (OMS, 2015)."
        ),
    },
    {
        "id": 10, "categorie": "D - sujets differents, meme registre socio-economique burkinabe",
        "label": "NON_PLAGIAT",
        "texte1": (
            "Créée le 27 mars 2019, YOUNOUSS NOUR SARL est une entreprise burkinabè spécialisée dans la "
            "distribution en gros de boissons sans alcool et ayant son siège social à Ouagadougou plus précisement "
            "à Projet ZACA. Son fondateur, Monsieur KABORE INOUSSA, s’est d’abord lancé dans le commerce de "
            "détail avant de développer une activité de grossiste, animée par la volonté d’assurer une "
            "distribution rapide, fiable et de qualité sur le marché national.\n"
            "L’entreprise, constituée sous forme de Société à Responsabilité Limitée (SARL), est aujourd’hui l’un "
            "des principaux grossistes de la capitale Ouagadougou dans le domaine des boissons non alcoolisées. "
            "Elle distribue plus de 100 produits, regroupant des eaux minérales, des jus de fruits, et des "
            "boissons énergisantes. Ses fournisseurs incluent plusieurs acteurs majeurs du secteur tels que : "
            "DAFANI SA, Twellium Industry, General Industry, Namaphone, SODIBO, Coca-Cola Burkina Bottling "
            "Company et la Société Étoile.\n"
            "YOUNOUSS NOUR SARL agit comme intermédiaire entre les usines de production et les dépôts "
            "secondaires, qui s’approvisionnent auprès d’elle pour revendre aux clients finaux (alimentation, "
            "boutiques, dépôts de boissons, restaurants, hôtels et bars).\n"
            "Grâce à son positionnement stratégique dans la chaîne de distribution, l’entreprise contribue de "
            "manière significative à l’approvisionnement du marché local en boissons de qualité. Elle combine "
            "ainsi un savoir-faire commercial fondé sur la proximité avec les clients et une gestion logistique "
            "pragmatique adaptée aux réalités du terrain."
        ),
        "texte2": (
            "1.2 Question de recherche\n"
            "Dans quelles mesures les pratiques d’enseignement à l’INFAS antenne de Korhogo tiennent-elles "
            "comptes de l’approche par les compétences ?\n"
            "1.3 Questions secondaires\n"
            "Quels sont les méthodes, techniques et procédés d’apprentissage que les enseignants utilisent dans "
            "leur pratiques d’enseignement à l’INFAS antenne de Korhogo?\n"
            "Quels domaines de connaissance et niveaux taxonomiques sont considérés dans l’évaluation des "
            "apprentissages des étudiants ?\n"
            "1.4 Objectifs de l’étude\n"
            "1.4.1 Objectif General\n"
            "Analyser les pratiques d’enseignement des enseignants de l’INFAS antenne de Korhogo en relation "
            "avec les exigences de l’Approche par les Compétences\n"
            "1.4.2- Objectifs spécifiques\n"
            "Décrire les méthodes, techniques et procédés utilisés par les enseignants en fonction des principes "
            "de l’Approche par les Compétences ;\n"
            "Identifier les domaines de connaissance et niveaux taxonomiques considérés dans l’évaluation des "
            "étudiants"
        ),
    },
]

GROUPE_PLAGIAT = [1, 2, 3, 5, 6, 9]
GROUPE_NON_PLAGIAT = [4, 7, 8, 10]
VALEURS_ALPHA = [round(a, 1) for a in np.arange(0.0, 1.01, 0.1)]


def cosinus(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def cosinus_sparse(v1, v2):
    num = v1.multiply(v2).sum()
    den = np.sqrt(v1.multiply(v1).sum()) * np.sqrt(v2.multiply(v2).sum())
    return float(num / den) if den > 0 else 0.0


if __name__ == "__main__":
    print(f"Chargement de {FICHIER_METADONNEES} pour refitter le TF-IDF du corpus...")
    df_meta = pd.read_parquet(FICHIER_METADONNEES)

    vectoriseur = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.85, stop_words=sorted(MOTS_VIDES_FR), lowercase=True
    )
    vectoriseur.fit([pretraiter(t) for t in df_meta["texte"]])
    print(f"TF-IDF du corpus refit : {len(vectoriseur.get_feature_names_out())} termes "
          f"(identique a construire_tfidf_corpus.py, meme corpus/parametres).")

    print(f"\nChargement de {NOM_MODELE_SEMANTIQUE}...")
    modele = SentenceTransformer(NOM_MODELE_SEMANTIQUE, device="cuda")
    modele.max_seq_length = FENETRE_REELLE

    print(f"\n{'=' * 100}\nCALCUL DES SCORES SEMANTIQUE ET LEXICAL SUR LES 10 PAIRES\n{'=' * 100}")
    for p in PAIRES:
        vec_sem = modele.encode([p["texte1"], p["texte2"]], convert_to_numpy=True, show_progress_bar=False)
        p["score_semantique"] = cosinus(vec_sem[0], vec_sem[1])

        vec_lex = vectoriseur.transform([pretraiter(p["texte1"]), pretraiter(p["texte2"])])
        p["score_lexical"] = cosinus_sparse(vec_lex.getrow(0), vec_lex.getrow(1))

        print(f"  Paire {p['id']:2d} [{p['label']:12s}] {p['categorie']:65s} "
              f"sem={p['score_semantique']:.4f}  lex={p['score_lexical']:.4f}")

    # -------------------------------------------------------------------
    # Balayage d'alpha
    # -------------------------------------------------------------------
    print(f"\n{'=' * 100}\nTABLEAU PAIRE x ALPHA (score fusionne = alpha*semantique + (1-alpha)*lexical)\n{'=' * 100}")
    entete = "Paire".ljust(6) + "Label".ljust(13) + "".join(f"a={a:.1f}".rjust(8) for a in VALEURS_ALPHA)
    print(entete)
    scores_par_alpha = {a: {} for a in VALEURS_ALPHA}
    for p in PAIRES:
        ligne = f"{p['id']:<6}{p['label']:<13}"
        for a in VALEURS_ALPHA:
            score = a * p["score_semantique"] + (1 - a) * p["score_lexical"]
            scores_par_alpha[a][p["id"]] = score
            ligne += f"{score:.4f}".rjust(8)
        print(ligne)

    # -------------------------------------------------------------------
    # Separation plagiat / non-plagiat par valeur d'alpha
    # -------------------------------------------------------------------
    print(f"\n{'=' * 100}\nSEPARATION PLAGIAT / NON-PLAGIAT PAR VALEUR D'ALPHA\n{'=' * 100}")
    print(f"{'alpha':>6} {'moy. plagiat':>14} {'moy. non-plagiat':>18} {'ecart (moy)':>12} "
          f"{'min plagiat':>12} {'max non-plag.':>14} {'marge':>10} {'chevauchement':>15}")

    resultats_alpha = []
    for a in VALEURS_ALPHA:
        scores_plagiat = [scores_par_alpha[a][i] for i in GROUPE_PLAGIAT]
        scores_non_plagiat = [scores_par_alpha[a][i] for i in GROUPE_NON_PLAGIAT]
        moy_plagiat = np.mean(scores_plagiat)
        moy_non_plagiat = np.mean(scores_non_plagiat)
        ecart = moy_plagiat - moy_non_plagiat
        min_plagiat = min(scores_plagiat)
        max_non_plagiat = max(scores_non_plagiat)
        marge = min_plagiat - max_non_plagiat
        chevauchement = "OUI" if marge < 0 else "non"
        resultats_alpha.append({
            "alpha": a, "moy_plagiat": moy_plagiat, "moy_non_plagiat": moy_non_plagiat,
            "ecart": ecart, "min_plagiat": min_plagiat, "max_non_plagiat": max_non_plagiat,
            "marge": marge,
        })
        print(f"{a:>6.1f} {moy_plagiat:>14.4f} {moy_non_plagiat:>18.4f} {ecart:>12.4f} "
              f"{min_plagiat:>12.4f} {max_non_plagiat:>14.4f} {marge:>10.4f} {chevauchement:>15}")

    meilleur = max(resultats_alpha, key=lambda r: r["ecart"])
    alpha_retenu = meilleur["alpha"]

    print(f"\n{'=' * 100}\nALPHA RETENU\n{'=' * 100}")
    print(f"alpha = {alpha_retenu:.1f} maximise l'ecart de moyennes "
          f"(ecart = {meilleur['ecart']:.4f}, marge min/max = {meilleur['marge']:.4f})")

    ref = next(r for r in resultats_alpha if r["alpha"] == ALPHA_REFERENCE_MUTSADDI)
    print(f"\nComparaison a la reference Mutsaddi & Choudhary (2025), alpha=0.6 :")
    print(f"  a={ALPHA_REFERENCE_MUTSADDI:.1f} (reference) : ecart={ref['ecart']:.4f}, marge={ref['marge']:.4f}")
    print(f"  a={alpha_retenu:.1f} (retenu ici)    : ecart={meilleur['ecart']:.4f}, marge={meilleur['marge']:.4f}")
    if alpha_retenu == ALPHA_REFERENCE_MUTSADDI:
        print("  -> Le contexte burkinabe CONFIRME la valeur de reference (meme alpha optimal).")
    else:
        print(f"  -> Le contexte burkinabe DIFFERE de la reference "
              f"(optimal local a={alpha_retenu:.1f} vs reference a={ALPHA_REFERENCE_MUTSADDI:.1f}), "
              f"a discuter dans le memoire plutot qu'a masquer.")

    print(f"\n{'=' * 100}\nPAIRES 7 et 8 (categorie E, reutilisation legitime) A L'ALPHA RETENU (a={alpha_retenu:.1f})\n{'=' * 100}")
    scores_plagiat_retenu = [scores_par_alpha[alpha_retenu][i] for i in GROUPE_PLAGIAT]
    min_plagiat_retenu = min(scores_plagiat_retenu)
    for i in (7, 8):
        p = next(p for p in PAIRES if p["id"] == i)
        score = scores_par_alpha[alpha_retenu][i]
        distinguable = "oui, reste sous le plagiat le plus bas" if score < min_plagiat_retenu else \
            "NON, chevauche ou depasse le plagiat le plus bas"
        print(f"  Paire {i} ({p['categorie']}) : score = {score:.4f} "
              f"(min groupe plagiat = {min_plagiat_retenu:.4f}) -> {distinguable}")
    autres_non_plagiat = [i for i in GROUPE_NON_PLAGIAT if i not in (7, 8)]
    scores_autres = [scores_par_alpha[alpha_retenu][i] for i in autres_non_plagiat]
    print(f"  (pour reference, autres non-plagiat {autres_non_plagiat} : "
          f"{[round(s, 4) for s in scores_autres]})")

    print(f"\n{'=' * 100}\nLIMITES DE CE CALIBRAGE\n{'=' * 100}")
    print(
        "- 10 paires construites/choisies a la main, pas un jeu de test statistique : la valeur\n"
        "  d'alpha retenue est indicative, sensible au choix precis de ces exemples (un autre\n"
        "  jeu de 10 paires pourrait deplacer legerement l'optimum).\n"
        "- Le mini-jeu ne couvre pas toutes les longueurs/styles de chunks reels (les chunks du\n"
        "  pipeline font ~400 tokens max, ces paires sont plus variees en taille).\n"
        "- A confirmer/ajuster avec le jeu de test complet de la section 3.6 avant de figer alpha\n"
        "  definitivement pour le pre-filtrage et la fusion en production."
    )

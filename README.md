# Détection de plagiat pour mémoires et thèses universitaires

Pipeline de traitement de documents académiques (mémoires et thèses de premier
et second cycle) développé dans le cadre d'un mémoire de Master en
Intelligence Artificielle, en vue de la mise en place d'un détecteur de
plagiat pour des documents universitaires burkinabè.

Le corpus traité est hétérogène : formats de PDF variés, qualité d'extraction
inégale, et surtout schémas de numérotation très divers d'un document à
l'autre (chapitres romains ou arabes, numérotation décimale continue,
sommaires mal formatés, tables des matières parfois dupliquées). Le pipeline
est conçu pour absorber cette hétérogénéité plutôt que de supposer une mise
en forme homogène.

## Objectif du dépôt

Ce dépôt couvre la première partie de la chaîne de traitement : la
**segmentation structurelle** des documents, c'est-à-dire le découpage d'un
PDF en sections cohérentes (chapitres, sections, sous-sections) prêtes à être
comparées entre documents. La détection de plagiat proprement dite
(encodage sémantique, comparaison, scoring de similarité) n'est pas encore
implémentée à ce stade.

## Pipeline

Le traitement d'un document se déroule en quatre étapes, chacune portée par
un module dédié.

### 1. `inspection.py` — Extraction et détection du sommaire

- Extraction du texte page par page via `pdfplumber`.
- Correction des cas d'extraction où chaque caractère semble dupliqué.
- Détection du bloc de sommaire (`detecter_sommaire`) par combinaison de
  trois stratégies complémentaires (lignes à points de suite, occurrences du
  mot « chapitre », structure numérotée sans points de suite), avec choix du
  bloc le plus probable et fusion des bornes des blocs pertinents qui se
  chevauchent.
- Détection des bandeaux répétés (en-têtes, pieds de page) et du taux de
  bruit du texte extrait.

### 2. `extraire_titre_sommaire.py` — Parsing des titres

- Transforme le bloc de sommaire brut en une liste structurée de titres :
  numéro, niveau hiérarchique, chapitre parent, clé unique, statut
  légitime ou non (dédicace, remerciements, bibliographie, etc.).
- Gère les schémas de numérotation mixtes au sein d'un même document
  (chapitres nommés explicitement ou numérotation décimale continue,
  imbrication romain puis arabe) et le rattachement des titres coupés sur
  plusieurs lignes.
- Filtre les bandeaux répétés détectés en amont.

### 3. `localiser_titres_corps.py` — Localisation des titres dans le corps

- Pour chaque titre extrait du sommaire, recherche sa position réelle dans
  le corps du document (le sommaire propose l'ordre attendu, le corps donne
  la position réelle).
- Recherche séquentielle : chaque titre est cherché à partir de la position
  où le précédent a été trouvé, pour éviter de confondre deux occurrences
  du même numéro sous des chapitres différents.
- Repli progressif en cas d'échec de la correspondance stricte (tolérance
  sur le nombre de mots comparés, comparaison sans espaces pour les
  sommaires aux mots collés, similarité approximative en dernier recours) ;
  un titre non localisé n'est jamais deviné, il reste marqué comme tel.

### 4. `decouper_segments.py` — Découpage en chunks

- Extrait le texte de chaque section délimitée à l'étape précédente et le
  nettoie (bandeaux répétés, lignes de bruit, numéros de page isolés).
- Découpe les sections trop longues en respectant en priorité les
  frontières de paragraphe, puis de phrase, avec un léger chevauchement
  entre chunks consécutifs pour préserver le contexte à la coupure.
- Fusionne les sections « titre seul » (sans texte propre avant la
  sous-section suivante) dans le premier chunk de contenu qui suit.
- Chaque chunk produit hérite des métadonnées de son titre d'origine
  (niveau, statut légitime, chapitre parent, clé unique), en vue de l'étape
  de comparaison à venir.

### `test_corpus_complet.py` — Exécution sur le corpus

Fait tourner la chaîne complète sur un corpus de PDF et produit un fichier
de statistiques (`statistiques_structure_corpus.xlsx`) : nombre de titres,
de chapitres, profondeur hiérarchique, taux de localisation dans le corps,
nombre de chunks produits, et repérage automatique des documents dont les
résultats sortent des plages attendues.

## Prérequis

- Python 3.10 ou supérieur
- `pdfplumber`
- `pandas`
- `openpyxl` (pour l'export des statistiques au format Excel)

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pdfplumber pandas openpyxl
```

## Utilisation

Le corpus de documents n'est pas inclus dans ce dépôt (documents
universitaires de tiers, non redistribuables). Pour exécuter le pipeline sur
un corpus local, renseigner le chemin du dossier contenant les PDF dans
`DOSSIER_CORPUS` (`test_corpus_complet.py`), puis lancer :

```bash
python test_corpus_complet.py
```

## Limites connues

- La localisation des titres dans le corps et le découpage en chunks
  reposent sur une exclusion explicite de la zone du sommaire ; un document
  contenant une deuxième copie complète de sa table des matières ailleurs
  que dans cette zone peut produire de faux résultats.
- La taille de chunk par défaut (350 mots) est une valeur indicative,
  choisie en l'absence de modèle d'encodage sémantique déjà défini. Elle
  devra être recalée en nombre de tokens une fois le modèle d'embedding
  choisi (par exemple sentence-camembert-large).

## Prochaines étapes

- Choix et intégration du modèle d'encodage sémantique.
- Comparaison des chunks entre documents et scoring de similarité.
- Prise en compte du statut légitime des sections dans le scoring (une
  section légitime comme une bibliographie ou des remerciements ne doit pas
  être traitée comme du contenu académique comparable).

## Auteur

Sawadogo Rimalguedo Rahimata — mémoire de Master en Intelligence
Artificielle.

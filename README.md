# Détection de plagiat pour mémoires et thèses universitaires

Pipeline de détection de plagiat pour documents académiques (mémoires et
thèses de premier et second cycle), développé dans le cadre d'un mémoire de
Master en Intelligence Artificielle, pour des documents universitaires
burkinabè.

Le corpus traité est hétérogène : formats de PDF variés, qualité
d'extraction inégale, et schémas de numérotation très divers d'un document
à l'autre (chapitres romains ou arabes, numérotation décimale continue,
sommaires mal formatés). Le pipeline est conçu pour absorber cette
hétérogénéité plutôt que de supposer une mise en forme homogène.

**Cet outil signale des similarités ; il ne prononce aucun verdict de
plagiat. L'appréciation finale revient à l'enseignant.**

## Vue d'ensemble en une phrase

Un corpus de PDF de référence est traité **une fois** pour produire des
vecteurs sémantiques et une matrice lexicale ; un document suspect est
ensuite comparé à ce corpus **à chaque vérification**, sans jamais retraiter
le corpus entier.

```
        PDF de référence (dossier corpus/)
                    │
                    ▼
      [A. CONSTRUCTION DU CORPUS]  ──── une fois, ou à l'ajout de documents
                    │
                    ▼
  vecteurs_corpus.npy + metadonnees_corpus.parquet + tfidf_corpus.npz
                    │
                    ▼
      [B. DÉTECTION SUR UN NOUVEAU DOCUMENT]  ──── à chaque vérification
                    │
                    ▼
   rapport (taux global + passages suspects, côte à côte avec leur source)
```

Dernière construction validée : **123 documents, 11 674 chunks** (dont 1 276
chunks « légitimes », soit 10,9 %), après retrait de documents en double.

**Détection en deux étages** (`detecter_plagiat.py`) : l'étage 1 retient, pour
chaque chunk du document suspect, les 50 chunks du corpus les plus proches
en TF-IDF ; l'étage 2 calcule le score sémantique (`sentence-camembert-large`)
uniquement sur ces 50 candidats et fusionne : `score = 0.6 × sémantique +
0.4 × lexical`. Le score lexical TF-IDF est calculé **une seule fois** par
chunk (contre tout le corpus) et réutilisé tel quel pour la sélection des 50
candidats et pour la fusion.

**Mesures de similarité utilisées** (toutes deux des similarités cosinus) :
- **voie lexicale** : chaque texte est transformé en vecteur de poids TF-IDF
  (poids d'un mot = fréquence dans le texte × rareté dans le corpus ;
  unigrammes et bigrammes, mots vides français retirés, vecteurs normalisés en
  L2), puis les deux textes sont comparés par le **cosinus** de ces vecteurs.
  TF-IDF désigne la pondération ; le cosinus est la mesure de comparaison. Les
  poids étant positifs, le score est entre 0 et 1 (1 = vecteurs identiques).
  Les vecteurs du corpus étant déjà normalisés, le cosinus se calcule par un
  simple produit scalaire ;
- **voie sémantique** : **cosinus** entre les vecteurs d'embedding
  `sentence-camembert-large` (non normalisés, d'où le calcul explicite
  produit scalaire / produit des normes) ;
- **fusion** : `0.6 × cosinus sémantique + 0.4 × cosinus TF-IDF`.

## Environnement

**`.venv-embed`** (Python 3.11) est le seul environnement à utiliser pour
l'ensemble du pipeline. Une carte NVIDIA avec CUDA est nécessaire (les
modèles sont chargés avec `device="cuda"`).

```bash
.venv-embed\Scripts\python.exe <script>.py
```

Si `.venv-embed` doit être recréé :
```bash
uv venv .venv-embed --python 3.11
uv pip install --python .venv-embed/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv-embed/Scripts/python.exe -r requirements.txt
```

**Ne jamais relancer `python -m venv .venv-embed` avec un autre Python** :
cela réécrit l'interpréteur de l'environnement (les paquets restent compilés
pour 3.11 et plus rien ne s'importe).

Le dossier du corpus est `C:\Users\1\Documents\MASTER_SOUTENANCE\corpus` par
défaut ; la variable d'environnement `CORPUS_DIR` le remplace (utile sous
Colab). Les dossiers `.venv/`, `.venv-1/`, `.venv-2/`, `venv-embed/` sont
d'anciens environnements, obsolètes.

---

## Interfaces graphiques (Streamlit)

Deux applications distinctes, qui partagent le code du pipeline (aucune
logique métier dupliquée). Le seuil de sensibilité par défaut est 0,65
(réglable de 0,30 à 0,90).

```bash
.venv-embed/Scripts/python.exe -m streamlit run app_detection.py                          # http://localhost:8501
.venv-embed/Scripts/python.exe -m streamlit run app_corpus.py --server.port 8502          # http://localhost:8502
```

- **`app_detection.py`** — dépose d'un PDF à vérifier, taux global, tableau
  triable des passages suspects (score, document source, section, statut
  « légitime »), comparaison côte à côte des deux textes.
- **`app_corpus.py`** — ajout de PDF au corpus et reconstruction de l'index,
  avec barre de progression et état par document (normal / limitation /
  échec).

Points d'attention :
- Les deux applications chargent un modèle sur le GPU : ne pas lancer une
  détection pendant une reconstruction.
- **La reconstruction depuis `app_corpus.py` est toujours complète** (tous
  les documents sont retraités) et **supprime d'abord les anciens artefacts**.
  Si la page est rafraîchie ou fermée en cours de route, le calcul s'arrête
  et l'index reste absent : garder l'onglet ouvert, ou utiliser les scripts
  en ligne de commande (section A), qui ne dépendent pas du navigateur.
  Faire une copie des artefacts avant de reconstruire.
- L'application de détection n'exclut de la recherche que les chunks du
  **document de même nom de fichier** (cas d'un document déjà présent dans
  le corpus). Un même contenu sous un autre nom ressortirait à 100 % : voir
  la vérification des doublons ci-dessous.

---

## A. Construction du corpus de référence

À lancer **une fois**, ou à chaque ajout de nouveaux documents. Chaque
script attend le précédent : l'ordre est important.

**Avant d'ajouter des documents :**
- les noms de fichiers doivent être **uniques** dans tout le corpus
  (sous-dossiers compris) : le document est identifié par son nom ;
- retirer les **doublons** (copies renommées `- Copie.pdf`, `(1).pdf`...) :
  ils sont indexés comme des documents distincts et se « trouvent » entre
  eux à 100 %. Un test rapide : comparer les chunks de chaque paire de
  documents (une paire dont les textes de chunks sont identiques est un
  doublon).

### Modules de bibliothèque (jamais lancés seuls)

`inspection.py`, `extraire_titre_sommaire.py`, `localiser_titres_corps.py`,
`decouper_segments.py` — extraction, détection du sommaire, parsing des
titres, localisation dans le corps et découpage en chunks. Importés par tous
les scripts ci-dessous. `extraire_titre_sommaire.py` contient la liste
`MOTS_CLES_LEGITIMES` qui détermine les sections « légitimes »
(remerciements, bibliographie, présentation de la structure d'accueil...).

### 1. `test_corpus_complet.py` (optionnel — contrôle qualité)

- **Entrée** : dossier de PDF — **Sortie** : `statistiques_structure_corpus.xlsx`
- `.venv-embed\Scripts\python.exe test_corpus_complet.py`

### 2. `encoder_corpus_complet.py`

- **Entrée** : dossier de PDF (constante `CORPUS`)
- **Sortie** : `metadonnees_corpus.parquet`, `vecteurs_corpus.npy`
  (vecteurs sémantiques 1024 dim), `checkpoint_encodage.json` (reprise en
  cas d'interruption)
- Segmente tous les documents (token-based, 400 tokens max, chevauchement
  40) puis encode chaque chunk avec `sentence-camembert-large`
  (`max_seq_length=512`).
- Durées observées pour 123 à 128 documents : segmentation environ 25 à 35 min
  (CPU, sans affichage de progression), encodage ≈ 14 min pour 12 000 chunks
  (GPU).
- **Le fichier de métadonnées n'est créé que s'il n'existe pas** (sinon le
  script reprend un encodage interrompu). Pour reconstruire après un
  changement du corpus, supprimer d'abord `metadonnees_corpus.parquet`,
  `vecteurs_corpus.npy`, `checkpoint_encodage.json`, `tfidf_corpus.npz`,
  `vocabulaire_tfidf.txt` et `correspondance_tfidf.parquet`.

### 3. `construire_tfidf_corpus.py`

- **Entrée** : `metadonnees_corpus.parquet`
- **Sortie** : `tfidf_corpus.npz`, `vocabulaire_tfidf.txt`,
  `correspondance_tfidf.parquet`

```bash
.venv-embed\Scripts\python.exe encoder_corpus_complet.py
.venv-embed\Scripts\python.exe construire_tfidf_corpus.py
```

Trois fichiers suffisent ensuite à toute détection : `vecteurs_corpus.npy`,
`metadonnees_corpus.parquet`, `tfidf_corpus.npz`. Ils contiennent le texte
de documents de tiers : **ne jamais les versionner** (ils sont dans le
`.gitignore`, comme `corpus/`, `sauvegarde_index*/` et `corpus_doublons/`).

**Vérification d'un index reconstruit** : mêmes nombres de lignes dans les
métadonnées, les vecteurs, la matrice TF-IDF et la correspondance, même
ordre des identifiants de chunks, aucun NaN ni vecteur nul, tous les PDF du
dossier présents dans l'index.

---

## B. Détection sur un nouveau document

Via `app_detection.py`, ou en Python :

```python
from detecter_plagiat import charger_ressources, detecter_plagiat
ressources = charger_ressources()   # chargement unique (modèle + corpus)
df_rapport, taux_global = detecter_plagiat("document.pdf", *ressources, seuil=0.65)
```

- **Entrée** : `metadonnees_corpus.parquet`, `tfidf_corpus.npz`,
  `vecteurs_corpus.npy` + le PDF à vérifier
- **Sortie** : un DataFrame trié par score décroissant (chunk suspect, source
  candidate, scores fusionné / sémantique / lexical, statut légitime des deux
  côtés, textes complets) et le taux global
- **Taux global** : proportion des chunks **non légitimes** du document dont
  le meilleur score fusionné dépasse le seuil. Les chunks légitimes restent
  visibles dans le rapport mais sont exclus du taux.
- `python detecter_plagiat.py` lance seulement une démonstration sur un
  document fixé dans le script (non utile en usage normal).
- Le seuil par défaut de la constante `SEUIL_SIMILARITE = 0.75` de
  `detecter_plagiat.py` est une valeur de départ **périmée** ; l'application
  passe explicitement `seuil=0.65`.

---

## Utilisation sur Google Colab

1. Mettre sur Google Drive (dossier **privé**, sans lien de partage) l'index
   (`metadonnees_corpus.parquet`, `vecteurs_corpus.npy`, `tfidf_corpus.npz`)
   et, seulement pour reconstruire l'index, les PDF du corpus.
2. Exécution → type d'exécution → GPU, puis :

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone -b interface https://github.com/Nellehmdlll/plagiat_detection.git
%cd plagiat_detection
!pip install -q -r requirements.txt
!cp /content/drive/MyDrive/plagiat/index/* .

from detecter_plagiat import charger_ressources, detecter_plagiat
ressources = charger_ressources()
df, taux = detecter_plagiat("/content/mon_doc.pdf", *ressources, seuil=0.65)
```

Pour reconstruire l'index sur Colab : `%env CORPUS_DIR=/content/corpus`, puis
`!python encoder_corpus_complet.py` et `!python construire_tfidf_corpus.py`,
et recopier les trois artefacts sur Drive (le disque de Colab est effacé en
fin de session). Le fichier `requirements.txt` ne contient pas `torch`
(préinstallé sur Colab).

---

## Résultats de l'évaluation (175 paires étiquetées)

Jeu de test : 175 paires (101 plagiat, 74 non-plagiat) — 165 paires construites
(catégories A à E) + 10 paires du noyau de calibrage. Chaque paire est comparée
directement (`texte_1` contre `texte_2`). Script : `evaluer_jeu_test_complet.py`.

| Configuration | F1 | précision | rappel | FP / FN |
|---|---|---|---|---|
| Sémantique seul (α = 1) | 0.9754 | 0.9706 | 0.9802 | 3 / 2 |
| Lexical TF-IDF seul (α = 0) | 0.9806 | 0.9619 | 1.0000 | 4 / 0 |
| **Hybride (α = 0.6, seuil 0.60–0.65)** | **0.9854** | 0.9712 | 1.0000 | 3 / 0 |

(grille α × seuil par pas de 0,05 ; 18 couples atteignent ce même F1 — le
plateau couvre α de 0,2 à 0,75, le choix α = 0,6 vient du calibrage initial.)
À α = 0,6, passer le seuil à 0,70 fait chuter le rappel (2 faux négatifs,
F1 = 0,9754) : 0,65 est en bordure de plateau.

Par catégorie : rappel 100 % sur les plagiats A, B et C ; faux positifs 0 / 42
sur la catégorie D (contenu sans rapport) et 3 / 32 (9,4 %) sur la catégorie E
(contenu de nature légitime, ex. présentation d'une même structure d'accueil).

**Ce jeu de test est presque saturé** : les erreurs restantes sont 3 ou 4 paires
de la catégorie E. Les écarts entre configurations portent sur une ou deux
paires et sont du même ordre que l'incertitude statistique (voir ci-dessous).

---

## Scripts de méthodologie et de calibration

Ces scripts ont été exécutés pour prendre des décisions de conception ; ils
sont conservés pour la traçabilité du mémoire et n'ont pas besoin d'être
relancés en usage normal.

### Choix du modèle sémantique — `comparer_modeles_definitif.py`

Comparaison de deux modèles Hugging Face d'embedding : `dangvantuan/sentence-camembert-large`
(**retenu**) et `BAAI/bge-m3` (dense). Critères : séparation paraphrase /
sans rapport, comportement sur les zones riches en citations, vitesse et
mémoire GPU, accès à la sortie lexicale native de BGE-M3.

### Choix du signal lexical

BM25 est une **alternative** à TF-IDF, pas un complément : le pipeline n'utilise
que TF-IDF. BM25 (formule de score de recherche, sans apprentissage, années
1990) est la référence classique de la voie lexicale ; il figure dans le
comparatif comme concurrent testé et non retenu, au même titre que les modèles
récents.

- `comparer_bm25_tfidf.py` — premier comparatif BM25 / TF-IDF / sémantique sur 3
  paires seulement (2 paraphrases, 1 paire à terme rare partagé). Les trois
  scores sont recalculés à chaque exécution. Illustratif, sans précision ni
  rappel : ne suffit pas à justifier un choix.
- `comparer_lexical_10_paires.py` — TF-IDF et BM25 sur les 10 paires « vérité
  connue » (5 paraphrases, 5 sans rapport) : scores moyens par catégorie,
  écart, symétrie. Sortie : `comparatif_lexical_10_paires.csv`.
- **`comparer_lexical_recent.py`** — **comparatif de référence**, quantitatif,
  sur les 175 paires, avec les mêmes métriques et la même grille que
  l'évaluation finale. Signaux comparés : TF-IDF (référence du pipeline), BM25,
  **BGE-M3 sparse** (poids lexicaux appris, `BAAI/bge-m3`), **OpenSearch neural
  sparse multilingue v1** (famille SPLADE, appris, français parmi les langues
  d'entraînement) et le **recouvrement de n-grammes de 3 mots** (Jaccard et
  containment, méthode classique de détection de copie littérale). BGE-M3 sparse
  est calculé sans la librairie FlagEmbedding (même algorithme, fichier officiel
  `sparse_linear.pt` ; non comparé numériquement à l'implémentation officielle).
  Le script imprime aussi le score moyen par catégorie A à E (section D).
  Sorties : `comparatif_lexical_resultats.csv`, `scores_lexical_compares.csv`.

Résultats (corpus de 123 documents ; F1 optimiste = seuil choisi sur les 175
paires ; F1 CV = seuil choisi sur les folds d'entraînement, validation croisée
5 × 10) :

| Signal lexical | AUC seul [IC 95 %] | F1 seul (opt. / CV) | F1 fusion avec camembert (opt. / CV) | faux positifs (fusion) |
|---|---|---|---|---|
| TF-IDF | 0.9835 [0.961 ; 1.000] | 0.9806 / 0.972 | **0.9854** / 0.976 (α = 0.1) | 69, calib-7, calib-8 |
| BM25 | 0.9779 [0.950 ; 0.998] | 0.9806 / 0.976 | 0.9806 / 0.973 (α = 0) | + 44 |
| BGE-M3 sparse | 0.9884 [0.969 ; 0.999] | 0.9806 / 0.976 | 0.9806 / 0.970 (α = 0) | + 44 |
| OpenSearch sparse | 0.9817 [0.955 ; 1.000] | 0.9806 / 0.975 | 0.9806 / 0.972 (α = 0) | + 44 |
| N-grammes 3 mots (Jaccard) | 0.9704 [0.935 ; 0.998] | 0.9665 / 0.954 | 0.9756 / 0.968 (α = 0.4) | 65, 69, calib-7, calib-8 ; 1 faux négatif (99) |
| N-grammes 3 mots (containment) | 0.9660 [0.930 ; 0.993] | 0.9615 / 0.950 | 0.9756 / 0.965 (α = 0.35) | + 44 ; 1 faux négatif (99) |
| (réf.) sémantique seul | 0.9837 [0.960 ; 1.000] | 0.9802 / 0.974 | — | — |

Score moyen par catégorie (scores normalisés dans [0, 1]) ; A = copie quasi
identique, B = modification légère, C = paraphrase profonde (plagiats), D =
sujets différents, E = réutilisation légitime (non-plagiats) :

| Catégorie | n | TF-IDF | BM25 | BGE-M3 sparse | OpenSearch | 3-gr Jaccard | 3-gr containment | Sémantique |
|---|---|---|---|---|---|---|---|---|
| A | 27 | 0.9959 | 0.9957 | 0.9941 | 0.9992 | 0.9409 | 0.9722 | 0.9995 |
| B | 32 | 0.6484 | 0.6875 | 0.7631 | 0.8932 | 0.1955 | 0.3088 | 0.9566 |
| C | 42 | 0.6097 | 0.6622 | 0.7526 | 0.8732 | 0.1492 | 0.2691 | 0.9589 |
| D | 42 | 0.0062 | 0.0158 | 0.0337 | 0.0790 | 0.0004 | 0.0010 | 0.2925 |
| E | 32 | 0.1307 | 0.1813 | 0.2059 | 0.3058 | 0.0366 | 0.0759 | 0.6086 |

Sur la catégorie A, le score sémantique (0,9995) dépasse légèrement TF-IDF
(0,9959) : un embedding traite un synonyme comme un mot quasi identique,
alors que TF-IDF voit un mot rare remplacé (le vecteur de mots change). Les
échelles des colonnes ne sont pas comparables entre méthodes (chacune a sa
propre normalisation) ; c'est le classement entre catégories qui compte.

**Lecture, de façon factuelle :**
- Les différences entre TF-IDF, BM25, BGE-M3 sparse et OpenSearch sparse ne
  sont **pas démontrables** sur ce jeu de test : les écarts d'AUC avec TF-IDF
  (−0,006, +0,005, −0,002) ont des intervalles de confiance qui contiennent
  tous 0, et les écarts de F1 sont inférieurs à l'écart-type de la validation
  croisée (≈ 0,02). Chaque méthode est en tête sur certaines lignes (AUC de BGE-M3
  sparse la plus haute ; F1 de fusion de TF-IDF le plus haut), par de très
  petites marges.
- Les **n-grammes sont significativement moins bons** que TF-IDF (écart d'AUC
  de −0,013 [−0,028 ; −0,002] pour Jaccard et de −0,018 [−0,035 ; −0,005] pour
  le containment, intervalles qui ne contiennent pas 0) : ils détectent la copie
  littérale (catégorie A) mais ratent une partie des reformulations légères
  (1 faux négatif en catégorie B).
- L'écart de F1 optimiste (0,9854 contre 0,9806) correspond à **une seule
  paire** sur 175.
- Les erreurs sont les mêmes pour toutes les méthodes (paires 69, calib-7,
  calib-8 de la catégorie E ; la paire 44 en plus pour BM25 et les modèles
  appris) : ce sont des contenus légitimement similaires, que le choix du
  signal lexical ne corrige pas.
- Avec BM25 et les modèles appris, l'optimum de fusion est α = 0 (le lexical
  seul fait aussi bien que la fusion) : ce jeu de test ne permet pas de
  démontrer l'apport de la composante sémantique sur ces signaux.
- TF-IDF est retenu parce qu'il fait aussi bien, sans modèle supplémentaire
  à charger ni à télécharger : environ 2 s pour les 175 paires contre environ
  28 s (BGE-M3 sparse) et 105 s (OpenSearch sparse), chargement des modèles
  inclus (durées indicatives).

**Critère de décision** (défini avec l'auteure, puis appliqué aux résultats) :
TF-IDF a été construit en premier ; ce comparatif a été fait ensuite, pour
vérifier si une méthode lexicale plus récente ferait mieux. La règle est :
1. critère principal : le **F1 en validation croisée de la fusion** avec le score
   sémantique (c'est ce qui tourne dans le pipeline, et la validation croisée
   évite l'optimisme d'un seuil choisi sur les mêmes paires) ;
2. confirmation : l'**AUC**, qui ne dépend d'aucun seuil ;
3. critère pratique : le **rappel** (l'outil ne rend pas de verdict, un plagiat
   raté est plus grave qu'une fausse alerte relue par l'enseignant) ;
4. si une méthode est à moins d'un écart-type du meilleur F1 CV, on garde **la
   plus simple**.

Métriques : l'**AUC** (aire sous la courbe ROC) est la probabilité qu'une paire de
plagiat tirée au hasard ait un score plus élevé qu'une paire non plagiat tirée
au hasard (1 = parfait, 0,5 = hasard) ; l'**AP** (précision moyenne) est la
moyenne des précisions mesurées à chaque plagiat rencontré en parcourant les
paires par score décroissant ; toutes deux ne dépendent d'aucun seuil. Le **F1
opt.** choisit le seuil sur les paires évaluées (optimiste) ; le **F1 CV** choisit
le seuil sur une partie des paires et le mesure sur le reste (5 plis × 10
répétitions), c'est la mesure à privilégier.

Application : TF-IDF a le meilleur F1 CV en fusion (0,9762 ± 0,017) ; les autres
méthodes sont à moins d'un écart-type, donc non démontrablement moins bonnes,
mais TF-IDF est la plus simple (aucun modèle à charger, environ 2 s contre 19 à
23 s pour les modèles appris sur les 175 paires) : il est conservé. Avec l'AUC
seule, BGE-M3 sparse est devant (0,9884 contre 0,9835), mais l'intervalle de
confiance de la différence contient 0.

**Limites de ce comparatif** : il mesure la capacité à **séparer** des paires
déjà appariées, pas à **retrouver** la bonne source parmi tout le corpus
(rôle de l'étage 1) ; c'est là que des modèles lexicaux appris pourraient
différer, et ce n'est pas mesuré. Le jeu de test est petit et saturé (peu
d'erreurs), donc peu discriminant. Les scores BM25 et OpenSearch (asymétriques,
requête → document) sont symétrisés et ramenés dans [0, 1] pour la fusion
(score(A→B) / score(A→A), moyenne des deux sens).

Deux biais de construction du jeu de test, à connaître : (1) la catégorie D a
été filtrée avec TF-IDF (paires rejetées au-dessus de 0,05), ce qui avantage
légèrement TF-IDF sur cette catégorie ; (2) les paraphrases B et C, générées
par un modèle de langage, conservent beaucoup de vocabulaire (TF-IDF moyen
0,61 à 0,65), alors que des paraphrases écrites à la main n'en gardent presque
aucun (≈ 0,10 sur les paires de `comparer_bm25_tfidf.py`) : un plagiat par
reformulation profonde passerait sous le radar du lexical, c'est le rôle de la
composante sémantique.

Sur les 10 paires « vérité connue » (`comparer_lexical_10_paires.py`) :
TF-IDF donne 0,3003 en moyenne sur les paraphrases contre 0,0032 sur les
paires sans rapport (écart 0,2972) ; BM25 normalisé 0,3455 contre 0,0097
(écart 0,3357). La séparation est parfaite pour les deux. TF-IDF est symétrique
(écart A→B / B→A de 0) ; BM25 ne l'est pas (asymétrie relative moyenne 9,7 %,
maximale 48,6 %).

### Choix du modèle sémantique : comparatif élargi

`comparer_modeles_supplementaires.py` compare `sentence-camembert-large`
(retenu) à trois autres modèles, sur les mêmes 10 paires et les mêmes chunks
que `comparer_modeles_definitif.py` (un modèle chargé à la fois,
`max_seq_length = 512`, préfixe `query: ` pour E5, temps = médiane de 3
passes) :

| | camembert-large (retenu) | paraphrase-multilingual-mpnet-base-v2 | multilingual-e5-large | camembert-base |
|---|---|---|---|---|
| Moyenne paraphrases | 0.8584 | 0.8999 | 0.9700 | 0.8627 |
| Moyenne sans rapport | 0.2150 | 0.2899 | 0.8124 | 0.3191 |
| **Écart** | **0.6434** | 0.6100 | 0.1576 | 0.5436 |
| Paire 9 (ambiguë) | 0.1735 | 0.2801 | 0.8073 | 0.3012 |
| Temps / 346 chunks (médiane) | 11.3 s | 4.1 s | 12.1 s | 4.3 s |
| VRAM pic | 1.61 Go | 1.33 Go | 2.47 Go | 0.69 Go |
| Chunks tronqués (> 512 tokens) | 0 | 17 | 19 | 0 |

Les quatre modèles séparent parfaitement les 10 paires (mini-jeu trop petit pour
les départager) ; camembert-large a le plus grand écart. Les chunks sont découpés
avec le tokenizer de camembert, ce qui avantage les modèles camembert (aucun
chunk tronqué) ; E5 concentre ses similarités entre 0,80 et 0,98 (seuils à
recalibrer). Les temps sont indicatifs.

### Autres scripts de calibration et d'évaluation

- `calibrer_alpha_fusion.py` — calibrage du poids α sur 10 paires (valeur de
  départ retenue : 0.6, d'après Mutsaddi & Choudhary, 2025).
- `construire_jeu_test.py` + scripts satellites (`generer_echantillon_bc.py`,
  `regenerer_categorie_b.py`, `remplacer_sources_suspectes.py`,
  `generer_reste_categorie_b.py`, `generer_reste_categorie_c.py`,
  `regenerer_categorie_e.py`, `relire_echantillon_de.py`) — construction du jeu
  de test (`jeu_test_etiquete.parquet`, 165 paires).
- `evaluer_jeu_test_complet.py` — évaluation finale précision / rappel / F1 ;
  produit `scores_bruts_jeu_test.csv`, `grille_alpha_seuil.csv`,
  `courbe_precision_rappel.csv`.
- `tester_encodage_chunks.py`, `verifier_decoupage_tokens.py` — exploration et
  mise au point ponctuelles, conservées pour la traçabilité.
- Scripts supprimés car obsolètes (récupérables dans l'historique git) :
  `prefiltrer_tfidf.py` (prototype de l'étage 1, logique reprise dans
  `detecter_plagiat.py`), `diagnostic_3_docs.py`, `diagnostic_doc2.py`,
  `diagnostic_specifique.py`, `tester_modeles_embedding.py` (remplacé par
  `comparer_modeles_definitif.py`).

---

## Points d'attention (nettoyage restant)

1. **La construction du jeu de test est fragmentée en 8 scripts** à lancer
   dans un ordre précis (`construire_jeu_test.py` → `generer_echantillon_bc.py`
   → `regenerer_categorie_b.py` → `remplacer_sources_suspectes.py` →
   `generer_reste_categorie_b.py` → `generer_reste_categorie_c.py` →
   `regenerer_categorie_e.py`). Résultat final correct et validé, à
   consolider si le jeu de test doit être régénéré.
2. **Anciens environnements virtuels** (`.venv/`, `.venv-1/`, `.venv-2/`,
   `venv-embed/`) à supprimer.
3. **`comparer_lexical_recent.py` et `comparer_lexical_10_paires.py` importent
   des fonctions de `comparer_bm25_tfidf.py` et `comparer_modeles_definitif.py`**
   (tokenisation BM25, mini-jeu de 10 paires) : ces deux scripts ne peuvent pas
   être supprimés sans déplacer ces fonctions.
4. **`statuts_corpus.json`** (détail par document) n'est produit que par
   `app_corpus.py`, pas par les scripts en ligne de commande.

## Limites connues du pipeline

- La localisation des titres dans le corps et le découpage en chunks reposent
  sur une exclusion explicite de la zone du sommaire ; un document contenant
  une deuxième copie complète de sa table des matières ailleurs peut produire
  de faux résultats.
- **Héritage du statut « légitime » lors de la fusion des « titres seuls »** :
  quand un titre sans contenu est fusionné avec la section suivante
  (`decouper_segments.py`), le chunk obtenu reprend les métadonnées de la
  section absorbante, pas celles du titre fusionné. Un titre de chapitre
  légitime (ex. « Structure d'accueil ») suivi d'une sous-section générique
  peut donc produire un chunk non légitime. Cas connu, non corrigé.
- Certains documents produisent très peu de chunks (sommaire mal analysé) ;
  seule une petite partie de leur texte est alors indexée.
- Le taux global (v1) ne porte que sur les chunks non légitimes du document
  suspect ; formule volontairement simple.
- Les chunks légitimes du corpus restent dans la recherche de candidats
  (transparence et robustesse), mais génèrent plus de faux positifs (9,4 % sur
  la catégorie E contre 0 % sur la catégorie D).
- Plus le corpus est grand, plus il y a de candidats parmi lesquels retenir le
  meilleur : le seuil calibré sur des paires directes peut devenir un peu trop
  permissif sur un très grand corpus.
- Pas de recalibrage nécessaire quand le corpus grandit (α et seuil sont
  calibrés sur des paires fixes) ; seules les statistiques TF-IDF changent
  légèrement. Relancer l'évaluation si le modèle, le prétraitement ou le
  découpage changent.

## Auteur

Sawadogo Rimalguedo Rahimata — mémoire de Master en Intelligence
Artificielle.

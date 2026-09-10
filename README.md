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

## Vue d'ensemble en une phrase

Un corpus de PDF de référence est traité **une fois** pour produire des
vecteurs sémantiques et une matrice lexicale ; un document suspect est
ensuite comparé à ce corpus **à chaque vérification**, sans jamais retraiter
le corpus entier.

```
   PDF de référence (63 documents)
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
     rapport de détection (taux global + chunks les plus suspects)
```

## Environnement virtuel à utiliser

**`.venv-embed`** — c'est le seul environnement à utiliser pour l'ensemble
du pipeline (extraction PDF, segmentation, embeddings, TF-IDF, détection).
Il contient : `pdfplumber`, `pandas`, `openpyxl`, `torch` (CUDA), `sentence-
transformers`, `transformers`, `scikit-learn`, `spacy` (mots vides
français), `scipy`, `pyarrow`, `rank-bm25`.

Toutes les commandes ci-dessous s'exécutent donc avec :
```bash
.venv-embed\Scripts\python.exe <script>.py
```

Les dossiers `.venv/`, `.venv-1/`, `.venv-2/` sont d'anciens environnements
de test, obsolètes — à ignorer (ou supprimer).

Si `.venv-embed` doit être recréé :
```bash
uv venv .venv-embed --python 3.11
uv pip install --python .venv-embed/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv-embed/Scripts/python.exe sentence-transformers transformers scikit-learn spacy scipy pyarrow rank-bm25 sentencepiece protobuf pdfplumber pandas openpyxl
```

---

## A. Construction du corpus de référence

À lancer **une fois**, ou à chaque ajout de nouveaux documents de référence
au corpus. Chaque script attend le précédent : l'ordre est important.

### Modules de bibliothèque (jamais lancés seuls)

`inspection.py`, `extraire_titre_sommaire.py`, `localiser_titres_corps.py`,
`decouper_segments.py` — fonctions d'extraction, détection du sommaire,
parsing des titres, localisation dans le corps et découpage en chunks.
Importées par tous les scripts ci-dessous, pas de sortie propre.

### 1. `test_corpus_complet.py` (optionnel — contrôle qualité)

- **Entrée** : dossier de PDF (constante `DOSSIER_CORPUS` dans le script)
- **Sortie** : `statistiques_structure_corpus.xlsx` (titres, chapitres,
  profondeur, taux de localisation, chunks, documents atypiques)
- **Commande** :
  ```bash
  .venv-embed\Scripts\python.exe test_corpus_complet.py
  ```

### 2. `encoder_corpus_complet.py`

- **Entrée** : dossier de PDF (constante `CORPUS`)
- **Sortie** : `vecteurs_corpus.npy` (vecteurs sémantiques, 1024 dim),
  `metadonnees_corpus.parquet` (texte + métadonnées de chaque chunk),
  `checkpoint_encodage.json` (reprise en cas d'interruption)
- **Commande** :
  ```bash
  .venv-embed\Scripts\python.exe encoder_corpus_complet.py
  ```
- Segmente tous les documents (découpage token-based, 400 tokens max,
  chevauchement 40 tokens) et encode chaque chunk avec
  `sentence-camembert-large` (`max_seq_length=512`, correctif du bug de
  position CamemBERT/RoBERTa).

### 3. `construire_tfidf_corpus.py`

- **Entrée** : `metadonnees_corpus.parquet` (produit à l'étape précédente)
- **Sortie** : `tfidf_corpus.npz` (matrice sparse), `vocabulaire_tfidf.txt`,
  `correspondance_tfidf.parquet`
- **Commande** :
  ```bash
  .venv-embed\Scripts\python.exe construire_tfidf_corpus.py
  ```

À l'issue de ces étapes, le corpus de référence est prêt : trois fichiers
(`vecteurs_corpus.npy`, `metadonnees_corpus.parquet`, `tfidf_corpus.npz`)
suffisent à toute détection ultérieure.

---

## B. Détection sur un nouveau document

À lancer **à chaque fois** qu'on veut vérifier un document. Ne retraite pas
le corpus de référence (déjà construit en A).

### `detecter_plagiat.py`

- **Entrée** : `metadonnees_corpus.parquet`, `tfidf_corpus.npz`,
  `vecteurs_corpus.npy` (issus de A) + le chemin du document à vérifier
- **Sortie** : `rapport_detection_<nom>.csv` (rapport complet, trié par
  score décroissant) + rapport console (taux global, top 15 chunks les
  plus suspects)
- **Commande** :
  ```bash
  .venv-embed\Scripts\python.exe detecter_plagiat.py
  ```

**Point d'attention important** : ce script n'accepte pas encore le chemin
du document en argument de ligne de commande. Le document à vérifier est
fixé en dur ligne ~196 (`chemin_simpore = f"{CORPUS}\\SIMPORE Némata Version
Finale.pdf"`) — pour vérifier un autre document, il faut éditer cette ligne
directement dans le script avant de le lancer. C'est le premier point à
corriger avant l'interface graphique (voir plus bas).

Le pipeline segmente le document suspect, cherche pour chaque chunk ses 50
meilleurs candidats dans le corpus par similarité TF-IDF (excluant les
chunks du document lui-même s'il est déjà dans le corpus), calcule le score
sémantique seulement sur ces 50 candidats (réutilise les vecteurs déjà
encodés du corpus, pas de réencodage), fusionne les deux scores
(`0.6 × sémantique + 0.4 × lexical`) et retient le meilleur candidat par
chunk suspect.

---

## Exemple de bout en bout

Corpus déjà construit une première fois (étapes A.2 et A.3 exécutées) ;
pour vérifier un nouveau document, seule l'étape B est à relancer :

```bash
# 1. (une seule fois) construire le corpus de reference
.venv-embed\Scripts\python.exe encoder_corpus_complet.py
.venv-embed\Scripts\python.exe construire_tfidf_corpus.py

# 2. (a chaque nouveau document) editer la ligne "chemin_simpore" dans
#    detecter_plagiat.py pour pointer vers le document a verifier, puis :
.venv-embed\Scripts\python.exe detecter_plagiat.py
```

---

## Scripts de méthodologie et de calibration

Ces scripts ont déjà été exécutés pour prendre des décisions de conception
(choix du modèle, du signal lexical, du poids α, du seuil) ; ils sont
conservés pour la traçabilité du mémoire mais ne font pas partie du
pipeline de production courant (A et B ci-dessus) et n'ont pas besoin
d'être relancés en usage normal :

- `comparer_modeles_definitif.py` — comparatif sentence-camembert-large vs
  BGE-M3 (modèle retenu : sentence-camembert-large).
- `comparer_bm25_tfidf.py` — comparatif BM25 vs TF-IDF (signal retenu :
  TF-IDF).
- `calibrer_alpha_fusion.py` — calibrage du poids α sur un mini-jeu de 10
  paires (valeur retenue : 0.6).
- `construire_jeu_test.py` + scripts satellites (`generer_echantillon_bc.py`,
  `regenerer_categorie_b.py`, `remplacer_sources_suspectes.py`,
  `generer_reste_categorie_b.py`, `generer_reste_categorie_c.py`,
  `regenerer_categorie_e.py`, `relire_echantillon_de.py`) — construction du
  jeu de test étiqueté de 175 paires. Voir « Points d'attention » ci-dessous.
- `evaluer_jeu_test_complet.py` — évaluation finale précision/rappel/F1 sur
  les 175 paires (F1=0.9854 au couple α=0.6/seuil≈0.6).
- `tester_modeles_embedding.py`, `tester_encodage_chunks.py`,
  `verifier_decoupage_tokens.py`, `prefiltrer_tfidf.py`,
  `diagnostic_3_docs.py`, `diagnostic_doc2.py`, `diagnostic_specifique.py`
  — scripts d'exploration et de mise au point ponctuels. Voir « Points
  d'attention ».

---

## Points d'attention avant l'interface graphique

À nettoyer ou fusionner pour arriver à un pipeline compréhensible avant de
l'envelopper dans une application :

1. **`detecter_plagiat.py` n'a pas de véritable interface en ligne de
   commande** : le document cible et le seuil sont des constantes en dur
   dans le fichier. C'est le point bloquant n°1 avant toute interface.
2. **Le seuil `SEUIL_SIMILARITE = 0.75` dans `detecter_plagiat.py` est
   obsolète** : c'était une valeur de départ documentée comme provisoire ;
   l'évaluation complète (`evaluer_jeu_test_complet.py`) a depuis trouvé un
   optimum autour de 0.60–0.65. À aligner.
3. **`prefiltrer_tfidf.py` est redondant** : c'était le prototype de
   l'étage 1 (recherche des 50 candidats), mais `detecter_plagiat.py`
   réimplémente la même logique en interne plutôt que d'importer ce
   module. Soit factoriser (`detecter_plagiat.py` importe la fonction de
   `prefiltrer_tfidf.py`), soit supprimer ce dernier.
4. **La construction du jeu de test est fragmentée en 8 scripts** qui
   doivent être lancés dans un ordre précis et non évident
   (`construire_jeu_test.py` → `generer_echantillon_bc.py` →
   `regenerer_categorie_b.py` → `remplacer_sources_suspectes.py` →
   `generer_reste_categorie_b.py` → `generer_reste_categorie_c.py` →
   `regenerer_categorie_e.py`). Résultat final (`jeu_test_etiquete.parquet`)
   correct et validé, mais à consolider en un seul script si le jeu de
   test doit être régénéré un jour.
5. **`tester_modeles_embedding.py` est obsolète** : remplacé par le
   comparatif plus complet `comparer_modeles_definitif.py`.
6. **`diagnostic_3_docs.py`, `diagnostic_doc2.py`,
   `diagnostic_specifique.py`** sont des scripts de mise au point ponctuels
   liés à des bugs déjà corrigés (bin-packing, chevauchement) — obsolètes,
   à supprimer ou archiver hors du dépôt de code actif.
7. **Trois environnements virtuels obsolètes** (`.venv/`, `.venv-1/`,
   `.venv-2/`) ne servent plus à rien — seul `.venv-embed` est utilisé.

Rien n'a été modifié à cette étape : ce sont des constats, pas des
corrections déjà faites.

## Limites connues du pipeline

- La localisation des titres dans le corps et le découpage en chunks
  reposent sur une exclusion explicite de la zone du sommaire ; un document
  contenant une deuxième copie complète de sa table des matières ailleurs
  que dans cette zone peut produire de faux résultats.
- Le taux global de détection (v1) ne porte que sur les chunks
  non-légitimes du document suspect ; la formule est documentée comme
  volontairement simple et pourra être affinée.
- Les chunks `legitime=True` du corpus restent dans la recherche de
  candidats (étage 1) par choix de conception (transparence et
  robustesse), mais génèrent plus de faux positifs que le contenu
  clairement sans rapport (confirmé par l'évaluation complète : 9.4% sur
  la catégorie E contre 0% sur la catégorie D).

## Auteur

Sawadogo Rimalguedo Rahimata — mémoire de Master en Intelligence
Artificielle.

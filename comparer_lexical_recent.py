# -*- coding: utf-8 -*-
"""
Comparatif des signaux lexicaux sur les 175 paires etiquetees du jeu de test
(memes paires, memes metriques, meme grille (alpha, seuil) que
evaluer_jeu_test_complet.py) :

  - TF-IDF (reference actuelle du pipeline, unigrammes + bigrammes)
  - BM25 Okapi (classique, deja compare a TF-IDF sur 3 paires dans
    comparer_bm25_tfidf.py)
  - BGE-M3 sparse (BAAI/bge-m3, poids lexicaux appris par token)
  - OpenSearch neural sparse multilingue v1 (famille SPLADE, apprise,
    francais parmi les langues d'entrainement)
  - recouvrement de n-grammes de mots (3 mots consecutifs, sans retrait des
    mots vides) : Jaccard et containment - methode classique de detection de
    copie litterale, sans ponderation ni apprentissage

Deux evaluations par methode :
  A. signal lexical SEUL : AUC ROC, precision moyenne (AP), meilleur F1
     (seuil optimal choisi sur les 175 paires = optimiste) et F1 en
     validation croisee (seuil choisi sur les folds d'entrainement) ;
  B. FUSION avec le score semantique (sentence-camembert-large, repris de
     scores_bruts_jeu_test.csv) : meilleur (alpha, seuil) sur la meme grille
     que l'evaluation finale, F1 optimiste et F1 en validation croisee.

Pour la fusion, chaque score lexical doit etre dans [0, 1] comme un cosinus :
  - TF-IDF : cosinus (deja borne) ;
  - BGE-M3 sparse : produit scalaire des poids / produit des normes ;
  - BM25 et OpenSearch sparse (scores asymetriques, requete -> document) :
    score(A->B) / score(A->A), symetrise par la moyenne des deux sens,
    ramene dans [0, 1].
Les scores bruts (non normalises) sont conserves dans le CSV de sortie.

PORTEE ET LIMITES : on mesure la capacite du signal lexical a SEPARER des
paires plagiat / non-plagiat deja appariees, pas sa capacite a RETROUVER la
bonne source parmi tout le corpus (role de l'etage 1 du pipeline). Cette
seconde mesure n'est pas faite ici.

BGE-M3 sparse est calcule sans la librairie FlagEmbedding (dont
l'installation mettrait a jour transformers/huggingface-hub dans l'environnement
du pipeline) : meme algorithme que BGEM3FlagModel (relu(sparse_linear(dernier
etat cache)), poids maximal par identifiant de token, tokens speciaux exclus,
score = somme des produits des poids sur les tokens communs), avec le fichier
officiel sparse_linear.pt du depot BAAI/bge-m3.

A executer avec l'environnement .venv-embed (apres evaluer_jeu_test_complet.py,
dont on reutilise scores_bruts_jeu_test.csv) :
    .venv-embed\\Scripts\\python.exe comparer_lexical_recent.py
"""

import re
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from spacy.lang.fr.stop_words import STOP_WORDS as MOTS_VIDES_FR
from transformers import AutoModel, AutoTokenizer

from comparer_bm25_tfidf import tokeniser, score_bm25_paire, K1, B
from construire_tfidf_corpus import pretraiter, FICHIER_METADONNEES
from evaluer_jeu_test_complet import (
    FICHIER_SCORES_BRUTS, VALEURS_ALPHA, VALEURS_SEUIL,
    charger_jeu_complet, cosinus_sparse, precision_rappel_f1,
)

NOM_BGE = "BAAI/bge-m3"
NOM_OPENSEARCH = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
LONGUEUR_MAX_BGE = 1024

FICHIER_SCORES_LEXICAUX = "scores_lexical_compares.csv"
FICHIER_RESULTATS = "comparatif_lexical_resultats.csv"

N_FOLDS, N_REPETITIONS, GRAINE = 5, 10, 0


# -----------------------------------------------------------------------------
# SCORES LEXICAUX PAR METHODE (une valeur brute et une valeur normalisee [0,1]
# par paire)
# -----------------------------------------------------------------------------

def normaliser_asymetrique(s_ab, s_aa, s_ba, s_bb):
    """score(A->B)/score(A->A) et score(B->A)/score(B->B), moyennes, bornees a [0, 1]."""
    s_ab, s_aa, s_ba, s_bb = (np.asarray(x, dtype=float) for x in (s_ab, s_aa, s_ba, s_bb))
    n_ab = np.where(s_aa > 0, s_ab / np.where(s_aa > 0, s_aa, 1), 0.0)
    n_ba = np.where(s_bb > 0, s_ba / np.where(s_bb > 0, s_bb, 1), 0.0)
    return np.clip((n_ab + n_ba) / 2, 0.0, 1.0)


def scores_tfidf(df, df_meta):
    vectoriseur = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.85, stop_words=sorted(MOTS_VIDES_FR), lowercase=True
    )
    vectoriseur.fit([pretraiter(t) for t in df_meta["texte"]])
    v1 = vectoriseur.transform([pretraiter(t) for t in df["texte_1"]])
    v2 = vectoriseur.transform([pretraiter(t) for t in df["texte_2"]])
    s = np.array([cosinus_sparse(v1.getrow(i), v2.getrow(i)) for i in range(len(df))])
    return s, s


def scores_bm25(df, df_meta):
    bm25 = BM25Okapi([tokeniser(t) for t in df_meta["texte"]], k1=K1, b=B)
    t1, t2 = df["texte_1"].tolist(), df["texte_2"].tolist()
    s_ab = [score_bm25_paire(bm25, a, b) for a, b in zip(t1, t2)]
    s_ba = [score_bm25_paire(bm25, b, a) for a, b in zip(t1, t2)]
    s_aa = [score_bm25_paire(bm25, a, a) for a in t1]
    s_bb = [score_bm25_paire(bm25, b, b) for b in t2]
    brut = (np.array(s_ab) + np.array(s_ba)) / 2
    return brut, normaliser_asymetrique(s_ab, s_aa, s_ba, s_bb)


@torch.no_grad()
def poids_lexicaux_bge(textes, tokenizer, modele, tete_sparse):
    ids_exclus = {tokenizer.cls_token_id, tokenizer.eos_token_id,
                  tokenizer.pad_token_id, tokenizer.unk_token_id}
    resultats = []
    for i in range(0, len(textes), 16):
        lot = textes[i:i + 16]
        enc = tokenizer(lot, padding=True, truncation=True, max_length=LONGUEUR_MAX_BGE,
                        return_tensors="pt").to("cuda")
        etats = modele(**enc).last_hidden_state
        poids = torch.relu(tete_sparse(etats)).squeeze(-1).float().cpu()
        for j in range(len(lot)):
            d = {}
            for tid, w in zip(enc["input_ids"][j].tolist(), poids[j].tolist()):
                if tid in ids_exclus or w <= 0:
                    continue
                if w > d.get(tid, 0.0):
                    d[tid] = w
            resultats.append(d)
    return resultats


def scores_bge_sparse(df):
    tokenizer = AutoTokenizer.from_pretrained(NOM_BGE)
    modele = AutoModel.from_pretrained(NOM_BGE, dtype=torch.float16).to("cuda").eval()
    tete = torch.nn.Linear(modele.config.hidden_size, 1)
    tete.load_state_dict(torch.load(hf_hub_download(NOM_BGE, "sparse_linear.pt"), map_location="cpu"))
    tete = tete.half().to("cuda").eval()

    w1 = poids_lexicaux_bge(df["texte_1"].tolist(), tokenizer, modele, tete)
    w2 = poids_lexicaux_bge(df["texte_2"].tolist(), tokenizer, modele, tete)

    brut, norme = [], []
    for a, b in zip(w1, w2):
        s = sum(w * b[t] for t, w in a.items() if t in b)
        na = sum(w * w for w in a.values()) ** 0.5
        nb = sum(w * w for w in b.values()) ** 0.5
        brut.append(s)
        norme.append(s / (na * nb) if na > 0 and nb > 0 else 0.0)
    del modele, tete
    torch.cuda.empty_cache()
    return np.array(brut), np.clip(np.array(norme), 0.0, 1.0)


def scores_opensearch_sparse(df):
    from sentence_transformers.sparse_encoder import SparseEncoder
    modele = SparseEncoder(NOM_OPENSEARCH, device="cuda")

    def encoder(textes):
        q = modele.encode_query(textes, convert_to_sparse_tensor=False, batch_size=16)
        d = modele.encode_document(textes, convert_to_sparse_tensor=False, batch_size=16)
        return q.float().cpu().numpy(), d.float().cpu().numpy()

    q1, d1 = encoder(df["texte_1"].tolist())
    q2, d2 = encoder(df["texte_2"].tolist())
    s_ab = (q1 * d2).sum(axis=1)   # requete = texte_1, document = texte_2
    s_ba = (q2 * d1).sum(axis=1)
    s_aa = (q1 * d1).sum(axis=1)
    s_bb = (q2 * d2).sum(axis=1)
    del modele
    torch.cuda.empty_cache()
    return (s_ab + s_ba) / 2, normaliser_asymetrique(s_ab, s_aa, s_ba, s_bb)


MOT = re.compile(r"\w+", re.UNICODE)
TAILLE_NGRAMME = 3


def ngrammes(texte, n=TAILLE_NGRAMME):
    mots = MOT.findall(pretraiter(texte).lower())
    if len(mots) < n:
        return {tuple(mots)} if mots else set()
    return {tuple(mots[i:i + n]) for i in range(len(mots) - n + 1)}


def scores_ngrammes(df, mode):
    """Jaccard = |A inter B| / |A union B| ; containment = |A inter B| / min(|A|, |B|). Deja dans [0, 1]."""
    scores = []
    for a, b in zip(df["texte_1"], df["texte_2"]):
        ga, gb = ngrammes(a), ngrammes(b)
        inter = len(ga & gb)
        if mode == "jaccard":
            union = len(ga | gb)
            scores.append(inter / union if union else 0.0)
        else:
            scores.append(inter / min(len(ga), len(gb)) if ga and gb else 0.0)
    s = np.array(scores)
    return s, s


# -----------------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------------

def f1_sur_indices(P, y, idx):
    """F1 de chaque configuration (lignes de P, booleen) restreint aux indices idx."""
    p, t = P[:, idx], y[idx]
    vp = (p & t).sum(axis=1)
    fp = (p & ~t).sum(axis=1)
    fn = (~p & t).sum(axis=1)
    denom = 2 * vp + fp + fn
    return np.where(denom > 0, 2 * vp / np.where(denom > 0, denom, 1), 0.0)


def evaluer_signal_seul(y, s, decoupages):
    """Seuils candidats = milieux entre scores consecutifs distincts."""
    valeurs = np.unique(s)
    seuils = np.concatenate([[valeurs[0] - 1e-9], (valeurs[:-1] + valeurs[1:]) / 2])
    P = s[None, :] > seuils[:, None]
    f1_tous = f1_sur_indices(P, y, np.arange(len(y)))
    k = int(np.argmax(f1_tous))
    precision, rappel, f1, *_ = precision_rappel_f1(y, P[k])

    f1_cv = []
    for train, test in decoupages:
        k_train = int(np.argmax(f1_sur_indices(P, y, train)))
        f1_cv.append(f1_sur_indices(P[k_train:k_train + 1], y, test)[0])
    return {
        "auc": roc_auc_score(y, s), "ap": average_precision_score(y, s),
        "f1_optimiste": f1, "precision": precision, "rappel": rappel,
        "seuil": float(seuils[k]), "f1_cv": float(np.mean(f1_cv)), "f1_cv_ecart_type": float(np.std(f1_cv)),
    }


def bootstrap_auc(y, scores, n_rep=2000, graine=0):
    """IC 95% de l'AUC de chaque methode et de sa difference appariee a la premiere methode."""
    rng = np.random.default_rng(graine)
    echantillons = [rng.integers(0, len(y), len(y)) for _ in range(n_rep)]
    echantillons = [i for i in echantillons if 0 < y[i].sum() < len(y)]
    auc = {nom: np.array([roc_auc_score(y[i], s[i]) for i in echantillons]) for nom, s in scores.items()}
    return echantillons, auc


def evaluer_fusion(y, sem, lex, categories, ids, decoupages):
    configs = [(a, s) for a in VALEURS_ALPHA for s in VALEURS_SEUIL]
    P = np.stack([(a * sem + (1 - a) * lex) > s for a, s in configs])
    f1_tous = f1_sur_indices(P, y, np.arange(len(y)))
    k = int(np.argmax(f1_tous))
    alpha, seuil = configs[k]
    precision, rappel, f1, vp, fp, vn, fn = precision_rappel_f1(y, P[k])

    f1_cv = []
    for train, test in decoupages:
        k_train = int(np.argmax(f1_sur_indices(P, y, train)))
        f1_cv.append(f1_sur_indices(P[k_train:k_train + 1], y, test)[0])

    pred = P[k]
    detail = {}
    for cat in ["A", "B", "C"]:
        m = categories == cat
        detail[f"rappel_{cat}"] = float((pred & m).sum() / m.sum())
    for cat in ["D", "E"]:
        m = categories == cat
        detail[f"fp_{cat}"] = float((pred & m).sum() / m.sum())
    return {
        "alpha": alpha, "seuil": seuil, "f1_optimiste": f1, "precision": precision, "rappel": rappel,
        "VP": vp, "FP": fp, "VN": vn, "FN": fn,
        "faux_positifs": ";".join(ids[pred & ~y]), "faux_negatifs": ";".join(ids[~pred & y]),
        "f1_cv": float(np.mean(f1_cv)), "f1_cv_ecart_type": float(np.std(f1_cv)), **detail,
    }


if __name__ == "__main__":
    df = charger_jeu_complet().reset_index(drop=True)
    df["id_paire"] = df["id_paire"].astype(str)
    y = (df["etiquette"] == "PLAGIAT").to_numpy()
    categories = df["categorie"].to_numpy()
    print(f"{len(df)} paires : {int(y.sum())} PLAGIAT, {int((~y).sum())} NON-PLAGIAT ; "
          f"par categorie {df['categorie'].value_counts().sort_index().to_dict()}")

    anciens = pd.read_csv(FICHIER_SCORES_BRUTS)
    anciens["id_paire"] = anciens["id_paire"].astype(str)
    anciens = anciens.set_index("id_paire")
    assert set(df["id_paire"]) == set(anciens.index), "scores_bruts_jeu_test.csv ne correspond pas au jeu de test"
    sem = anciens.loc[df["id_paire"], "score_semantique"].to_numpy()
    lex_ancien = anciens.loc[df["id_paire"], "score_lexical"].to_numpy()

    df_meta = pd.read_parquet(FICHIER_METADONNEES)
    print(f"Corpus de reference (TF-IDF / BM25 ajustes dessus) : {len(df_meta)} chunks, "
          f"{df_meta['document'].nunique()} documents")

    decoupages = list(RepeatedStratifiedKFold(
        n_splits=N_FOLDS, n_repeats=N_REPETITIONS, random_state=GRAINE).split(np.zeros(len(y)), y))

    brut, norme, durees = {}, {}, {}
    for nom, fonction in [
        ("TF-IDF", lambda: scores_tfidf(df, df_meta)),
        ("BM25", lambda: scores_bm25(df, df_meta)),
        ("BGE-M3 sparse", lambda: scores_bge_sparse(df)),
        ("OpenSearch neural sparse (multilingue)", lambda: scores_opensearch_sparse(df)),
        (f"N-grammes {TAILLE_NGRAMME} mots (Jaccard)", lambda: scores_ngrammes(df, "jaccard")),
        (f"N-grammes {TAILLE_NGRAMME} mots (containment)", lambda: scores_ngrammes(df, "containment")),
    ]:
        print(f"\nCalcul des scores : {nom}...")
        t0 = time.time()
        brut[nom], norme[nom] = fonction()
        durees[nom] = time.time() - t0
        print(f"  {durees[nom]:.1f}s ; score brut min/moy/max = "
              f"{brut[nom].min():.4f} / {brut[nom].mean():.4f} / {brut[nom].max():.4f}")

    sortie = df[["id_paire", "categorie", "etiquette"]].copy()
    sortie["score_semantique"] = sem
    for nom in brut:
        sortie[f"brut_{nom}"] = brut[nom]
        sortie[f"norme_{nom}"] = norme[nom]
    sortie.to_csv(FICHIER_SCORES_LEXICAUX, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 110}\nCONTROLE DE REPRODUCTIBILITE - TF-IDF de l'evaluation finale (score_lexical de "
          f"{FICHIER_SCORES_BRUTS})\n{'=' * 110}")
    ids = df["id_paire"].to_numpy()
    controle = evaluer_fusion(y, sem, lex_ancien, categories, ids, decoupages)
    print(f"Fusion, alpha={controle['alpha']}, seuil={controle['seuil']} : "
          f"F1={controle['f1_optimiste']:.4f} (precision={controle['precision']:.4f}, rappel={controle['rappel']:.4f})")

    lignes = []
    print(f"\n{'=' * 110}\nA. SIGNAL LEXICAL SEUL\n{'=' * 110}")
    print(f"{'Methode':42s} {'AUC':>7s} {'AP':>7s} {'F1 opt.':>8s} {'prec.':>7s} {'rappel':>7s} {'F1 CV':>14s}")
    for nom in brut:
        r = evaluer_signal_seul(y, norme[nom], decoupages)
        print(f"{nom:42s} {r['auc']:7.4f} {r['ap']:7.4f} {r['f1_optimiste']:8.4f} {r['precision']:7.4f} "
              f"{r['rappel']:7.4f} {r['f1_cv']:7.4f}+-{r['f1_cv_ecart_type']:.3f}")
        lignes.append({"evaluation": "lexical_seul", "methode": nom, **r})
    r = evaluer_signal_seul(y, sem, decoupages)
    print(f"{'(reference) semantique seul':42s} {r['auc']:7.4f} {r['ap']:7.4f} {r['f1_optimiste']:8.4f} "
          f"{r['precision']:7.4f} {r['rappel']:7.4f} {r['f1_cv']:7.4f}+-{r['f1_cv_ecart_type']:.3f}")
    lignes.append({"evaluation": "semantique_seul", "methode": "camembert", **r})

    print(f"\n{'=' * 110}\nB. FUSION avec le score semantique (alpha x semantique + (1-alpha) x lexical)\n{'=' * 110}")
    print(f"{'Methode':42s} {'alpha':>6s} {'seuil':>6s} {'F1 opt.':>8s} {'prec.':>7s} {'rappel':>7s} "
          f"{'F1 CV':>14s}  {'VP/FP/VN/FN':>14s}")
    fusions = {}
    for nom in brut:
        r = evaluer_fusion(y, sem, norme[nom], categories, ids, decoupages)
        fusions[nom] = r
        print(f"{nom:42s} {r['alpha']:6.2f} {r['seuil']:6.2f} {r['f1_optimiste']:8.4f} {r['precision']:7.4f} "
              f"{r['rappel']:7.4f} {r['f1_cv']:7.4f}+-{r['f1_cv_ecart_type']:.3f}  "
              f"{r['VP']:>3d}/{r['FP']:>2d}/{r['VN']:>3d}/{r['FN']:>2d}")
        lignes.append({"evaluation": "fusion_avec_semantique", "methode": nom, **r})

    print(f"\nDetail par categorie au couple optimal de chaque fusion "
          f"(rappel A/B/C = plagiats detectes ; FP D/E = non-plagiats a tort signales)")
    print(f"{'Methode':42s} {'rappel A':>9s} {'rappel B':>9s} {'rappel C':>9s} {'FP D':>7s} {'FP E':>7s}")
    for nom, r in fusions.items():
        print(f"{nom:42s} {r['rappel_A']:9.4f} {r['rappel_B']:9.4f} {r['rappel_C']:9.4f} "
              f"{r['fp_D']:7.4f} {r['fp_E']:7.4f}")

    print(f"\nPaires mal classees au couple optimal de chaque fusion (id_paire ; les categories "
          f"sont A-C = plagiat, D-E = non-plagiat)")
    for nom, r in fusions.items():
        print(f"  {nom:42s} faux positifs : {r['faux_positifs'] or '-'} | faux negatifs : {r['faux_negatifs'] or '-'}")

    print(f"\n{'=' * 110}\nC. INCERTITUDE - AUC et intervalle de confiance a 95% (bootstrap sur les paires)\n{'=' * 110}")
    scores_auc = {nom: norme[nom] for nom in brut}
    scores_auc["(reference) semantique seul"] = sem
    echantillons, auc = bootstrap_auc(y, scores_auc)
    print(f"{len(echantillons)} reechantillons.")
    print(f"{'Methode':42s} {'AUC':>7s} {'IC 95%':>20s}   difference d'AUC vs TF-IDF (IC 95%)")
    auc_tfidf = roc_auc_score(y, norme["TF-IDF"])
    for nom, s in scores_auc.items():
        a = roc_auc_score(y, s)
        ic = f"[{np.percentile(auc[nom], 2.5):.4f} ; {np.percentile(auc[nom], 97.5):.4f}]"
        if nom == "TF-IDF":
            diff = "(reference)"
        else:
            d = auc[nom] - auc["TF-IDF"]
            lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
            diff = f"{a - auc_tfidf:+.4f} [{lo:+.4f} ; {hi:+.4f}] " + ("contient 0" if lo <= 0 <= hi else "NE contient PAS 0")
        print(f"{nom:42s} {a:7.4f} {ic:>20s}   {diff}")

    print(f"\n{'=' * 110}\nD. SCORE MOYEN PAR CATEGORIE (scores normalises dans [0, 1]) : moyenne [minimum]\n{'=' * 110}")
    print("A = copie quasi identique, B = modification legere, C = paraphrase profonde (plagiats) ; "
          "D = sujets differents, E = reutilisation legitime (non-plagiats)")
    sigles = {
        "TF-IDF": "TF-IDF", "BM25": "BM25", "BGE-M3 sparse": "BGE-M3 sp.",
        "OpenSearch neural sparse (multilingue)": "OpenSearch",
        f"N-grammes {TAILLE_NGRAMME} mots (Jaccard)": f"{TAILLE_NGRAMME}-gr Jaccard",
        f"N-grammes {TAILLE_NGRAMME} mots (containment)": f"{TAILLE_NGRAMME}-gr contain.",
    }
    colonnes = {sigles[nom]: norme[nom] for nom in brut}
    colonnes["Semantique"] = sem
    tableau = pd.DataFrame(colonnes)
    tableau["categorie"] = categories
    print(f"{'Cat.':5s} {'n':>3s} " + " ".join(f"{c:>15s}" for c in colonnes))
    for cat in sorted(set(categories)):
        sous = tableau[tableau["categorie"] == cat]
        print(f"{cat:5s} {len(sous):3d} "
              + " ".join(f"{sous[c].mean():7.4f} [{sous[c].min():5.3f}]" for c in colonnes))

    print(f"\nDuree de calcul des scores (175 paires, chargement/telechargement des modeles inclus) : "
          + ", ".join(f"{n} {d:.1f}s" for n, d in durees.items()))
    print(f"Validation croisee : {N_FOLDS} folds x {N_REPETITIONS} repetitions, graine {GRAINE}")

    pd.DataFrame(lignes).to_csv(FICHIER_RESULTATS, index=False, encoding="utf-8-sig")
    print(f"\nResultats sauvegardes : {FICHIER_RESULTATS}, {FICHIER_SCORES_LEXICAUX}")

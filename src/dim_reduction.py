"""
dim_reduction.py
Dimensionality reduction for sparse and dense text feature spaces.

Provides:
  - reduce_tfidf()           : Truncated SVD on sparse TF-IDF
  - reduce_embeddings()      : PCA on dense DistilBERT / GloVe
  - compute_variance_curves(): Full variance sweep for scree-plot generation
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_N   = 100     # default compression target (components)
RANDOM_SEED = 42


# ── Public API ───────────────────────────────────────────────────────────────

def reduce_tfidf(
    tfidf_features,
    n_components: int = TARGET_N,
) -> tuple[np.ndarray, TruncatedSVD]:
    """
    Compress sparse TF-IDF features via Truncated SVD (LSA).

    Returns
    -------
    reduced : dense ndarray  (N, n_components)
    svd     : fitted TruncatedSVD (kept for variance analysis)
    """
    svd     = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
    reduced = svd.fit_transform(tfidf_features)
    var_pct = svd.explained_variance_ratio_.sum() * 100
    print(f"[SVD]  TF-IDF  {tfidf_features.shape[1]}d → {n_components}d  "
          f"| variance retained: {var_pct:.2f}%")
    return reduced, svd


def reduce_embeddings(
    embeddings: np.ndarray,
    n_components: int = TARGET_N,
) -> tuple[np.ndarray, PCA]:
    """
    Compress dense embeddings (DistilBERT / GloVe) via PCA.

    Returns
    -------
    reduced : ndarray  (N, n_components)
    pca     : fitted PCA (kept for variance analysis / inverse_transform)
    """
    n_comp_actual = min(n_components, embeddings.shape[1])
    pca     = PCA(n_components=n_comp_actual, random_state=RANDOM_SEED)
    reduced = pca.fit_transform(embeddings)
    var_pct = pca.explained_variance_ratio_.sum() * 100
    print(f"[PCA]  Embeddings  {embeddings.shape[1]}d → {n_comp_actual}d  "
          f"| variance retained: {var_pct:.2f}%")
    return reduced, pca


def compute_variance_curves(
    tfidf_features,
    distilbert_embeddings: np.ndarray,
    glove_embeddings: Optional[np.ndarray] = None,
    max_components: int = 300,
) -> dict:
    """
    Compute *cumulative* explained-variance curves for the scree plot.

    Fits one max-rank decomposition per feature space, then takes
    the cumulative sum of the explained-variance ratios.

    Returns
    -------
    dict with keys:
      'n_components'    : list of ints  (x-axis values)
      'tfidf_svd'       : cumulative % variance for TF-IDF SVD
      'distilbert_pca'  : cumulative % variance for DistilBERT PCA
      'glove_pca'       : cumulative % variance for GloVe PCA  (if provided)
    """
    print("[Variance] Computing variance curves …")

    # ── TF-IDF: Truncated SVD ─────────────────────────────────────────────────
    max_svd = min(max_components, min(tfidf_features.shape) - 1)
    svd_full = TruncatedSVD(n_components=max_svd, random_state=RANDOM_SEED)
    svd_full.fit(tfidf_features)
    tfidf_cumvar = np.cumsum(svd_full.explained_variance_ratio_) * 100

    # ── DistilBERT: PCA ───────────────────────────────────────────────────────
    max_bert = min(max_components, distilbert_embeddings.shape[1])
    pca_bert = PCA(n_components=max_bert, random_state=RANDOM_SEED)
    pca_bert.fit(distilbert_embeddings)
    bert_cumvar = np.cumsum(pca_bert.explained_variance_ratio_) * 100

    n_axis = list(range(1, min(max_svd, max_bert) + 1))
    curves: dict = {
        "n_components":   n_axis,
        "tfidf_svd":      tfidf_cumvar[:len(n_axis)].tolist(),
        "distilbert_pca": bert_cumvar[:len(n_axis)].tolist(),
    }

    # ── GloVe: PCA (optional) ─────────────────────────────────────────────────
    if glove_embeddings is not None:
        max_glove = glove_embeddings.shape[1]   # 100 for glove-wiki-gigaword-100
        pca_glove = PCA(n_components=max_glove, random_state=RANDOM_SEED)
        pca_glove.fit(glove_embeddings)
        glove_cumvar = np.cumsum(pca_glove.explained_variance_ratio_) * 100
        # extend / truncate to match n_axis length
        padded = np.full(len(n_axis), glove_cumvar[-1])
        padded[:max_glove] = glove_cumvar
        curves["glove_pca"] = padded.tolist()
        print(f"  GloVe PCA   at {max_glove:3d} comps: {glove_cumvar[-1]:.1f}%")

    idx100 = min(99, len(n_axis) - 1)
    print(f"  TF-IDF SVD  at 100 comps: {tfidf_cumvar[idx100]:.1f}%")
    print(f"  DistilBERT  at 100 comps: {bert_cumvar[idx100]:.1f}%")
    return curves
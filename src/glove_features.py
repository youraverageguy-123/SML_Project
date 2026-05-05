"""
glove_features.py  (BONUS — explicitly listed as optional in the brief)
Generates mean-pooled GloVe word embeddings as a static dense feature space.

Motivation
----------
Adding GloVe creates a three-way comparison that separates two distinct axes:

  Sparse (BoW/TF-IDF)  →  Static dense (GloVe)  →  Contextual dense (DistilBERT)
       |                         |                          |
  No semantics            Word-level semantics       Token-level context

This lets us argue precisely that DistilBERT's advantage comes from *contextual*
understanding, not merely from using dense vectors.

Usage
-----
embeddings = generate_glove_features(texts)
# Returns ndarray (N, 100) or None if download fails
"""

from __future__ import annotations

import os
import numpy as np
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────
# glove-wiki-gigaword-100: ~128 MB, 400 k vocabulary, 100-dimensional vectors
# Trained on Wikipedia 2014 + Gigaword 5 (6B tokens).
GLOVE_MODEL = "glove-wiki-gigaword-100"
GLOVE_DIM   = 100
CACHE_PATH  = "data/glove_embeddings.npy"


# ── Public API ───────────────────────────────────────────────────────────────

def generate_glove_features(
    texts: list[str],
    cache_path: str = CACHE_PATH,
    model_name: str = GLOVE_MODEL,
) -> Optional[np.ndarray]:
    """
    Return mean-pooled GloVe embeddings for every article in `texts`.

    Fast path:  loads from cache_path if the .npy file already exists.
    Slow path:  downloads model via gensim (≈128 MB, one-time cost) then
                computes and caches embeddings.

    Parameters
    ----------
    texts      : list of raw article strings
    cache_path : where to save / load the cached .npy file
    model_name : gensim model key (default: glove-wiki-gigaword-100)

    Returns
    -------
    ndarray of shape (N, 100), or None if gensim is unavailable.
    """
    # ── Cache hit ─────────────────────────────────────────────────────────────
    if os.path.exists(cache_path):
        embeddings = np.load(cache_path)
        print(f"[GloVe] Cache hit — loaded {embeddings.shape} from {cache_path}")
        return embeddings

    # ── Download via gensim ───────────────────────────────────────────────────
    try:
        import gensim.downloader as api
        print(f"[GloVe] Downloading '{model_name}' via gensim (~128 MB, one-time) …")
        model = api.load(model_name)
        dim   = model.vector_size
        print(f"[GloVe] Model loaded. dim={dim}. Computing mean-pool embeddings …")

        embeddings = _mean_pool(texts, model, dim)
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, embeddings)
        print(f"[GloVe] Saved to {cache_path}.  shape={embeddings.shape}")
        return embeddings

    except Exception as exc:
        print(f"[GloVe] ⚠  Could not load GloVe: {exc}")
        print("[GloVe]    Install with:  pip install gensim")
        print("[GloVe]    GloVe plots will be skipped.")
        return None


# ── Internal helpers ─────────────────────────────────────────────────────────

def _mean_pool(
    texts: list[str],
    model,
    dim: int,
) -> np.ndarray:
    """
    Represent each text as the arithmetic mean of its known GloVe word vectors.
    Words not in the GloVe vocabulary are silently skipped.
    Documents with zero known words receive a zero vector.
    """
    embeddings = np.zeros((len(texts), dim), dtype=np.float32)
    n_oov_docs = 0

    for i, text in enumerate(texts):
        tokens  = text.lower().split()
        vectors = [model[w] for w in tokens if w in model]
        if vectors:
            embeddings[i] = np.mean(vectors, axis=0)
        else:
            n_oov_docs += 1

    coverage = 1.0 - n_oov_docs / len(texts)
    print(f"[GloVe] Document coverage: {coverage*100:.1f}%  "
          f"({n_oov_docs} docs had no known tokens)")
    return embeddings
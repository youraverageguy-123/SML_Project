"""
sparse_features.py
Generates sparse text feature representations: Bag-of-Words and TF-IDF.

Fixes from v1:
  - TF-IDF ngram_range changed from (2,2) → (1,2)   [BUG FIX]
  - Added sublinear_tf=True (log(1+tf) normalisation — standard NLP practice)
  - Added min_df=2 to prune hapax legomena
  - token_pattern now applied consistently to both vectorisers
  - Returns fitted vectorisers for downstream use (word clouds, feature importance)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from datasets import load_dataset

# ── Constants ────────────────────────────────────────────────────────────────
VOCAB_SIZE    = 5_000
RANDOM_SEED   = 42
N_SAMPLES     = 8_000
# Only alphabetic tokens of length ≥ 3 (reduces noise from numbers / symbols)
TOKEN_PATTERN = r"(?u)\b[a-zA-Z]{3,}\b"


# ── Public API ───────────────────────────────────────────────────────────────

def get_raw_text() -> tuple[list[str], np.ndarray]:
    """
    Load N_SAMPLES examples from AG News via Hugging Face datasets.

    Returns
    -------
    texts  : list of raw article strings
    labels : integer ndarray  (0=World, 1=Sports, 2=Business, 3=Sci/Tech)
    """
    print("[Data] Loading AG News from Hugging Face …")
    dataset = load_dataset("ag_news", split="train")
    dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(N_SAMPLES))
    df      = pd.DataFrame(dataset)
    texts   = df["text"].tolist()
    labels  = df["label"].values
    unique, counts = np.unique(labels, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"[Data] {len(texts)} samples loaded.  Distribution: {dist}")
    return texts, labels


def generate_sparse_features(
    texts: list[str],
) -> tuple:
    """
    Build Bag-of-Words (unigram) and TF-IDF (unigram + bigram) feature matrices.

    Parameters
    ----------
    texts : list of raw article strings

    Returns
    -------
    bow_features   : sparse CSR matrix  (N, VOCAB_SIZE)
    tfidf_features : sparse CSR matrix  (N, VOCAB_SIZE)
    bow_vec        : fitted CountVectorizer  (for word clouds / EDA)
    tfidf_vec      : fitted TfidfVectorizer
    """
    # ── Unigram Bag-of-Words ─────────────────────────────────────────────────
    bow_vec = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 1),
        max_features=VOCAB_SIZE,
        min_df=2,
        token_pattern=TOKEN_PATTERN,
    )
    bow_features = bow_vec.fit_transform(texts)

    # ── Unigram + Bigram TF-IDF  (FIXED: was ngram_range=(2,2)) ─────────────
    # sublinear_tf=True applies log(1 + tf), reducing the weight of very
    # frequent terms — standard practice in text classification.
    tfidf_vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),       # unigrams AND bigrams
        max_features=VOCAB_SIZE,
        min_df=2,
        sublinear_tf=True,
        token_pattern=TOKEN_PATTERN,
    )
    tfidf_features = tfidf_vec.fit_transform(texts)

    _log_stats("Bag-of-Words (unigram)",       bow_features)
    _log_stats("TF-IDF (unigram + bigram)",    tfidf_features)
    return bow_features, tfidf_features, bow_vec, tfidf_vec


# ── Internal helpers ─────────────────────────────────────────────────────────

def _log_stats(name: str, X) -> None:
    density = X.nnz / (X.shape[0] * X.shape[1])
    print(f"  [{name:30s}]  shape={X.shape}  density={density:.5f}")
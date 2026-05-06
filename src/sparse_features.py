from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from datasets import load_dataset
VOCAB_SIZE    = 5_000
RANDOM_SEED   = 42
N_SAMPLES     = 8_000
TOKEN_PATTERN = r"(?u)\b[a-zA-Z]{3,}\b"

def get_raw_text() -> tuple[list[str], np.ndarray]:
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
    bow_vec = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 1),
        max_features=VOCAB_SIZE,
        min_df=2,
        token_pattern=TOKEN_PATTERN,
    )
    bow_features = bow_vec.fit_transform(texts)
    tfidf_vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=VOCAB_SIZE,
        min_df=2,
        sublinear_tf=True,
        token_pattern=TOKEN_PATTERN,
    )
    tfidf_features = tfidf_vec.fit_transform(texts)
    _log_stats("Bag-of-Words (unigram)",       bow_features)
    _log_stats("TF-IDF (unigram + bigram)",    tfidf_features)
    return bow_features, tfidf_features, bow_vec, tfidf_vec

def _log_stats(name: str, X) -> None:
    density = X.nnz / (X.shape[0] * X.shape[1])
    print(f"  [{name:30s}]  shape={X.shape}  density={density:.5f}")
from __future__ import annotations

import os
import numpy as np
from typing import Optional
GLOVE_MODEL = "glove-wiki-gigaword-100"
GLOVE_DIM   = 100
CACHE_PATH  = "data/glove_embeddings.npy"
def generate_glove_features(
    texts: list[str],
    cache_path: str = CACHE_PATH,
    model_name: str = GLOVE_MODEL,
) -> Optional[np.ndarray]:
    if os.path.exists(cache_path):
        embeddings = np.load(cache_path)
        print(f"[GloVe] Cache hit — loaded {embeddings.shape} from {cache_path}")
        return embeddings
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

def _mean_pool(
    texts: list[str],
    model,
    dim: int,
) -> np.ndarray:
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
from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import learning_curve
LABEL_NAMES  = ["World", "Sports", "Business", "Sci/Tech"]
RANDOM_SEED  = 42
N_CLASSES    = 4
def _class_centroid(X, y: np.ndarray, class_idx: int) -> np.ndarray:
    """Return the mean vector for samples of class `class_idx`."""
    mask = y == class_idx
    subset = X[mask]
    if issparse(subset):
        return np.asarray(subset.mean(axis=0)).flatten()
    return subset.mean(axis=0)

def centroid_cosine_sim_matrices(
    feature_sets: dict[str, any],
    y: np.ndarray,
) -> dict[str, np.ndarray]:
    print("\n[Analysis] Class-centroid cosine similarity …")
    print(f"  {'Feature space':45s}  avg off-diagonal")
    print("  " + "─" * 60)
    sim_matrices: dict[str, np.ndarray] = {}
    for name, X in feature_sets.items():
        centroids = np.array([_class_centroid(X, y, i) for i in range(N_CLASSES)])
        # L2-normalise before cosine similarity
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        centroids_normed = centroids / norms
        sim = cosine_similarity(centroids_normed)
        sim_matrices[name] = sim
        off_mask  = ~np.eye(N_CLASSES, dtype=bool)
        avg_off   = sim[off_mask].mean()
        print(f"  {name:45s}  {avg_off:.4f}")
    return sim_matrices

def get_lr_top_features(
    bow_features,
    y: np.ndarray,
    vectorizer,
    n_top: int = 40,
) -> dict[str, list[tuple[str, float]]]:
    print("[Analysis] Fitting LR for feature importance (word clouds) …")
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
    model.fit(bow_features, y)
    vocab = vectorizer.get_feature_names_out()
    top_features: dict[str, list[tuple[str, float]]] = {}
    for i, label in enumerate(LABEL_NAMES):
        coefs    = model.coef_[i]
        top_idx  = coefs.argsort()[-n_top:][::-1]
        # Keep only positive coefficients (push toward this class)
        top_features[label] = [
            (vocab[j], float(coefs[j]))
            for j in top_idx
            if coefs[j] > 0
        ]
    return top_features

def compute_learning_curve_data(
    X,
    y: np.ndarray,
    model_name: str,
    n_cv: int = 3,
    n_points: int = 8,
) -> dict:
    from .classical_models import build_model
    model = build_model(model_name)

    train_fracs = np.linspace(0.05, 1.0, n_points)
    sizes, tr_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_fracs,
        cv=n_cv,
        scoring="accuracy",
        n_jobs=1,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    return {
        "train_sizes": sizes,
        "mean_train":  tr_scores.mean(axis=1),
        "std_train":   tr_scores.std(axis=1),
        "mean_val":    val_scores.mean(axis=1),
        "std_val":     val_scores.std(axis=1),
    }
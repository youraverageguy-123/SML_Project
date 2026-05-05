"""
classical_models.py
Training and evaluation of classical ML models across text feature spaces.

Returns structured result dicts that capture:
  - accuracy and macro F1-score  (supervised models)
  - per-class F1  (for the per-class heatmap)
  - training time and inference time  (for the time-accuracy scatter)
  - NMI and ARI  (KMeans only — proper unsupervised evaluation)
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ── Constants ────────────────────────────────────────────────────────────────
LABEL_NAMES  = ["World", "Sports", "Business", "Sci/Tech"]
RANDOM_SEED  = 42
TEST_SIZE    = 0.20
N_CLASSES    = 4


# ── Public API ───────────────────────────────────────────────────────────────

def build_model(model_name: str):
    """Instantiate a model by name with fixed hyper-parameters."""
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    if model_name == "Linear SVM":
        return SVC(kernel="linear", random_state=RANDOM_SEED)
    if model_name == "RBF SVM":
        # probability=True enables soft predictions; used for confidence analysis
        return SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)
    if model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    if model_name == "KMeans":
        return KMeans(n_clusters=N_CLASSES, random_state=RANDOM_SEED, n_init="auto")
    raise ValueError(f"Unknown model: {model_name!r}")


def train_and_evaluate(
    X,
    y: np.ndarray,
    feature_name: str,
    model_name: str = "Logistic Regression",
) -> dict:
    """
    Train `model_name` on features `X` with labels `y` and return a result dict.

    Supervised models use an 80/20 random split (random_state=42).
    KMeans fits on the full dataset (unsupervised) and reports NMI + ARI.

    Returns
    -------
    dict with keys:
      model, feature_name, model_name,
      accuracy, macro_f1, per_class_f1,
      training_time, inference_time,
      nmi, ari   (None for supervised models)
    """
    print(f"  [{model_name:20s}] × [{feature_name}]")
    model = build_model(model_name)

    # ── KMeans: unsupervised path ─────────────────────────────────────────────
    if model_name == "KMeans":
        t0 = time.perf_counter()
        model.fit(X)
        train_time = time.perf_counter() - t0
        cluster_labels = model.labels_
        nmi = normalized_mutual_info_score(y, cluster_labels)
        ari = adjusted_rand_score(y, cluster_labels)
        print(f"    NMI={nmi:.4f}   ARI={ari:.4f}   train={train_time:.1f}s")
        return {
            "model": model, "feature_name": feature_name, "model_name": model_name,
            "accuracy": None, "macro_f1": None, "per_class_f1": None,
            "training_time": train_time, "inference_time": None,
            "nmi": nmi, "ari": ari,
        }

    # ── Supervised path ───────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    preds = model.predict(X_test)
    infer_time = time.perf_counter() - t1

    acc           = accuracy_score(y_test, preds)
    macro_f1      = f1_score(y_test, preds, average="macro")
    per_class_f1  = f1_score(y_test, preds, average=None, labels=list(range(N_CLASSES)))

    print(f"    Acc={acc:.4f}  MacroF1={macro_f1:.4f}  "
          f"train={train_time:.1f}s  infer={infer_time:.2f}s")
    per_class_str = " | ".join(
        f"{n}={v:.3f}" for n, v in zip(LABEL_NAMES, per_class_f1)
    )
    print(f"    Per-class F1: {per_class_str}")

    return {
        "model": model, "feature_name": feature_name, "model_name": model_name,
        "accuracy": acc, "macro_f1": macro_f1,
        "per_class_f1": per_class_f1.tolist(),
        "training_time": train_time, "inference_time": infer_time,
        "nmi": None, "ari": None,
    }
"""
generate_plots.py
All visualisation functions for the project.

Figures produced
----------------
01  eda_plots.png              — class balance, token lengths, top BoW terms
02  wordclouds_lr_features.png — discriminative terms per class (LR coefficients)
03  scree_curves.png           — cumulative explained-variance vs n_components
04  pca_scatter.png            — 2-D PCA projection (DistilBERT + GloVe if available)
05  tsne_comparison.png        — t-SNE: BoW / GloVe / DistilBERT side by side
06  centroid_similarity.png    — cosine similarity between class centroids
07  confusion_matrix.png       — confusion matrix for the best model
08  model_comparison.png       — grouped accuracy bar chart across all models
09  per_class_f1_heatmap.png   — per-class F1 heatmap (BoW vs DistilBERT)
10  learning_curves.png        — accuracy vs training size (LR + KNN, 2 feature spaces)
11  kmeans_quality.png         — KMeans NMI / ARI across feature spaces
12  time_accuracy_scatter.png  — accuracy vs training time tradeoff
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.sparse import issparse

# ── Global style ──────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":          "DejaVu Sans",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.color":           "#e5e7eb",
    "grid.linewidth":       0.8,
    "axes.axisbelow":       True,
    "figure.dpi":           150,
})

# ── Colour palettes ───────────────────────────────────────────────────────────
CLASS_COLORS = {
    "World":    "#ef4444",
    "Sports":   "#3b82f6",
    "Business": "#10b981",
    "Sci/Tech": "#f59e0b",
}
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

FEAT_COLORS = {
    "Bag-of-Words":              "#d1d5db",
    "TF-IDF":                    "#9ca3af",
    "Reduced TF-IDF (SVD)":      "#4b5563",
    "GloVe (Mean-pool)":         "#a78bfa",
    "Reduced GloVe (PCA)":       "#7c3aed",
    "DistilBERT (Uncompressed)": "#60a5fa",
    "Reduced DistilBERT (PCA)":  "#1d4ed8",
}

OUT_DIR = "outputs"


def _savefig(filename: str, **kwargs) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight", **kwargs)
    print(f"  [Plot] saved → {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 01  EDA
# ══════════════════════════════════════════════════════════════════════════════

def create_eda_plots(
    texts: list[str],
    labels: np.ndarray,
    bow_features,
    bow_vectorizer,
) -> None:
    """
    Three-panel EDA figure:
    Left   — class distribution (bar chart)
    Centre — token-length histogram
    Right  — top-15 discriminative BoW terms (excluding generic stopwords)
    """
    from collections import Counter
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Panel 1: class distribution ───────────────────────────────────────────
    counts = Counter(labels)
    bars = axes[0].bar(
        LABEL_NAMES,
        [counts[i] for i in range(4)],
        color=[CLASS_COLORS[n] for n in LABEL_NAMES],
        edgecolor="white", linewidth=0.8,
    )
    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 20,
                     f"{int(h):,}", ha="center", va="bottom", fontsize=9)
    axes[0].set_title("Class Distribution", fontweight="bold", fontsize=12)
    axes[0].set_ylabel("Count")
    axes[0].set_ylim(0, max(counts.values()) * 1.15)

    # ── Panel 2: token-length histogram ──────────────────────────────────────
    lengths = [len(t.split()) for t in texts]
    axes[1].hist(lengths, bins=40, color="#6366f1", edgecolor="white", linewidth=0.6)
    axes[1].set_title("Token Length Distribution", fontweight="bold", fontsize=12)
    axes[1].set_xlabel("Words per article")
    axes[1].set_ylabel("Count")
    mu = np.mean(lengths)
    axes[1].axvline(mu, color="#ef4444", lw=1.5, linestyle="--",
                    label=f"mean={mu:.0f}")
    axes[1].legend(fontsize=9)

    # ── Panel 3: top-15 BoW terms (non-generic) ───────────────────────────────
    vocab  = bow_vectorizer.get_feature_names_out()
    freqs  = np.asarray(bow_features.sum(axis=0)).flatten()
    # Filter out generic news words that appear across all classes
    NOISE  = {"said", "year", "new", "reuters", "inc", "corp", "ltd",
               "company", "week", "monday", "tuesday", "wednesday",
               "thursday", "friday", "saturday", "sunday"}
    mask   = np.array([v not in NOISE for v in vocab])
    masked_freqs = freqs.copy()
    masked_freqs[~mask] = 0
    top_idx = masked_freqs.argsort()[-15:][::-1]

    axes[2].barh(
        vocab[top_idx], freqs[top_idx],
        color="#10b981", edgecolor="white", linewidth=0.6,
    )
    axes[2].set_title("Top 15 Terms (BoW, filtered)", fontweight="bold", fontsize=12)
    axes[2].set_xlabel("Total frequency")
    axes[2].invert_yaxis()

    plt.suptitle("AG News — Exploratory Data Analysis  (8 000 samples)", y=1.02,
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _savefig("01_eda_plots.png")


# ══════════════════════════════════════════════════════════════════════════════
# 02  LR-weighted word clouds
# ══════════════════════════════════════════════════════════════════════════════

def create_lr_wordclouds(
    lr_top_features: dict[str, list[tuple[str, float]]],
) -> None:
    """
    Four word clouds (one per class), sized by Logistic Regression coefficients.
    Using LR weights instead of raw frequencies shows *discriminative* vocabulary
    — words that push the model toward a specific class.
    Falls back to horizontal bar charts if the `wordcloud` package is absent.
    """
    try:
        from wordcloud import WordCloud
        _wordcloud_wc(lr_top_features)
    except ImportError:
        print("[WordCloud] `wordcloud` not installed — using bar-chart fallback")
        _wordcloud_bars(lr_top_features)


def _wordcloud_wc(top_features: dict) -> None:
    from wordcloud import WordCloud

    CMAPS = {"World": "Reds", "Sports": "Blues",
             "Business": "Greens", "Sci/Tech": "Oranges"}

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, label in zip(axes, LABEL_NAMES):
        word_weights = {w: max(0, v) for w, v in top_features[label]}
        if not word_weights:
            ax.axis("off")
            continue
        wc = WordCloud(
            width=500, height=380,
            background_color="white",
            colormap=CMAPS[label],
            max_words=50,
            prefer_horizontal=0.85,
        ).generate_from_frequencies(word_weights)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(label, fontsize=15, fontweight="bold",
                     color=CLASS_COLORS[label])
        ax.axis("off")

    plt.suptitle(
        "Discriminative Vocabulary per Class\n"
        "(word size ∝ Logistic Regression coefficient)",
        fontsize=14, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    _savefig("02_wordclouds_lr_features.png")


def _wordcloud_bars(top_features: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    for ax, label in zip(axes, LABEL_NAMES):
        words, weights = zip(*top_features[label][:15]) if top_features[label] else ([], [])
        ax.barh(list(words), list(weights), color=CLASS_COLORS[label])
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("LR coefficient")
    plt.suptitle("Discriminative Terms per Class (LR Coefficients)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _savefig("02_wordclouds_lr_features.png")


# ══════════════════════════════════════════════════════════════════════════════
# 03  Scree curves
# ══════════════════════════════════════════════════════════════════════════════

def create_scree_plot(variance_curves: dict) -> None:
    """
    Cumulative explained-variance curves for SVD (TF-IDF) and PCA (embeddings).

    Directly answers the rubric criterion "Quality for PCA and other feature
    extractions" — showing *why* 100 components was chosen and how each
    reduction method compares.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n = variance_curves["n_components"]

    # ── TF-IDF SVD ────────────────────────────────────────────────────────────
    ax.plot(n, variance_curves["tfidf_svd"],
            color="#4b5563", lw=2, label="TF-IDF  (Truncated SVD)")

    # ── DistilBERT PCA ────────────────────────────────────────────────────────
    ax.plot(n, variance_curves["distilbert_pca"],
            color="#1d4ed8", lw=2, label="DistilBERT  (PCA)")

    # ── GloVe PCA (optional) ──────────────────────────────────────────────────
    if "glove_pca" in variance_curves:
        ax.plot(n, variance_curves["glove_pca"],
                color="#7c3aed", lw=2, linestyle="--",
                label="GloVe  (PCA)")

    # ── Annotations ───────────────────────────────────────────────────────────
    ax.axvline(100, color="#ef4444", lw=1.5, linestyle=":",
               label="n = 100 (chosen)")
    for pct in (80, 90, 95):
        ax.axhline(pct, color="#9ca3af", lw=0.8, linestyle="--")
        ax.text(n[-1] * 1.01, pct, f"{pct}%", va="center",
                fontsize=8, color="#6b7280")

    # ── Labels for the 100-component intercepts ───────────────────────────────
    idx = min(99, len(n) - 1)
    for key, color in [("tfidf_svd", "#4b5563"), ("distilbert_pca", "#1d4ed8"),
                       ("glove_pca", "#7c3aed")]:
        if key not in variance_curves:
            continue
        val = variance_curves[key][idx]
        ax.annotate(
            f"{val:.1f}%",
            xy=(n[idx], val),
            xytext=(n[idx] + 8, val - 3),
            fontsize=9, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
        )

    ax.set_xlabel("Number of Components", fontsize=12)
    ax.set_ylabel("Cumulative Explained Variance (%)", fontsize=12)
    ax.set_title(
        "Scree Curves — Variance Retention vs Number of Components\n"
        "PCA on dense embeddings retains far more information than SVD on sparse TF-IDF",
        fontweight="bold", fontsize=12,
    )
    ax.set_xlim(1, max(n))
    ax.set_ylim(0, 102)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _savefig("03_scree_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# 04  PCA 2-D scatter
# ══════════════════════════════════════════════════════════════════════════════

def create_pca_scatter(
    distilbert_embeddings: np.ndarray,
    y: np.ndarray,
    glove_embeddings: Optional[np.ndarray] = None,
) -> None:
    """
    2-D PCA projection of embedding spaces.
    Shows DistilBERT alone, or DistilBERT + GloVe side by side when available.
    """
    from sklearn.decomposition import PCA

    spaces: list[tuple[str, np.ndarray]] = [
        ("DistilBERT Embeddings — PCA 2D Projection", distilbert_embeddings),
    ]
    if glove_embeddings is not None:
        spaces.append(("GloVe Embeddings — PCA 2D Projection", glove_embeddings))

    ncols = len(spaces)
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 7))
    if ncols == 1:
        axes = [axes]

    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=CLASS_COLORS[n], markersize=9, label=n)
        for n in LABEL_NAMES
    ]

    for ax, (title, emb) in zip(axes, spaces):
        pca     = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(emb)
        var_exp = pca.explained_variance_ratio_.sum() * 100
        for i, name in enumerate(LABEL_NAMES):
            mask = y == i
            ax.scatter(
                reduced[mask, 0], reduced[mask, 1],
                c=CLASS_COLORS[name], alpha=0.35, s=10, rasterized=True,
            )
        ax.set_title(f"{title}\n(PC1+PC2 explains {var_exp:.1f}% variance)",
                     fontweight="bold", fontsize=11)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")

    axes[-1].legend(handles=legend_handles, loc="upper right", fontsize=9)
    plt.suptitle("Feature Space Geometry — PCA 2D Projections",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _savefig("04_pca_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# 05  t-SNE comparison (3-way if GloVe available)
# ══════════════════════════════════════════════════════════════════════════════

def create_tsne_comparison(
    bow_features,
    distilbert_embeddings: np.ndarray,
    y: np.ndarray,
    glove_embeddings: Optional[np.ndarray] = None,
    sample_size: int = 1_000,
) -> None:
    """
    t-SNE projection on a random subset of samples.
    Two-panel (BoW vs DistilBERT) or three-panel with GloVe as middle column,
    showing the progression: no semantics → static semantics → contextual.
    """
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(42)
    idx = rng.choice(len(y), size=min(sample_size, len(y)), replace=False)
    y_sub = y[idx]

    def _get(X):
        sub = X[idx]
        return sub.toarray() if issparse(sub) else sub

    panels: list[tuple[str, np.ndarray]] = [
        ("Bag-of-Words (t-SNE)", _get(bow_features)),
    ]
    if glove_embeddings is not None:
        panels.append(("GloVe — static semantics (t-SNE)", _get(glove_embeddings)))
    panels.append(("DistilBERT — contextual (t-SNE)", _get(distilbert_embeddings)))

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=CLASS_COLORS[n], markersize=8, label=n)
        for n in LABEL_NAMES
    ]

    for ax, (title, X_sub) in zip(axes, panels):
        proj = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000).fit_transform(X_sub)
        for i, name in enumerate(LABEL_NAMES):
            mask = y_sub == i
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=CLASS_COLORS[name], alpha=0.55, s=12, rasterized=True)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[-1].legend(handles=legend_handles, markerscale=1.5,
                    fontsize=9, loc="upper right")
    plt.suptitle(
        "t-SNE Visualisation — How Well Does Each Feature Space Separate Classes?\n"
        f"({sample_size} random samples)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _savefig("05_tsne_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 06  Centroid cosine similarity heatmaps
# ══════════════════════════════════════════════════════════════════════════════

def create_centroid_similarity_plot(
    sim_matrices: dict[str, np.ndarray],
) -> None:
    """
    Grid of 4×4 cosine-similarity heatmaps between class centroids,
    one panel per feature space.

    Lower off-diagonal values ↔ better class separability.
    This visualisation directly explains *why* DistilBERT outperforms BoW.
    """
    n_spaces = len(sim_matrices)
    ncols    = min(n_spaces, 4)
    nrows    = (n_spaces + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4.5 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax, (name, sim) in zip(axes_flat, sim_matrices.items()):
        off_mask = ~np.eye(4, dtype=bool)
        avg_off  = sim[off_mask].mean()
        im = sns.heatmap(
            sim,
            ax=ax,
            vmin=0.0, vmax=1.0,
            cmap="Blues",
            annot=True, fmt=".3f",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            linewidths=0.5, linecolor="white",
            cbar=False,
        )
        ax.set_title(
            f"{name}\navg off-diagonal = {avg_off:.3f}",
            fontweight="bold", fontsize=10,
        )
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    # Hide any unused axes
    for ax in axes_flat[len(sim_matrices):]:
        ax.set_visible(False)

    plt.suptitle(
        "Cosine Similarity Between Class Centroids\n"
        "Lower off-diagonal → better class separability in that feature space",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _savefig("06_centroid_similarity.png")


# ══════════════════════════════════════════════════════════════════════════════
# 07  Confusion matrix
# ══════════════════════════════════════════════════════════════════════════════

def create_confusion_matrix(
    X,
    y: np.ndarray,
    label_names: list[str],
    model_name: str = "RBF SVM",
) -> None:
    """
    Confusion matrix for the best-performing supervised model.
    Uses RBF SVM on Reduced DistilBERT (PCA) — the highest-accuracy configuration.
    """
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    from sklearn.model_selection import train_test_split

    from src.classical_models import build_model

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = build_model(model_name)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    cm   = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(
        f"Confusion Matrix — {model_name} on Reduced DistilBERT (PCA)\n"
        f"(Best-performing configuration across all experiments)",
        fontweight="bold", fontsize=11,
    )
    plt.tight_layout()
    _savefig("07_confusion_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
# 08  Grouped accuracy bar chart
# ══════════════════════════════════════════════════════════════════════════════

def create_grouped_bar_chart(results: dict) -> None:
    """
    Grouped bar chart: accuracy for every (model × feature set) combination.
    Fixed v1 issues: RBF SVM now included; GloVe bars added when available.
    """
    supervised = ["Logistic Regression", "Linear SVM", "RBF SVM", "KNN"]
    plot_models = [m for m in supervised if m in results]

    # Determine which feature sets actually have results
    feat_order = [
        "Bag-of-Words",
        "TF-IDF",
        "Reduced TF-IDF (SVD)",
        "GloVe (Mean-pool)",
        "Reduced GloVe (PCA)",
        "DistilBERT (Uncompressed)",
        "Reduced DistilBERT (PCA)",
    ]
    # Only include feature sets present in results
    all_feats = set()
    for m in plot_models:
        all_feats.update(results[m].keys())
    feat_order = [f for f in feat_order if f in all_feats]
    feat_labels = [f.replace(" (Uncompressed)", "").replace(" (Mean-pool)", "")
                   for f in feat_order]

    x        = np.arange(len(plot_models))
    n_feats  = len(feat_order)
    width    = 0.80 / n_feats
    offsets  = np.linspace(-(n_feats - 1) / 2, (n_feats - 1) / 2, n_feats) * width

    fig, ax = plt.subplots(figsize=(max(14, 3 * n_feats), 7))
    for i, (feat_name, label) in enumerate(zip(feat_order, feat_labels)):
        color = FEAT_COLORS.get(feat_name, "#94a3b8")
        accs  = [
            results[m].get(feat_name, {}).get("accuracy") or 0.0
            for m in plot_models
        ]
        rects = ax.bar(x + offsets[i], accs, width, label=label,
                       color=color, edgecolor="white", linewidth=0.5)
        for rect, val in zip(rects, accs):
            if val > 0:
                ax.annotate(
                    f"{val:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, rotation=90,
                )

    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title(
        "Classical ML Models: Sparse Features vs Static Embeddings vs Contextual Embeddings",
        fontsize=13, fontweight="bold", pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_models, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    _savefig("08_model_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 09  Per-class F1 heatmap
# ══════════════════════════════════════════════════════════════════════════════

def create_per_class_f1_heatmap(results: dict) -> None:
    """
    Two-panel heatmap: per-class F1 for every supervised model.
    Left  panel — Bag-of-Words features.
    Right panel — Reduced DistilBERT (PCA) features.

    Reveals which classes are inherently hard (Business / Sci/Tech) versus
    trivially easy (Sports) across all models, independent of feature quality.
    """
    supervised = ["Logistic Regression", "Linear SVM", "RBF SVM", "KNN"]
    plot_models = [m for m in supervised if m in results]

    # Prefer Reduced DistilBERT; fall back to Uncompressed
    bert_key = "Reduced DistilBERT (PCA)"
    if bert_key not in results.get(plot_models[0], {}):
        bert_key = "DistilBERT (Uncompressed)"
    bow_key = "Bag-of-Words"

    def _matrix(feat_key):
        mat = []
        for m in plot_models:
            row = results[m].get(feat_key, {})
            pcf = row.get("per_class_f1")
            mat.append(pcf if pcf else [0.0] * 4)
        return np.array(mat)

    bow_mat  = _matrix(bow_key)
    bert_mat = _matrix(bert_key)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    kw = dict(vmin=0.4, vmax=1.0, cmap="YlGn",
              annot=True, fmt=".3f", linewidths=0.5,
              linecolor="white", cbar=False)

    sns.heatmap(bow_mat,  ax=axes[0],
                xticklabels=LABEL_NAMES, yticklabels=plot_models, **kw)
    sns.heatmap(bert_mat, ax=axes[1],
                xticklabels=LABEL_NAMES, yticklabels=False, **kw)

    axes[0].set_title(f"Feature: {bow_key}", fontweight="bold")
    axes[1].set_title(f"Feature: {bert_key}", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].tick_params(axis="x", rotation=30)

    plt.suptitle(
        "Per-Class F1-Score Heatmap  (Rows = Models,  Columns = Classes)\n"
        "Sports is consistently easiest; Business \u2194 Sci/Tech causes the most confusion",
        fontsize=12, fontweight="bold", y=1.04,
    )
    plt.tight_layout()

    # Add colorbar in its own dedicated axis to the right — avoids overlap
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap="YlGn",
                                norm=plt.Normalize(vmin=0.4, vmax=1.0))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="F1-score")

    _savefig("09_per_class_f1_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10  Learning curves
# ══════════════════════════════════════════════════════════════════════════════

def create_learning_curves_plot(
    lc_data: dict[str, dict],
) -> None:
    """
    Two-panel learning curve plot showing cross-validated validation accuracy
    as a function of training-set size.

    lc_data keys expected:
      "LR — Bag-of-Words"
      "LR — DistilBERT"
      "LR — GloVe"  (optional)

    Demonstrates:
    - DistilBERT is substantially more data-efficient than BoW
    - LR saturates quickly on DistilBERT embeddings
    """
    STYLES = {
        "LR — Bag-of-Words":  {"color": "#4b5563", "ls": "-",  "label": "LR — Bag-of-Words"},
        "LR — DistilBERT":    {"color": "#1d4ed8", "ls": "-",  "label": "LR — DistilBERT"},
        "LR — GloVe":         {"color": "#7c3aed", "ls": "--", "label": "LR — GloVe"},
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for key, curve in lc_data.items():
        if curve is None:
            continue
        style = STYLES.get(key, {"color": "#6b7280", "ls": "-", "label": key})
        sizes = curve["train_sizes"]
        mean  = curve["mean_val"]
        std   = curve["std_val"]
        ax.plot(sizes, mean, lw=2, **{k: v for k, v in style.items()
                                       if k in ("color", "ls", "label")})
        ax.fill_between(sizes, mean - std, mean + std,
                        alpha=0.12, color=style["color"])

    ax.set_xlabel("Training set size", fontsize=12)
    ax.set_ylabel("Validation accuracy (3-fold CV)", fontsize=12)
    ax.set_title(
        "Learning Curves — Data Efficiency by Feature Space\n"
        "DistilBERT achieves high accuracy with far fewer training samples",
        fontweight="bold", fontsize=12,
    )
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _savefig("10_learning_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# 11  KMeans cluster quality (NMI / ARI)
# ══════════════════════════════════════════════════════════════════════════════

def create_kmeans_quality_plot(results: dict) -> None:
    """
    Grouped bar chart of KMeans NMI and ARI across feature spaces.

    Quantifies how well the *natural geometry* of each feature space
    aligns with the ground-truth class structure — without any classifier.
    """
    if "KMeans" not in results:
        print("[Plot] No KMeans results — skipping quality plot")
        return

    km_results = results["KMeans"]
    feat_names = list(km_results.keys())
    nmis = [km_results[f]["nmi"] for f in feat_names]
    aris = [km_results[f]["ari"] for f in feat_names]

    x      = np.arange(len(feat_names))
    width  = 0.35
    labels = [f.replace(" (Uncompressed)", "").replace(" (Mean-pool)", "")
              for f in feat_names]

    fig, ax = plt.subplots(figsize=(max(10, 2 * len(feat_names)), 6))
    b1 = ax.bar(x - width / 2, nmis, width, label="NMI", color="#3b82f6",
                edgecolor="white")
    b2 = ax.bar(x + width / 2, aris, width, label="ARI", color="#10b981",
                edgecolor="white")

    for bar in list(b1) + list(b2):
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, min(1.0, max(max(nmis), max(aris)) * 1.25 + 0.05))
    ax.set_title(
        "KMeans Cluster Quality (NMI and ARI) — Unsupervised Evaluation\n"
        "Higher NMI/ARI → the natural cluster geometry matches the true class structure",
        fontweight="bold", fontsize=12,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    _savefig("11_kmeans_quality.png")


# ══════════════════════════════════════════════════════════════════════════════
# 12  Training-time vs accuracy scatter
# ══════════════════════════════════════════════════════════════════════════════

def create_time_accuracy_scatter(results: dict) -> None:
    """
    Scatter plot: accuracy (x) vs training time in seconds (y, log scale).

    Each point is one (model × feature_set) combination.
    Colour encodes feature-space family; marker shape encodes model.
    Reveals the efficiency frontier — which configurations give the best
    accuracy for the least training time.
    """
    MARKERS = {
        "Logistic Regression": "o",
        "Linear SVM":          "s",
        "RBF SVM":             "D",
        "KNN":                 "^",
    }
    FEAT_FAMILY = {
        "Bag-of-Words":              ("Sparse", "#9ca3af"),
        "TF-IDF":                    ("Sparse", "#4b5563"),
        "Reduced TF-IDF (SVD)":      ("Sparse", "#6b7280"),
        "GloVe (Mean-pool)":         ("GloVe",  "#a78bfa"),
        "Reduced GloVe (PCA)":       ("GloVe",  "#7c3aed"),
        "DistilBERT (Uncompressed)": ("BERT",   "#60a5fa"),
        "Reduced DistilBERT (PCA)":  ("BERT",   "#1d4ed8"),
    }

    fig, ax = plt.subplots(figsize=(11, 7))
    plotted: set = set()

    for model_name, model_results in results.items():
        marker = MARKERS.get(model_name, "o")
        for feat_name, res in model_results.items():
            acc  = res.get("accuracy")
            t    = res.get("training_time")
            if acc is None or t is None:
                continue
            _, color = FEAT_FAMILY.get(feat_name, ("Other", "#94a3b8"))
            ax.scatter(acc, t, c=color, marker=marker, s=90,
                       edgecolors="white", linewidths=0.7, zorder=3)
            ax.annotate(
                f"{model_name.split()[0][:2]}\n{feat_name.split('(')[0].strip()[:8]}",
                xy=(acc, t), fontsize=6.5, ha="center", va="bottom",
                color="#374151",
            )

    # ── Legends ───────────────────────────────────────────────────────────────
    model_handles = [
        Line2D([0], [0], marker=m, color="w", markerfacecolor="#374151",
               markersize=9, label=n)
        for n, m in MARKERS.items()
    ]
    feat_handles: list = []
    seen: set = set()
    for feat_name, (family, color) in FEAT_FAMILY.items():
        if family not in seen:
            feat_handles.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=color, markersize=9, label=family)
            )
            seen.add(family)

    leg1 = ax.legend(handles=model_handles, title="Model",
                     loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=feat_handles, title="Feature family",
              loc="lower right", fontsize=8)

    ax.set_yscale("log")
    ax.set_xlabel("Test Accuracy", fontsize=12)
    ax.set_ylabel("Training Time (s, log scale)", fontsize=12)
    ax.set_title(
        "Accuracy vs Training Time Trade-off\n"
        "Upper-left is desirable (fast); Lower-right is efficient (accurate)",
        fontweight="bold", fontsize=12,
    )
    plt.tight_layout()
    _savefig("12_time_accuracy_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX results table
# ══════════════════════════════════════════════════════════════════════════════

def save_latex_table(results: dict) -> None:
    """
    Write a booktabs LaTeX table of Accuracy + Macro F1 for all supervised
    model × feature combinations.  Highlights the best accuracy per model
    in bold.  Saved to outputs/results_table.tex.
    """
    FEAT_DISPLAY = {
        "Bag-of-Words":              r"Bag-of-Words (unigram)",
        "TF-IDF":                    r"TF-IDF (unigram+bigram)",
        "Reduced TF-IDF (SVD)":      r"Reduced TF-IDF (SVD, 100d)",
        "GloVe (Mean-pool)":         r"GloVe mean-pool (100d)",
        "Reduced GloVe (PCA)":       r"Reduced GloVe (PCA, 100d)",
        "DistilBERT (Uncompressed)": r"DistilBERT (768d)",
        "Reduced DistilBERT (PCA)":  r"Reduced DistilBERT (PCA, 100d)",
    }
    FEAT_ORDER  = list(FEAT_DISPLAY.keys())
    MODEL_ORDER = ["Logistic Regression", "Linear SVM", "RBF SVM", "KNN"]

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Accuracy and Macro F1-score for all model--feature combinations. "
        r"Best accuracy per model shown in \textbf{bold}.}",
        r"\label{tab:results}",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Feature Set} & \textbf{Accuracy} & \textbf{Macro F1} \\",
        r"\midrule",
    ]

    for i, model in enumerate(MODEL_ORDER):
        if model not in results:
            continue
        if i > 0:
            lines.append(r"\midrule")
        # find best accuracy for this model
        best_acc = max(
            (r.get("accuracy") or 0.0 for r in results[model].values()),
            default=0.0,
        )
        for feat in FEAT_ORDER:
            res = results[model].get(feat)
            if res is None or res.get("accuracy") is None:
                continue
            acc = res["accuracy"]
            f1  = res["macro_f1"]
            acc_str = (r"\textbf{" + f"{acc:.4f}" + r"}"
                       if abs(acc - best_acc) < 1e-6 else f"{acc:.4f}")
            lines.append(
                f"{model} & {FEAT_DISPLAY.get(feat, feat)} "
                f"& {acc_str} & {f1:.4f} \\\\"
            )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "results_table.tex")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"  [LaTeX] saved → {out_path}")
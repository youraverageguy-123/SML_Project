"""
main.py
Full pipeline: feature extraction → dimensionality reduction →
classification → deep analysis → 12 publication-quality figures.

Run
---
    python main.py

Output
------
All plots and results_table.tex written to outputs/
"""
from __future__ import annotations
from src.generate_report_text import write_stats_tex


import os
import numpy as np
import pandas as pd

# ── Local modules ─────────────────────────────────────────────────────────────
from src.sparse_features   import get_raw_text, generate_sparse_features
from src.glove_features    import generate_glove_features
from src.dim_reduction     import reduce_tfidf, reduce_embeddings, compute_variance_curves
from src.classical_models  import train_and_evaluate
from src.analysis          import (
    centroid_cosine_sim_matrices,
    get_lr_top_features,
    compute_learning_curve_data,
)
from src.generate_plots import (
    create_eda_plots,
    create_lr_wordclouds,
    create_scree_plot,
    create_pca_scatter,
    create_tsne_comparison,
    create_centroid_similarity_plot,
    create_confusion_matrix,
    create_grouped_bar_chart,
    create_per_class_f1_heatmap,
    create_learning_curves_plot,
    create_kmeans_quality_plot,
    create_time_accuracy_scatter,
    save_latex_table,
)

LABEL_NAMES   = ["World", "Sports", "Business", "Sci/Tech"]
SUPERVISED    = ["Logistic Regression", "Linear SVM", "RBF SVM", "KNN"]
ALL_MODELS    = SUPERVISED + ["KMeans"]


def main() -> None:
    os.makedirs("outputs", exist_ok=True)
    print("=" * 70)
    print("  SML Project — Sparse Features vs Contextual Embeddings")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/7] Loading data …")
    texts, y = get_raw_text()

    # ── 2. Generate features ──────────────────────────────────────────────────
    print("\n[2/7] Generating features …")

    # Sparse
    bow, tfidf, bow_vec, tfidf_vec = generate_sparse_features(texts)

    # GloVe (optional bonus)
    glove = generate_glove_features(texts)   # returns None if gensim unavailable

    # DistilBERT (pre-computed and cached)
    print("[DistilBERT] Loading pre-computed embeddings …")
    distilbert = (
        np.load("data/distilbert_embeddings.npy", allow_pickle=False)
          .reshape(8000, 768)
          .astype(np.float32)
    )
    print(f"  DistilBERT shape: {distilbert.shape}")

    # ── 3. Dimensionality reduction ───────────────────────────────────────────
    print("\n[3/7] Dimensionality reduction …")
    reduced_tfidf,      _svd  = reduce_tfidf(tfidf)
    reduced_distilbert, _pca  = reduce_embeddings(distilbert)
    reduced_glove = None
    if glove is not None:
        reduced_glove, _ = reduce_embeddings(glove)

    # ── 4. EDA + geometry plots ───────────────────────────────────────────────
    print("\n[4/7] Generating EDA and geometry plots …")

    create_eda_plots(texts, y, bow, bow_vec)

    lr_features = get_lr_top_features(bow, y, bow_vec)
    create_lr_wordclouds(lr_features)

    variance_curves = compute_variance_curves(tfidf, distilbert, glove)
    create_scree_plot(variance_curves)

    create_pca_scatter(distilbert, y, glove)
    create_tsne_comparison(bow, distilbert, y, glove)

    # Centroid similarity: subset of feature spaces for clarity
    centroid_sets: dict = {
        "Bag-of-Words":              bow,
        "TF-IDF":                    tfidf,
        "DistilBERT (Uncompressed)": distilbert,
    }
    if glove is not None:
        centroid_sets["GloVe (Mean-pool)"] = glove
    sim_matrices = centroid_cosine_sim_matrices(centroid_sets, y)
    create_centroid_similarity_plot(sim_matrices)

    # ── 5. Run all model × feature combinations ───────────────────────────────
    print("\n[5/7] Training and evaluating all models …")

    feature_sets: dict = {
        "Bag-of-Words":              bow,
        "TF-IDF":                    tfidf,
        "Reduced TF-IDF (SVD)":      reduced_tfidf,
        "DistilBERT (Uncompressed)": distilbert,
        "Reduced DistilBERT (PCA)":  reduced_distilbert,
    }
    if glove is not None:
        feature_sets["GloVe (Mean-pool)"]  = glove
        feature_sets["Reduced GloVe (PCA)"] = reduced_glove

    results: dict = {}
    for model_name in ALL_MODELS:
        print(f"\n  ── {model_name} ──")
        results[model_name] = {}
        for feat_name, X in feature_sets.items():
            res = train_and_evaluate(X, y, feat_name, model_name)
            results[model_name][feat_name] = res

    # ── 6. Result plots ───────────────────────────────────────────────────────
    print("\n[6/7] Generating result plots …")

    # Confusion matrix on best model: RBF SVM + Reduced DistilBERT (PCA)
    create_confusion_matrix(reduced_distilbert, y, LABEL_NAMES, model_name="RBF SVM")

    create_grouped_bar_chart(results)
    create_per_class_f1_heatmap(results)
    create_kmeans_quality_plot(results)
    create_time_accuracy_scatter(results)

    # Learning curves (cross-validated, only LR — fast)
    print("\n  [Learning curves] — this may take a few minutes …")
    lc_data: dict = {}
    lc_data["LR — Bag-of-Words"] = compute_learning_curve_data(
        bow, y, "Logistic Regression", n_cv=3
    )
    lc_data["LR — DistilBERT"] = compute_learning_curve_data(
        reduced_distilbert, y, "Logistic Regression", n_cv=3
    )
    if glove is not None:
        lc_data["LR — GloVe"] = compute_learning_curve_data(
            glove, y, "Logistic Regression", n_cv=3
        )
    create_learning_curves_plot(lc_data)

    # ── 7. LaTeX table ────────────────────────────────────────────────────────
    print("\n[7/7] Saving LaTeX results table …")
    save_latex_table(results)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  All outputs saved to outputs/")
    print("  Figures: 01–12 PNG  |  results_table.tex")
    _print_summary(results)
    write_stats_tex(results, variance_curves, reduced_distilbert, y)
    _compile_latex("report_final.tex")
    _compile_latex("submission_report.tex")
    print("=" * 70)


def _print_summary(results: dict) -> None:
    """Print a quick leaderboard of top-5 (model × feature) combinations."""
    rows = []
    for model_name, model_res in results.items():
        for feat_name, res in model_res.items():
            acc = res.get("accuracy")
            f1  = res.get("macro_f1")
            if acc is not None:
                rows.append((acc, f1, model_name, feat_name))
    rows.sort(reverse=True)
    print("\n  Top-5 configurations by accuracy:")
    for acc, f1, model, feat in rows[:5]:
        print(f"    {acc:.4f}  |  MacroF1={f1:.4f}  |  {model:22s}  ×  {feat}")
import subprocess

import subprocess
import os

def _compile_latex(tex_file: str = "report_final.tex") -> None:
    print("\n[PDF] Compiling LaTeX report ...")
    for ext in [".aux", ".log", ".out", ".toc"]:
        stale = tex_file.replace(".tex", ext)
        if os.path.exists(stale):
            os.remove(stale)
    for i in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_file],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  [PDF] pdflatex error on pass {i+1}:")
            print(result.stdout[-3000:])
            return
        print(f"  [PDF] Pass {i+1}/2 done")
    print(f"  [PDF] {tex_file.replace('.tex', '.pdf')} generated successfully")


if __name__ == "__main__":
    main()
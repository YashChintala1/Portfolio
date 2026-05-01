"""
Visualization suite: elbow/silhouette plots, 2D scatter (PCA, t-SNE, UMAP),
dendrograms, topic word clouds, and coherence curves.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from wordcloud import WordCloud

logger = logging.getLogger(__name__)

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    logger.warning("umap-learn not installed; UMAP plots will be skipped.")

OUTPUT_DIR = Path("output")
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)


def _savefig(fig, name: str):
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_elbow_silhouette(eval_results: dict):
    """Dual-axis plot of inertia (elbow) and silhouette score vs. k."""
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia", color="steelblue")
    ax1.plot(eval_results["k"], eval_results["inertia"], "o-", color="steelblue", label="Inertia")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette Score", color="coral")
    ax2.plot(eval_results["k"], eval_results["silhouette"], "s--", color="coral", label="Silhouette")
    ax2.tick_params(axis="y", labelcolor="coral")

    fig.suptitle("K-Means: Elbow & Silhouette Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "elbow_silhouette")


def _reduce_2d(X, method: str = "pca"):
    """Reduce features to 2D for plotting."""
    if issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)

    if method == "pca":
        svd = TruncatedSVD(n_components=2, random_state=42)
        return svd.fit_transform(X if issparse(X) else X_dense)
    elif method == "tsne":
        n_components_pre = min(50, X_dense.shape[1])
        if X_dense.shape[1] > 50:
            svd = TruncatedSVD(n_components=n_components_pre, random_state=42)
            X_dense = svd.fit_transform(X if issparse(X) else X_dense)
        perp = min(30, max(5, len(X_dense) // 4))
        return TSNE(n_components=2, perplexity=perp, random_state=42, init="pca", learning_rate="auto").fit_transform(X_dense)
    elif method == "umap" and HAS_UMAP:
        n_neighbors = min(15, max(2, len(X_dense) // 5))
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
        return reducer.fit_transform(X_dense)
    else:
        raise ValueError(f"Unknown method: {method}")


def plot_clusters_2d(X, labels, titles, method: str = "pca", name_suffix: str = ""):
    """Scatter plot of books in 2D, colored by cluster label."""
    coords = _reduce_2d(X, method=method)
    n_clusters = len(set(labels))

    fig, ax = plt.subplots(figsize=(11, 8))
    palette = sns.color_palette("husl", n_clusters)

    for cid in range(n_clusters):
        mask = labels == cid
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[palette[cid]], label=f"Cluster {cid}", s=60, alpha=0.7, edgecolors="white", linewidth=0.5)

        for i in np.where(mask)[0][:3]:
            title_short = str(titles.iloc[i])[:30]
            ax.annotate(title_short, (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.8)

    method_name = method.upper()
    ax.set_title(f"Book Clusters ({method_name} projection)", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    _savefig(fig, f"clusters_{method}{name_suffix}")


def plot_dendrogram(Z, labels=None, max_display: int = 40):
    """Plot a truncated dendrogram from hierarchical clustering."""
    fig, ax = plt.subplots(figsize=(14, 7))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=max_display,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax,
        color_threshold=0,
    )
    ax.set_title("Hierarchical Clustering Dendrogram", fontsize=14, fontweight="bold")
    ax.set_xlabel("Books (or clusters)")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    _savefig(fig, "dendrogram")


def plot_topic_wordclouds(topic_words: list, n_cols: int = 3):
    """Generate word clouds for each LDA topic."""
    n_topics = len(topic_words)
    n_rows = (n_topics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_topics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, words in enumerate(topic_words):
        freq = {w: p for w, p in words}
        wc = WordCloud(
            width=600, height=400,
            background_color="white",
            colormap="viridis",
            max_words=30,
        ).generate_from_frequencies(freq)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"Topic {i}", fontsize=12, fontweight="bold")
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("LDA Topic Word Clouds", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, "topic_wordclouds")


def plot_coherence_curve(eval_results: dict):
    """Plot topic coherence vs number of topics."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eval_results["n_topics"], eval_results["coherence"], "o-", color="teal", linewidth=2)
    ax.set_xlabel("Number of Topics")
    ax.set_ylabel("Coherence Score (c_v)")
    ax.set_title("LDA Topic Coherence", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "topic_coherence")


def plot_cluster_summary_heatmap(summary_df: pd.DataFrame):
    """Heatmap of numeric cluster characteristics."""
    numeric_cols = ["size", "avg_rating", "avg_pages"]
    available = [c for c in numeric_cols if c in summary_df.columns and summary_df[c].notna().any()]
    if not available:
        return

    data = summary_df.set_index("cluster")[available].astype(float)
    fig, ax = plt.subplots(figsize=(8, max(3, len(data) * 0.8)))
    sns.heatmap(data, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Cluster Characteristics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "cluster_heatmap")

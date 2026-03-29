"""
Clustering algorithms: K-Means and Hierarchical (Agglomerative).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


def find_optimal_k(X, k_range=range(2, 11)) -> dict:
    """Evaluate K-Means for each k using inertia and silhouette score."""
    if issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)

    results = {"k": [], "inertia": [], "silhouette": []}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_dense)
        sil = silhouette_score(X_dense, labels, sample_size=min(len(X_dense), 2000))
        results["k"].append(k)
        results["inertia"].append(km.inertia_)
        results["silhouette"].append(sil)
        logger.info("K=%d  inertia=%.1f  silhouette=%.3f", k, km.inertia_, sil)

    return results


def run_kmeans(X, n_clusters: int) -> tuple:
    """Fit K-Means and return (labels, model)."""
    if issparse(X):
        X = X.toarray()
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=min(len(X), 2000))
    logger.info("K-Means (k=%d): silhouette=%.3f", n_clusters, sil)
    return labels, km


def run_hierarchical(X, n_clusters: int, method: str = "ward") -> tuple:
    """Agglomerative clustering. Returns (labels, linkage_matrix)."""
    if issparse(X):
        X = X.toarray()

    Z = linkage(X, method=method, metric="euclidean")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1  # 0-indexed
    sil = silhouette_score(X, labels, sample_size=min(len(X), 2000))
    logger.info("Hierarchical (k=%d, %s): silhouette=%.3f", n_clusters, method, sil)
    return labels, Z


def summarize_clusters(df: pd.DataFrame, labels: np.ndarray, corpus: pd.Series) -> pd.DataFrame:
    """Produce a summary of what characterizes each cluster."""
    df = df.copy()
    df["cluster"] = labels
    df["_corpus"] = corpus.values

    summaries = []
    for cid in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cid]
        top_authors = subset["author"].value_counts().head(3).index.tolist() if "author" in df.columns else []
        avg_rating = subset["avg_rating"].mean() if "avg_rating" in df.columns else None
        avg_pages = subset["pages"].mean() if "pages" in df.columns else None

        shelves_all = []
        if "shelves" in df.columns:
            for s in subset["shelves"].dropna():
                shelves_all.extend([x.strip().lower() for x in s.split(",") if x.strip()])
        top_shelves = pd.Series(shelves_all).value_counts().head(5).index.tolist() if shelves_all else []

        summaries.append({
            "cluster": cid,
            "size": len(subset),
            "top_authors": ", ".join(top_authors),
            "top_shelves": ", ".join(top_shelves),
            "avg_rating": round(avg_rating, 2) if avg_rating else None,
            "avg_pages": round(avg_pages, 0) if avg_pages else None,
            "sample_titles": " | ".join(subset["title"].head(5).tolist()) if "title" in df.columns else "",
        })

    return pd.DataFrame(summaries)

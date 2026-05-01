"""
Book Pattern Discovery — main pipeline.

Usage:
    python main.py --csv goodreads_library_export.csv [options]

Run `python main.py --help` for all options.
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from data_loader import load_goodreads_csv, enrich_with_open_library, filter_liked_books
from feature_engineering import build_combined_features, build_text_corpus
from clustering import find_optimal_k, run_kmeans, run_hierarchical, summarize_clusters
from topic_modeling import build_lda_model, find_optimal_topics, get_topic_words, assign_topics
from visualize import (
    plot_elbow_silhouette,
    plot_clusters_2d,
    plot_dendrogram,
    plot_topic_wordclouds,
    plot_coherence_curve,
    plot_cluster_summary_heatmap,
    HAS_UMAP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def parse_args():
    p = argparse.ArgumentParser(description="Discover patterns in your Goodreads library.")
    p.add_argument("--csv", required=True, help="Path to Goodreads CSV export file.")
    p.add_argument("--min-rating", type=int, default=0,
                   help="Only include books you rated >= this value (0 = all read books).")
    p.add_argument("--enrich", action="store_true",
                   help="Fetch descriptions & subjects from Open Library (slow, cached after first run).")
    p.add_argument("--cache", default="data/ol_cache.csv",
                   help="Path to cache enrichment data.")
    p.add_argument("--k", type=int, default=0,
                   help="Number of clusters. 0 = auto-select via silhouette.")
    p.add_argument("--topics", type=int, default=0,
                   help="Number of LDA topics. 0 = auto-select via coherence.")
    p.add_argument("--max-tfidf", type=int, default=500,
                   help="Max TF-IDF features.")
    p.add_argument("--skip-topic-search", action="store_true",
                   help="Skip exhaustive topic number search (faster).")
    p.add_argument("--skip-cluster-search", action="store_true",
                   help="Skip exhaustive cluster k search (faster).")
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    # ── 1. Load & filter ─────────────────────────────────────────────
    logger.info("Loading Goodreads export: %s", csv_path)
    df = load_goodreads_csv(csv_path)
    logger.info("Loaded %d books total.", len(df))

    df = filter_liked_books(df, min_rating=args.min_rating)
    logger.info("After filtering: %d books.", len(df))

    if len(df) < 5:
        logger.error("Need at least 5 books to find meaningful patterns. Exiting.")
        sys.exit(1)

    # ── 2. Enrich (optional) ─────────────────────────────────────────
    if args.enrich:
        Path("data").mkdir(exist_ok=True)
        df = enrich_with_open_library(df, cache_path=args.cache)
        logger.info("Enrichment complete. Columns: %s", list(df.columns))

    # ── 3. Feature engineering ───────────────────────────────────────
    combined, corpus, vectorizer, genre_classes = build_combined_features(df, max_tfidf=args.max_tfidf)
    logger.info("Feature matrix ready: %s", combined.shape)

    # ── 4. Clustering ────────────────────────────────────────────────
    Path("output").mkdir(exist_ok=True)

    if args.k > 0:
        best_k = args.k
    elif not args.skip_cluster_search and len(df) >= 10:
        max_k = min(10, len(df) - 1)
        eval_k = find_optimal_k(combined, k_range=range(2, max_k + 1))
        plot_elbow_silhouette(eval_k)
        best_k = eval_k["k"][int(pd.Series(eval_k["silhouette"]).idxmax())]
        logger.info("Best k by silhouette: %d", best_k)
    else:
        best_k = min(4, max(2, len(df) // 10))
        logger.info("Using default k=%d", best_k)

    km_labels, km_model = run_kmeans(combined, n_clusters=best_k)
    hc_labels, linkage_Z = run_hierarchical(combined, n_clusters=best_k)

    km_summary = summarize_clusters(df, km_labels, corpus)
    logger.info("\n── K-Means Cluster Summary ──\n%s", km_summary.to_string(index=False))
    km_summary.to_csv("output/kmeans_clusters.csv", index=False)

    hc_summary = summarize_clusters(df, hc_labels, corpus)
    logger.info("\n── Hierarchical Cluster Summary ──\n%s", hc_summary.to_string(index=False))
    hc_summary.to_csv("output/hierarchical_clusters.csv", index=False)

    # Visualize clusters
    titles = df["title"] if "title" in df.columns else pd.Series(range(len(df)))
    plot_clusters_2d(combined, km_labels, titles, method="pca", name_suffix="_kmeans")
    plot_clusters_2d(combined, km_labels, titles, method="tsne", name_suffix="_kmeans")
    if HAS_UMAP:
        plot_clusters_2d(combined, km_labels, titles, method="umap", name_suffix="_kmeans")

    plot_dendrogram(linkage_Z)
    plot_cluster_summary_heatmap(km_summary)

    # ── 5. Topic modeling ────────────────────────────────────────────
    if args.topics > 0:
        best_n = args.topics
    elif not args.skip_topic_search and len(df) >= 10:
        max_t = min(12, len(df) // 2)
        topic_eval = find_optimal_topics(corpus, topic_range=range(3, max_t + 1))
        plot_coherence_curve(topic_eval)
        best_n = topic_eval["n_topics"][int(pd.Series(topic_eval["coherence"]).idxmax())]
        logger.info("Best n_topics by coherence: %d", best_n)
    else:
        best_n = min(5, max(2, len(df) // 5))
        logger.info("Using default n_topics=%d", best_n)

    lda, bow_corpus, dictionary, tokenized = build_lda_model(corpus, n_topics=best_n)
    topic_words = get_topic_words(lda, n_words=15)

    logger.info("\n── LDA Topics ──")
    for i, tw in enumerate(topic_words):
        words = ", ".join(f"{w} ({p:.3f})" for w, p in tw[:10])
        logger.info("  Topic %d: %s", i, words)

    dominant_topics = assign_topics(lda, bow_corpus)
    df["dominant_topic"] = dominant_topics
    df["cluster_kmeans"] = km_labels
    df["cluster_hierarchical"] = hc_labels

    plot_topic_wordclouds(topic_words)

    # ── 6. Save full results ─────────────────────────────────────────
    out_cols = [c for c in ["title", "author", "avg_rating", "pages", "shelves",
                            "cluster_kmeans", "cluster_hierarchical", "dominant_topic"] if c in df.columns]
    df[out_cols].to_csv("output/books_analyzed.csv", index=False)
    logger.info("Results saved to output/ directory.")
    logger.info("Done! Check the output/ folder for plots and CSV files.")


if __name__ == "__main__":
    main()

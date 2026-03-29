# Book Pattern Discovery

Unsupervised learning pipeline that discovers patterns, themes, and clusters in your Goodreads library.

## What It Does

Given your Goodreads book collection, this tool applies:

- **K-Means Clustering** — groups similar books by features (genres, keywords, authors, ratings, page counts)
- **Hierarchical Clustering** — creates a dendrogram showing book relationships at multiple levels
- **LDA Topic Modeling** — uncovers latent themes across book descriptions, shelves, and subjects
- **Dimensionality Reduction** — PCA, t-SNE, and UMAP projections to visualize your library in 2D

## Setup

```bash
cd book-pattern-discovery
pip install -r requirements.txt
```

## Getting Your Goodreads Data

1. Go to [goodreads.com/review/import](https://www.goodreads.com/review/import)
2. Click **Export Library** at the top
3. Download the CSV file when it's ready
4. Place it in this directory (or pass the full path)

## Usage

**Basic run (Goodreads CSV only):**

```bash
python main.py --csv goodreads_library_export.csv
```

**With Open Library enrichment (adds descriptions & subjects — recommended for better results):**

```bash
python main.py --csv goodreads_library_export.csv --enrich
```

**Only analyze books you rated 4+:**

```bash
python main.py --csv goodreads_library_export.csv --enrich --min-rating 4
```

**Specify exact cluster/topic counts:**

```bash
python main.py --csv goodreads_library_export.csv --enrich --k 5 --topics 7
```

**Quick run (skip exhaustive search for optimal k and topics):**

```bash
python main.py --csv goodreads_library_export.csv --skip-cluster-search --skip-topic-search
```

### All Options

| Flag | Description | Default |
|---|---|---|
| `--csv` | Path to Goodreads CSV export | (required) |
| `--min-rating` | Only include books rated >= N | 0 (all read) |
| `--enrich` | Fetch extra data from Open Library | off |
| `--cache` | Cache path for enrichment | `data/ol_cache.csv` |
| `--k` | Number of clusters (0 = auto) | 0 |
| `--topics` | Number of LDA topics (0 = auto) | 0 |
| `--max-tfidf` | Max TF-IDF features | 500 |
| `--skip-topic-search` | Skip coherence search | off |
| `--skip-cluster-search` | Skip silhouette search | off |

## Output

Everything goes into the `output/` folder:

| File | Description |
|---|---|
| `books_analyzed.csv` | Your books with cluster and topic assignments |
| `kmeans_clusters.csv` | Summary of each K-Means cluster |
| `hierarchical_clusters.csv` | Summary of each hierarchical cluster |
| `elbow_silhouette.png` | Elbow + silhouette plot for choosing k |
| `clusters_pca_kmeans.png` | PCA 2D scatter colored by cluster |
| `clusters_tsne_kmeans.png` | t-SNE 2D scatter colored by cluster |
| `clusters_umap_kmeans.png` | UMAP 2D scatter colored by cluster |
| `dendrogram.png` | Hierarchical clustering dendrogram |
| `cluster_heatmap.png` | Cluster characteristics heatmap |
| `topic_wordclouds.png` | Word clouds for each LDA topic |
| `topic_coherence.png` | Coherence vs. number of topics |

## How It Works

1. **Data Loading** — Parses the Goodreads CSV export format (handles their specific column names, ISBN formatting quirks, shelf structure)
2. **Enrichment** (optional) — Queries Open Library API by title+author to get book descriptions and subject tags (cached after first run)
3. **Feature Engineering** — Builds a combined feature matrix from:
   - TF-IDF on all text (titles, shelves, descriptions, reviews)
   - One-hot encoded genres from Goodreads shelves + Open Library subjects
   - Scaled numeric features (avg rating, page count, publication year)
4. **Clustering** — Runs K-Means and Agglomerative clustering with automatic k selection via silhouette analysis
5. **Topic Modeling** — Gensim LDA with automatic topic count selection via coherence scoring
6. **Visualization** — PCA, t-SNE, UMAP scatter plots, dendrograms, word clouds, heatmaps

## Data Sources

- **Goodreads CSV Export** — your personal library with ratings, shelves, reviews
- **Open Library API** (optional enrichment) — free, no API key required, provides descriptions and subject classifications

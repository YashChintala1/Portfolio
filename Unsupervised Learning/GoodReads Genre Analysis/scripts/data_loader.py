"""
Load books from Goodreads CSV export and enrich via Open Library API.

Goodreads CSV export columns:
  Book Id, Title, Author, Author l-f, Additional Authors, ISBN, ISBN13,
  My Rating, Average Rating, Publisher, Binding, Number of Pages, Year Published,
  Original Publication Year, Date Read, Date Added, Bookshelves,
  Bookshelves with positions, Exclusive Shelf, My Review, Spoiler, Private Notes,
  Read Count, Owned Copies
"""

import time
import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OPEN_LIBRARY_SEARCH = "https://openlibrary.org/search.json"
OPEN_LIBRARY_WORKS = "https://openlibrary.org/works/{}.json"
OL_RATE_LIMIT_DELAY = 0.5  # seconds between requests


def load_goodreads_csv(path: str | Path) -> pd.DataFrame:
    """Parse the Goodreads library-export CSV into a clean DataFrame."""
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = df.columns.str.strip()

    rename = {
        "Book Id": "book_id",
        "Title": "title",
        "Author": "author",
        "Additional Authors": "additional_authors",
        "ISBN": "isbn",
        "ISBN13": "isbn13",
        "My Rating": "my_rating",
        "Average Rating": "avg_rating",
        "Publisher": "publisher",
        "Number of Pages": "pages",
        "Year Published": "year_published",
        "Original Publication Year": "original_year",
        "Bookshelves": "shelves",
        "Exclusive Shelf": "exclusive_shelf",
        "My Review": "my_review",
        "Binding": "binding",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    for col in ["isbn", "isbn13"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip('="').str.strip('"')

    if "shelves" in df.columns:
        df["shelves"] = df["shelves"].fillna("").astype(str)

    return df


def _query_open_library(title: str, author: str) -> dict | None:
    """Search Open Library for a book and return subjects + description."""
    try:
        resp = requests.get(
            OPEN_LIBRARY_SEARCH,
            params={"title": title, "author": author, "limit": 1},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("docs"):
            return None

        doc = data["docs"][0]
        result = {
            "ol_subjects": doc.get("subject", []),
            "ol_description": "",
            "ol_first_sentence": "",
        }

        if doc.get("first_sentence"):
            result["ol_first_sentence"] = doc["first_sentence"][0] if isinstance(
                doc["first_sentence"], list
            ) else str(doc["first_sentence"])

        work_key = doc.get("key", "")
        if work_key:
            work_id = work_key.split("/")[-1]
            time.sleep(OL_RATE_LIMIT_DELAY)
            wresp = requests.get(
                OPEN_LIBRARY_WORKS.format(work_id), timeout=10
            )
            if wresp.ok:
                wdata = wresp.json()
                desc = wdata.get("description", "")
                if isinstance(desc, dict):
                    desc = desc.get("value", "")
                result["ol_description"] = str(desc)

        return result
    except Exception as e:
        logger.warning("Open Library lookup failed for '%s': %s", title, e)
        return None


def enrich_with_open_library(df: pd.DataFrame, cache_path: str | Path | None = None) -> pd.DataFrame:
    """Add subjects and descriptions from Open Library to each book row."""
    if cache_path and Path(cache_path).exists():
        logger.info("Loading cached enrichment data from %s", cache_path)
        cached = pd.read_csv(cache_path)
        return df.merge(cached, on="book_id", how="left", suffixes=("", "_cached"))

    records = []
    total = len(df)
    for idx, row in df.iterrows():
        logger.info("Enriching %d/%d: %s", idx + 1, total, row.get("title", "?"))
        result = _query_open_library(
            row.get("title", ""), row.get("author", "")
        )
        if result is None:
            result = {"ol_subjects": [], "ol_description": "", "ol_first_sentence": ""}

        result["book_id"] = row["book_id"]
        result["ol_subjects"] = "|".join(result["ol_subjects"][:30])
        records.append(result)
        time.sleep(OL_RATE_LIMIT_DELAY)

    enriched = pd.DataFrame(records)

    if cache_path:
        enriched.to_csv(cache_path, index=False)
        logger.info("Saved enrichment cache to %s", cache_path)

    return df.merge(enriched, on="book_id", how="left")


def filter_liked_books(df: pd.DataFrame, min_rating: int = 0) -> pd.DataFrame:
    """
    Keep only books the user actually liked.
    Goodreads 'exclusive_shelf' of 'read' means the user has read the book.
    Optionally filter by minimum personal rating.
    """
    if "exclusive_shelf" in df.columns:
        df = df[df["exclusive_shelf"] == "read"].copy()

    if "my_rating" in df.columns and min_rating > 0:
        df = df[df["my_rating"] >= min_rating].copy()

    return df.reset_index(drop=True)

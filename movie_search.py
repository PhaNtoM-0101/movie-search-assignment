# movie_search.py
"""
Movie semantic search utility.

- Function: search_movies(query, top_n=5)
  Loads movies.csv (in same directory) and returns top_n matches for the query.

Designed to work with your provided movies.csv which has ['title', 'plot'] columns.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import Optional, List

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(THIS_DIR, "movies.csv")

_model: Optional[SentenceTransformer] = None
_df: Optional[pd.DataFrame] = None
_embeddings: Optional[np.ndarray] = None
_text_col = "plot"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _ensure_model(model_name: str = DEFAULT_MODEL_NAME):
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def load_movies(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load movies.csv which must contain ['title', 'plot'].
    """
    path = csv_path or DEFAULT_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(f"Movies CSV not found at {path}")
    df = pd.read_csv(path)
    if "title" not in df.columns or "plot" not in df.columns:
        raise ValueError("CSV must contain 'title' and 'plot' columns.")
    df = df[["title", "plot"]].copy()
    df["plot"] = df["plot"].astype(str).str.strip()
    df = df.dropna(subset=["plot"])
    df = df[df["plot"].str.len() > 0].reset_index(drop=True)
    return df


def _ensure_index(df: Optional[pd.DataFrame] = None, model_name: str = DEFAULT_MODEL_NAME):
    """
    Ensure global df and embeddings are loaded/created. If df provided, use it.
    """
    global _df, _embeddings
    if df is not None:
        _df = df.reset_index(drop=True).copy()
        _embeddings = None

    if _df is None:
        _df = load_movies(DEFAULT_CSV)

    if _embeddings is None:
        model = _ensure_model(model_name)
        texts: List[str] = _df[_text_col].astype(str).tolist()
        _embeddings = model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32
        )
    return _df, _embeddings


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def search_movies(query: str, top_n: int = 5) -> pd.DataFrame:
    """
    Search for top_n movies similar to query.

    Returns a DataFrame with columns: ['title', 'plot', 'similarity'] sorted by similarity desc.
    Similarity is mapped into [0,1].
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    if not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    df, embeddings = _ensure_index()
    model = _ensure_model()

    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).reshape(-1)
    sims = np.dot(embeddings, q_emb.T).squeeze()  # values in [-1, 1]
    sims = (sims + 1.0) / 2.0  # map to [0,1]
    sims = np.clip(sims, 0.0, 1.0)

    n_results = min(top_n, len(df))
    order = np.argsort(-sims)[:n_results]

    out = df.iloc[order].copy().reset_index(drop=True)
    out["similarity"] = sims[order]
    return out[["title", "plot", "similarity"]]


# ---------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------
def _cli():
    parser = argparse.ArgumentParser(description="Semantic search over movie plots (uses SentenceTransformers).")
    parser.add_argument("query", type=str, help="Search query, e.g., 'spy thriller in Paris'")
    parser.add_argument("--top_n", type=int, default=5, help="Number of results to return")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to movies CSV")
    args = parser.parse_args()

    if args.csv:
        df = load_movies(args.csv)
        global _df, _embeddings
        _df = df
        _embeddings = None

    res = search_movies(args.query, top_n=args.top_n)
    pd.set_option("display.max_colwidth", 200)
    print(res.to_string(index=False))


if __name__ == "__main__":
    _cli()

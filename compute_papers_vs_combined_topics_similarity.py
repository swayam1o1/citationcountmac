"""compute_papers_vs_combined_topics_similarity.py

Compute a SINGLE TF-IDF cosine similarity score per paper against a *single*
combined topic-document.

Steps implemented (deterministic):
1) Load full text papers from /Users/swayam/Downloads/1992/*.txt
2) Load topics from consolidated_topics_with_counts.json
3) Combine all (parent) topics into one document (space-joined)
4) Build TF-IDF vectors for [combined_topic_doc] + [paper_docs]
5) Compute cosine similarity of each paper vs the combined topic vector
6) Output CSV with one similarity score per paper
7) Print top 5 papers by similarity

Inputs:
  - /Users/swayam/Downloads/1992/*.txt
  - consolidated_topics_with_counts.json

Outputs:
  - papers_vs_combined_topics_tfidf_cosine.csv

Notes:
- We use the *topic strings* as the topic-document content.
- We use unigrams+bigrams and English stopwords to reduce noise.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PAPERS_DIR = Path("/Users/swayam/Downloads/1992")
TOPICS_FILE = Path("consolidated_topics_with_counts.json")
OUTPUT_CSV = Path("papers_vs_combined_topics_tfidf_cosine.csv")


def load_paper_text(paper_path: Path) -> str:
    """Load full text from a paper file."""
    try:
        return paper_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Warning: Could not read {paper_path}: {e}")
        return ""


def load_topics(topics_file: Path) -> List[str]:
    """Load canonical topics list from consolidated_topics_with_counts.json."""
    with topics_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Expect: {"topics": [{"topic": "...", ...}, ...]}
    topics = data.get("topics", [])
    return [t["topic"] for t in topics if isinstance(t, dict) and t.get("topic")]


def main() -> None:
    print("\n" + "=" * 70)
    print("COMPUTING SINGLE TF-IDF COSINE: each paper vs COMBINED TOPICS")
    print("=" * 70 + "\n")

    if not TOPICS_FILE.exists():
        raise FileNotFoundError(f"Topics file not found: {TOPICS_FILE.resolve()}")

    if not PAPERS_DIR.exists():
        raise FileNotFoundError(f"Papers dir not found: {PAPERS_DIR}")

    # 1) Load topics
    print("Loading topics...")
    topics = load_topics(TOPICS_FILE)
    if not topics:
        raise ValueError("No topics found in consolidated_topics_with_counts.json")
    print(f"  ✓ Loaded {len(topics)} topics")

    # 2) Build ONE combined topic document
    # Joining with spaces ensures TF-IDF sees them as normal tokens/bigrams.
    combined_topic_doc = " ".join(topics)

    # 3) Load papers
    print("\nLoading papers...")
    paper_files = sorted(PAPERS_DIR.glob("*.txt"))
    paper_ids: List[str] = []
    paper_docs: List[str] = []

    for p in paper_files:
        text = load_paper_text(p)
        if not text.strip():
            continue
        paper_ids.append(p.stem)
        paper_docs.append(text)

    print(f"  ✓ Loaded {len(paper_docs)} papers")

    # 4) TF-IDF vectorization
    # Documents order: [combined topics] first, then all papers.
    print("\nComputing TF-IDF vectors...")
    all_docs = [combined_topic_doc] + paper_docs

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    tfidf = vectorizer.fit_transform(all_docs)
    print(f"  ✓ TF-IDF matrix shape: {tfidf.shape}")

    topics_vec = tfidf[0:1]  # shape: (1, vocab)
    papers_vec = tfidf[1:]   # shape: (num_papers, vocab)

    # 5) Cosine similarity per paper vs combined topic vector
    print("\nComputing cosine similarities...")
    sims = cosine_similarity(papers_vec, topics_vec).ravel()  # shape: (num_papers,)

    # 6) Output CSV
    print("\nWriting CSV...")
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["paper_id", "cosine_tfidf_similarity"])
        for pid, s in zip(paper_ids, sims):
            writer.writerow([pid, f"{float(s):.6f}"])
    print(f"  ✓ Saved to: {OUTPUT_CSV}")

    # 7) Print top 5
    print("\n" + "=" * 70)
    print("TOP 5 PAPERS (highest similarity to combined topics)")
    print("=" * 70)

    top_idx = np.argsort(sims)[::-1][:5]
    for rank, idx in enumerate(top_idx, 1):
        print(f"{rank}. Paper {paper_ids[idx]}  similarity={sims[idx]:.4f}")

    print("\n" + "=" * 70)
    print(
        f"Similarity range: [{float(sims.min()):.4f}, {float(sims.max()):.4f}]  "
        f"mean={float(sims.mean()):.4f}  std={float(sims.std()):.4f}"
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

"""compute_deeptext_papers_vs_consolidated_topics_similarity.py

Compute a SINGLE TF-IDF cosine similarity score per paper (deeptext) against a
single combined topic-document (consolidated canonical topics).

This mirrors compute_papers_vs_combined_topics_similarity.py but:
- Uses deeptext instead of full paper text.
- Runs separately for n=5 and n=10 consolidated topic lists.

Default inputs:
- outputs/1992_deeptext_topics_2021_gt20/consolidated_topics_5_with_counts.json
- outputs/1992_deeptext_topics_2021_gt20/consolidated_topics_10_with_counts.json
 - ~/Downloads/1992_deeptext (ALL deeptext files)

Default outputs:
- outputs/1992_deeptext_topics_2021_gt20/deeptext_ALL_papers_vs_consolidated_topics_tfidf_cosine_n5.csv
- outputs/1992_deeptext_topics_2021_gt20/deeptext_ALL_papers_vs_consolidated_topics_tfidf_cosine_n10.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_OUT_DIR = Path("outputs/1992_deeptext_topics_2021_gt20")
DEFAULT_CONSOLIDATED_5 = DEFAULT_OUT_DIR / "consolidated_topics_5_with_counts.json"
DEFAULT_CONSOLIDATED_10 = DEFAULT_OUT_DIR / "consolidated_topics_10_with_counts.json"
DEFAULT_DEEPTEXT_DIR = Path("~/Downloads/1992_deeptext").expanduser()


def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Warning: Could not read {path}: {e}")
        return ""


def load_consolidated_topics(consolidated_file: Path) -> List[str]:
    with consolidated_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    topics = data.get("topics", [])
    return [t["topic"] for t in topics if isinstance(t, dict) and t.get("topic")]


def load_all_deeptext_papers(deeptext_dir: Path) -> Tuple[List[str], List[str]]:
    """Return (paper_ids, deeptext_docs) for ALL files in deeptext_dir."""

    if not deeptext_dir.exists():
        raise FileNotFoundError(f"Deeptext directory not found: {deeptext_dir}")

    file_paths = sorted([p for p in deeptext_dir.iterdir() if p.is_file()])
    paper_ids: List[str] = []
    paper_docs: List[str] = []

    empty = 0
    unreadable = 0

    for p in file_paths:
        paper_id = p.stem
        text = load_text(p)
        if text == "":
            unreadable += 1
            continue

        text = text.replace("\x00", " ")
        if not text.strip():
            empty += 1
            continue

        paper_ids.append(paper_id)
        paper_docs.append(text)

    if unreadable:
        print(f"Note: unreadable deeptext files: {unreadable}")
    if empty:
        print(f"Note: empty deeptext files: {empty}")

    return paper_ids, paper_docs


def compute_and_write(
    *,
    paper_ids: List[str],
    paper_docs: List[str],
    topics: List[str],
    out_csv: Path,
) -> None:
    if not topics:
        raise ValueError("No topics found in consolidated file")
    if not paper_docs:
        raise ValueError("No deeptext docs loaded")

    combined_topic_doc = " ".join(topics)

    all_docs = [combined_topic_doc] + paper_docs

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    tfidf = vectorizer.fit_transform(all_docs)

    topics_vec = tfidf[0:1]
    papers_vec = tfidf[1:]

    sims = cosine_similarity(papers_vec, topics_vec).ravel()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["paper_id", "cosine_tfidf_similarity"])
        for pid, s in zip(paper_ids, sims):
            writer.writerow([pid, f"{float(s):.6f}"])

    top_idx = np.argsort(sims)[::-1][:5]
    print(f"\nSaved: {out_csv}")
    print("Top 5 papers by similarity:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"{rank}. {paper_ids[idx]}  similarity={sims[idx]:.4f}")
    print(
        f"Range: [{float(sims.min()):.4f}, {float(sims.max()):.4f}]  mean={float(sims.mean()):.4f}  std={float(sims.std()):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute TF-IDF cosine similarity using deeptexts vs consolidated topics (n=5 and n=10)"
    )
    parser.add_argument(
        "--deeptext-dir",
        type=Path,
        default=DEFAULT_DEEPTEXT_DIR,
        help="Directory containing ALL deeptext files to score (default: ~/Downloads/1992_deeptext)",
    )
    parser.add_argument(
        "--consolidated-5",
        type=Path,
        default=DEFAULT_CONSOLIDATED_5,
        help="Consolidated topics file for n=5",
    )
    parser.add_argument(
        "--consolidated-10",
        type=Path,
        default=DEFAULT_CONSOLIDATED_10,
        help="Consolidated topics file for n=10",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for the similarity CSV files",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("COMPUTING TF-IDF COSINE (DEEPTEXT): papers vs CONSOLIDATED TOPICS")
    print("=" * 70 + "\n")

    paper_ids, paper_docs = load_all_deeptext_papers(args.deeptext_dir.expanduser())
    print(f"Loaded deeptexts: {len(paper_docs)} papers from {args.deeptext_dir}")

    if not args.consolidated_5.exists():
        raise FileNotFoundError(f"Missing: {args.consolidated_5}")
    if not args.consolidated_10.exists():
        raise FileNotFoundError(f"Missing: {args.consolidated_10}")

    topics_5 = load_consolidated_topics(args.consolidated_5)
    topics_10 = load_consolidated_topics(args.consolidated_10)

    out_5 = args.out_dir / "deeptext_ALL_papers_vs_consolidated_topics_tfidf_cosine_n5.csv"
    out_10 = args.out_dir / "deeptext_ALL_papers_vs_consolidated_topics_tfidf_cosine_n10.csv"

    print(f"\nRunning n=5 (topics={len(topics_5)})...")
    compute_and_write(paper_ids=paper_ids, paper_docs=paper_docs, topics=topics_5, out_csv=out_5)

    print(f"\nRunning n=10 (topics={len(topics_10)})...")
    compute_and_write(paper_ids=paper_ids, paper_docs=paper_docs, topics=topics_10, out_csv=out_10)


if __name__ == "__main__":
    main()

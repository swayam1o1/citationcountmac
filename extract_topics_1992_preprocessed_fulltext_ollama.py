"""extract_topics_1992_preprocessed_fulltext_ollama.py

Reads the citations Excel (default: config.CITATIONS_FILE), filters papers with
2021 citations > threshold, loads corresponding *preprocessed full text* from
outputs/preprocessed_papers/, and uses Ollama (LLaMA 3.2) to extract EXACTLY 10
technical/trending physics topics per paper.

Then consolidates the extracted topics into a single canonical list with counts,
merging plural/singular variants (same logic as consolidate_all_topics.py).

Outputs (default out-dir: outputs/1992_preprocessed_fulltext_topics_2021_gt20):
- per-paper: <paper_id>_topics_10.json
- aggregate: all_topics_10.json
- consolidated: consolidated_topics_10_with_counts.json

Example:
  /Users/swayam/citation_count_chatgpt/.venv/bin/python extract_topics_1992_preprocessed_fulltext_ollama.py \
    --papers-dir outputs/preprocessed_papers \
    --out-dir outputs/1992_preprocessed_fulltext_topics_2021_gt20 \
    --year 2021 --threshold 20
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import pandas as pd
import requests

import config
from consolidate_all_topics import normalize_topic, merge_similar_topics
from summarize_physics_papers_llama import TopicExtractor
from utils import setup_logging, get_logger, save_json


def _find_year_column(df: pd.DataFrame, year: int) -> Optional[str]:
    year_str = str(year)
    for col in df.columns:
        if year_str in str(col):
            return col
    return None


def _coerce_paper_id(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        maybe_int = text[:-2]
        if maybe_int.isdigit():
            return maybe_int
    return text


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _normalize_text(text: str) -> str:
    return text.replace("\x00", " ")


def _test_ollama(logger, api_url: str) -> bool:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("✓ Ollama is running and accessible")
            return True
        logger.warning(f"Ollama responded with status {response.status_code}")
        return False
    except requests.exceptions.ConnectionError:
        logger.error("✗ Cannot connect to Ollama at http://localhost:11434")
        return False
    except Exception as e:
        logger.error(f"Error testing Ollama connection: {e}")
        return False


def _load_filtered_ids_and_citations(
    *,
    excel_path: Path,
    citation_year: int,
    citation_threshold: float,
    inclusive: bool,
    id_column: Optional[str] = None,
) -> Tuple[dict[str, float], str, str]:
    df = pd.read_excel(excel_path)

    year_col = _find_year_column(df, citation_year)
    if year_col is None:
        raise ValueError(
            f"Could not find a {citation_year} citation column in Excel. Columns={df.columns.tolist()}"
        )

    if id_column is None:
        id_column = df.columns[0]

    mask = df[year_col] >= citation_threshold if inclusive else df[year_col] > citation_threshold
    filtered = df.loc[mask, [id_column, year_col]].copy()

    id_to_citations: dict[str, float] = {}
    for _, row in filtered.iterrows():
        paper_id = _coerce_paper_id(row[id_column])
        try:
            citations = float(row[year_col])
        except Exception:
            continue
        if paper_id:
            id_to_citations[paper_id] = citations

    return id_to_citations, str(id_column), str(year_col)


def _consolidate_topics_dict(all_topics_by_paper: Dict[str, List[str]]) -> dict:
    """Consolidate topics with counts, merging plural/singular variants."""

    normalized_topics = defaultdict(list)
    for paper_id, topics in all_topics_by_paper.items():
        for topic in topics:
            normalized = normalize_topic(str(topic))
            if normalized:
                normalized_topics[normalized].append((normalized, paper_id))

    merged_topics = merge_similar_topics(normalized_topics)

    topic_to_papers = defaultdict(set)
    topic_counts = defaultdict(int)
    merge_log = []

    for canonical_topic, paper_data in merged_topics.items():
        original_forms = defaultdict(lambda: {"count": 0, "papers": set()})

        for orig_topic, paper_id in paper_data:
            original_forms[orig_topic]["count"] += 1
            original_forms[orig_topic]["papers"].add(paper_id)
            topic_to_papers[canonical_topic].add(paper_id)
            topic_counts[canonical_topic] += 1

        if len(original_forms) > 1:
            merged_details = []
            for form, data in sorted(original_forms.items()):
                merged_details.append(
                    {
                        "variant": form,
                        "count": data["count"],
                        "papers_mentioning": len(data["papers"]),
                    }
                )

            merge_log.append(
                {
                    "canonical": canonical_topic,
                    "total_count": topic_counts[canonical_topic],
                    "total_papers_mentioning": len(topic_to_papers[canonical_topic]),
                    "merged_from": merged_details,
                }
            )

    consolidated = []
    for topic in sorted(topic_to_papers.keys()):
        papers = sorted(list(topic_to_papers[topic]))
        consolidated.append(
            {
                "topic": topic,
                "count": topic_counts[topic],
                "papers_mentioning": len(papers),
                "paper_ids": papers,
            }
        )

    consolidated.sort(key=lambda x: (-x["papers_mentioning"], -x["count"], x["topic"]))

    total_mentions = sum(x["count"] for x in consolidated)
    total_papers = len(all_topics_by_paper) if all_topics_by_paper else 0

    return {
        "summary": {
            "total_unique_topics": len(consolidated),
            "total_topic_mentions": total_mentions,
            "total_papers": total_papers,
            "average_topics_per_paper": (total_mentions / total_papers) if total_papers else 0.0,
            "topics_merged": len(merge_log),
        },
        "topics": consolidated,
        "merge_log": merge_log,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract EXACTLY 10 topics from preprocessed full text via Ollama LLaMA 3.2 and consolidate"
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=config.CITATIONS_FILE,
        help="Path to Excel file with citations (default: config.CITATIONS_FILE)",
    )
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=Path("outputs/preprocessed_papers"),
        help="Directory containing preprocessed paper .txt files (default: outputs/preprocessed_papers)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/1992_preprocessed_fulltext_topics_2021_gt20"),
        help="Output directory for per-paper and consolidated JSON",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=config.CITATION_YEAR,
        help="Citation year to filter on (default: config.CITATION_YEAR)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(config.CITATION_THRESHOLD),
        help="Citation threshold (default: config.CITATION_THRESHOLD)",
    )
    parser.add_argument(
        "--inclusive",
        action="store_true",
        help="Use >= threshold instead of > threshold",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default=None,
        help="Override the paper-id column name in Excel (default: first column)",
    )

    args = parser.parse_args()

    setup_logging()
    logger = get_logger()

    logger.info("=" * 60)
    logger.info("Starting preprocessed-fulltext topic extraction (10)")
    logger.info("=" * 60)

    if not args.excel.exists():
        logger.error(f"Excel file not found: {args.excel}")
        return
    if not args.papers_dir.exists():
        logger.error(f"Preprocessed papers directory not found: {args.papers_dir}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not _test_ollama(logger, config.LLAMA_API_URL):
        logger.error("Please start Ollama: 'ollama serve'")
        logger.error(f"And ensure model is present: 'ollama pull {config.LLAMA_MODEL}'")
        return

    try:
        id_to_citations, id_col, year_col = _load_filtered_ids_and_citations(
            excel_path=args.excel,
            citation_year=args.year,
            citation_threshold=args.threshold,
            inclusive=args.inclusive,
            id_column=args.id_column,
        )
    except Exception as e:
        logger.error(f"Failed to load/filter Excel: {e}")
        return

    logger.info(
        f"Filtered to {len(id_to_citations)} papers with {args.year} citations "
        f"{'>= ' if args.inclusive else '> '}{args.threshold} (id_col='{id_col}', year_col='{year_col}')"
    )

    extractor = TopicExtractor()

    papers_payload: Dict[str, dict] = {}
    topics_by_paper: Dict[str, List[str]] = {}

    missing_text = 0
    failed = 0

    paper_ids = sorted(id_to_citations.keys())
    for idx, paper_id in enumerate(paper_ids, start=1):
        citations = id_to_citations[paper_id]
        logger.info(f"[{idx}/{len(paper_ids)}] {paper_id} (citations_{args.year}={citations})")

        paper_path = args.papers_dir / f"{paper_id}.txt"
        if not paper_path.exists():
            logger.warning(f"No preprocessed fulltext found for paper_id={paper_id} at {paper_path}")
            missing_text += 1
            continue

        try:
            paper_text = _normalize_text(_read_text_file(paper_path))
        except Exception as e:
            logger.warning(f"Failed to read preprocessed fulltext for {paper_id} at {paper_path}: {e}")
            missing_text += 1
            continue

        topics_10 = extractor.extract_topics(paper_id, paper_text, topics_per_paper=10)
        if not topics_10:
            logger.error(f"Topic extraction failed for {paper_id}")
            failed += 1
            continue

        record = {
            "paper_id": paper_id,
            "paper_file": str(paper_path),
            "citation_year": args.year,
            "citations": citations,
            "topics_10": topics_10,
            "model": config.LLAMA_MODEL,
            "api_url": config.LLAMA_API_URL,
            "created_at": dt.datetime.utcnow().isoformat() + "Z",
        }

        papers_payload[paper_id] = record
        topics_by_paper[paper_id] = topics_10

        per_paper_path = args.out_dir / f"{paper_id}_topics_10.json"
        save_json(record, per_paper_path)

    aggregate_path = args.out_dir / "all_topics_10.json"
    save_json(
        {
            "citation_year": args.year,
            "threshold": args.threshold,
            "inclusive": bool(args.inclusive),
            "total_filtered": len(id_to_citations),
            "total_extracted": len(papers_payload),
            "missing_fulltext": missing_text,
            "failed_extractions": failed,
            "papers": papers_payload,
        },
        aggregate_path,
    )

    consolidated = _consolidate_topics_dict(topics_by_paper)
    consolidated_path = args.out_dir / "consolidated_topics_10_with_counts.json"
    with consolidated_path.open("w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Filtered papers:        {len(id_to_citations)}")
    logger.info(f"Extracted (10 topics):  {len(papers_payload)}")
    logger.info(f"Missing fulltext:       {missing_text}")
    logger.info(f"Failed extractions:     {failed}")
    logger.info(f"Unique topics (merged): {consolidated['summary']['total_unique_topics']}")
    logger.info(f"Topics merged:          {consolidated['summary']['topics_merged']}")
    logger.info(f"Outputs: {args.out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

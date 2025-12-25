"""extract_topics_1992_deeptext_ollama.py

Reads ~/Downloads/1992.xlsx, filters papers with 2021 citations > threshold,
loads corresponding deeptext from ~/Downloads/1992_deeptext, and uses Ollama
(LLaMA 3.2) to extract EXACTLY 5 and EXACTLY 10 technical topics per paper.

Outputs per-paper JSON plus a consolidated JSON into a separate output folder.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests

import config
from summarize_physics_papers_llama import TopicExtractor
from utils import setup_logging, get_logger, save_json


def _find_year_column(df: pd.DataFrame, year: int) -> Optional[str]:
    year_str = str(year)
    for col in df.columns:
        if year_str in str(col):
            return col
    return None


def _coerce_paper_id(value) -> str:
    # Make IDs robust against numeric formatting in Excel (e.g., 123.0)
    text = str(value).strip()
    if text.endswith(".0"):
        maybe_int = text[:-2]
        if maybe_int.isdigit():
            return maybe_int
    return text


def _read_text_file(path: Path) -> str:
    # Deeptext can contain odd bytes; be forgiving.
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _normalize_text(text: str) -> str:
    # Minimal cleanup only (not full preprocessing): remove NULs.
    return text.replace("\x00", " ")


def _find_deeptext_file(deeptext_dir: Path, paper_id: str) -> Optional[Path]:
    # Try common filename patterns.
    candidates = []
    for ext in (".txt", ".text", ".md"):
        p = deeptext_dir / f"{paper_id}{ext}"
        if p.exists():
            candidates.append(p)

    # Fallback: any file with matching stem
    if not candidates:
        matches = sorted(deeptext_dir.glob(f"{paper_id}.*"))
        for m in matches:
            if m.is_file():
                candidates.append(m)

    return candidates[0] if candidates else None


def _test_ollama(logger, api_url: str) -> bool:
    # Ollama tags endpoint is stable for a simple health check.
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract EXACTLY 5 and EXACTLY 10 topics from deeptext via Ollama LLaMA 3.2"
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=config.CITATIONS_FILE,
        help="Path to Excel file with citations (default: config.CITATIONS_FILE)",
    )
    parser.add_argument(
        "--deeptext-dir",
        type=Path,
        default=Path("~/Downloads/1992_deeptext").expanduser(),
        help="Directory containing deeptext files (default: ~/Downloads/1992_deeptext)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/1992_deeptext_topics_2021_gt20"),
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
    logger.info("Starting deeptext topic extraction (5 + 10)")
    logger.info("=" * 60)

    if not args.excel.exists():
        logger.error(f"Excel file not found: {args.excel}")
        return
    if not args.deeptext_dir.exists():
        logger.error(f"Deeptext directory not found: {args.deeptext_dir}")
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

    consolidated: dict[str, dict] = {}
    missing_text = 0
    failed = 0

    paper_ids = sorted(id_to_citations.keys())
    for idx, paper_id in enumerate(paper_ids, start=1):
        citations = id_to_citations[paper_id]
        logger.info(f"[{idx}/{len(paper_ids)}] {paper_id} (citations_{args.year}={citations})")

        deeptext_file = _find_deeptext_file(args.deeptext_dir, paper_id)
        if deeptext_file is None:
            logger.warning(f"No deeptext file found for paper_id={paper_id}")
            missing_text += 1
            continue

        try:
            paper_text = _normalize_text(_read_text_file(deeptext_file))
        except Exception as e:
            logger.warning(f"Failed to read deeptext for {paper_id} at {deeptext_file}: {e}")
            missing_text += 1
            continue

        topics_5 = extractor.extract_topics(paper_id, paper_text, topics_per_paper=5)
        topics_10 = extractor.extract_topics(paper_id, paper_text, topics_per_paper=10)

        if not topics_5 or not topics_10:
            logger.error(f"Topic extraction failed for {paper_id}")
            failed += 1
            continue

        record = {
            "paper_id": paper_id,
            "deeptext_file": str(deeptext_file),
            "citation_year": args.year,
            "citations": citations,
            "topics_5": topics_5,
            "topics_10": topics_10,
            "model": config.LLAMA_MODEL,
            "api_url": config.LLAMA_API_URL,
            "created_at": dt.datetime.utcnow().isoformat() + "Z",
        }

        consolidated[paper_id] = record

        per_paper_path = args.out_dir / f"{paper_id}_topics.json"
        save_json(record, per_paper_path)

    consolidated_path = args.out_dir / "all_topics_5_and_10.json"
    save_json(
        {
            "citation_year": args.year,
            "threshold": args.threshold,
            "inclusive": bool(args.inclusive),
            "total_filtered": len(id_to_citations),
            "total_extracted": len(consolidated),
            "missing_deeptext": missing_text,
            "failed_extractions": failed,
            "papers": consolidated,
        },
        consolidated_path,
    )

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Filtered papers: {len(id_to_citations)}")
    logger.info(f"Extracted (both 5 and 10): {len(consolidated)}")
    logger.info(f"Missing deeptext: {missing_text}")
    logger.info(f"Failed extractions: {failed}")
    logger.info(f"Outputs: {args.out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

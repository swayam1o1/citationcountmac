"""
Research-grade text preprocessing for academic papers.

Preprocesses all .txt files in a folder using a deterministic,
documented pipeline suitable for similarity computation.
"""

import re
import string
from pathlib import Path
from typing import List
import logging

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ============================
# CONFIGURATION
# ============================

INPUT_DIR = Path("/Users/swayam/Downloads/1992")
OUTPUT_DIR = Path("outputs/preprocessed_papers")
LANGUAGE = "english"

REMOVE_STOPWORDS = True
REMOVE_NUMBERS = True

LOG_FILE = "preprocessing.log"


# ============================
# LOGGING
# ============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================
# NLTK SETUP
# ============================

def ensure_nltk():
    required = ["punkt", "stopwords"]
    for item in required:
        try:
            nltk.data.find(f"tokenizers/{item}" if item == "punkt" else f"corpora/{item}")
        except LookupError:
            nltk.download(item, quiet=True)

ensure_nltk()

STOPWORDS = set(stopwords.words(LANGUAGE))


# ============================
# PREPROCESSING
# ============================

def clean_text(text: str) -> str:
    """Remove URLs, emails, normalize whitespace."""
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Lowercase and tokenize."""
    return word_tokenize(text.lower())


def filter_tokens(tokens: List[str]) -> List[str]:
    """Remove punctuation, numbers, and stopwords."""
    cleaned = []

    for token in tokens:
        if all(c in string.punctuation for c in token):
            continue

        if REMOVE_NUMBERS and token.isdigit():
            continue

        if REMOVE_STOPWORDS and token in STOPWORDS:
            continue

        cleaned.append(token)

    return cleaned


def preprocess_text(text: str) -> str:
    """Complete preprocessing pipeline."""
    if not text or not text.strip():
        return ""

    text = clean_text(text)
    tokens = tokenize(text)
    tokens = filter_tokens(tokens)

    return " ".join(tokens)


# ============================
# FOLDER PROCESSING
# ============================

def preprocess_folder(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.txt"))
    logger.info(f"Found {len(files)} text files")

    for idx, file in enumerate(files, 1):
        if idx % 100 == 0:
            logger.info(f"Processed {idx}/{len(files)} files")

        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
            processed = preprocess_text(text)

            out_file = output_dir / file.name
            out_file.write_text(processed, encoding="utf-8")

        except Exception as e:
            logger.error(f"Failed to process {file.name}: {e}")

    logger.info("Preprocessing complete")


# ============================
# MAIN
# ============================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("STARTING TEXT PREPROCESSING")
    logger.info("=" * 80)

    preprocess_folder(INPUT_DIR, OUTPUT_DIR)

    logger.info("=" * 80)
    logger.info("DONE")
    logger.info("=" * 80)

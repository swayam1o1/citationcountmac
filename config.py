"""
Configuration module for citation-count analysis pipeline
Centralizes all paths, parameters, and settings
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
RAW_PAPERS_DIR = Path("/Users/swayam/Downloads/1992")
CITATIONS_FILE = Path("/Users/swayam/Downloads/1992.xlsx")

# Internal data directory
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_PAPERS_DIR = DATA_DIR / "raw_papers"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TRENDING_TOPICS_PER_PAPER = OUTPUT_DIR / "trending_topics_per_paper.json"
TRENDING_TOPICS_GLOBAL = OUTPUT_DIR / "trending_topics_global.json"
TFIDF_SIMILARITY_CSV = OUTPUT_DIR / "tfidf_similarity.csv"
JACCARD_SIMILARITY_CSV = OUTPUT_DIR / "jaccard_similarity.csv"

# Logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "pipeline.log"

# Citation filtering parameters
CITATION_YEAR = 2021
CITATION_THRESHOLD = 20

# Topic extraction parameters
TOPICS_PER_PAPER = 5
MIN_TOPIC_WORDS = 1
MAX_TOPIC_WORDS = 6
MAX_RETRY_ATTEMPTS = 3

# LLaMA 3.2 configuration
# Adjust these based on your LLaMA setup (local or API)
LLAMA_MODEL = "llama3.2:latest"  # Or your specific model variant
LLAMA_API_URL = "http://localhost:11434/api/generate"  # For Ollama local
LLAMA_TEMPERATURE = 0.3  # Lower for more consistent extraction
LLAMA_MAX_TOKENS = 200

# Text preprocessing
STOPWORDS_LANGUAGE = "english"

# Similarity computation
TFIDF_MAX_FEATURES = 10000
TFIDF_MIN_DF = 1
TFIDF_MAX_DF = 0.95

# File extensions to process
VALID_EXTENSIONS = ['.pdf', '.txt']

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_RAW_PAPERS_DIR.mkdir(parents=True, exist_ok=True)

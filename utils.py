"""
Utility functions for logging, file I/O, and common operations
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any
import sys

from config import LOG_FILE, LOG_DIR


def setup_logging(log_level=logging.INFO):
    """
    Configure logging for the entire pipeline
    
    Args:
        log_level: Logging level (default: INFO)
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('citation_count')
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = 'citation_count'):
    """Get logger instance"""
    return logging.getLogger(name)


def save_json(data: Any, filepath: Path, indent: int = 2):
    """
    Save data to JSON file with error handling
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation level
    """
    logger = get_logger()
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.info(f"Successfully saved JSON to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        raise


def load_json(filepath: Path) -> Any:
    """
    Load data from JSON file with error handling
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    logger = get_logger()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON from {filepath}: {e}")
        raise


def validate_file_exists(filepath: Path) -> bool:
    """
    Check if file exists and is readable
    
    Args:
        filepath: File path to validate
        
    Returns:
        True if file exists and is readable
    """
    return filepath.exists() and filepath.is_file()


def get_paper_identifier(filepath: Path) -> str:
    """
    Extract paper identifier from filepath
    
    Args:
        filepath: Path to paper file
        
    Returns:
        Paper identifier (filename without extension)
    """
    return filepath.stem


def count_words(text: str) -> int:
    """
    Count words in text
    
    Args:
        text: Input text
        
    Returns:
        Word count
    """
    return len(text.split())


def validate_topic(topic: str, min_words: int, max_words: int) -> bool:
    """
    Validate if topic meets word count requirements
    
    Args:
        topic: Topic string
        min_words: Minimum word count
        max_words: Maximum word count
        
    Returns:
        True if valid
    """
    word_count = count_words(topic.strip())
    return min_words <= word_count <= max_words


def consolidate_topics(topics_per_paper: Dict[str, List[str]]) -> List[str]:
    """
    Consolidate topics across all papers (case-insensitive unique)
    
    Args:
        topics_per_paper: Dictionary mapping paper_id to list of topics
        
    Returns:
        List of unique topics (case-insensitive)
    """
    logger = get_logger()
    
    all_topics = []
    for topics in topics_per_paper.values():
        all_topics.extend(topics)
    
    # Case-insensitive unique - preserve first occurrence case
    seen = {}
    unique_topics = []
    for topic in all_topics:
        topic_lower = topic.lower()
        if topic_lower not in seen:
            seen[topic_lower] = True
            unique_topics.append(topic)
    
    logger.info(f"Consolidated {len(all_topics)} topics into {len(unique_topics)} unique topics")
    return unique_topics

"""
Verify and fix topics in all_topics.json
Re-extract topics for papers with questionable/meaningless terms
"""

import json
import requests
from pathlib import Path
from summarize_physics_papers_llama import TopicExtractor
from utils import get_logger

# Potentially meaningless or generic terms to flag
QUESTIONABLE_TERMS = [
    "coprime",
    "analysis",
    "study",
    "method",
    "approach",
    "technique",
    "framework",
    "model",  # Only if alone, not in compound terms
    "theory",  # Only if alone
    "equation",  # Only if alone
    "system",  # Only if alone
]

def is_questionable_topic(topic):
    """Check if a topic seems meaningless or too generic"""
    topic_lower = topic.lower().strip()
    
    # Check for single generic words
    if topic_lower in QUESTIONABLE_TERMS:
        return True
    
    # Check if it's just a number or very short
    if len(topic_lower) <= 2:
        return True
    
    # Check for common filler words
    if topic_lower in ["paper", "abstract", "introduction", "conclusion", "results"]:
        return True
    
    return False

def main():
    logger = get_logger()
    logger.info("="*60)
    logger.info("Verifying Topics")
    logger.info("="*60)
    
    # Load all_topics.json
    topics_file = Path("summaries_2021/all_topics.json")
    with open(topics_file, 'r') as f:
        all_topics = json.load(f)
    
    logger.info(f"Loaded {len(all_topics)} papers with topics")
    
    # Check for questionable topics
    papers_to_reprocess = []
    
    for paper_id, topics in all_topics.items():
        questionable = []
        for topic in topics:
            if is_questionable_topic(topic):
                questionable.append(topic)
        
        if questionable:
            logger.warning(f"Paper {paper_id} has questionable topics: {questionable}")
            papers_to_reprocess.append(paper_id)
    
    if not papers_to_reprocess:
        logger.info("✓ All topics look good!")
        return
    
    logger.info(f"\nFound {len(papers_to_reprocess)} papers with questionable topics")
    logger.info(f"Papers to reprocess: {papers_to_reprocess}")
    
    # Re-extract topics for these papers
    logger.info("\n" + "="*60)
    logger.info("Re-extracting topics for questionable papers")
    logger.info("="*60)
    
    extractor = TopicExtractor()
    preprocessed_dir = Path("outputs/preprocessed_papers")
    
    for paper_id in papers_to_reprocess:
        logger.info(f"\nReprocessing {paper_id}...")
        
        # Read paper text
        paper_file = preprocessed_dir / f"{paper_id}.txt"
        if not paper_file.exists():
            logger.error(f"Preprocessed file not found: {paper_file}")
            continue
        
        with open(paper_file, 'r', encoding='utf-8') as f:
            paper_text = f.read()
        
        # Extract topics
        new_topics = extractor.extract_topics(paper_id, paper_text)
        
        if new_topics:
            logger.info(f"Old topics: {all_topics[paper_id]}")
            logger.info(f"New topics: {new_topics}")
            
            # Update
            all_topics[paper_id] = new_topics
            
            # Save individual file
            output_file = Path("summaries_2021") / f"{paper_id}_topics.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "paper_id": paper_id,
                    "topics": new_topics,
                    "topic_count": len(new_topics)
                }, f, indent=2)
        else:
            logger.error(f"Failed to re-extract topics for {paper_id}")
    
    # Save updated all_topics.json
    logger.info("\n" + "="*60)
    logger.info("Saving updated all_topics.json")
    logger.info("="*60)
    
    with open(topics_file, 'w', encoding='utf-8') as f:
        json.dump(all_topics, f, indent=2)
    
    logger.info("✓ Done!")

if __name__ == "__main__":
    main()

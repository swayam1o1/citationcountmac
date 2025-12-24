"""
Topic extraction module - integrates with LLaMA 3.2 for topic extraction
Enforces strict 5-topic constraint with auto-retry
"""

import json
import requests
import time
from typing import List, Dict, Optional
import config

from config import (
    TOPICS_PER_PAPER,
    MIN_TOPIC_WORDS,
    MAX_TOPIC_WORDS,
    MAX_RETRY_ATTEMPTS,
    LLAMA_MODEL,
    LLAMA_API_URL,
    LLAMA_TEMPERATURE,
    LLAMA_MAX_TOKENS,
    CITATIONS_FILE,
    CITATION_YEAR,
    CITATION_THRESHOLD
)
from utils import get_logger, validate_topic


class TopicExtractor:
    """
    Extracts exactly 5 technical/physics topics from papers using LLaMA 3.2
    """
    
    def __init__(self, api_url: str = LLAMA_API_URL, model: str = LLAMA_MODEL):
        self.logger = get_logger()
        self.api_url = api_url
        self.model = model
    
    def _build_prompt(self, paper_text: str) -> str:
        """
        Build prompt for LLaMA to extract topics
        
        Args:
            paper_text: Full paper text
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a scientific topic extraction specialist. Analyze the following research paper and extract EXACTLY 5 trending technical/physics topics.

STRICT REQUIREMENTS:
- Extract EXACTLY 5 topics, no more, no less
- Each topic must be 2-6 words
- Topics must be technical/physics-specific (equations, laws, effects, models, algorithms, mathematical formulations)
- NO generic terms like "analysis", "study", "method"
- NO full sentences
- NO explanations

OUTPUT FORMAT (JSON only):
{{
  "topics": ["topic1", "topic2", "topic3", "topic4", "topic5"]
}}

PAPER TEXT:
{paper_text[:8000]}

Return ONLY valid JSON with exactly 5 topics."""

        return prompt
    
    def _call_llama(self, prompt: str) -> Optional[str]:
        """
        Call LLaMA API (Ollama local endpoint)
        
        Args:
            prompt: Input prompt
            
        Returns:
            Raw response text or None if failed
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": LLAMA_TEMPERATURE,
                "max_tokens": LLAMA_MAX_TOKENS,
                "stream": False,
                "format": "json"
            }
            
            self.logger.debug(f"Calling LLaMA API: {self.api_url}")
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                # Ollama returns response in 'response' field
                return result.get('response', '')
            else:
                self.logger.error(f"LLaMA API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Cannot connect to LLaMA API at {self.api_url}. Is Ollama running?")
            return None
        except Exception as e:
            self.logger.error(f"LLaMA API call failed: {e}")
            return None
    
    def _parse_topics(self, response_text: str) -> Optional[List[str]]:
        """
        Parse topics from LLaMA response
        
        Args:
            response_text: Raw LLaMA response
            
        Returns:
            List of topics or None if parsing failed
        """
        try:
            # Try to parse JSON
            data = json.loads(response_text)
            
            if 'topics' in data and isinstance(data['topics'], list):
                topics = data['topics']
                return topics
            else:
                self.logger.warning("Response JSON missing 'topics' field")
                return None
                
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")
            # Try to extract topics from malformed response
            try:
                # Look for list patterns
                import re
                matches = re.findall(r'"([^"]+)"', response_text)
                if len(matches) >= TOPICS_PER_PAPER:
                    return matches[:TOPICS_PER_PAPER]
            except:
                pass
            return None
    
    def _validate_topics(self, topics: List[str]) -> bool:
        """
        Validate extracted topics against requirements
        
        Args:
            topics: List of topics
            
        Returns:
            True if valid
        """
        # Check count
        if len(topics) != TOPICS_PER_PAPER:
            self.logger.warning(f"Invalid topic count: {len(topics)} (expected {TOPICS_PER_PAPER})")
            return False
        
        # Check each topic
        for i, topic in enumerate(topics):
            if not isinstance(topic, str) or not topic.strip():
                self.logger.warning(f"Topic {i+1} is empty or invalid")
                return False
            
            # Check word count
            if not validate_topic(topic, MIN_TOPIC_WORDS, MAX_TOPIC_WORDS):
                self.logger.warning(f"Topic {i+1} '{topic}' has invalid word count")
                return False
        
        return True
    
    def extract_topics(self, paper_id: str, paper_text: str) -> Optional[List[str]]:
        """
        Extract exactly 5 topics from paper with retry logic
        
        Args:
            paper_id: Paper identifier
            paper_text: Full paper text
            
        Returns:
            List of 5 topics or None if extraction failed
        """
        self.logger.info(f"Extracting topics for paper: {paper_id}")
        
        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            self.logger.debug(f"Attempt {attempt}/{MAX_RETRY_ATTEMPTS}")
            
            # Build prompt
            prompt = self._build_prompt(paper_text)
            
            # Call LLaMA
            response = self._call_llama(prompt)
            
            if response is None:
                self.logger.warning(f"No response from LLaMA (attempt {attempt})")
                time.sleep(2)
                continue
            
            # Parse topics
            topics = self._parse_topics(response)
            
            if topics is None:
                self.logger.warning(f"Failed to parse topics (attempt {attempt})")
                time.sleep(2)
                continue
            
            # Validate topics
            if self._validate_topics(topics):
                self.logger.info(f"Successfully extracted {len(topics)} topics for {paper_id}")
                self.logger.debug(f"Topics: {topics}")
                return topics
            else:
                self.logger.warning(f"Topics validation failed (attempt {attempt})")
                time.sleep(2)
        
        self.logger.error(f"Failed to extract valid topics for {paper_id} after {MAX_RETRY_ATTEMPTS} attempts")
        return None
    
    def extract_batch(self, papers: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """
        Extract topics from multiple papers
        
        Args:
            papers: List of dicts with 'paper_id' and 'text' keys
            
        Returns:
            Dictionary mapping paper_id to list of topics
        """
        self.logger.info(f"Starting batch topic extraction for {len(papers)} papers...")
        
        results = {}
        successful = 0
        failed = 0
        
        for i, paper in enumerate(papers, 1):
            paper_id = paper['paper_id']
            paper_text = paper['text']
            
            self.logger.info(f"Processing paper {i}/{len(papers)}: {paper_id}")
            
            topics = self.extract_topics(paper_id, paper_text)
            
            if topics is not None:
                results[paper_id] = topics
                successful += 1
            else:
                failed += 1
                self.logger.error(f"Skipping paper {paper_id} - topic extraction failed")
        
        self.logger.info(f"Batch extraction complete: {successful} successful, {failed} failed")
        return results


def main():
    """
    Main execution function - processes papers with citations >= threshold in citation year
    """
    import os
    from pathlib import Path
    import pandas as pd
    
    logger = get_logger()
    logger.info("="*60)
    logger.info("Starting Topic Extraction Pipeline")
    logger.info("="*60)
    
    # Set up paths
    preprocessed_dir = Path("outputs/preprocessed_papers")
    output_dir = Path("summaries_2021")
    output_dir.mkdir(exist_ok=True)
    
    # Check if preprocessed papers exist
    if not preprocessed_dir.exists():
        logger.error(f"Preprocessed papers directory not found: {preprocessed_dir}")
        logger.error("Please run preprocess.py first!")
        return
    
    # Load citation data
    logger.info(f"Loading citation data from {config.CITATIONS_FILE}...")
    if not config.CITATIONS_FILE.exists():
        logger.error(f"Citation file not found: {config.CITATIONS_FILE}")
        return
    
    try:
        # Read Excel file
        df = pd.read_excel(config.CITATIONS_FILE)
        logger.info(f"Loaded citation data: {len(df)} papers")
        logger.debug(f"Columns: {df.columns.tolist()}")
        
        # Find the column for year 2021 (might be named '2021', 'year_2021', etc.)
        year_col = None
        for col in df.columns:
            if '2021' in str(col):
                year_col = col
                break
        
        if year_col is None:
            logger.error("Could not find 2021 citation column in Excel file")
            logger.info(f"Available columns: {df.columns.tolist()}")
            return
        
        logger.info(f"Using column '{year_col}' for 2021 citations")
        
        # Filter papers with citations >= threshold
        # Assuming first column is paper ID
        id_col = df.columns[0]
        logger.info(f"Using column '{id_col}' as paper ID")
        
        # Filter papers
        filtered_df = df[df[year_col] >= config.CITATION_THRESHOLD]
        logger.info(f"Found {len(filtered_df)} papers with citations >= {config.CITATION_THRESHOLD} in {config.CITATION_YEAR}")
        
        # Get list of paper IDs to process
        paper_ids_to_process = set(str(pid).strip() for pid in filtered_df[id_col].values)
        logger.info(f"Paper IDs to process: {len(paper_ids_to_process)}")
        
    except Exception as e:
        logger.error(f"Error loading citation data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Load all preprocessed papers and filter by citation threshold
    all_paper_files = sorted(preprocessed_dir.glob("*.txt"))
    
    if not all_paper_files:
        logger.error(f"No preprocessed papers found in {preprocessed_dir}")
        return
    
    # Filter to only papers that meet citation threshold
    paper_files = [pf for pf in all_paper_files if pf.stem in paper_ids_to_process]
    
    logger.info(f"Found {len(all_paper_files)} preprocessed papers")
    logger.info(f"Filtered to {len(paper_files)} papers with citations >= {config.CITATION_THRESHOLD} in {config.CITATION_YEAR}")
    
    # Initialize topic extractor
    logger.info("Initializing LLaMA Topic Extractor...")
    extractor = TopicExtractor()
    
    # Test connection to LLaMA
    logger.info(f"Testing connection to LLaMA API at {LLAMA_API_URL}...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("✓ LLaMA/Ollama is running and accessible")
        else:
            logger.warning("LLaMA/Ollama responded but with unexpected status")
    except requests.exceptions.ConnectionError:
        logger.error("✗ Cannot connect to LLaMA/Ollama!")
        logger.error("Please ensure Ollama is running: 'ollama serve'")
        logger.error(f"And that the model is available: 'ollama pull {LLAMA_MODEL}'")
        return
    except Exception as e:
        logger.error(f"Error testing LLaMA connection: {e}")
        return
    
    # Process each paper
    logger.info("="*60)
    logger.info("Starting paper processing...")
    logger.info("="*60)
    
    all_topics = {}
    
    for i, paper_file in enumerate(paper_files, 1):
        paper_id = paper_file.stem
        
        logger.info(f"\n[{i}/{len(paper_files)}] Processing {paper_id}...")
        
        # Read paper text
        try:
            with open(paper_file, 'r', encoding='utf-8') as f:
                paper_text = f.read()
        except Exception as e:
            logger.error(f"Failed to read {paper_file}: {e}")
            continue
        
        # Extract topics
        topics = extractor.extract_topics(paper_id, paper_text)
        
        if topics:
            all_topics[paper_id] = topics
            
            # Save individual paper topics
            output_file = output_dir / f"{paper_id}_topics.json"
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "paper_id": paper_id,
                        "topics": topics,
                        "topic_count": len(topics)
                    }, f, indent=2)
                logger.info(f"✓ Saved topics to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save topics for {paper_id}: {e}")
        else:
            logger.error(f"✗ Failed to extract topics for {paper_id}")
    
    # Save consolidated results
    logger.info("="*60)
    logger.info("Saving consolidated results...")
    logger.info("="*60)
    
    consolidated_file = output_dir / "all_topics.json"
    try:
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            json.dump(all_topics, f, indent=2)
        logger.info(f"✓ Saved consolidated topics to {consolidated_file}")
    except Exception as e:
        logger.error(f"Failed to save consolidated file: {e}")
    
    # Summary
    logger.info("="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Total papers processed: {len(paper_files)}")
    logger.info(f"Topics successfully extracted: {len(all_topics)}")
    logger.info(f"Failed extractions: {len(paper_files) - len(all_topics)}")
    logger.info(f"Success rate: {len(all_topics)/len(paper_files)*100:.1f}%")
    logger.info("="*60)


if __name__ == "__main__":
    main()

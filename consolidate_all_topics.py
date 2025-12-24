"""
Consolidate all topics from all_topics.json into a single file with counts.
Normalizes text (lowercase, removes extra spaces) and merges similar topics.

Features:
- Lowercases all topics
- Merges plurals (e.g., "algebra" and "algebras")
- Merges minor variations (e.g., "theory" vs "theories")
- Prefers singular forms

Input:  summaries_2021/all_topics.json
Output: consolidated_topics_with_counts.json
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import re


def normalize_topic(topic: str) -> str:
    """
    Normalize topic text:
    - Convert to lowercase
    - Strip whitespace
    - Replace multiple spaces with single space
    """
    return ' '.join(topic.lower().strip().split())


def get_canonical_form(topic: str) -> str:
    """
    Get canonical form of a topic by singularizing common endings.
    This helps merge similar topics like "algebra" and "algebras".
    """
    topic = topic.lower().strip()
    
    # Special cases: words that should NOT be singularized
    no_singularize = {
        'mass', 'process', 'class', 'gross', 'ross', 'basis', 
        'analysis', 'thesis', 'genesis', 'axis', 'crisis',
        'emphasis', 'oasis', 'parenthesis', 'synopsis', 'diagnosis'
    }
    
    if topic in no_singularize:
        return topic
    
    # Handle common plural forms
    # theories -> theory
    if topic.endswith('theories'):
        return topic[:-3] + 'y'
    # properties -> property, symmetries -> symmetry
    elif topic.endswith('ies') and len(topic) > 4 and topic[-4] not in 'aeiou':
        return topic[:-3] + 'y'
    # matrices -> matrix
    elif topic.endswith('ices') and len(topic) > 5:
        return topic[:-4] + 'ix'
    # forms, terms, models, algebras, manifolds, states, constraints, etc.
    elif topic.endswith('s') and len(topic) > 4:
        # Don't singularize if it ends with ss, us, is, os
        if any(topic.endswith(x) for x in ['ss', 'ous', 'ius', 'sis', 'xis']):
            return topic
        # Don't singularize single words ending with 'as' or specific patterns
        if topic.endswith('as') and ' ' not in topic:
            return topic
        # Otherwise, remove the 's'
        return topic[:-1]
    
    return topic


def merge_similar_topics(topics_dict: Dict[str, List[Tuple[str, str]]]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Merge topics that are the same except for plural/singular variations.
    Returns a dict mapping canonical form to list of (original_topic, paper_id) tuples.
    """
    canonical_to_originals = defaultdict(list)
    
    for topic, paper_data in topics_dict.items():
        canonical = get_canonical_form(topic)
        canonical_to_originals[canonical].extend(paper_data)
    
    return canonical_to_originals


def consolidate_topics(input_file: Path, output_file: Path) -> None:
    """
    Consolidate topics from all papers with normalization, merging, and counting.
    """
    # Load all topics
    with open(input_file, 'r', encoding='utf-8') as f:
        all_topics_by_paper = json.load(f)
    
    # First pass: normalize and collect all topic-paper pairs
    normalized_topics = defaultdict(list)
    
    for paper_id, topics in all_topics_by_paper.items():
        for topic in topics:
            normalized = normalize_topic(topic)
            normalized_topics[normalized].append((normalized, paper_id))
    
    # Second pass: merge similar topics (plurals, variations)
    merged_topics = merge_similar_topics(normalized_topics)
    
    # Third pass: build final consolidated structure
    topic_to_papers = defaultdict(set)
    topic_counts = defaultdict(int)
    merge_log = []  # Track what was merged
    
    for canonical_topic, paper_data in merged_topics.items():
        # Get unique original forms that mapped to this canonical
        original_forms = defaultdict(lambda: {"count": 0, "papers": set()})
        
        for orig_topic, paper_id in paper_data:
            original_forms[orig_topic]["count"] += 1
            original_forms[orig_topic]["papers"].add(paper_id)
            topic_to_papers[canonical_topic].add(paper_id)
            topic_counts[canonical_topic] += 1
        
        # Log merges if multiple forms were combined
        if len(original_forms) > 1:
            merged_details = []
            for form, data in sorted(original_forms.items()):
                merged_details.append({
                    "variant": form,
                    "count": data["count"],
                    "papers_mentioning": len(data["papers"])
                })
            
            merge_log.append({
                "canonical": canonical_topic,
                "total_count": topic_counts[canonical_topic],
                "total_papers_mentioning": len(topic_to_papers[canonical_topic]),
                "merged_from": merged_details
            })
    
    # Build output structure
    consolidated = []
    for topic in sorted(topic_to_papers.keys()):
        papers = sorted(list(topic_to_papers[topic]))
        consolidated.append({
            "topic": topic,
            "count": topic_counts[topic],
            "papers_mentioning": len(papers),
            "paper_ids": papers
        })
    
    # Sort by papers_mentioning (desc), then count (desc), then alphabetically
    consolidated.sort(key=lambda x: (-x["papers_mentioning"], -x["count"], x["topic"]))
    
    # Create summary
    output = {
        "summary": {
            "total_unique_topics": len(consolidated),
            "total_topic_mentions": sum(x["count"] for x in consolidated),
            "total_papers": len(all_topics_by_paper),
            "average_topics_per_paper": sum(x["count"] for x in consolidated) / len(all_topics_by_paper),
            "topics_merged": len(merge_log)
        },
        "topics": consolidated,
        "merge_log": merge_log
    }
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("TOPIC CONSOLIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total unique topics: {output['summary']['total_unique_topics']}")
    print(f"Total topic mentions: {output['summary']['total_topic_mentions']}")
    print(f"Total papers: {output['summary']['total_papers']}")
    print(f"Average topics per paper: {output['summary']['average_topics_per_paper']:.1f}")
    print(f"Topics merged: {output['summary']['topics_merged']}")
    print(f"\nOutput written to: {output_file}")
    
    # Show merge log
    if merge_log:
        print(f"\n{'='*70}")
        print("TOPICS MERGED (with counts)")
        print(f"{'='*70}")
        for entry in merge_log:
            print(f"  {entry['canonical']}")
            print(f"    Total: {entry['total_count']} mentions, {entry['total_papers_mentioning']} papers")
            print(f"    Merged from:")
            for variant in entry['merged_from']:
                print(f"      - {variant['variant']:40} ({variant['count']} mentions, {variant['papers_mentioning']} papers)")
    
    # Show top 20
    print(f"\n{'='*70}")
    print("TOP 20 TOPICS BY PAPERS MENTIONING")
    print(f"{'='*70}")
    for i, item in enumerate(consolidated[:20], 1):
        print(f"{i:2}. {item['topic']:45} ({item['papers_mentioning']} papers, {item['count']} mentions)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    input_file = Path("summaries_2021/all_topics.json")
    output_file = Path("consolidated_topics_with_counts.json")
    
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found!")
        exit(1)
    
    consolidate_topics(input_file, output_file)
    print("✅ Consolidation complete!")

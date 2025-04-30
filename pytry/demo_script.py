#!/usr/bin/env python3
"""
Demo script for GPTCore entity matching
This script demonstrates how to use the GPTCore to replace FasttextCore
for entity matching tasks.
"""

import os
import yaml
import argparse
from typing import Dict, Any

from triwave.datatype.workflow import WorkflowConfig, CacheMode
from triwave.file_container import RecordContainer
from triwave.datatype.record import RecordType, RecordMG
from triwave.gptcore import GPTCore  # Import the new GPTCore

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Demo script for GPTCore entity matching")
    parser.add_argument("--config", type=str, required=True, help="Path to project config file")
    parser.add_argument("--target", type=str, required=True, help="Path to target data file")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="GPT model to use")
    parser.add_argument("--output", type=str, default="matches.yaml", help="Output file for matches")
    parser.add_argument("--detailed", action="store_true", help="Get detailed match recommendations")
    args = parser.parse_args()
    
    if 'version' in project_config:
        del project_config['version']
    
    workflow_config = WorkflowConfig(**project_config)

    # Load project configuration
    project_config = load_config(args.config)
    workflow_config = WorkflowConfig(**project_config)
    
    # Define inference attributes (this would normally come from the project config)
    inf_attr = {
        "title": RecordType.TEXT_KEY,
        "author": RecordType.TEXT_EN,
        "publisher": RecordType.TEXT_JA,
        "year": RecordType.COMPLEMENT_DATE
    }
    
    # Initialize GPTCore
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key must be provided via --api-key or OPENAI_API_KEY env var")
    
    gpt_core = GPTCore(
        config=workflow_config,
        target_filepath=args.target,
        log_filepath="entity_matching.log",
        inf_attr=inf_attr,
        api_key=api_key,
        model=args.model
    )
    
    # Load target file
    print(f"Loading records from {args.target}")
    rc = RecordContainer()
    rc.load_file(args.target)
    records = rc.get_recordmg()
    print(f"Loaded {len(records)} records")
    
    # Method 1: Batch entity resolution
    print("Performing batch entity resolution...")
    matches = gpt_core.batch_entity_resolution(
        records,
        threshold=0.85,  # Adjust based on your needs
        detailed=args.detailed
    )
    
    # Print results
    print(f"\nFound {len(matches)} potential matches:")
    for i, j, similarity, details in matches[:10]:  # Show top 10
        print(f"Match: {records[i].re.id} <-> {records[j].re.id} (similarity: {similarity:.4f})")
        if details and 'explanation' in details:
            print(f"Explanation: {details['explanation']}")
        print()
    
    # Method 2: Direct comparison of two records
    if len(records) >= 2:
        print("\nDetailed comparison of two example records:")
        recommendation = gpt_core.get_similarity_recommendation(records[0], records[1])
        print(f"Records: {records[0].re.id} <-> {records[1].re.id}")
        print(f"Is match: {recommendation['is_match']}")
        print(f"Confidence: {recommendation['confidence']:.4f}")
        print(f"Explanation: {recommendation['explanation']}")
        print()
    
    # Save results to file
    output_data = {
        "matches": [
            {
                "record1": records[i].re.id,
                "record2": records[j].re.id,
                "similarity": float(similarity),
                "is_match": details.get("is_match", similarity >= 0.85),
                "confidence": float(details.get("confidence", similarity)),
                "explanation": details.get("explanation", "")
            }
            for i, j, similarity, details in matches
        ]
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Results saved to {args.output}")
    
if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# expense-categorizer/main.py
"""
Command-line interface for expense categorization using Ollama
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List

from classifier import ExpenseClassifier, SimpleClassifier
from models import TransactionResult, DEFAULT_CATEGORIES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
LOG = logging.getLogger(__name__)


def process_single(text: str, classifier, args) -> TransactionResult:
    """Process a single transaction description"""
    return classifier.classify(text, threshold=args.threshold)


def process_file(filepath: Path, classifier, args) -> List[TransactionResult]:
    """Process transaction descriptions from a file"""
    results = []

    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            # List of strings or list of objects with 'description' field
            descriptions = []
            for item in data:
                if isinstance(item, str):
                    descriptions.append(item)
                elif isinstance(item, dict) and 'description' in item:
                    descriptions.append(item['description'])
            results = classifier.classify_batch(descriptions, threshold=args.threshold)

    elif filepath.suffix in ['.txt', '.csv']:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        descriptions = [line.strip() for line in lines if line.strip()]
        results = classifier.classify_batch(descriptions, threshold=args.threshold)

    return results


def save_results(results: List[TransactionResult], output_path: Path):
    """Save classification results to file"""
    output_data = [r.to_dict() for r in results]

    if output_path.suffix == '.json':
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
    elif output_path.suffix == '.csv':
        import csv
        with open(output_path, 'w', newline='') as f:
            if output_data:
                writer = csv.DictWriter(f, fieldnames=output_data[0].keys())
                writer.writeheader()
                writer.writerows(output_data)
    else:
        # Default to JSON
        output_path = output_path.with_suffix('.json')
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

    LOG.info(f"Results saved to {output_path}")


def print_results(results: List[TransactionResult], verbose: bool = False):
    """Print classification results to console"""
    for r in results:
        if verbose:
            print(f"\nDescription: {r.description}")
            print(f"Category: {r.category}")
            if r.subcategory:
                print(f"Subcategory: {r.subcategory}")
            print(f"Confidence: {r.confidence:.2%}")
        else:
            subcat = f" ({r.subcategory})" if r.subcategory else ""
            print(f"{r.description[:50]:50} -> {r.category}{subcat} [{r.confidence:.1%}]")


def main():
    parser = argparse.ArgumentParser(
        description='Categorize expense transactions using Ollama LLM'
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '-t', '--text',
        help='Single transaction description to classify'
    )
    input_group.add_argument(
        '-f', '--file',
        type=Path,
        help='File containing transaction descriptions (JSON, TXT, or CSV)'
    )
    input_group.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode - enter descriptions one by one'
    )

    # Model options
    parser.add_argument(
        '--model',
        default='qwen3:8b',
        help='Ollama model name (default: qwen3:8b)'
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Use simple rule-based classifier (no LLM)'
    )
    parser.add_argument(
        '--ollama-host',
        default='http://10.112.30.10:11434',
        help='Ollama API host URL (default: http://10.112.30.10:11434)'
    )

    # Classification options
    parser.add_argument(
        '--categories',
        nargs='+',
        default=DEFAULT_CATEGORIES,
        help='Custom expense categories'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='Minimum confidence threshold (0-1)'
    )

    # Output options
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output file for results (JSON or CSV)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Initialize classifier
    if args.simple:
        LOG.info("Using simple rule-based classifier")
        classifier = SimpleClassifier(categories=args.categories)
    else:
        LOG.info(f"Using Ollama LLM classifier with model: {args.model}")
        try:
            classifier = ExpenseClassifier(
                model_name=args.model,
                categories=args.categories,
                ollama_host=args.ollama_host
            )
        except ConnectionError as e:
            LOG.error(f"Failed to connect to Ollama: {e}")
            LOG.info("Make sure Ollama is running (ollama serve)")
            LOG.info("Falling back to simple classifier")
            classifier = SimpleClassifier(categories=args.categories)
        except Exception as e:
            LOG.error(f"Failed to initialize classifier: {e}")
            LOG.info("Falling back to simple classifier")
            classifier = SimpleClassifier(categories=args.categories)

    results = []

    # Process input
    if args.text:
        result = process_single(args.text, classifier, args)
        results = [result]

    elif args.file:
        if not args.file.exists():
            LOG.error(f"File not found: {args.file}")
            sys.exit(1)
        results = process_file(args.file, classifier, args)

    elif args.interactive:
        print("Interactive mode. Type 'quit' to exit.")
        while True:
            try:
                text = input("\nEnter transaction description: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                if text:
                    result = process_single(text, classifier, args)
                    results.append(result)
                    print_results([result], verbose=True)
            except KeyboardInterrupt:
                print("\nExiting...")
                break

    # Output results
    if results:
        if not args.interactive:
            print_results(results, verbose=args.verbose)

        if args.output:
            save_results(results, args.output)

        # Print summary
        if len(results) > 1:
            print(f"\n=== Summary ===")
            categories_count = {}
            for r in results:
                categories_count[r.category] = categories_count.get(r.category, 0) + 1

            for cat, count in sorted(categories_count.items()):
                print(f"{cat}: {count} ({count / len(results):.1%})")


if __name__ == "__main__":
    main()
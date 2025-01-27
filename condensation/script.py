print("Setting things up...")
import os
import sys
import pandas as pd
import asyncio
import textwrap
import json
import csv
from typing import List
import argparse

parent_dir = os.path.abspath(os.path.join(os.getcwd()))
sys.path.insert(0, parent_dir)

from chatbot_api.providers.openai import OpenAIProvider
from condensation.argument import Argument
from condensation.condenser import Condenser

# Configuration
api_key = os.getenv("OPENAI_API_KEY")
model = "o1-mini-2024-09-12"  
batch_size = 20
n_comments = 100
file_name = "results_v0.txt"
save_path = os.path.join(parent_dir, 'condensation', 'results', file_name)

def export_results(arguments: List[Argument], base_path: str, formats: List[str] = ['txt', 'json', 'csv']):
    """Export results in multiple formats.
    
    Args:
        arguments: List of Argument objects to export
        base_path: Base path for the output files (without extension)
        formats: List of format strings to export to ('txt', 'json', 'csv')
    """
    for fmt in formats:
        if fmt == 'txt':
            with open(f"{base_path}.txt", 'w', encoding='utf-8') as file:
                for i, arg in enumerate(arguments, 1):
                    file.write(f"\n                                      *Argumentti {i}*\n")
                    file.write(textwrap.fill(arg.main_argument, width=125) + "\n")
                    
                    # optional: show sources
                    """
                    file.write("\nLähteet:\n")
                    for source in arg.sources:
                        file.write(textwrap.fill(source, width=125) + "\n")
                    file.write("\n" + "-"*125 + "\n") """
        
        elif fmt == 'json':
            json_data = [
                {
                    'argument_id': i,
                    'topic': arg.topic,
                    'main_argument': arg.main_argument,
                    'sources': arg.sources,
                    'source_indices': arg.source_indices
                }
                for i, arg in enumerate(arguments, 1)
            ]
            with open(f"{base_path}.json", 'w', encoding='utf-8') as file:
                json.dump(json_data, file, ensure_ascii=False, indent=2)
        
        elif fmt == 'csv':
            with open(f"{base_path}.csv", 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['argument_id', 'topic', 'main_argument', 'sources', 'source_indices'])
                for i, arg in enumerate(arguments, 1):
                    writer.writerow([
                        i,
                        arg.topic,
                        arg.main_argument,
                        '|'.join(arg.sources),
                        ','.join(map(str, arg.source_indices))
                    ])

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process and export arguments from comments.')
    parser.add_argument('--formats', nargs='+', default=['txt'],
                       choices=['txt', 'json', 'csv'],
                       help='Export formats (multiple can be specified)')
    return parser.parse_args()

# Main function
async def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize OpenAI provider
    provider = OpenAIProvider(
        api_key=api_key, 
        model=model
    )

    # Initialize condenser
    condenser = Condenser(provider)

    # Load data
    data_path = os.path.join(parent_dir, 'data', 'sources', 'kuntavaalit2021.csv')
    df = pd.read_csv(data_path)

    # Process comments
    print("Processing comments...")

    # Get comments from specific question (to do: make this dynamic)
    topic = "Kuntavero ja pääomatulovero"  # make this dynamic
    comments_with_index = [(i, comment) for i, comment in enumerate(df['q9.explanation_fi'].dropna(), 1)]
    comments = [comment for _, comment in comments_with_index]
    
    arguments = await condenser.process_comments(comments, topic, n_comments, batch_size)

    # Get base path
    base_path = os.path.splitext(save_path)[0]
    
    # Export results in specified formats
    export_results(arguments, base_path, formats=args.formats)
    
    # Create format-specific message
    format_msg = ', '.join(f'.{fmt}' for fmt in args.formats)
    print(f"Results exported to {base_path}{{{format_msg}}}")

if __name__ == "__main__":
    asyncio.run(main())
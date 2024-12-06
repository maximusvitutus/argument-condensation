import json
from typing import List, Dict
import spacy
from spacy.lang.en import English
from pypdf import PdfReader
from pathlib import Path

def load_data(file_path: str) -> list:
    """Load data from a JSON or text file."""
    with open(file_path, "r") as file:
        if file_path.endswith(".json"):
            return json.load(file)
        else:
            return [line.strip() for line in file.readlines()]

def save_data(data: dict, file_path: str) -> None:
    """Save data to a JSON file."""
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def chunk_text(filepath: str, n_words: int = 100, party_name: str = None) -> List[Dict]:
    """
    Chunks political party text into semantically meaningful pieces based on sentence boundaries.
    
    Args:
        filepath: Path to text file containing party information
        n_words: Minimum number of words per chunk (default 100)
        party_name: Name of the political party (or unknown)
    
    Returns:
        List of dictionaries containing chunked content and metadata
    """
    # Load spaCy model for better sentence boundary detection
    nlp = spacy.load("en_core_web_sm")
    
    # Read the file
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read().strip()
    
    # Process the text with spaCy
    doc = nlp(text)
    
    # Initialize variables
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    # Process sentence by sentence
    for sent in doc.sents:
        # Add sentence to current chunk
        current_chunk.append(sent.text)
        current_word_count += len(sent.text.split())
        
        # Check if chunk is complete
        if current_word_count >= n_words:
            # Create chunk content
            chunk_text = ' '.join(current_chunk)
            
            # Create metadata
            metadata = {
                "party": party_name or "Unknown",
                "word_count": current_word_count
            }
            
            # Add chunk to results
            chunks.append({
                "content": chunk_text,
                "metadata": metadata
            })
            
            # Reset for next chunk
            current_chunk = []
            current_word_count = 0
    
    # Add remaining text as final chunk
    if current_chunk and current_word_count > n_words / 2:
        chunks.append({
            "content": ' '.join(current_chunk),
            "metadata": {
                "party": party_name or "Unknown",
                "word_count": current_word_count
            }
        })
    
    return chunks


# Path from project root
def from_pdf_to_string(pdf_path: str) -> str:
    reader = PdfReader(Path(__file__).resolve().parents[2] / pdf_path)
    manifesto = ""
    for page in reader.pages:
        manifesto += page.extract_text()
    return manifesto

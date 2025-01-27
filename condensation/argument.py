from typing import List
from dataclasses import dataclass

@dataclass
class Argument:
    """A static argument class that represents a single perspective found in source comments."""
    main_argument: str          # The main argument text
    sources: List[str]          # The source comments that led to this argument
    source_indices: List[int]   # The indices of the sources in the original data
    topic: str                  # The topic this argument relates to

    def __init__(self, main_argument, sources, source_indices=None, topic=None):
        self.main_argument = main_argument
        self.sources = sources
        self.source_indices = source_indices if source_indices is not None else list(range(1, len(sources) + 1))
        self.topic = topic
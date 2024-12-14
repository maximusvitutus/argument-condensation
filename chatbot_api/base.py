from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from enum import Enum
import numpy as np
from numpy.typing import NDArray

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class Message:
    role: Role
    content: str

@dataclass
class UsageStats:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    @property
    def estimated_cost(self) -> float:
        """Calculate estimated cost based on token usage."""
        pass

@dataclass
class EmbeddingResponse:
    embedding: NDArray[np.float32]  # The actual embedding vector
    tokens: int                     # Number of tokens processed
    model: str                      # Model used for embedding
    
    def __post_init__(self):
        # Convert list to numpy array if needed
        if isinstance(self.embedding, list):
            self.embedding = np.array(self.embedding, dtype=np.float32)
    
    @property
    def estimated_cost(self) -> float:
        """Calculate cost based on OpenAI's ada v2 embedding model pricing."""
        cost_per_1k_tokens = 0.0001  # $0.0001 per 1K tokens
        return (self.tokens / 1000) * cost_per_1k_tokens
    
    def __len__(self) -> int:
        return len(self.embedding)

@dataclass
class LLMResponse:
    content: str
    usage: UsageStats
    model: str
    finish_reason: Optional[str] = None
    
    @property
    def was_truncated(self) -> bool:
        """Check if response was truncated due to length."""
        return self.finish_reason == "length"

class LLMProvider(ABC): 
    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of conversation messages
            temperature: Controls randomness (0-1)
            max_tokens: Maximum tokens to generate
            stop_sequences: Custom stop sequences
            
        Returns:
            LLMResponse object containing response and metadata
            
        Raises:
            TokenLimitException: If input exceeds model's context window
            LLMException: For other LLM-related errors
        """
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using provider's tokenizer.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        pass

    @property
    @abstractmethod
    def max_context_tokens(self) -> int:
        """Maximum tokens supported by model's context window."""
        pass

    # to do: implement such that the fitting doesn't happen according to context length but performance
    # this needs to be done because model performance drops if the input is too large
    @abstractmethod
    async def fit_comment_args_count(self) -> int:
        """Count the number of Argument class entities you can fit to the context."""
        pass


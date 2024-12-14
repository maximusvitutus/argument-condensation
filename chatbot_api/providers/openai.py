import os
import tiktoken
from openai import OpenAI, OpenAIError
from typing import List, Optional
from chatbot_api.base import LLMProvider, LLMResponse, Message, UsageStats, EmbeddingResponse
from ..base import (
    LLMProvider,
    LLMResponse,
    Message,
    UsageStats
)
from ..exceptions import (
    LLMException, 
    TokenLimitException,
)

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", embedding_model: str = "text-embedding-3-small"):
        self.api_key = api_key 
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.embedding_model = embedding_model
        self.avg_chars_per_token = 4  # Average characters per token for most ChatGPTs

    async def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role.value, "content": m.content} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
            )
            
            usage = UsageStats(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                usage=usage,
                model=self.model,
                finish_reason=response.choices[0].finish_reason
            )
            
        except OpenAIError as e:
            if "maximum context length" in str(e):
                raise TokenLimitException(str(e)) from e
            raise LLMException(str(e)) from e


    # ------TOKEN STUFF STARTS------

    async def count_tokens(self, text: str) -> int:
        encoding = tiktoken.encoding_for_model(self.model)
        try:
            token_list = encoding.encode(text)
            tokens = len(token_list)
            return tokens
            
        except OpenAIError as e:
            raise LLMException(str(e)) from e
        
    async def max_context_tokens(self) -> int:    
        if (self.model == 'gpt-4o' or self.model == 'gpt-4o-2024-11-20' or 
        self.model == 'gpt-4o-2024-08-06' or self.model == 'gpt-4o-2024-05-13' or
        self.model == 'chatgpt-4o-latest' or self.model == 'gpt-4o-mini'
        or self.model == 'gpt-4o-mini-2024-07-18'):
            return 120000 # 128k tokens for GPT-4o models, but leave a buffer (to do: get optimal token limit that is safe)
        return 4096 # Set a super safe fallback value
        
    async def fit_comment_args_count(self, comments: List[str], prompt_template: str) -> int:
        """
        Calculate how many complete messages can fit in the context window with the prompt template.
        Returns the amount of messages that fit.
        """
        template_tokens = await self.count_tokens(prompt_template.format(comments_text=""))
        max_tokens = await self.max_context_tokens()
        available_tokens = max_tokens - template_tokens
        
        if available_tokens <= 0:
            return 0
            
        total_tokens = 0
        for i, message in enumerate(comments):
            message_tokens = await self.count_tokens(message)
            if total_tokens + message_tokens > available_tokens:
                return i
            total_tokens += message_tokens
            
        return len(comments) - 10 # Leave a 10 comment buffer

    # to do: implement sentence splitting or similar more advanced trimming method
    async def truncate_to_token_limit(self, text: str) -> str:
        """
        Truncate text to fit the model's context window. 
        Currently, this method simply truncates the text on the character level.
        Returns the trimmed text. 
        """
        max_tokens = await self.max_context_tokens()
        tokens = await self.count_tokens(text)
        
        if tokens <= max_tokens:
            return text
        
        target_length = max_tokens * self.avg_chars_per_token # approximate target length 

        # make sure the text can fit with target_length
        text = text[:target_length]
        tokens = await self.count_tokens(text)

        # remove 10 characters at a time until it fits
        while tokens > max_tokens:
            text = text[:-10]
            tokens = await self.count_tokens(text)
        
        return text

    # to do: implement based on the model that is being used
    def estimated_cost(self, prompt_tokens, completion_tokens) -> float:
        input_cost_per_1k = 0.0015
        output_cost_per_1k = 0.002
        
        input_cost = (prompt_tokens / 1000) * input_cost_per_1k
        output_cost = (completion_tokens / 1000) * output_cost_per_1k
        
        return input_cost + output_cost
        
    # ------END TOKEN STUFF------



    # ------EMBEDDING STUFF STARTS------
    # to do: makes sense to move this functionality to a separate EmbeddingProvider class

    async def get_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> EmbeddingResponse:
        """
        Get embedding vector for a text string.
        
        Args:
            text: Text to get embedding for
            model: Optional model override (defaults to self.embedding_model)
            
        Returns:
            EmbeddingResponse containing the embedding vector and metadata
            
        Raises:
            LLMException: If the API call fails
        """
        try:
            response = self.client.embeddings.create(
                model=model or self.embedding_model,
                input=text
            )
            
            return EmbeddingResponse(
                embedding=response.data[0].embedding,
                tokens=response.usage.total_tokens,
                model=response.model
            )
            
        except Exception as e:
            raise LLMException(f"OpenAI embedding error: {str(e)}") from e
    
    async def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[EmbeddingResponse]:
        """
        Get embedding vectors for multiple texts.
        
        Args:
            texts: List of texts to get embeddings for
            model: Optional model override (defaults to self.embedding_model)
            
        Returns:
            List of EmbeddingResponse objects
            
        Raises:
            LLMException: If the API call fails
        """
        try:
            response = self.client.embeddings.create(
                model=model or self.embedding_model,
                input=texts
            )
            
            # Create EmbeddingResponse objects for each embedding
            results = []
            tokens_per_response = response.usage.total_tokens // len(texts)
            
            for data in response.data:
                results.append(
                    EmbeddingResponse(
                        embedding=data.embedding,
                        tokens=tokens_per_response,  # Approximate tokens per text
                        model=response.model
                    )
                )
            
            return results
            
        except Exception as e:
            raise LLMException(f"OpenAI embedding error: {str(e)}") from e

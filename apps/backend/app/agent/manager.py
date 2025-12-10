import os
from typing import Dict, Any

from ..core import settings
from .strategies.wrapper import JSONWrapper, MDWrapper
from .providers.base import Provider, EmbeddingProvider

class AgentManager:
    def __init__(self,
                 strategy: str | None = None,
                 model: str = settings.LL_MODEL,
                 model_provider: str = settings.LLM_PROVIDER
                 ) -> None:
        match strategy:
            case "md":
                self.strategy = MDWrapper()
            case "json":
                self.strategy = JSONWrapper()
            case _:
                self.strategy = JSONWrapper()
        self.model = model
        self.model_provider = model_provider

    async def _get_provider(self, **kwargs: Any) -> Provider:
        # Default options for any LLM. Not all can handle them
        # (e.g. OpenAI doesn't take top_k) but each provider can make
        # best effort.
        opts = {
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
        }
        opts.update(kwargs)
        match self.model_provider:
            case 'openai':
                from .providers.openai import OpenAIProvider
                api_key = opts.get("llm_api_key", settings.LLM_API_KEY)
                return OpenAIProvider(model_name=self.model,
                                      api_key=api_key,
                                      opts=opts)
            case 'ollama':
                from .providers.ollama import OllamaProvider
                model = opts.get("model", self.model)
                return OllamaProvider(model_name=model,
                                      opts=opts)
            case _:
                from .providers.llama_index import LlamaIndexProvider
                llm_api_key = opts.get("llm_api_key", settings.LLM_API_KEY)
                llm_api_base_url = opts.get("llm_base_url", settings.LLM_BASE_URL)
                return LlamaIndexProvider(api_key=llm_api_key,
                                          model_name=self.model,
                                          api_base_url=llm_api_base_url,
                                          provider=self.model_provider,
                                          opts=opts)

    async def run(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Run the agent with the given prompt and generation arguments.
        """
        provider = await self._get_provider(**kwargs)
        return await self.strategy(prompt, provider, **kwargs)

class EmbeddingManager:
    """
    Manages embedding generation for CV-to-job matching.
    
    IMPORTANT: This project uses LOCAL EMBEDDINGS ONLY.
    
    Default provider: local_tfidf (no external API required)
    
    External embedding APIs (Ollama, OpenAI, Voyage, etc.) are DISABLED.
    Attempting to use them will raise an error to prevent accidental API calls.
    """
    
    def __init__(self,
                 model: str = settings.EMBEDDING_MODEL or "",
                 model_provider: str = settings.EMBEDDING_PROVIDER or "local_tfidf") -> None:
        # Hard default to local_tfidf if provider is empty or None
        self._model = model
        self._model_provider = (model_provider or "local_tfidf").lower().strip()

    async def _get_embedding_provider(
        self, **kwargs: Any
    ) -> EmbeddingProvider:
        # Route to the correct provider based on configuration
        # Default: local_tfidf (no external deps, always works)
        # 
        # SECURITY: External embedding APIs are DISABLED to prevent:
        #   1. Accidental API costs
        #   2. Network dependency failures
        #   3. Data leakage to external services
        #
        match self._model_provider:
            case 'openai':
                # DISABLED: External API not allowed
                raise RuntimeError(
                    "External embedding API is disabled; use local_tfidf. "
                    "Set EMBEDDING_PROVIDER='local_tfidf' in .env"
                )
            case 'ollama':
                # DISABLED: External API not allowed
                raise RuntimeError(
                    "External embedding API is disabled; use local_tfidf. "
                    "Set EMBEDDING_PROVIDER='local_tfidf' in .env"
                )
            case 'local_tfidf' | _:
                # Default case: always fall back to local_tfidf
                # This ensures we never accidentally call external APIs
                from .providers.local_tfidf import LocalTfidfEmbeddingProvider
                return LocalTfidfEmbeddingProvider()

    async def embed(self, text: str, **kwargs: Any) -> list[float]:
        """
        Get the embedding for the given text.
        """
        provider = await self._get_embedding_provider(**kwargs)
        return await provider.embed(text)

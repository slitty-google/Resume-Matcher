import asyncio
import logging
import ollama
from ollama._types import ResponseError as OllamaResponseError

from typing import Any, Dict, List, Optional
from fastapi.concurrency import run_in_threadpool

from ..exceptions import ProviderError, EmbeddingProviderError
from .base import Provider, EmbeddingProvider
from ...core import settings

logger = logging.getLogger(__name__)


class OllamaBaseProvider:
    """Base class for Ollama providers with shared model management utilities."""
    
    _client: ollama.Client
    
    @staticmethod
    async def _get_installed_models(host: Optional[str] = None) -> List[str]:
        """List all installed models."""
        def _list_sync() -> List[str]:
            client = ollama.Client(host=host) if host else ollama.Client()
            return [model_class.model for model_class in client.list().models]
        return await run_in_threadpool(_list_sync)

    def _ensure_model_pulled(self, model_name: str) -> None:
        """
        Ensure model is available locally.
        - If it's already in /api/tags, skip pulling.
        - If pull fails but model is actually present, continue.
        - Raises ProviderError with clear message if model unavailable.
        """
        # First check if model is already installed
        try:
            installed = [m.model for m in self._client.list().models]
            # Check both exact match and partial match (e.g., "nomic-embed-text" vs "nomic-embed-text:latest")
            if model_name in installed or any(m.startswith(model_name) for m in installed):
                logger.debug(f"Ollama model '{model_name}' already installed")
                return
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
            # If listing fails, we'll try pulling below
            pass

        # Try to pull the model
        try:
            logger.info(f"Pulling Ollama model '{model_name}'...")
            self._client.pull(model_name)
            logger.info(f"Successfully pulled Ollama model '{model_name}'")
            return
        except Exception as e:
            # If pull failed (e.g., offline), check again—maybe it actually exists locally
            try:
                installed = [m.model for m in self._client.list().models]
                if model_name in installed or any(m.startswith(model_name) for m in installed):
                    logger.debug(f"Ollama model '{model_name}' found after failed pull")
                    return
            except Exception:
                pass
            
            error_msg = (
                f"Ollama model '{model_name}' is unavailable. "
                f"Please run 'ollama pull {model_name}' and restart the backend. "
                f"Original error: {e}"
            )
            logger.error(error_msg)
            raise ProviderError(error_msg) from e


class OllamaProvider(Provider, OllamaBaseProvider):
    """Ollama LLM provider for text generation."""
    
    def __init__(
        self,
        model_name: str = settings.LL_MODEL,
        api_base_url: Optional[str] = settings.LLM_BASE_URL,
        opts: Optional[Dict[str, Any]] = None
    ):
        self.opts = opts or {}
        self.model = model_name
        self._client = ollama.Client(host=api_base_url) if api_base_url else ollama.Client()
        self._ensure_model_pulled(model_name)

    def _generate_sync(self, prompt: str, options: Dict[str, Any]) -> str:
        """Generate a response from the model synchronously."""
        try:
            response = self._client.generate(
                prompt=prompt,
                model=self.model,
                options=options,
            )
            return response["response"].strip()
        except OllamaResponseError as e:
            logger.error(f"Ollama generation error: status={e.status_code}, message={e}")
            raise ProviderError(f"Ollama - Error generating response: {e}") from e
        except Exception as e:
            logger.error(f"Ollama sync error: {e}")
            raise ProviderError(f"Ollama - Error generating response: {e}") from e

    async def __call__(self, prompt: str, **generation_args: Any) -> str:
        if generation_args:
            logger.warning(f"OllamaProvider ignoring generation_args {generation_args}")
        return await run_in_threadpool(self._generate_sync, prompt, self.opts)


class OllamaEmbeddingProvider(EmbeddingProvider, OllamaBaseProvider):
    """
    Ollama embedding provider with robust error handling and concurrency protection.
    
    Features:
    - Single long-lived client instance
    - Asyncio lock to serialize requests (prevents EOF errors from concurrent calls)
    - Specific exception type (EmbeddingProviderError) for embedding failures
    - Detailed error logging
    - Healthcheck method for startup validation
    """
    
    # Class-level lock shared across all instances to prevent concurrent Ollama requests
    _embed_lock: asyncio.Lock = asyncio.Lock()
    
    def __init__(
        self,
        embedding_model: str = settings.EMBEDDING_MODEL,
        api_base_url: Optional[str] = settings.EMBEDDING_BASE_URL,
    ):
        self._model = embedding_model
        self._host = api_base_url
        self._client = ollama.Client(host=api_base_url) if api_base_url else ollama.Client()
        self._ensure_model_pulled(embedding_model)
    
    def _embed_sync(self, text: str) -> List[float]:
        """
        Synchronous embedding call - runs in threadpool.
        Returns the embedding vector as a list of floats.
        """
        response = self._client.embed(
            model=self._model,
            input=text,
        )
        
        # Handle both response shapes from ollama client:
        # - {"embedding": [...]} for single input
        # - {"embeddings": [[...]]} for batch input
        if hasattr(response, "embedding") and response.embedding:
            return list(response.embedding)
        if hasattr(response, "embeddings") and response.embeddings:
            return list(response.embeddings[0])
        
        # Also handle dict-style response
        if isinstance(response, dict):
            if "embedding" in response and response["embedding"]:
                return list(response["embedding"])
            if "embeddings" in response and response["embeddings"]:
                return list(response["embeddings"][0])
        
        raise EmbeddingProviderError("Ollama returned empty embedding response")

    async def embed(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        
        Uses a class-level lock to serialize requests, preventing the EOF errors
        that occur when Ollama receives concurrent embedding requests.
        
        Raises:
            EmbeddingProviderError: If embedding generation fails
        """
        async with self._embed_lock:
            try:
                return await run_in_threadpool(self._embed_sync, text)
            except OllamaResponseError as e:
                error_msg = f"Ollama embedding failed: status={e.status_code}, message={e}"
                logger.error(error_msg)
                raise EmbeddingProviderError(error_msg) from e
            except EmbeddingProviderError:
                # Re-raise our own exceptions
                raise
            except Exception as e:
                error_msg = f"Ollama embedding error: {e}"
                logger.error(error_msg)
                raise EmbeddingProviderError(error_msg) from e
    
    async def healthcheck(self) -> None:
        """
        Verify the embedding provider is operational.
        
        Checks:
        1. Model is pulled and available
        2. Can successfully generate an embedding for a trivial input
        
        Raises:
            EmbeddingProviderError: If the healthcheck fails
        """
        # Re-verify model is available
        try:
            installed = [m.model for m in self._client.list().models]
            model_found = (
                self._model in installed or 
                any(m.startswith(self._model) for m in installed)
            )
            if not model_found:
                raise EmbeddingProviderError(
                    f"Ollama embedding model '{self._model}' not found. "
                    f"Available models: {installed}. "
                    f"Please run 'ollama pull {self._model}' and restart."
                )
        except EmbeddingProviderError:
            raise
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to list Ollama models: {e}") from e
        
        # Try a trivial embedding
        try:
            result = await self.embed("healthcheck ping")
            if not result or len(result) == 0:
                raise EmbeddingProviderError("Healthcheck embedding returned empty vector")
            logger.info(f"Ollama embedding healthcheck passed: {len(result)} dimensions")
        except EmbeddingProviderError:
            raise
        except Exception as e:
            raise EmbeddingProviderError(f"Ollama embedding healthcheck failed: {e}") from e

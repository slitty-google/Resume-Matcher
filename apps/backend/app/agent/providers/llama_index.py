"""
LlamaIndex Provider Integration

This module provides LLM integration via LlamaIndex's provider abstraction.

=============================================================================
ANTHROPIC KWARGS REFERENCE
=============================================================================

When using LLM_PROVIDER="llama_index.llms.anthropic.Anthropic", only the
following kwargs are supported by the Anthropic constructor:

    ALLOWED KWARGS:
    ---------------
    - model (str)           : Model name, e.g., "claude-sonnet-4-5"
    - api_key (str)         : Anthropic API key (sk-ant-...)
    - base_url (str)        : Optional API base URL
    - temperature (float)   : Sampling temperature (0.0-1.0)
    - max_tokens (int)      : Maximum tokens in response

    FORBIDDEN KWARGS (will cause TypeError):
    -----------------------------------------
    - model_name            : Use 'model' instead
    - token                 : Use 'api_key' instead
    - is_chat_model         : Not supported by Anthropic
    - is_function_calling_model : Not supported
    - context_window        : Not supported
    - num_output            : Use 'max_tokens' instead

If you encounter constructor errors, check that only allowed kwargs are passed.
The LlamaIndexProvider class below implements this filtering automatically.

=============================================================================
"""

import logging

from typing import Any, Dict, List
from fastapi.concurrency import run_in_threadpool
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM

from ..exceptions import ProviderError
from .base import Provider, EmbeddingProvider
from ...core import settings

logger = logging.getLogger(__name__)

def _get_real_provider(provider_name):
    # The format this method expects is something like:
    # llama_index.llms.openai_like.OpenAILike
    # llama_index.embeddings.openai_like.OpenAILikeEmbedding
    if not isinstance(provider_name, str):
        raise ValueError("provider_name must be a string denoting a fully-qualified Python class name")
    dotpos = provider_name.rfind('.')
    if dotpos < 0:
        raise ValueError("provider_name not correctly formatted")
    classname = provider_name[dotpos+1:]
    modname = provider_name[:dotpos]
    from importlib import import_module
    rm = import_module(modname)
    return getattr(rm, classname), modname, classname

class LlamaIndexProvider(Provider):
    def __init__(self,
                 api_key: str = settings.LLM_API_KEY,
                 api_base_url: str = settings.LLM_BASE_URL,
                 model_name: str = settings.LL_MODEL,
                 provider: str = settings.LLM_PROVIDER,
                 opts: Dict[str, Any] = None):
        if opts is None:
            opts = {}
        self.opts = opts
        self._api_key = api_key
        self._api_base_url = api_base_url
        self._model = model_name
        self._provider = provider
        if not provider:
            raise ValueError("Provider string is required")
        provider_obj, self._modname, self._classname = _get_real_provider(provider)
        if not issubclass(provider_obj, BaseLLM):
            raise TypeError("LLM provider must be e.g. a llama_index.llms.* class - a subclass of llama_index.core.base.llms.base.BaseLLM")

        # Minimal kwargs that Anthropic (and most LlamaIndex LLM integrations) accept.
        kwargs_for_provider = {
            'model': model_name,
            'api_key': api_key,
        }
        if api_base_url:
            kwargs_for_provider['base_url'] = api_base_url
        # Add temperature and max_tokens from opts if present
        if opts.get('temperature') is not None:
            kwargs_for_provider['temperature'] = opts['temperature']
        if opts.get('max_tokens') is not None:
            kwargs_for_provider['max_tokens'] = opts['max_tokens']
        try:
            self._client = provider_obj(**kwargs_for_provider)
        except TypeError as e:
            # Fallback for providers that still expect `model_name` instead of `model`.
            if 'model' in str(e) or 'unexpected keyword argument' in str(e):
                legacy_kwargs = {**kwargs_for_provider}
                legacy_kwargs.pop('model', None)
                legacy_kwargs['model_name'] = model_name
                self._client = provider_obj(**legacy_kwargs)
            else:
                raise

    def _generate_sync(self, prompt: str, **options) -> str:
        """
        Generate a response from the model.
        """
        try:
            cr = self._client.complete(prompt)
            return cr.text
        except Exception as e:
            logger.error(f"llama_index sync error: {e}")
            raise ProviderError(f"llama_index - Error generating response: {e}") from e

    async def __call__(self, prompt: str, **generation_args: Any) -> str:
        if generation_args:
            logger.warning(f"LlamaIndexProvider ignoring generation_args: {generation_args}")
        return await run_in_threadpool(self._generate_sync, prompt)

class LlamaIndexEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        embedding_model: str = settings.EMBEDDING_MODEL,
        api_key: str = settings.EMBEDDING_API_KEY,
        api_base_url: str = settings.EMBEDDING_BASE_URL,
        provider: str = settings.EMBEDDING_PROVIDER):

        self._model = embedding_model
        self._provider = provider
        self._api_key = api_key
        self._api_base_url = api_base_url
        provider_obj, self._modname, self._classname = _get_real_provider(provider)
        if not issubclass(provider_obj, BaseEmbedding):
            raise TypeError("Embedding provider must be e.g. a llama_index.embeddings.* class - a subclass of llama_index.core.base.embeddings.base.BaseEmbedding")
        # Minimal kwargs that work across LlamaIndex embedding integrations.
        kwargs_for_provider = {
            'model': embedding_model,
            'api_key': self._api_key,
        }
        if self._api_base_url:
            kwargs_for_provider['base_url'] = \
                kwargs_for_provider['api_base'] = self._api_base_url
        kwargs_for_provider['context_window'] = \
            kwargs_for_provider["max_tokens"] = kwargs_for_provider.get('num_ctx', 20000)

        self._client = provider_obj(**kwargs_for_provider)

    async def embed(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        """
        try:
            return await run_in_threadpool(self._client.get_text_embedding, text)
        except Exception as e:
            logger.error(f"llama_index embedding error: {e}")
            raise ProviderError(f"llama_index - Error generating embedding: {e}") from e

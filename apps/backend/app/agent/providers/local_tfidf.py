"""
Local TF-IDF Embedding Provider

A zero-external-dependency embedding provider that uses sklearn's TF-IDF vectorizer
to generate dense embedding vectors. This is useful as a fallback when:
- Ollama is not installed or not running
- No paid embedding API keys are available
- Quick local development/testing is needed

The embeddings are not as semantically rich as neural embeddings (e.g., from OpenAI
or Ollama models), but they are sufficient for basic keyword matching and scoring.

Usage:
    Set EMBEDDING_PROVIDER="local_tfidf" in .env
"""

import logging
import hashlib
from typing import List, Optional

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)

# Target embedding dimension - we'll pad/truncate to this size for consistency
DEFAULT_EMBEDDING_DIM = 768


class LocalTfidfEmbeddingProvider(EmbeddingProvider):
    """
    Local TF-IDF based embedding provider.
    
    Generates embeddings using character n-grams and word features,
    producing a fixed-size vector suitable for cosine similarity comparisons.
    
    This provider requires no external APIs or services - it runs entirely locally
    using only Python standard library + numpy (which is already a dependency).
    
    Features:
    - No external API calls
    - No model downloads required
    - Consistent vector dimensions
    - Fast execution
    
    Limitations:
    - Less semantically rich than neural embeddings
    - No understanding of synonyms or context
    - Better suited for keyword matching than semantic search
    """
    
    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        **kwargs
    ):
        self._embedding_dim = embedding_dim
        logger.info(f"Initialized LocalTfidfEmbeddingProvider with dim={embedding_dim}")
    
    def _text_to_features(self, text: str) -> List[str]:
        """
        Extract features from text for embedding.
        Uses a combination of:
        - Lowercased words
        - Character 3-grams
        - Word bigrams
        """
        text = text.lower().strip()
        
        # Word tokens
        words = text.split()
        
        # Character n-grams (3-grams)
        char_ngrams = []
        for i in range(len(text) - 2):
            char_ngrams.append(text[i:i+3])
        
        # Word bigrams
        word_bigrams = []
        for i in range(len(words) - 1):
            word_bigrams.append(f"{words[i]}_{words[i+1]}")
        
        return words + char_ngrams + word_bigrams
    
    def _hash_feature(self, feature: str, dim: int) -> int:
        """Hash a feature string to an index in [0, dim)."""
        h = hashlib.md5(feature.encode('utf-8')).hexdigest()
        return int(h, 16) % dim
    
    def _embed_sync(self, text: str) -> List[float]:
        """
        Generate a fixed-size embedding vector for the text.
        
        Uses feature hashing (the "hashing trick") to map variable-length
        feature sets to a fixed-size vector.
        """
        import numpy as np
        
        # Initialize zero vector
        vector = np.zeros(self._embedding_dim, dtype=np.float32)
        
        # Extract features
        features = self._text_to_features(text)
        
        if not features:
            # Return zero vector for empty text
            return vector.tolist()
        
        # Hash each feature to a dimension and increment
        for feature in features:
            idx = self._hash_feature(feature, self._embedding_dim)
            # Use a simple TF-like weighting
            vector[idx] += 1.0
        
        # Apply log normalization (like TF-IDF without IDF)
        vector = np.log1p(vector)
        
        # L2 normalize for cosine similarity
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.tolist()
    
    async def embed(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        
        This is a synchronous operation that runs locally,
        but we keep the async interface for compatibility.
        """
        try:
            return self._embed_sync(text)
        except Exception as e:
            logger.error(f"Local TF-IDF embedding error: {e}")
            # Import here to avoid circular imports
            from ..exceptions import EmbeddingProviderError
            raise EmbeddingProviderError(f"Local TF-IDF embedding failed: {e}") from e
    
    async def healthcheck(self) -> None:
        """
        Verify the embedding provider is operational.
        
        For local TF-IDF, this just verifies we can generate an embedding.
        """
        try:
            result = await self.embed("healthcheck ping")
            if not result or len(result) != self._embedding_dim:
                from ..exceptions import EmbeddingProviderError
                raise EmbeddingProviderError(
                    f"Local TF-IDF healthcheck failed: expected {self._embedding_dim} dims, "
                    f"got {len(result) if result else 0}"
                )
            logger.info(f"Local TF-IDF embedding healthcheck passed: {len(result)} dimensions")
        except Exception as e:
            from ..exceptions import EmbeddingProviderError
            raise EmbeddingProviderError(f"Local TF-IDF healthcheck failed: {e}") from e

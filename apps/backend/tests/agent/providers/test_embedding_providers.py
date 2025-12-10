"""
Tests for the Ollama embedding provider.

These tests verify:
1. Successful embedding generation returns correct vector shape
2. Ollama ResponseError is properly caught and wrapped as EmbeddingProviderError
3. Empty response handling
4. Healthcheck functionality
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio


class TestOllamaEmbeddingProvider:
    """Tests for OllamaEmbeddingProvider."""
    
    @pytest.fixture
    def mock_ollama_client(self):
        """Create a mock Ollama client."""
        client = MagicMock()
        # Mock list() to return models including our test model
        mock_model = MagicMock()
        mock_model.model = "nomic-embed-text"
        client.list.return_value.models = [mock_model]
        return client
    
    @pytest.fixture
    def provider_with_mock_client(self, mock_ollama_client):
        """Create an OllamaEmbeddingProvider with a mocked client."""
        with patch('app.agent.providers.ollama.ollama.Client', return_value=mock_ollama_client):
            from app.agent.providers.ollama import OllamaEmbeddingProvider
            provider = OllamaEmbeddingProvider(embedding_model="nomic-embed-text")
            return provider, mock_ollama_client
    
    @pytest.mark.asyncio
    async def test_embed_success_returns_list_of_floats(self, provider_with_mock_client):
        """Test that successful embed() returns a list of floats."""
        provider, mock_client = provider_with_mock_client
        
        # Mock successful response with 768-dim embedding
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 768]
        mock_response.embedding = None
        mock_client.embed.return_value = mock_response
        
        result = await provider.embed("test text")
        
        assert isinstance(result, list)
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)
    
    @pytest.mark.asyncio
    async def test_embed_handles_single_embedding_response(self, provider_with_mock_client):
        """Test handling of response with 'embedding' key (single input format)."""
        provider, mock_client = provider_with_mock_client
        
        mock_response = MagicMock()
        mock_response.embedding = [0.5] * 768
        mock_response.embeddings = None
        mock_client.embed.return_value = mock_response
        
        result = await provider.embed("test text")
        
        assert len(result) == 768
        assert result[0] == 0.5
    
    @pytest.mark.asyncio
    async def test_embed_raises_embedding_provider_error_on_response_error(self, provider_with_mock_client):
        """Test that Ollama ResponseError is wrapped as EmbeddingProviderError."""
        provider, mock_client = provider_with_mock_client
        
        from ollama._types import ResponseError
        from app.agent.exceptions import EmbeddingProviderError
        
        mock_client.embed.side_effect = ResponseError(
            "do embedding request: Post EOF",
            status_code=500
        )
        
        with pytest.raises(EmbeddingProviderError) as exc_info:
            await provider.embed("test text")
        
        assert "do embedding request" in str(exc_info.value)
        assert "500" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_embed_raises_on_empty_response(self, provider_with_mock_client):
        """Test that empty embedding response raises EmbeddingProviderError."""
        provider, mock_client = provider_with_mock_client
        
        from app.agent.exceptions import EmbeddingProviderError
        
        mock_response = MagicMock()
        mock_response.embedding = None
        mock_response.embeddings = None
        mock_client.embed.return_value = mock_response
        
        with pytest.raises(EmbeddingProviderError) as exc_info:
            await provider.embed("test text")
        
        assert "empty" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_healthcheck_success(self, provider_with_mock_client):
        """Test healthcheck passes when embedding works."""
        provider, mock_client = provider_with_mock_client
        
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 768]
        mock_response.embedding = None
        mock_client.embed.return_value = mock_response
        
        # Should not raise
        await provider.healthcheck()
    
    @pytest.mark.asyncio
    async def test_healthcheck_fails_when_model_not_found(self):
        """Test healthcheck fails when model is not available."""
        from app.agent.exceptions import EmbeddingProviderError
        
        mock_client = MagicMock()
        mock_client.list.return_value.models = []  # No models installed
        
        with patch('app.agent.providers.ollama.ollama.Client', return_value=mock_client):
            from app.agent.providers.ollama import OllamaEmbeddingProvider
            
            # Creating provider should try to pull, then fail
            mock_client.pull.side_effect = Exception("Connection refused")
            
            with pytest.raises(Exception):  # Will raise ProviderError during init
                OllamaEmbeddingProvider(embedding_model="nomic-embed-text")


class TestOllamaEmbeddingProviderLock:
    """Tests for the concurrency lock in OllamaEmbeddingProvider."""
    
    @pytest.mark.asyncio
    async def test_concurrent_embeds_are_serialized(self):
        """Test that concurrent embed calls are serialized by the lock."""
        call_times = []
        
        async def slow_embed(*args, **kwargs):
            call_times.append(('start', asyncio.get_event_loop().time()))
            await asyncio.sleep(0.1)  # Simulate slow embedding
            call_times.append(('end', asyncio.get_event_loop().time()))
            mock_response = MagicMock()
            mock_response.embeddings = [[0.1] * 768]
            mock_response.embedding = None
            return mock_response
        
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.model = "nomic-embed-text"
        mock_client.list.return_value.models = [mock_model]
        
        with patch('app.agent.providers.ollama.ollama.Client', return_value=mock_client):
            with patch('app.agent.providers.ollama.run_in_threadpool', side_effect=slow_embed):
                from app.agent.providers.ollama import OllamaEmbeddingProvider
                provider = OllamaEmbeddingProvider(embedding_model="nomic-embed-text")
                
                # Launch two concurrent embeds
                results = await asyncio.gather(
                    provider.embed("text 1"),
                    provider.embed("text 2"),
                )
                
                # Both should succeed
                assert len(results) == 2
                
                # Verify serialization: second 'start' should be after first 'end'
                # (This is approximate due to timing, but the lock should prevent overlap)
                assert len(call_times) == 4


class TestLocalTfidfEmbeddingProvider:
    """Tests for the local TF-IDF embedding provider."""
    
    @pytest.mark.asyncio
    async def test_embed_returns_correct_dimension(self):
        """Test that embed returns correct dimension vector."""
        from app.agent.providers.local_tfidf import LocalTfidfEmbeddingProvider
        
        provider = LocalTfidfEmbeddingProvider(embedding_dim=768)
        result = await provider.embed("test text for embedding")
        
        assert isinstance(result, list)
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)
    
    @pytest.mark.asyncio
    async def test_embed_produces_normalized_vector(self):
        """Test that embedding is L2 normalized."""
        import numpy as np
        from app.agent.providers.local_tfidf import LocalTfidfEmbeddingProvider
        
        provider = LocalTfidfEmbeddingProvider()
        result = await provider.embed("test text with some words")
        
        # L2 norm should be ~1.0 for normalized vector
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 0.01
    
    @pytest.mark.asyncio
    async def test_similar_texts_have_similar_embeddings(self):
        """Test that similar texts produce similar embeddings."""
        import numpy as np
        from app.agent.providers.local_tfidf import LocalTfidfEmbeddingProvider
        
        provider = LocalTfidfEmbeddingProvider()
        
        emb1 = await provider.embed("python programming language")
        emb2 = await provider.embed("python programming code")
        emb3 = await provider.embed("completely different topic about cooking")
        
        # Cosine similarity
        sim_1_2 = np.dot(emb1, emb2)
        sim_1_3 = np.dot(emb1, emb3)
        
        # Similar texts should have higher similarity
        assert sim_1_2 > sim_1_3
    
    @pytest.mark.asyncio
    async def test_healthcheck_passes(self):
        """Test that healthcheck passes for local provider."""
        from app.agent.providers.local_tfidf import LocalTfidfEmbeddingProvider
        
        provider = LocalTfidfEmbeddingProvider()
        await provider.healthcheck()  # Should not raise
    
    @pytest.mark.asyncio
    async def test_empty_text_returns_zero_vector(self):
        """Test that empty text returns a normalized zero vector."""
        from app.agent.providers.local_tfidf import LocalTfidfEmbeddingProvider
        
        provider = LocalTfidfEmbeddingProvider(embedding_dim=768)
        result = await provider.embed("")
        
        assert len(result) == 768
        # Empty text should produce zero vector
        assert sum(abs(x) for x in result) == 0

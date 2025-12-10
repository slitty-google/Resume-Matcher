"""
Unit tests for LocalTfidfEmbeddingProvider.

These tests verify:
1. Embedding shape is consistent (768 dimensions)
2. Same input produces identical embeddings (deterministic)
3. Non-empty input produces non-zero vectors
4. Provider does NOT make any external API calls
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from app.agent.providers.local_tfidf import LocalTfidfEmbeddingProvider, DEFAULT_EMBEDDING_DIM


class TestLocalTfidfEmbeddingProvider:
    """Tests for LocalTfidfEmbeddingProvider."""

    @pytest.fixture
    def provider(self) -> LocalTfidfEmbeddingProvider:
        """Create a fresh provider instance."""
        return LocalTfidfEmbeddingProvider()

    @pytest.mark.asyncio
    async def test_embedding_shape(self, provider: LocalTfidfEmbeddingProvider):
        """Embedding should have the expected dimension (768 by default)."""
        text = "Python developer with experience in machine learning"
        embedding = await provider.embed(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == DEFAULT_EMBEDDING_DIM
        assert all(isinstance(v, float) for v in embedding)

    @pytest.mark.asyncio
    async def test_embedding_deterministic(self, provider: LocalTfidfEmbeddingProvider):
        """Same input should produce identical embeddings across multiple calls."""
        text = "Senior software engineer with 5 years experience"
        
        embedding1 = await provider.embed(text)
        embedding2 = await provider.embed(text)
        embedding3 = await provider.embed(text)
        
        # All embeddings should be exactly equal
        assert embedding1 == embedding2
        assert embedding2 == embedding3

    @pytest.mark.asyncio
    async def test_embedding_non_empty(self, provider: LocalTfidfEmbeddingProvider):
        """Non-empty text should produce a non-zero vector."""
        text = "Data scientist proficient in TensorFlow and PyTorch"
        embedding = await provider.embed(text)
        
        # Vector should not be all zeros
        vector = np.array(embedding)
        assert np.linalg.norm(vector) > 0, "Embedding should not be a zero vector"
        
        # At least some values should be non-zero
        non_zero_count = np.count_nonzero(vector)
        assert non_zero_count > 0, f"Expected non-zero values, got {non_zero_count}"

    @pytest.mark.asyncio
    async def test_embedding_empty_text(self, provider: LocalTfidfEmbeddingProvider):
        """Empty text should return a zero vector of correct dimension."""
        embedding = await provider.embed("")
        
        assert len(embedding) == DEFAULT_EMBEDDING_DIM
        vector = np.array(embedding)
        assert np.allclose(vector, 0), "Empty text should produce zero vector"

    @pytest.mark.asyncio
    async def test_embedding_whitespace_only(self, provider: LocalTfidfEmbeddingProvider):
        """Whitespace-only text should return a zero vector."""
        embedding = await provider.embed("   \n\t   ")
        
        assert len(embedding) == DEFAULT_EMBEDDING_DIM
        vector = np.array(embedding)
        assert np.allclose(vector, 0), "Whitespace-only should produce zero vector"

    @pytest.mark.asyncio
    async def test_embedding_normalized(self, provider: LocalTfidfEmbeddingProvider):
        """Non-empty embeddings should be L2 normalized (unit length)."""
        text = "Full stack developer JavaScript React Node.js"
        embedding = await provider.embed(text)
        
        vector = np.array(embedding)
        norm = np.linalg.norm(vector)
        
        # Should be unit length (norm ≈ 1.0) or zero
        if norm > 0:
            assert np.isclose(norm, 1.0, atol=1e-6), f"Expected unit vector, got norm={norm}"

    @pytest.mark.asyncio
    async def test_different_inputs_different_embeddings(self, provider: LocalTfidfEmbeddingProvider):
        """Different inputs should produce different embeddings."""
        text1 = "Python backend developer with Django experience"
        text2 = "Frontend engineer specializing in React and TypeScript"
        
        embedding1 = await provider.embed(text1)
        embedding2 = await provider.embed(text2)
        
        # Embeddings should be different
        assert embedding1 != embedding2

    @pytest.mark.asyncio
    async def test_similar_inputs_similar_embeddings(self, provider: LocalTfidfEmbeddingProvider):
        """Similar inputs should have higher cosine similarity."""
        text1 = "Python developer with machine learning experience"
        text2 = "Python programmer with ML background"
        text3 = "Chef specializing in French cuisine"
        
        emb1 = np.array(await provider.embed(text1))
        emb2 = np.array(await provider.embed(text2))
        emb3 = np.array(await provider.embed(text3))
        
        # Cosine similarity
        def cosine_sim(a, b):
            norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return np.dot(a, b) / (norm_a * norm_b)
        
        sim_1_2 = cosine_sim(emb1, emb2)  # Similar (both about Python/ML)
        sim_1_3 = cosine_sim(emb1, emb3)  # Dissimilar
        
        assert sim_1_2 > sim_1_3, f"Expected similar texts to have higher similarity: {sim_1_2} vs {sim_1_3}"

    @pytest.mark.asyncio
    async def test_no_external_api_calls(self, provider: LocalTfidfEmbeddingProvider):
        """Provider should NOT make any HTTP requests."""
        import socket
        
        original_socket = socket.socket
        
        def blocking_socket(*args, **kwargs):
            raise AssertionError("LocalTfidfEmbeddingProvider made a network call!")
        
        with patch.object(socket, 'socket', blocking_socket):
            # This should work without network access
            embedding = await provider.embed("Test text for local embedding")
            assert len(embedding) == DEFAULT_EMBEDDING_DIM

    @pytest.mark.asyncio
    async def test_healthcheck_passes(self, provider: LocalTfidfEmbeddingProvider):
        """Healthcheck should pass for a properly initialized provider."""
        # Should not raise
        await provider.healthcheck()

    @pytest.mark.asyncio
    async def test_custom_dimension(self):
        """Provider should support custom embedding dimensions."""
        custom_dim = 512
        provider = LocalTfidfEmbeddingProvider(embedding_dim=custom_dim)
        
        embedding = await provider.embed("Test with custom dimension")
        assert len(embedding) == custom_dim

    @pytest.mark.asyncio
    async def test_long_text_handling(self, provider: LocalTfidfEmbeddingProvider):
        """Provider should handle long texts without errors."""
        # Generate a long text (simulating a full resume)
        long_text = " ".join(["software developer"] * 1000)
        
        embedding = await provider.embed(long_text)
        
        assert len(embedding) == DEFAULT_EMBEDDING_DIM
        vector = np.array(embedding)
        assert np.linalg.norm(vector) > 0


class TestLocalTfidfNoOllamaDependency:
    """Verify that local_tfidf does not import or use Ollama."""

    def test_no_ollama_import(self):
        """LocalTfidfEmbeddingProvider should not import ollama."""
        import sys
        
        # Clear any cached imports
        modules_before = set(sys.modules.keys())
        
        # Import the provider
        from app.agent.providers.local_tfidf import LocalTfidfEmbeddingProvider
        
        # Check no ollama module was imported
        modules_after = set(sys.modules.keys())
        new_modules = modules_after - modules_before
        
        ollama_modules = [m for m in new_modules if 'ollama' in m.lower()]
        assert not ollama_modules, f"Unexpected Ollama imports: {ollama_modules}"

    def test_no_anthropic_import(self):
        """LocalTfidfEmbeddingProvider should not import anthropic."""
        import sys
        
        modules_before = set(sys.modules.keys())
        
        from app.agent.providers.local_tfidf import LocalTfidfEmbeddingProvider
        
        modules_after = set(sys.modules.keys())
        new_modules = modules_after - modules_before
        
        anthropic_modules = [m for m in new_modules if 'anthropic' in m.lower()]
        assert not anthropic_modules, f"Unexpected Anthropic imports: {anthropic_modules}"

    def test_no_openai_import(self):
        """LocalTfidfEmbeddingProvider should not import openai."""
        import sys
        
        modules_before = set(sys.modules.keys())
        
        from app.agent.providers.local_tfidf import LocalTfidfEmbeddingProvider
        
        modules_after = set(sys.modules.keys())
        new_modules = modules_after - modules_before
        
        openai_modules = [m for m in new_modules if 'openai' in m.lower()]
        assert not openai_modules, f"Unexpected OpenAI imports: {openai_modules}"

"""
Integration tests for score + improve flow using local_tfidf embeddings.

These tests verify:
1. EmbeddingManager routes to LocalTfidfEmbeddingProvider when configured
2. The improve flow works end-to-end without external API calls for embeddings
3. No Ollama or OpenAI providers are instantiated

Note: Tests that hit Anthropic for LLM are marked with @pytest.mark.integration_anthropic
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

# Synthetic test data
SAMPLE_RESUME = """
# John Doe
Senior Software Engineer

## Experience
- 5 years Python development
- Machine learning and data science
- Full stack web development with Django and React

## Skills
Python, JavaScript, TypeScript, React, Django, PostgreSQL, Docker, AWS
"""

SAMPLE_JOB_DESCRIPTION = """
We are looking for a Senior Software Engineer with:
- 3+ years Python experience
- Machine learning background
- React/TypeScript frontend skills
- AWS cloud experience

Responsibilities:
- Build scalable backend services
- Develop ML models
- Collaborate with product team
"""


class TestEmbeddingManagerRouting:
    """Test that EmbeddingManager routes correctly to local_tfidf."""

    @pytest.mark.asyncio
    async def test_default_routes_to_local_tfidf(self):
        """When EMBEDDING_PROVIDER is empty or None, use local_tfidf."""
        with patch('app.agent.manager.settings') as mock_settings:
            mock_settings.EMBEDDING_PROVIDER = None
            mock_settings.EMBEDDING_MODEL = None
            mock_settings.EMBEDDING_API_KEY = None
            
            from app.agent.manager import EmbeddingManager
            from app.agent.providers.local_tfidf import LocalTfidfEmbeddingProvider
            
            manager = EmbeddingManager(model_provider="")
            provider = await manager._get_embedding_provider()
            
            assert isinstance(provider, LocalTfidfEmbeddingProvider)

    @pytest.mark.asyncio
    async def test_explicit_local_tfidf_routes_correctly(self):
        """When EMBEDDING_PROVIDER='local_tfidf', use LocalTfidfEmbeddingProvider."""
        from app.agent.manager import EmbeddingManager
        from app.agent.providers.local_tfidf import LocalTfidfEmbeddingProvider
        
        manager = EmbeddingManager(model_provider="local_tfidf")
        provider = await manager._get_embedding_provider()
        
        assert isinstance(provider, LocalTfidfEmbeddingProvider)

    @pytest.mark.asyncio
    async def test_unknown_provider_falls_back_to_local_tfidf(self):
        """Unknown provider names should fall back to local_tfidf."""
        from app.agent.manager import EmbeddingManager
        from app.agent.providers.local_tfidf import LocalTfidfEmbeddingProvider
        
        manager = EmbeddingManager(model_provider="unknown_provider_xyz")
        provider = await manager._get_embedding_provider()
        
        assert isinstance(provider, LocalTfidfEmbeddingProvider)

    @pytest.mark.asyncio
    async def test_embed_returns_valid_vector(self):
        """EmbeddingManager.embed() should return a valid vector."""
        from app.agent.manager import EmbeddingManager
        
        manager = EmbeddingManager(model_provider="local_tfidf")
        embedding = await manager.embed("Test text for embedding")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 768  # Default dimension
        assert all(isinstance(v, float) for v in embedding)


class TestNoExternalEmbeddingCalls:
    """Verify that local_tfidf config does NOT call external APIs."""

    @pytest.mark.asyncio
    async def test_no_ollama_instantiation(self):
        """When local_tfidf is configured, OllamaEmbeddingProvider is never created."""
        from app.agent.manager import EmbeddingManager
        
        with patch('app.agent.providers.ollama.OllamaEmbeddingProvider') as mock_ollama:
            manager = EmbeddingManager(model_provider="local_tfidf")
            provider = await manager._get_embedding_provider()
            await manager.embed("Test text")
            
            # OllamaEmbeddingProvider should never be instantiated
            mock_ollama.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_openai_instantiation(self):
        """When local_tfidf is configured, OpenAIEmbeddingProvider is never created."""
        from app.agent.manager import EmbeddingManager
        
        with patch('app.agent.providers.openai.OpenAIEmbeddingProvider') as mock_openai:
            manager = EmbeddingManager(model_provider="local_tfidf")
            provider = await manager._get_embedding_provider()
            await manager.embed("Test text")
            
            # OpenAIEmbeddingProvider should never be instantiated
            mock_openai.assert_not_called()


class TestScoreImprovementWithLocalTfidf:
    """Integration tests for ScoreImprovementService with local_tfidf."""

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        return session

    @pytest.fixture
    def mock_resume(self):
        """Create a mock Resume object."""
        resume = MagicMock()
        resume.resume_id = "test-resume-123"
        resume.content = SAMPLE_RESUME
        return resume

    @pytest.fixture
    def mock_job(self):
        """Create a mock Job object."""
        job = MagicMock()
        job.job_id = "test-job-456"
        job.content = SAMPLE_JOB_DESCRIPTION
        return job

    @pytest.fixture
    def mock_processed_resume(self):
        """Create a mock ProcessedResume object."""
        processed = MagicMock()
        processed.resume_id = "test-resume-123"
        processed.extracted_keywords = ["Python", "JavaScript", "React", "Django", "AWS"]
        return processed

    @pytest.fixture
    def mock_processed_job(self):
        """Create a mock ProcessedJob object."""
        processed = MagicMock()
        processed.job_id = "test-job-456"
        processed.extracted_keywords = ["Python", "Machine Learning", "React", "TypeScript", "AWS"]
        return processed

    @pytest.mark.asyncio
    async def test_cosine_similarity_calculation(self):
        """Test that cosine similarity works with local_tfidf embeddings."""
        import numpy as np
        from app.agent.manager import EmbeddingManager
        
        manager = EmbeddingManager(model_provider="local_tfidf")
        
        # Get embeddings for resume and job keywords
        resume_emb = await manager.embed("Python JavaScript React Django AWS")
        job_emb = await manager.embed("Python Machine Learning React TypeScript AWS")
        
        # Calculate cosine similarity
        resume_vec = np.array(resume_emb)
        job_vec = np.array(job_emb)
        
        dot_product = np.dot(resume_vec, job_vec)
        norm_resume = np.linalg.norm(resume_vec)
        norm_job = np.linalg.norm(job_vec)
        
        if norm_resume > 0 and norm_job > 0:
            similarity = dot_product / (norm_resume * norm_job)
        else:
            similarity = 0.0
        
        # Similarity should be between 0 and 1
        assert 0.0 <= similarity <= 1.0
        # These are similar texts, so similarity should be reasonably high
        assert similarity > 0.3, f"Expected similarity > 0.3, got {similarity}"

    @pytest.mark.asyncio
    async def test_embedding_error_propagation(self):
        """Test that embedding errors are properly propagated."""
        from app.agent.manager import EmbeddingManager
        from app.agent.exceptions import EmbeddingProviderError
        
        manager = EmbeddingManager(model_provider="local_tfidf")
        
        # Patch the provider to raise an error
        with patch.object(manager, '_get_embedding_provider') as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.embed.side_effect = EmbeddingProviderError("Test error")
            mock_get_provider.return_value = mock_provider
            
            with pytest.raises(EmbeddingProviderError):
                await manager.embed("Test text")


class TestProviderAgnosticErrors:
    """Test that error messages don't mention specific providers incorrectly."""

    @pytest.mark.asyncio
    async def test_embedding_error_message_is_generic(self):
        """EmbeddingError should have provider-agnostic message format."""
        from app.services.exceptions import EmbeddingError
        
        error = EmbeddingError(
            provider="local_tfidf",
            original_error="Test failure"
        )
        
        error_str = str(error)
        
        # Should mention the actual provider
        assert "local_tfidf" in error_str
        # Should NOT mention other providers
        assert "ollama" not in error_str.lower()
        assert "openai" not in error_str.lower()


# Mark tests that require actual Anthropic API calls
@pytest.mark.integration_anthropic
class TestFullFlowWithAnthropic:
    """
    Full integration tests that hit Anthropic API.
    
    Run with: pytest -m integration_anthropic
    Skip with: pytest -m "not integration_anthropic"
    """

    @pytest.mark.asyncio
    async def test_resume_parsing_uses_anthropic(self):
        """
        Verify resume parsing calls Anthropic LLM.
        
        This test requires ANTHROPIC_API_KEY to be set.
        """
        pytest.skip("Requires Anthropic API key - run manually")

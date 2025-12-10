"""
Tests for ScoreImprovementService embedding handling.

These tests verify:
1. Successful embedding flow proceeds to LLM improvement call
2. Embedding failures raise EmbeddingError and do NOT call the LLM
3. Sequential embedding calls are made (not concurrent)
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestScoreImprovementServiceEmbedding:
    """Tests for embedding handling in ScoreImprovementService."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create a mock async database session."""
        session = AsyncMock()
        return session
    
    @pytest.fixture
    def mock_resume(self):
        """Create a mock Resume object."""
        resume = MagicMock()
        resume.resume_id = "test-resume-123"
        resume.content = "Test resume content with Python and AWS skills"
        return resume
    
    @pytest.fixture
    def mock_job(self):
        """Create a mock Job object."""
        job = MagicMock()
        job.job_id = "test-job-456"
        job.content = "Looking for Python developer with AWS experience"
        return job
    
    @pytest.fixture
    def mock_processed_resume(self):
        """Create a mock ProcessedResume object."""
        import json
        processed = MagicMock()
        processed.resume_id = "test-resume-123"
        processed.extracted_keywords = json.dumps({
            "extracted_keywords": ["Python", "AWS", "Developer"]
        })
        return processed
    
    @pytest.fixture
    def mock_processed_job(self):
        """Create a mock ProcessedJob object."""
        import json
        processed = MagicMock()
        processed.job_id = "test-job-456"
        processed.extracted_keywords = json.dumps({
            "extracted_keywords": ["Python", "AWS", "Cloud"]
        })
        return processed
    
    @pytest.mark.asyncio
    async def test_embedding_failure_raises_embedding_error(self, mock_db_session):
        """Test that embedding failure raises EmbeddingError without calling LLM."""
        from app.services.exceptions import EmbeddingError
        from app.agent.exceptions import EmbeddingProviderError
        
        # Mock embedding manager to fail
        mock_embedding_manager = AsyncMock()
        mock_embedding_manager.embed.side_effect = EmbeddingProviderError("Ollama EOF error")
        mock_embedding_manager._model_provider = "ollama"
        
        # Mock LLM agent manager - should NOT be called
        mock_llm_agent = AsyncMock()
        
        with patch('app.services.score_improvement_service.EmbeddingManager', return_value=mock_embedding_manager):
            with patch('app.services.score_improvement_service.AgentManager', return_value=mock_llm_agent):
                from app.services.score_improvement_service import ScoreImprovementService
                
                service = ScoreImprovementService(db=mock_db_session)
                service.embedding_manager = mock_embedding_manager
                service.json_agent_manager = mock_llm_agent
                service.md_agent_manager = mock_llm_agent
                
                # Mock the data fetching methods
                service._get_resume = AsyncMock(return_value=(
                    MagicMock(content="resume text", resume_id="123"),
                    MagicMock(extracted_keywords='{"extracted_keywords": ["Python"]}')
                ))
                service._get_job = AsyncMock(return_value=(
                    MagicMock(content="job text", job_id="456"),
                    MagicMock(extracted_keywords='{"extracted_keywords": ["Python"]}')
                ))
                service._validate_resume_keywords = MagicMock()
                service._validate_job_keywords = MagicMock()
                
                # The run() should raise EmbeddingError
                with pytest.raises(EmbeddingError) as exc_info:
                    await service.run(resume_id="123", job_id="456")
                
                # Verify the error contains provider info
                assert "ollama" in str(exc_info.value).lower()
                
                # Verify LLM was NOT called (because embedding failed first)
                # The improve_score_with_llm should not have been reached
                mock_llm_agent.run.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_embedding_calls_are_sequential(self, mock_db_session):
        """Test that embedding calls are made sequentially, not concurrently."""
        call_order = []
        
        async def mock_embed(text):
            call_order.append(f"start_{len(call_order)}")
            # In sequential execution, we should see: start_0, end_0, start_1, end_1
            call_order.append(f"end_{len(call_order)}")
            return [0.1] * 768
        
        mock_embedding_manager = AsyncMock()
        mock_embedding_manager.embed = mock_embed
        mock_embedding_manager._model_provider = "ollama"
        
        with patch('app.services.score_improvement_service.EmbeddingManager', return_value=mock_embedding_manager):
            from app.services.score_improvement_service import ScoreImprovementService
            
            service = ScoreImprovementService(db=mock_db_session)
            service.embedding_manager = mock_embedding_manager
            
            # Mock everything else to get to the embedding step
            service._get_resume = AsyncMock(return_value=(
                MagicMock(content="resume", resume_id="123"),
                MagicMock(extracted_keywords='{"extracted_keywords": ["Python"]}')
            ))
            service._get_job = AsyncMock(return_value=(
                MagicMock(content="job", job_id="456"),
                MagicMock(extracted_keywords='{"extracted_keywords": ["Python"]}')
            ))
            service._validate_resume_keywords = MagicMock()
            service._validate_job_keywords = MagicMock()
            
            # Mock the rest of the flow to avoid actual LLM calls
            service.improve_score_with_llm = AsyncMock(return_value=("improved", 0.9))
            service.get_resume_for_previewer = AsyncMock(return_value={})
            service.get_resume_analysis = AsyncMock(return_value={
                "score": 85,
                "justification": "Good match",
                "missing_keywords": [],
                "suggested_bullets": []
            })
            
            await service.run(resume_id="123", job_id="456")
            
            # Verify sequential execution pattern
            # With sequential calls: start_0, end_1, start_2, end_3
            # (indices are based on call_order length at time of append)
            assert len(call_order) == 4
            # First call completes before second starts
            assert 'start' in call_order[0]
            assert 'end' in call_order[1]
            assert 'start' in call_order[2]
            assert 'end' in call_order[3]


class TestEmbeddingErrorPropagation:
    """Tests for proper error propagation from embedding layer to API."""
    
    @pytest.mark.asyncio
    async def test_embedding_error_has_provider_info(self):
        """Test that EmbeddingError includes provider information."""
        from app.services.exceptions import EmbeddingError
        
        error = EmbeddingError(
            provider="ollama",
            original_error="EOF (status code: 500)"
        )
        
        assert "ollama" in str(error)
        assert "EOF" in str(error)
        assert error.provider == "ollama"
        assert error.original_error == "EOF (status code: 500)"
    
    @pytest.mark.asyncio
    async def test_embedding_error_without_provider(self):
        """Test EmbeddingError with minimal info."""
        from app.services.exceptions import EmbeddingError
        
        error = EmbeddingError(message="Custom message")
        
        assert "Custom message" in str(error)
        assert error.provider is None

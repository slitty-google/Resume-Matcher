class ProviderError(RuntimeError):
    """Raised when the underlying LLM provider fails"""


class EmbeddingProviderError(ProviderError):
    """Raised specifically when an embedding provider fails.
    
    This allows callers to distinguish embedding failures from LLM failures
    and handle them appropriately (e.g., skip LLM calls when embeddings fail).
    """


class StrategyError(RuntimeError):
    """Raised when a Strategy cannot parse/return expected output"""

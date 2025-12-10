#!/usr/bin/env python3
"""
Environment and Provider Diagnostics Script

This script prints the runtime configuration for LLM and embedding providers.
Use this to verify that:
1. LLM provider is Anthropic (via LlamaIndex)
2. Embedding provider is local_tfidf (no external API)
3. All environment variables are correctly set

Usage:
    cd apps/backend
    uv run python scripts/inspect_env.py
"""

import os
import sys

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 60)
    print("Resume-Matcher Provider Diagnostics")
    print("=" * 60)
    
    # Load settings
    from app.core.config import settings
    
    print("\n📋 ENVIRONMENT VARIABLES (from .env)")
    print("-" * 40)
    
    # LLM Configuration
    print("\n🤖 LLM Configuration:")
    print(f"  LLM_PROVIDER:     {settings.LLM_PROVIDER}")
    print(f"  LL_MODEL:         {settings.LL_MODEL}")
    print(f"  LLM_MAX_TOKENS:   {settings.LLM_MAX_TOKENS}")
    print(f"  LLM_TEMPERATURE:  {settings.LLM_TEMPERATURE}")
    print(f"  LLM_BASE_URL:     {settings.LLM_BASE_URL or '(not set)'}")
    print(f"  LLM_API_KEY:      {'✅ Set' if settings.LLM_API_KEY else '❌ NOT SET'}")
    
    # Embedding Configuration
    print("\n📊 Embedding Configuration:")
    print(f"  EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")
    print(f"  EMBEDDING_MODEL:    {settings.EMBEDDING_MODEL or '(not set)'}")
    print(f"  EMBEDDING_API_KEY:  {'Set' if settings.EMBEDDING_API_KEY else '(not set)'}")
    
    # Validate LLM Provider
    print("\n✅ VALIDATION CHECKS")
    print("-" * 40)
    
    errors = []
    warnings = []
    
    # Check LLM provider
    if settings.LLM_PROVIDER == "llama_index.llms.anthropic.Anthropic":
        print("✅ LLM Provider: Anthropic (correct)")
    elif settings.LLM_PROVIDER and "anthropic" in settings.LLM_PROVIDER.lower():
        print("⚠️  LLM Provider: Anthropic variant detected")
        warnings.append(f"LLM_PROVIDER should be 'llama_index.llms.anthropic.Anthropic', got: {settings.LLM_PROVIDER}")
    else:
        print(f"❌ LLM Provider: Expected Anthropic, got: {settings.LLM_PROVIDER}")
        errors.append("LLM_PROVIDER must be 'llama_index.llms.anthropic.Anthropic'")
    
    # Check embedding provider
    if settings.EMBEDDING_PROVIDER == "local_tfidf":
        print("✅ Embedding Provider: local_tfidf (correct)")
    elif settings.EMBEDDING_PROVIDER in ("ollama", "openai"):
        print(f"❌ Embedding Provider: {settings.EMBEDDING_PROVIDER} (EXTERNAL API - NOT ALLOWED)")
        errors.append(f"EMBEDDING_PROVIDER must be 'local_tfidf', got: {settings.EMBEDDING_PROVIDER}")
    else:
        print(f"⚠️  Embedding Provider: {settings.EMBEDDING_PROVIDER} (will fallback to local_tfidf)")
        warnings.append(f"Unknown EMBEDDING_PROVIDER '{settings.EMBEDDING_PROVIDER}' - will use local_tfidf fallback")
    
    # Check LLM_MAX_TOKENS
    if settings.LLM_MAX_TOKENS == 4000:
        print("✅ LLM_MAX_TOKENS: 4000 (correct)")
    elif settings.LLM_MAX_TOKENS < 2000:
        print(f"⚠️  LLM_MAX_TOKENS: {settings.LLM_MAX_TOKENS} (may cause truncation)")
        warnings.append(f"LLM_MAX_TOKENS={settings.LLM_MAX_TOKENS} is low, recommend 4000")
    else:
        print(f"ℹ️  LLM_MAX_TOKENS: {settings.LLM_MAX_TOKENS}")
    
    # Test actual provider instantiation
    print("\n🔧 RUNTIME INSTANTIATION TEST")
    print("-" * 40)
    
    try:
        from app.agent.manager import EmbeddingManager
        from app.agent.providers.local_tfidf import LocalTfidfEmbeddingProvider
        
        mgr = EmbeddingManager()
        print(f"  EmbeddingManager._model_provider: {mgr._model_provider}")
        
        import asyncio
        provider = asyncio.run(mgr._get_embedding_provider())
        
        if isinstance(provider, LocalTfidfEmbeddingProvider):
            print("✅ EmbeddingManager instantiates LocalTfidfEmbeddingProvider")
        else:
            print(f"❌ EmbeddingManager instantiated: {type(provider).__name__}")
            errors.append(f"Expected LocalTfidfEmbeddingProvider, got {type(provider).__name__}")
        
        # Test embedding generation
        embedding = asyncio.run(mgr.embed("test"))
        print(f"✅ Embedding generated: {len(embedding)} dimensions")
        
    except Exception as e:
        print(f"❌ Failed to instantiate EmbeddingManager: {e}")
        errors.append(f"EmbeddingManager instantiation failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if errors:
        print("❌ ERRORS FOUND:")
        for err in errors:
            print(f"   • {err}")
        print("\nFix these issues before running the application.")
        sys.exit(1)
    elif warnings:
        print("⚠️  WARNINGS (non-critical):")
        for warn in warnings:
            print(f"   • {warn}")
        print("\n✅ System should work, but consider addressing warnings.")
        sys.exit(0)
    else:
        print("✅ ALL CHECKS PASSED")
        print("\nSystem is correctly configured:")
        print("  • LLM: Anthropic Claude (via LlamaIndex)")
        print("  • Embeddings: Local TF-IDF (no external API)")
        sys.exit(0)


if __name__ == "__main__":
    main()

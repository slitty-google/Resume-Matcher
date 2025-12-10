# Backend Development Notes

## Running Locally Without External Embedding APIs

The backend supports multiple embedding providers for CV-to-job matching. By default, we use **Local TF-IDF embeddings** which require no external services.

### Quick Start

1. Ensure backend `.env` has:
   ```env
   EMBEDDING_PROVIDER="local_tfidf"
   EMBEDDING_MODEL=""
   ```

2. Start the dev stack from repo root:
   ```bash
   cd ~/Resume-Matcher
   source "$HOME/.local/bin/env" 2>/dev/null || true
   npm run dev
   ```

3. In the browser:
   - Visit http://localhost:3000/resume
   - Upload your resume PDF
   - Paste a job description
   - Click "Improve"

4. Expected behavior:
   - No calls to Ollama or external embedding APIs
   - No 500 errors from `/api/v1/resumes/improve`
   - Matching and improvements run using local TF-IDF embeddings plus Anthropic for LLM reasoning

---

## Embedding Provider Options

| Provider | Cost | Quality | Setup |
|----------|------|---------|-------|
| `local_tfidf` | Free | Good for keyword matching | No setup needed |
| `ollama` | Free | Better semantic matching | Requires `ollama pull nomic-embed-text` |
| `openai` | Paid | Best semantic matching | Requires OpenAI API key |

### Using Local TF-IDF (Default)

```env
EMBEDDING_PROVIDER="local_tfidf"
EMBEDDING_MODEL=""
EMBEDDING_API_KEY=""
```

- ✅ No external dependencies
- ✅ Fast execution
- ✅ No API costs
- ⚠️ Less semantically rich than neural embeddings

### Using Ollama (Optional)

```env
EMBEDDING_PROVIDER="ollama"
EMBEDDING_MODEL="nomic-embed-text"
EMBEDDING_API_KEY=""
```

Requires:
```bash
brew install ollama
ollama pull nomic-embed-text
ollama serve  # Keep running in background
```

⚠️ **Known Issue**: Ollama may return EOF errors under concurrent load. If you encounter `status code: 500` errors with Ollama, switch to `local_tfidf`.

### Using OpenAI (Optional, Paid)

```env
EMBEDDING_PROVIDER="openai"
EMBEDDING_MODEL="text-embedding-3-small"
EMBEDDING_API_KEY="sk-your-openai-key"
```

---

## LLM Provider Configuration

The backend uses **Anthropic Claude** for CV/job parsing and improvement generation:

```env
LLM_PROVIDER="llama_index.llms.anthropic.Anthropic"
LLM_API_KEY="sk-ant-your-anthropic-key"
LLM_BASE_URL="https://api.anthropic.com"
LL_MODEL="claude-sonnet-4-5"
LLM_MAX_TOKENS=4000
```

**Note:** `LLM_MAX_TOKENS=4000` is a good balance between safety and cost. Increase if you see truncated responses.

---

## Architecture Summary

| Component | Provider | Notes |
|-----------|----------|-------|
| CV Parsing | Anthropic (Claude) | Structured JSON extraction |
| Job Parsing | Anthropic (Claude) | Structured JSON extraction |
| Improvement Generation | Anthropic (Claude) | LLM reasoning |
| Embeddings | Local TF-IDF | No external API, no cost |

**Key principle:** Anthropic is used ONLY for LLM calls. Embeddings are ALWAYS local.

---

## Error Handling

The `/api/v1/resumes/improve` endpoint has fail-fast behavior:

1. **Embedding errors** → Returns HTTP 503 immediately, no LLM calls made
2. **Parsing errors** → Returns HTTP 422/500 with specific error message
3. **Generic errors** → Returns HTTP 500 with "sorry, something went wrong"

This ensures that if embeddings are misconfigured, no Anthropic API credits are wasted.

---

## Troubleshooting

### "Embedding provider failed" errors

1. Check `EMBEDDING_PROVIDER` in `.env`
2. If using `ollama`, ensure `ollama serve` is running
3. Switch to `local_tfidf` for maximum reliability

### Truncated JSON responses

Increase `LLM_MAX_TOKENS` in `.env` (default: 8000)

### API key errors

Ensure `LLM_API_KEY` is set with a valid Anthropic key starting with `sk-ant-`

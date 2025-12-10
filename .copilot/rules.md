⸻
<!-- copilot:ignore false -->
# Workspace Rules for Copilot
(
🧠 Goal

Ensure Resume-Matcher runs entirely on Anthropic Claude (LLM) and local TF-IDF embeddings (no external embedding API).
Copilot should audit, fix, and future-proof all LLM and embedding provider code so no regressions occur.

⸻

📌 Copilot Instructions

You are performing a consistency audit and refactor of this codebase. Apply the following rules globally:

1. LLM Provider = Anthropic Only
	•	Enforce the provider string:
LLM_PROVIDER="llama_index.llms.anthropic.Anthropic"
	•	Only pass Anthropic-supported kwargs:
	•	model
	•	api_key
	•	base_url
	•	temperature
	•	max_tokens
	•	Remove / refactor out any unsupported kwargs:
	•	model_name
	•	is_chat_model
	•	is_function_calling_model
	•	unknown provider fallback paths that reintroduce them
	•	Ensure dynamic import (_get_real_provider) remains intact.
	•	Confirm the constructor signature matches LlamaIndex Anthropic wrapper.

1. Embedding Provider = Local TF-IDF Only

Never call external embedding APIs.
Copilot must enforce the following invariants:
	•	Default embedding provider = "local_tfidf"
	•	Fallback for any invalid/empty provider = "local_tfidf"
	•	Remove legacy providers:
	•	Ollama
	•	OpenAI
	•	Voyage
	•	LlamaIndex Embeddings
	•	Ensure no import paths reference:
	•	ollama
	•	openai
	•	anthropic embedding classes
	•	llama_index.embeddings.*

3. Environment Variable Enforcement

Copilot should confirm:

EMBEDDING_PROVIDER="local_tfidf"
EMBEDDING_MODEL=""
LLM_MAX_TOKENS=4000
LLM_PROVIDER="llama_index.llms.anthropic.Anthropic"
LL_MODEL="claude-sonnet-4-5"

Any place the code uses fallback values, ensure they match.

4. Tests Must Pass

Copilot should:
	•	Ensure all test_local_tfidf.py and test_score_improve_local_tfidf.py pass without modification.
	•	Ensure no reintroduction of external network calls.
	•	Keep deterministic embeddings and 768-dimensional hashing.

5. Add Guards Against Regression

Copilot should add defensive checks:
	•	If Anthropic model constructor is passed unknown kwargs → raise a readable error.
	•	If a provider attempts to call Ollama / OpenAI → fail-fast with an explicit message:
"External embedding API is disabled; use local_tfidf"

6. Manual Verification Actions for Copilot

Copilot should also:
	•	Add a diagnostics helper script (scripts/inspect_env.py) that prints which provider is actually being constructed at runtime.
	•	Add a comment header to llama_index.py explaining the Anthropic kwargs.

⸻

🎯 Deliverables Copilot Must Produce
	1.	Diffs for all updated files.
	2.	Explanations of why each change was required.
	3.	Confirmation script to verify runtime settings (Python snippet).
	4.	A regression-proof set of comments in each modified file.

⸻

End of Copilot Prompt

⸻
)
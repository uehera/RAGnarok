# Native Guardrails, Prompt Injection Defense, and Data Leakage Prevention

## Threat Modeling for Retrieval-Augmented Generation

Deploying generative AI into production environments exposes applications to unique adversarial vulnerabilities and data leakage risks. Retrieval-Augmented Generation (RAG) introduces a dual attack surface:
1. **Direct Prompt Injection:** Adversarial user inputs designed to override system personas, bypass safety filters, extract internal system prompts, or manipulate model output.
2. **Indirect Document Injection (Data Poisoning):** Malicious payloads hidden inside ingested documents (e.g., resumes, PDFs, web scrapings) that hijack the model once retrieved into the context window.
3. **Sensitive Credential & PII Leakage:** Models hallucinating or inadvertently reproducing API keys, database connection strings, passwords, customer emails, or private intellectual property.
4. **Ungrounded Hallucinations:** Confidently asserting non-factual claims unsupported by the retrieved reference texts.

---

## Two-Tier Input Guardrail Architecture

To balance maximum security with minimum latency, RAGnarok utilizes a tiered input checkpoint system:

```
User Query ──► [Tier 1: Deterministic Fast-Path (<1ms)] ──► [Tier 2: Semantic Intent Guard (~150ms)] ──► Search
                     │                                            │
                     ▼ (Violation)                                ▼ (Unsafe / Off-Topic)
            Immediate Refusal                            Standardized Refusal Notice
```

### Tier 1: Deterministic Fast-Path Filter (< 1ms)
Cheap, procedural checks executed entirely in pure Python before invoking database search or vector models:
- **Length Boundary Enforcement:** Hard token ceiling (default: 1,024 tokens approximated at 4 characters/token). Prevents buffer bloat and denial-of-service token flooding.
- **Control Character Sanitization:** Strips non-printable ASCII control bytes (`\x00-\x08`, `\x0B-\x0C`, `\x0E-\x1F`, `\x7F`) and ANSI terminal escape codes (`\x1b[...]`), safeguarding terminal renderers and downstream parsers.
- **Compiled Regex Heuristics:** High-speed pattern matching detecting overt injection keywords:
  - `"ignore all previous instructions"`
  - `"you are now in developer/DAN mode"`
  - `"system prompt override"`
  - Chat template injection delimiters (`<|im_start|>`, `<|im_end|>`, `[INST]`, `[/INST]`).

### Tier 2: Semantic Model Guard
For queries that pass deterministic checks, an ultra-fast small language model (configured via `config.toml`, e.g., `groq/qwen-2.5-32b`) evaluates query intent against a structured Pydantic v2 schema:

```json
{
  "is_safe": true,
  "is_topical": true,
  "violation_type": null,
  "fallback_message": null
}
```

- **Domain Topicality Alignment:** When `strict_mode = true`, off-topic questions (e.g., *"Write a recipe for chocolate cake"* on a software engineering corpus) are rejected with a polite domain boundary notice, saving retrieval compute.
- **Fail-Open Resilience:** If the cloud model endpoint encounters rate limits or network socket timeouts, Tier 2 automatically fails open (`tier2_fallback`), allowing legitimate queries through while Tier 1 deterministic protection remains active.

---

## Two-Layer Output Guardrail Protection

Generated answers undergo inspection before presentation on the terminal user interface:

### Layer 1: Secret & PII Redaction Scanner
A deterministic procedural scanner inspects synthesized text for exposed credentials and replaces matches with structured redaction tags:

| Target Credential | Regex Pattern Scope | Replacement Tag |
| :--- | :--- | :--- |
| **Anthropic Keys** | `sk-ant-[a-zA-Z0-9_-]{20,}` | `[REDACTED ANTHROPIC_KEY]` |
| **OpenAI Keys** | `sk-(?!ant-)[a-zA-Z0-9_-]{20,}` | `[REDACTED OPENAI_KEY]` |
| **Groq Keys** | `gsk_[a-zA-Z0-9_-]{20,}` | `[REDACTED GROQ_KEY]` |
| **Database URIs** | `(?:postgresql\|postgres\|mysql\|mongodb\|redis\|duckdb)://...` | `[REDACTED DATABASE_URL]` |
| **Generic Secrets** | `api_key=...`, `access_token=...` | `[REDACTED API_KEY]` |
| **Passwords** | `password=...`, `passwd: ...`, `secret=...` | `[REDACTED PASSWORD]` |
| **Email Addresses** | `[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+` | `[REDACTED EMAIL]` |
| **IPv4 Addresses** | `\b(?:\d{1,3}\.){3}\d{1,3}\b` | `[REDACTED IP_ADDRESS]` |

### Layer 2: Self-RAG Grounding & Faithfulness Verification
Verifies whether factual claims in the synthesized text are directly supported by the retrieved parent context blocks. If unsupported assertions or hallucinations are detected, the system supplements or replaces the response with a standardized refusal:

> *"I could not find sufficient factual support in the provided documents to answer this question."*

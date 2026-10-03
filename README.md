# DocuMind — Production-Grade RAG Platform

DocuMind is an enterprise-ready Retrieval-Augmented Generation (RAG) system engineered with:
- **Qdrant Vector Database** (Cloud & Embedded support)
- **Redis Response Caching** (Sub-millisecond frequent query retrieval)
- **Pydantic Logfire Observability** (OpenTelemetry distributed tracing & spans)
- **Multi-Layer LLM Guardrails** (Prompt injection, PII redaction, output sanitization)
- **Multi-Provider LLM Gateway** (Groq primary with OpenAI automatic fallback)

---

## ⚡ Architecture Flow

```
[ User Query ]
       │
       ▼
[ Redis / In-Memory Cache ] ──(Hit: < 2ms)──► [ Instant Return ]
       │ (Miss)
       ▼
[ Input Guardrails ] ───────(Blocked)──────► [ Safe Refusal ]
       │ (Passed / Redacted)
       ▼
[ Qdrant Vector Retrieval ] ──► [ Context Guardrails ]
                                          │
                                          ▼
                               [ LLM Gateway with Fallback ]
                               (Groq -> OpenAI -> Offline)
                                          │
                                          ▼
                               [ Output Guardrails ]
                                          │
                                          ▼
                               [ Save in Redis & Return ]
```

---

## 🚀 Key Features

### 1. Redis Caching (`src/cache.py`)
- **Sub-millisecond Latency**: Drops repeat query latency from ~800ms to < 1ms.
- **Cost Reduction**: Drastically reduces LLM API token consumption and prevents rate limits.
- **Multi-Backend**: Supports standard `REDIS_URL` (Render Redis, Redis Cloud), Upstash REST, with automatic in-memory TTL fallback.
- **Cache Invalidation**: Automatically clears cache whenever documents are uploaded or deleted.

### 2. Logfire Observability (`src/observability.py`)
- Built on OpenTelemetry by Pydantic.
- Automatically traces FastAPI routes, HTTP requests, and custom RAG pipeline spans:
  - `rag.pipeline`
  - `rag.cache.lookup`
  - `rag.guardrails.input`
  - `rag.qdrant.retrieval`
  - `rag.guardrails.context`
  - `rag.llm_gateway.generate`
  - `rag.guardrails.output`
- Set `LOGFIRE_TOKEN` to stream live traces to your Logfire web console.

### 3. Qdrant Vector Database (`src/vector_store.py`)
- Connected to remote **Qdrant Cloud** cluster with cosine similarity search.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).

### 4. LLM Guardrails (`src/guardrails.py`)
- Prompt injection & jailbreak defense (DAN, system override, hidden prompt extraction).
- Automatic PII detection and redaction (emails, phone numbers, SSNs, credit cards).
- Safety checks & output credential leak prevention.

### 5. Multi-Tier LLM Gateway (`src/llm_gateway.py`)
- **Tier 1 (Primary)**: Groq (`openai/gpt-oss-20b` or `llama-3.3-70b-versatile`).
- **Tier 2 (Secondary Fallback)**: OpenAI (`gpt-4o-mini`).
- **Tier 3 (Tertiary Fallback)**: Offline extractive context summarizer.

---

## 🛠️ Environment Configuration (`.env`)

```env
# LLM Providers
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
GROQ_MODEL=openai/gpt-oss-20b
OPENAI_MODEL=gpt-4o-mini

# Qdrant Vector Database
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=documind_docs

# Redis Cache (Optional - uses in-memory if empty)
REDIS_URL=
# Or Upstash:
UPSTASH_REDIS_REST_URL=
UPSTASH_REDIS_REST_TOKEN=
CACHE_TTL_SECONDS=3600

# Observability (Optional - uses local trace console if empty)
LOGFIRE_TOKEN=
```

---

## 🏃 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run Web Application
uvicorn temp:app --port 8080 --reload

# Run CLI
python app.py
```

---

## 🚢 Deployment Guide: Render vs Vercel

### Recommendation: **Render** (Recommended ⭐⭐⭐⭐⭐)
Render is significantly better suited for Python RAG applications because:
- **PyTorch / SentenceTransformers**: SentenceTransformers + PyTorch weigh ~250MB+. Vercel serverless has a strict 250MB zip limit and 10s execution timeouts.
- **Fast In-Memory Embeddings**: Render keeps the model loaded in RAM, giving instant vector conversions without cold starts.
- **Managed Free Redis**: Render allows 1-click free Redis provisioning linked directly via `render.yaml`.

#### How to Deploy to Render:
1. Push your repository to GitHub.
2. In [Render Dashboard](https://dashboard.render.com), click **New +** $\rightarrow$ **Blueprint**.
3. Connect your GitHub repository.
4. Render will automatically read `render.yaml` and provision:
   - **FastAPI Web Service** (`documind-rag`)
   - **Render Redis Cache** (`documind-redis`)
5. Fill in your secrets (`GROQ_API_KEY`, `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `LOGFIRE_TOKEN`).
6. Done! Your RAG platform is live with SSL.

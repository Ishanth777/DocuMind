"""
DocuMind RAG Search Engine
Integrates:
- In-Memory Response Caching
- Logfire Observability & Tracing
- Qdrant Vector Store retrieval
- LLM Guardrails (Input safety, context grounding, output sanitization)
- LLM Gateway (Groq -> OpenAI -> Offline Fallback)
"""

import time
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

from src.vector_store import vector_store_example, similarity_search
from src.guardrails import RAGGuardrails
from src.llm_gateway import LLMGateway
from src.cache import RAGCache
from src.observability import trace_span

load_dotenv()
logger = logging.getLogger("DocuMind.RAGSearch")
logging.basicConfig(level=logging.INFO)


class RAGSearch:
    def __init__(
        self,
        llm_model: str = "openai/gpt-oss-20b",
        openai_model: str = "gpt-4o-mini",
        redact_pii: bool = True,
    ):
        self.gateway = LLMGateway(groq_model=llm_model, openai_model=openai_model)
        self.guardrails = RAGGuardrails(redact_pii=redact_pii)
        self.cache = RAGCache()
        logger.info("[RAGSearch] Initialized with Qdrant, Guardrails, LLM Gateway, Logfire, and Cache.")

    def _build_structured_prompt(self, query: str, context_text: str) -> str:
        """Constructs a prompt enforcing structured, professional Markdown output."""
        return f"""You are DocuMind, an enterprise AI research assistant. Your task is to provide a comprehensive, well-structured, and accurate answer to the user's question strictly based on the provided context documents.

### Instructions:
1. **Direct Overview**: Begin with a clear 1-2 sentence direct answer or summary.
2. **Structured Breakdown**: Organize information into logical sections using Markdown headings (e.g. `### Skills & Technologies`, `### Experience & Roles`, `### Key Findings`).
3. **Bullet Points & Bold Highlights**: Use bullet lists with **bold keywords** for readability.
4. **Tables where applicable**: If the data involves dates, scores, categories, or metrics, format them into a clean Markdown table.
5. **Strict Grounding**: Base all statements strictly on the context. If specific details requested are not found in the context, explicitly mention what is unavailable.

---

### Context Documents:
{context_text}

---

### User Query:
{query}

### Structured Response:"""

    def search_and_summarize_with_meta(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Full Production RAG pipeline with Caching, Tracing, and Structured Output.
        """
        start_time = time.time()

        with trace_span("rag.pipeline", query=query, top_k=top_k) as root_span:
            # Step 0: Cache Check
            with trace_span("rag.cache.lookup", query=query):
                cached = self.cache.get(query, top_k=top_k)
                if cached:
                    total_latency = round((time.time() - start_time) * 1000, 2)
                    cached["gateway"]["latency_ms"] = total_latency
                    cached["cached"] = True
                    cached["cache_backend"] = self.cache.backend
                    logger.info(f"[RAGSearch] Served from Cache ({self.cache.backend}) in {total_latency}ms")
                    return cached

            # Step 1: Input Guardrails
            with trace_span("rag.guardrails.input", query=query):
                input_guard = self.guardrails.check_input(query)
                if not input_guard.passed:
                    logger.warning(f"[RAGSearch] Input blocked by guardrails: {input_guard.message}")
                    return {
                        "answer": f"🛡️ **[Guardrail Notice]** {input_guard.message}",
                        "guardrails": {
                            "input": {
                                "passed": False,
                                "action": input_guard.action,
                                "flags": input_guard.flags,
                                "details": input_guard.details,
                            }
                        },
                        "gateway": {"provider": "none", "fallback_triggered": False, "latency_ms": 0},
                        "sources": [],
                        "cached": False,
                    }

            effective_query = input_guard.sanitized_text or query

            # Step 2: Vector Retrieval via Qdrant
            with trace_span("rag.qdrant.retrieval", query=effective_query, top_k=top_k):
                results = vector_store_example(effective_query, top_k=top_k)

            # Step 3: Context Guardrails
            with trace_span("rag.guardrails.context", retrieved_count=len(results)):
                context_guard = self.guardrails.check_context(effective_query, results)

            if not results:
                return {
                    "answer": "ℹ️ **No relevant documents found.** Please upload documents to the knowledge base to answer this query.",
                    "guardrails": {
                        "input": {"passed": True, "action": input_guard.action, "flags": input_guard.flags},
                        "context": {"passed": False, "flags": ["no_context_found"]},
                    },
                    "gateway": {"provider": "none", "fallback_triggered": False, "latency_ms": 0},
                    "sources": [],
                    "cached": False,
                }

            # Format context for prompt
            context_text = "\n\n---\n\n".join(
                [f"[Document: {getattr(doc, 'metadata', {}).get('source', 'Unknown')} | Page: {getattr(doc, 'metadata', {}).get('page', 1)}]\n{doc.page_content}" for doc in results]
            )

            prompt = self._build_structured_prompt(effective_query, context_text)

            # Step 4: LLM Gateway invocation (Groq -> OpenAI -> Offline Fallback)
            with trace_span("rag.llm_gateway.generate", prompt_len=len(prompt)):
                gateway_resp = self.gateway.generate(prompt=prompt, raw_context=context_text)

            # Step 5: Output Guardrail check
            with trace_span("rag.guardrails.output"):
                output_guard = self.guardrails.check_output(gateway_resp.content, context=context_text)
                final_answer = output_guard.sanitized_text or gateway_resp.content

            # Package sources metadata
            sources = [
                {
                    "source": getattr(doc, "metadata", {}).get("source", "Unknown"),
                    "page": getattr(doc, "metadata", {}).get("page", 1),
                    "snippet": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                }
                for doc in results
            ]

            response_payload = {
                "answer": final_answer,
                "guardrails": {
                    "input": {
                        "passed": input_guard.passed,
                        "action": input_guard.action,
                        "flags": input_guard.flags,
                    },
                    "context": {
                        "passed": context_guard.passed,
                        "action": context_guard.action,
                        "flags": context_guard.flags,
                    },
                    "output": {
                        "passed": output_guard.passed,
                        "action": output_guard.action,
                        "flags": output_guard.flags,
                    },
                },
                "gateway": {
                    "provider": gateway_resp.provider,
                    "model": gateway_resp.model,
                    "latency_ms": gateway_resp.latency_ms,
                    "fallback_triggered": gateway_resp.fallback_triggered,
                    "fallback_chain": gateway_resp.fallback_chain,
                    "metadata": gateway_resp.metadata,
                },
                "sources": sources,
                "cached": False,
                "cache_backend": self.cache.backend,
            }

            # Step 6: Cache the structured response
            self.cache.set(query, top_k, response_payload)

            return response_payload

    def search_and_summarize(self, query: str, top_k: int = 3) -> str:
        """Standard interface for backward compatibility."""
        result = self.search_and_summarize_with_meta(query, top_k=top_k)
        return result["answer"]

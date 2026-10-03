"""
DocuMind LLM Gateway with Multi-Provider Fallback
Provides a resilient gateway with automatic failover across multiple LLM providers:
1. Groq (Primary - Fast inference with llama-3.1-8b-instant / llama-3.3-70b-versatile)
2. OpenAI (Secondary Fallback - gpt-4o-mini / gpt-4o)
3. Extractive Context Fallback (Offline Fallback - produces grounded answer directly from context if all external APIs fail or are unconfigured)
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("DocuMind.LLMGateway")


@dataclass
class GatewayResponse:
    content: str
    provider: str
    model: str
    latency_ms: float
    fallback_triggered: bool
    fallback_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMGateway:
    """
    LLM Gateway providing automatic provider failover, retry handling,
    and graceful degradation for RAG pipelines.
    """

    def __init__(
        self,
        groq_model: str = "llama-3.1-8b-instant",
        openai_model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        request_timeout: float = 15.0,
    ):
        self.groq_model = os.getenv("GROQ_MODEL", groq_model)
        self.openai_model = os.getenv("OPENAI_MODEL", openai_model)
        self.temperature = temperature
        self.request_timeout = request_timeout

        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        self._groq_client = None
        self._openai_client = None

        self._init_providers()

    def _init_providers(self):
        """Lazy load or initialize clients depending on available API keys."""
        if self.groq_api_key and self.groq_api_key.strip():
            try:
                from langchain_groq import ChatGroq
                self._groq_client = ChatGroq(
                    groq_api_key=self.groq_api_key.strip(),
                    model_name=self.groq_model,
                    temperature=self.temperature,
                    timeout=self.request_timeout,
                    max_retries=1,
                )
                logger.info(f"[LLMGateway] Primary Provider configured: Groq ({self.groq_model})")
            except Exception as e:
                logger.warning(f"[LLMGateway] Failed to initialize Groq client: {e}")

        if self.openai_api_key and self.openai_api_key.strip():
            try:
                from langchain_openai import ChatOpenAI
                self._openai_client = ChatOpenAI(
                    api_key=self.openai_api_key.strip(),
                    model=self.openai_model,
                    temperature=self.temperature,
                    timeout=self.request_timeout,
                    max_retries=1,
                )
                logger.info(f"[LLMGateway] Secondary Provider configured: OpenAI ({self.openai_model})")
            except Exception as e:
                logger.warning(f"[LLMGateway] Failed to initialize OpenAI client: {e}")

    def _call_groq(self, prompt: str) -> str:
        if not self._groq_client:
            raise ValueError("Groq provider not configured or missing GROQ_API_KEY.")
        response = self._groq_client.invoke(prompt)
        return response.content

    def _call_openai(self, prompt: str) -> str:
        if not self._openai_client:
            raise ValueError("OpenAI provider not configured or missing OPENAI_API_KEY.")
        response = self._openai_client.invoke(prompt)
        return response.content

    def _extractive_fallback(self, prompt: str, raw_context: Optional[str] = None) -> str:
        """
        Deterministic, offline fallback when all external LLM APIs are unreachable or unconfigured.
        Extracts top sentences directly from the context.
        """
        logger.warning("[LLMGateway] Invoking Extractive Fallback (Offline Mode).")
        if not raw_context or not raw_context.strip():
            return "Unable to contact LLM service and no local context was provided."

        sentences = [s.strip() for s in raw_context.replace("\n", " ").split(".") if len(s.strip()) > 20]
        summary_sentences = sentences[:5]
        if summary_sentences:
            return "[Offline Extractive Fallback]\n" + ". ".join(summary_sentences) + "."
        return "[Offline Extractive Fallback]\n" + raw_context[:500] + "..."

    def generate(self, prompt: str, raw_context: Optional[str] = None) -> GatewayResponse:
        """
        Executes generation through the provider chain:
        Groq -> OpenAI -> Offline Fallback
        """
        fallback_history: List[str] = []
        start_time = time.time()

        # 1. Attempt Primary: Groq
        if self._groq_client:
            try:
                logger.info(f"[LLMGateway] Routing request to Primary Provider: Groq ({self.groq_model})")
                t0 = time.time()
                content = self._call_groq(prompt)
                latency = (time.time() - t0) * 1000
                return GatewayResponse(
                    content=content,
                    provider="groq",
                    model=self.groq_model,
                    latency_ms=round(latency, 2),
                    fallback_triggered=False,
                    fallback_chain=["groq"],
                )
            except Exception as e:
                msg = f"Groq failure: {type(e).__name__} - {str(e)[:100]}"
                logger.error(f"[LLMGateway] {msg}. Initiating fallback...")
                fallback_history.append(msg)

        # 2. Attempt Secondary: OpenAI
        if self._openai_client:
            try:
                logger.info(f"[LLMGateway] Routing request to Fallback Provider: OpenAI ({self.openai_model})")
                t0 = time.time()
                content = self._call_openai(prompt)
                latency = (time.time() - t0) * 1000
                return GatewayResponse(
                    content=content,
                    provider="openai",
                    model=self.openai_model,
                    latency_ms=round(latency, 2),
                    fallback_triggered=True,
                    fallback_chain=["groq" if self._groq_client else "none", "openai"],
                    metadata={"fallback_reasons": fallback_history},
                )
            except Exception as e:
                msg = f"OpenAI failure: {type(e).__name__} - {str(e)[:100]}"
                logger.error(f"[LLMGateway] {msg}. Falling back to offline mode...")
                fallback_history.append(msg)

        # 3. Tertiary: Offline Context Extractive Fallback
        t0 = time.time()
        content = self._extractive_fallback(prompt, raw_context=raw_context)
        latency = (time.time() - t0) * 1000

        return GatewayResponse(
            content=content,
            provider="offline_extractive_fallback",
            model="local-rule-based",
            latency_ms=round(latency, 2),
            fallback_triggered=True,
            fallback_chain=fallback_history + ["offline_fallback"],
            metadata={
                "fallback_reasons": fallback_history or ["No remote API keys available in environment"],
                "notice": "All remote LLM endpoints unavailable or unconfigured; generated using offline fallback.",
            },
        )

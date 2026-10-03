"""
DocuMind Observability Module (powered by Pydantic Logfire)
Provides structured tracing, OpenTelemetry metrics, and span management across:
- FastAPI HTTP requests
- Guardrails evaluation
- Vector search retrieval
- LLM Gateway invocations & fallbacks
- Redis cache hits/misses
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("DocuMind.Observability")

_logfire_initialized = False

try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False


def setup_observability(service_name: str = "documind-rag", app: Optional[object] = None):
    """Initializes Logfire observability with OpenTelemetry instrumentation."""
    global _logfire_initialized
    if _logfire_initialized or not LOGFIRE_AVAILABLE:
        return

    token = os.getenv("LOGFIRE_TOKEN")
    
    if token and token.strip():
        logger.info("[Observability] Configuring Logfire with remote cloud exporter...")
        logfire.configure(
            token=token.strip(),
            service_name=service_name,
            service_version="0.2.0",
        )
    else:
        logger.info("[Observability] LOGFIRE_TOKEN not set. Running Logfire in local mode (console traces only).")
        logfire.configure(
            send_to_logfire=False,
            service_name=service_name,
        )

    # Instrument httpx if available
    try:
        logfire.instrument_httpx()
    except Exception as e:
        logger.debug(f"[Observability] httpx instrumentation skipped: {e}")

    # Instrument FastAPI app if provided
    if app:
        try:
            logfire.instrument_fastapi(app)
            logger.info("[Observability] FastAPI app instrumented with Logfire.")
        except Exception as e:
            logger.debug(f"[Observability] FastAPI instrumentation skipped: {e}")

    _logfire_initialized = True


class SpanContext:
    """Helper context manager that safely wraps logfire spans or falls back gracefully."""
    def __init__(self, span_name: str, **attributes):
        self.span_name = span_name
        self.attributes = attributes
        self._span = None

    def __enter__(self):
        if LOGFIRE_AVAILABLE and _logfire_initialized:
            try:
                self._span = logfire.span(self.span_name, **self.attributes)
                return self._span.__enter__()
            except Exception:
                pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._span:
            try:
                return self._span.__exit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass
        return False

    def set_attribute(self, key: str, value: object):
        if self._span and hasattr(self._span, "set_attribute"):
            try:
                self._span.set_attribute(key, value)
            except Exception:
                pass


def trace_span(name: str, **attributes):
    """Convenience span wrapper for RAG components."""
    return SpanContext(name, **attributes)

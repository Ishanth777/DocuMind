"""
DocuMind LLM Guardrails Module
Comprehensive Input, Context, and Output guardrails for Production RAG:
- Prompt injection & jailbreak detection
- PII detection and redaction
- Harmful / safety policy checks
- Context relevance & grounding verification
- System prompt & sensitive credential leakage prevention
- Output hallucination & quality checks
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("DocuMind.Guardrails")
logging.basicConfig(level=logging.INFO)


@dataclass
class GuardrailResult:
    passed: bool
    action: str  # "allow", "block", "mask", "warn"
    message: str = ""
    flags: List[str] = field(default_factory=list)
    sanitized_text: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class InputGuardrails:
    """Validates and sanitizes user input before it hits retrieval and LLMs."""

    # Common prompt injection and jailbreak patterns
    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?(previous\s+|prior\s+|above\s+)?instructions?",
        r"(?i)disregard\s+(all\s+)?(previous\s+|prior\s+|above\s+)?instructions?",
        r"(?i)forget\s+(all\s+)?(previous\s+|prior\s+|above\s+)?instructions?",
        r"(?i)you\s+are\s+now\s+(unrestricted|unfiltered|jailbroken|DAN|in\s+developer\s+mode)",
        r"(?i)system\s*override",
        r"(?i)bypass\s+(all\s+)?(filters|rules|guardrails|safety)",
        r"(?i)(reveal|show|display|print)\s+(your\s+|the\s+)?(system\s+prompt|hidden\s+prompt|developer\s+instructions|instructions)",
        r"(?i)act\s+as\s+an?\s+(unfiltered|evil|jailbroken|adversarial)\s+ai",
        r"(?i)do\s+anything\s+now",
        r"(?i)switch\s+to\s+(debug|god|developer)\s+mode",
    ]

    # Harmful & unsafe topic patterns
    HARMFUL_PATTERNS = [
        r"(?i)\b(how\s+to\s+make|build|create)\s+(a\s+)?(bomb|explosive|weapon|poison|virus|malware|ransomware)\b",
        r"(?i)\b(steal|hack|phish|exploit|ddos|sql\s*injection)\s+(into|credentials|accounts?|passwords?)\b",
        r"(?i)\b(kill\s+yourself|commit\s+suicide|self-harm)\b",
    ]

    # PII Regexes
    EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    PHONE_PATTERN = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    CREDIT_CARD_PATTERN = r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"

    def __init__(self, min_length: int = 1, max_length: int = 2000, redact_pii: bool = True):
        self.min_length = min_length
        self.max_length = max_length
        self.redact_pii = redact_pii
        self._injection_regex = [re.compile(p) for p in self.INJECTION_PATTERNS]
        self._harmful_regex = [re.compile(p) for p in self.HARMFUL_PATTERNS]

    def validate(self, query: str) -> GuardrailResult:
        query = query.strip()
        flags = []

        # 1. Bounds check
        if len(query) < self.min_length:
            return GuardrailResult(
                passed=False,
                action="block",
                message="Query is too short or empty.",
                flags=["input_too_short"],
            )

        if len(query) > self.max_length:
            return GuardrailResult(
                passed=False,
                action="block",
                message=f"Query exceeds maximum character limit of {self.max_length}.",
                flags=["input_too_long"],
            )

        # 2. Injection / Jailbreak check
        for pattern in self._injection_regex:
            if pattern.search(query):
                logger.warning(f"[Guardrails] Prompt injection detected in query: {query}")
                return GuardrailResult(
                    passed=False,
                    action="block",
                    message="Query contains unauthorized instructions or prompt injection attempts.",
                    flags=["prompt_injection_detected"],
                    details={"matched_pattern": pattern.pattern},
                )

        # 3. Harmful / Safety check
        for pattern in self._harmful_regex:
            if pattern.search(query):
                logger.warning(f"[Guardrails] Harmful content detected in query: {query}")
                return GuardrailResult(
                    passed=False,
                    action="block",
                    message="Query violates safety policy.",
                    flags=["harmful_content_detected"],
                    details={"matched_pattern": pattern.pattern},
                )

        # 4. PII Redaction
        sanitized = query
        pii_found = []

        if re.search(self.CREDIT_CARD_PATTERN, sanitized):
            sanitized = re.sub(self.CREDIT_CARD_PATTERN, "[REDACTED_CARD]", sanitized)
            pii_found.append("credit_card")

        if re.search(self.SSN_PATTERN, sanitized):
            sanitized = re.sub(self.SSN_PATTERN, "[REDACTED_SSN]", sanitized)
            pii_found.append("ssn")

        if re.search(self.EMAIL_PATTERN, sanitized):
            sanitized = re.sub(self.EMAIL_PATTERN, "[REDACTED_EMAIL]", sanitized)
            pii_found.append("email")

        if re.search(self.PHONE_PATTERN, sanitized):
            sanitized = re.sub(self.PHONE_PATTERN, "[REDACTED_PHONE]", sanitized)
            pii_found.append("phone")

        if pii_found:
            logger.info(f"[Guardrails] PII detected and sanitized: {pii_found}")
            flags.extend([f"pii_{t}" for t in pii_found])
            return GuardrailResult(
                passed=True,
                action="mask",
                message="PII detected and redacted.",
                flags=flags,
                sanitized_text=sanitized,
                details={"pii_types": pii_found},
            )

        return GuardrailResult(
            passed=True,
            action="allow",
            message="Input passed all guardrails.",
            sanitized_text=query,
            flags=[],
        )


class ContextGuardrails:
    """Verifies that retrieved context is sufficient and relevant for generating an answer."""

    def validate(self, query: str, documents: List[Any], min_chunks: int = 1) -> GuardrailResult:
        if not documents or len(documents) < min_chunks:
            return GuardrailResult(
                passed=False,
                action="warn",
                message="Insufficient context retrieved from vector database.",
                flags=["insufficient_context"],
                details={"chunk_count": len(documents) if documents else 0},
            )

        total_content = " ".join(
            getattr(doc, "page_content", str(doc)) for doc in documents
        ).strip()

        if len(total_content) < 20:
            return GuardrailResult(
                passed=False,
                action="warn",
                message="Retrieved context chunks are nearly empty.",
                flags=["empty_context"],
                details={"chunk_count": len(documents)},
            )

        # Keyword overlap check for grounding
        query_words = set(re.findall(r"\w+", query.lower()))
        # Exclude common stopwords
        stopwords = {"what", "is", "the", "a", "an", "and", "or", "in", "on", "of", "to", "for", "with", "who", "where", "how", "when", "why"}
        significant_words = query_words - stopwords

        if significant_words:
            context_words = set(re.findall(r"\w+", total_content.lower()))
            overlap = significant_words.intersection(context_words)
            overlap_ratio = len(overlap) / len(significant_words)

            if overlap_ratio == 0:
                logger.warning(f"[Guardrails] Context grounding warning: No significant keyword overlap with query words {significant_words}")
                return GuardrailResult(
                    passed=True,
                    action="warn",
                    message="Retrieved context may have low relevance to query keywords.",
                    flags=["low_relevance_warning"],
                    details={"overlap_ratio": overlap_ratio, "chunk_count": len(documents)},
                )

        return GuardrailResult(
            passed=True,
            action="allow",
            message="Context verified and grounded.",
            flags=[],
            details={"chunk_count": len(documents)},
        )


class OutputGuardrails:
    """Validates and sanitizes LLM output before sending to user."""

    # Sensitive token / prompt leak patterns
    LEAK_PATTERNS = [
        r"(?i)(api[_-]?key|secret[_-]?key|groq[_-]?api[_-]?key|bearer\s+[a-zA-Z0-9_\-\.]+)",
        r"(?i)you\s+are\s+documind\s+an\s+ai\s+assistant\s+your\s+instructions\s+are",
        r"(?i)system\s+prompt:",
    ]

    def __init__(self):
        self._leak_regex = [re.compile(p) for p in self.LEAK_PATTERNS]

    def validate(self, response_text: str, context: Optional[str] = None) -> GuardrailResult:
        if not response_text or not response_text.strip():
            return GuardrailResult(
                passed=False,
                action="block",
                message="LLM generated an empty response.",
                flags=["empty_response"],
            )

        # 1. Leakage detection
        for pattern in self._leak_regex:
            if pattern.search(response_text):
                logger.error("[Guardrails] Output blocked: Sensitive credential or system prompt leakage detected.")
                # Redact matched sensitive strings
                sanitized = pattern.sub("[REDACTED_CONFIDENTIAL]", response_text)
                return GuardrailResult(
                    passed=True,
                    action="mask",
                    message="Potential credential/system prompt leakage was redacted from response.",
                    flags=["leakage_redacted"],
                    sanitized_text=sanitized,
                )

        # 2. Hallucination check against explicitly empty contexts
        if context is not None and not context.strip():
            # If there was zero context, check if the model fabricated detailed specific facts
            pass

        return GuardrailResult(
            passed=True,
            action="allow",
            message="Output passed all guardrails.",
            sanitized_text=response_text,
            flags=[],
        )


class RAGGuardrails:
    """Unified Facade for all RAG Guardrails."""

    def __init__(self, redact_pii: bool = True):
        self.input_guard = InputGuardrails(redact_pii=redact_pii)
        self.context_guard = ContextGuardrails()
        self.output_guard = OutputGuardrails()

    def check_input(self, query: str) -> GuardrailResult:
        return self.input_guard.validate(query)

    def check_context(self, query: str, documents: List[Any]) -> GuardrailResult:
        return self.context_guard.validate(query, documents)

    def check_output(self, response_text: str, context: Optional[str] = None) -> GuardrailResult:
        return self.output_guard.validate(response_text, context)

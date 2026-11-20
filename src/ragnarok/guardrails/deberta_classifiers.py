"""
Specialized DeBERTa-v3 Guardrail Classifiers.

Provides localized, model-based evaluation for:
1. Prompt Injection Detection using ProtectAI/deberta-v3-base-prompt-injection-v2.
2. Grounding & Faithfulness Verification using cross-encoder/nli-deberta-v3-base.
Implements lazy-loading, structural protocols, and graceful offline fallback.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

import structlog

logger = structlog.get_logger(__name__)


class PromptInjectionClassifierProtocol(Protocol):
    """Protocol contract for semantic prompt injection classifiers."""

    def predict_injection_score(self, text: str) -> float:
        """Returns the probability of prompt injection in range [0.0, 1.0]."""
        ...


class NLIEntailmentClassifierProtocol(Protocol):
    """Protocol contract for natural language inference grounding classifiers."""

    def predict_grounding_score(self, premise: str, hypothesis: str) -> dict[str, float]:
        """
        Returns softmax probabilities for entailment, neutral, and contradiction:
        {'entailment': float, 'neutral': float, 'contradiction': float}
        """
        ...


class DebertaPromptInjectionClassifier:
    """
    Local prompt injection classifier using ProtectAI/deberta-v3-base-prompt-injection-v2.
    Evaluates direct and indirect injection vectors with calibrated probabilities.
    """

    def __init__(
        self,
        model_name: str = "ProtectAI/deberta-v3-base-prompt-injection-v2",
        model_mock: Any = None,
    ) -> None:
        self.model_name = model_name
        self._mock = model_mock
        self._tokenizer: Any = None
        self._model: Any = None
        self._loaded: bool = False

    def _load(self) -> None:
        if self._loaded or self._mock is not None:
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._loaded = True
            logger.info("Loaded local DeBERTa prompt injection classifier", model=self.model_name)
        except Exception as exc:
            logger.warning(
                "DeBERTa prompt injection model unavailable locally; failing open",
                model=self.model_name,
                error=str(exc),
            )
            self._loaded = False

    def predict_injection_score(self, text: str) -> float:
        """
        Calculates probability [0.0, 1.0] that text represents a prompt injection.
        """
        if self._mock is not None:
            if callable(self._mock):
                return float(self._mock(text))
            return float(self._mock)

        if not text or not text.strip():
            return 0.0

        self._load()
        if not self._loaded or self._model is None or self._tokenizer is None:
            return 0.0

        try:
            import torch

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits[0]
                probabilities = torch.softmax(logits, dim=-1)

            # Check model id2label mapping for INJECTION / SAFE index
            id2label = getattr(self._model.config, "id2label", {0: "SAFE", 1: "INJECTION"})
            injection_idx = 1
            for idx, label in id2label.items():
                if "INJECTION" in str(label).upper():
                    injection_idx = int(idx)
                    break

            return float(probabilities[injection_idx].item())
        except Exception as exc:
            logger.warning("DeBERTa injection inference failed; failing open", error=str(exc))
            return 0.0


class DebertaNLIGroundingClassifier:
    """
    Local grounding verification cross-encoder using cross-encoder/nli-deberta-v3-base.
    Computes mathematical contradiction probabilities between retrieved context and answer.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        model_mock: Any = None,
    ) -> None:
        self.model_name = model_name
        self._mock = model_mock
        self._tokenizer: Any = None
        self._model: Any = None
        self._loaded: bool = False

    def _load(self) -> None:
        if self._loaded or self._mock is not None:
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._loaded = True
            logger.info("Loaded local DeBERTa NLI grounding classifier", model=self.model_name)
        except Exception as exc:
            logger.warning(
                "DeBERTa NLI model unavailable locally; failing open",
                model=self.model_name,
                error=str(exc),
            )
            self._loaded = False

    def predict_grounding_score(self, premise: str, hypothesis: str) -> dict[str, float]:
        """
        Calculates NLI entailment probabilities.
        Returns:
            {'entailment': float, 'neutral': float, 'contradiction': float}
        """
        default_safe = {"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0}

        if self._mock is not None:
            if callable(self._mock):
                return dict(self._mock(premise, hypothesis))
            return dict(self._mock)

        if not premise or not hypothesis:
            return {"entailment": 0.0, "neutral": 0.0, "contradiction": 1.0}

        self._load()
        if not self._loaded or self._model is None or self._tokenizer is None:
            return default_safe

        try:
            import torch

            inputs = self._tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits[0]
                probabilities = torch.softmax(logits, dim=-1)

            # Map NLI labels (cross-encoder/nli-deberta-v3-base: 0=contradiction, 1=entailment, 2=neutral)
            id2label = getattr(self._model.config, "id2label", {0: "contradiction", 1: "entailment", 2: "neutral"})
            result: dict[str, float] = {}
            for idx, label in id2label.items():
                clean_label = str(label).lower()
                result[clean_label] = float(probabilities[int(idx)].item())

            return {
                "entailment": result.get("entailment", 0.0),
                "neutral": result.get("neutral", 0.0),
                "contradiction": result.get("contradiction", 0.0),
            }
        except Exception as exc:
            logger.warning("DeBERTa NLI inference failed; failing open", error=str(exc))
            return default_safe

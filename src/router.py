"""
router.py — Semantic Router for Intent Classification
=======================================================

Upgrade features:
  1. Intent Classification Layer with 6 supported intents:
     - factual (RAG)
     - reasoning (LoRA)
     - summarization (LoRA)
     - document_qa (RAG)
     - email_generation (LoRA)
     - conversational (LoRA)
  2. Softmax-based confidence scoring.
  3. Ambiguity detection (warns if top two intents are very close).
  4. Fallback routing (defaults to Route.RAG if confidence is low).
  5. Multi-intent detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Route and Intent Enums                                             #
# ------------------------------------------------------------------ #

class Route(str, Enum):
    RAG = "RAG"
    LORA = "LORA"


class Intent(str, Enum):
    FACTUAL = "factual"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    DOCUMENT_QA = "document_qa"
    EMAIL_GENERATION = "email_generation"
    CONVERSATIONAL = "conversational"


# Map intents to core routes
INTENT_TO_ROUTE_MAP = {
    Intent.FACTUAL: Route.RAG,
    Intent.DOCUMENT_QA: Route.RAG,
    Intent.REASONING: Route.LORA,
    Intent.SUMMARIZATION: Route.LORA,
    Intent.EMAIL_GENERATION: Route.LORA,
    Intent.CONVERSATIONAL: Route.LORA,
}


@dataclass
class IntentCluster:
    """A named group of example phrases that define an intent classification category."""
    name: Intent
    route: Route
    examples: List[str] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None


# ------------------------------------------------------------------ #
#  Intent Examples                                                    #
# ------------------------------------------------------------------ #

INTENT_EXAMPLES: Dict[Intent, List[str]] = {
    Intent.FACTUAL: [
        "What is the capital of France?",
        "When was the company founded?",
        "List all the features of product X.",
        "What is the definition of machine learning?",
        "Who is the CEO of the company?",
        "What are the pricing options?",
        "What is the boiling point of water?",
        "Tell me the date of the next solar eclipse.",
        "What is the GDP of Japan?",
        "What is the speed of light?",
    ],
    Intent.REASONING: [
        "Explain the trade-offs between microservices and monoliths.",
        "Compare and contrast supervised and unsupervised learning.",
        "Why might a distributed system choose eventual consistency?",
        "Analyze the pros and cons of using Kubernetes.",
        "Debate the merits of functional vs. object-oriented programming.",
        "Provide your reasoning on how to optimize this algorithm.",
        "Solve this coding problem step-by-step.",
        "How do I troubleshoot a memory leak in Java?",
        "Derive the formula for quadratic equations.",
        "What are the ethical implications of artificial intelligence?",
    ],
    Intent.SUMMARIZATION: [
        "Summarize this article in three sentences.",
        "Give me a brief summary of the project status report.",
        "Condense this meeting transcript into key action points.",
        "What is the main takeaway from this text?",
        "Provide an outline of this book chapter.",
        "Write a TL;DR for this documentation page.",
        "Summarize the key arguments in this paragraph.",
        "Can you make this shorter and hit the main highlights?",
    ],
    Intent.DOCUMENT_QA: [
        "What does the documentation say about API rate limits?",
        "According to the uploaded files, what is the refund policy?",
        "Find the section about health insurance benefits in the PDF.",
        "What is the target growth rate according to the annual report?",
        "What does page 4 of the manual say about troubleshooting?",
        "Is there any mention of remote work policies in the employee handbook?",
        "Check the uploaded text files for information about the holiday schedule.",
        "Find answers in my documents.",
    ],
    Intent.EMAIL_GENERATION: [
        "Write a professional email to a client about a project delay.",
        "Draft a polite follow-up email asking about job application status.",
        "Write an out-of-office response email for my vacation.",
        "Draft a proposal email to request budget approval.",
        "Write a cold email to potential sales leads.",
        "Help me reply to this customer complaint email.",
        "Send a thank-you note email to the interviewer.",
        "Create a template for a monthly newsletter email.",
    ],
    Intent.CONVERSATIONAL: [
        "Hello, how are you today?",
        "Hi there! What's up?",
        "Good morning! Can you help me?",
        "Hey! Tell me a joke.",
        "Who are you and what do you do?",
        "Let's chat about nothing in particular.",
        "How is the weather today?",
        "Thank you so much!",
        "Goodbye, see you later!",
    ],
}


class SemanticRouter:
    """Classifies user queries into 6 specific intents using cosine similarity."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        confidence_threshold: float = 0.28,
        default_route: Route = Route.RAG,
        ambiguity_threshold: float = 0.12,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.default_route = default_route
        self.ambiguity_threshold = ambiguity_threshold

        logger.info("Loading router embedding model: %s", model_name)
        self._encoder = SentenceTransformer(model_name)

        # Initialize clusters
        self._clusters: List[IntentCluster] = []
        for intent, examples in INTENT_EXAMPLES.items():
            self._clusters.append(
                IntentCluster(
                    name=intent,
                    route=INTENT_TO_ROUTE_MAP[intent],
                    examples=examples,
                )
            )

        # Pre-compute embeddings
        self._build_cluster_embeddings()

    def _build_cluster_embeddings(self) -> None:
        """Embed examples for all intent clusters."""
        for cluster in self._clusters:
            cluster.embeddings = self._encoder.encode(
                cluster.examples,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            logger.debug(
                "Embedded %d examples for intent '%s'.",
                len(cluster.examples),
                cluster.name.value,
            )

    def add_examples(self, intent: Intent, new_examples: List[str]) -> None:
        """Dynamically add examples to an intent and re-encode."""
        for cluster in self._clusters:
            if cluster.name == intent:
                cluster.examples.extend(new_examples)
                cluster.embeddings = self._encoder.encode(
                    cluster.examples,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                logger.info(
                    "Added %d examples to '%s' (total: %d).",
                    len(new_examples),
                    cluster.name.value,
                    len(cluster.examples),
                )
                return
        raise ValueError(f"No cluster found for intent: {intent}")

    def route(self, query: str) -> Tuple[Route, Dict[str, Any]]:
        """Classify query, compute confidence, detect ambiguity and return route.

        Returns
        -------
        route : Route
            Selected destination route (RAG or LORA).
        metadata : dict
            Detailed classification scores, confidence, intents, and flags.
        """
        # 1. Encode query
        query_embedding = self._encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # 2. Score each cluster
        # Using top-3 mean cosine similarity for each intent
        raw_similarities: Dict[Intent, float] = {}
        for cluster in self._clusters:
            # sims shape: (1, n_examples) -> [0] to get flat array
            sims = cosine_similarity(query_embedding, cluster.embeddings)[0]
            top_k = min(3, len(sims))
            top_sims = np.sort(sims)[-top_k:]
            mean_score = float(np.mean(top_sims))
            raw_similarities[cluster.name] = max(0.0, mean_score)  # avoid negative scores

        # 3. Softmax probabilities over raw similarities to compute relative confidence
        temp = 0.05  # softmax temperature
        intents_list = list(raw_similarities.keys())
        sim_scores = np.array([raw_similarities[i] for i in intents_list])
        
        # Softmax formula
        max_sim = np.max(sim_scores)
        exp_sims = np.exp((sim_scores - max_sim) / temp)
        probs = exp_sims / np.sum(exp_sims)
        
        confidence_scores = {intent.value: round(float(prob), 4) for intent, prob in zip(intents_list, probs)}
        raw_scores = {intent.value: round(raw_similarities[intent], 4) for intent in intents_list}

        # 4. Determine winning intent
        sorted_intents = sorted(intents_list, key=lambda x: raw_similarities[x], reverse=True)
        top_intent = sorted_intents[0]
        second_intent = sorted_intents[1]

        top_raw_score = raw_similarities[top_intent]
        top_prob = confidence_scores[top_intent.value]
        second_prob = confidence_scores[second_intent.value]

        # 5. Check confidence threshold (based on raw similarity to avoid softmax hallucination)
        if top_raw_score < self.confidence_threshold:
            logger.info(
                "Router confidence low (top raw score %.4f < %.4f). Fallback to default route: %s",
                top_raw_score,
                self.confidence_threshold,
                self.default_route.value,
            )
            return self.default_route, {
                "route": self.default_route.value,
                "intent": "fallback",
                "confidence": 1.0,
                "all_scores": confidence_scores,
                "raw_scores": raw_scores,
                "ambiguous": False,
                "multi_intents": [],
            }

        # 6. Ambiguity Detection
        prob_gap = top_prob - second_prob
        is_ambiguous = prob_gap < self.ambiguity_threshold
        if is_ambiguous:
            logger.warning(
                "Ambiguity detected! Top intent: '%s' (prob=%.4f), Second intent: '%s' (prob=%.4f), Gap=%.4f",
                top_intent.value,
                top_prob,
                second_intent.value,
                second_prob,
                prob_gap,
            )

        # 7. Multi-intent Detection
        # If second intent is close to top intent and both have significant raw matching scores
        multi_intents = []
        if is_ambiguous and raw_similarities[second_intent] > 0.35:
            multi_intents = [top_intent.value, second_intent.value]

        target_route = INTENT_TO_ROUTE_MAP[top_intent]
        logger.info(
            "Routed query to Route=%s via Intent=%s (confidence=%.4f)",
            target_route.value,
            top_intent.value,
            top_prob,
        )

        return target_route, {
            "route": target_route.value,
            "intent": top_intent.value,
            "confidence": top_prob,
            "all_scores": confidence_scores,
            "raw_scores": raw_scores,
            "ambiguous": is_ambiguous,
            "multi_intents": multi_intents,
        }

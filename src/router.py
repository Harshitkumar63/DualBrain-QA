"""
router.py — Semantic Router for Intent Classification
=======================================================

Responsibilities:
  1. Maintain two "intent clusters" (Factual Query & Reasoning/Style
     Query), each defined by a set of representative example phrases.
  2. Embed the user query and every example phrase using the same
     sentence-transformer model used by the RAG pipeline.
  3. Compute cosine similarity between the query embedding and each
     cluster's example embeddings.
  4. Route to RAG if the query is closest to the Factual cluster,
     or to the LoRA model if closest to the Reasoning/Style cluster.

Design Trade-offs:
  • **Why not an LLM-based classifier?**  A lightweight embedding +
    cosine-similarity approach adds < 5 ms of latency vs. hundreds of
    milliseconds for an LLM call.  This is critical when the router
    sits on the hot path of every request.
  • **Why example-based clusters instead of learned centroids?**
    Example-based clusters are transparent, auditable, and easy to
    extend without retraining.  You simply add more example phrases
    to improve coverage.
  • **Threshold-based fallback.**  If neither cluster scores above a
    configurable confidence threshold, we default to RAG (the safer
    option for factual accuracy).  This avoids confidently routing
    ambiguous queries to the wrong pipeline.
  • **Cosine similarity vs. L2 distance.**  Cosine similarity is
    scale-invariant and directly interpretable (1 = identical, 0 =
    orthogonal).  With normalized embeddings, cosine and dot-product
    are equivalent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Route Enum                                                         #
# ------------------------------------------------------------------ #

class Route(str, Enum):
    """The two possible destinations for an incoming query."""
    RAG = "RAG"
    LORA = "LORA"


# ------------------------------------------------------------------ #
#  Intent Cluster Definition                                          #
# ------------------------------------------------------------------ #

@dataclass
class IntentCluster:
    """A named group of example phrases that define a routing intent.

    Attributes
    ----------
    name : str
        Human-readable cluster label.
    route : Route
        Which pipeline this cluster maps to.
    examples : list[str]
        Representative phrases.  The more diverse and numerous, the
        better the router's coverage.
    embeddings : np.ndarray | None
        Computed lazily by ``SemanticRouter.build()``.
    """
    name: str
    route: Route
    examples: List[str] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None


# ------------------------------------------------------------------ #
#  Default cluster definitions                                        #
# ------------------------------------------------------------------ #

DEFAULT_FACTUAL_EXAMPLES: List[str] = [
    "What is the capital of France?",
    "When was the company founded?",
    "List all the features of product X.",
    "What does the documentation say about API rate limits?",
    "How many employees does the organization have?",
    "What is the return policy?",
    "Give me the technical specifications.",
    "What are the system requirements?",
    "Find information about the pricing plan.",
    "What is the definition of machine learning?",
    "Who is the CEO of the company?",
    "Summarize the key facts from the annual report.",
]

DEFAULT_REASONING_EXAMPLES: List[str] = [
    "Explain the trade-offs between microservices and monoliths.",
    "Write a professional email to a client about a project delay.",
    "Rewrite this paragraph in a more formal tone.",
    "Compare and contrast supervised and unsupervised learning.",
    "Draft a creative marketing tagline for our new product.",
    "Why might a distributed system choose eventual consistency?",
    "Analyze the pros and cons of using Kubernetes.",
    "Help me brainstorm ideas for a team-building event.",
    "Debate the merits of functional vs. object-oriented programming.",
    "Generate a step-by-step tutorial on setting up CI/CD.",
    "Write a poem about artificial intelligence.",
    "Provide your reasoning on how to optimize this algorithm.",
]


# ------------------------------------------------------------------ #
#  Semantic Router                                                     #
# ------------------------------------------------------------------ #

class SemanticRouter:
    """Embedding-based intent router using cosine similarity.

    Parameters
    ----------
    model_name : str
        SentenceTransformer model for encoding queries and examples.
    confidence_threshold : float
        Minimum similarity gap required to confidently pick a route.
        If the best cluster's mean similarity is below this absolute
        threshold, the router falls back to the ``default_route``.
    default_route : Route
        Fallback when confidence is too low.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        confidence_threshold: float = 0.30,
        default_route: Route = Route.RAG,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.default_route = default_route

        logger.info("Loading router embedding model: %s", model_name)
        self._encoder = SentenceTransformer(model_name)

        # Initialize default clusters
        self._clusters: List[IntentCluster] = [
            IntentCluster(
                name="Factual Query",
                route=Route.RAG,
                examples=DEFAULT_FACTUAL_EXAMPLES,
            ),
            IntentCluster(
                name="Reasoning / Style Query",
                route=Route.LORA,
                examples=DEFAULT_REASONING_EXAMPLES,
            ),
        ]

        # Pre-compute example embeddings
        self._build_cluster_embeddings()

    # -------------------------------------------------------------- #
    #  Cluster Management                                              #
    # -------------------------------------------------------------- #

    def _build_cluster_embeddings(self) -> None:
        """Embed all example phrases for every cluster."""
        for cluster in self._clusters:
            cluster.embeddings = self._encoder.encode(
                cluster.examples, convert_to_numpy=True, normalize_embeddings=True
            )
            logger.debug(
                "Embedded %d examples for cluster '%s'.",
                len(cluster.examples),
                cluster.name,
            )

    def add_examples(self, route: Route, new_examples: List[str]) -> None:
        """Dynamically add examples to an existing cluster and
        recompute its embeddings.
        """
        for cluster in self._clusters:
            if cluster.route == route:
                cluster.examples.extend(new_examples)
                cluster.embeddings = self._encoder.encode(
                    cluster.examples,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                logger.info(
                    "Added %d examples to '%s' (total: %d).",
                    len(new_examples),
                    cluster.name,
                    len(cluster.examples),
                )
                return
        raise ValueError(f"No cluster found for route: {route}")

    # -------------------------------------------------------------- #
    #  Routing Logic                                                   #
    # -------------------------------------------------------------- #

    def route(self, query: str) -> Tuple[Route, Dict[str, float]]:
        """Determine the best route for *query*.

        Returns
        -------
        route : Route
            The selected pipeline destination.
        scores : dict[str, float]
            Per-cluster similarity scores for observability.

        Algorithm:
          1. Embed the query.
          2. For each cluster, compute the cosine similarity between
             the query and *every* example, then take the **mean of
             the top-3** similarities as the cluster score.
             ─ Using top-3 mean (instead of max or global mean) makes
               the router robust to outlier examples while still
               rewarding clusters that have multiple strong matches.
          3. Pick the cluster with the highest score.
          4. If the winning score < ``confidence_threshold``, fall
             back to ``default_route``.
        """
        # 1. Encode query
        query_embedding = self._encoder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )  # shape: (1, dim)

        # 2. Score each cluster
        cluster_scores: Dict[str, float] = {}
        best_score = -1.0
        best_cluster: Optional[IntentCluster] = None

        for cluster in self._clusters:
            # cosine_similarity returns shape (1, n_examples)
            sims = cosine_similarity(query_embedding, cluster.embeddings)[0]

            # Take the mean of the top-3 similarities
            top_k = min(3, len(sims))
            top_sims = np.sort(sims)[-top_k:]
            score = float(np.mean(top_sims))

            cluster_scores[cluster.name] = round(score, 4)

            if score > best_score:
                best_score = score
                best_cluster = cluster

        # 3. Apply confidence threshold
        if best_score < self.confidence_threshold:
            logger.info(
                "Low confidence (%.4f < %.4f) — defaulting to %s.",
                best_score,
                self.confidence_threshold,
                self.default_route.value,
            )
            return self.default_route, cluster_scores

        assert best_cluster is not None
        logger.info(
            "Routed to '%s' (score=%.4f). Scores: %s",
            best_cluster.name,
            best_score,
            cluster_scores,
        )
        return best_cluster.route, cluster_scores

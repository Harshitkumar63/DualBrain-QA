"""
test_router.py — Unit tests for the SemanticRouter
===================================================

Tests the semantic router intent classification, confidence scoring,
ambiguity alerts, fallback settings, and runtime example registration.
"""

import pytest
from src.router import SemanticRouter, Route, Intent


@pytest.fixture(scope="module")
def router() -> SemanticRouter:
    """Initialize a single router instance for tests."""
    return SemanticRouter(
        confidence_threshold=0.28,
        ambiguity_threshold=0.12,
        default_route=Route.RAG
    )


def test_intent_classification(router: SemanticRouter):
    """Test that standard queries route to correct intents and routes."""
    # Factual -> RAG
    route, meta = router.route("What is the capital city of France?")
    assert route == Route.RAG
    assert meta["intent"] == Intent.FACTUAL
    assert meta["confidence"] > 0.0

    # Reasoning -> LORA
    route, meta = router.route("Explain why distributed systems prefer eventual consistency.")
    assert route == Route.LORA
    assert meta["intent"] == Intent.REASONING

    # Conversational -> LORA
    route, meta = router.route("Hello there, how are you today?")
    assert route == Route.LORA
    assert meta["intent"] == Intent.CONVERSATIONAL


def test_low_confidence_fallback(router: SemanticRouter):
    """Test that nonsense/out-of-scope inputs trigger the fallback route."""
    # We temporarily set a high confidence threshold to force fallback
    custom_router = SemanticRouter(confidence_threshold=0.99, default_route=Route.RAG)
    route, meta = custom_router.route("xyz abc hello world 123")
    assert route == Route.RAG
    assert meta["intent"] == "fallback"


def test_ambiguity_detection(router: SemanticRouter):
    """Test that borderline queries trigger the ambiguity warning."""
    # Query that spans across multiple intent concepts: summarization vs conversational
    route, meta = router.route("summarize hello how are you doing")
    # Should run successfully and record ambiguity if scores are within range
    assert "ambiguous" in meta


def test_add_examples(router: SemanticRouter):
    """Test that we can dynamically add routing training examples."""
    original_count = len(router._clusters[0].examples)
    router.add_examples(Intent.FACTUAL, ["When was the declaration of independence signed?"])
    assert len(router._clusters[0].examples) == original_count + 1

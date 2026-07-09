"""Unit tests for the provider cache-token accounting classification."""
from __future__ import annotations

import pytest

from app.ai.llm.cache_accounting import cache_tokens_are_additive


@pytest.mark.parametrize("provider", ["anthropic", "bedrock", "Anthropic", " BEDROCK "])
def test_additive_providers(provider):
    assert cache_tokens_are_additive(provider) is True


@pytest.mark.parametrize("provider", ["openai", "azure", "google", "custom", "", None, "unknown"])
def test_folded_or_unknown_providers(provider):
    assert cache_tokens_are_additive(provider) is False

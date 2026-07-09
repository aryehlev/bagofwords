"""Provider classification for prompt-cache token accounting.

Providers disagree on how cached tokens relate to the reported prompt/input
token count:

  * **Additive** (Anthropic, and Bedrock which serves Anthropic models with the
    same accounting): ``cache_read_input_tokens`` and
    ``cache_creation_input_tokens`` are reported *separately* from
    ``input_tokens``. Totals (quota, spend) must add them in, and cost math
    prices them at their own rates (0.1x reads / 1.25x writes).

  * **Folded** (OpenAI/Azure, Google): cached tokens are already *included* in
    the prompt token count. Adding them again would double-count; cost math
    instead applies a discount to the cached portion.

Shared by the quota accounting in ``app.ai.llm.llm`` and the cost recorder in
``app.services.llm_usage_recorder`` so the two never drift apart.
"""

from __future__ import annotations

from typing import Optional

# Providers whose cache read/creation token counts are reported separately from
# (i.e. in addition to) the prompt/input token count.
_ADDITIVE_CACHE_PROVIDERS = frozenset({"anthropic", "bedrock"})


def cache_tokens_are_additive(provider_type: Optional[str]) -> bool:
    """True when ``provider_type`` reports cache tokens separately from prompt
    tokens (Anthropic/Bedrock), False for providers that fold cached tokens into
    the prompt count (OpenAI, Azure, Google, ...) or when unknown."""
    if not provider_type:
        return False
    return str(provider_type).strip().lower() in _ADDITIVE_CACHE_PROVIDERS

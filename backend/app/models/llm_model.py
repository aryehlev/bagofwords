from sqlalchemy import Column, String, JSON, ForeignKey, Boolean, Integer, Float
from sqlalchemy.orm import relationship
from app.models.base import BaseSchema

LLM_MODEL_DETAILS = [
    {
        "name": "GPT-5.5",
        "model_id": "gpt-5.5",
        "provider_type": "openai",
        "is_preset": True,
        "is_enabled": True,
        "is_default": True,
        "supports_vision": True,
        "context_window_tokens": 1050000,
        "input_cost_per_million_tokens_usd": 5.00,
        "output_cost_per_million_tokens_usd": 30.00
    },
    {
        "name": "GPT-5.4",
        "model_id": "gpt-5.4",
        "provider_type": "openai",
        "is_preset": True,
        "is_enabled": True,
        "is_default": False,
        "supports_vision": True,
        "context_window_tokens": 400000,
        "input_cost_per_million_tokens_usd": 2.50,
        "output_cost_per_million_tokens_usd": 15.00
    },
    {
        "name": "GPT-5.4 Mini",
        "model_id": "gpt-5.4-mini",
        "provider_type": "openai",
        "is_preset": True,
        "is_enabled": True,
        "is_default": False,
        "is_small_default": True,
        "supports_vision": True,
        "context_window_tokens": 400000,
        "input_cost_per_million_tokens_usd": 0.75,
        "output_cost_per_million_tokens_usd": 4.50
    },
    {
        "name": "GPT-5.2",
        "model_id": "gpt-5.2",
        "provider_type": "openai",
        "is_preset": True,
        "is_enabled": True,
        "is_default": False,
        "supports_vision": True,
        "context_window_tokens": 400000,
        "input_cost_per_million_tokens_usd": 1.75,
        "output_cost_per_million_tokens_usd": 14.00
    },
    {
        "name": "Claude 4.6 Sonnet",
        "model_id": "claude-sonnet-4-6",
        "provider_type": "anthropic",
        "is_preset": True,
        "is_enabled": True,
        "is_default": True,
        "supports_vision": True,
        "context_window_tokens": 200000,
        "input_cost_per_million_tokens_usd": 3.00,
        "output_cost_per_million_tokens_usd": 15.00
    },
  
    {
        "name": "Claude 4.6 Opus",
        "model_id": "claude-opus-4-6",
        "provider_type": "anthropic",
        "is_preset": True,
        "is_enabled": True,
        "is_default": False,
        "supports_vision": True,
        "context_window_tokens": 1000000,
        "input_cost_per_million_tokens_usd": 5.00,
        "output_cost_per_million_tokens_usd": 25.00
    },
  {
        "name": "Claude 4.5 Sonnet",
        "model_id": "claude-sonnet-4-5-20250929",
        "provider_type": "anthropic",
        "is_preset": True,
        "is_enabled": True,
        "is_default": False,
        "supports_vision": True,
        "context_window_tokens": 200000,
        "input_cost_per_million_tokens_usd": 3.00,
        "output_cost_per_million_tokens_usd": 15.00
    },
    {
        "name": "Claude 4.5 Opus",
        "model_id": "claude-opus-4-5-20251101",
        "provider_type": "anthropic",
        "is_preset": True,
        "is_enabled": True,
        "is_default": False,
        "supports_vision": True,
        "context_window_tokens": 200000,
        "input_cost_per_million_tokens_usd": 5.00,
        "output_cost_per_million_tokens_usd": 25.00
    },
    {
        "name": "Claude 4.5 Haiku",
        "model_id": "claude-haiku-4-5-20251001",
        "provider_type": "anthropic",
        "is_preset": True,
        "is_enabled": True,
        "is_small_default": True,
        "is_default": False,
        "supports_vision": True,
        "context_window_tokens": 200000,
        "input_cost_per_million_tokens_usd": 1,
        "output_cost_per_million_tokens_usd": 5.00
    },
    {
        "name": "Gemini 3.1 Pro Preview",
        "model_id": "gemini-3.1-pro-preview",
        "provider_type": "google",
        "is_preset": True,
        "is_enabled": True,
        "is_default": False,
        "is_small_default": False,
        "supports_vision": True,
        "context_window_tokens": 200000,
        "input_cost_per_million_tokens_usd": 2.00,
        "output_cost_per_million_tokens_usd": 12.00
    },
    {
        "name": "Gemini 2.5 Pro",
        "model_id": "gemini-2.5-pro",
        "provider_type": "google",
        "is_preset": True,
        "is_enabled": True,
        "is_default": True,
        "supports_vision": True,
        "context_window_tokens": 1047576,
        "input_cost_per_million_tokens_usd": 1.25,
        "output_cost_per_million_tokens_usd": 10.00
    },
    {
        "name": "Gemini 2.5 Flash",
        "model_id": "gemini-2.5-flash",
        "provider_type": "google",
        "is_preset": True,
        "is_enabled": True,
        "is_small_default": True,
        "supports_vision": True,
        "context_window_tokens": 1047576,
        "input_cost_per_million_tokens_usd": 0.30,
        "output_cost_per_million_tokens_usd": 2.50
    }
]

# Embedding-model catalog (model_type="embedding"). Kept *separate* from
# LLM_MODEL_DETAILS so the chat-model seeder (which force-enables every entry and
# may pick the first enabled one as the org default) never treats these as chat
# models. The API embedding backend is opt-in: an admin creates/enables one of
# these and points `organization_settings.default_embedding_model_id` at it. A
# deployment standardizes on one active model/dimension at a time (switching
# requires a re-embed); `embedding_dim` is the fixed vector width.
EMBEDDING_MODEL_DETAILS = [
    {
        "name": "OpenAI text-embedding-3-small",
        "model_id": "text-embedding-3-small",
        "provider_type": "openai",
        "embedding_dim": 1536,
    },
    {
        "name": "OpenAI text-embedding-3-large",
        "model_id": "text-embedding-3-large",
        "provider_type": "openai",
        "embedding_dim": 3072,
    },
    {
        "name": "Gemini text-embedding-004",
        "model_id": "text-embedding-004",
        "provider_type": "google",
        "embedding_dim": 768,
    },
]

# Fallback dimensions for known embedding model ids, used when a model row has
# no explicit `embedding_dim`.
EMBEDDING_MODEL_DIMS = {
    d["model_id"]: d["embedding_dim"] for d in EMBEDDING_MODEL_DETAILS
}

class LLMModel(BaseSchema):
    __tablename__ = "llm_models"
    
    name = Column(String, nullable=False)
    model_id = Column(String, nullable=False)  # The actual model ID used with the provider
    # Role of the model: "chat" (default) drives inference; "embedding" powers
    # semantic retrieval. A NULL value is treated as "chat" for back-compat.
    model_type = Column(String, nullable=False, default="chat", server_default="chat")
    embedding_dim = Column(Integer, nullable=True)  # Vector width for embedding models
    is_custom = Column(Boolean, default=False)  # Whether this is a custom model ID
    config = Column(JSON, nullable=True)  # Model-specific configurations
    is_preset = Column(Boolean, default=False, nullable=False)  # If True, cannot be deleted
    is_enabled = Column(Boolean, default=True, nullable=False)  # Can be disabled but not deleted
    is_default = Column(Boolean, default=False, nullable=False)  # If True, this is the default model for the organization
    is_small_default = Column(Boolean, default=False, nullable=False)  # Optional small default model per organization
    supports_vision = Column(Boolean, default=False, nullable=False)  # Whether model accepts image inputs
    # Token limits
    context_window_tokens = Column(Integer, nullable=True)  # Max prompt+completion tokens
    max_output_tokens = Column(Integer, nullable=True)  # Max model output tokens
    # Pricing (USD per million tokens)
    input_cost_per_million_tokens_usd = Column(Float, nullable=True)
    output_cost_per_million_tokens_usd = Column(Float, nullable=True)
    
    provider_id = Column(String, ForeignKey('llm_providers.id'), nullable=False)
    provider = relationship("LLMProvider", back_populates="models", lazy="selectin")
    organization_id = Column(String, ForeignKey('organizations.id'), nullable=False)
    organization = relationship("Organization", back_populates="llm_models", lazy="selectin")

    # Pricing helpers -----------------------------------------------------
    def _get_static_details(self) -> dict | None:
        for detail in LLM_MODEL_DETAILS:
            if detail.get("model_id") == self.model_id:
                return detail
        return None

    def get_input_cost_rate(self) -> float | None:
        if self.input_cost_per_million_tokens_usd is not None:
            return float(self.input_cost_per_million_tokens_usd)
        detail = self._get_static_details()
        if detail:
            return detail.get("input_cost_per_million_tokens_usd")
        return None

    def get_output_cost_rate(self) -> float | None:
        if self.output_cost_per_million_tokens_usd is not None:
            return float(self.output_cost_per_million_tokens_usd)
        detail = self._get_static_details()
        if detail:
            return detail.get("output_cost_per_million_tokens_usd")
        return None

    def get_embedding_dim(self) -> int | None:
        """Resolve the vector width for an embedding model.

        Prefers the explicit column, then the static preset detail, then the
        known-id fallback map.
        """
        if self.embedding_dim is not None:
            return int(self.embedding_dim)
        detail = self._get_static_details()
        if detail and detail.get("embedding_dim"):
            return int(detail["embedding_dim"])
        return EMBEDDING_MODEL_DIMS.get(self.model_id)

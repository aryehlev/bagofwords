"""Local, CPU-only embedding backend.

Wraps ``fastembed`` (ONNX runtime, no torch) with the small, fast
``BAAI/bge-small-en-v1.5`` model (384-dim). The model is downloaded and cached
to disk on first use and runs fully offline afterwards, so a self-hosted
deployment with no external API access still gets semantic retrieval out of the
box.

The wrapper is a lazy process-wide singleton: the (~tens of MB) model is loaded
once on first ``embed()`` and reused. Loading + inference are synchronous and
CPU-bound, so callers must run this off the event loop (see
``EmbeddingService.embed_texts`` which dispatches via ``run_in_executor``).
"""

from __future__ import annotations

import threading
from typing import List, Optional

from app.settings.logging_config import get_logger

logger = get_logger(__name__)

# Public identifiers for the local model. ``MODEL_ID`` is stored on every
# embedding row (provenance + dimension guard); ``DIM`` is the fixed vector
# width and the default for the embeddings column.
MODEL_ID = "bge-small-en-v1.5"
FASTEMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384


class LocalEmbedder:
    """Lazy singleton around a fastembed ``TextEmbedding`` model."""

    _instance: Optional["LocalEmbedder"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self.model_id = MODEL_ID
        self.dim = DIM
        self._model = None
        self._load_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "LocalEmbedder":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                # Imported lazily so the dependency (and its ONNX runtime) is
                # only required when local embeddings are actually used.
                from fastembed import TextEmbedding

                logger.info("Loading local embedding model %s", FASTEMBED_MODEL_NAME)
                self._model = TextEmbedding(model_name=FASTEMBED_MODEL_NAME)
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Blocking/CPU-bound — call off the loop."""
        if not texts:
            return []
        model = self._ensure_model()
        # fastembed yields numpy arrays; normalize to plain Python floats so the
        # vectors serialize cleanly into JSON / pgvector / libSQL.
        return [list(map(float, vec)) for vec in model.embed(texts)]

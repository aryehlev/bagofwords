import asyncio
import base64
import hashlib
import json
import os
import time
from typing import AsyncGenerator, AsyncIterator, Optional

from google import genai
from google.genai import types

from app.ai.llm.clients.base import LLMClient
from app.ai.utils.token_counter import estimate_tokens_fast
from app.settings.logging_config import get_logger
from app.ai.llm.types import (
    ImageInput,
    LLMResponse,
    LLMStreamEvent,
    LLMUsage,
    Message,
    MessageStopEvent,
    ReasoningCompleteEvent,
    ReasoningDeltaEvent,
    ReasoningStartEvent,
    TextDeltaEvent,
    ToolSpec,
    ToolUseCompleteEvent,
    ToolUseInputDeltaEvent,
    ToolUseStartEvent,
    UsageEvent,
)

logger = get_logger(__name__)


def _explicit_cache_enabled() -> bool:
    return os.environ.get("BOW_GEMINI_EXPLICIT_CACHE", "1").lower() in ("1", "true", "yes")


def _cache_ttl_seconds() -> int:
    try:
        return max(60, int(os.environ.get("BOW_GEMINI_CACHE_TTL_SECONDS", "1800")))
    except (TypeError, ValueError):
        return 1800


# Gemini rejects explicit caches whose payload is below a model-specific token
# floor (1k–4k depending on the model). Guard with a conservative estimate so we
# don't pay a round-trip to create a cache the API will refuse; small prefixes
# fall through to the inline (uncached) path. Override via env for tuning.
def _cache_min_tokens() -> int:
    try:
        return max(0, int(os.environ.get("BOW_GEMINI_CACHE_MIN_TOKENS", "2048")))
    except (TypeError, ValueError):
        return 2048


class _GeminiCacheManager:
    """Process-local registry of Gemini explicit caches (``CachedContent``).

    The static prefix of an agent request — the system instruction plus the
    tool declarations — is byte-identical across every iteration of a planner
    run and, because it derives from org-level config, across users and turns
    hitting the same model. Caching it explicitly lets Gemini bill those tokens
    at the cache-hit rate instead of full price on every call.

    Keyed by a hash of ``(model_id, system, tools)``. Entries carry a local
    expiry slightly inside the server-side TTL so we recreate before the cache
    is evicted server-side. All failures degrade to ``None`` → the caller uses
    the inline path, so caching never breaks a request.
    """

    def __init__(self) -> None:
        # key -> (cache_name, local_expiry_monotonic)
        self._entries: dict[str, tuple[str, float]] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _signature(model_id: str, system: Optional[str], tools_payload: str) -> str:
        h = hashlib.sha256()
        h.update((model_id or "").encode("utf-8"))
        h.update(b"\x00")
        h.update((system or "").encode("utf-8"))
        h.update(b"\x00")
        h.update(tools_payload.encode("utf-8"))
        return h.hexdigest()

    async def get_or_create(
        self,
        *,
        client: "genai.Client",
        model_id: str,
        system: Optional[str],
        tools: Optional[list["types.Tool"]],
        tools_payload: str,
        estimated_tokens: int,
    ) -> Optional[str]:
        if not _explicit_cache_enabled():
            return None
        if estimated_tokens < _cache_min_tokens():
            # Below the model's cache floor — not worth a create round-trip.
            return None

        key = self._signature(model_id, system, tools_payload)
        now = time.monotonic()

        async with self._lock:
            entry = self._entries.get(key)
            if entry and entry[1] > now:
                return entry[0]

            ttl = _cache_ttl_seconds()
            cfg_kwargs: dict = {"ttl": f"{ttl}s", "display_name": f"bow-prefix-{key[:16]}"}
            if system:
                cfg_kwargs["system_instruction"] = system
            if tools:
                cfg_kwargs["tools"] = tools

            try:
                loop = asyncio.get_running_loop()
                cached = await loop.run_in_executor(
                    None,
                    lambda: client.caches.create(
                        model=model_id,
                        config=types.CreateCachedContentConfig(**cfg_kwargs),
                    ),
                )
            except Exception as exc:  # noqa: BLE001 — caching must never break a call
                logger.warning(
                    "Gemini explicit cache create failed (model=%s); using inline path: %s",
                    model_id,
                    exc,
                )
                return None

            name = getattr(cached, "name", None)
            if not name:
                return None
            # Expire locally 60s before the server TTL to avoid a race where we
            # reference a cache the server has just evicted.
            self._entries[key] = (name, now + max(30, ttl - 60))
            logger.debug(
                "Gemini explicit cache created (model=%s, name=%s, ttl=%ss, ~%s tokens)",
                model_id, name, ttl, estimated_tokens,
            )
            return name

    async def evict(self, name: str) -> None:
        async with self._lock:
            for key, (cname, _) in list(self._entries.items()):
                if cname == name:
                    self._entries.pop(key, None)


_CACHE_MANAGER = _GeminiCacheManager()


class Google(LLMClient):
    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.client = genai.Client(api_key=api_key)
        self.temperature = 0.3

    @staticmethod
    def _build_contents(prompt: str, images: Optional[list[ImageInput]] = None) -> str | list:
        """Build contents, either as string or list with Parts for multimodal."""
        if not images:
            return prompt.strip()

        contents = []
        for img in images:
            if img.source_type == "url":
                # For URLs, use Part.from_uri (works with gs:// or https://)
                contents.append(types.Part.from_uri(file_uri=img.data, mime_type=img.media_type))
            else:
                # For base64, decode and use Part.from_bytes
                image_bytes = base64.b64decode(img.data)
                contents.append(types.Part.from_bytes(data=image_bytes, mime_type=img.media_type))
        contents.append(types.Part.from_text(text=prompt.strip()))
        return contents

    def inference(self, model_id: str, prompt: str, images: Optional[list[ImageInput]] = None) -> LLMResponse:
        thinking_budget = 128 if "pro" in model_id else 0

        response = self.client.models.generate_content(
            model=model_id,
            contents=self._build_contents(prompt, images),
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
                temperature=self.temperature,
            ),
        )
        usage_meta = getattr(response, "usage_metadata", None)
        usage = LLMUsage(
            prompt_tokens=getattr(usage_meta, "prompt_token_count", 0) if usage_meta else 0,
            completion_tokens=getattr(usage_meta, "candidates_token_count", 0) if usage_meta else 0,
        )
        self._set_last_usage(usage)
        text = getattr(response, "text", "") or ""
        return LLMResponse(text=text, usage=usage)

    async def inference_stream(
        self, model_id: str, prompt: str, images: Optional[list[ImageInput]] = None
    ) -> AsyncGenerator[str, None]:
        thinking_budget = 128 if "pro" in model_id else 0

        prompt_tokens = 0
        completion_tokens = 0
        for chunk in self.client.models.generate_content_stream(
            model=model_id,
            contents=self._build_contents(prompt, images),
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
                temperature=self.temperature,
            ),
        ):
            text = getattr(chunk, "text", None)
            if text:
                yield text
            usage_meta = getattr(chunk, "usage_metadata", None)
            if usage_meta:
                prompt_tokens = getattr(usage_meta, "prompt_token_count", prompt_tokens) or prompt_tokens
                completion_tokens = getattr(usage_meta, "candidates_token_count", completion_tokens) or completion_tokens

        self._set_last_usage(
            LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    @staticmethod
    def _translate_messages(messages: list[Message]) -> list[types.Content]:
        # First pass: build tool_use_id → name map for function_response translation
        id_to_name: dict[str, str] = {}
        for msg in messages:
            if isinstance(msg.content, list):
                for b in msg.content:
                    if b.get("type") == "tool_use":
                        id_to_name[b["id"]] = b["name"]

        out: list[types.Content] = []
        for msg in messages:
            role = "model" if msg.role == "assistant" else "user"
            if isinstance(msg.content, str):
                out.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))
                continue

            blocks = msg.content
            text_blocks = [b for b in blocks if b.get("type") == "text"]
            tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
            tool_results = [b for b in blocks if b.get("type") == "tool_result"]

            if tool_results:
                parts = []
                for tr in tool_results:
                    tool_id = tr["tool_use_id"]
                    name = id_to_name.get(tool_id, tool_id)
                    content = tr.get("content", "")
                    if not isinstance(content, str):
                        content = json.dumps(content, default=str)
                    parts.append(types.Part.from_function_response(
                        name=name,
                        response={"output": content},
                    ))
                out.append(types.Content(role="user", parts=parts))
            elif tool_uses:
                parts = []
                for tc in tool_uses:
                    parts.append(types.Part.from_function_call(
                        name=tc["name"],
                        args=tc.get("input", {}),
                    ))
                out.append(types.Content(role="model", parts=parts))
            else:
                text = " ".join(b.get("text", "") for b in text_blocks)
                out.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))
        return out

    # Fields not accepted by Google's FunctionDeclaration schema validator
    _GOOGLE_SCHEMA_STRIP = frozenset({
        "$defs", "$schema", "choices", "examples", "default", "title",
        "additionalProperties",
    })

    @staticmethod
    def _resolve_schema_refs(schema: dict) -> dict:
        """Inline $ref references and strip / convert fields Google's SDK doesn't accept."""
        defs = schema.get("$defs", {})

        def _resolve(node: any) -> any:
            if isinstance(node, dict):
                if "$ref" in node:
                    ref = node["$ref"]
                    if ref.startswith("#/$defs/"):
                        def_name = ref[len("#/$defs/"):]
                        resolved = defs.get(def_name, {})
                        return _resolve(resolved)
                    return {"type": "string"}  # unresolvable ref → fallback
                # Convert JSON Schema "const" → "enum" (Google supports enum, not const)
                if "const" in node:
                    result = {k: _resolve(v) for k, v in node.items() if k not in Google._GOOGLE_SCHEMA_STRIP and k != "const"}
                    result["enum"] = [node["const"]]
                    return result
                result = {
                    k: _resolve(v)
                    for k, v in node.items()
                    if k not in Google._GOOGLE_SCHEMA_STRIP
                }
                # Drop required entries that reference undefined properties
                if "required" in result and "properties" in result:
                    defined = set(result["properties"].keys())
                    result["required"] = [r for r in result["required"] if r in defined]
                    if not result["required"]:
                        del result["required"]
                return result
            if isinstance(node, list):
                return [_resolve(i) for i in node]
            return node

        return _resolve(schema)

    @staticmethod
    def _translate_tools(tools: list[ToolSpec]) -> list[types.Tool]:
        declarations = [
            types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters=Google._resolve_schema_refs(t.input_schema),
            )
            for t in tools
        ]
        return [types.Tool(function_declarations=declarations)]

    async def inference_stream_v2(
        self,
        model_id: str,
        messages: list[Message],
        system: Optional[str] = None,
        tools: Optional[list[ToolSpec]] = None,
        images: Optional[list[ImageInput]] = None,
        thinking: Optional[dict] = None,
        disable_parallel_tools: bool = True,
    ) -> AsyncIterator[LLMStreamEvent]:
        if thinking:
            budget = int(thinking.get("budget_tokens") or 1024)
            thinking_config = types.ThinkingConfig(thinking_budget=budget, include_thoughts=True)
        else:
            default_budget = 128 if "pro" in model_id else 0
            thinking_config = types.ThinkingConfig(thinking_budget=default_budget, include_thoughts=False)
        config_kwargs: dict = {
            "thinking_config": thinking_config,
            "temperature": self.temperature,
        }

        translated_tools = self._translate_tools(tools) if tools else None

        # Explicit caching: hoist the static (system + tools) prefix into a
        # server-side CachedContent and reference it by name. When a cache is in
        # play we MUST NOT also pass system_instruction/tools inline — they live
        # in the cache, and the API rejects supplying both. Falls back to the
        # inline path whenever caching is disabled, the prefix is too small, or
        # cache creation fails (get_or_create returns None).
        tools_payload = json.dumps(
            [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
             for t in (tools or [])],
            sort_keys=True, default=str,
        )
        estimated_prefix_tokens = estimate_tokens_fast((system or "") + tools_payload)
        cache_name = await _CACHE_MANAGER.get_or_create(
            client=self.client,
            model_id=model_id,
            system=system,
            tools=translated_tools,
            tools_payload=tools_payload,
            estimated_tokens=estimated_prefix_tokens,
        )
        if cache_name:
            config_kwargs["cached_content"] = cache_name
        else:
            if system:
                config_kwargs["system_instruction"] = system
            if translated_tools:
                config_kwargs["tools"] = translated_tools

        contents = self._translate_messages(messages)
        prompt_tokens = 0
        completion_tokens = 0
        stop_reason = "end_turn"

        loop = asyncio.get_running_loop()

        def _collect(cfg_kwargs: dict):
            chunks = []
            for chunk in self.client.models.generate_content_stream(
                model=model_id,
                contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs),
            ):
                chunks.append(chunk)
            return chunks

        try:
            chunks = await loop.run_in_executor(None, lambda: _collect(config_kwargs))
        except Exception as exc:  # noqa: BLE001
            # A stale/evicted cache reference fails the request. Evict our record
            # and retry once on the inline path so a bad cache never breaks a call.
            if cache_name:
                logger.warning(
                    "Gemini generate failed with cached_content=%s; retrying inline: %s",
                    cache_name, exc,
                )
                await _CACHE_MANAGER.evict(cache_name)
                config_kwargs.pop("cached_content", None)
                if system:
                    config_kwargs["system_instruction"] = system
                if translated_tools:
                    config_kwargs["tools"] = translated_tools
                chunks = await loop.run_in_executor(None, lambda: _collect(config_kwargs))
            else:
                raise

        tool_call_counter = 0
        reasoning_started = False
        cache_read_tokens = 0
        for chunk in chunks:
            usage_meta = getattr(chunk, "usage_metadata", None)
            if usage_meta:
                prompt_tokens = getattr(usage_meta, "prompt_token_count", prompt_tokens) or prompt_tokens
                completion_tokens = getattr(usage_meta, "candidates_token_count", completion_tokens) or completion_tokens
                # Tokens served from the explicit cache (billed at the cache-hit
                # rate). Gemini reports them separately from prompt_token_count.
                cache_read_tokens = getattr(usage_meta, "cached_content_token_count", cache_read_tokens) or cache_read_tokens

            candidate = chunk.candidates[0] if chunk.candidates else None
            if not candidate:
                continue

            finish = getattr(candidate, "finish_reason", None)
            if finish:
                finish_name = getattr(finish, "name", str(finish))
                if finish_name in ("STOP",):
                    stop_reason = "end_turn"
                elif finish_name in ("MAX_TOKENS",):
                    stop_reason = "max_tokens"

            for part in (candidate.content.parts if candidate.content else []):
                is_thought = getattr(part, "thought", False)
                if part.text and is_thought:
                    if not reasoning_started:
                        reasoning_started = True
                        yield ReasoningStartEvent()
                    yield ReasoningDeltaEvent(text=part.text)
                elif part.text and not is_thought:
                    if reasoning_started:
                        reasoning_started = False
                        yield ReasoningCompleteEvent(text="")
                    yield TextDeltaEvent(text=part.text)
                fc = getattr(part, "function_call", None)
                if fc:
                    stop_reason = "tool_use"
                    call_id = f"call_{tool_call_counter}"
                    tool_call_counter += 1
                    yield ToolUseStartEvent(id=call_id, name=fc.name)
                    args_json = json.dumps(dict(fc.args))
                    yield ToolUseInputDeltaEvent(id=call_id, partial_json=args_json)
                    yield ToolUseCompleteEvent(id=call_id, name=fc.name, input=dict(fc.args))

        if reasoning_started:
            yield ReasoningCompleteEvent(text="")

        yield MessageStopEvent(stop_reason=stop_reason)
        yield UsageEvent(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens,
        )
        self._set_last_usage(LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens,
        ))


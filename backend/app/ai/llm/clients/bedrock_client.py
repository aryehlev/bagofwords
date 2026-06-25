import asyncio
import base64
import json
import os
from collections.abc import AsyncGenerator, AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import boto3

from app.ai.llm.clients.base import LLMClient
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

_STREAM_EXECUTOR = ThreadPoolExecutor(max_workers=4)

logger = get_logger(__name__)


def _prompt_cache_enabled() -> bool:
    return os.environ.get("BOW_BEDROCK_PROMPT_CACHE", "1").lower() in ("1", "true", "yes")


# Bedrock Converse marks cacheable prefixes with explicit `cachePoint` blocks:
# everything up to and including the marker is cached, billed at the cache-read
# rate on subsequent identical-prefix requests. We mark the system block and the
# end of the tool list — both static across a planner run — mirroring the
# Anthropic client. Only a subset of Bedrock models support caching (Claude 3.5+,
# Nova); unsupported models raise a ValidationException, which we recover from by
# retrying once without cache points (see `_converse_stream_with_fallback`).
_CACHE_POINT = {"cachePoint": {"type": "default"}}

# Models that have rejected a cache-point request at least once this process.
# Once a modelId lands here we skip the cached attempt entirely and go straight
# to the plain request, avoiding a redundant ValidationException on every call.
_CACHE_POINT_UNSUPPORTED_MODELS: set[str] = set()


# Map MIME types to Bedrock image format strings
_MIME_TO_FORMAT = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}


class BedrockClient(LLMClient):
    """
    AWS Bedrock client using the native Converse API (boto3).

    Auth modes:
      - iam: uses the standard AWS credential chain (IRSA, env vars, instance role, etc.)
      - access_keys: uses explicit AWS access key ID and secret access key

    Supports application inference profiles — pass the profile ARN as model_id.
    """

    # TODO: Add api_key auth mode support.
    # When boto3 supports passing Bedrock API keys as a client parameter
    # (see https://github.com/boto/boto3/issues/4723), add api_key auth here.
    # Current workaround would be os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
    # but that's process-global and unsafe for multi-tenant setups.

    _SUPPORTED_AUTH_MODES = {"iam", "access_keys"}

    def __init__(
        self,
        region: str,
        auth_mode: str = "iam",
        api_key: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        super().__init__()
        if auth_mode not in self._SUPPORTED_AUTH_MODES:
            raise ValueError(
                f"Unsupported auth_mode '{auth_mode}'. "
                f"Supported modes: {', '.join(sorted(self._SUPPORTED_AUTH_MODES))}."
            )

        if auth_mode == "access_keys":
            if not aws_access_key_id or not aws_secret_access_key:
                raise ValueError(
                    "Bedrock auth_mode 'access_keys' requires both "
                    "aws_access_key_id and aws_secret_access_key."
                )
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region,
            )
            self.client = session.client("bedrock-runtime")
        else:
            self.client = boto3.client("bedrock-runtime", region_name=region)

        self._region = region
        self._auth_mode = auth_mode

    async def embed(self, model_id: str, texts: list[str]) -> list[list[float]]:
        """Embed via Bedrock ``invoke_model`` (Titan-style, one input per call).

        Titan Text Embeddings accept a single ``inputText`` per request, so we
        issue one call per text. Runs the blocking boto3 calls on the shared
        stream executor to avoid blocking the event loop.
        """
        if not texts:
            return []

        def _embed_one(text: str) -> list[float]:
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps({"inputText": text}),
            )
            payload = json.loads(response["body"].read())
            return [float(x) for x in payload["embedding"]]

        loop = asyncio.get_running_loop()
        futures = [
            loop.run_in_executor(_STREAM_EXECUTOR, _embed_one, t)
            for t in texts
        ]
        return await asyncio.gather(*futures)

    @staticmethod
    def _build_content(prompt: str, images: Optional[list[ImageInput]] = None) -> list[dict]:
        """Build Bedrock message content blocks."""
        content: list[dict] = []

        if images:
            for img in images:
                if img.source_type == "url":
                    # Bedrock converse only supports bytes/S3 for images, skip URLs
                    continue
                fmt = _MIME_TO_FORMAT.get(img.media_type, "png")
                image_bytes = base64.b64decode(img.data)
                content.append({
                    "image": {
                        "format": fmt,
                        "source": {"bytes": image_bytes},
                    }
                })

        content.append({"text": prompt.strip()})
        return content

    def inference(self, model_id: str, prompt: str, images: Optional[list[ImageInput]] = None) -> LLMResponse:
        response = self.client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": self._build_content(prompt, images)}],
        )

        # Extract text from response
        output_message = response["output"]["message"]
        text = ""
        for block in output_message.get("content", []):
            if "text" in block:
                text += block["text"]

        # Extract usage
        usage_data = response.get("usage", {})
        usage = LLMUsage(
            prompt_tokens=usage_data.get("inputTokens", 0),
            completion_tokens=usage_data.get("outputTokens", 0),
        )
        self._set_last_usage(usage)
        return LLMResponse(text=text, usage=usage)

    async def inference_stream(
        self, model_id: str, prompt: str, images: Optional[list[ImageInput]] = None
    ) -> AsyncGenerator[str, None]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        usage_holder: dict = {"inputTokens": 0, "outputTokens": 0}

        def _sync_stream():
            """Run the blocking boto3 stream in a worker thread."""
            try:
                response = self.client.converse_stream(
                    modelId=model_id,
                    messages=[{"role": "user", "content": self._build_content(prompt, images)}],
                )
                for event in response["stream"]:
                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"].get("delta", {})
                        text = delta.get("text")
                        if text:
                            loop.call_soon_threadsafe(queue.put_nowait, text)

                    if "metadata" in event:
                        usage = event["metadata"].get("usage", {})
                        usage_holder["inputTokens"] = usage.get("inputTokens", usage_holder["inputTokens"])
                        usage_holder["outputTokens"] = usage.get("outputTokens", usage_holder["outputTokens"])
            finally:
                # Always signal end of stream so the async generator unblocks
                loop.call_soon_threadsafe(queue.put_nowait, None)

        future = loop.run_in_executor(_STREAM_EXECUTOR, _sync_stream)

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

        # Ensure the thread has finished and propagate any exceptions
        await future

        self._set_last_usage(
            LLMUsage(
                prompt_tokens=usage_holder["inputTokens"],
                completion_tokens=usage_holder["outputTokens"],
            )
        )

    @staticmethod
    def _translate_messages(messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for msg in messages:
            role = "assistant" if msg.role == "assistant" else "user"
            if isinstance(msg.content, str):
                out.append({"role": role, "content": [{"text": msg.content}]})
                continue

            blocks = msg.content
            text_blocks = [b for b in blocks if b.get("type") == "text"]
            tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
            tool_results = [b for b in blocks if b.get("type") == "tool_result"]

            if tool_results:
                content = []
                for tr in tool_results:
                    tr_content = tr.get("content", "")
                    if not isinstance(tr_content, str):
                        tr_content = json.dumps(tr_content, default=str)
                    content.append({
                        "toolResult": {
                            "toolUseId": tr["tool_use_id"],
                            "content": [{"text": tr_content}],
                        }
                    })
                out.append({"role": "user", "content": content})
            elif tool_uses:
                content = []
                for tc in tool_uses:
                    content.append({
                        "toolUse": {
                            "toolUseId": tc["id"],
                            "name": tc["name"],
                            "input": tc.get("input", {}),
                        }
                    })
                if text_blocks:
                    text = " ".join(b.get("text", "") for b in text_blocks)
                    content.insert(0, {"text": text})
                out.append({"role": "assistant", "content": content})
            else:
                text = " ".join(b.get("text", "") for b in text_blocks)
                out.append({"role": role, "content": [{"text": text}]})
        return out

    @staticmethod
    def _translate_tools(tools: list[ToolSpec]) -> dict:
        return {
            "tools": [
                {
                    "toolSpec": {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": {"json": t.input_schema},
                    }
                }
                for t in tools
            ]
        }

    @staticmethod
    def _with_cache_points(request_kwargs: dict) -> dict:
        """Return a copy of the request with cachePoint markers on the static
        (system + tools) prefix. Everything up to and including each marker is
        cached by Bedrock.

        Only ``system`` and ``toolConfig`` are rebuilt; ``messages`` (which may
        carry large image bytes) is shared by reference since it isn't mutated.
        """
        cached = dict(request_kwargs)
        if cached.get("system"):
            cached["system"] = list(cached["system"]) + [_CACHE_POINT]
        tool_cfg = cached.get("toolConfig")
        if tool_cfg and tool_cfg.get("tools"):
            cached["toolConfig"] = {**tool_cfg, "tools": list(tool_cfg["tools"]) + [_CACHE_POINT]}
        return cached

    def _converse_stream_with_fallback(self, cached_kwargs: Optional[dict], plain_kwargs: dict):
        """Start a converse_stream, preferring the cache-marked request.

        Request validation (including unsupported cachePoint blocks) happens when
        converse_stream() is called, before any event is iterated — so on a
        cache-related ValidationException we can transparently retry the plain
        request. Non-cache errors propagate unchanged.
        """
        model_id = str(plain_kwargs.get("modelId") or "")
        if cached_kwargs is None or model_id in _CACHE_POINT_UNSUPPORTED_MODELS:
            return self.client.converse_stream(**plain_kwargs)
        try:
            return self.client.converse_stream(**cached_kwargs)
        except Exception as exc:  # noqa: BLE001 — caching must never break a call
            msg = str(exc).lower()
            if "cachepoint" in msg or "cache" in msg or "validationexception" in msg:
                if model_id:
                    _CACHE_POINT_UNSUPPORTED_MODELS.add(model_id)
                logger.warning(
                    "Bedrock prompt caching rejected (model=%s); retrying without cache points: %s",
                    plain_kwargs.get("modelId"), exc,
                )
                return self.client.converse_stream(**plain_kwargs)
            raise

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
        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue = asyncio.Queue()

        bedrock_messages = self._translate_messages(messages)
        request_kwargs: dict = {"modelId": model_id, "messages": bedrock_messages}
        if system:
            request_kwargs["system"] = [{"text": system}]
        if thinking:
            budget = int(thinking.get("budget_tokens") or 5000)
            request_kwargs["additionalModelRequestFields"] = {
                "thinking": {"type": "enabled", "budget_tokens": budget}
            }
        if tools:
            tc = self._translate_tools(tools)
            # disableParallelToolUse in toolChoice.auto requires botocore ≥ 1.37;
            # skip it to keep compatibility with older botocore versions.
            request_kwargs["toolConfig"] = tc

        # Prompt caching: mark the static (system + tools) prefix with cachePoint
        # blocks. Disabled when thinking is on — interleaved reasoning changes the
        # cacheable prefix and several models reject cache points alongside it.
        cache_enabled = _prompt_cache_enabled() and not thinking
        cached_kwargs = self._with_cache_points(request_kwargs) if cache_enabled else None

        def _sync_stream():
            try:
                response = self._converse_stream_with_fallback(cached_kwargs, request_kwargs)
                for event in response["stream"]:
                    loop.call_soon_threadsafe(event_queue.put_nowait, event)
            finally:
                loop.call_soon_threadsafe(event_queue.put_nowait, None)

        future = loop.run_in_executor(_STREAM_EXECUTOR, _sync_stream)

        # State for tracking open tool calls and reasoning blocks
        open_calls: dict[int, dict] = {}  # block_index → {id, name, args_buffer}
        open_reasoning: set[int] = set()  # block indices that are reasoning blocks
        current_block_index: int = -1
        prompt_tokens = 0
        completion_tokens = 0
        cache_read_tokens = 0
        cache_creation_tokens = 0
        stop_reason = "end_turn"

        while True:
            event = await event_queue.get()
            if event is None:
                break

            if "contentBlockStart" in event:
                block_start = event["contentBlockStart"]
                current_block_index = block_start.get("contentBlockIndex", current_block_index + 1)
                start = block_start.get("start", {})
                tool_use = start.get("toolUse")
                # reasoningContent may be stripped by old botocore as an unknown
                # tagged-union member; we also detect it lazily from delta events.
                reasoning = start.get("reasoningContent")
                if tool_use:
                    open_calls[current_block_index] = {
                        "id": tool_use["toolUseId"],
                        "name": tool_use["name"],
                        "args_buffer": "",
                    }
                    yield ToolUseStartEvent(
                        id=tool_use["toolUseId"],
                        name=tool_use["name"],
                    )
                elif reasoning is not None:
                    open_reasoning.add(current_block_index)
                    yield ReasoningStartEvent()

            elif "contentBlockDelta" in event:
                block_delta = event["contentBlockDelta"]
                idx = block_delta.get("contentBlockIndex", current_block_index)
                delta = block_delta.get("delta", {})

                if "text" in delta:
                    yield TextDeltaEvent(text=delta["text"])

                if "toolUse" in delta:
                    fragment = delta["toolUse"].get("input", "")
                    if fragment and idx in open_calls:
                        open_calls[idx]["args_buffer"] += fragment
                        yield ToolUseInputDeltaEvent(
                            id=open_calls[idx]["id"],
                            partial_json=fragment,
                        )

                if "reasoningContent" in delta:
                    rc = delta["reasoningContent"]
                    # botocore ≥1.37 exposes "text"; older raw API used "thinkingDelta"
                    text = rc.get("text", "") or rc.get("thinkingDelta", "")
                    if text:
                        # Lazily open a reasoning block if botocore stripped the start event
                        if idx not in open_reasoning:
                            open_reasoning.add(idx)
                            yield ReasoningStartEvent()
                        yield ReasoningDeltaEvent(text=text)

            elif "contentBlockStop" in event:
                idx = event["contentBlockStop"].get("contentBlockIndex", current_block_index)
                if idx in open_calls:
                    pending = open_calls[idx]
                    raw = pending["args_buffer"]
                    try:
                        parsed = json.loads(raw) if raw.strip() else {}
                    except Exception:
                        parsed = {"_unparsable": True, "_raw": raw}
                    yield ToolUseCompleteEvent(
                        id=pending["id"],
                        name=pending["name"],
                        input=parsed,
                    )
                elif idx in open_reasoning:
                    open_reasoning.discard(idx)
                    yield ReasoningCompleteEvent(text="")

            elif "messageStop" in event:
                bedrock_stop = event["messageStop"].get("stopReason", "end_turn")
                _stop_map = {"end_turn": "end_turn", "tool_use": "tool_use", "max_tokens": "max_tokens"}
                stop_reason = _stop_map.get(bedrock_stop, "other")

            elif "metadata" in event:
                usage = event["metadata"].get("usage", {})
                prompt_tokens = usage.get("inputTokens", prompt_tokens)
                completion_tokens = usage.get("outputTokens", completion_tokens)
                # Cache hit/write counts (present only on cache-capable models).
                cache_read_tokens = usage.get("cacheReadInputTokens", cache_read_tokens) or cache_read_tokens
                cache_creation_tokens = usage.get("cacheWriteInputTokens", cache_creation_tokens) or cache_creation_tokens

        await future

        yield MessageStopEvent(stop_reason=stop_reason)
        yield UsageEvent(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )
        self._set_last_usage(LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        ))

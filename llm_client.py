"""
LLM client: one robust way to get parsed JSON out of any provider's chat model
==============================================================================

Every LLM call in the pipeline goes through ``chat_json`` (sync) or
``achat_json`` (async). They exist because the raw calls failed silently, and
because "any provider" means the same request must survive very different
APIs.

Two backends
------------
``OpenAICompatBackend``  every provider that speaks OpenAI chat-completions
                         (OpenAI, Groq, Gemini, Mistral, DeepSeek, xAI,
                         OpenRouter, Together, ... and custom endpoints).
``AnthropicBackend``     Claude, through the official ``anthropic`` SDK.

Recovery, in the order it is tried
----------------------------------
* A parameter the model rejects (``response_format``, ``reasoning_effort``,
  ``seed``, ``temperature``, ``max_tokens`` vs ``max_completion_tokens``) is
  dropped or renamed and the call retried. What was actually sent is recorded.
* Truncated output (reasoning models spend output tokens on hidden reasoning)
  is retried with a larger budget. A truncated answer is never parsed: an
  inner fragment of cut-off JSON would look like a valid, wrong, answer.
* Per-minute limits are honoured: the provider's retry hint is obeyed, and a
  request larger than the whole per-minute allowance is shrunk, not repeated.
* A model that is retired, out of quota, refuses the request, or cannot fit the
  answer hands over to the next fallback model - on any connected provider -
  and the substitution is reported.

Each call returns an ``LLMResult`` carrying the model actually used, what
sampling controls were applied, the attempts made and token usage, so
provenance can be exported.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from model_registry import (
    MIN_CONTEXT_WINDOW, is_chat_model, make_ref, model_family, rank_models, split_ref,
)
from providers import PROVIDERS, Connection

MAX_TOKENS_CEILING = 16000
MAX_RATE_LIMIT_WAIT = 65.0
REQUEST_TIMEOUT = 120.0

# Hidden reasoning is billed as output tokens. "low" keeps it short for most
# models; Qwen3 ignores "low" and reasons at length, which alone exceeds a free
# tier's per-minute output allowance, so it is switched off there. A model that
# rejects the parameter simply has it dropped.
REASONING_EFFORT_BY_FAMILY = {"qwen": "none"}
EFFORT_LEVELS = ("low", "medium", "high")


class LLMError(RuntimeError):
    """The model could not produce usable JSON after all recovery attempts."""


@dataclass
class LLMResult:
    data: Any
    model: str                                   # the model that actually answered
    attempts: int = 1
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    notes: List[str] = field(default_factory=list)
    applied: Dict[str, Any] = field(default_factory=dict)   # sampling controls really sent

    def provenance(self) -> Dict:
        return {
            'model': self.model, 'attempts': self.attempts,
            'finish_reason': self.finish_reason,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'notes': self.notes,
            'applied': self.applied,
        }


@dataclass
class Completion:
    """A provider response, normalised."""
    text: Optional[str]
    finish_reason: str = "stop"                  # stop | length | refusal
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    served_model: Optional[str] = None           # provider-reported model ID
    applied: Dict[str, Any] = field(default_factory=dict)
    refusal_category: Optional[str] = None


@dataclass
class ErrorInfo:
    """What went wrong with a request, in provider-neutral terms."""
    kind: str               # model_not_found | bad_param | json_failed | quota | too_large |
                            # token_limit | rate_limit | auth | network | other
    message: str = ""
    param: Optional[str] = None
    retry_after: Optional[float] = None
    overshoot: Optional[int] = None      # tokens over a per-minute allowance
    limit: Optional[int] = None          # a model's hard output-token limit


# ============================================================
# JSON EXTRACTION
# ============================================================

def _is_truncated(text: str) -> bool:
    """True if ``text`` (starting at a '{' or '[') ends before its JSON closes."""
    depth = 0
    in_string = escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                return False
    return True


def extract_json(text: Optional[str]) -> Any:
    """
    Parse the first complete JSON value in ``text``.

    Tolerates markdown fences, <think> blocks and prose before or after the
    JSON. Raises ValueError when there is no JSON or when the JSON is cut off -
    it never returns an inner fragment of a truncated answer.
    """
    if not text or not text.strip():
        raise ValueError("empty response")

    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    fenced = re.search(r'```(?:json)?\s*(.*?)```', cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    for match in re.finditer(r'[\[{]', cleaned):
        candidate = cleaned[match.start():]
        try:
            value, _ = decoder.raw_decode(candidate)
            return value
        except json.JSONDecodeError:
            if _is_truncated(candidate):
                raise ValueError("truncated JSON in response") from None
            continue
    raise ValueError("no complete JSON value in response")


# ============================================================
# ERROR CLASSIFICATION (shared text heuristics)
# ============================================================

_OPENAI_OPTIONAL = ("reasoning_effort", "response_format", "seed", "temperature")


def _retry_after_from_text(text: str) -> Optional[float]:
    match = re.search(r'try again in (?:(\d+)m)?([0-9.]+)\s*(ms|s)', text.lower())
    if not match:
        return None
    seconds = float(match.group(2)) / (1000.0 if match.group(3) == 'ms' else 1.0)
    return seconds + 60.0 * int(match.group(1) or 0)


def _retry_after_from_headers(exc: Exception) -> Optional[float]:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def _token_overshoot(text: str) -> Optional[int]:
    """
    Tokens by which a request exceeds a per-minute limit OUTRIGHT, or None.

    Groq counts the requested ``max_tokens`` against the limit, so a request
    asking for more than the whole allowance can never succeed however long we
    wait ("Limit 1000, Requested 1920"). "Limit 1000, Used 396, Requested 900"
    is different: that passes once the window rolls over.
    """
    match = re.search(r'Limit (\d+),(?: Used \d+,)? Requested (\d+)', text)
    if not match:
        return None
    limit, requested = int(match.group(1)), int(match.group(2))
    return requested - limit if requested > limit else None


def _output_limit(text: str) -> Optional[int]:
    """A model's stated maximum output tokens, from a 'max_tokens too large' error."""
    lowered = text.lower()
    if "max_tokens" not in lowered and "max_completion_tokens" not in lowered \
            and "output tokens" not in lowered and "completion tokens" not in lowered:
        return None
    match = re.search(r'(?:at most|maximum(?: value)?(?: is| of| allowed is)?|less than or equal to|<=|'
                      r'up to|limit(?: is| of)?)\s*:?\s*(\d{3,7})', lowered)
    return int(match.group(1)) if match else None


def classify_text(text: str, status: Optional[int] = None) -> ErrorInfo:
    """Classify an error from its message (and HTTP status when known)."""
    lowered = text.lower()

    if "model_not_found" in lowered or "decommissioned" in lowered or "no longer supported" in lowered \
            or ("model" in lowered and ("does not exist" in lowered or "unknown model" in lowered
                                        or "is not a valid model" in lowered
                                        or (status == 404 and "not found" in lowered))):
        return ErrorInfo("model_not_found", text)

    if status == 429 or "429" in text or "rate limit" in lowered or "rate_limit" in lowered \
            or "insufficient_quota" in lowered or "exceeded your current quota" in lowered \
            or "resource_exhausted" in lowered or "too many requests" in lowered:
        # Only explicit daily / account-level markers. Generic words such as
        # "billing" appear in the upgrade link of ordinary per-minute 429s.
        if "per day" in lowered or "(tpd)" in lowered or "(rpd)" in lowered \
                or "insufficient_quota" in lowered or "exceeded your current quota" in lowered \
                or "credit balance" in lowered:
            return ErrorInfo("quota", text)
        overshoot = _token_overshoot(text)
        if overshoot is not None:
            return ErrorInfo("too_large", text, overshoot=overshoot)
        return ErrorInfo("rate_limit", text, retry_after=_retry_after_from_text(text))

    if "json_validate_failed" in lowered or "failed to generate json" in lowered:
        return ErrorInfo("json_failed", text)

    bad_request = status in (400, 422) or "400" in text or "422" in text or "invalid" in lowered \
        or "unsupported" in lowered or "not supported" in lowered or "unrecognized" in lowered \
        or "unknown parameter" in lowered or "extra inputs" in lowered or "not permitted" in lowered
    if bad_request:
        if re.search(r"unsupported parameter:?\s*'?max_tokens", lowered) \
                or re.search(r"'?\bmax_tokens'? is not supported", lowered):
            return ErrorInfo("bad_param", text, param="max_tokens")
        limit = _output_limit(text)
        if limit is not None:
            return ErrorInfo("token_limit", text, limit=limit)
        for name in _OPENAI_OPTIONAL:
            if name in lowered:
                return ErrorInfo("bad_param", text, param=name)

    if status in (401, 403) or "401" in text or "invalid api key" in lowered \
            or "incorrect api key" in lowered or "authentication" in lowered or "unauthorized" in lowered:
        return ErrorInfo("auth", text)

    return ErrorInfo("other", text)


# ============================================================
# BACKENDS
# ============================================================

def _split_system(messages: Sequence[Dict]) -> Tuple[str, List[Dict]]:
    """Separate system text from the turns (Anthropic takes it as a top-level field)."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    return system, turns


class OpenAICompatBackend:
    """Any endpoint that implements OpenAI chat completions."""

    def __init__(self, provider_id: str, api_key: str = "", base_url: Optional[str] = None,
                 client=None):
        self.provider_id = provider_id
        provider = PROVIDERS.get(provider_id)
        self.api_key = api_key or "not-needed"       # local servers ignore it, the SDK requires one
        self.base_url = base_url or (provider.base_url if provider else None)
        self._client = client
        self._async_client = None

    # ------------------------------------------------------------ clients

    def _sync(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                  max_retries=1, timeout=REQUEST_TIMEOUT)
        return self._client

    def _async(self):
        if self._async_client is None:
            from openai import AsyncOpenAI
            self._async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url,
                                             max_retries=1, timeout=REQUEST_TIMEOUT)
        return self._async_client

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None

    # ------------------------------------------------------------ requests

    def _params(self, model: str, messages: Sequence[Dict], max_tokens: int,
                temperature: Optional[float], seed: Optional[int], effort: Optional[str],
                disabled: set) -> Tuple[Dict, Dict]:
        params: Dict[str, Any] = {"model": model, "messages": list(messages)}
        # Newer OpenAI reasoning models only accept max_completion_tokens.
        params["max_completion_tokens" if "max_tokens" in disabled else "max_tokens"] = max_tokens
        applied: Dict[str, Any] = {"temperature": None, "seed": None, "effort": None, "json_mode": False}
        if temperature is not None and "temperature" not in disabled:
            params["temperature"] = applied["temperature"] = temperature
        if "response_format" not in disabled:
            params["response_format"] = {"type": "json_object"}
            applied["json_mode"] = True
        if effort and "reasoning_effort" not in disabled:
            params["reasoning_effort"] = applied["effort"] = effort
        if seed is not None and "seed" not in disabled:
            params["seed"] = applied["seed"] = seed
        return params, applied

    @staticmethod
    def _completion(response, applied: Dict) -> Completion:
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        finish = choice.finish_reason
        return Completion(
            text=choice.message.content,
            finish_reason={"length": "length", "content_filter": "refusal"}.get(finish, "stop"),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            served_model=getattr(response, "model", None),
            applied=applied,
        )

    def complete(self, model, messages, *, max_tokens, temperature, seed, effort, disabled) -> Completion:
        params, applied = self._params(model, messages, max_tokens, temperature, seed, effort, disabled)
        return self._completion(self._sync().chat.completions.create(**params), applied)

    async def acomplete(self, model, messages, *, max_tokens, temperature, seed, effort,
                        disabled) -> Completion:
        params, applied = self._params(model, messages, max_tokens, temperature, seed, effort, disabled)
        return self._completion(await self._async().chat.completions.create(**params), applied)

    # ------------------------------------------------------------ discovery

    def list_models(self) -> Dict[str, Dict]:
        """{model_id: {context_window, created, owned_by, active}} as the provider reports it."""
        page = self._sync().models.list()
        items = getattr(page, "data", None)
        if items is None:
            items = list(page)
        models: Dict[str, Dict] = {}
        for item in items:
            model_id = getattr(item, "id", None)
            if not model_id:
                continue
            if model_id.startswith("models/"):          # Gemini lists "models/<id>"
                model_id = model_id[len("models/"):]
            models[model_id] = {
                "context_window": (getattr(item, "context_window", None)
                                   or getattr(item, "context_length", None)),
                "created": getattr(item, "created", None),
                "owned_by": getattr(item, "owned_by", None),
                "active": getattr(item, "active", None),
            }
        return models

    # ------------------------------------------------------------ errors

    def classify(self, exc: Exception) -> ErrorInfo:
        status = getattr(exc, "status_code", None)
        try:
            import openai
            if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
                return ErrorInfo("network", str(exc))
        except ImportError:      # pragma: no cover - openai is a hard dependency
            pass
        info = classify_text(str(exc), status)
        if info.kind == "rate_limit":
            header = _retry_after_from_headers(exc)
            if header is not None:
                info.retry_after = header
        return info


class AnthropicBackend:
    """
    Claude through the official ``anthropic`` SDK.

    Differences from the OpenAI-style providers, all deliberate:
      * ``temperature`` and ``seed`` are never sent - current Claude models
        reject sampling parameters - so runs are not seed-reproducible, and the
        export says so (``applied.temperature`` is null).
      * Depth of reasoning is set with ``output_config.effort``.
      * ``max_tokens`` has a 16,000 floor: thinking counts against it, and a
        low cap would truncate answers mid-thought.
      * A request Claude declines (``stop_reason == "refusal"``) is reported as
        a refusal, never read as an answer. On the models that support it the
        server-side ``fallbacks`` beta is enabled so a declined request is
        re-run on another Claude model within the same call.
    """

    MIN_MAX_TOKENS = 16000
    FALLBACK_BETA = "server-side-fallback-2026-07-01"
    FALLBACK_MODELS = frozenset({
        "claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5",
        "claude-sonnet-5-5",
    })
    EFFORTS = ("low", "medium", "high", "xhigh", "max")

    def __init__(self, provider_id: str = "anthropic", api_key: str = "",
                 base_url: Optional[str] = None, client=None):
        self.provider_id = provider_id
        self.api_key = api_key
        self.base_url = base_url
        self._client = client
        self._async_client = None

    @staticmethod
    def _sdk():
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("The Anthropic provider needs the `anthropic` package: "
                           "pip install anthropic") from exc
        return anthropic

    def _client_kwargs(self) -> Dict:
        kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return kwargs

    def _sync(self):
        if self._client is None:
            self._client = self._sdk().Anthropic(**self._client_kwargs())
        return self._client

    def _async(self):
        if self._async_client is None:
            self._async_client = self._sdk().AsyncAnthropic(**self._client_kwargs())
        return self._async_client

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None

    def _request(self, model: str, messages: Sequence[Dict], max_tokens: int,
                 effort: Optional[str], disabled: set) -> Tuple[Dict, bool, Dict]:
        system, turns = _split_system(messages)
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max(max_tokens, self.MIN_MAX_TOKENS),
            "messages": turns,
        }
        if system:
            kwargs["system"] = system
        applied: Dict[str, Any] = {"temperature": None, "seed": None, "effort": None, "json_mode": False}
        if effort in self.EFFORTS and "effort" not in disabled:
            kwargs["output_config"] = {"effort": effort}
            applied["effort"] = effort
        use_fallbacks = model in self.FALLBACK_MODELS and "fallbacks" not in disabled
        applied["refusal_fallbacks"] = use_fallbacks
        return kwargs, use_fallbacks, applied

    def _completion(self, response, applied: Dict) -> Completion:
        usage = getattr(response, "usage", None)
        common = dict(
            prompt_tokens=getattr(usage, "input_tokens", None),
            completion_tokens=getattr(usage, "output_tokens", None),
            served_model=getattr(response, "model", None),
            applied=applied,
        )
        # Always check stop_reason before reading content.
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            return Completion(text=None, finish_reason="refusal",
                              refusal_category=getattr(details, "category", None), **common)
        text = "".join(block.text for block in response.content if block.type == "text")
        truncated = response.stop_reason in ("max_tokens", "model_context_window_exceeded")
        return Completion(text=text, finish_reason="length" if truncated else "stop", **common)

    def complete(self, model, messages, *, max_tokens, temperature, seed, effort, disabled) -> Completion:
        kwargs, use_fallbacks, applied = self._request(model, messages, max_tokens, effort, disabled)
        client = self._sync()
        if use_fallbacks:
            response = client.beta.messages.create(betas=[self.FALLBACK_BETA], fallbacks="default", **kwargs)
        else:
            response = client.messages.create(**kwargs)
        return self._completion(response, applied)

    async def acomplete(self, model, messages, *, max_tokens, temperature, seed, effort,
                        disabled) -> Completion:
        kwargs, use_fallbacks, applied = self._request(model, messages, max_tokens, effort, disabled)
        client = self._async()
        if use_fallbacks:
            response = await client.beta.messages.create(betas=[self.FALLBACK_BETA],
                                                         fallbacks="default", **kwargs)
        else:
            response = await client.messages.create(**kwargs)
        return self._completion(response, applied)

    def list_models(self) -> Dict[str, Dict]:
        models: Dict[str, Dict] = {}
        for item in self._sync().models.list():          # auto-paginates
            if getattr(item, "lifecycle", None) == "retired":
                continue
            models[item.id] = {
                "context_window": getattr(item, "max_input_tokens", None),
                "created": getattr(item, "created_at", None),
                "owned_by": "anthropic",
                "active": True,
            }
        return models

    def classify(self, exc: Exception) -> ErrorInfo:
        sdk = self._sdk()
        text = str(exc)
        lowered = text.lower()
        if isinstance(exc, sdk.NotFoundError):
            return ErrorInfo("model_not_found", text)
        if isinstance(exc, sdk.RateLimitError):
            return ErrorInfo("rate_limit", text, retry_after=_retry_after_from_headers(exc))
        if isinstance(exc, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
            return ErrorInfo("auth", text)
        if isinstance(exc, sdk.BadRequestError):
            if "credit balance" in lowered or "billing" in lowered:
                return ErrorInfo("quota", text)
            if "fallback" in lowered or "beta" in lowered:
                return ErrorInfo("bad_param", text, param="fallbacks")
            if "effort" in lowered or "output_config" in lowered:
                return ErrorInfo("bad_param", text, param="effort")
            return ErrorInfo("other", text)
        if isinstance(exc, (sdk.APIConnectionError, sdk.APITimeoutError)):
            return ErrorInfo("network", text)
        return ErrorInfo("other", text)


def make_backend(provider_id: str, api_key: str = "", base_url: Optional[str] = None):
    provider = PROVIDERS.get(provider_id)
    if provider is not None and provider.kind == "anthropic":
        return AnthropicBackend(provider_id, api_key, base_url)
    return OpenAICompatBackend(provider_id, api_key, base_url)


# ============================================================
# ROUTER
# ============================================================

class LLMRouter:
    """
    Maps a model reference to the backend that serves it.

    Holds the user's connections (provider -> key, endpoint, models). A bare
    model ID is routed to the first connected provider.
    """

    def __init__(self, connections: Dict[str, Connection], effort: Optional[str] = None):
        if not connections:
            raise LLMError("No LLM provider is connected.")
        self.connections = dict(connections)
        self.effort = effort if effort in EFFORT_LEVELS else None
        self._backends: Dict[str, Any] = {}

    @classmethod
    def single(cls, provider_id: str, api_key: str = "", base_url: Optional[str] = None,
               effort: Optional[str] = None) -> "LLMRouter":
        """A router with one provider and no model list - for scripts and tests."""
        return cls({provider_id: Connection(provider_id, api_key, base_url)}, effort=effort)

    @property
    def default_provider(self) -> str:
        return next(iter(self.connections))

    def resolve(self, ref: str) -> Tuple[str, str]:
        provider_id, model_id = split_ref(ref)
        if provider_id is None:
            provider_id = self.default_provider
        if provider_id not in self.connections:
            raise LLMError(f"No API key is connected for provider '{provider_id}' "
                           f"(needed for model {ref}).")
        return provider_id, model_id

    def new_backend(self, provider_id: str):
        connection = self.connections[provider_id]
        return make_backend(provider_id, connection.api_key, connection.base_url)

    def backend(self, provider_id: str):
        if provider_id not in self._backends:
            self._backends[provider_id] = self.new_backend(provider_id)
        return self._backends[provider_id]

    def effort_for(self, provider_id: str, model_id: str) -> Optional[str]:
        """Reasoning effort to request: the user's setting, else a per-provider default."""
        if self.effort:
            return self.effort
        family_default = REASONING_EFFORT_BY_FAMILY.get(model_family(make_ref(provider_id, model_id)))
        if family_default:
            return family_default
        provider = PROVIDERS.get(provider_id)
        return provider.default_effort if provider else None

    def models(self) -> List[str]:
        """Every connected model, as references, best default first within each provider."""
        return [ref for connection in self.connections.values() for ref in connection.refs()]

    def public(self) -> Dict[str, Dict]:
        """Connection details safe to export (no keys)."""
        return {pid: connection.public() for pid, connection in self.connections.items()}

    def async_pool(self) -> "AsyncPool":
        return AsyncPool(self)


class AsyncPool:
    """
    Async backends for ONE event loop.

    Async HTTP clients are bound to the loop they first run on, and each debate
    runs in its own loop, so they are created per run and closed with it.
    """

    def __init__(self, router: LLMRouter):
        self.router = router
        self._backends: Dict[str, Any] = {}

    def backend(self, provider_id: str):
        if provider_id not in self._backends:
            self._backends[provider_id] = self.router.new_backend(provider_id)
        return self._backends[provider_id]

    async def __aenter__(self) -> "AsyncPool":
        return self

    async def __aexit__(self, *exc_info) -> None:
        for backend in self._backends.values():
            await backend.aclose()
        self._backends.clear()


# ============================================================
# CONNECTING A PROVIDER
# ============================================================

def connect_provider(provider_id: str, api_key: str = "", base_url: Optional[str] = None) -> Connection:
    """
    Validate a key by asking the provider for its models, and return the
    connection with chat models ranked best-default-first.

    Raises the provider's own exception on a bad key or unreachable endpoint.
    A provider with no usable model list still connects; the user then types
    model IDs (``Connection.listed`` is False).
    """
    provider = PROVIDERS[provider_id]
    api_key = (api_key or "").strip()
    base_url = (base_url or "").strip() or provider.base_url
    if provider.needs_key and not api_key:
        raise LLMError(f"{provider.label} needs an API key.")
    if provider.kind == "openai" and not base_url:
        raise LLMError(f"{provider.label} needs a base URL (ending in /v1).")

    connection = Connection(provider_id, api_key, base_url)
    backend = make_backend(provider_id, api_key, base_url)
    raw: Dict[str, Dict] = {}
    if provider.lists_models:
        try:
            raw = backend.list_models()
        except LLMError:
            raise
        except Exception as exc:
            kind = backend.classify(exc).kind
            if kind in ("auth", "network", "quota", "rate_limit"):
                raise
            connection.listed = False
            connection.warning = (f"{provider.label} did not return a model list "
                                  f"({str(exc)[:120]}). Add model IDs manually.")
    else:
        connection.listed = False

    chat = {
        model_id: meta for model_id, meta in raw.items()
        if is_chat_model(model_id)
        and meta.get("active") is not False
        and (meta.get("context_window") is None or meta["context_window"] >= MIN_CONTEXT_WINDOW)
    }
    connection.models = {model_id: chat[model_id] for model_id in rank_models(provider_id, chat)}
    if not connection.models:
        connection.models = {model_id: {} for model_id in provider.suggested_models}
        if raw and not connection.warning:
            connection.warning = (f"{provider.label} listed {len(raw)} models but none is a "
                                  "usable chat model.")
    return connection


# ============================================================
# RETRY STATE MACHINE
# ============================================================

class _Attempt:
    """
    State machine shared by the sync and async drivers.

    Two kinds of recovery: adjust the request and retry the same model
    (drop a parameter, shrink or grow the token budget, wait out a per-minute
    limit), or give up on the model and move to the next fallback (retired,
    quota exhausted, refused, per-minute allowance too small for the job).
    """

    def __init__(self, model: str, fallbacks: Sequence[str], max_tokens: int, max_attempts: int):
        self.model = model
        self.fallbacks = [m for m in fallbacks if m and m != model]
        self.requested_tokens = min(max_tokens, MAX_TOKENS_CEILING)
        self.max_tokens = self.requested_tokens
        self.max_attempts = max_attempts
        self.disabled: set = set()
        self.notes: List[str] = []
        self.attempts = 0
        self.rate_limit_hits = 0
        self.json_failures = 0
        self.token_cap = MAX_TOKENS_CEILING     # lowered if the account or model allows less
        self.last_error: Optional[str] = None

    def _switch_model(self, reason: str, error: Exception) -> float:
        """Move to the next fallback model, or raise ``error`` if none is left."""
        if not self.fallbacks:
            raise error
        replacement = self.fallbacks.pop(0)
        self.notes.append(f"{self.model} {reason}; used {replacement}")
        self.model = replacement
        self.disabled.clear()
        self.max_tokens = self.requested_tokens
        self.token_cap = MAX_TOKENS_CEILING
        self.rate_limit_hits = 0
        self.json_failures = 0
        self.max_attempts += 2                  # a new model deserves fresh attempts
        return 0.0

    def _grow(self, why: str) -> bool:
        if self.max_tokens >= self.token_cap:
            return False
        grown = min(self.max_tokens * 2, self.token_cap)
        self.notes.append(f"{why} at {self.max_tokens} tokens; retried with {grown}")
        self.max_tokens = grown
        return True

    def on_error(self, info: ErrorInfo, exc: Exception) -> float:
        """Return seconds to sleep before retrying, or raise if unrecoverable."""
        self.last_error = info.message or str(exc)

        if info.kind == "model_not_found":
            return self._switch_model("is no longer available", exc)

        if info.kind == "bad_param" and info.param and info.param not in self.disabled:
            self.disabled.add(info.param)
            detail = ("retried with max_completion_tokens" if info.param == "max_tokens"
                      else "retried without it")
            self.notes.append(f"{self.model} rejected '{info.param}'; {detail}")
            return 0.0

        if info.kind == "json_failed":
            # Usually the answer was cut off mid-JSON by the token budget, so
            # give it room first; only then give up on JSON mode.
            self.json_failures += 1
            if self.json_failures == 1 and self._grow("JSON mode produced invalid output"):
                return 0.0
            if "response_format" not in self.disabled:
                self.disabled.add("response_format")
                self.notes.append(f"{self.model} rejected 'response_format'; retried without it")
                return 0.0

        if info.kind == "token_limit" and info.limit and info.limit < self.max_tokens:
            self.token_cap = self.max_tokens = info.limit
            self.notes.append(f"{self.model} allows at most {info.limit} output tokens; budget capped")
            return 0.0

        if info.kind == "quota":
            # Waiting does not help: the allowance refills over hours, or needs credit.
            return self._switch_model(
                "has used up its quota on this API key",
                LLMError(f"{self.model} has used up its daily quota or credit on this API key "
                         f"(rate limit, 429). Choose another model or provider, or try again "
                         f"later. Provider said: {self.last_error[:200]}"))

        if info.kind == "too_large":
            reduced = self.max_tokens - (info.overshoot or 0) - 50
            if reduced < 200:
                return self._switch_model(
                    "cannot take a request this large within its per-minute token limit",
                    LLMError(f"{self.model}: this request is larger than the account's "
                             "per-minute token limit for the model (rate limit, 429). "
                             "Use a smaller batch or another model."))
            self.token_cap = self.max_tokens = reduced
            self.notes.append(f"{self.model}: request exceeded the per-minute token limit; "
                              f"output budget capped at {reduced}")
            return 0.0

        if info.kind == "rate_limit":
            if self.rate_limit_hits < 4:
                self.rate_limit_hits += 1
                hint = info.retry_after
                wait = min(hint + 0.5, MAX_RATE_LIMIT_WAIT) if hint is not None \
                    else min(2.0 * (2 ** self.rate_limit_hits), MAX_RATE_LIMIT_WAIT)
                self.notes.append(f"rate limited; waited {wait:.0f}s")
                return wait
            return self._switch_model("stayed rate limited after 4 waits", exc)

        raise exc

    def on_completion(self, completion: Completion) -> Optional[LLMResult]:
        """Return a result, or None to retry (request already adjusted)."""
        if completion.finish_reason == "refusal":
            category = f" ({completion.refusal_category})" if completion.refusal_category else ""
            self.last_error = f"request declined by the model's safety filter{category}"
            self._switch_model(
                f"declined the request{category}",
                LLMError(f"{self.model} declined this request{category}. Rephrase the research "
                         "question or choose another model."))
            return None
        try:
            data = extract_json(completion.text)
        except ValueError as parse_error:
            self.last_error = f"{parse_error} (finish_reason={completion.finish_reason})"
            truncated = completion.finish_reason == "length"
            if truncated and self.max_tokens >= self.token_cap < MAX_TOKENS_CEILING:
                self._switch_model(
                    "cannot fit the answer inside its output limit",
                    LLMError(
                        f"{self.model} needs more than {self.token_cap} output tokens for this "
                        "request, but that is this account's per-minute output limit for the "
                        "model. Pick a different model or a smaller batch."))
            elif truncated and self._grow("output truncated"):
                pass
            else:
                self.notes.append(f"unparseable response ({parse_error}); retried")
            return None

        served = self.model
        reported = completion.served_model
        provider_id, requested = split_ref(self.model)
        if reported and completion.applied.get("refusal_fallbacks") and reported != requested:
            # Anthropic re-ran a declined request on another Claude model.
            served = make_ref(provider_id, reported) if provider_id else reported
            self.notes.append(f"{self.model} declined; answered by {served} (server-side fallback)")
        return LLMResult(
            data=data, model=served, attempts=self.attempts,
            finish_reason=completion.finish_reason,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            notes=self.notes, applied=completion.applied,
        )

    def exhausted(self) -> LLMError:
        return LLMError(
            f"{self.model} did not return valid JSON after {self.attempts} attempts "
            f"(last problem: {self.last_error})"
        )


# ============================================================
# PUBLIC API
# ============================================================

def _route(llm, ref: str, backend_for) -> Tuple[Any, str, Optional[str]]:
    """(backend, bare model id, effort) for a model reference."""
    provider_id, model_id = llm.resolve(ref)
    return backend_for(provider_id), model_id, llm.effort_for(provider_id, model_id)


def chat_json(
    llm,
    model: str,
    messages: List[Dict],
    *,
    max_tokens: int = 4000,
    temperature: Optional[float] = 0.0,
    seed: Optional[int] = 0,
    fallbacks: Sequence[str] = (),
    max_attempts: int = 6,
    client=None,
) -> LLMResult:
    """
    Chat completion that returns parsed JSON or raises.

    ``llm`` is an ``LLMRouter``. ``model`` and ``fallbacks`` are model
    references; fallbacks may live on other providers. ``client`` (an
    OpenAI-style client object) bypasses the router - used by tests.
    """
    preset = OpenAICompatBackend("groq", client=client) if client is not None else None
    state = _Attempt(model, fallbacks, max_tokens, max_attempts)
    while state.attempts < state.max_attempts:
        state.attempts += 1
        if preset is not None:
            backend, model_id = preset, state.model
            effort = REASONING_EFFORT_BY_FAMILY.get(model_family(model_id), "low")
        else:
            backend, model_id, effort = _route(llm, state.model, llm.backend)
        try:
            completion = backend.complete(
                model_id, messages, max_tokens=state.max_tokens, temperature=temperature,
                seed=seed, effort=effort, disabled=state.disabled)
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - classified by the backend
            delay = state.on_error(backend.classify(exc), exc)
            if delay:
                time.sleep(delay)
            continue
        result = state.on_completion(completion)
        if result is not None:
            return result
    raise state.exhausted()


async def achat_json(
    pool: AsyncPool,
    model: str,
    messages: List[Dict],
    *,
    max_tokens: int = 4000,
    temperature: Optional[float] = 0.0,
    seed: Optional[int] = 0,
    fallbacks: Sequence[str] = (),
    max_attempts: int = 6,
) -> LLMResult:
    """Async twin of ``chat_json``. ``pool`` comes from ``LLMRouter.async_pool()``."""
    state = _Attempt(model, fallbacks, max_tokens, max_attempts)
    while state.attempts < state.max_attempts:
        state.attempts += 1
        backend, model_id, effort = _route(pool.router, state.model, pool.backend)
        try:
            completion = await backend.acomplete(
                model_id, messages, max_tokens=state.max_tokens, temperature=temperature,
                seed=seed, effort=effort, disabled=state.disabled)
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            delay = state.on_error(backend.classify(exc), exc)
            if delay:
                await asyncio.sleep(delay)
            continue
        result = state.on_completion(completion)
        if result is not None:
            return result
    raise state.exhausted()

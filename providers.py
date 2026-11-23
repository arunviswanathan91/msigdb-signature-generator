"""
LLM provider registry
=====================

Every provider the app can talk to, in one table. Adding a provider that
speaks the OpenAI chat-completions protocol is one entry here - no other code
changes. Anthropic is the exception: it is reached through the official
``anthropic`` SDK (see ``llm_client.AnthropicBackend``), never through an
OpenAI-compatibility shim.

Nothing in this file asserts that a particular model exists. ``preference``
lists are ordered *patterns* used only to choose a default from whatever the
provider's own model list returns at run time; a pattern that matches nothing
is skipped. Users can always pick any listed model or type a model ID.

A model is referred to everywhere as ``"<provider id>:<model id>"``, e.g.
``groq:openai/gpt-oss-120b`` or ``anthropic:claude-opus-5-5``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    kind: str                         # "openai" (OpenAI-compatible) | "anthropic"
    base_url: Optional[str]
    env_keys: Tuple[str, ...]         # environment variables that may hold the key
    key_url: str                      # where a user gets a key
    needs_key: bool = True
    edit_base_url: bool = False       # user may / must supply the endpoint
    lists_models: bool = True         # has a usable /models endpoint
    default_effort: Optional[str] = "low"
    preference: Tuple[str, ...] = ()  # ordered model-id patterns for the default
    avoid: str = ""                   # regex: variants never chosen as a default
    suggested_models: Tuple[str, ...] = ()   # only for providers without /models
    note: str = ""


# Specialised variants that are never a sensible default for reasoning tasks.
_OPENAI_AVOID = (r"(mini|nano|-pro\b|codex|chat-latest|audio|realtime|search|transcribe|"
                 r"tts|image|instruct|deep-research|moderation|embedding)")
_GOOGLE_AVOID = r"(lite|image|tts|audio|live|embedding|aqa|nano|gemma|computer|robotics|learnlm)"
_MISTRAL_AVOID = r"(embed|moderation|ocr|voxtral|codestral|devstral|pixtral|saba)"

PROVIDERS: Dict[str, Provider] = {p.id: p for p in (
    Provider(
        id="groq", label="Groq", kind="openai",
        base_url="https://api.groq.com/openai/v1",
        env_keys=("GROQ_API_KEY",), key_url="https://console.groq.com/keys",
        preference=("openai/gpt-oss-120b", "qwen/qwen3*", "moonshotai/*", "minimaxai/*",
                    "meta-llama/llama-4*", "llama-3.3-70b*", "openai/gpt-oss-20b"),
        note="Free tier available, with tight per-minute and per-day token limits.",
    ),
    Provider(
        id="openai", label="OpenAI", kind="openai",
        base_url="https://api.openai.com/v1",
        env_keys=("OPENAI_API_KEY",), key_url="https://platform.openai.com/api-keys",
        preference=("gpt-5*", "gpt-4.1*", "gpt-4o*", "o4*", "o3*"), avoid=_OPENAI_AVOID,
    ),
    Provider(
        id="anthropic", label="Anthropic (Claude)", kind="anthropic",
        base_url=None,
        env_keys=("ANTHROPIC_API_KEY",), key_url="https://platform.claude.com/settings/keys",
        default_effort="medium",
        preference=("claude-opus-5-5", "claude-opus-*", "claude-sonnet-*", "claude-haiku-*"),
        note="Called through the official Anthropic SDK. Sampling temperature is fixed by "
             "the API on current Claude models, so runs are not seed-reproducible.",
    ),
    Provider(
        id="google", label="Google Gemini", kind="openai",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        env_keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"), key_url="https://aistudio.google.com/apikey",
        preference=("gemini-*-pro*", "gemini-*-flash*", "gemini-*"), avoid=_GOOGLE_AVOID,
    ),
    Provider(
        id="mistral", label="Mistral AI", kind="openai",
        base_url="https://api.mistral.ai/v1",
        env_keys=("MISTRAL_API_KEY",), key_url="https://console.mistral.ai/api-keys",
        default_effort=None,
        preference=("mistral-large*", "magistral-medium*", "mistral-medium*", "mistral-small*"),
        avoid=_MISTRAL_AVOID,
    ),
    Provider(
        id="deepseek", label="DeepSeek", kind="openai",
        base_url="https://api.deepseek.com",
        env_keys=("DEEPSEEK_API_KEY",), key_url="https://platform.deepseek.com/api_keys",
        default_effort=None,
        preference=("deepseek-chat", "deepseek-reasoner", "deepseek-*"),
    ),
    Provider(
        id="xai", label="xAI (Grok)", kind="openai",
        base_url="https://api.x.ai/v1",
        env_keys=("XAI_API_KEY",), key_url="https://console.x.ai",
        default_effort=None,
        preference=("grok-*",), avoid=r"(image|imagine|vision|mini|code)",
    ),
    Provider(
        id="openrouter", label="OpenRouter", kind="openai",
        base_url="https://openrouter.ai/api/v1",
        env_keys=("OPENROUTER_API_KEY",), key_url="https://openrouter.ai/keys",
        preference=("anthropic/claude-opus-*", "openai/gpt-5*", "google/gemini-*-pro*",
                    "anthropic/claude-sonnet-*", "deepseek/deepseek-chat*", "x-ai/grok-*",
                    "mistralai/mistral-large*", "meta-llama/*"),
        avoid=r"(:free|:extended|mini|nano|lite|image|audio|codex|search|embed|guard)",
        note="One key for hundreds of models from many vendors - the easiest way to give "
             "the debate three genuinely different model families.",
    ),
    Provider(
        id="together", label="Together AI", kind="openai",
        base_url="https://api.together.ai/v1",
        env_keys=("TOGETHER_API_KEY",), key_url="https://api.together.ai/settings/api-keys",
        preference=("openai/gpt-oss-120b", "deepseek-ai/*", "Qwen/*", "meta-llama/*", "moonshotai/*"),
        avoid=r"(embed|rerank|guard|vision|image|flux|whisper|tts|free)",
    ),
    Provider(
        id="fireworks", label="Fireworks AI", kind="openai",
        base_url="https://api.fireworks.ai/inference/v1",
        env_keys=("FIREWORKS_API_KEY",), key_url="https://fireworks.ai/account/api-keys",
        avoid=r"(embed|rerank|guard|vision|image|flux|whisper|tts)",
    ),
    Provider(
        id="cerebras", label="Cerebras", kind="openai",
        base_url="https://api.cerebras.ai/v1",
        env_keys=("CEREBRAS_API_KEY",), key_url="https://cloud.cerebras.ai",
        preference=("gpt-oss-120b", "qwen-3*", "llama-3.3-70b", "llama*"),
    ),
    Provider(
        id="cohere", label="Cohere", kind="openai",
        base_url="https://api.cohere.ai/compatibility/v1",
        env_keys=("COHERE_API_KEY", "CO_API_KEY"), key_url="https://dashboard.cohere.com/api-keys",
        default_effort=None,
        preference=("command-a*", "command-r-plus*", "command-r*", "command*"),
        avoid=r"(embed|rerank|vision|translate)",
    ),
    Provider(
        id="perplexity", label="Perplexity (Sonar)", kind="openai",
        base_url="https://api.perplexity.ai",
        env_keys=("PERPLEXITY_API_KEY", "PPLX_API_KEY"),
        key_url="https://www.perplexity.ai/account/api", lists_models=False,
        default_effort=None, suggested_models=("sonar-pro", "sonar"),
        note="Sonar models search the web while answering, so answers can change between "
             "runs. No model list is published: type the model ID.",
    ),
    Provider(
        id="huggingface", label="Hugging Face Inference Providers", kind="openai",
        base_url="https://router.huggingface.co/v1",
        env_keys=("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"),
        key_url="https://huggingface.co/settings/tokens",
        preference=("openai/gpt-oss-120b*", "deepseek-ai/*", "Qwen/*", "meta-llama/*"),
        avoid=r"(embed|rerank|guard|vision|image|whisper|tts)",
        note="Use a token with the 'Make calls to Inference Providers' permission.",
    ),
    Provider(
        id="nvidia", label="NVIDIA NIM", kind="openai",
        base_url="https://integrate.api.nvidia.com/v1",
        env_keys=("NVIDIA_API_KEY",), key_url="https://build.nvidia.com",
        avoid=r"(embed|rerank|guard|vision|image|whisper|tts|nv-|parakeet)",
    ),
    Provider(
        id="ollama", label="Ollama (local)", kind="openai",
        base_url="http://localhost:11434/v1",
        env_keys=("OLLAMA_API_KEY",), key_url="https://ollama.com/download",
        needs_key=False, edit_base_url=True, default_effort=None,
        note="Only works when this app runs on the same machine as Ollama (not on a "
             "hosted deployment).",
    ),
    Provider(
        id="custom", label="Custom (OpenAI-compatible)", kind="openai",
        base_url=None,
        env_keys=("CUSTOM_LLM_API_KEY",), key_url="",
        needs_key=False, edit_base_url=True, default_effort=None,
        note="Any endpoint that implements /chat/completions: Azure OpenAI (v1 endpoint), "
             "LM Studio, vLLM, LiteLLM, a self-hosted gateway. Give the base URL ending in /v1.",
    ),
)}


@dataclass
class Connection:
    """One provider the user has connected, with the models it reported."""
    provider_id: str
    api_key: str = ""
    base_url: Optional[str] = None
    # Chat models, best default first: {model_id: {"context_window", "created", "owned_by"}}
    models: Dict[str, Dict] = field(default_factory=dict)
    listed: bool = True          # False when the provider returned no model list
    warning: Optional[str] = None

    @property
    def provider(self) -> Provider:
        return PROVIDERS[self.provider_id]

    def refs(self):
        return [f"{self.provider_id}:{model_id}" for model_id in self.models]

    def public(self) -> Dict:
        """What may be written to an export: never the key."""
        return {
            "provider": self.provider.label,
            "base_url": self.base_url or self.provider.base_url or "official SDK default",
            "models_listed": len(self.models) if self.listed else "not published by provider",
        }


def key_from_environment(provider_id: str) -> Optional[str]:
    """First non-empty API key found in the provider's environment variables."""
    for name in PROVIDERS[provider_id].env_keys:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def providers_in_environment() -> Dict[str, str]:
    """{provider_id: key} for every provider whose key is set in the environment."""
    found = {}
    for provider_id in PROVIDERS:
        key = key_from_environment(provider_id)
        if key:
            found[provider_id] = key
    return found

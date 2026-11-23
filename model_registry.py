"""
Model registry - references, filters and default selection
==========================================================

WHY THIS MODULE EXISTS
----------------------
Providers retire models on a rolling basis. Every model ID this project ever
hardcoded has since been decommissioned:

    llama-3.3-70b-versatile     shutdown 2026-08-16
    llama-3.1-8b-instant        shutdown 2026-08-16
    gemma2-9b-it                shutdown 2025-10-08
    llama-3.1-70b-versatile     shutdown 2025-01-24
    llama-3.2-90b-text-preview  shutdown 2024-11-25

So nothing here asserts which models exist. The app ASKS each connected
provider at run time (``llm_client.connect_provider``), filters to chat models
and ranks them with the provider's preference *patterns* from ``providers.py``.
A pattern that matches nothing is skipped; if none match, whatever the provider
reports is still usable.

MODEL REFERENCES
----------------
A model is ``"<provider id>:<model id>"`` - ``groq:openai/gpt-oss-120b``,
``anthropic:claude-opus-5-5``, ``openrouter:google/gemini-2.5-pro``. A bare
model ID (no known provider prefix) is resolved against the first connected
provider, which keeps single-provider scripts simple.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from providers import PROVIDERS

# ============================================================
# SERVICE ENDPOINTS
# ============================================================

GROQ_BASE_URL = PROVIDERS["groq"].base_url

# Backend pathway API.
MSIGDB_API_URL = "https://arunviswanathan91-msigdb-api.hf.space"

# The Space page itself. Free HF Spaces sleep after inactivity; opening this
# page wakes the container back up (takes ~30-60s). Every "backend offline"
# message points users here so they can self-serve.
MSIGDB_SPACE_URL = "https://huggingface.co/spaces/arunviswanathan91/msigdb-api"

WAKE_BACKEND_HINT = (
    "The pathway database sleeps when unused. "
    "**[Click here to wake it up](" + MSIGDB_SPACE_URL + ")** — leave the page "
    "open for ~30-60 seconds until it says *Running*, then come back and retry."
)

DEBATE_ROLES = ("skeptic", "discoverer", "mediator")

# Preference patterns for Groq, kept under the historical name for scripts
# that pass bare Groq model IDs. Each provider's own list lives in providers.py.
REASONING_PREFERENCE = list(PROVIDERS["groq"].preference)

# The pipeline sends a full signature plus an evidence table in one prompt.
# Models with a smaller window (e.g. allam-2-7b at 4,096) cannot hold it and
# are never offered.
MIN_CONTEXT_WINDOW = 16000


# ============================================================
# REFERENCES
# ============================================================

def split_ref(ref: str) -> Tuple[Optional[str], str]:
    """``"groq:openai/gpt-oss-20b"`` -> ``("groq", "openai/gpt-oss-20b")``.

    A string without a known provider prefix is returned as ``(None, ref)``;
    model IDs may themselves contain colons (``llama3:8b``, ``...:free``).
    """
    head, sep, tail = ref.partition(":")
    if sep and head in PROVIDERS and tail:
        return head, tail
    return None, ref


def make_ref(provider_id: str, model_id: str) -> str:
    return f"{provider_id}:{model_id}"


def bare_model(ref: str) -> str:
    return split_ref(ref)[1]


# ============================================================
# CHAT-MODEL FILTER
# ============================================================

# /models endpoints also list speech, embedding, image and moderation models.
_NON_CHAT = re.compile(
    r"(whisper|tts|text-to-speech|speech|transcri|embed|rerank|moderation|guard|"
    r"dall-e|image|imagen|veo|sora|flux|stable-diffusion|audio|realtime|orpheus|"
    r"playai|parakeet|ocr\b|aqa\b)", re.IGNORECASE)


def is_chat_model(model_id: str) -> bool:
    """True if this model ID can plausibly serve chat completions."""
    return not _NON_CHAT.search(model_id)


# ============================================================
# MODEL FAMILIES (for debate diversity)
# ============================================================

# Namespace -> the organisation that trained the model.
_VENDOR_ALIASES = {
    "meta-llama": "meta", "meta": "meta", "mistralai": "mistral", "mistral": "mistral",
    "deepseek-ai": "deepseek", "deepseek": "deepseek", "x-ai": "xai", "xai": "xai",
    "moonshotai": "moonshot", "minimaxai": "minimax", "alibaba": "qwen", "qwen": "qwen",
    "google": "google", "openai": "openai", "anthropic": "anthropic", "cohere": "cohere",
    "perplexity": "perplexity", "microsoft": "microsoft", "amazon": "amazon",
    "nvidia": "nvidia", "zai-org": "zhipu", "z-ai": "zhipu",
}

# For IDs without a namespace ("gpt-5", "claude-opus-5-5", "llama-3.3-70b").
_NAME_FAMILIES = (
    ("chatgpt", "openai"), ("gpt", "openai"), ("o1", "openai"), ("o3", "openai"), ("o4", "openai"),
    ("claude", "anthropic"), ("gemini", "google"), ("gemma", "google"),
    ("llama", "meta"), ("mistral", "mistral"), ("mixtral", "mistral"), ("magistral", "mistral"),
    ("ministral", "mistral"), ("codestral", "mistral"), ("pixtral", "mistral"),
    ("qwen", "qwen"), ("qwq", "qwen"), ("deepseek", "deepseek"), ("grok", "xai"),
    ("command", "cohere"), ("sonar", "perplexity"), ("kimi", "moonshot"), ("phi", "microsoft"),
    ("nova", "amazon"), ("glm", "zhipu"), ("minimax", "minimax"), ("allam", "sdaia"),
)

# Providers that only serve their own models.
_NATIVE_VENDORS = {"openai", "anthropic", "google", "mistral", "deepseek", "xai", "cohere", "perplexity"}


def model_family(ref: str) -> str:
    """
    The organisation that trained a model, whichever provider serves it:
    ``groq:openai/gpt-oss-20b`` and ``openai:gpt-5`` are both ``openai``.
    Two debaters from one family share training data and blind spots.
    """
    provider_id, name = split_ref(ref)
    name = name.lower()
    if "/" in name:
        vendor = name.split("/", 1)[0]
        if vendor in ("accounts", "models"):          # e.g. accounts/fireworks/models/<name>
            name = name.rsplit("/", 1)[1]
        else:
            return _VENDOR_ALIASES.get(vendor, vendor)
    for prefix, family in _NAME_FAMILIES:
        if name.startswith(prefix):
            return family
    if provider_id in _NATIVE_VENDORS:
        return provider_id
    return re.split(r"[-_.:]", name)[0]


# ============================================================
# RANKING AND DEFAULTS
# ============================================================

def _size_hint(model_id: str) -> float:
    """Parameter count in billions parsed from the ID, 0 if absent."""
    match = re.search(r"(\d+(?:\.\d+)?)b\b", model_id.lower())
    return float(match.group(1)) if match else 0.0


def _version_key(model_id: str) -> Tuple[float, ...]:
    """Version numbers in an ID, ignoring date stamps: 'gpt-5.2-2026-03-01' -> (5.0, 2.0)."""
    undated = re.sub(r"\d{4}-\d{2}-\d{2}|\d{6,8}", "", bare_model(model_id))
    undated = re.sub(r"\d+(?:\.\d+)?b\b", "", undated, flags=re.IGNORECASE)   # not a version
    return tuple(float(n) for n in re.findall(r"\d+", undated))


@lru_cache(maxsize=512)
def _pattern(pattern: str):
    return re.compile(".*".join(re.escape(part) for part in pattern.lower().split("*")) + r"\Z")


def _matches(pattern: str, model: str) -> bool:
    """Exact ID or '*' wildcard pattern, tested against the full ref and the bare ID."""
    compiled = _pattern(pattern)
    return bool(compiled.match(model.lower()) or compiled.match(bare_model(model).lower()))


def _timestamp(value) -> float:
    if value is None:
        return 0.0
    if hasattr(value, "timestamp"):
        return float(value.timestamp())
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def rank_models(provider_id: str, models: Dict[str, Dict]) -> List[str]:
    """
    Order a provider's chat models best-default-first.

    Sort keys, in order: not a specialised variant (the provider's ``avoid``
    pattern); position of the first matching preference pattern; newest
    release; largest parameter count; highest version number; shortest ID (an
    alias before its dated snapshot); the ID itself, for determinism.
    """
    provider = PROVIDERS[provider_id]
    avoid = re.compile(provider.avoid, re.IGNORECASE) if provider.avoid else None
    unmatched = len(provider.preference)

    def key(model_id: str):
        meta = models.get(model_id) or {}
        position = next((i for i, pattern in enumerate(provider.preference)
                         if _matches(pattern, model_id)), unmatched)
        return (
            bool(avoid and avoid.search(model_id)),
            position,
            -_timestamp(meta.get("created")),
            -_size_hint(model_id),
            tuple(-v for v in _version_key(model_id)),
            len(model_id),
            model_id,
        )

    return sorted(models, key=key)


def pick_default(available: Sequence[str], preference: Optional[Sequence[str]] = None) -> Optional[str]:
    """
    The default model from ``available``.

    Model lists built by ``connect_provider`` are already ranked, so with no
    ``preference`` the first entry is the answer. With an explicit preference
    list, the first pattern that matches a live model wins, and if none match
    the largest live model is used - the app keeps working even when every
    preferred ID has been retired.
    """
    if not available:
        return None
    if preference is None:
        return available[0]
    for pattern in preference:
        hits = sorted((m for m in available if _matches(pattern, m)),
                      key=lambda m: (-_size_hint(m), m))
        if hits:
            return hits[0]
    return max(available, key=lambda m: (_size_hint(m), m))


def resolve_debate_models(available: Sequence[str]) -> Dict[str, str]:
    """
    Assign models to the three debate roles, maximising family diversity.

    Pass 1 walks the (ranked) list and gives each role the best model from a
    family no other role is using - across every connected provider, so
    OpenAI + Anthropic + Google keys yield three unrelated debaters. Pass 2
    fills any remaining role with an unused model. Only if fewer than three
    models are live do roles share one; ``debate_diversity`` reports it.
    """
    resolved: Dict[str, str] = {}
    used_models: set = set()
    used_families: set = set()

    def take(role: str, model: str) -> None:
        resolved[role] = model
        used_models.add(bare_model(model))
        used_families.add(model_family(model))

    for role in DEBATE_ROLES:
        hit = next((m for m in available if bare_model(m) not in used_models
                    and model_family(m) not in used_families), None)
        if hit:
            take(role, hit)

    for role in DEBATE_ROLES:
        if role in resolved:
            continue
        spare = next((m for m in available if bare_model(m) not in used_models), None)
        if spare:
            take(role, spare)
        elif available:
            resolved[role] = available[0]

    return resolved


def debate_diversity(models: Dict[str, str]) -> Dict:
    """How independent the three debaters actually are."""
    ids = list(models.values())
    return {
        "distinct_models": len({bare_model(m) for m in ids}),
        "distinct_families": len({model_family(m) for m in ids}),
        "roles": len(ids),
    }


def fallback_chain(current: str, available: Sequence[str],
                   preference: Optional[Sequence[str]] = None) -> List[str]:
    """Ordered replacements to try if ``current`` is retired or out of quota."""
    chain: List[str] = []
    for pattern in preference or ():
        for hit in sorted((m for m in available if _matches(pattern, m)),
                          key=lambda m: (-_size_hint(m), m)):
            if hit != current and hit not in chain:
                chain.append(hit)
    rest = available if preference is None else sorted(available, key=lambda m: (-_size_hint(m), m))
    for model in rest:
        if model != current and model not in chain:
            chain.append(model)
    return chain


def next_fallback(current: str, available: Sequence[str],
                  preference: Optional[Sequence[str]] = None) -> Optional[str]:
    chain = fallback_chain(current, available, preference)
    return chain[0] if chain else None


def describe_model(ref: str) -> str:
    """Short human label for a model, for captions and spinners."""
    if not ref:
        return "Unknown model"
    provider_id, model_id = split_ref(ref)
    vendor, _, name = model_id.partition("/")
    if not name:
        vendor, name = "", model_id

    pretty = name.replace("-", " ").strip()
    pretty = re.sub(r"\b(\d+)b\b", lambda m: m.group(1) + "B", pretty, flags=re.IGNORECASE)
    labels = {"openai": "OpenAI", "qwen": "Qwen", "minimaxai": "MiniMax", "meta-llama": "Meta",
              "groq": "Groq", "anthropic": "Anthropic", "google": "Google", "mistralai": "Mistral",
              "deepseek": "DeepSeek", "deepseek-ai": "DeepSeek", "x-ai": "xAI", "moonshotai": "Moonshot"}
    label = labels.get(vendor.lower())
    text = (label + " " + pretty).strip() if label else pretty
    return f"{PROVIDERS[provider_id].label} · {text}" if provider_id else text


# ============================================================
# ERROR EXPLANATION
# ============================================================

def is_model_not_found(exc: Exception) -> bool:
    """True if this exception means the requested model no longer exists."""
    text = str(exc).lower()
    return "model_not_found" in text or "does not exist" in text or "decommissioned" in text


def explain_llm_error(exc: Exception, model_id: str = None) -> str:
    """Turn a raw provider exception into something a user can act on."""
    text = str(exc)
    lowered = text.lower()
    where = " (`" + model_id + "`)" if model_id else ""

    if is_model_not_found(exc):
        return (
            "**Model unavailable**" + where + "\n\n"
            "The provider no longer serves this model. Open **Providers & model** in the "
            "sidebar, click **Refresh model lists**, then pick a different model."
        )

    if "daily quota" in lowered or "per day" in lowered or "(tpd)" in lowered or "(rpd)" in lowered \
            or "insufficient_quota" in lowered or "credit balance" in lowered:
        return (
            "**Quota used up**" + where + "\n\n"
            "This API key has spent its allowance for the model (a free tier's daily limit, "
            "or the account is out of credit). Pick a different model or provider in the "
            "sidebar to continue now.\n\n`" + text[:300] + "`"
        )

    if "401" in text or "invalid api key" in lowered or "invalid x-api-key" in lowered \
            or "authentication" in lowered or "unauthorized" in lowered or "incorrect api key" in lowered:
        return (
            "**Invalid API key**\n\n"
            "The provider rejected the key. Check that it is complete, belongs to the "
            "selected provider and has not been revoked."
        )

    if "429" in text or "rate limit" in lowered or "rate_limit" in lowered:
        return (
            "**Rate limited**\n\n"
            "The provider is throttling this key. Wait a moment and retry, or switch to "
            "another model or provider in the sidebar."
        )

    if "timeout" in lowered or "timed out" in lowered:
        return (
            "**Request timed out**\n\n"
            "The model took too long to respond. Retry, or pick a faster model."
        )

    if "connection" in lowered or "resolve" in lowered or "network" in lowered:
        return (
            "**Network error**\n\n"
            "Could not reach the provider. Check the endpoint and your connection.\n\n`" + text[:300] + "`"
        )

    return "**LLM call failed**" + where + "\n\n`" + text[:500] + "`"

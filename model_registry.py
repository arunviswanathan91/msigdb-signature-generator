"""
Model Registry - Single Source of Truth for LLM Model Selection
================================================================

WHY THIS MODULE EXISTS
----------------------
Groq retires models on a rolling basis. Every hardcoded model ID in this
repo has been decommissioned at least once:

    llama-3.3-70b-versatile     shutdown 2026-08-16
    llama-3.1-8b-instant        shutdown 2026-08-16
    gemma2-9b-it                shutdown 2025-10-08
    llama-3.1-70b-versatile     shutdown 2025-01-24
    llama-3.2-90b-text-preview  shutdown 2024-11-25

Hand-maintaining a model list does not work: a previous "fix" swapped in two
IDs that had already been dead for over a year. So this module never asserts
which models exist. It ASKS the provider at runtime via /models and picks the
best live option from an ordered preference list.

If Groq retires everything named below, the app still works - it falls back to
whatever the API actually reports.

USAGE
-----
    from model_registry import fetch_available_models, pick_default, REASONING_PREFERENCE

    models = fetch_available_models(api_key)
    model  = pick_default(models, REASONING_PREFERENCE)
"""

from typing import Dict, List, Optional
import re

# ============================================================
# PROVIDER CONFIG
# ============================================================

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Backend pathway API (centralised - was duplicated in 4 places in app.py)
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


# ============================================================
# PREFERENCE LISTS
# ============================================================
# These are PREFERENCES, not requirements. The first entry that is actually
# live wins; if none are live we fall back to the first available chat model.
# A retired ID here is harmless - it is simply skipped.

REASONING_PREFERENCE = [
    "openai/gpt-oss-120b",     # Groq migration target for llama-3.3-70b-versatile
    "qwen/qwen3.6-27b",
    "minimaxai/minimax-m2.7",
    "openai/gpt-oss-20b",
]

FAST_PREFERENCE = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-120b",
    "minimaxai/minimax-m2.7",
]

# The debate system wants three DISTINCT models for adversarial diversity.
# Preferences are ordered so the roles naturally land on different families.
DEBATE_PREFERENCE: Dict[str, List[str]] = {
    "skeptic":    ["openai/gpt-oss-120b", "qwen/qwen3.6-27b", "openai/gpt-oss-20b"],
    "discoverer": ["qwen/qwen3.6-27b", "minimaxai/minimax-m2.7", "openai/gpt-oss-120b"],
    "mediator":   ["openai/gpt-oss-20b", "minimaxai/minimax-m2.7", "qwen/qwen3.6-27b"],
}

# Models the /models endpoint returns that cannot serve chat completions.
NON_CHAT_PREFIXES = (
    "whisper-",                        # speech-to-text
    "canopylabs/",                     # text-to-speech (orpheus)
    "meta-llama/llama-prompt-guard",   # prompt-injection classifier
    "openai/gpt-oss-safeguard",        # moderation classifier
    "playai-tts",                      # text-to-speech
    "distil-whisper",                  # speech-to-text
)


# ============================================================
# DISCOVERY
# ============================================================

def is_chat_model(model_id: str) -> bool:
    """True if this model ID can plausibly serve chat completions."""
    lowered = model_id.lower()
    return not any(lowered.startswith(p) for p in NON_CHAT_PREFIXES)


def fetch_available_models(api_key: str, base_url: str = GROQ_BASE_URL) -> List[str]:
    """
    Ask the provider which models actually exist right now.

    Args:
        api_key: Groq API key
        base_url: OpenAI-compatible endpoint

    Returns:
        Sorted list of chat-capable model IDs.

    Raises:
        Exception: propagated from the provider so the caller can distinguish
                   a bad key (401) from a network failure.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.models.list()

    ids = [m.id for m in response.data if is_chat_model(m.id)]
    return sorted(ids)


def pick_default(available: List[str], preference: List[str]) -> Optional[str]:
    """
    Return the first preferred model that is actually available.

    Falls back to the first available model so the app keeps working even if
    every preferred ID has been retired.
    """
    if not available:
        return None

    for candidate in preference:
        if candidate in available:
            return candidate

    return available[0]


def resolve_debate_models(available: List[str]) -> Dict[str, str]:
    """
    Assign three distinct models to the debate roles.

    Tries to give each role a different model so the adversarial setup is
    genuinely adversarial. If fewer than three chat models are live, roles
    reuse models rather than failing - a degraded debate beats no debate.
    """
    resolved: Dict[str, str] = {}
    used: set = set()

    # First pass: give each role its top *unused* preference.
    for role, preference in DEBATE_PREFERENCE.items():
        for candidate in preference:
            if candidate in available and candidate not in used:
                resolved[role] = candidate
                used.add(candidate)
                break

    # Second pass: any role still unfilled takes an unused model, then any model.
    for role in DEBATE_PREFERENCE:
        if role in resolved:
            continue
        spare = next((m for m in available if m not in used), None)
        if spare:
            resolved[role] = spare
            used.add(spare)
        else:
            resolved[role] = pick_default(available, DEBATE_PREFERENCE[role])

    return resolved


def next_fallback(current: str, available: List[str], preference: List[str]) -> Optional[str]:
    """
    Pick a replacement when `current` turns out to be dead mid-flight.

    Used for the single automatic retry on model_not_found.
    """
    for candidate in preference:
        if candidate != current and candidate in available:
            return candidate

    return next((m for m in available if m != current), None)


def describe_model(model_id: str) -> str:
    """Short human label for a model ID, for chat bubbles and captions."""
    if not model_id:
        return "Unknown model"

    vendor, _, name = model_id.partition("/")
    if not name:
        vendor, name = "", model_id

    pretty = name.replace("-", " ").strip()
    pretty = re.sub(r"\b(\d+)b\b", lambda m: m.group(1) + "B", pretty, flags=re.I)

    vendor_labels = {
        "openai": "OpenAI",
        "qwen": "Qwen",
        "minimaxai": "MiniMax",
        "meta-llama": "Meta",
        "groq": "Groq",
    }
    label = vendor_labels.get(vendor.lower())

    return (label + " " + pretty).strip() if label else pretty


# ============================================================
# ERROR EXPLANATION
# ============================================================

def is_model_not_found(exc: Exception) -> bool:
    """True if this exception means the requested model no longer exists."""
    text = str(exc).lower()
    return "model_not_found" in text or "does not exist" in text


def explain_llm_error(exc: Exception, model_id: str = None) -> str:
    """
    Turn a raw provider exception into something a user can act on.

    Replaces the previous behaviour of dumping a raw 404 into st.error, and of
    reporting every failure - timeouts included - as "Invalid API key".
    """
    text = str(exc)
    lowered = text.lower()
    where = " (`" + model_id + "`)" if model_id else ""

    if is_model_not_found(exc):
        return (
            "**Model unavailable**" + where + "\n\n"
            "Groq retires models on a rolling basis and this one is gone. "
            "Open **API & Models** in the sidebar and click **Refresh model list**, "
            "then pick a different model."
        )

    if "401" in text or "invalid api key" in lowered or "authentication" in lowered:
        return (
            "**Invalid API key**\n\n"
            "Check the key in the sidebar. Free keys are at console.groq.com/keys."
        )

    if "429" in text or "rate limit" in lowered:
        return (
            "**Rate limited**\n\n"
            "The Groq free tier throttles requests. Wait a moment and retry, or "
            "switch to a smaller model in the sidebar."
        )

    if "timeout" in lowered or "timed out" in lowered:
        return (
            "**Request timed out**\n\n"
            "The model took too long to respond. Retry, or pick a smaller model."
        )

    if "connection" in lowered or "resolve" in lowered or "network" in lowered:
        return (
            "**Network error**\n\n"
            "Could not reach the provider. Check your connection.\n\n`" + text + "`"
        )

    return "**LLM call failed**" + where + "\n\n`" + text + "`"

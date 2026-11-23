from model_registry import (
    REASONING_PREFERENCE, bare_model, debate_diversity, describe_model, explain_llm_error,
    fallback_chain, is_chat_model, is_model_not_found, make_ref, model_family, pick_default,
    rank_models, resolve_debate_models, split_ref,
)

LIVE = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]


# ----------------------------------------------------------------- references

def test_model_references_round_trip():
    assert split_ref("groq:openai/gpt-oss-20b") == ("groq", "openai/gpt-oss-20b")
    assert split_ref("anthropic:claude-opus-5-5") == ("anthropic", "claude-opus-5-5")
    # A colon that is part of the model ID is not a provider separator.
    assert split_ref("llama3:8b") == (None, "llama3:8b")
    assert split_ref("ollama:llama3:8b") == ("ollama", "llama3:8b")
    assert split_ref("openrouter:meta-llama/llama-3.3-70b-instruct:free") == (
        "openrouter", "meta-llama/llama-3.3-70b-instruct:free")
    assert make_ref("groq", "x/y") == "groq:x/y" and bare_model("groq:x/y") == "x/y"
    assert bare_model("x/y") == "x/y"


def test_non_chat_models_are_filtered():
    for model_id in ["openai/gpt-oss-120b", "gpt-5", "claude-opus-5-5", "gemini-2.5-pro",
                     "meta-llama/llama-3.2-90b-vision-preview", "deepseek-chat"]:
        assert is_chat_model(model_id), model_id
    for model_id in ["whisper-large-v3", "canopylabs/orpheus-v1-english",
                     "meta-llama/llama-prompt-guard-2-86m", "openai/gpt-oss-safeguard-20b",
                     "text-embedding-3-large", "gpt-image-1", "tts-1", "gpt-4o-realtime-preview",
                     "omni-moderation-latest", "gemini-embedding-001", "rerank-v3.5", "dall-e-3"]:
        assert not is_chat_model(model_id), model_id


# ------------------------------------------------------------------- families

def test_family_is_the_organisation_that_trained_the_model_whoever_serves_it():
    assert model_family("openai/gpt-oss-20b") == "openai"
    assert model_family("groq:openai/gpt-oss-120b") == model_family("openai:gpt-5") == "openai"
    assert model_family("anthropic:claude-opus-5-5") == model_family("openrouter:anthropic/claude-sonnet-5-5")
    assert model_family("google:gemini-2.5-pro") == model_family("openrouter:google/gemini-2.5-pro") == "google"
    assert model_family("groq:llama-3.3-70b-versatile") == model_family("together:meta-llama/Llama-4-Scout") == "meta"
    assert model_family("fireworks:accounts/fireworks/models/deepseek-v3") == model_family("deepseek:deepseek-chat")
    assert model_family("mistral:mistral-large-latest") == "mistral"
    assert model_family("xai:grok-4") == "xai" and model_family("cohere:command-a-03-2025") == "cohere"
    assert model_family("custom:my-finetune-v2") == "my"            # unknown: first token


# -------------------------------------------------------------------- ranking

def test_ranking_prefers_listed_patterns_then_newest_then_aliases():
    groq = {m: {} for m in ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b", "x/other-9b"]}
    assert rank_models("groq", groq) == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b",
                                         "openai/gpt-oss-20b", "x/other-9b"]

    openai = {"gpt-5": {"created": 100}, "gpt-5.2": {"created": 300}, "gpt-5.2-2026-03-01": {"created": 300},
              "gpt-5.2-mini": {"created": 310}, "gpt-5.2-pro": {"created": 320}, "gpt-4o": {"created": 50},
              "gpt-5.2-codex": {"created": 330}}
    ranked = rank_models("openai", openai)
    assert ranked[:3] == ["gpt-5.2", "gpt-5.2-2026-03-01", "gpt-5"]      # newest; alias before snapshot
    assert ranked[3] == "gpt-4o"
    assert set(ranked[4:]) == {"gpt-5.2-mini", "gpt-5.2-pro", "gpt-5.2-codex"}   # variants last

    claude = {m: {} for m in ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-opus-5-5",
                              "claude-fable-5-2"]}
    ranked = rank_models("anthropic", claude)
    assert ranked[:3] == ["claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"]

    gemini = {m: {} for m in ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-pro"]}
    assert rank_models("google", gemini)[:2] == ["gemini-3-pro", "gemini-2.5-pro"]

    # A provider with no preference list still gets a sensible, deterministic order.
    assert rank_models("fireworks", {"a/small-8b": {}, "a/big-70b": {}})[0] == "a/big-70b"


def test_pick_default_prefers_listed_models_then_patterns_then_largest():
    assert pick_default(LIVE, REASONING_PREFERENCE) == "openai/gpt-oss-120b"
    # A version bump must not demote the family: 'qwen/qwen3*' still matches.
    assert pick_default(["qwen/qwen3.8-27b", "x/tiny-3b"], ["qwen/qwen3.6-27b", "qwen/qwen3*"]) == "qwen/qwen3.8-27b"
    # Every preference retired -> largest live model, not alphabetical first.
    assert pick_default(["a/small-7b", "z/large-70b"], ["gone/model"]) == "z/large-70b"
    assert pick_default([], REASONING_PREFERENCE) is None
    # Lists from connect_provider are already ranked: with no preference, take the first.
    assert pick_default(["anthropic:claude-opus-5-5", "groq:openai/gpt-oss-120b"]) == "anthropic:claude-opus-5-5"
    assert pick_default(["groq:qwen/qwen3.8-27b"], ["qwen/qwen3*"]) == "groq:qwen/qwen3.8-27b"


# --------------------------------------------------------------- debate roles

def test_debate_roles_get_distinct_models_and_maximal_family_spread():
    roles = resolve_debate_models(LIVE)
    assert roles == {"skeptic": "openai/gpt-oss-120b", "discoverer": "qwen/qwen3.8-27b",
                     "mediator": "openai/gpt-oss-20b"}
    assert debate_diversity(roles) == {"distinct_models": 3, "distinct_families": 2, "roles": 3}

    three_vendors = resolve_debate_models(LIVE + ["meta-llama/llama-4-scout-17b"])
    assert debate_diversity(three_vendors)["distinct_families"] == 3
    assert len(set(three_vendors.values())) == 3


def test_debate_roles_span_providers_when_several_are_connected():
    available = ["openai:gpt-5.2", "openai:gpt-5", "anthropic:claude-opus-5-5",
                 "anthropic:claude-sonnet-5-5", "google:gemini-2.5-pro", "groq:openai/gpt-oss-120b"]
    roles = resolve_debate_models(available)
    assert roles == {"skeptic": "openai:gpt-5.2", "discoverer": "anthropic:claude-opus-5-5",
                     "mediator": "google:gemini-2.5-pro"}
    assert debate_diversity(roles) == {"distinct_models": 3, "distinct_families": 3, "roles": 3}
    # The same model reached through two providers is one model, not two voices.
    same = {"skeptic": "groq:openai/gpt-oss-120b", "discoverer": "together:openai/gpt-oss-120b",
            "mediator": "cerebras:gpt-oss-120b"}
    assert debate_diversity(same)["distinct_families"] == 1
    assert debate_diversity({"a": "groq:openai/gpt-oss-120b", "b": "together:openai/gpt-oss-120b"})[
        "distinct_models"] == 1


def test_debate_roles_degrade_gracefully():
    assert resolve_debate_models(["a/only-8b"]) == {
        "skeptic": "a/only-8b", "discoverer": "a/only-8b", "mediator": "a/only-8b"}
    assert resolve_debate_models([]) == {}
    two = resolve_debate_models(["openai/gpt-oss-120b", "qwen/qwen3.8-27b"])
    assert set(two) == {"skeptic", "discoverer", "mediator"}
    assert debate_diversity(two)["distinct_models"] == 2


def test_fallback_chain_excludes_the_dead_model_and_orders_by_preference():
    assert fallback_chain("openai/gpt-oss-120b", LIVE, REASONING_PREFERENCE) == [
        "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
    assert fallback_chain("dead/model", LIVE, REASONING_PREFERENCE)[0] == "openai/gpt-oss-120b"
    assert fallback_chain("only/model", ["only/model"], REASONING_PREFERENCE) == []
    # Without a preference list the (already ranked) order is kept, across providers.
    assert fallback_chain("groq:a", ["groq:a", "groq:b", "openai:c"]) == ["groq:b", "openai:c"]


def test_helpers():
    assert describe_model("openai/gpt-oss-120b") == "OpenAI gpt oss 120B"
    assert describe_model("groq:openai/gpt-oss-120b") == "Groq · OpenAI gpt oss 120B"
    assert describe_model("anthropic:claude-opus-5-5") == "Anthropic (Claude) · claude opus 5 5"
    assert is_model_not_found(Exception("404 model_not_found"))
    assert "Invalid API key" in explain_llm_error(Exception("Error code: 401"))
    assert "Rate limited" in explain_llm_error(Exception("Error code: 429 rate limit"))
    assert "Quota used up" in explain_llm_error(
        Exception("429 Rate limit reached on tokens per day (TPD): Limit 200000"), "m")
    assert "Quota used up" in explain_llm_error(Exception("400 Your credit balance is too low"))
    assert "Model unavailable" in explain_llm_error(Exception("The model `x` does not exist"), "x")

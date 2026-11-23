import types

import pytest

import llm_client
from llm_client import (
    AnthropicBackend, LLMError, LLMRouter, OpenAICompatBackend, chat_json, classify_text,
    connect_provider, extract_json,
)
from providers import PROVIDERS, Connection


def response(content, finish="stop"):
    message = types.SimpleNamespace(content=content)
    usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message, finish_reason=finish)],
                                 usage=usage)


class FakeClient:
    """Plays back a script of responses / exceptions and records every request."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **params):
        self.requests.append(params)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    waits = []
    monkeypatch.setattr(llm_client.time, "sleep", waits.append)
    return waits


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": [1, 2]}\n```\nDone.') == {"a": [1, 2]}
    assert extract_json('<think>maybe {"x": 0}</think>{"a": 2} trailing') == {"a": 2}
    assert extract_json('[1, 2, 3]') == [1, 2, 3]
    for bad in ["", None, "no json here", '{"a": [1, 2']:      # the last one is truncated
        with pytest.raises(ValueError):
            extract_json(bad)


def test_happy_path_sends_json_mode_low_reasoning_and_seed():
    client = FakeClient([response('{"ok": true}')])
    result = chat_json("k", "openai/gpt-oss-120b", MESSAGES, max_tokens=300, client=client)
    assert result.data == {"ok": True} and result.attempts == 1 and result.notes == []
    sent = client.requests[0]
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["reasoning_effort"] == "low" and sent["seed"] == 0 and sent["temperature"] == 0.0


def test_qwen_family_has_hidden_reasoning_switched_off():
    client = FakeClient([response('{"ok": 1}')])
    chat_json("k", "qwen/qwen3.8-27b", MESSAGES, client=client)
    assert client.requests[0]["reasoning_effort"] == "none"


def test_truncated_output_is_retried_with_a_larger_budget():
    client = FakeClient([response('{"a": [1, 2', finish="length"), response('{"a": [1, 2]}')])
    result = chat_json("k", "m", MESSAGES, max_tokens=500, client=client)
    assert result.data == {"a": [1, 2]} and result.attempts == 2
    assert [r["max_tokens"] for r in client.requests] == [500, 1000]
    assert "truncated at 500" in result.notes[0]


def test_empty_content_from_a_reasoning_model_is_retried():
    client = FakeClient([response(None, finish="length"), response('{"a": 1}')])
    assert chat_json("k", "m", MESSAGES, max_tokens=400, client=client).data == {"a": 1}


def test_unsupported_parameter_is_dropped_and_retried():
    client = FakeClient([
        Exception("Error code: 400 - {'error': {'message': '`reasoning_effort` is not supported with this model'}}"),
        response('{"ok": 1}'),
    ])
    result = chat_json("k", "m", MESSAGES, client=client)
    assert result.data == {"ok": 1}
    assert "reasoning_effort" in client.requests[0] and "reasoning_effort" not in client.requests[1]
    assert result.applied["effort"] is None            # provenance shows it was not applied
    assert "rejected 'reasoning_effort'" in result.notes[0]


def test_openai_reasoning_models_get_max_completion_tokens_and_no_temperature():
    client = FakeClient([
        Exception("Error code: 400 - Unsupported parameter: 'max_tokens' is not supported with this "
                  "model. Use 'max_completion_tokens' instead."),
        Exception("Error code: 400 - Unsupported value: 'temperature' does not support 0.0 with this "
                  "model. Only the default (1) value is supported."),
        response('{"ok": 1}'),
    ])
    result = chat_json("k", "gpt-5", MESSAGES, max_tokens=700, client=client)
    final = client.requests[2]
    assert final["max_completion_tokens"] == 700 and "max_tokens" not in final
    assert "temperature" not in final
    assert result.applied["temperature"] is None       # not deterministic, and recorded as such


def test_json_mode_failure_first_gets_more_room_then_drops_json_mode():
    failed = Exception("Error code: 400 - {'error': {'code': 'json_validate_failed', "
                       "'message': 'Failed to generate JSON'}}")
    client = FakeClient([failed, failed, response('{"ok": 1}')])
    result = chat_json("k", "m", MESSAGES, max_tokens=600, client=client)
    assert [r["max_tokens"] for r in client.requests] == [600, 1200, 1200]
    assert "response_format" in client.requests[1]     # truncation is the usual cause
    assert "response_format" not in client.requests[2]
    assert result.data == {"ok": 1}


def test_a_models_hard_output_limit_caps_the_budget():
    client = FakeClient([
        Exception("Error code: 400 - max_tokens is too large: 9000. This model supports at most "
                  "4096 completion tokens"),
        response('{"ok": 1}'),
    ])
    result = chat_json("k", "m", MESSAGES, max_tokens=9000, client=client)
    assert client.requests[1]["max_tokens"] == 4096
    assert "at most 4096" in result.notes[0]


def test_truncated_json_is_never_parsed_as_an_inner_fragment():
    cut_off = '{"keep": {"A": 0.9}, "remove": {"B": {"confidence": 0.8, "reason": "x"}, "C": {"conf'
    with pytest.raises(ValueError, match="truncated"):
        extract_json(cut_off)
    # ...whereas prose containing a brace before the real JSON is still handled.
    assert extract_json('Use {curly} braces. Answer: {"a": 1}') == {"a": 1}
    client = FakeClient([response(cut_off, finish="length"), response('{"keep": {"A": 0.9}}')])
    assert chat_json("k", "m", MESSAGES, max_tokens=300, client=client).data == {"keep": {"A": 0.9}}


def test_content_filter_refusal_moves_to_another_model():
    client = FakeClient([response(None, finish="content_filter"), response('{"ok": 1}')])
    result = chat_json("k", "strict/model", MESSAGES, fallbacks=["other/model"], client=client)
    assert result.model == "other/model" and "declined the request" in result.notes[0]
    with pytest.raises(LLMError, match="declined this request"):
        chat_json("k", "strict/model", MESSAGES, client=FakeClient([response(None, finish="content_filter")]))


def test_retired_model_falls_back_and_reports_it():
    client = FakeClient([
        Exception("Error code: 404 - {'error': {'message': 'The model `dead/model` does not exist', 'code': 'model_not_found'}}"),
        response('{"ok": 1}'),
    ])
    result = chat_json("k", "dead/model", MESSAGES, fallbacks=["dead/model", "live/model"], client=client)
    assert result.model == "live/model"
    assert client.requests[1]["model"] == "live/model"
    assert "no longer available" in result.notes[0]


def test_retired_model_without_fallback_raises_the_original_error():
    client = FakeClient([Exception("model_not_found: gone")])
    with pytest.raises(Exception, match="model_not_found"):
        chat_json("k", "dead/model", MESSAGES, client=client)


def test_rate_limit_waits_for_the_providers_hint(no_sleep):
    client = FakeClient([
        Exception("Error code: 429 - Rate limit reached ... on output tokens per minute (OTPM): "
                  "Limit 1000, Used 396, Requested 900. Please try again in 17.7s."),
        response('{"ok": 1}'),
    ])
    result = chat_json("k", "m", MESSAGES, max_tokens=900, client=client)
    assert result.data == {"ok": 1}
    assert no_sleep == [pytest.approx(18.2)]
    assert client.requests[1]["max_tokens"] == 900            # the request itself was fine


def test_request_larger_than_the_allowance_is_shrunk_not_waited_on(no_sleep):
    client = FakeClient([
        Exception("Error code: 429 - Request too large ... on output tokens per minute (OTPM): "
                  "Limit 1000, Requested 1920."),
        response('{"ok": 1}'),
    ])
    result = chat_json("k", "m", MESSAGES, max_tokens=1920, client=client)
    assert client.requests[1]["max_tokens"] == 1920 - 920 - 50
    assert no_sleep == []
    assert "capped" in result.notes[0]


def test_truncation_at_the_account_cap_fails_with_an_actionable_message():
    client = FakeClient([
        Exception("429 - output tokens per minute (OTPM): Limit 1000, Requested 1920."),
        response('{"a": [', finish="length"),
    ])
    with pytest.raises(LLMError, match="per-minute output limit"):
        chat_json("k", "m", MESSAGES, max_tokens=1920, client=client)


def test_gives_up_after_max_attempts():
    client = FakeClient([response("not json")] * 3)
    with pytest.raises(LLMError, match="after 3 attempts"):
        chat_json("k", "m", MESSAGES, max_attempts=3, client=client)


def test_other_errors_propagate_untouched():
    client = FakeClient([Exception("Error code: 401 - Invalid API Key")])
    with pytest.raises(Exception, match="401"):
        chat_json("k", "m", MESSAGES, client=client)
    assert len(client.requests) == 1


DAILY = ("Error code: 429 - {'error': {'message': 'Rate limit reached for model `big/model` ... on tokens "
         "per day (TPD): Limit 200000, Used 199969, Requested 97. Please try again in 28.512s.'}}")
PER_MINUTE = ("Error code: 429 - Rate limit reached ... on tokens per minute (TPM): "
              "Limit 8000, Used 7900, Requested 900. Please try again in 6s.")


def test_daily_quota_switches_model_immediately_instead_of_waiting(no_sleep):
    client = FakeClient([Exception(DAILY), response('{"ok": 1}')])
    result = chat_json("k", "big/model", MESSAGES, fallbacks=["small/model"], client=client)
    assert result.model == "small/model"
    assert no_sleep == []                                   # no pointless 28 s wait
    assert "used up its quota" in result.notes[0] and "small/model" in result.notes[0]


def test_daily_quota_without_fallback_is_an_actionable_error(no_sleep):
    client = FakeClient([Exception(DAILY)])
    with pytest.raises(LLMError, match="daily quota"):
        chat_json("k", "big/model", MESSAGES, client=client)
    assert no_sleep == [] and len(client.requests) == 1


def test_persistent_per_minute_limit_eventually_switches_model(no_sleep):
    client = FakeClient([Exception(PER_MINUTE)] * 5 + [response('{"ok": 1}')])
    result = chat_json("k", "busy/model", MESSAGES, fallbacks=["idle/model"], max_attempts=6,
                       client=client)
    assert result.model == "idle/model"
    assert len(no_sleep) == 4                               # waited four times, then moved on
    assert [r["model"] for r in client.requests] == ["busy/model"] * 5 + ["idle/model"]


def test_model_that_cannot_fit_the_answer_hands_over_with_the_original_budget():
    client = FakeClient([
        Exception("429 - output tokens per minute (OTPM): Limit 1000, Requested 1920."),
        response('{"a": [', finish="length"),
        response('{"a": [1]}'),
    ])
    result = chat_json("k", "tight/model", MESSAGES, max_tokens=1920,
                       fallbacks=["roomy/model"], client=client)
    assert result.data == {"a": [1]} and result.model == "roomy/model"
    assert client.requests[1]["max_tokens"] == 950          # shrunk for the tight model
    assert client.requests[2]["max_tokens"] == 1920         # restored for the roomy one
    assert any("cannot fit the answer" in note for note in result.notes)


# ============================================================
# ROUTER
# ============================================================

def test_router_resolves_references_and_picks_backends():
    router = LLMRouter({
        "groq": Connection("groq", "g-key", models={"openai/gpt-oss-120b": {}}),
        "anthropic": Connection("anthropic", "a-key", models={"claude-opus-5-5": {}}),
        "ollama": Connection("ollama", "", "http://localhost:11434/v1", models={"llama3:8b": {}}),
    })
    assert router.resolve("anthropic:claude-opus-5-5") == ("anthropic", "claude-opus-5-5")
    assert router.resolve("ollama:llama3:8b") == ("ollama", "llama3:8b")       # colons in the ID survive
    assert router.resolve("openai/gpt-oss-120b") == ("groq", "openai/gpt-oss-120b")   # bare -> first provider
    with pytest.raises(LLMError, match="No API key is connected for provider 'openai'"):
        router.resolve("openai:gpt-5")
    assert isinstance(router.backend("groq"), OpenAICompatBackend)
    assert isinstance(router.backend("anthropic"), AnthropicBackend)
    assert router.backend("ollama").base_url == "http://localhost:11434/v1"
    assert router.models() == ["groq:openai/gpt-oss-120b", "anthropic:claude-opus-5-5", "ollama:llama3:8b"]
    assert "g-key" not in str(router.public()) and "a-key" not in str(router.public())


def test_reasoning_effort_policy():
    router = LLMRouter.single("groq", "k")
    assert router.effort_for("groq", "openai/gpt-oss-120b") == "low"
    assert router.effort_for("groq", "qwen/qwen3.8-27b") == "none"      # qwen ignores "low"
    anthropic = LLMRouter.single("anthropic", "k")
    assert anthropic.effort_for("anthropic", "claude-opus-5-5") == "medium"
    assert LLMRouter.single("mistral", "k").effort_for("mistral", "mistral-large-latest") is None
    assert LLMRouter.single("groq", "k", effort="high").effort_for("groq", "qwen/qwen3.8-27b") == "high"
    assert LLMRouter.single("groq", "k", effort="bogus").effort is None
    with pytest.raises(LLMError, match="No LLM provider"):
        LLMRouter({})


def test_fallback_can_cross_providers(monkeypatch):
    served = []

    class Backend:
        def __init__(self, name, error=None):
            self.name, self.error = name, error

        def complete(self, model, messages, **kwargs):
            served.append((self.name, model))
            if self.error:
                raise self.error
            return llm_client.Completion(text='{"ok": 1}', applied={"temperature": 0.0})

        def classify(self, exc):
            return classify_text(str(exc))

    router = LLMRouter({"groq": Connection("groq", "k"), "openai": Connection("openai", "k")})
    backends = {"groq": Backend("groq", Exception("429 ... tokens per day (TPD): Limit 200000")),
                "openai": Backend("openai")}
    monkeypatch.setattr(router, "backend", lambda provider_id: backends[provider_id])
    result = chat_json(router, "groq:openai/gpt-oss-120b", MESSAGES, fallbacks=["openai:gpt-5"])
    assert served == [("groq", "openai/gpt-oss-120b"), ("openai", "gpt-5")]
    assert result.model == "openai:gpt-5"
    assert "used up its quota" in result.notes[0]


@pytest.mark.parametrize("text, status, kind", [
    ("The model `x` does not exist or you do not have access to it", 404, "model_not_found"),
    ("model_decommissioned: this model has been decommissioned", 400, "model_not_found"),
    ("Rate limit reached ... tokens per day (TPD): Limit 200000, Used 199969", 429, "quota"),
    ("You exceeded your current quota, please check your plan and billing details", 429, "quota"),
    ("Rate limit reached ... (OTPM): Limit 1000, Requested 1920.", 429, "too_large"),
    ("Rate limit reached. Please try again in 7.5s.", 429, "rate_limit"),
    # Real Groq per-minute message: its upgrade link contains "billing" and must not read as quota.
    ("Rate limit reached for model `qwen/qwen3.8-27b` ... on output tokens per minute (OTPM): Limit 1000, "
     "Used 396, Requested 900. Please try again in 17.7s. Need more tokens? Upgrade to Dev Tier today at "
     "https://console.groq.com/settings/billing", 429, "rate_limit"),
    ("json_validate_failed: Failed to generate JSON", 400, "json_failed"),
    ("Unrecognized request argument supplied: seed", 400, "bad_param"),
    ("Incorrect API key provided", 401, "auth"),
    ("upstream connect error", 502, "other"),
])
def test_classify_text(text, status, kind):
    assert classify_text(text, status).kind == kind


# ============================================================
# ANTHROPIC BACKEND (official SDK, faked transport)
# ============================================================

def claude_response(text='{"ok": 1}', stop_reason="end_turn", model="claude-opus-5-5", category=None):
    return types.SimpleNamespace(
        content=[types.SimpleNamespace(type="thinking", thinking=""),
                 types.SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason, model=model,
        stop_details=types.SimpleNamespace(category=category) if category else None,
        usage=types.SimpleNamespace(input_tokens=120, output_tokens=40))


class FakeAnthropic:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.messages = types.SimpleNamespace(create=lambda **kw: self._create("messages", kw))
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(
            create=lambda **kw: self._create("beta.messages", kw)))

    def _create(self, endpoint, kwargs):
        self.calls.append((endpoint, kwargs))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


CLAUDE_MESSAGES = [{"role": "system", "content": "Be terse."}, {"role": "system", "content": "JSON only."},
                   {"role": "user", "content": "context"}, {"role": "user", "content": "question"}]
CALL = dict(max_tokens=700, temperature=0.0, seed=0, effort="medium", disabled=set())


def test_anthropic_request_follows_the_messages_api():
    client = FakeAnthropic([claude_response()])
    backend = AnthropicBackend(client=client)
    completion = backend.complete("claude-opus-5-5", CLAUDE_MESSAGES, **CALL)

    endpoint, sent = client.calls[0]
    assert endpoint == "beta.messages"                       # refusal fallbacks enabled on this model
    assert sent["betas"] == ["server-side-fallback-2026-07-01"] and sent["fallbacks"] == "default"
    assert sent["system"] == "Be terse.\n\nJSON only."        # system is top-level, not a message
    assert [m["role"] for m in sent["messages"]] == ["user", "user"]
    assert sent["output_config"] == {"effort": "medium"}
    assert sent["max_tokens"] == 16000                       # thinking counts against it: never lowball
    for forbidden in ("temperature", "seed", "top_p", "response_format", "thinking"):
        assert forbidden not in sent                         # rejected by current Claude models
    assert completion.text == '{"ok": 1}' and completion.finish_reason == "stop"
    assert (completion.prompt_tokens, completion.completion_tokens) == (120, 40)
    assert completion.applied == {"temperature": None, "seed": None, "effort": "medium",
                                  "json_mode": False, "refusal_fallbacks": True}


def test_anthropic_models_outside_the_fallback_set_use_the_plain_endpoint():
    client = FakeAnthropic([claude_response(model="claude-haiku-5-5")])
    AnthropicBackend(client=client).complete("claude-haiku-5-5", CLAUDE_MESSAGES, **CALL)
    endpoint, sent = client.calls[0]
    assert endpoint == "messages" and "fallbacks" not in sent and "betas" not in sent


def test_anthropic_refusal_is_reported_not_read_as_an_answer():
    client = FakeAnthropic([claude_response(text="", stop_reason="refusal", category="bio")])
    completion = AnthropicBackend(client=client).complete("claude-opus-5-5", CLAUDE_MESSAGES, **CALL)
    assert completion.finish_reason == "refusal" and completion.refusal_category == "bio"
    assert completion.text is None


def test_anthropic_truncation_and_dropped_parameters():
    client = FakeAnthropic([claude_response(text='{"a": [', stop_reason="max_tokens"), claude_response()])
    backend = AnthropicBackend(client=client)
    assert backend.complete("claude-opus-5-5", CLAUDE_MESSAGES, **CALL).finish_reason == "length"
    backend.complete("claude-opus-5-5", CLAUDE_MESSAGES,
                     **{**CALL, "disabled": {"effort", "fallbacks"}, "effort": "max"})
    endpoint, sent = client.calls[1]
    assert endpoint == "messages" and "output_config" not in sent and "fallbacks" not in sent


def test_anthropic_errors_are_classified_from_typed_exceptions():
    import anthropic
    import httpx2

    def error(cls, status, message):
        request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        return cls(message, response=httpx2.Response(status, request=request, headers={"retry-after": "12"}),
                   body=None)

    backend = AnthropicBackend(api_key="k")
    assert backend.classify(error(anthropic.NotFoundError, 404, "model: nope")).kind == "model_not_found"
    limited = backend.classify(error(anthropic.RateLimitError, 429, "slow down"))
    assert limited.kind == "rate_limit" and limited.retry_after == 12.0
    assert backend.classify(error(anthropic.AuthenticationError, 401, "invalid x-api-key")).kind == "auth"
    assert backend.classify(error(anthropic.BadRequestError, 400,
                                  "Your credit balance is too low")).kind == "quota"
    effort = backend.classify(error(anthropic.BadRequestError, 400, "output_config.effort: not supported"))
    assert effort.kind == "bad_param" and effort.param == "effort"
    fallbacks = backend.classify(error(anthropic.BadRequestError, 400, "fallbacks: unknown beta"))
    assert fallbacks.kind == "bad_param" and fallbacks.param == "fallbacks"


def test_refusal_on_claude_hands_over_to_another_provider(monkeypatch):
    claude = AnthropicBackend(client=FakeAnthropic(
        [claude_response(text="", stop_reason="refusal", category="bio")]))
    other = OpenAICompatBackend("openai", client=FakeClient([response('{"ok": 1}')]))
    router = LLMRouter({"anthropic": Connection("anthropic", "k"), "openai": Connection("openai", "k")})
    monkeypatch.setattr(router, "backend", lambda pid: {"anthropic": claude, "openai": other}[pid])
    result = chat_json(router, "anthropic:claude-opus-5-5", MESSAGES, fallbacks=["openai:gpt-5"])
    assert result.model == "openai:gpt-5"
    assert "declined the request (bio)" in result.notes[0]


def test_server_side_fallback_is_recorded_with_the_model_that_answered(monkeypatch):
    claude = AnthropicBackend(client=FakeAnthropic([claude_response(model="claude-opus-4-8")]))
    router = LLMRouter.single("anthropic", "k")
    monkeypatch.setattr(router, "backend", lambda pid: claude)
    result = chat_json(router, "anthropic:claude-opus-5-5", MESSAGES)
    assert result.model == "anthropic:claude-opus-4-8"
    assert "server-side fallback" in result.notes[0]


# ============================================================
# CONNECTING PROVIDERS
# ============================================================

def model_item(model_id, **extra):
    return types.SimpleNamespace(id=model_id, **extra)


def fake_openai_models(items):
    client = FakeClient([])
    client.models = types.SimpleNamespace(list=lambda: types.SimpleNamespace(data=items))
    return client


def test_connect_filters_and_ranks_what_the_provider_reports(monkeypatch):
    items = [
        model_item("whisper-large-v3", context_window=448),
        model_item("openai/gpt-oss-20b", context_window=131072),
        model_item("allam-2-7b", context_window=4096),                 # too small to hold a prompt
        model_item("qwen/qwen3.8-27b", context_window=131072),
        model_item("openai/gpt-oss-safeguard-20b", context_window=131072),
        model_item("openai/gpt-oss-120b", context_window=131072),
        model_item("retired/model-99b", context_window=131072, active=False),
    ]
    monkeypatch.setattr(llm_client, "make_backend",
                        lambda pid, key, url: OpenAICompatBackend(pid, key, url, client=fake_openai_models(items)))
    connection = connect_provider("groq", " gsk_test ")
    assert list(connection.models) == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
    assert connection.refs()[0] == "groq:openai/gpt-oss-120b"
    assert connection.api_key == "gsk_test" and connection.listed
    assert "gsk_test" not in str(connection.public())


def test_connect_strips_the_gemini_models_prefix(monkeypatch):
    items = [model_item("models/gemini-2.5-flash"), model_item("models/gemini-2.5-pro"),
             model_item("models/gemini-embedding-001"), model_item("models/gemini-2.5-flash-lite")]
    monkeypatch.setattr(llm_client, "make_backend",
                        lambda pid, key, url: OpenAICompatBackend(pid, key, url, client=fake_openai_models(items)))
    connection = connect_provider("google", "key")
    assert list(connection.models) == ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]


def test_connect_requirements_and_providers_without_a_model_list(monkeypatch):
    with pytest.raises(LLMError, match="needs an API key"):
        connect_provider("openai", "")
    with pytest.raises(LLMError, match="needs a base URL"):
        connect_provider("custom", "")
    perplexity = connect_provider("perplexity", "pplx-key")             # no /models: nothing is fetched
    assert not perplexity.listed and list(perplexity.models) == ["sonar-pro", "sonar"]

    class NoListing(OpenAICompatBackend):
        def list_models(self):
            raise Exception("Error code: 404 - page not found")

    monkeypatch.setattr(llm_client, "make_backend", lambda pid, key, url: NoListing(pid, key, url))
    custom = connect_provider("custom", "", "http://localhost:8000/v1")
    assert not custom.listed and custom.models == {} and "Add model IDs manually" in custom.warning

    class BadKey(OpenAICompatBackend):
        def list_models(self):
            raise Exception("Error code: 401 - Incorrect API key provided")

    monkeypatch.setattr(llm_client, "make_backend", lambda pid, key, url: BadKey(pid, key, url))
    with pytest.raises(Exception, match="Incorrect API key"):
        connect_provider("openai", "sk-wrong")


def test_every_provider_entry_is_usable():
    assert {"groq", "openai", "anthropic", "google", "mistral", "deepseek", "xai", "openrouter",
            "together", "fireworks", "cerebras", "cohere", "perplexity", "huggingface", "nvidia",
            "ollama", "custom"} <= set(PROVIDERS)
    for provider in PROVIDERS.values():
        assert provider.kind in ("openai", "anthropic")
        assert provider.env_keys and provider.label
        if provider.kind == "openai" and not provider.edit_base_url:
            assert provider.base_url.startswith("https://")
        if not provider.lists_models:
            assert provider.suggested_models
    assert PROVIDERS["anthropic"].base_url is None          # official SDK, not an OpenAI shim

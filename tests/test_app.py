"""
End-to-end tests of the Streamlit app, driven headlessly with every external
service mocked. They use the real knowledge base in data/, so they also guard
the data files against drifting out of sync with the code.
"""

import json
import os
import types

import numpy as np
import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

import db_client
import debate_engine
import llm_client
import ncbi_client
import pipeline
from llm_client import LLMResult
from providers import PROVIDERS, Connection

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]     # ranked, as connect_provider returns
REFS = [f"groq:{m}" for m in MODELS]

FACETS = {"facets": [
    {"facet_name": "Inhibitory checkpoint receptors",
     "description": "Co-inhibitory receptors on chronically stimulated CD8 T cells.",
     "seed_genes": ["PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "CD160", "CD244", "B7H3", "TWOWAY", "NOPE9"]},
    {"facet_name": "Type I interferon signaling",
     "description": "Sustained interferon alpha/beta signalling and interferon-stimulated genes.",
     "seed_genes": ["IFNAR1", "STAT1", "IRF7", "ISG15", "OAS1", "MX1", "RSAD2", "IFIT1"]},
]}

# TWOWAY stands for a name two unrelated genes share; nothing in a checkpoint
# mechanism points at keratin 5 or albumin, so it must stay ambiguous.
NCBI = {"B7H3": ["CD276"], "TWOWAY": ["KRT5", "ALB"]}


class FakeEncoder:
    """Stands in for SentenceTransformer so tests need no model download."""

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, text, normalize_embeddings=True):
        vector = np.zeros(384, dtype=np.float32)
        vector[0] = 1.0
        return vector


def fake_api(self, method, endpoint, **kwargs):
    if endpoint.startswith("/api/gene-neighbors/"):
        # Both clear the default cut-off (strength 0.5 -> link probability >= 0.03).
        return [{"target": "NETGENE1", "probability": 0.045, "rank": 1},
                {"target": "NETGENE2", "probability": 0.035, "rank": 2},
                {"target": "TOOWEAK1", "probability": 0.012, "rank": 3}]
    if endpoint == "/api/evidence/batch":
        return {"results": {g: {"pathway_count": 50, "evidence_strength": "strong"} for g in kwargs["json"]}}
    if endpoint == "/api/expression/batch":
        return {"results": {g: {"Blood/PBMC": 12.0} for g in kwargs["json"]}}
    raise AssertionError(f"unexpected endpoint {endpoint}")


def fake_ncbi_lookup(self, names, organism="Homo sapiens"):
    return {name: [{"symbol": s, "match": "alias"} for s in NCBI.get(name, [])] for name in names}


def signature_genes(messages):
    return messages[1]["content"].split(" genes): ")[1].split("\n")[0].split(", ")


async def fake_debater(pool, model, messages, **kwargs):
    """All agents vote out the third-from-last gene; the skeptic also contests gene 3."""
    genes = signature_genes(messages)
    if len(messages) > 2:
        contested = [line for line in messages[2]["content"].splitlines() if line in genes]
        return LLMResult(data={"votes": [{"gene": g, "vote": "keep", "confidence": 0.8, "reason": "ok"}
                                         for g in contested]}, model=model)
    remove = {genes[-3]: {"confidence": 0.9, "reason": "unrelated"}}
    if "SKEPTIC" in messages[0]["content"]:
        remove[genes[2]] = {"confidence": 0.7, "reason": "weak"}
    return LLMResult(data={"keep": {g: 0.9 for g in genes if g not in remove}, "remove": remove},
                     model=model)


def fake_reviewer(llm, model, messages, **kwargs):
    """Flags the second gene of every signature and proposes one real, one fake addition."""
    out = {}
    for i, block in enumerate(messages[-1]["content"].split("\nS")[1:], start=1):
        genes = block.split("): ")[-1].split("\n")[0].split(", ")
        out[f"S{i}"] = {"remove": {genes[1]: "generic hub"},
                        "add": {"TOX": "exhaustion TF", "NOTREAL99": "x"}}
    return LLMResult(data=out, model=model)


def groq_connection(models=MODELS):
    return Connection("groq", "test-key", PROVIDERS["groq"].base_url, models={m: {} for m in models})


@pytest.fixture
def at(monkeypatch):
    import sentence_transformers

    for provider in PROVIDERS.values():                     # no key may leak in from the environment
        for name in provider.env_keys:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeEncoder)
    monkeypatch.setattr(requests, "get", lambda *a, **k: types.SimpleNamespace(status_code=200))
    monkeypatch.setattr(db_client.DatabaseClient, "_json", fake_api)
    monkeypatch.setattr(ncbi_client.NCBIGene, "lookup", fake_ncbi_lookup)
    monkeypatch.setattr(pipeline, "chat_json",
                        lambda llm, model, messages, **kw: LLMResult(data=FACETS, model=model))
    monkeypatch.setattr(debate_engine, "achat_json", fake_debater)
    st.cache_data.clear()
    st.cache_resource.clear()

    app = AppTest.from_file(APP, default_timeout=300)
    app.session_state["connections"] = {"groq": groq_connection()}
    return app


def click(app, label):
    matches = [b for b in app.button if b.label == label]
    assert matches, f"no button {label!r}; available: {[b.label for b in app.button]}"
    return matches[0].click().run()


def problems(app):
    return [str(e.value) for e in app.exception] + [e.value for e in app.error]


def build_signatures(app):
    app.run()
    app.text_area(key="main_query").set_value("T cell exhaustion in the tumor microenvironment").run()
    click(app, "Generate mechanisms")
    click(app, "Approve selected & continue")
    click(app, "Build signatures")
    assert problems(app) == []
    return app.session_state["signatures"]


# ----------------------------------------------------------------- start-up

def test_app_boots_without_any_provider(monkeypatch):
    for provider in PROVIDERS.values():
        for name in provider.env_keys:
            monkeypatch.delenv(name, raising=False)
    st.cache_data.clear()
    app = AppTest.from_file(APP, default_timeout=300).run()
    assert not app.exception
    assert any("Connect an LLM provider" in w.value for w in app.warning)
    # Every registered provider is offered.
    picker = app.selectbox(key="provider_choice")
    assert set(picker.options) == {p.label for p in PROVIDERS.values()}
    assert {"OpenAI", "Anthropic (Claude)", "Google Gemini", "DeepSeek", "Groq"} <= set(picker.options)


def test_a_key_in_the_environment_connects_its_provider_automatically(monkeypatch):
    seen = []

    def fake_connect(provider_id, api_key="", base_url=None):
        seen.append((provider_id, api_key))
        return Connection(provider_id, api_key, models={"deepseek-chat": {}})

    for provider in PROVIDERS.values():
        for name in provider.env_keys:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
    monkeypatch.setattr(llm_client, "connect_provider", fake_connect)
    monkeypatch.setattr(requests, "get", lambda *a, **k: types.SimpleNamespace(status_code=200))
    st.cache_data.clear()
    app = AppTest.from_file(APP, default_timeout=300).run()
    assert seen == [("deepseek", "sk-env")]
    assert list(app.session_state["connections"]) == ["deepseek"]
    app.run()
    assert len(seen) == 1                                   # once per session, not on every rerun


# ---------------------------------------------------------------- providers

def test_connecting_several_providers_gives_the_debate_independent_models(at, monkeypatch):
    catalogue = {
        "openai": ["gpt-5.2", "gpt-5"],
        "anthropic": ["claude-opus-5-5", "claude-sonnet-5-5"],
        "deepseek": ["deepseek-chat"],
    }
    calls = []

    def fake_connect(provider_id, api_key="", base_url=None):
        calls.append((provider_id, api_key))
        if api_key == "bad":
            raise Exception("Error code: 401 - Incorrect API key provided")
        return Connection(provider_id, api_key, models={m: {} for m in catalogue[provider_id]})

    monkeypatch.setattr(llm_client, "connect_provider", fake_connect)
    at.session_state["connections"] = {}
    at.run()

    def connect(label, provider_id, key):
        at.selectbox(key="provider_choice").set_value(label).run()
        at.text_input(key=f"key_{provider_id}").set_value(key).run()
        click(at, "Connect")

    connect("OpenAI", "openai", "bad")
    assert at.session_state["connections"] == {}
    assert any("Invalid API key" in e.value for e in at.error)

    connect("OpenAI", "openai", "sk-openai")
    connect("Anthropic (Claude)", "anthropic", "sk-ant")
    connect("DeepSeek", "deepseek", "sk-deep")
    assert list(at.session_state["connections"]) == ["openai", "anthropic", "deepseek"]
    assert not at.error
    assert at.selectbox(key="model_reasoning").options == [
        "openai:gpt-5.2", "openai:gpt-5", "anthropic:claude-opus-5-5", "anthropic:claude-sonnet-5-5",
        "deepseek:deepseek-chat"]
    assert at.selectbox(key="model_reasoning").value == "openai:gpt-5.2"

    # Adding a model the provider does not list.
    at.selectbox(key="manual_provider").set_value("DeepSeek").run()
    at.text_input(key="manual_model").set_value("deepseek-reasoner").run()
    click(at, "Add model")
    assert "deepseek:deepseek-reasoner" in at.selectbox(key="model_reasoning").options

    # With three providers, the three debate roles default to three different families.
    build_signatures(at)
    at.radio(key="verification_method").set_value("Three-agent debate").run()
    roles = {role: at.selectbox(key=f"debate_model_{role}").value
             for role in ("skeptic", "discoverer", "mediator")}
    assert roles == {"skeptic": "openai:gpt-5.2", "discoverer": "anthropic:claude-opus-5-5",
                     "mediator": "deepseek:deepseek-chat"}

    click(at, "Remove")                                     # disconnects the first provider
    assert "openai" not in at.session_state["connections"]
    assert at.selectbox(key="model_reasoning").value == "anthropic:claude-opus-5-5"   # repaired, no crash
    assert problems(at) == []


# ------------------------------------------------------------------ layers

def test_layers_1_and_2_build_grounded_signatures(at):
    signatures = build_signatures(at)
    assert len(signatures) == 2
    facets = at.session_state["facets"]
    assert [f["facet_id"] for f in facets] == ["F1", "F2"]

    # Aliases: resolved through NCBI, ambiguous ones reported, unknown ones dropped.
    assert facets[0]["seeds_resolved"]["B7H3"]["symbol"] == "CD276"
    assert "CD276" in facets[0]["seed_genes"]
    assert facets[0]["seeds_ambiguous"] == {"TWOWAY": ["KRT5", "ALB"]}
    assert facets[0]["seeds_unknown"] == ["NOPE9"]
    provenance = at.session_state["l1_provenance"]
    assert provenance["model"] == "groq:openai/gpt-oss-120b"
    assert provenance["alias_resolution"]["resolved"] == 1 and provenance["alias_resolution"]["ambiguous"] == 1

    checkpoint, interferon = signatures.values()
    assert {"PDCD1", "HAVCR2"} <= set(checkpoint.genes)
    assert {"STAT1", "IRF7", "ISG15"} <= set(interferon.genes)
    assert not set(checkpoint.genes) & set(interferon.genes)
    for sig in signatures.values():
        assert len(sig.genes) == 20 and len(set(sig.genes)) == 20
        assert 0 < sig.support() < 0.99                          # no longer saturated
        assert sig.stability_score is None                       # undebated != "stable"
        assert sig.source_pathways and sig.retrieval_mode == "hybrid"
    assert at.session_state["retrieval_params"]["top_k"] == 10


def test_ncbi_being_unreachable_does_not_block_layer_1(at, monkeypatch):
    def down(self, names, organism="Homo sapiens"):
        raise ncbi_client.NCBIUnavailable("NCBI E-utilities unavailable: timeout")

    monkeypatch.setattr(ncbi_client.NCBIGene, "lookup", down)
    at.run()
    at.text_area(key="main_query").set_value("T cell exhaustion").run()
    click(at, "Generate mechanisms")
    assert problems(at) == []
    assert len(at.session_state["facets"]) == 2
    assert at.session_state["facets"][0]["seeds_unknown"] == ["B7H3", "TWOWAY", "NOPE9"]
    assert "NCBI unavailable" in at.session_state["l1_provenance"]["alias_resolution"]["error"]
    assert any("NCBI unavailable" in w.value for w in at.warning)


def test_every_layer_composes_and_the_user_keeps_the_last_word(at):
    signatures = build_signatures(at)
    retrieved = {sid: list(sig.genes) for sid, sig in signatures.items()}

    # Layer 3: additions are appended after the ranked genes, and re-running is idempotent.
    click(at, "Expand signatures")
    click(at, "Expand signatures")
    for sid, sig in at.session_state["signatures"].items():
        assert sig.genes == retrieved[sid] + ["NETGENE1", "NETGENE2"]   # TOOWEAK1 is below the cut-off
        assert sig.gene_sources["NETGENE1"] == "network" and sig.dam_expanded

    # Layer 4a: single-model review. Flags become default deselections, not deletions.
    pipeline.chat_json = fake_reviewer
    click(at, "Run review")
    assert problems(at) == []
    for sid, sig in at.session_state["signatures"].items():
        suggestion = at.session_state["suggestions"][sid]
        flagged = retrieved[sid][1]
        assert suggestion["status"] == "ok"
        assert suggestion["genes_to_remove"] == [flagged]
        assert suggestion["genes_to_add"] == ["TOX"] and suggestion["rejected"] == ["NOTREAL99"]
        assert at.session_state["deselected"][sid] == {flagged: "llm review: generic hub"}
        assert flagged in sig.genes                              # still a candidate

    # Layer 4b: debate. Votes are recorded, and the voted-out gene is deselected with its reason.
    at.radio(key="verification_method").set_value("Three-agent debate").run()
    candidates = {sid: list(sig.genes) for sid, sig in at.session_state["signatures"].items()}
    click(at, "Start debate")
    assert problems(at) == []
    for sid, sig in at.session_state["signatures"].items():
        result = at.session_state["debates"][sid]
        victim = candidates[sid][-3]
        assert result.genes_to_remove == [victim]
        assert result.metrics["stop_reason"] == "unanimous" and result.total_rounds == 2
        assert sig.genes == candidates[sid]                       # nothing deleted behind the user's back
        assert at.session_state["deselected"][sid][victim] == "debate: P(remove)=0.90, 3/3 agents"
        assert at.session_state["deselected"][sid][retrieved[sid][1]].startswith("llm review")
        assert sig.debate_verified and 0 < sig.stability_score <= 1 and sig.debate_agreement == 1.0
    assert at.session_state["debate_models"] == {
        "skeptic": REFS[0], "discoverer": REFS[1], "mediator": REFS[2]}

    # Layer 5: approval freezes exactly the selected candidates, in ranked order.
    click(at, "Approve all")
    approved = at.session_state["approved"]
    for sid, sig in at.session_state["signatures"].items():
        final = approved[sid]
        assert candidates[sid][-3] not in final                  # debate vote respected
        assert retrieved[sid][1] not in final                    # review flag respected
        assert "NETGENE2" in final                               # network addition survives
        assert final == [g for g in sig.genes if g in final]     # ranked order preserved
        assert len(final) == len(sig.genes) - 2
    assert len([e for e in at.main if type(e).__name__ == "DownloadButton"]) == 3
    assert problems(at) == []

    # The user overrules the debate: re-selecting the gene puts it back and voids the approval.
    sid, sig = next(iter(at.session_state["signatures"].items()))
    victim = candidates[sid][-3]
    picker = next(m for m in at.multiselect if m.key.startswith(f"genes_{sid}_"))
    picker.set_value([g for g in sig.genes if g != retrieved[sid][1]]).run()
    assert victim not in at.session_state["deselected"][sid]
    assert sid not in at.session_state["approved"]

    # Re-running an upstream layer invalidates everything built on it.
    click(at, "Build signatures")
    for key in ("approved", "debates", "suggestions", "deselected", "network_additions"):
        assert len(at.session_state[key]) == 0, key


def test_a_failed_review_is_reported_as_failed(at):
    build_signatures(at)

    def broken(llm, model, messages, **kwargs):
        raise RuntimeError("m did not return valid JSON after 6 attempts")

    pipeline.chat_json = broken
    click(at, "Run review")
    assert all(s["status"] == "failed" for s in at.session_state["suggestions"].values())
    assert any("NOT reviewed" in e.value for e in at.error)
    assert not any(sig.llm2_verified for sig in at.session_state["signatures"].values())
    assert all(not v for v in at.session_state["deselected"].values())


def test_a_retired_model_falls_back_instead_of_failing_the_layer(at, monkeypatch):
    import openai

    asked = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        def create(self, **params):
            asked.append(params["model"])
            if params["model"] == "dead/model-90b":
                raise Exception("Error code: 404 - {'error': {'message': 'The model `dead/model-90b` "
                                "does not exist', 'code': 'model_not_found'}}")
            message = types.SimpleNamespace(content=json.dumps(FACETS))
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=message, finish_reason="stop")], usage=None)

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(pipeline, "chat_json", llm_client.chat_json)

    at.session_state["connections"] = {"groq": groq_connection(["dead/model-90b", "openai/gpt-oss-120b"])}
    at.session_state["model_reasoning"] = "groq:dead/model-90b"
    at.run()
    at.text_area(key="main_query").set_value("T cell exhaustion").run()
    click(at, "Generate mechanisms")

    assert problems(at) == []
    assert asked == ["dead/model-90b", "openai/gpt-oss-120b"]
    assert len(at.session_state["facets"]) == 2
    provenance = at.session_state["l1_provenance"]
    assert provenance["model"] == "groq:openai/gpt-oss-120b"
    assert any("no longer available" in note for note in provenance["notes"])
    assert any("no longer available" in w.value for w in at.warning)
    assert provenance["applied"]["temperature"] == 0.0 and provenance["applied"]["seed"] == 0


def test_offline_backend_disables_expansion_and_debate_runs_ungrounded(at, monkeypatch):
    def down(*args, **kwargs):
        raise requests.ConnectionError("space asleep")

    monkeypatch.setattr(requests, "get", down)
    st.cache_data.clear()
    build_signatures(at)
    expand = [b for b in at.button if b.label == "Expand signatures"][0]
    assert expand.disabled

    at.radio(key="verification_method").set_value("Three-agent debate").run()
    click(at, "Start debate")
    assert problems(at) == []
    for result in at.session_state["debates"].values():
        assert result.parameters["database_grounded"] is False
        assert result.evidence is None

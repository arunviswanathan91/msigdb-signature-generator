import pytest

import pipeline
from conftest import axis_encoder
from llm_client import LLMError, LLMResult
from pipeline import (
    apply_debate_result, apply_network_expansion, apply_review, decompose_query,
    resolve_seed_aliases, retrieved_genes, verify_signatures,
)
from signature_engine import RetrievalParams, build_mechanism_signatures


def fake_chat(payloads, seen=None):
    queue = list(payloads)

    def chat(api_key, model, messages, **kwargs):
        if seen is not None:
            seen.append({"model": model, "prompt": messages[-1]["content"], **kwargs})
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResult(data=item, model=model)

    return chat


# ------------------------------------------------------------- decompose

def test_decompose_cleans_validates_and_numbers_mechanisms(monkeypatch, toy_index):
    seen = []
    payload = {"facets": [
        {"facet_name": "Checkpoints", "description": "Inhibitory receptors.",
         "seed_genes": ["pdcd1", "PDCD1", "CTLA4", "B7H3", 5]},
        "garbage",
        {"facet_name": "checkpoints", "description": "duplicate name", "seed_genes": ["LAG3"]},
        {"facet_name": "", "seed_genes": ["TOX"]},
        {"facet_name": "Interferon", "description": "ISGs.", "seed_genes": ["STAT1", "IRF7"]},
        {"facet_name": "Extra", "description": "beyond n", "seed_genes": ["MX1"]},
    ]}
    monkeypatch.setattr(pipeline, "chat_json", fake_chat([payload], seen))
    facets, result = decompose_query("k", "m", "T cell exhaustion", 2,
                                     context={"tissue": "Blood", "disease": "melanoma"},
                                     index=toy_index)
    assert [f["facet_id"] for f in facets] == ["F1", "F2"]
    assert [f["facet_name"] for f in facets] == ["Checkpoints", "Interferon"]
    assert facets[0]["seed_genes"] == ["PDCD1", "CTLA4"]        # normalised, de-duplicated
    assert facets[0]["seeds_unknown"] == ["B7H3"]               # alias the KB does not know
    assert result.notes == []
    prompt = seen[0]["prompt"]
    assert "EXACTLY 2" in prompt and "melanoma" in prompt and "T cell exhaustion" in prompt
    assert seen[0]["max_tokens"] >= 700


def test_decompose_reports_a_short_answer_and_rejects_an_empty_one(monkeypatch):
    one = {"facets": [{"facet_name": "Only one", "description": "d", "seed_genes": ["A"]}]}
    monkeypatch.setattr(pipeline, "chat_json", fake_chat([one]))
    facets, result = decompose_query("k", "m", "q", 5)
    assert len(facets) == 1
    assert "1 distinct mechanisms, 5 requested" in result.notes[0]

    monkeypatch.setattr(pipeline, "chat_json", fake_chat([{"facets": []}]))
    with pytest.raises(LLMError, match="no usable mechanisms"):
        decompose_query("k", "m", "q", 5)


def test_decompose_caps_the_number_of_mechanisms(monkeypatch):
    seen = []
    many = {"facets": [{"facet_name": f"M{i}", "description": "d", "seed_genes": []} for i in range(80)]}
    monkeypatch.setattr(pipeline, "chat_json", fake_chat([many], seen))
    facets, _ = decompose_query("k", "m", "q", 500)
    assert len(facets) == pipeline.MAX_MECHANISMS
    assert f"EXACTLY {pipeline.MAX_MECHANISMS}" in seen[0]["prompt"]


# ---------------------------------------------------------------- verify

@pytest.fixture
def signatures(toy_index):
    params = RetrievalParams(min_genes=8, max_genes=10)
    tcell = build_mechanism_signatures(toy_index, {
        "facet_name": "T cell exhaustion", "description": "d",
        "seed_genes": ["PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT"]}, axis_encoder, params, id_prefix="F1")
    ifn = build_mechanism_signatures(toy_index, {
        "facet_name": "Interferon response", "description": "interferon",
        "seed_genes": ["STAT1", "IRF7", "ISG15", "OAS1", "MX1"]}, axis_encoder, params, id_prefix="F2")
    return tcell + ifn


def test_verify_maps_labels_back_and_validates_every_gene(monkeypatch, toy_index, signatures):
    first, second = signatures
    victim = first.genes[-1]
    seen = []
    payload = {
        "S1": {"remove": {victim.lower(): "generic", "NOT_IN_SIG": "x"},
               "add": {"stat1": "interferon link", "MADEUP7": "x", first.genes[0]: "already there",
                       "IRF7": "a", "ISG15": "b", "OAS1": "over the cap"}},
        "S2": {"remove": {}, "add": {}},
    }
    monkeypatch.setattr(pipeline, "chat_json", fake_chat([payload], seen))
    suggestions, calls = verify_signatures("k", "m", signatures, "query", index=toy_index, batch_size=5)

    result = suggestions[first.signature_id]
    assert result["status"] == "ok"
    assert result["genes_to_remove"] == [victim]                # matched case-insensitively
    assert result["genes_to_add"] == ["STAT1", "IRF7", "ISG15"]  # capped at 3, KB-validated
    assert set(result["rejected"]) == {"NOT_IN_SIG", "MADEUP7"}
    assert result["reasoning"][victim] == "generic"
    assert suggestions[second.signature_id] == {
        "status": "ok", "error": None, "genes_to_remove": [], "genes_to_add": [],
        "reasoning": {}, "rejected": []}
    assert calls == [{"batch": 1, "status": "ok", "model": "m", "attempts": 1, "finish_reason": None,
                      "prompt_tokens": None, "completion_tokens": None, "notes": [], "applied": {}}]
    assert f"S1 - {first.signature_name}" in seen[0]["prompt"]
    assert ", ".join(first.genes) in seen[0]["prompt"]


def test_verify_never_reports_a_failed_batch_as_no_changes(monkeypatch, toy_index, signatures):
    first, second = signatures
    monkeypatch.setattr(pipeline, "chat_json", fake_chat([
        LLMError("m did not return valid JSON after 6 attempts"),
        {"S1": {"remove": {}, "add": {}}},
    ]))
    suggestions, calls = verify_signatures("k", "m", signatures, "q", index=toy_index, batch_size=1)
    assert suggestions[first.signature_id]["status"] == "failed"
    assert "valid JSON" in suggestions[first.signature_id]["error"]
    assert suggestions[second.signature_id]["status"] == "ok"
    assert [c["status"] for c in calls] == ["failed", "ok"]


def test_verify_flags_signatures_the_model_skipped(monkeypatch, toy_index, signatures):
    monkeypatch.setattr(pipeline, "chat_json", fake_chat([{"S1": {"remove": {}, "add": {}}}]))
    suggestions, _ = verify_signatures("k", "m", signatures, "q", index=toy_index, batch_size=5)
    assert suggestions[signatures[1].signature_id]["status"] == "failed"
    assert "omitted S2" in suggestions[signatures[1].signature_id]["error"]


def test_verify_can_forbid_additions(monkeypatch, toy_index, signatures):
    seen = []
    monkeypatch.setattr(pipeline, "chat_json", fake_chat(
        [{"S1": {"remove": [], "add": {"STAT1": "x"}}, "S2": {}}], seen))
    suggestions, _ = verify_signatures("k", "m", signatures, "q", suggest_additions=False,
                                       index=toy_index, publication_mode=True)
    assert suggestions[signatures[0].signature_id]["genes_to_add"] == []
    assert "PUBLICATION" in seen[0]["prompt"] and "leave empty" in seen[0]["prompt"]


# ----------------------------------------------------- alias resolution

def lookup_from(table):
    """Fake NCBIGene.lookup: {name: [official symbols]}."""
    seen = []

    def lookup(names):
        seen.append(list(names))
        return {name: [{"symbol": s, "match": "alias"} for s in table.get(name, [])] for name in names}

    lookup.seen = seen
    return lookup


def test_aliases_are_resolved_reported_or_left_unknown(toy_index):
    facets = [
        {"facet_id": "F1", "seed_genes": ["PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT"],
         "seeds_unknown": ["TIM3", "CD39", "AMBIG", "WEAKAMBIG", "OUTSIDEKB", "NOPE"]},
        {"facet_id": "F2", "seed_genes": ["STAT1"], "seeds_unknown": []},
    ]
    lookup = lookup_from({
        "TIM3": ["HAVCR2"],                 # unique, and already a seed
        "CD39": ["ENTPD1"],                 # unique, new
        "AMBIG": ["TOX", "BG7"],            # TOX co-occurs with the other seeds; BG7 does not
        "WEAKAMBIG": ["BG1", "BG2"],        # neither is supported by the context
        "OUTSIDEKB": ["NOTINKB1"],          # NCBI knows it, the knowledge base does not
    })
    summary = resolve_seed_aliases(facets, toy_index, lookup)

    first = facets[0]
    assert first["seed_genes"] == ["PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "ENTPD1", "TOX"]
    assert first["seeds_resolved"]["TIM3"] == {"symbol": "HAVCR2", "basis": "unique", "candidates": ["HAVCR2"]}
    assert first["seeds_resolved"]["CD39"]["symbol"] == "ENTPD1"
    assert first["seeds_resolved"]["AMBIG"] == {"symbol": "TOX", "basis": "context", "candidates": ["TOX", "BG7"]}
    assert first["seeds_ambiguous"] == {"WEAKAMBIG": ["BG1", "BG2"]}       # reported, never guessed
    assert first["seeds_unknown"] == ["OUTSIDEKB", "NOPE"]
    assert summary == {"source": "NCBI Gene (E-utilities)", "looked_up": 6,
                       "resolved": 3, "ambiguous": 1, "unresolved": 2}
    assert facets[1]["seeds_resolved"] == {} and facets[1]["seed_genes"] == ["STAT1"]
    assert len(lookup.seen) == 1                                            # one batched lookup


def test_alias_resolution_does_nothing_without_unknown_seeds(toy_index):
    lookup = lookup_from({})
    facets = [{"facet_id": "F1", "seed_genes": ["PDCD1"], "seeds_unknown": []}]
    assert resolve_seed_aliases(facets, toy_index, lookup)["looked_up"] == 0
    assert lookup.seen == [] and "seeds_resolved" not in facets[0]


def test_alias_resolution_propagates_an_unreachable_ncbi(toy_index):
    def down(names):
        raise ConnectionError("NCBI E-utilities unavailable")

    facets = [{"facet_id": "F1", "seed_genes": ["PDCD1"], "seeds_unknown": ["GLUT1"]}]
    with pytest.raises(ConnectionError):
        resolve_seed_aliases(facets, toy_index, down)
    assert facets[0]["seeds_unknown"] == ["GLUT1"]                          # untouched on failure


# ------------------------------------------------- writing outcomes back

def test_network_expansion_is_appended_and_idempotent(signatures):
    sig = signatures[0]
    base = list(sig.genes)
    first = apply_network_expansion(sig, [{"gene": "NET1"}, {"gene": base[0]}, {"gene": "NET2"}])
    assert first == ["NET1", "NET2"] and sig.genes == base + ["NET1", "NET2"]
    assert sig.gene_sources["NET1"] == "network" and sig.dam_expanded
    assert retrieved_genes(sig) == base
    apply_network_expansion(sig, [{"gene": "NET3"}])                         # re-run replaces, never stacks
    assert sig.genes == base + ["NET3"] and "NET1" not in sig.gene_sources
    apply_network_expansion(sig, [])
    assert sig.genes == base and not sig.dam_expanded


def test_review_outcomes_are_applied_only_when_the_review_succeeded(signatures):
    sig = signatures[0]
    victim, base = sig.genes[2], list(sig.genes)
    failed = {"status": "failed", "genes_to_remove": [victim], "genes_to_add": ["STAT1"], "reasoning": {}}
    assert apply_review(sig, failed) == {"removed": [], "added": []}
    assert sig.genes == base and not sig.llm2_verified

    ok = {"status": "ok", "genes_to_remove": [victim, "NOT_THERE"], "genes_to_add": ["STAT1"],
          "reasoning": {victim: "generic hub"}}
    assert apply_review(sig, ok, remove=False) == {"removed": [], "added": []}
    assert sig.genes == base and sig.llm2_verified                           # recorded, nothing changed
    outcome = apply_review(sig, ok, remove=True, add=True)
    assert outcome == {"removed": [victim], "added": ["STAT1"]}
    assert sig.genes == [g for g in base if g != victim] + ["STAT1"]
    assert sig.removed_genes[victim] == "llm review: generic hub" and sig.gene_sources["STAT1"] == "llm"


class FakeDebate:
    def __init__(self, genes, removed, failed=False):
        import types
        self.failed = failed
        self.genes_to_remove = removed
        self.metrics = {"stability": 0.81, "entropy": 0.3, "conflict": 0.05, "agreement": 0.9}
        self.decisions = {g: types.SimpleNamespace(p_remove=0.9, n_remove=3, n_voters=3) for g in genes}


def test_debate_outcomes_respect_failure_and_the_apply_switch(signatures):
    sig = signatures[0]
    base, victim = list(sig.genes), sig.genes[-1]

    assert apply_debate_result(sig, FakeDebate(base, [victim], failed=True)) == []
    assert sig.genes == base and sig.stability_score is None and not sig.debate_verified

    assert apply_debate_result(sig, FakeDebate(base, [victim]), apply_removals=False) == []
    assert sig.genes == base and sig.stability_score == 0.81 and sig.debate_verified
    assert sig.quality_label() == "HIGH_CONFIDENCE"

    assert apply_debate_result(sig, FakeDebate(base, [victim, "NOT_THERE"])) == [victim]
    assert victim not in sig.genes
    assert sig.removed_genes[victim] == "debate: P(remove)=0.90, 3/3 agents"

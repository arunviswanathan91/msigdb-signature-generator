import pytest

import debate_engine
from debate_engine import (
    DebateEngine, GeneVote, aggregate, agreement_rate, binary_entropy,
    build_opening_prompt, build_rebuttal_prompt, build_recap, clip_confidence,
    contested_genes, estimate_tokens_per_model, fleiss_kappa, opening_budget, parse_votes,
)
from llm_client import LLMError, LLMResult, LLMRouter

GENES = ["PDCD1", "GAPDH", "TOX", "ALB"]
MODELS = {"skeptic": "vendor-a/model-70b", "discoverer": "vendor-b/model-30b",
          "mediator": "vendor-c/model-20b"}
LLM = LLMRouter.single("groq", "test-key")        # never contacted: achat_json is faked


def positions(**by_role):
    """positions(skeptic={'G': ('remove', 0.9)}, ...) -> role -> gene -> GeneVote"""
    return {role: {g: GeneVote(g, v, c) for g, (v, c) in votes.items()}
            for role, votes in by_role.items()}


# --------------------------------------------------------------- parsing

def test_parse_compact_opening_shape():
    votes, stray = parse_votes(
        {"keep": {"pdcd1": 0.9, "XYZ1": 0.8},
         "remove": {"GAPDH": {"confidence": 95, "reason": "housekeeping"}}}, GENES)
    assert votes["PDCD1"].vote == "keep" and votes["PDCD1"].confidence == 0.9
    assert votes["GAPDH"].vote == "remove"
    assert votes["GAPDH"].confidence == 0.95          # percentages are rescaled
    assert votes["GAPDH"].reason == "housekeeping"
    assert stray == ["XYZ1"]                          # hallucinated symbol: counted, never used


def test_parse_list_and_rebuttal_shapes():
    votes, _ = parse_votes({"keep": ["PDCD1"],
                            "remove": [{"gene": "TOX", "confidence": 0.7, "reason": "r"}]}, GENES)
    assert votes["PDCD1"].confidence == 0.5           # no confidence given -> neutral
    assert votes["TOX"].p_remove == 0.7
    votes, _ = parse_votes({"votes": [{"gene": "ALB", "vote": "Remove", "confidence": "0.8"},
                                      {"gene": "TOX", "vote": "maybe"}, "junk"]}, GENES)
    assert list(votes) == ["ALB"]                     # invalid vote values are dropped
    assert parse_votes("not json", GENES) == ({}, [])


@pytest.mark.parametrize("raw, expected", [
    (0.9, 0.9), (0.2, 0.5), (1.7, 0.5), (87, 0.87), (500, 1.0), ("0.75", 0.75),
    (None, 0.5), ("high", 0.5), (float("nan"), 0.5),
])
def test_clip_confidence(raw, expected):
    assert clip_confidence(raw) == pytest.approx(expected)


# ------------------------------------------------------------ statistics

def test_binary_entropy():
    assert binary_entropy(0.5) == pytest.approx(1.0)
    assert binary_entropy(0.0) == binary_entropy(1.0) == 0.0
    assert binary_entropy(0.9) == pytest.approx(0.4690, abs=1e-4)


def test_aggregate_matches_hand_calculation():
    pos = positions(skeptic={"G": ("remove", 0.9)}, discoverer={"G": ("remove", 0.8)},
                    mediator={"G": ("keep", 0.7)})
    decision = aggregate(["G"], pos)["G"]
    assert decision.p_remove == pytest.approx((0.9 + 0.8 + 0.3) / 3)
    assert decision.decision == "remove" and decision.n_remove == 2 and decision.n_voters == 3
    assert decision.entropy == pytest.approx(binary_entropy(2 / 3))
    mean = 2 / 3
    variance = ((0.9 - mean) ** 2 + (0.8 - mean) ** 2 + (0.3 - mean) ** 2) / 3
    assert decision.conflict == pytest.approx(4 * variance)


def test_removal_needs_both_pooled_probability_and_majority():
    # Majority votes remove, but weakly, against one certain keep -> pooled p < 0.5 -> keep.
    weak_majority = positions(skeptic={"G": ("remove", 0.55)}, discoverer={"G": ("remove", 0.55)},
                              mediator={"G": ("keep", 1.0)})
    assert aggregate(["G"], weak_majority)["G"].decision == "keep"
    # One certain remove against two coin-flip keeps -> pooled p >= 0.5 but no majority -> keep.
    lone_voice = positions(skeptic={"G": ("remove", 1.0)}, discoverer={"G": ("keep", 0.5)},
                           mediator={"G": ("keep", 0.5)})
    decision = aggregate(["G"], lone_voice)["G"]
    assert decision.p_remove >= 0.5 and decision.decision == "keep"
    # A lukewarm 2-1 split is not a consensus: pooled p = 0.52 stays below the default 0.6.
    lukewarm = positions(skeptic={"G": ("remove", 0.85)}, discoverer={"G": ("keep", 0.9)},
                         mediator={"G": ("remove", 0.6)})
    decision = aggregate(["G"], lukewarm)["G"]
    assert decision.p_remove == pytest.approx((0.85 + 0.1 + 0.6) / 3) and decision.decision == "keep"
    assert aggregate(["G"], lukewarm, threshold=0.5)["G"].decision == "remove"
    # A stricter threshold can overturn a removal.
    strong = positions(skeptic={"G": ("remove", 0.7)}, discoverer={"G": ("remove", 0.7)},
                       mediator={"G": ("remove", 0.7)})
    assert aggregate(["G"], strong, threshold=0.5)["G"].decision == "remove"
    assert aggregate(["G"], strong, threshold=0.8)["G"].decision == "keep"


def test_quorum_and_database_flags():
    pos = positions(skeptic={"A": ("remove", 0.9), "B": ("remove", 0.9)},
                    discoverer={"A": ("remove", 0.9)})
    evidence = {"genes": {"A": {"flags": ["isolated"]}, "B": {"flags": []}}}
    decisions = aggregate(["A", "B"], pos, evidence=evidence)
    assert decisions["A"].decision == "remove" and decisions["A"].db_flags == ["isolated"]
    assert decisions["B"].decision == "no_quorum" and decisions["B"].p_remove is None


def test_agreement_contested_and_kappa():
    def table(remove_counts):
        roles = ["skeptic", "discoverer", "mediator"]
        pos = {r: {} for r in roles}
        for g, k in enumerate(remove_counts):
            for i, role in enumerate(roles):
                pos[role][f"G{g}"] = GeneVote(f"G{g}", "remove" if i < k else "keep", 0.9)
        return [f"G{g}" for g in range(len(remove_counts))], pos

    genes, pos = table([3, 2, 0, 0])
    assert contested_genes(genes, pos) == ["G1"]
    assert agreement_rate(genes, pos) == pytest.approx(0.75)
    assert fleiss_kappa(genes, pos) == pytest.approx(0.6571, abs=1e-4)   # worked by hand

    genes, pos = table([3, 0, 3, 0])
    assert fleiss_kappa(genes, pos) == pytest.approx(1.0)
    genes, pos = table([0, 0, 0])
    assert fleiss_kappa(genes, pos) is None           # no variation: kappa undefined
    assert agreement_rate(genes, pos) == 1.0


# --------------------------------------------------------------- prompts

def test_opening_prompt_contains_everything_the_agents_need():
    evidence = {"tissue": None, "tissue_requested": None, "genes": {
        g: {"links_in_signature": 1, "pathway_count": 10, "tissue_tpm": None, "flags": []}
        for g in GENES}}
    prompt = build_opening_prompt(GENES, query="T cell exhaustion", mechanism="Checkpoints",
                                  description="Inhibitory receptors.",
                                  context={"tissue": "Blood", "disease": "melanoma"},
                                  evidence=evidence)
    for needle in ["T cell exhaustion", "Checkpoints", "Inhibitory receptors.", "melanoma",
                   "PDCD1, GAPDH, TOX, ALB", "links_to_signature", "ALL 4 genes"]:
        assert needle in prompt
    assert "unavailable" in build_opening_prompt(GENES, query="q", mechanism="m", description="",
                                                 context=None, evidence=None)


def test_rebuttal_prompt_labels_own_and_others_positions():
    pos = positions(skeptic={"TOX": ("remove", 0.9)}, discoverer={"TOX": ("keep", 0.8)})
    pos["skeptic"]["TOX"].reason = "not a receptor"
    prompt = build_rebuttal_prompt(["TOX"], pos, "skeptic", 2, None)
    assert "YOU: remove (confidence 0.90) - not a receptor" in prompt
    assert "Advocate: keep (confidence 0.80)" in prompt


def test_budgets_grow_with_signature_size():
    assert opening_budget(30) > opening_budget(10) > 0
    assert opening_budget(30) < 1000      # must fit a 1,000 token/min output allowance
    assert estimate_tokens_per_model(20, 3) > estimate_tokens_per_model(20, 1) > 0
    assert estimate_tokens_per_model(40, 2) > estimate_tokens_per_model(20, 2)
    assert estimate_tokens_per_model(20, 2, grounded=True) > estimate_tokens_per_model(20, 2, grounded=False)


def test_recap_names_the_signature_but_omits_the_evidence_table():
    recap = build_recap(GENES, query="q", mechanism="Checkpoints", description="Receptors.",
                        context={"disease": "melanoma"})
    assert "Checkpoints" in recap and "PDCD1, GAPDH, TOX, ALB" in recap and "melanoma" in recap
    assert "links_to_signature" not in recap


# ---------------------------------------------------------------- engine

def scripted(script, calls=None):
    """Fake achat_json. ``script[role]`` is a list of payloads/exceptions per round."""
    counters = {}

    async def fake(client, model, messages, **kwargs):
        role = next(r for r, m in MODELS.items() if m == model)
        step = counters.get(role, 0)
        counters[role] = step + 1
        if calls is not None:
            calls.append((role, step + 1, messages, kwargs))
        item = script[role][step]
        if isinstance(item, Exception):
            raise item
        return LLMResult(data=item, model=model)

    return fake


def keep_all(conf=0.9, **removals):
    return {"keep": {g: conf for g in GENES if g not in removals},
            "remove": {g: {"confidence": c, "reason": f"{g} is unrelated"} for g, c in removals.items()}}


def test_debate_converges_and_only_contested_genes_are_reopened(monkeypatch):
    calls = []
    script = {
        "skeptic":    [keep_all(GAPDH=0.95, ALB=0.9, TOX=0.8),
                       {"votes": [{"gene": "TOX", "vote": "keep", "confidence": 0.7, "reason": "TF of exhaustion"}]}],
        "discoverer": [keep_all(GAPDH=0.9, ALB=0.9),
                       {"votes": [{"gene": "TOX", "vote": "keep", "confidence": 0.9, "reason": "core TF"}]}],
        "mediator":   [keep_all(GAPDH=0.9, ALB=0.8),
                       {"votes": [{"gene": "TOX", "vote": "keep", "confidence": 0.9, "reason": "core TF"}]}],
    }
    monkeypatch.setattr(debate_engine, "achat_json", scripted(script, calls))
    result = DebateEngine(LLM, MODELS).run(GENES, None, query="q", mechanism="m", max_rounds=3)

    assert result.genes_to_remove == ["GAPDH", "ALB"]
    assert result.decisions["TOX"].decision == "keep"
    assert result.total_rounds == 2
    assert result.rounds[0].contested == ["TOX"] and result.rounds[1].contested == []
    assert result.metrics["stop_reason"] == "unanimous"
    assert result.metrics["vote_flips"] == 1
    assert result.metrics["agreement"] == 1.0
    assert result.metrics["agents_responding"] == 3 and not result.failed
    assert 0 < result.metrics["stability"] <= 1
    # Round 2 asked about TOX only, and showed the other agents' reasons.
    round2 = [c for c in calls if c[1] == 2]
    assert len(round2) == 3
    rebuttal = round2[0][2][-1]["content"]
    assert "TOX" in rebuttal and "GAPDH" not in rebuttal.split("Positions from the previous round:")[1]
    # Rebuttals carry a recap, not the full evidence table again.
    assert all(len(c[2]) == 3 and "SIGNATURE (4 genes)" in c[2][1]["content"] for c in round2)
    # A role whose model fails may borrow another role's model only as a last resort.
    skeptic_call = next(c for c in calls if c[0] == "skeptic")
    assert skeptic_call[3]["fallbacks"] == ["vendor-b/model-30b", "vendor-c/model-20b"]
    # Each role got its own system prompt.
    systems = {c[0]: c[2][0]["content"] for c in calls if c[1] == 1}
    assert "SKEPTIC" in systems["skeptic"] and "ADVOCATE" in systems["discoverer"]
    assert result.parameters["database_grounded"] is False
    assert result.parameters["diversity"] == {"distinct_models": 3, "distinct_families": 3, "roles": 3}
    record = result.to_dict()
    assert record["genes_removed"] == ["GAPDH", "ALB"] and len(record["gene_decisions"]) == 4


def test_debate_survives_one_failed_agent(monkeypatch):
    script = {"skeptic": [keep_all(GAPDH=0.9)], "discoverer": [LLMError("rate limited")],
              "mediator": [keep_all(GAPDH=0.9)]}
    monkeypatch.setattr(debate_engine, "achat_json", scripted(script))
    result = DebateEngine(LLM, MODELS).run(GENES, None, max_rounds=3)
    assert result.metrics["agents_responding"] == 2 and not result.failed
    assert result.genes_to_remove == ["GAPDH"]
    assert any(t.error for t in result.rounds[0].turns)


def test_debate_with_a_single_agent_takes_no_decision(monkeypatch):
    script = {"skeptic": [keep_all(GAPDH=0.99, ALB=0.99)], "discoverer": [LLMError("down")],
              "mediator": [LLMError("down")]}
    monkeypatch.setattr(debate_engine, "achat_json", scripted(script))
    result = DebateEngine(LLM, MODELS).run(GENES, None, max_rounds=3)
    assert result.failed
    assert result.genes_to_remove == []
    assert result.metrics["stop_reason"] == "insufficient_agents"
    assert all(d.decision == "no_quorum" for d in result.decisions.values())


def test_stalemate_stops_the_debate_and_out_of_set_genes_are_ignored(monkeypatch):
    stubborn_remove = {"votes": [{"gene": "TOX", "vote": "remove", "confidence": 0.9, "reason": "no"},
                                 {"gene": "FAKE9", "vote": "remove", "confidence": 0.9, "reason": "x"}]}
    stubborn_keep = {"votes": [{"gene": "TOX", "vote": "keep", "confidence": 0.9, "reason": "yes"}]}
    script = {"skeptic": [keep_all(TOX=0.9), stubborn_remove, stubborn_remove],
              "discoverer": [keep_all(), stubborn_keep, stubborn_keep],
              "mediator": [keep_all(), stubborn_keep, stubborn_keep]}
    monkeypatch.setattr(debate_engine, "achat_json", scripted(script))
    result = DebateEngine(LLM, MODELS).run(GENES, None, max_rounds=4)
    assert result.metrics["stop_reason"] == "stalemate" and result.total_rounds == 2
    assert result.decisions["TOX"].decision == "keep"          # 1 of 3 is not a majority
    assert result.metrics["out_of_set_mentions"] == 1
    assert "FAKE9" not in result.decisions
    assert result.decisions["TOX"].conflict > result.decisions["PDCD1"].conflict


def test_engine_requires_all_roles():
    with pytest.raises(ValueError, match="mediator"):
        DebateEngine(LLM, {"skeptic": "a", "discoverer": "b"})


def test_spare_models_are_preferred_over_sharing_when_a_role_needs_a_fallback(monkeypatch):
    calls = []
    script = {role: [keep_all()] for role in MODELS}
    monkeypatch.setattr(debate_engine, "achat_json", scripted(script, calls))
    DebateEngine(LLM, MODELS, fallbacks=["spare/model-9b", *MODELS.values()]).run(GENES, None)
    mediator = next(c for c in calls if c[0] == "mediator")
    assert mediator[3]["fallbacks"] == ["spare/model-9b", "vendor-a/model-70b", "vendor-b/model-30b"]


def test_membership_rule_is_part_of_every_agents_instructions():
    for role in MODELS:
        assert "MEMBERSHIP RULE" in debate_engine.SHARED_RULES
        assert debate_engine.ROLE_PROMPTS[role].startswith("ROLE - ")
    # Direct partners must not be voted out for failing to be the namesake molecule.
    assert "direct regulator, ligand, receptor, effector" in debate_engine.SHARED_RULES


def test_roles_may_come_from_different_providers(monkeypatch):
    calls = []
    models = {"skeptic": "openai:gpt-5", "discoverer": "anthropic:claude-opus-5-5",
              "mediator": "google:gemini-2.5-pro"}

    async def fake(pool, model, messages, **kwargs):
        calls.append(model)
        return LLMResult(data=keep_all(), model=model)

    monkeypatch.setattr(debate_engine, "achat_json", fake)
    result = DebateEngine(LLM, models).run(GENES, None)
    assert sorted(calls) == sorted(models.values())
    assert result.models == models
    assert result.parameters["diversity"]["distinct_families"] == 3

import csv
import io
import json

from conftest import axis_encoder
from exporters import (
    build_csv, build_gmt, build_json, build_provenance, file_sha256, package_versions,
)
from signature_engine import RetrievalParams, build_mechanism_signatures


def make(toy_index):
    sig = build_mechanism_signatures(toy_index, {
        "facet_name": "T cell | exhaustion", "description": "d",
        "seed_genes": ["PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT"]},
        axis_encoder, RetrievalParams(min_genes=8, max_genes=10), id_prefix="F1")[0]
    return sig


def test_gmt_is_valid_and_carries_quality_tags(toy_index):
    sig = make(toy_index)
    lines = build_gmt([sig]).splitlines()
    assert len(lines) == 1
    fields = lines[0].split("\t")
    assert fields[0] == sig.signature_id
    assert fields[2:] == sig.genes
    assert "stability:NA" in fields[1] and "DEBATE" not in fields[1]
    assert fields[1].count("|") == 4            # a '|' inside the facet name cannot break the format

    sig.debate_verified, sig.stability_score, sig.dam_expanded = True, 0.8771, True
    description = build_gmt([sig]).split("\t")[1]
    assert "stability:0.877" in description and description.endswith("NETWORK|DEBATE")
    assert build_gmt([]) == ""


def test_csv_is_long_format_one_row_per_gene(toy_index):
    sig = make(toy_index)
    sig.genes.append("ADDED1")
    sig.gene_sources["ADDED1"] = "network"
    rows = list(csv.DictReader(io.StringIO(build_csv([sig]))))
    assert [r["gene"] for r in rows] == sig.genes
    assert [int(r["rank"]) for r in rows] == list(range(1, len(sig.genes) + 1))
    assert rows[0]["source"] == "knowledge_base" and rows[0]["quality"] == "NOT_ASSESSED"
    assert rows[-1]["source"] == "network" and rows[-1]["score"] == ""
    assert float(rows[-1]["weighted_support"]) == 0.0
    assert rows[0]["llm_seed"] in ("True", "False")


def test_json_export_is_self_describing(toy_index):
    sig = make(toy_index)
    assessed = make(toy_index)
    assessed.signature_id = "F9_OTHER__CORE"
    assessed.genes = assessed.genes[:5]
    assessed.stability_score, assessed.debate_entropy, assessed.debate_conflict = 0.9, 0.2, 0.05
    record = json.loads(build_json(
        [sig, assessed], query="T cell exhaustion", generation_mode="publication",
        biological_context={"tissue": "Blood"},
        provenance={"pipeline_version": "3.0.0", "retrieval_parameters": {"top_k": 30}},
        debates={"F9_OTHER__CORE": {"metrics": {"stability": 0.9}}},
        verification={"model": "m"}, timings={"Layer 1: mechanisms": "2.0s"}))

    assert record["schema"].endswith("/3")
    assert record["query"] == "T cell exhaustion"
    assert record["provenance"]["pipeline_version"] == "3.0.0"
    assert record["provenance"]["engine_version"]
    assert "numpy" in record["provenance"]["packages"]
    summary = record["summary"]
    assert summary["total_signatures"] == 2
    assert summary["overlap"]["pairs"] == 1
    # Undebated signatures are NOT counted as high confidence.
    assert summary["debate"] == {
        "signatures_assessed": 1, "signatures_not_assessed": 1, "high_confidence": 1,
        "moderate_confidence": 0, "requires_manual_review": 0, "mean_stability": 0.9,
        "mean_entropy": 0.2, "mean_conflict": 0.05}
    assert record["signatures"][0]["debate_quality"]["interpretation"] == "NOT_ASSESSED"
    assert record["signatures"][0]["retrieval"]["source_gene_sets"]
    assert record["debates"]["F9_OTHER__CORE"]["metrics"]["stability"] == 0.9


def test_helpers(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    assert file_sha256(str(path)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert file_sha256(str(tmp_path / "missing")) is None
    assert "python" in package_versions()


def test_provenance_identifies_data_code_and_providers_but_never_keys(toy_index):
    from llm_client import LLMRouter
    from providers import Connection

    router = LLMRouter({"groq": Connection("groq", "gsk_SECRET", models={"openai/gpt-oss-120b": {}}),
                        "anthropic": Connection("anthropic", "sk-ant-SECRET", models={"claude-opus-5-5": {}})},
                       effort="high")
    resources = {"index": toy_index, "kb_metadata": {"version": "2025.1", "sources": {"toy": 80}},
                 "embedding_model": "all-MiniLM-L6-v2", "kb_sha256": "abc", "embeddings_sha256": "def"}
    provenance = build_provenance(
        resources, pipeline_version="3.0.0", debate_version="3.0.0",
        llm_providers=router.public(), reasoning_effort=router.effort, pathway_api_url="https://api",
        decomposition={"model": "groq:openai/gpt-oss-120b"}, mechanisms=[{"facet_id": "F1"}],
        mechanisms_selected=["F1"], retrieval_parameters={"top_k": 10},
        debate_models={"skeptic": "groq:openai/gpt-oss-120b"})

    assert provenance["knowledge_base"] == {"version": "2025.1", "created": None, "sources": {"toy": 80},
                                            "filters": None, "gene_sets": 80,
                                            "genes": toy_index.n_genes, "sha256": "abc"}
    assert provenance["embeddings"]["sha256"] == "def" and "first 10 genes" in provenance["embeddings"]["text_recipe"]
    assert set(provenance["llm_providers"]) == {"groq", "anthropic"}
    assert provenance["llm_providers"]["anthropic"]["base_url"] == "official SDK default"
    assert provenance["reasoning_effort"] == "high"
    assert "SECRET" not in json.dumps(provenance)

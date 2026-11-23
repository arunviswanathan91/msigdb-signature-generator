import numpy as np
import pytest
from scipy.stats import fisher_exact

from conftest import IFN, TCELL, axis_encoder
from signature_engine import (
    GeneSignature, KnowledgeIndex, RetrievalParams, benjamini_hochberg,
    build_mechanism_signatures, cluster_gene_sets, deduplicate,
    humanise_gene_set_name, jaccard, overlap_summary, retrieve_gene_sets,
    seed_enrichment, slugify,
)

FACET = {"facet_name": "T cell exhaustion",
         "description": "Chronically stimulated T cells express inhibitory receptors.",
         "seed_genes": ["PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "TOX"]}


# ------------------------------------------------------------------ index

def test_index_shapes_and_symbol_normalisation(toy_index):
    assert toy_index.n_pathways == 80
    assert toy_index.matrix.shape == (80, 8)
    known, unknown = toy_index.normalise_symbols(["pdcd1", "PDCD1", " Tox ", "NOTAGENE", "", 7])
    assert known == ["PDCD1", "TOX"]          # case-insensitive, de-duplicated, order kept
    assert unknown == ["NOTAGENE"]


def test_index_rejects_mismatched_embeddings():
    with pytest.raises(ValueError, match="no embedding"):
        KnowledgeIndex({"A": ["G1", "G2"], "B": ["G2", "G3"]}, {"A": np.ones(4)})


def test_idf_penalises_promiscuous_genes(toy_index):
    # Find the most and least frequent genes; idf must order them inversely.
    frequency = {g: len(v) for g, v in toy_index.gene_to_idx.items()}
    common = max(frequency, key=frequency.get)
    rare = min(frequency, key=frequency.get)
    assert frequency[common] > frequency[rare]
    assert toy_index.idf[common] < toy_index.idf[rare]


# ------------------------------------------------------------- statistics

def test_benjamini_hochberg_matches_hand_calculation():
    adjusted = benjamini_hochberg(np.array([0.01, 0.02, 0.03]), n_tests=3)
    assert np.allclose(adjusted, [0.03, 0.03, 0.03])
    # Correcting for more hypotheses than were observed scales up accordingly.
    assert np.allclose(benjamini_hochberg(np.array([0.01]), n_tests=100), [1.0])


def test_seed_enrichment_matches_fisher_exact(toy_index):
    seeds = FACET["seed_genes"]
    counts, p_values, fdr = seed_enrichment(toy_index, seeds)
    i = toy_index.ids.index("TCELL_EXHAUSTION_0")
    members = toy_index.gene_set(i)
    a = len(members & set(seeds))
    assert counts[i] == a
    table = [[a, len(seeds) - a],
             [len(members) - a, toy_index.n_genes - len(members) - len(seeds) + a]]
    assert p_values[i] == pytest.approx(fisher_exact(table, alternative="greater")[1], rel=1e-9)
    assert np.all(fdr >= p_values - 1e-15) and np.all(fdr <= 1.0)
    # Gene sets sharing no seed are never significant.
    assert np.all(p_values[counts == 0] == 1.0)


# -------------------------------------------------------------- retrieval

def test_hybrid_retrieval_finds_the_right_family(toy_index):
    hits, mode = retrieve_gene_sets(toy_index, axis_encoder("t cell"), FACET["seed_genes"],
                                    RetrievalParams(top_k=8))
    assert mode == "hybrid"
    assert len(hits) == 8
    assert all(h.pathway_id.startswith("TCELL_EXHAUSTION") for h in hits)
    assert hits[0].weight == pytest.approx(1.0)
    assert all(0 < h.weight <= 1 for h in hits)


def test_retrieval_degrades_to_single_channels(toy_index):
    _, mode = retrieve_gene_sets(toy_index, None, FACET["seed_genes"], RetrievalParams())
    assert mode == "enrichment"
    hits, mode = retrieve_gene_sets(toy_index, axis_encoder("interferon"), [], RetrievalParams(top_k=5))
    assert mode == "semantic"
    assert all(h.pathway_id.startswith("INTERFERON") for h in hits)
    assert retrieve_gene_sets(toy_index, None, [], RetrievalParams()) == ([], "none")


def test_seeds_overrule_a_misleading_description(toy_index):
    # Description points at interferon, seeds are T-cell genes: enrichment must
    # put the T-cell sets on top, because they win on BOTH pools' rank fusion
    # only if supported by genes.
    hits, mode = retrieve_gene_sets(toy_index, axis_encoder("interferon"), FACET["seed_genes"],
                                    RetrievalParams(top_k=16, semantic_pool=8))
    assert mode == "hybrid"
    families = {h.pathway_id.rsplit("_", 1)[0] for h in hits}
    assert families == {"TCELL_EXHAUSTION", "INTERFERON_RESPONSE"}
    tcell_hits = [h for h in hits if h.pathway_id.startswith("TCELL")]
    assert all(h.overlap >= 2 and h.fdr <= 0.05 for h in tcell_hits)


def test_exclusion_mask_is_respected(toy_index):
    exclude = np.array([p.startswith("TCELL") for p in toy_index.ids])
    hits, _ = retrieve_gene_sets(toy_index, axis_encoder("t cell"), FACET["seed_genes"],
                                 RetrievalParams(), exclude=exclude)
    assert not any(h.pathway_id.startswith("TCELL") for h in hits)


# ------------------------------------------------------------- signatures

def test_signature_is_grounded_and_deterministic(toy_index):
    params = RetrievalParams(min_genes=8, max_genes=12)
    first = build_mechanism_signatures(toy_index, FACET, axis_encoder, params)
    second = build_mechanism_signatures(toy_index, FACET, axis_encoder, params)
    assert len(first) == 1
    sig = first[0]
    assert sig.genes == second[0].genes                      # bit-for-bit reproducible
    assert set(sig.genes) <= set(TCELL)                      # no background, no interferon genes
    assert 8 <= len(sig.genes) <= 12
    assert sig.retrieval_mode == "hybrid"
    assert 0.0 < sig.confidence <= 1.0
    assert sig.support() == pytest.approx(sig.confidence)
    assert all(sig.gene_support[g] >= 2 for g in sig.genes)
    # Genes the LLM never proposed must be able to enter the signature.
    assert sig.current_novel_fraction() > 0
    assert sig.stability_score is None and sig.quality_label() == "NOT_ASSESSED"


def test_unsupported_and_unknown_seeds_are_reported_not_kept(toy_index):
    facet = dict(FACET, seed_genes=FACET["seed_genes"] + ["BG7", "FAKEGENE1"])
    sig = build_mechanism_signatures(toy_index, facet, axis_encoder,
                                     RetrievalParams(min_genes=8, max_genes=12))[0]
    assert sig.seeds_unknown == ["FAKEGENE1"]
    assert "BG7" in sig.seeds_unsupported and "BG7" not in sig.genes
    assert sig.current_seed_recovery() < 1.0


def test_support_drops_when_unsupported_genes_are_added(toy_index):
    sig = build_mechanism_signatures(toy_index, FACET, axis_encoder,
                                     RetrievalParams(min_genes=8, max_genes=12))[0]
    before = sig.support()
    sig.genes.append("BG399")                 # in no retrieved gene set
    assert sig.support() < before
    assert sig.to_dict()["confidence"] < sig.to_dict()["confidence_at_build"]


def test_housekeeping_is_flagged_in_exploratory_and_removed_in_publication(toy_index_housekeeping):
    base = dict(min_genes=8, max_genes=13)
    exploratory = build_mechanism_signatures(
        toy_index_housekeeping, FACET, axis_encoder, RetrievalParams(**base))[0]
    assert "GAPDH" in exploratory.genes
    assert exploratory.housekeeping_flagged == ["GAPDH"]
    assert exploratory.removed_genes == {}

    publication = build_mechanism_signatures(
        toy_index_housekeeping, FACET, axis_encoder,
        RetrievalParams(drop_housekeeping=True, **base))[0]
    assert "GAPDH" not in publication.genes
    assert "GAPDH" in publication.removed_genes
    assert len(publication.genes) == len(exploratory.genes)   # backfilled, not shortened


def test_too_few_genes_returns_nothing(toy_index):
    assert build_mechanism_signatures(toy_index, FACET, axis_encoder,
                                      RetrievalParams(min_genes=500, max_genes=600)) == []


def test_submodules_separate_distinct_gene_set_families(toy_index):
    mixed = {"facet_name": "Mixed", "description": "",
             "seed_genes": TCELL[:6] + IFN[:6]}
    hits, _ = retrieve_gene_sets(toy_index, None, mixed["seed_genes"], RetrievalParams(top_k=16))
    groups = cluster_gene_sets(toy_index, hits, 2)
    assert len(groups) == 2
    for group in groups:
        assert len({h.pathway_id.rsplit("_", 1)[0] for h in group}) == 1   # homogeneous

    signatures = build_mechanism_signatures(
        toy_index, mixed, None, RetrievalParams(min_genes=6, max_genes=10, top_k=16), n_variants=3)
    assert [s.mechanism for s in signatures][0] == "Core"
    assert len(signatures) == 3
    assert len({s.signature_id for s in signatures}) == 3


def test_deduplicate_and_overlap_summary():
    def sig(sid, genes):
        return GeneSignature(sid, sid, genes, "f", "m", 0.5)

    a, b, c = sig("A", list("abcdef")), sig("B", list("abcdeg")), sig("C", list("uvwxyz"))
    assert jaccard(a.genes, b.genes) == pytest.approx(5 / 7)
    assert [s.signature_id for s in deduplicate([a, b, c], 0.5)] == ["A", "C"]
    assert [s.signature_id for s in deduplicate([a, b, c], 0.8)] == ["A", "B", "C"]
    summary = overlap_summary([a, b, c])
    assert summary["pairs"] == 3
    assert summary["max_jaccard"] == pytest.approx(0.7143)
    assert summary["most_similar_pair"] == ["A", "B"]
    assert overlap_summary([a])["pairs"] == 0


def test_naming_helpers():
    assert slugify("Hypoxia‑induced HIF‑1α / signalling!") == "HYPOXIA_INDUCED_HIF_1_SIGNALLING"
    assert slugify("") == "SIGNATURE"
    assert humanise_gene_set_name("HALLMARK_INTERFERON_GAMMA_RESPONSE") == "Interferon gamma response"
    assert humanise_gene_set_name("Interferon gamma signaling") == "Interferon gamma signaling"


def test_export_dict_is_complete(toy_index):
    sig = build_mechanism_signatures(toy_index, FACET, axis_encoder,
                                     RetrievalParams(min_genes=8, max_genes=12))[0]
    record = sig.to_dict()
    assert record["gene_count"] == len(sig.genes) == len(record["gene_table"])
    assert [row["rank"] for row in record["gene_table"]] == list(range(1, len(sig.genes) + 1))
    assert record["retrieval"]["source_gene_sets"]
    assert record["debate_quality"] == {"stability": None, "entropy": None, "conflict": None,
                                        "agreement": None, "interpretation": "NOT_ASSESSED"}
    assert "support_pool" not in record

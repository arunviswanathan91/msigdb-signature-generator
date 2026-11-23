# Methods

This document specifies what the software computes, how it was validated, and what it does not establish. Version: pipeline 3.0.0, engine 3.0.0, debate 3.0.0.

## 1. Inputs

**Knowledge base.** 34,021 gene sets over 42,302 genes: MSigDB v2025.1.Hs (33,902 sets) and ImmPort (119), keeping sets of 5–500 genes (`kb_builder.py`). Each gene set has one 384-dimensional embedding from `all-MiniLM-L6-v2` of the text *"set name with underscores as spaces, then its first 10 genes"*, L2-normalised. `scripts/build_embeddings.py --verify` reproduces the committed embeddings to cosine 1.0000.

**Mechanisms.** For a research question and optional context (species, tissue, disease, cell type, treatment), one LLM call returns N mechanisms. Each has a title, a one-sentence plain-language description without gene symbols, and 8–12 seed genes (`pipeline.decompose_query`). Mechanisms with duplicate titles are dropped; the count returned is recorded.

**Alias resolution.** Seed names absent from the knowledge base are looked up in NCBI Gene with two E-utilities requests (`esearch` over the Gene Name field restricted to *Homo sapiens* and live records, then `esummary`). A name is accepted only on an exact match, ignoring case and punctuation, with a gene's official symbol or a listed alias (`ncbi_client.py`). If exactly one matching gene is in the knowledge base, the name is resolved. If several are, the name is resolved only when the mechanism's other seeds single one out: the candidate's gene sets must be enriched for sets containing at least two other seeds (hypergeometric p ≤ 1e-4), with at least 5 such sets and at least 5 times as many as the runner-up. Otherwise the name is reported as ambiguous and not used. Every mapping is exported. The use of NCBI's APIs to ground gene nomenclature follows GeneGPT (Jin *et al.*, 2024); here the queries are built deterministically rather than by a model.

## 2. Signature construction

For one mechanism with description *d* and seed set *S* (`signature_engine.py`):

**Enrichment channel.** For every gene set *p*, a one-sided hypergeometric test of |S ∩ p| against a universe of all 42,302 knowledge-base genes, Benjamini–Hochberg corrected across all 34,021 sets. A set is *enriched* if FDR ≤ 0.05 and |S ∩ p| ≥ 2.

**Semantic channel.** Cosine similarity between the embedding of "*title*. *d*" and each gene-set embedding.

**Selection.** Each channel ranks the gene sets. The fused score of a set is the reciprocal-rank fusion 1/(60 + r_sem) + 1/(60 + r_enr), where the second term is present only for enriched sets. Enriched sets are taken first, in order of fused score; the 20 semantically nearest sets may fill any remaining slots; K = 10 sets are kept. The weight w_p of a kept set is its fused score divided by the largest fused score among the kept sets.

**Gene scoring.** score(g) = Σ_p w_p · [g ∈ p] / √|p| over the kept sets. A gene is eligible if at least 2 kept sets contain it. The top 20 eligible genes, ties broken alphabetically, form the signature. If fewer than 10 genes are eligible the support requirement is relaxed to 1; if still fewer than 10, no signature is produced for that mechanism. Seed genes get no exemption: a seed that fails the support requirement is dropped and listed.

**Support (reported as "confidence").** For each gene, the weighted fraction of kept sets containing it, Σ_p w_p [g ∈ p] / Σ_p w_p. A signature's support is the mean over its genes, in [0, 1]. Genes added later that occur in no kept set count as 0.

**Housekeeping genes.** A fixed list plus ribosomal (`RPL*`, `RPS*`, `MRPL*`, `MRPS*` by pattern, excluding kinases such as RPS6KA5) and mitochondrially encoded genes (`gene_filters.py`). Exploratory mode flags them; publication mode removes them before selection and records each removal.

**Sub-modules (optional).** The kept sets are clustered by average linkage on Jaccard distance into the requested number of clusters; each cluster of at least 2 sets yields a signature named after its highest-weight set.

**De-duplication.** A signature whose Jaccard similarity to an earlier one exceeds 0.5 is dropped. Mean and maximum pairwise Jaccard are reported.

All of Section 2 is deterministic.

## 3. Optional layers

**Network expansion.** The pathway API stores, for each gene, its 50 strongest partners with weights normalised to sum to 1 (observed range about 0.01–0.07). A candidate must be a partner, at weight ≥ τ, of at least 2 signature genes; candidates are ranked by summed weight and at most *m* are appended (default 3). τ = 0.05 − 0.04 × strength, so strength 0 admits only the strongest links and 1 admits all stored links. Appended genes are tagged `network`.

**Single-model review.** One model receives each signature with its mechanism and top supporting sets, and returns genes to remove and up to 3 to add, with reasons. Removals not in the signature and additions not in the knowledge base are rejected and listed. A failed call marks its signatures as not reviewed.

**Structured debate** (`debate_engine.py`). Three agents (skeptic: burden of proof on inclusion; advocate: on exclusion; mediator: none) receive the question, mechanism, gene list, and one evidence row per gene: number of other signature genes it is linked to in the network, knowledge-base coverage, GTEx v10 median TPM in the matched tissue if one was given, and flags (isolated, low expression < 1 TPM, housekeeping). All agents are given the same membership rule: a gene belongs if it is a component, direct regulator, ligand, receptor, effector, transcriptional target or established marker of the mechanism in context.

- Round 1: each agent votes keep/remove with a confidence in [0.5, 1] for every gene.
- Later rounds: only genes without unanimous votes are reopened; each agent sees all agents' previous votes and reasons for those genes.
- Stop: unanimity, an agreement threshold, a round with no changed vote, or the round limit (default 3).

For gene *g* and agent *a*, P_a(g) = c if the vote is remove and 1 − c if keep. The pooled probability is p(g) = mean_a P_a(g), agents equally weighted. A gene is voted out if p(g) ≥ 0.6, a strict majority voted remove, and at least 2 agents voted on it. Reported per gene: entropy H₂(p(g)) and conflict 4·Var_a P_a(g). Reported per signature: stability = mean_g |1 − 2p(g)| · (1 − conflict(g)); agreement (fraction of genes with unanimous final votes); Fleiss' κ (undefined when all votes are identical); vote flips between rounds; symbols mentioned that are not in the signature (discarded); and database concordance (flagged genes voted out, flagged genes kept, genes voted out without any flag). Labels: stability ≥ 0.8 high, ≥ 0.5 moderate, below that requires manual review; a signature that was not debated is labelled not assessed.

In the app, voted-out genes are deselected with the reason shown and the user decides; the command-line runner applies them.

## 4. Validation

**Held-out recovery** (`benchmarks/heldout_recovery.py`). 400 gene sets with descriptive names (Hallmark, Reactome, GO-BP, WikiPathways, KEGG; 30–200 genes) were sampled with a fixed seed: 150 for development, 250 for test. For each, the set and every set with Jaccard ≥ 0.5 to it are hidden; the method receives the humanised set name and *n* random member genes, of which a fraction is replaced by random knowledge-base genes; the top 20 non-seed genes returned are scored against the hidden members (precision@20). All defaults in Section 2 were chosen on the development split.

Test split, n = 250, mean precision@20 (results file: `benchmarks/results/heldout_recovery_test.json`):

| Arm | 10 seeds | 10 seeds, 30% wrong | 10 seeds, 50% wrong | 5 seeds |
|---|---|---|---|---|
| v2 pipeline (embed the gene list, sum cosine) | 0.036 | 0.026 | 0.019 | 0.032 |
| semantic only | 0.346 | 0.341 | 0.334 | 0.364 |
| enrichment only | 0.679 | 0.608 | 0.548 | 0.647 |
| **shipped method** | **0.698** | **0.642** | **0.599** | **0.686** |
| pure RRF selection instead of gated | 0.684 | 0.619 | 0.590 | 0.661 |
| without size normalisation | 0.692 | 0.640 | 0.610 | 0.696 |
| with hub penalty (inverse set frequency) | 0.681 | 0.626 | 0.591 | 0.679 |
| K = 5 | 0.671 | 0.614 | 0.599 | 0.684 |
| K = 30 | 0.676 | 0.593 | 0.531 | 0.612 |
| random genes | 0.001 | 0.001 | 0.001 | 0.001 |

Paired one-sided Wilcoxon tests, shipped method against: the v2 pipeline, p ≤ 7e-43 in every condition; semantic only, p ≤ 3e-28; enrichment only, p = 0.11, 5.0e-4, 3.5e-5 and 2.8e-4 across the four conditions.

What this supports: the method recovers related genes far better than the previous pipeline; enrichment carries most of the signal; adding the semantic channel gives no significant gain with ten correct seeds and a significant one when seeds are wrong or few, which is the regime of LLM-proposed seeds.

What did not hold up: size normalisation was chosen on the development split (+0.026) but shows no consistent effect on test (−0.003 on average). A hub penalty lowered recovery slightly on both splits and is off by default. Gene-level specificity scoring (hypergeometric over the kept sets) was tried on development and was worse than weighted support, so it is not included.

**What the benchmark does not show.** Seeds are drawn from the hidden set, and the target is a curated set, so this measures retrieval of related genes, not the biological validity of a signature for a new question. The debate layer has no quantitative validation here: in spot checks it removed deliberately inserted wrong genes (GAPDH, ACTB, KRT5, ALB, INS, MKI67) from a checkpoint signature with all three agents agreeing, but there is no benchmark of its precision. External validation on expression data is needed for both.

**Software tests.** 180 offline tests cover the statistics (hypergeometric against Fisher's exact test, BH, Fleiss' κ by hand calculation), determinism, the debate protocol, every recovery path of the LLM client, both provider backends, alias resolution, exports, and the app driven end to end with mocked services.

**Live verification.** The full pipeline was run against Groq, the pathway API and NCBI Gene. Other LLM providers are implemented from their documentation and were not exercised against live services.

## 5. Provenance and reproducibility

Each export contains the data identity (knowledge-base version and SHA-256 of both files), all parameters, mechanisms with seeds and alias mappings, per-signature supporting sets with statistics, per-gene score, support and origin, the provider and model that answered each LLM call with the sampling controls actually applied, complete debate records, every gene left out with the reason, and package versions. Sections 1 (after the LLM call) and 2 reproduce exactly from an export. LLM calls use temperature 0 and a fixed seed where accepted; models that fix their own sampling are recorded as such and may differ between runs.

## 6. To document before publication

- How `gene_gene.db`, `gene_pathway.db`, `pathway_pathway.db` and the gene evidence table in the Hugging Face dataset were constructed. This repository contains no build code for them.
- The licence status of redistributed gene sets (KEGG and BioCarta subsets of MSigDB carry separate terms).
- An external validation of the signatures and of the debate's decisions.

## References

- Jin Q, Yang Y, Chen Q, Lu Z. GeneGPT: augmenting large language models with domain tools for improved access to biomedical information. *Bioinformatics* 2024;40(2):btae075.
- Liberzon A *et al.* The Molecular Signatures Database hallmark gene set collection. *Cell Systems* 2015;1(6):417–425.
- Cormack GV, Clarke CLA, Büttcher S. Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. *SIGIR* 2009.
- Benjamini Y, Hochberg Y. Controlling the false discovery rate. *J R Stat Soc B* 1995;57(1):289–300.
- Fleiss JL. Measuring nominal scale agreement among many raters. *Psychol Bull* 1971;76(5):378–382.

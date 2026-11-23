"""
Held-out gene recovery benchmark
================================

Question: given a mechanism's name and a handful of its genes, how well does
each retrieval strategy recover the *other* genes of that mechanism from the
rest of the knowledge base - and how does that degrade when the seed genes are
few or partly wrong, as LLM-proposed seeds are?

Protocol (deterministic, no LLM, no network beyond the embedding model)
----------------------------------------------------------------------
1. Sample 400 curated gene sets with descriptive names (Hallmark, Reactome,
   GO-BP, WikiPathways, KEGG; 30-200 genes) with a fixed random seed.
   The first 150 are the DEVELOPMENT split, the remaining 250 the TEST split.
   Every default in ``RetrievalParams`` was chosen on development only.
2. For each set, hide it AND every near-duplicate (Jaccard >= 0.5) from the KB.
3. Give the method the humanised set name plus N random member genes; with
   seed noise f, a fraction f of those genes is replaced by random KB genes.
4. Score the top 20 non-seed genes it returns against the held-out members
   (precision@20).

Arms
----
  legacy        the pre-3.0 pipeline: embed the gene list, sum cosine similarity
  semantic      description embedding only
  enrichment    seed hypergeometric enrichment only
  hybrid        the shipped method (gated selection, both channels)
  ablations     pure RRF selection; no size normalisation; hub penalty on;
                top_k 5 and 30
  random        20 random KB genes

Usage
-----
    python benchmarks/heldout_recovery.py                 # test split, all conditions
    python benchmarks/heldout_recovery.py --split dev
    python benchmarks/heldout_recovery.py --json results.json
"""

import argparse
import gzip
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_engine import (  # noqa: E402
    ENGINE_VERSION, KnowledgeIndex, RetrievalParams, humanise_gene_set_name,
    retrieve_gene_sets, score_genes, seed_enrichment,
)

TOP_N = 20
SAMPLE = 400
DEV_SIZE = 150
PREFIXES = ('HALLMARK_', 'REACTOME_', 'GOBP_', 'WP_', 'KEGG_')
CONDITIONS = ((0.0, 10), (0.3, 10), (0.5, 10), (0.0, 5))     # (seed noise, number of seeds)

ARMS = {
    'semantic only': RetrievalParams(use_enrichment=False),
    'enrichment only': RetrievalParams(use_semantic=False),
    'hybrid (shipped)': RetrievalParams(),
    '  - pure RRF selection': RetrievalParams(selection="fused"),
    '  - no size normalisation': RetrievalParams(size_normalise=False),
    '  + hub penalty (idf)': RetrievalParams(idf_power=1.0),
    '  - top_k = 5': RetrievalParams(top_k=5),
    '  - top_k = 30': RetrievalParams(top_k=30),
}


def near_duplicate_mask(index, target_idx, threshold=0.5):
    target = index.gene_set(target_idx)
    shared = np.zeros(index.n_pathways, dtype=np.int64)
    for gene in target:
        shared[index.gene_to_idx[gene]] += 1
    union = index.sizes + len(target) - shared
    mask = shared / union >= threshold
    mask[target_idx] = True
    return mask


def top_non_seed(ranked, seeds, n=TOP_N):
    seed_set = set(seeds)
    return [g for g in ranked if g not in seed_set][:n]


def legacy_arm(index, embedding, seeds, exclude):
    """v2 behaviour: top-50 gene sets by cosine to the embedded gene list, summed."""
    sims = index.matrix @ embedding
    sims[exclude] = -np.inf
    scores = {}
    for i in np.argsort(-sims)[:50]:
        for gene in index.pathways[index.ids[i]]:
            scores[gene] = scores.get(gene, 0.0) + float(sims[i])
    return top_non_seed(sorted(scores, key=lambda g: -scores[g]), seeds)


def engine_arm(index, embedding, seeds, exclude, params, enrichment=None):
    hits, _ = retrieve_gene_sets(index, embedding if params.use_semantic else None, seeds,
                                 params, exclude=exclude, enrichment=enrichment)
    if not hits:
        return []
    score, support, _ = score_genes(index, hits, params)
    need = min(params.min_support, len(hits))
    ranked = [g for g in sorted(score, key=lambda g: (-score[g], g)) if support[g] >= need]
    return top_non_seed(ranked, seeds)


def split_indices(index, split):
    rng = np.random.default_rng(0)
    eligible = [i for i, p in enumerate(index.ids)
                if p.startswith(PREFIXES) and 30 <= index.sizes[i] <= 200]
    sample = rng.choice(eligible, size=SAMPLE, replace=False)
    return sample[:DEV_SIZE] if split == "dev" else sample[DEV_SIZE:]


def make_cases(index, chosen, noise, n_seeds, frequent):
    """Deterministic per condition: (set index, seeds given, held-out members)."""
    rng = np.random.default_rng(1000 + int(noise * 100) + n_seeds)
    cases = []
    for idx in chosen:
        members = sorted(index.gene_set(idx))
        true_seeds = list(rng.choice(members, size=n_seeds, replace=False))
        n_wrong = int(round(noise * n_seeds))
        seeds = true_seeds[:n_seeds - n_wrong] + list(rng.choice(frequent, size=n_wrong, replace=False))
        cases.append((int(idx), seeds, set(members) - set(true_seeds)))
    return cases


def bootstrap_ci(values, rng, n=2000):
    values = np.asarray(values)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--json", default=None, help="write full results to this file")
    args = parser.parse_args()

    with gzip.open('data/knowledge_base.json.gz', 'rt') as handle:
        pathways = json.load(handle)['pathways']
    with open('data/pathway_embeddings.pkl', 'rb') as handle:
        embeddings = pickle.load(handle)['embeddings']
    index = KnowledgeIndex(pathways, embeddings)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    chosen = split_indices(index, args.split)
    frequent = sorted(g for g, v in index.gene_to_idx.items() if len(v) >= 5)
    all_genes = sorted(index.gene_to_idx)
    descriptions = model.encode([humanise_gene_set_name(index.ids[i]) for i in chosen],
                                normalize_embeddings=True, batch_size=64)
    boot = np.random.default_rng(7)
    results = {'engine_version': ENGINE_VERSION, 'split': args.split, 'n_sets': int(len(chosen)),
               'metric': f'precision@{TOP_N} over non-seed genes', 'conditions': []}

    for noise, n_seeds in CONDITIONS:
        cases = make_cases(index, chosen, noise, n_seeds, frequent)
        gene_lists = model.encode([' OR '.join(s) for _, s, _ in cases],
                                  normalize_embeddings=True, batch_size=64)
        rng = np.random.default_rng(99)
        scores = {name: [] for name in ['legacy (v2)', *ARMS, 'random genes']}
        for n, (idx, seeds, held) in enumerate(cases):
            exclude = near_duplicate_mask(index, idx)
            known, _ = index.normalise_symbols(seeds)
            enrichment = seed_enrichment(index, known)

            def record(name, genes):
                scores[name].append(len(set(genes) & held) / TOP_N)

            record('legacy (v2)', legacy_arm(index, gene_lists[n], seeds, exclude))
            for name, params in ARMS.items():
                record(name, engine_arm(index, descriptions[n], known, exclude, params, enrichment))
            record('random genes', list(rng.choice(all_genes, size=TOP_N, replace=False)))

        from scipy.stats import wilcoxon
        label = f"{n_seeds} seeds, {int(noise * 100)}% wrong"
        print(f"\n{args.split.upper()} split, n={len(cases)} gene sets - {label}")
        print(f"{'arm':28s} {'mean':>6s} {'median':>7s}   95% CI")
        condition = {'seed_noise': noise, 'n_seeds': n_seeds, 'arms': {}}
        for name, values in scores.items():
            low, high = bootstrap_ci(values, boot)
            condition['arms'][name] = {'mean': float(np.mean(values)), 'median': float(np.median(values)),
                                       'ci95': [low, high]}
            print(f"{name:28s} {np.mean(values):6.3f} {np.median(values):7.3f}   [{low:.3f}, {high:.3f}]")
        tests = {}
        for baseline in ('legacy (v2)', 'semantic only', 'enrichment only', '  - pure RRF selection'):
            diff = np.asarray(scores['hybrid (shipped)']) - np.asarray(scores[baseline])
            p = float(wilcoxon(diff, alternative='greater', zero_method='zsplit').pvalue) if diff.any() else 1.0
            tests[baseline] = p
            print(f"  hybrid > {baseline.strip():24s} paired Wilcoxon p = {p:.2e}")
        condition['hybrid_vs'] = tests
        results['conditions'].append(condition)

    if args.json:
        with open(args.json, 'w') as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.json}")
    return results


if __name__ == '__main__':
    main()

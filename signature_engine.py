"""
Signature engine: seed-anchored hybrid retrieval and evidence-weighted scoring
==============================================================================

Pure Python / NumPy. No Streamlit, no network. Everything here is
deterministic, so a signature can be regenerated exactly from its exported
provenance.

METHOD
------
For one mechanism the LLM supplies a natural-language *description* and a
short list of *seed genes*. The LLM proposes; the knowledge base decides.

1. Two independent relevance channels rank every gene set in the KB:

   * enrichment - one-sided hypergeometric test of the seed genes against the
                  gene set, Benjamini-Hochberg corrected across the whole KB;
   * semantic   - cosine similarity between the description and the
                  precomputed gene-set embedding.

   A gene set is *enriched* if FDR <= ``fdr_threshold`` with at least
   ``min_overlap`` seeds. Selection is gated: enriched sets are taken first,
   ordered by the reciprocal-rank fusion (RRF) of both channels; sets that are
   only semantically similar (the ``semantic_pool`` nearest) fill whatever
   slots remain, up to ``top_k``. Semantic similarity therefore re-ranks
   statistical evidence and backs it up when seeds are few or wrong, but never
   outvotes it.

2. Genes are scored by relevance-weighted support across the kept sets::

       score(g) = sum_p  w_p * [g in p] / sqrt(|p|)

   ``w_p`` is the fused relevance of set p and ``1/sqrt(|p|)`` stops large
   sets from dominating. An optional hub penalty (``idf_power`` > 0 multiplies
   by the inverse gene-set frequency of g) is available but off by default: it
   did not improve held-out recovery.

3. A gene is admitted only if it is supported by >= ``min_support`` kept sets.
   Seed genes get no exemption: a seed the KB does not support is dropped and
   reported, which is what makes the result database-grounded rather than an
   echo of the LLM.

4. Signature confidence is the mean weighted support of its genes - the
   relevance-weighted fraction of kept gene sets that contain each gene. It is
   bounded in [0, 1] and does not saturate.

5. Optional sub-modules: the kept gene sets are clustered on Jaccard distance
   (average linkage) and one signature is built per cluster, named after the
   cluster's most relevant gene set.

Defaults (``RetrievalParams``) were chosen on a development split of the
held-out recovery benchmark and reported on a disjoint test split - see
``benchmarks/heldout_recovery.py`` and ``docs/METHODS.md``.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gene_filters import is_housekeeping

ENGINE_VERSION = "3.0.0"
RRF_K = 60


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class RetrievalParams:
    """Every knob that affects the output. Exported verbatim with results."""
    top_k: int = 10               # gene sets kept per mechanism
    semantic_pool: int = 20       # nearest semantic-only sets allowed to fill empty slots
    fdr_threshold: float = 0.05   # BH-FDR for the enrichment channel
    min_overlap: int = 2          # seeds a set must contain to count as enriched
    min_support: int = 2          # kept sets that must contain a gene
    min_genes: int = 10
    max_genes: int = 20
    use_semantic: bool = True
    use_enrichment: bool = True
    selection: str = "gated"      # "gated": enriched sets first, semantic sets fill; "fused": pure RRF
    idf_power: float = 0.0        # > 0 penalises genes found in many gene sets (off by default)
    size_normalise: bool = True
    drop_housekeeping: bool = False
    dedup_jaccard: float = 0.5    # signatures more similar than this are merged away

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


@dataclass
class PathwayHit:
    pathway_id: str
    size: int
    semantic: float            # cosine similarity
    overlap: int               # seed genes contained
    p_value: float
    fdr: float
    weight: float              # fused relevance in (0, 1]
    seeds_hit: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'pathway': self.pathway_id, 'size': self.size,
            'semantic_similarity': round(self.semantic, 4),
            'seed_overlap': self.overlap,
            'p_value': float(f"{self.p_value:.3e}"), 'fdr': float(f"{self.fdr:.3e}"),
            'weight': round(self.weight, 4), 'seeds_hit': self.seeds_hit,
        }


@dataclass
class GeneSignature:
    """A gene signature plus the evidence that produced it."""
    signature_id: str
    signature_name: str
    genes: List[str]                       # ranked, best first
    facet: str
    mechanism: str
    confidence: float
    description: str = ""
    gene_scores: Dict[str, float] = field(default_factory=dict)
    gene_support: Dict[str, int] = field(default_factory=dict)
    gene_sources: Dict[str, str] = field(default_factory=dict)   # knowledge_base | network | llm | user
    source_pathways: List[Dict] = field(default_factory=list)
    seed_genes: List[str] = field(default_factory=list)
    seeds_unknown: List[str] = field(default_factory=list)       # not in the KB at all
    seeds_unsupported: List[str] = field(default_factory=list)   # in the KB, not supported here
    seed_recovery: float = 0.0             # fraction of valid seeds that survived
    novel_fraction: float = 0.0            # fraction of genes the LLM did not propose
    retrieval_mode: str = "hybrid"
    housekeeping_flagged: List[str] = field(default_factory=list)
    removed_genes: Dict[str, str] = field(default_factory=dict)  # gene -> why it was removed
    # Weighted support of every gene found in the retrieved gene sets, so the
    # support of an edited gene list can be recomputed exactly. Not exported.
    support_pool: Dict[str, float] = field(default_factory=dict, repr=False)

    dam_expanded: bool = False
    llm2_verified: bool = False
    debate_verified: bool = False

    # None means "not assessed by debate" - never a default of 1.0.
    stability_score: Optional[float] = None
    debate_entropy: Optional[float] = None
    debate_conflict: Optional[float] = None
    debate_agreement: Optional[float] = None

    def support(self) -> float:
        """
        Mean weighted support of the CURRENT gene list. Genes added later
        (network, LLM, user) that occur in none of the retrieved gene sets
        count as 0, so additions without knowledge-base backing lower it.
        """
        if not self.genes:
            return 0.0
        if not self.support_pool:
            return self.confidence
        return float(np.mean([self.support_pool.get(g, 0.0) for g in self.genes]))

    def current_seed_recovery(self) -> float:
        if not self.seed_genes:
            return 0.0
        present = set(self.genes)
        return sum(1 for s in self.seed_genes if s in present) / len(self.seed_genes)

    def current_novel_fraction(self) -> float:
        if not self.genes:
            return 0.0
        seeds = set(self.seed_genes)
        return sum(1 for g in self.genes if g not in seeds) / len(self.genes)

    def quality_label(self) -> str:
        if self.stability_score is None:
            return "NOT_ASSESSED"
        if self.stability_score >= 0.8:
            return "HIGH_CONFIDENCE"
        if self.stability_score >= 0.5:
            return "MODERATE_CONFIDENCE"
        return "REQUIRES_MANUAL_REVIEW"

    def gene_set_hash(self) -> str:
        return hashlib.sha256('|'.join(sorted(self.genes)).encode()).hexdigest()[:16]

    def to_dict(self) -> Dict:
        return {
            'signature_id': self.signature_id,
            'signature_name': self.signature_name,
            'facet': self.facet,
            'mechanism': self.mechanism,
            'description': self.description,
            'genes': list(self.genes),
            'gene_count': len(self.genes),
            'gene_set_hash': self.gene_set_hash(),
            'gene_table': [
                {
                    'gene': g, 'rank': i + 1,
                    'score': round(self.gene_scores[g], 5) if g in self.gene_scores else None,
                    'supporting_gene_sets': self.gene_support.get(g),
                    'weighted_support': round(self.support_pool.get(g, 0.0), 4),
                    'source': self.gene_sources.get(g, 'knowledge_base'),
                    'llm_seed': g in self.seed_genes,
                }
                for i, g in enumerate(self.genes)
            ],
            # support of the exported gene list; *_at_build is before any edits
            'confidence': round(self.support(), 4),
            'confidence_at_build': round(self.confidence, 4),
            'retrieval': {
                'mode': self.retrieval_mode,
                'seed_genes': self.seed_genes,
                'seeds_unknown_to_kb': self.seeds_unknown,
                'seeds_not_supported': self.seeds_unsupported,
                'seed_recovery': round(self.current_seed_recovery(), 4),
                'novel_fraction': round(self.current_novel_fraction(), 4),
                'source_gene_sets': self.source_pathways,
            },
            'housekeeping_flagged': self.housekeeping_flagged,
            'removed_genes': self.removed_genes,
            'dam_expanded': self.dam_expanded,
            'llm2_verified': self.llm2_verified,
            'debate_verified': self.debate_verified,
            'debate_quality': {
                'stability': self.stability_score,
                'entropy': self.debate_entropy,
                'conflict': self.debate_conflict,
                'agreement': self.debate_agreement,
                'interpretation': self.quality_label(),
            },
        }


# ============================================================
# KNOWLEDGE INDEX
# ============================================================

class KnowledgeIndex:
    """Gene-set knowledge base with an inverted index and aligned embeddings."""

    def __init__(self, pathways: Dict[str, List[str]],
                 embeddings: Optional[Dict[str, np.ndarray]] = None):
        # json.load creates a separate str object for every gene occurrence
        # (~3M for the full KB). Interning them here lets the caller's copy be
        # freed and cuts resident memory by well over 100 MB.
        canon: Dict[str, str] = {}
        self.pathways: Dict[str, List[str]] = {
            pid: list(dict.fromkeys(canon.setdefault(g, g) for g in genes))
            for pid, genes in pathways.items()
        }
        self.ids: List[str] = list(self.pathways)
        self.sizes = np.array([len(self.pathways[p]) for p in self.ids], dtype=np.int64)
        self.n_pathways = len(self.ids)

        inverted: Dict[str, List[int]] = {}
        for idx, pid in enumerate(self.ids):
            for gene in self.pathways[pid]:
                inverted.setdefault(gene, []).append(idx)
        self.gene_to_idx: Dict[str, np.ndarray] = {
            g: np.asarray(v, dtype=np.int32) for g, v in inverted.items()
        }
        self.n_genes = len(self.gene_to_idx)
        self._symbol_map = {g.upper(): g for g in self.gene_to_idx}

        log_n = math.log(1 + self.n_pathways)
        self.idf: Dict[str, float] = {
            g: math.log(1 + self.n_pathways / len(v)) / log_n
            for g, v in self.gene_to_idx.items()
        }

        self.matrix: Optional[np.ndarray] = None
        if embeddings:
            missing = [p for p in self.ids if p not in embeddings]
            if missing:
                raise ValueError(
                    f"{len(missing)} gene sets have no embedding (e.g. {missing[0]}). "
                    "The embeddings file does not match the knowledge base."
                )
            self.matrix = np.stack([embeddings[p] for p in self.ids]).astype(np.float32)

    def normalise_symbols(self, symbols: Sequence[str]) -> Tuple[List[str], List[str]]:
        """Map symbols onto KB spelling. Returns (known, unknown), de-duplicated."""
        known, unknown, seen = [], [], set()
        for raw in symbols:
            if not isinstance(raw, str):
                continue
            symbol = raw.strip()
            if not symbol:
                continue
            canonical = self._symbol_map.get(symbol.upper())
            if canonical is None:
                if symbol.upper() not in seen:
                    unknown.append(symbol)
                    seen.add(symbol.upper())
            elif canonical not in seen:
                known.append(canonical)
                seen.add(canonical)
        return known, unknown

    def has_gene(self, symbol: str) -> bool:
        return symbol.upper() in self._symbol_map

    def gene_set(self, i: int) -> frozenset:
        """Members of the i-th gene set."""
        return frozenset(self.pathways[self.ids[i]])


# ============================================================
# STATISTICS
# ============================================================

def benjamini_hochberg(p_values: np.ndarray, n_tests: int) -> np.ndarray:
    """BH-adjusted p-values for a subset of ``n_tests`` hypotheses."""
    if p_values.size == 0:
        return p_values
    order = np.argsort(p_values)
    ranked = p_values[order] * n_tests / np.arange(1, p_values.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted


def seed_enrichment(index: KnowledgeIndex, seeds: Sequence[str]
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hypergeometric over-representation of ``seeds`` in every gene set.

    Returns (overlap_counts, p_values, fdr) aligned to ``index.ids``. Sets with
    no overlap get p = fdr = 1. BH is corrected for all gene sets in the KB,
    not only those with overlap.
    """
    from scipy.stats import hypergeom

    counts = np.zeros(index.n_pathways, dtype=np.int64)
    for gene in seeds:
        counts[index.gene_to_idx[gene]] += 1

    p_values = np.ones(index.n_pathways)
    fdr = np.ones(index.n_pathways)
    hit = np.nonzero(counts)[0]
    if hit.size:
        p_hit = hypergeom.sf(counts[hit] - 1, index.n_genes, index.sizes[hit], len(seeds))
        p_values[hit] = p_hit
        fdr[hit] = benjamini_hochberg(p_hit, index.n_pathways)
    return counts, p_values, fdr


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    union = len(a | b)
    return len(a & b) / union if union else 0.0


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_gene_sets(
    index: KnowledgeIndex,
    query_embedding: Optional[np.ndarray],
    seeds: Sequence[str],
    params: RetrievalParams,
    exclude: Optional[np.ndarray] = None,
    enrichment: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[List[PathwayHit], str]:
    """
    Rank gene sets for one mechanism. Returns (hits, mode) where mode is one of
    'hybrid', 'enrichment', 'semantic' depending on which channels contributed.

    ``exclude`` is a boolean mask of gene sets to ignore (used by benchmarks).
    ``enrichment`` is a precomputed ``seed_enrichment`` result for these seeds.
    """
    n = index.n_pathways
    allowed = np.ones(n, dtype=bool) if exclude is None else ~exclude

    use_sem = params.use_semantic and query_embedding is not None and index.matrix is not None
    use_enr = params.use_enrichment and len(seeds) >= params.min_overlap

    sims = np.zeros(n)
    sem_rank = np.full(n, np.inf)
    sem_pool = np.zeros(n, dtype=bool)
    if use_sem:
        sims = (index.matrix @ np.asarray(query_embedding, dtype=np.float32)).astype(float)
        masked = np.where(allowed, sims, -np.inf)
        order = np.argsort(-masked, kind='stable')
        sem_rank[order] = np.arange(1, n + 1)
        sem_pool[order[:params.semantic_pool]] = True
        sem_pool &= allowed

    counts = np.zeros(n, dtype=np.int64)
    p_values = np.ones(n)
    fdr = np.ones(n)
    enr_rank = np.full(n, np.inf)
    enr_pool = np.zeros(n, dtype=bool)
    if use_enr:
        counts, p_values, fdr = enrichment if enrichment is not None else seed_enrichment(index, seeds)
        enr_pool = (fdr <= params.fdr_threshold) & (counts >= params.min_overlap) & allowed
        hits = np.nonzero(enr_pool)[0]
        if hits.size:
            # Rank by p-value; break ties toward the semantically closer set.
            order = hits[np.lexsort((-sims[hits], p_values[hits]))]
            enr_rank[order] = np.arange(1, order.size + 1)

    pool = sem_pool | enr_pool
    if not pool.any():
        return [], 'none'

    fused = np.zeros(n)
    if use_sem:
        fused += 1.0 / (RRF_K + sem_rank)
    if enr_pool.any():
        fused += np.where(enr_pool, 1.0 / (RRF_K + enr_rank), 0.0)

    candidates = np.nonzero(pool)[0]
    if params.selection == "gated":
        # Statistically enriched sets always outrank sets admitted on semantic
        # similarity alone; the latter only fill slots the former leave empty.
        # Within each tier the fused score decides.
        tier = np.where(enr_pool[candidates], 0, 1)
        order = candidates[np.lexsort((-fused[candidates], tier))][:params.top_k]
    else:
        order = candidates[np.argsort(-fused[candidates], kind='stable')][:params.top_k]
    top = fused[order].max()

    seed_set = set(seeds)
    hits_out = [
        PathwayHit(
            pathway_id=index.ids[i], size=int(index.sizes[i]),
            semantic=float(sims[i]), overlap=int(counts[i]),
            p_value=float(p_values[i]), fdr=float(fdr[i]),
            weight=float(fused[i] / top),
            seeds_hit=sorted(seed_set & index.gene_set(i)),
        )
        for i in order
    ]
    if use_sem and enr_pool.any():
        mode = 'hybrid'
    elif enr_pool.any():
        mode = 'enrichment'
    else:
        mode = 'semantic'
    return hits_out, mode


# ============================================================
# GENE SCORING
# ============================================================

def score_genes(index: KnowledgeIndex, hits: Sequence[PathwayHit], params: RetrievalParams
                ) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, float]]:
    """
    Returns (score, support_count, weighted_support) per gene.

    weighted_support is the relevance-weighted fraction of kept gene sets that
    contain the gene, in [0, 1].
    """
    raw: Dict[str, float] = {}
    support: Dict[str, int] = {}
    weighted: Dict[str, float] = {}
    total_weight = sum(h.weight for h in hits) or 1.0

    for hit in hits:
        contribution = hit.weight / math.sqrt(hit.size) if params.size_normalise else hit.weight
        for gene in index.pathways[hit.pathway_id]:
            raw[gene] = raw.get(gene, 0.0) + contribution
            support[gene] = support.get(gene, 0) + 1
            weighted[gene] = weighted.get(gene, 0.0) + hit.weight

    if params.idf_power:
        score = {g: v * (index.idf[g] ** params.idf_power) for g, v in raw.items()}
    else:
        score = raw
    weighted = {g: v / total_weight for g, v in weighted.items()}
    return score, support, weighted


def slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r'[^A-Z0-9]+', '_', text.upper()).strip('_')
    return slug[:max_len].rstrip('_') or 'SIGNATURE'


def humanise_gene_set_name(name: str) -> str:
    """HALLMARK_INTERFERON_GAMMA_RESPONSE -> 'Interferon gamma response'."""
    stripped = re.sub(
        r'^(HALLMARK|REACTOME|GOBP|GOMF|GOCC|KEGG_MEDICUS_REFERENCE|KEGG_MEDICUS|KEGG|WP|PID|BIOCARTA|HP)_',
        '', name)
    text = stripped.replace('_', ' ').strip()
    return text[:1].upper() + text[1:].lower() if text.isupper() else text


def build_signature(
    index: KnowledgeIndex,
    hits: Sequence[PathwayHit],
    *,
    signature_id: str,
    facet: str,
    mechanism: str,
    description: str,
    seeds: Sequence[str],
    seeds_unknown: Sequence[str],
    mode: str,
    params: RetrievalParams,
) -> Optional[GeneSignature]:
    """Turn ranked gene sets into one ranked, evidence-annotated signature."""
    if not hits:
        return None

    score, support, weighted = score_genes(index, hits, params)
    ranked = sorted(score, key=lambda g: (-score[g], g))   # ties broken alphabetically

    # With a single retrieved set, 2-set support is impossible by construction.
    required = min(params.min_support, len(hits))

    def select(candidates: List[str]) -> List[str]:
        chosen = [g for g in candidates if support[g] >= required][:params.max_genes]
        if len(chosen) < params.min_genes:
            # Not enough multiply-supported genes: relax to single support rather
            # than return nothing, and let the lower support score say so.
            taken = set(chosen)
            for g in candidates:
                if len(chosen) >= params.min_genes:
                    break
                if g not in taken:
                    chosen.append(g)
                    taken.add(g)
        return chosen

    unfiltered = select(ranked)
    housekeeping = [g for g in unfiltered if is_housekeeping(g)]
    removed: Dict[str, str] = {}
    if params.drop_housekeeping and housekeeping:
        genes = select([g for g in ranked if not is_housekeeping(g)])
        removed = {g: "housekeeping filter (publication mode)" for g in housekeeping}
    else:
        genes = unfiltered
    if len(genes) < params.min_genes:
        return None

    gene_set = set(genes)
    seed_list = list(seeds)
    kept_seeds = [s for s in seed_list if s in gene_set]
    confidence = float(np.mean([weighted[g] for g in genes]))

    return GeneSignature(
        signature_id=signature_id,
        signature_name=f"{facet} - {mechanism}",
        genes=genes,
        facet=facet,
        mechanism=mechanism,
        description=description,
        confidence=confidence,
        gene_scores={g: float(score[g]) for g in genes},
        gene_support={g: int(support[g]) for g in genes},
        gene_sources={g: 'knowledge_base' for g in genes},
        source_pathways=[h.to_dict() for h in hits],
        seed_genes=seed_list,
        seeds_unknown=list(seeds_unknown),
        seeds_unsupported=[s for s in seed_list if s not in gene_set and s not in removed],
        seed_recovery=len(kept_seeds) / len(seed_list) if seed_list else 0.0,
        novel_fraction=1.0 - len(kept_seeds) / len(genes),
        retrieval_mode=mode,
        housekeeping_flagged=housekeeping,
        removed_genes=removed,
        support_pool={g: float(v) for g, v in weighted.items()},
    )


# ============================================================
# SUB-MODULES
# ============================================================

def cluster_gene_sets(index: KnowledgeIndex, hits: Sequence[PathwayHit], n_clusters: int
                      ) -> List[List[PathwayHit]]:
    """
    Split retrieved gene sets into ``n_clusters`` groups of overlapping sets
    (average-linkage clustering on Jaccard distance). Deterministic. Clusters
    are returned most-relevant first.
    """
    if n_clusters <= 1 or len(hits) < 2:
        return [list(hits)]
    n_clusters = min(n_clusters, len(hits))

    from scipy.cluster.hierarchy import fcluster, linkage

    sets = [frozenset(index.pathways[h.pathway_id]) for h in hits]
    condensed = [
        1.0 - (len(sets[i] & sets[j]) / len(sets[i] | sets[j]))
        for i in range(len(sets)) for j in range(i + 1, len(sets))
    ]
    labels = fcluster(linkage(np.asarray(condensed), method='average'),
                      t=n_clusters, criterion='maxclust')

    groups: Dict[int, List[PathwayHit]] = {}
    for hit, label in zip(hits, labels):
        groups.setdefault(int(label), []).append(hit)
    return sorted(groups.values(), key=lambda grp: -sum(h.weight for h in grp))


# ============================================================
# MECHANISM -> SIGNATURES
# ============================================================

def build_mechanism_signatures(
    index: KnowledgeIndex,
    facet: Dict,
    encode: Optional[Callable[[str], np.ndarray]],
    params: RetrievalParams,
    n_variants: int = 1,
    id_prefix: str = "SIG",
) -> List[GeneSignature]:
    """
    Build ``n_variants`` signatures for one LLM-proposed mechanism.

    ``facet`` needs 'facet_name'; 'description' and 'seed_genes' are optional
    but both channels degrade without them. ``encode`` maps text to a
    unit-norm embedding (None disables the semantic channel).
    """
    name = str(facet.get('facet_name') or 'Unnamed mechanism').strip()
    description = str(facet.get('description') or '').strip()
    seeds, unknown = index.normalise_symbols(facet.get('seed_genes') or [])

    query_embedding = None
    if encode is not None and params.use_semantic:
        text = f"{name}. {description}" if description else name
        query_embedding = np.asarray(encode(text), dtype=np.float32)
        norm = float(np.linalg.norm(query_embedding))
        if norm > 0:
            query_embedding = query_embedding / norm

    hits, mode = retrieve_gene_sets(index, query_embedding, seeds, params)
    if not hits:
        return []

    common = dict(facet=name, description=description, seeds=seeds,
                  seeds_unknown=unknown, mode=mode, params=params)
    signatures: List[GeneSignature] = []

    core = build_signature(index, hits, signature_id=f"{id_prefix}_{slugify(name)}__CORE",
                           mechanism="Core", **common)
    if core:
        signatures.append(core)

    if n_variants > 1 and len(hits) >= 4:
        # Clusters of overlapping gene sets = distinct facets of the mechanism.
        # A lone gene set is not a module, so singleton clusters are skipped.
        for k, group in enumerate(cluster_gene_sets(index, hits, n_variants), start=1):
            if len(signatures) >= n_variants:
                break
            if len(group) < 2:
                continue
            label = humanise_gene_set_name(max(group, key=lambda h: h.weight).pathway_id)
            sub = build_signature(
                index, group, signature_id=f"{id_prefix}_{slugify(name)}__M{k}",
                mechanism=f"Sub-module {k}: {label}", **common)
            if sub and all(jaccard(sub.genes, s.genes) <= params.dedup_jaccard for s in signatures):
                signatures.append(sub)

    return signatures


def deduplicate(signatures: Sequence[GeneSignature], max_jaccard: float) -> List[GeneSignature]:
    """Drop later signatures that overlap an earlier one above ``max_jaccard``."""
    kept: List[GeneSignature] = []
    for sig in signatures:
        if all(jaccard(sig.genes, other.genes) <= max_jaccard for other in kept):
            kept.append(sig)
    return kept


def overlap_summary(signatures: Sequence[GeneSignature]) -> Dict:
    """Pairwise Jaccard statistics - reported so 'distinct' is a measured claim."""
    values, worst = [], None
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            value = jaccard(signatures[i].genes, signatures[j].genes)
            values.append(value)
            if worst is None or value > worst[0]:
                worst = (value, signatures[i].signature_id, signatures[j].signature_id)
    if not values:
        return {'pairs': 0, 'mean_jaccard': 0.0, 'max_jaccard': 0.0, 'most_similar_pair': None}
    return {
        'pairs': len(values),
        'mean_jaccard': round(float(np.mean(values)), 4),
        'max_jaccard': round(float(worst[0]), 4),
        'most_similar_pair': [worst[1], worst[2]],
    }

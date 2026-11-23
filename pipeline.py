"""
Pipeline steps shared by the Streamlit app and the command-line runner
======================================================================

Resources  - ``load_resources``: knowledge base + embeddings + their identity.
Layer 1    - ``decompose_query``: research question -> N mechanisms, each with
             a natural-language description and seed genes;
             ``resolve_seed_aliases``: gene aliases -> official symbols (NCBI).
Layer 3    - ``apply_network_expansion``: append network-supported genes.
Layer 4    - ``verify_signatures``: single-model review;
             ``apply_review``, ``apply_debate_result``: write outcomes back.

No UI code lives here, so the app and ``run_pipeline.py`` cannot drift apart.
LLM steps go through ``llm_client.chat_json`` and return provenance alongside
results. Failures are returned as failures - never as empty successes.
"""

from __future__ import annotations

import gzip
import json
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

from exporters import file_sha256
from llm_client import LLMError, LLMResult, chat_json
from signature_engine import (
    GeneSignature, KnowledgeIndex, RetrievalParams, build_mechanism_signatures, deduplicate,
)

PIPELINE_VERSION = "3.0.0"
MAX_MECHANISMS = 30
MAX_ADDITIONS_PER_SIGNATURE = 3


DEFAULT_KB_PATH = "data/knowledge_base.json.gz"
DEFAULT_EMBEDDINGS_PATH = "data/pathway_embeddings.pkl"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_resources(kb_path: str = DEFAULT_KB_PATH,
                   embeddings_path: str = DEFAULT_EMBEDDINGS_PATH) -> Dict:
    """
    Load the knowledge base and its embeddings.

    Returns {index, kb_metadata, embedding_model, kb_sha256, embeddings_sha256}.
    The hashes go into every export so a result can be tied to the exact data
    that produced it. Raises FileNotFoundError / ValueError on bad inputs.
    """
    with gzip.open(kb_path, 'rt', encoding='utf-8') as handle:
        kb = json.load(handle)
    with open(embeddings_path, 'rb') as handle:
        package = pickle.load(handle)
    if not isinstance(package, dict) or 'embeddings' not in package:
        found = sorted(package) if isinstance(package, dict) else type(package).__name__
        raise ValueError(f"{embeddings_path} has no 'embeddings' key (found {found})")
    return {
        'index': KnowledgeIndex(kb.get('pathways', {}), package['embeddings']),
        'kb_metadata': kb.get('metadata', {}),
        'embedding_model': package.get('model_name', DEFAULT_EMBEDDING_MODEL),
        'kb_sha256': file_sha256(kb_path),
        'embeddings_sha256': file_sha256(embeddings_path),
    }


def context_lines(context: Optional[Dict]) -> str:
    if not context:
        return "BIOLOGICAL CONTEXT: not specified"
    fields = [('species', 'Species'), ('tissue', 'Tissue'), ('disease', 'Disease'),
              ('cell_type', 'Cell type'), ('treatment', 'Treatment')]
    lines = [f"- {label}: {context[key]}" for key, label in fields if context.get(key)]
    return "BIOLOGICAL CONTEXT:\n" + "\n".join(lines) if lines else "BIOLOGICAL CONTEXT: not specified"


# ============================================================
# LAYER 1
# ============================================================

DECOMPOSE_PROMPT = """Decompose this research question into EXACTLY {n} specific, mutually distinct molecular mechanisms.

Research question: "{query}"
{context}

For each mechanism provide:
- "facet_name": a short specific title (max 8 words)
- "description": ONE sentence in plain language naming the processes, pathways and cell types involved. Do NOT put gene symbols here - it is used for semantic search over pathway names.
- "seed_genes": 8 to 12 official HGNC gene symbols that are the most SPECIFIC markers or effectors of this mechanism. Prefer mechanism-specific genes over generic signalling hubs (AKT1, TP53, MAPK1, TNF). A gene may appear in only one mechanism.

Rules:
1. Exactly {n} mechanisms.
2. Mechanisms must not overlap: different processes, different seed genes.
3. Use only real, current human gene symbols.

Return JSON only, minified:
{{"facets": [{{"facet_name": "...", "description": "...", "seed_genes": ["GENE1", "GENE2"]}}]}}"""


def decompose_query(
    llm,
    model: str,
    query: str,
    n_mechanisms: int,
    *,
    context: Optional[Dict] = None,
    fallbacks: Sequence[str] = (),
    index: Optional[KnowledgeIndex] = None,
) -> Tuple[List[Dict], LLMResult]:
    """
    ``llm`` is an ``llm_client.LLMRouter``; ``model`` a model reference.

    Returns (facets, llm_result). Each facet has facet_id, facet_name,
    description, seed_genes and, if ``index`` is given, seeds_unknown (symbols
    the knowledge base does not contain).

    Raises LLMError if the model cannot produce a usable decomposition.
    """
    n_mechanisms = max(1, min(int(n_mechanisms), MAX_MECHANISMS))
    result = chat_json(
        llm, model,
        [{"role": "user", "content": DECOMPOSE_PROMPT.format(
            n=n_mechanisms, query=query.strip(), context=context_lines(context))}],
        max_tokens=700 + 140 * n_mechanisms,
        fallbacks=fallbacks,
    )

    raw = result.data.get("facets") if isinstance(result.data, dict) else result.data
    facets: List[Dict] = []
    seen_names = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("facet_name") or "").strip()
        if not name or name.lower() in seen_names:
            continue
        seen_names.add(name.lower())
        seeds = [str(g).strip() for g in (item.get("seed_genes") or []) if isinstance(g, str) and g.strip()]
        facet = {
            "facet_id": f"F{len(facets) + 1}",
            "facet_name": name,
            "description": str(item.get("description") or "").strip(),
            "seed_genes": list(dict.fromkeys(seeds)),
        }
        if index is not None:
            known, unknown = index.normalise_symbols(facet["seed_genes"])
            facet["seed_genes"] = known
            facet["seeds_unknown"] = unknown
        facets.append(facet)
        if len(facets) == n_mechanisms:
            break

    if not facets:
        raise LLMError(f"{result.model} returned no usable mechanisms")
    if len(facets) < n_mechanisms:
        result.notes.append(f"model returned {len(facets)} distinct mechanisms, {n_mechanisms} requested")
    return facets, result


# ============================================================
# LAYER 1b - alias resolution (NCBI Gene)
# ============================================================

# An ambiguous alias (several official genes carry the name) is resolved only
# when the mechanism's other seeds single out one candidate:
#   * its gene sets are significantly enriched for sets containing >= 2 of the
#     other seeds (hypergeometric p below ALIAS_MAX_P), and
#   * it shares such sets at least ALIAS_DOMINANCE times more often than the
#     runner-up (and at least ALIAS_MIN_SUPPORT times in absolute terms).
# Otherwise the name is reported as ambiguous and not used.
ALIAS_DOMINANCE = 5.0
ALIAS_MIN_SUPPORT = 5
ALIAS_MAX_P = 1e-4


def _context_support(index: KnowledgeIndex, candidate: str, context: Sequence[str]) -> Tuple[int, float]:
    """
    How strongly the mechanism's other seeds point at ``candidate``.

    Returns (support, p): the number of the candidate's gene sets that also
    contain >= 2 other context seeds, and the hypergeometric probability of
    seeing that many by chance given how many such sets exist in the KB.
    """
    import numpy as np
    from scipy.stats import hypergeom

    counts = np.zeros(index.n_pathways, dtype=np.int32)
    for seed in context:
        if seed != candidate:
            counts[index.gene_to_idx[seed]] += 1
    context_sets = counts >= 2
    own = index.gene_to_idx[candidate]
    support = int(context_sets[own].sum())
    p_value = float(hypergeom.sf(support - 1, index.n_pathways, int(context_sets.sum()), len(own)))
    return support, p_value


def resolve_seed_aliases(facets: Sequence[Dict], index: KnowledgeIndex, lookup) -> Dict:
    """
    Map seed names the knowledge base does not know onto official symbols.

    ``lookup`` is ``NCBIGene().lookup`` (or any callable with that contract).
    For every facet, each entry of ``seeds_unknown`` is looked up:

    * exactly one official gene in the KB carries the name -> resolved;
    * several do -> resolved only if the facet's other seeds single one out
      (significant and dominant co-membership, see ``ALIAS_*``), else left
      ambiguous and reported with its candidates - nothing is guessed;
    * none -> stays unknown.

    Facets are updated in place: resolved symbols are appended to
    ``seed_genes``; ``seeds_resolved`` records {name: {symbol, basis,
    candidates}}; ``seeds_ambiguous`` records {name: [candidates]}. Returns a
    summary for provenance. Raises whatever ``lookup`` raises when NCBI is
    unreachable - callers treat that as "skip this step".
    """
    unknown = sorted({name for facet in facets for name in facet.get('seeds_unknown', [])})
    summary = {'source': 'NCBI Gene (E-utilities)', 'looked_up': len(unknown),
               'resolved': 0, 'ambiguous': 0, 'unresolved': 0}
    if not unknown:
        return summary
    table = lookup(unknown)

    for facet in facets:
        seeds = list(facet.get('seed_genes', []))
        resolved: Dict[str, Dict] = {}
        ambiguous: Dict[str, List[str]] = {}
        still_unknown: List[str] = []

        for name in facet.get('seeds_unknown', []):
            candidates: List[str] = []
            for hit in table.get(name, []):
                known, _ = index.normalise_symbols([hit['symbol']])
                if known and known[0] not in candidates:
                    candidates.append(known[0])

            choice, basis = None, None
            if len(candidates) == 1:
                choice, basis = candidates[0], 'unique'
            elif len(candidates) > 1:
                scored = sorted(((*_context_support(index, c, seeds), c) for c in candidates),
                                key=lambda item: (-item[0], item[2]))
                (best_support, best_p, best), (runner_up_support, _, _) = scored[0], scored[1]
                if (best_support >= ALIAS_MIN_SUPPORT and best_p <= ALIAS_MAX_P
                        and best_support >= ALIAS_DOMINANCE * max(runner_up_support, 1)):
                    choice, basis = best, 'context'

            if choice:
                resolved[name] = {'symbol': choice, 'basis': basis, 'candidates': candidates}
                if choice not in seeds:
                    seeds.append(choice)
            elif candidates:
                ambiguous[name] = candidates
            else:
                still_unknown.append(name)

        facet['seed_genes'] = seeds
        facet['seeds_resolved'] = resolved
        facet['seeds_ambiguous'] = ambiguous
        facet['seeds_unknown'] = still_unknown
        summary['resolved'] += len(resolved)
        summary['ambiguous'] += len(ambiguous)
        summary['unresolved'] += len(still_unknown)
    return summary


# ============================================================
# LAYER 2
# ============================================================

def sentence_encoder(model=None, model_name: str = DEFAULT_EMBEDDING_MODEL):
    """
    Text -> unit-norm embedding function for the semantic channel.

    Pass an already loaded SentenceTransformer to reuse it (the app caches
    one); otherwise the model is loaded here.
    """
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
    return lambda text: model.encode(text, normalize_embeddings=True)


def build_signatures(
    index: KnowledgeIndex,
    facets: Sequence[Dict],
    encode,
    params: RetrievalParams,
    n_variants: int = 1,
    progress=None,
) -> Tuple[List[GeneSignature], Dict]:
    """
    Build signatures for every mechanism and drop near-duplicates.

    Returns (signatures, info). ``info`` records the settings that shaped the
    result plus what was dropped, and is exported verbatim.
    """
    built: List[GeneSignature] = []
    empty: List[str] = []
    for i, facet in enumerate(facets, start=1):
        signatures = build_mechanism_signatures(
            index, facet, encode, params, n_variants=n_variants,
            id_prefix=str(facet.get('facet_id') or f"F{i}"))
        if not signatures:
            empty.append(str(facet.get('facet_name')))
        built.extend(signatures)
        if progress:
            progress(i, len(facets))

    kept = deduplicate(built, params.dedup_jaccard)
    info = {
        **params.to_dict(),
        'signatures_per_mechanism': n_variants,
        'semantic_channel': encode is not None and params.use_semantic,
        'dropped_as_duplicates': len(built) - len(kept),
        'mechanisms_without_signature': empty,
    }
    return kept, info


# ============================================================
# LAYER 4 (debate)
# ============================================================

def debate_signature(engine, sig: GeneSignature, *, db=None, query: str = "",
                     context: Optional[Dict] = None, max_rounds: int = 3,
                     agreement_threshold: float = 1.0):
    """
    Fetch database evidence for one signature and run the debate on it.

    Returns (result, evidence_error). If the pathway API is unreachable the
    debate still runs, ungrounded, and ``evidence_error`` says why - the
    result's parameters record ``database_grounded: False``.
    """
    evidence, evidence_error = None, None
    if db is not None:
        try:
            tissue = context.get('tissue') if context else None
            evidence = db.signature_evidence(sig.genes, tissue_context=tissue)
        except ConnectionError as exc:
            evidence_error = str(exc)
    result = engine.run(
        sig.genes, evidence, query=query, mechanism=sig.signature_name,
        description=sig.description, context=context, max_rounds=max_rounds,
        agreement_threshold=agreement_threshold)
    return result, evidence_error


# ============================================================
# LAYER 4 (single-model review)
# ============================================================

VERIFY_PROMPT = """Review these {count} gene signatures for biological correctness.

Research question: {query}
{context}
Validation standard: {standard}

{blocks}

Task: {task}
- "remove": genes that do not belong to the stated mechanism (housekeeping genes, generic hubs, unrelated genes). Only genes listed in that signature.
{add_rule}- Give a reason of at most 12 words for every gene you list.

Return JSON only, minified, keyed by the signature labels above:
{{"S1": {{"remove": {{"GENE": "reason"}}, "add": {{"GENE": "reason"}}}}, "S2": {{...}}}}
Use empty objects when no change is needed."""


def verify_signatures(
    llm,
    model: str,
    signatures: Sequence[GeneSignature],
    query: str,
    *,
    suggest_additions: bool = True,
    context: Optional[Dict] = None,
    publication_mode: bool = False,
    batch_size: int = 5,
    fallbacks: Sequence[str] = (),
    index: Optional[KnowledgeIndex] = None,
    progress=None,
) -> Tuple[Dict[str, Dict], List[Dict]]:
    """
    Review signatures in batches.

    Returns (suggestions, calls):
      suggestions[signature_id] = {
          "status": "ok" | "failed", "error": str | None,
          "genes_to_remove": [...], "genes_to_add": [...], "reasoning": {gene: why},
          "rejected": [symbols the model named that were not valid],
      }
      calls = one provenance record per batch.

    A failed batch is marked failed. It is never reported as "no changes".
    """
    suggestions: Dict[str, Dict] = {}
    calls: List[Dict] = []
    signatures = list(signatures)
    batches = [signatures[i:i + batch_size] for i in range(0, len(signatures), batch_size)]

    for batch_no, batch in enumerate(batches, start=1):
        if progress:
            progress(batch_no, len(batches))
        blocks = "\n\n".join(
            f"S{i} - {sig.signature_name}\n"
            f"Mechanism: {sig.description or sig.facet}\n"
            f"Top supporting gene sets: {', '.join(p['pathway'] for p in sig.source_pathways[:4]) or 'n/a'}\n"
            f"Genes ({len(sig.genes)}): {', '.join(sig.genes)}"
            for i, sig in enumerate(batch, start=1)
        )
        prompt = VERIFY_PROMPT.format(
            count=len(batch), query=query, context=context_lines(context),
            standard=("PUBLICATION - require strong, specific evidence for every gene"
                      if publication_mode else "EXPLORATORY - flag only clear errors"),
            blocks=blocks,
            task=("Identify genes to remove and suggest genes to add."
                  if suggest_additions else "Identify genes to remove."),
            add_rule=(f'- "add": at most {MAX_ADDITIONS_PER_SIGNATURE} well-established genes '
                      "missing from the signature.\n" if suggest_additions else
                      '- "add": leave empty.\n'),
        )
        try:
            result = chat_json(llm, model, [{"role": "user", "content": prompt}],
                               max_tokens=500 + 260 * len(batch), fallbacks=fallbacks)
        except Exception as exc:  # noqa: BLE001 - recorded per signature below
            calls.append({'batch': batch_no, 'status': 'failed', 'error': str(exc)[:300]})
            for sig in batch:
                suggestions[sig.signature_id] = {
                    'status': 'failed', 'error': str(exc)[:300],
                    'genes_to_remove': [], 'genes_to_add': [], 'reasoning': {}, 'rejected': [],
                }
            continue

        calls.append({'batch': batch_no, 'status': 'ok', **result.provenance()})
        data = result.data if isinstance(result.data, dict) else {}
        for i, sig in enumerate(batch, start=1):
            entry = data.get(f"S{i}")
            if not isinstance(entry, dict):
                suggestions[sig.signature_id] = {
                    'status': 'failed', 'error': f"model omitted S{i} from its answer",
                    'genes_to_remove': [], 'genes_to_add': [], 'reasoning': {}, 'rejected': [],
                }
                continue
            suggestions[sig.signature_id] = _clean_suggestion(entry, sig, index, suggest_additions)

    return suggestions, calls


def _as_reason_map(block) -> Dict[str, str]:
    if isinstance(block, dict):
        return {str(k).strip(): str(v or "").strip()[:200] for k, v in block.items()}
    if isinstance(block, list):
        return {str(g).strip(): "" for g in block if isinstance(g, str)}
    return {}


def _clean_suggestion(entry: Dict, sig: GeneSignature, index: Optional[KnowledgeIndex],
                      allow_additions: bool) -> Dict:
    in_signature = {g.upper(): g for g in sig.genes}
    reasoning: Dict[str, str] = {}
    rejected: List[str] = []

    removals = []
    for symbol, reason in _as_reason_map(entry.get("remove", entry.get("genes_to_remove"))).items():
        gene = in_signature.get(symbol.upper())
        if gene is None:
            rejected.append(symbol)
        elif gene not in removals:
            removals.append(gene)
            reasoning[gene] = reason

    additions = []
    if allow_additions:
        for symbol, reason in _as_reason_map(entry.get("add", entry.get("genes_to_add"))).items():
            if not symbol or symbol.upper() in in_signature:
                continue
            if index is not None:
                known, _ = index.normalise_symbols([symbol])
                if not known:
                    rejected.append(symbol)      # not a symbol the knowledge base knows
                    continue
                symbol = known[0]
            if symbol not in additions and len(additions) < MAX_ADDITIONS_PER_SIGNATURE:
                additions.append(symbol)
                reasoning[symbol] = reason

    return {'status': 'ok', 'error': None, 'genes_to_remove': removals,
            'genes_to_add': additions, 'reasoning': reasoning, 'rejected': rejected}


# ============================================================
# WRITING LAYER OUTCOMES BACK ONTO A SIGNATURE
# ============================================================
# ``sig.genes`` is the single ordered candidate list. Every function below
# edits it in place, keeps ranked order, and records what it did and why.

def apply_network_expansion(sig: GeneSignature, candidates: Sequence[Dict]) -> List[str]:
    """
    Replace any previous network additions with ``candidates`` (from
    ``DatabaseClient.network_expansion``), appended after the retrieved genes.
    Idempotent. Returns the genes added.
    """
    base = [g for g in sig.genes if sig.gene_sources.get(g) != 'network']
    present = set(base)
    added = [c['gene'] for c in candidates if c['gene'] not in present]
    sig.gene_sources = {g: s for g, s in sig.gene_sources.items() if s != 'network'}
    for gene in added:
        sig.gene_sources[gene] = 'network'
    sig.genes = base + added
    sig.dam_expanded = bool(added)
    return added


def retrieved_genes(sig: GeneSignature) -> List[str]:
    """The signature without network additions - the input to expansion."""
    return [g for g in sig.genes if sig.gene_sources.get(g) != 'network']


def apply_review(sig: GeneSignature, suggestion: Dict, *, remove: bool = True,
                 add: bool = False) -> Dict[str, List[str]]:
    """
    Apply a single-model review to a signature (used by the command-line
    runner; the app leaves the decision to the user). A failed review changes
    nothing. Returns {'removed': [...], 'added': [...]}.
    """
    outcome: Dict[str, List[str]] = {'removed': [], 'added': []}
    if suggestion.get('status') != 'ok':
        return outcome
    sig.llm2_verified = True
    reasoning = suggestion.get('reasoning', {})
    if remove:
        flagged = [g for g in suggestion.get('genes_to_remove', []) if g in sig.genes]
        for gene in flagged:
            sig.removed_genes[gene] = f"llm review: {reasoning.get(gene) or 'no reason given'}"
        sig.genes = [g for g in sig.genes if g not in set(flagged)]
        outcome['removed'] = flagged
    if add:
        for gene in suggestion.get('genes_to_add', []):
            if gene not in sig.genes:
                sig.genes.append(gene)
                sig.gene_sources[gene] = 'llm'
                outcome['added'].append(gene)
    return outcome


def apply_debate_result(sig: GeneSignature, result, apply_removals: bool = True) -> List[str]:
    """
    Write a debate's metrics onto the signature and, if asked, remove the
    genes the agents voted out. A debate in which fewer than two agents
    answered (``result.failed``) changes nothing. Returns the genes removed.
    """
    if result.failed:
        return []
    metrics = result.metrics
    sig.debate_verified = True
    sig.stability_score = metrics['stability']
    sig.debate_entropy = metrics['entropy']
    sig.debate_conflict = metrics['conflict']
    sig.debate_agreement = metrics['agreement']
    if not apply_removals:
        return []
    removed = [g for g in result.genes_to_remove if g in sig.genes]
    for gene in removed:
        decision = result.decisions[gene]
        sig.removed_genes[gene] = (f"debate: P(remove)={decision.p_remove:.2f}, "
                                   f"{decision.n_remove}/{decision.n_voters} agents")
    drop = set(removed)
    sig.genes = [g for g in sig.genes if g not in drop]
    return removed

"""
Exports: GMT, CSV and a fully provenanced JSON record.

The JSON is meant to be sufficient to reproduce a run: it carries the query,
every parameter, the models actually used, knowledge-base identity (version
and file hashes), package versions, per-gene evidence and full debate records.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from signature_engine import ENGINE_VERSION, GeneSignature, overlap_summary


def file_sha256(path: str) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def package_versions() -> Dict[str, str]:
    from importlib import metadata
    out = {'python': platform.python_version()}
    for name in ('numpy', 'scipy', 'sentence-transformers', 'transformers', 'torch',
                 'openai', 'streamlit', 'pandas'):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = 'not installed'
    return out


def build_provenance(
    resources: Dict,
    *,
    pipeline_version: str,
    debate_version: str,
    llm_providers: Dict,
    pathway_api_url: str,
    reasoning_effort: Optional[str] = None,
    decomposition: Optional[Dict] = None,
    mechanisms: Optional[List[Dict]] = None,
    mechanisms_selected: Optional[List[str]] = None,
    retrieval_parameters: Optional[Dict] = None,
    network_expansion: Optional[Dict] = None,
    debate_models: Optional[Dict] = None,
) -> Dict:
    """Everything needed to identify the code, data and settings behind a run."""
    index = resources['index']
    metadata = resources.get('kb_metadata', {})
    return {
        'pipeline_version': pipeline_version,
        'debate_version': debate_version,
        'knowledge_base': {
            **{k: metadata.get(k) for k in ('version', 'created', 'sources', 'filters')},
            'gene_sets': index.n_pathways, 'genes': index.n_genes,
            'sha256': resources.get('kb_sha256'),
        },
        'embeddings': {
            'model': resources.get('embedding_model'),
            'sha256': resources.get('embeddings_sha256'),
            'text_recipe': "gene-set name with '_' as spaces, then its first 10 genes, space-joined",
        },
        # Providers and endpoints used; API keys are never recorded.
        'llm_providers': llm_providers,
        'reasoning_effort': reasoning_effort,
        'pathway_api': pathway_api_url,
        'decomposition': decomposition,
        'mechanisms': mechanisms or [],
        'mechanisms_selected': mechanisms_selected or [],
        'retrieval_parameters': retrieval_parameters,
        'network_expansion': network_expansion,
        'debate_models': debate_models or {},
    }


def _clean(text: str) -> str:
    return str(text).replace('\t', ' ').replace('\n', ' ').replace('|', '/')


def build_gmt(signatures: Sequence[GeneSignature]) -> str:
    """GMT: id <tab> description <tab> genes. One signature per line."""
    lines = []
    for sig in signatures:
        tags = [_clean(sig.facet), _clean(sig.mechanism), f"{len(sig.genes)}genes",
                f"support:{sig.support():.3f}",
                "stability:NA" if sig.stability_score is None else f"stability:{sig.stability_score:.3f}"]
        if sig.dam_expanded:
            tags.append("NETWORK")
        if sig.llm2_verified:
            tags.append("LLM_REVIEW")
        if sig.debate_verified:
            tags.append("DEBATE")
        lines.append('\t'.join([sig.signature_id, '|'.join(tags), *sig.genes]))
    return '\n'.join(lines) + ('\n' if lines else '')


def build_csv(signatures: Sequence[GeneSignature]) -> str:
    """Long format, one row per (signature, gene), ready for R / pandas."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(['signature_id', 'signature_name', 'facet', 'mechanism', 'gene', 'rank',
                     'score', 'supporting_gene_sets', 'weighted_support', 'source', 'llm_seed',
                     'signature_support', 'signature_stability', 'quality'])
    for sig in signatures:
        for rank, gene in enumerate(sig.genes, start=1):
            score = sig.gene_scores.get(gene)
            writer.writerow([
                sig.signature_id, sig.signature_name, sig.facet, sig.mechanism, gene, rank,
                '' if score is None else f"{score:.5f}",
                sig.gene_support.get(gene, ''),
                f"{sig.support_pool.get(gene, 0.0):.4f}",
                sig.gene_sources.get(gene, 'knowledge_base'),
                gene in sig.seed_genes,
                f"{sig.support():.4f}",
                '' if sig.stability_score is None else f"{sig.stability_score:.4f}",
                sig.quality_label(),
            ])
    return buffer.getvalue()


def build_json(
    signatures: Sequence[GeneSignature],
    *,
    query: str,
    generation_mode: str,
    biological_context: Optional[Dict],
    provenance: Dict,
    debates: Optional[Dict[str, Dict]] = None,
    verification: Optional[Dict] = None,
    timings: Optional[Dict[str, str]] = None,
) -> str:
    assessed = [s for s in signatures if s.stability_score is not None]

    def mean(values: List[float]) -> Optional[float]:
        return round(sum(values) / len(values), 4) if values else None

    record = {
        'schema': 'msigdb-signature-generator/export/3',
        'exported_at': datetime.now(timezone.utc).isoformat(),
        'query': query,
        'generation_mode': generation_mode,
        'biological_context': biological_context,
        'provenance': {'engine_version': ENGINE_VERSION, 'packages': package_versions(), **provenance},
        'summary': {
            'total_signatures': len(signatures),
            'mean_support_confidence': mean([s.support() for s in signatures]),
            'overlap': overlap_summary(signatures),
            'debate': {
                'signatures_assessed': len(assessed),
                'signatures_not_assessed': len(signatures) - len(assessed),
                'high_confidence': sum(1 for s in assessed if s.stability_score >= 0.8),
                'moderate_confidence': sum(1 for s in assessed if 0.5 <= s.stability_score < 0.8),
                'requires_manual_review': sum(1 for s in assessed if s.stability_score < 0.5),
                'mean_stability': mean([s.stability_score for s in assessed]),
                'mean_entropy': mean([s.debate_entropy for s in assessed if s.debate_entropy is not None]),
                'mean_conflict': mean([s.debate_conflict for s in assessed if s.debate_conflict is not None]),
            },
        },
        'signatures': [s.to_dict() for s in signatures],
        'llm_review': verification or {},
        'debates': debates or {},
        'layer_timings': timings or {},
    }
    return json.dumps(record, indent=2, default=str)

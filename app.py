"""
MSigDB Signature Generator - Streamlit app
==========================================

UI only. The method lives in importable, tested modules:

    signature_engine.py   seed-anchored hybrid retrieval + evidence-weighted scoring
    pipeline.py           LLM decomposition (Layer 1), alias resolution, review (Layer 4)
    debate_engine.py      structured three-agent, per-gene debate (Layer 4)
    db_client.py          pathway API: network expansion (Layer 3), evidence tables
    ncbi_client.py        NCBI Gene: gene aliases -> official symbols
    llm_client.py         robust JSON chat calls for every provider
    providers.py          the provider registry
    model_registry.py     model references, ranking and debate role assignment
    exporters.py          GMT / CSV / provenanced JSON

State model: ``signatures[sid].genes`` is the single ordered candidate list.
Layer 3 appends to it. Layer 4 never deletes from it in the app: a gene the
reviewer or the debate votes out is *deselected with its reason recorded*, so
the user can overrule it in Layer 5. Approval freezes an ordered copy of the
selected genes. Nothing downstream can silently undo an upstream layer.
"""

import dataclasses
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from db_client import BackendUnavailable, DatabaseClient, strength_to_min_prob
from debate_engine import (
    DEBATE_VERSION, DEFAULT_REMOVAL_THRESHOLD, ROLE_LABELS, DebateEngine,
    estimate_tokens_per_model,
)
from exporters import build_csv, build_gmt, build_json, build_provenance
from gene_filters import is_housekeeping
from llm_client import EFFORT_LEVELS, LLMRouter, connect_provider
from model_registry import (
    DEBATE_ROLES, MSIGDB_API_URL, WAKE_BACKEND_HINT, debate_diversity, describe_model,
    explain_llm_error, fallback_chain, pick_default, resolve_debate_models,
)
from ncbi_client import NCBIGene
from pipeline import (
    DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDINGS_PATH, DEFAULT_KB_PATH, MAX_MECHANISMS,
    PIPELINE_VERSION, apply_debate_result, apply_network_expansion, build_signatures,
    debate_signature, decompose_query, load_resources, resolve_seed_aliases, retrieved_genes,
    sentence_encoder, verify_signatures,
)
from providers import PROVIDERS, providers_in_environment
from signature_engine import (
    ENGINE_VERSION, GeneSignature, KnowledgeIndex, RetrievalParams, overlap_summary,
)

KB_PATH = DEFAULT_KB_PATH
EMBEDDINGS_PATH = DEFAULT_EMBEDDINGS_PATH
EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
EFFORT_DEFAULT = "provider default"


# ============================================================
# CACHED RESOURCES
# ============================================================

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_index() -> Optional[Dict]:
    """Knowledge base + embeddings as a KnowledgeIndex, with identifying metadata."""
    if not (os.path.exists(KB_PATH) and os.path.exists(EMBEDDINGS_PATH)):
        return None
    return load_resources(KB_PATH, EMBEDDINGS_PATH)


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def get_db_client() -> DatabaseClient:
    return DatabaseClient(MSIGDB_API_URL)


@st.cache_data(ttl=60, show_spinner=False)
def backend_health(api_url: str = MSIGDB_API_URL) -> bool:
    """True if the pathway API Space answers. Cached so reruns do not re-probe."""
    import requests
    try:
        return requests.get(api_url + "/", timeout=5).status_code == 200
    except Exception:
        return False


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    'connections': {}, 'provider_error': None, 'environment_checked': False,
    'generation_mode': 'exploratory', 'bio_context': None,
    'run_id': 0,
    'facets': [], 'l1_provenance': None, 'granularity_approved': False, 'selected_facet_ids': [],
    'signatures': {}, 'retrieval_params': None, 'query_used': '',
    'network_additions': {}, 'expansion_settings': None,
    'suggestions': {}, 'review_calls': [], 'review_model': None,
    'debates': {}, 'debate_models': {}, 'debate_problems': [],
    'deselected': {}, 'approved': {},
    'layer_timings': {},
}

# Keys cleared when an upstream layer is re-run, so stale results never leak
# into a new run's export.
DOWNSTREAM = {
    1: ['facets', 'l1_provenance', 'granularity_approved', 'selected_facet_ids'],
    2: ['signatures', 'retrieval_params', 'network_additions', 'expansion_settings',
        'suggestions', 'review_calls', 'review_model', 'debates', 'debate_models',
        'debate_problems', 'deselected', 'approved'],
}


def _fresh(value):
    return type(value)() if isinstance(value, (dict, list)) else value


def initialize_session_state():
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = _fresh(value)


def reset_from(layer: int):
    """Clear the results of ``layer`` and everything after it."""
    for level, keys in DOWNSTREAM.items():
        if level >= layer:
            for key in keys:
                st.session_state[key] = _fresh(DEFAULTS[key])
    st.session_state.layer_timings = {
        name: t for name, t in st.session_state.layer_timings.items()
        if int(name.split(':')[0].split()[-1]) < layer
    }
    st.session_state.run_id += 1


def record_timing(name: str, started: float):
    seconds = time.time() - started
    st.session_state.layer_timings[name] = f"{seconds:.1f}s" if seconds < 60 else f"{seconds / 60:.1f}m"


def invalidate_approval(signature_id: str):
    """A signature edited after approval must be approved again."""
    st.session_state.approved.pop(signature_id, None)


def ordered_signatures() -> List[GeneSignature]:
    return list(st.session_state.signatures.values())


# ---------------------------------------------------------------- providers

def available_models() -> List[str]:
    """Every model of every connected provider, as 'provider:model' references."""
    return [ref for connection in st.session_state.connections.values() for ref in connection.refs()]


def get_router() -> LLMRouter:
    effort = st.session_state.get('effort_choice')
    return LLMRouter(st.session_state.connections, effort=effort if effort in EFFORT_LEVELS else None)


def reasoning_model() -> Optional[str]:
    available = available_models()
    chosen = st.session_state.get('model_reasoning')
    return chosen if chosen in available else pick_default(available)


def fallbacks_for(model: str) -> List[str]:
    return fallback_chain(model, available_models())


def model_selectbox(container, label: str, default: Optional[str], key: str, **kwargs) -> str:
    """
    Model picker that survives the live model list changing under it.

    Streamlit forbids assigning to a widget key after the widget exists, so a
    stale selection (a model retired or a provider disconnected since it was
    chosen) is repaired here, before instantiation.
    """
    available = available_models()
    if st.session_state.get(key) not in available:
        st.session_state[key] = default if default in available else available[0]
    return container.selectbox(label, available, key=key, **kwargs)


def connect_from_environment():
    """Once per session: connect every provider whose API key is in the environment."""
    if st.session_state.environment_checked:
        return
    st.session_state.environment_checked = True
    for provider_id, key in providers_in_environment().items():
        if provider_id in st.session_state.connections:
            continue
        try:
            st.session_state.connections[provider_id] = connect_provider(provider_id, key)
        except Exception as exc:
            st.session_state.provider_error = (f"{PROVIDERS[provider_id].label} (key from "
                                               f"environment): {explain_llm_error(exc)}")


# ============================================================
# SIDEBAR
# ============================================================

def render_provider_panel():
    connections = st.session_state.connections
    with st.expander("Providers & model", expanded=not connections):
        by_label = {provider.label: provider for provider in PROVIDERS.values()}
        provider = by_label[st.selectbox("Provider", list(by_label), key="provider_choice")]
        if provider.note:
            st.caption(provider.note)
        key_label = "API key" if provider.needs_key else "API key (optional)"
        key_input = st.text_input(f"{provider.label} {key_label}", type="password",
                                  key=f"key_{provider.id}")
        base_url = None
        if provider.edit_base_url:
            base_url = st.text_input("Base URL", value=provider.base_url or "",
                                     key=f"url_{provider.id}", placeholder="https://host/v1")
        if provider.key_url:
            st.caption(f"Get a key: {provider.key_url}")

        if st.button("Connect", type="primary"):
            try:
                connections[provider.id] = connect_provider(provider.id, key_input, base_url)
                st.session_state.provider_error = None
            except Exception as exc:
                st.session_state.provider_error = f"{provider.label}: {explain_llm_error(exc)}"
            st.rerun()

        if connections:
            st.markdown("**Connected**")
            for provider_id in list(connections):
                connection = connections[provider_id]
                col1, col2 = st.columns([3, 2])
                col1.caption(f"{connection.provider.label} · {len(connection.models)} models")
                if col2.button("Remove", key=f"disconnect_{provider_id}"):
                    del connections[provider_id]
                    st.rerun()
                if connection.warning:
                    st.caption(f"⚠️ {connection.warning}")

            connected = {connections[pid].provider.label: pid for pid in connections}
            st.caption("Model missing from a list? Add its ID exactly as the provider names it.")
            target = connected[st.selectbox("Add a model to", list(connected), key="manual_provider")]
            manual = st.text_input("Model ID", key="manual_model")
            if st.button("Add model") and manual.strip():
                connections[target].models.setdefault(manual.strip(), {})
                st.rerun()

        available = available_models()
        if available:
            model_selectbox(st, "Reasoning model", pick_default(available), "model_reasoning",
                            help="Used for query decomposition (Layer 1) and as the default "
                                 "for single-model review (Layer 4).")
            st.selectbox("Reasoning effort", [EFFORT_DEFAULT, *EFFORT_LEVELS], key="effort_choice",
                         help="How much hidden reasoning to request. 'provider default' keeps "
                              "free-tier models inside their token limits; raise it on paid keys "
                              "for harder questions. Models that do not support it ignore it.")
            if st.button("Refresh model lists"):
                for provider_id, connection in list(connections.items()):
                    try:
                        connections[provider_id] = connect_provider(
                            provider_id, connection.api_key, connection.base_url)
                    except Exception as exc:
                        st.session_state.provider_error = (
                            f"{connection.provider.label}: {explain_llm_error(exc)}")
                st.rerun()


def render_sidebar():
    with st.sidebar:
        st.markdown("### Configuration")
        render_provider_panel()

        connections = st.session_state.connections
        if st.session_state.provider_error:
            st.error(st.session_state.provider_error)
        if connections:
            total = len(available_models())
            names = ", ".join(c.provider.label for c in connections.values())
            st.success(f"Connected: {names} ({total} models)")
        elif not st.session_state.provider_error:
            st.info("Connect an LLM provider to begin. Keys stay in this browser session "
                    "and are never written to exports.")

        if backend_health():
            st.success("Pathway API: online")
        else:
            st.warning("**Pathway API: asleep or offline**\n\n" + WAKE_BACKEND_HINT +
                       "\n\nSignature generation works without it. Network expansion is paused "
                       "and the debate runs without database evidence.")
            if st.button("Retry backend"):
                backend_health.clear()
                st.rerun()

        st.markdown("---")
        st.info("**Mode:** " + ("Publication" if st.session_state.generation_mode == 'publication'
                                else "Exploratory"))

        if st.session_state.layer_timings:
            st.markdown("### Timings")
            for name, duration in st.session_state.layer_timings.items():
                st.caption(f"{name}: {duration}")

        st.markdown("---")
        st.caption(
            f"Pipeline {PIPELINE_VERSION} · engine {ENGINE_VERSION} · debate {DEBATE_VERSION}\n\n"
            "1. Mechanisms (LLM + NCBI aliases)\n2. Hybrid retrieval (local)\n"
            "3. Network expansion (API)\n4. Review or debate (LLM + API)\n5. Approve & export"
        )


# ============================================================
# KNOWLEDGE BASE TAB
# ============================================================

def render_kb_tab(resources: Optional[Dict]):
    st.markdown("## Knowledge base")
    if not resources:
        st.error(f"Knowledge base not found. `{KB_PATH}` and `{EMBEDDINGS_PATH}` are required.")
        return

    index: KnowledgeIndex = resources['index']
    meta = resources['kb_metadata']
    col1, col2, col3 = st.columns(3)
    col1.metric("Gene sets", f"{index.n_pathways:,}")
    col2.metric("Genes", f"{index.n_genes:,}")
    col3.metric("Version", meta.get('version', 'unknown'))

    if meta.get('sources'):
        st.caption("Sources: " + ", ".join(f"{k} ({v:,})" for k, v in meta['sources'].items()))
    st.caption(f"Embedding model: {resources['embedding_model']} · "
               f"KB sha256 {str(resources['kb_sha256'])[:12]}… · "
               f"embeddings sha256 {str(resources['embeddings_sha256'])[:12]}…")

    with st.expander("Example gene set"):
        sample = index.ids[0]
        genes = index.pathways[sample]
        st.caption(f"**{sample}** - {len(genes)} genes: {', '.join(genes[:12])}…")


# ============================================================
# GENERATION TAB
# ============================================================

def render_mode_and_context() -> bool:
    """Mode toggle and biological context. Returns False if required input is missing."""
    st.markdown("### Generation mode")
    col1, col2 = st.columns(2)
    current = st.session_state.generation_mode
    if col1.button("Exploratory", width="stretch",
                   type="primary" if current == 'exploratory' else "secondary"):
        st.session_state.generation_mode = 'exploratory'
        st.rerun()
    if col2.button("Publication", width="stretch",
                   type="primary" if current == 'publication' else "secondary"):
        st.session_state.generation_mode = 'publication'
        st.rerun()

    publication = st.session_state.generation_mode == 'publication'
    if publication:
        st.info("**Publication mode** - biological context is required, housekeeping genes are "
                "removed, and LLM review applies a strict evidence standard.")
    else:
        st.info("**Exploratory mode** - context is optional and housekeeping genes are flagged "
                "but kept. Treat output as hypotheses.")

    st.markdown("### Biological context" + (" (required)" if publication else " (optional)"))
    tissues = ["", "Blood/PBMC", "Adipose Tissue", "Liver", "Brain", "Lung", "Intestine",
               "Pancreas", "Kidney", "Heart", "Muscle", "Skin", "Spleen", "Breast", "Other"]
    col1, col2, col3 = st.columns(3)
    species = col1.selectbox("Species", ["Homo sapiens", "Mus musculus"], key="ctx_species")
    tissue = col2.selectbox("Tissue", tissues, key="ctx_tissue",
                            help="Matched to GTEx v10 for the expression evidence in the debate.")
    if tissue == "Other":
        tissue = col2.text_input("Specify tissue", key="ctx_tissue_other")
    disease = col3.text_input("Disease / condition", key="ctx_disease",
                              placeholder="e.g. melanoma")
    col1, col2 = st.columns(2)
    cell_type = col1.text_input("Cell type", key="ctx_cell", placeholder="e.g. CD8+ T cells")
    treatment = col2.text_input("Treatment", key="ctx_treatment", placeholder="e.g. anti-PD-1")

    if species == "Mus musculus":
        st.caption("The knowledge base holds human gene symbols; mouse context is passed to the "
                   "LLMs but genes are reported as human orthologue symbols.")

    context = {
        'species': species, 'tissue': (tissue or '').strip() or None,
        'disease': disease.strip() or None, 'cell_type': cell_type.strip() or None,
        'treatment': treatment.strip() or None,
    }
    has_context = any(context[k] for k in ('tissue', 'disease', 'cell_type', 'treatment'))
    st.session_state.bio_context = context if has_context else None

    if publication and not (context['tissue'] and context['disease']):
        st.error("Publication mode needs at least a tissue and a disease/condition.")
        return False
    return True


def render_generation_tab(resources: Optional[Dict]):
    st.markdown("## Signature generation")
    if not resources:
        st.error("Knowledge base not loaded - see the Knowledge base tab.")
        return
    if not available_models():
        st.warning("Connect an LLM provider in the sidebar to begin.")
        return

    if not render_mode_and_context():
        return

    st.markdown("---")
    st.markdown("### Research question")
    query = st.text_area("Biological question or mechanism", key="main_query", height=80,
                         placeholder="e.g. T cell exhaustion in the tumor microenvironment")

    render_layer1(query, resources)
    if not st.session_state.granularity_approved:
        return
    render_layer2(resources)
    if not st.session_state.signatures:
        return
    render_layer3()
    render_layer4(resources)
    render_layer5()
    render_export(resources)


# ============================================================
# LAYER 1 - MECHANISMS
# ============================================================

def render_layer1(query: str, resources: Dict):
    st.markdown("---")
    st.markdown("### Layer 1 - Mechanisms")
    st.caption("The LLM proposes distinct mechanisms, each with a plain-language description "
               "and seed genes. The knowledge base, not the LLM, decides the final genes.")

    n_mechanisms = st.slider("Number of mechanisms", 3, MAX_MECHANISMS, 10)
    resolve_aliases = st.checkbox(
        "Resolve gene aliases with NCBI Gene", value=True,
        help="Seed genes named by an alias (GLUT1, PD-L1, 4-1BB) are mapped to official "
             "symbols through NCBI's public API. Ambiguous names are reported, not guessed.")
    col1, col2 = st.columns(2)

    if col1.button("Generate mechanisms", type="primary", width="stretch"):
        if not query or not query.strip():
            st.error("Enter a research question first.")
        else:
            model = reasoning_model()
            started = time.time()
            facets, result = None, None
            with st.spinner(f"{describe_model(model)} is proposing {n_mechanisms} mechanisms..."):
                try:
                    facets, result = decompose_query(
                        get_router(), model, query, n_mechanisms,
                        context=st.session_state.bio_context,
                        fallbacks=fallbacks_for(model), index=resources['index'])
                except Exception as exc:
                    st.error(explain_llm_error(exc, model))
            if facets:
                aliases = {'enabled': resolve_aliases}
                if resolve_aliases:
                    try:
                        aliases.update(resolve_seed_aliases(facets, resources['index'], NCBIGene().lookup))
                    except Exception as exc:
                        aliases['error'] = f"NCBI unavailable, aliases not resolved: {str(exc)[:160]}"
                reset_from(1)
                st.session_state.facets = facets
                st.session_state.selected_facet_ids = [f['facet_id'] for f in facets]
                st.session_state.query_used = query.strip()
                st.session_state.l1_provenance = {
                    'requested': n_mechanisms, 'returned': len(facets),
                    'alias_resolution': aliases, **result.provenance()}
                record_timing("Layer 1: mechanisms", started)
                st.rerun()

    if st.session_state.facets and col2.button("Discard mechanisms", width="stretch"):
        reset_from(1)
        st.rerun()

    facets = st.session_state.facets
    if not facets:
        return

    provenance = st.session_state.l1_provenance or {}
    st.caption(f"Model: `{provenance.get('model')}` · {provenance.get('returned')} of "
               f"{provenance.get('requested')} mechanisms")
    for note in provenance.get('notes', []):
        st.warning(note)
    aliases = provenance.get('alias_resolution') or {}
    if aliases.get('error'):
        st.warning(aliases['error'])
    elif aliases.get('looked_up'):
        st.caption(f"NCBI Gene: {aliases['resolved']} alias(es) resolved, "
                   f"{aliases['ambiguous']} ambiguous, {aliases['unresolved']} not found.")

    selected = []
    for facet in facets:
        col1, col2 = st.columns([1, 24])
        keep = col1.checkbox("use", value=facet['facet_id'] in st.session_state.selected_facet_ids,
                             key=f"use_{facet['facet_id']}_{st.session_state.run_id}",
                             label_visibility="collapsed")
        if keep:
            selected.append(facet['facet_id'])
        with col2.expander(f"{facet['facet_id']}. {facet['facet_name']}"):
            st.write(facet['description'] or "_No description returned._")
            st.caption("Seed genes: " + (", ".join(facet['seed_genes']) or "none recognised"))
            if facet.get('seeds_resolved'):
                st.caption("Aliases resolved (NCBI Gene): " + ", ".join(
                    f"{name} → {info['symbol']}" for name, info in facet['seeds_resolved'].items()))
            if facet.get('seeds_ambiguous'):
                st.caption("Ambiguous names, not used: " + "; ".join(
                    f"{name} ({' / '.join(candidates[:4])})"
                    for name, candidates in facet['seeds_ambiguous'].items()))
            if facet.get('seeds_unknown'):
                st.caption("Not recognised, ignored: " + ", ".join(facet['seeds_unknown']))
    st.session_state.selected_facet_ids = selected

    st.caption(f"Selected {len(selected)} of {len(facets)} mechanisms")
    if st.session_state.granularity_approved:
        st.success("Mechanisms approved")
    elif st.button("Approve selected & continue", type="primary", width="stretch"):
        if not selected:
            st.error("Select at least one mechanism.")
        else:
            st.session_state.granularity_approved = True
            st.rerun()


# ============================================================
# LAYER 2 - HYBRID RETRIEVAL
# ============================================================

def render_layer2(resources: Dict):
    st.markdown("---")
    st.markdown("### Layer 2 - Evidence-grounded signatures")
    st.caption("Gene sets statistically enriched for the seed genes are taken first, re-ranked "
               "by semantic similarity to the description; genes are scored by weighted support "
               "across the retrieved sets. Runs locally and is fully deterministic.")

    defaults = RetrievalParams()
    col1, col2, col3 = st.columns(3)
    variants = col1.number_input("Signatures per mechanism", 1, 4, 1,
                                 help="1 = one core signature. More adds sub-modules: clusters "
                                      "of overlapping gene sets within the mechanism.")
    min_genes = col2.number_input("Min genes", 5, 30, defaults.min_genes)
    max_genes = col3.number_input("Max genes", 10, 50, defaults.max_genes)
    if min_genes > max_genes:
        st.error("Min genes must not exceed max genes.")
        return

    with st.expander("Retrieval settings"):
        col1, col2 = st.columns(2)
        top_k = col1.slider("Gene sets kept per mechanism", 5, 40, defaults.top_k, 1,
                            help="Fewer, more relevant gene sets gave the best held-out recovery.")
        fdr = col2.select_slider("Enrichment FDR threshold", [0.001, 0.01, 0.05, 0.1],
                                 value=defaults.fdr_threshold)
        min_support = col1.slider("Min supporting gene sets per gene", 1, 5, defaults.min_support)
        dedup = col2.slider("Max Jaccard overlap between signatures", 0.1, 0.9,
                            defaults.dedup_jaccard, 0.05,
                            help="A signature more similar than this to an earlier one is dropped.")

    if st.button("Build signatures", type="primary", width="stretch"):
        index: KnowledgeIndex = resources['index']
        params = RetrievalParams(
            top_k=top_k, fdr_threshold=float(fdr), min_support=min_support,
            min_genes=int(min_genes), max_genes=int(max_genes), dedup_jaccard=dedup,
            drop_housekeeping=st.session_state.generation_mode == 'publication')

        try:
            encode = sentence_encoder(load_embedding_model())
        except Exception as exc:
            st.warning(f"Embedding model unavailable ({exc}). Using seed enrichment only.")
            encode = None

        started = time.time()
        facets = [f for f in st.session_state.facets
                  if f['facet_id'] in st.session_state.selected_facet_ids]
        progress = st.progress(0.0)
        kept, info = build_signatures(index, facets, encode, params, n_variants=int(variants),
                                      progress=lambda done, total: progress.progress(done / total))
        progress.empty()

        reset_from(2)
        st.session_state.signatures = {s.signature_id: s for s in kept}
        st.session_state.retrieval_params = info
        record_timing("Layer 2: retrieval", started)
        st.rerun()

    signatures = ordered_signatures()
    if not signatures:
        return

    info = st.session_state.retrieval_params or {}
    overlap = overlap_summary(signatures)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Signatures", len(signatures))
    col2.metric("Mean support", f"{sum(s.support() for s in signatures) / len(signatures):.2f}",
                help="Mean relevance-weighted fraction of retrieved gene sets containing each gene.")
    col3.metric("Mean pairwise Jaccard", f"{overlap['mean_jaccard']:.3f}")
    col4.metric("Max pairwise Jaccard", f"{overlap['max_jaccard']:.2f}")
    if info.get('dropped_as_duplicates'):
        st.caption(f"{info['dropped_as_duplicates']} signature(s) dropped as near-duplicates.")
    if info.get('mechanisms_without_signature'):
        st.warning("No signature could be built for: " + ", ".join(info['mechanisms_without_signature']))

    st.dataframe(pd.DataFrame([{
        'Signature': s.signature_name, 'Genes': len(s.genes), 'Support': round(s.support(), 3),
        'Retrieval': s.retrieval_mode, 'Seeds kept': f"{s.current_seed_recovery():.0%}",
        'Not LLM-proposed': f"{s.current_novel_fraction():.0%}",
        'Top gene set': s.source_pathways[0]['pathway'] if s.source_pathways else '',
    } for s in signatures]), hide_index=True, width="stretch")

    with st.expander("Supporting gene sets per signature"):
        # Names are unique (one per mechanism and sub-module), so they can be
        # the options directly.
        by_name = {s.signature_name: s for s in signatures}
        sig = by_name[st.selectbox("Signature", list(by_name),
                                   key=f"evidence_view_{st.session_state.run_id}")]
        st.dataframe(pd.DataFrame(sig.source_pathways).drop(columns=['seeds_hit'], errors='ignore'),
                     hide_index=True, width="stretch")
        if sig.seeds_unsupported:
            st.caption("LLM seeds the knowledge base did not support here (dropped): "
                       + ", ".join(sig.seeds_unsupported))


# ============================================================
# LAYER 3 - NETWORK EXPANSION
# ============================================================

def render_layer3():
    st.markdown("---")
    st.markdown("### Layer 3 - Network expansion (optional)")
    st.caption("Adds genes that are network neighbours of at least two signature genes, ranked "
               "by summed link strength. Additions are appended after the retrieved genes and "
               "tagged `network`.")

    backend_up = backend_health()
    if not backend_up:
        st.warning("**Network expansion needs the pathway API.**\n\n" + WAKE_BACKEND_HINT)

    col1, col2 = st.columns(2)
    strength = col1.slider("Expansion strength", 0.0, 1.0, 0.5, 0.1,
                           help="Higher = weaker links accepted = more candidates.")
    max_additions = col2.slider("Max genes added per signature", 1, 10, 3)
    col1.caption(f"Link weight cut-off: {strength_to_min_prob(strength):.3f} "
                 "(each gene's 50 strongest links are stored, with weights of about 0.01-0.07)")

    if st.button("Expand signatures", width="stretch", disabled=not backend_up):
        db = get_db_client()
        publication = st.session_state.generation_mode == 'publication'
        started = time.time()
        failures = []
        progress = st.progress(0.0)
        signatures = ordered_signatures()
        for i, sig in enumerate(signatures):
            try:
                # Expand from the retrieved genes only, so re-running is idempotent.
                candidates = db.network_expansion(retrieved_genes(sig), strength=strength,
                                                  max_additions=max_additions,
                                                  drop_housekeeping=publication)
            except BackendUnavailable as exc:
                failures.append(f"{sig.signature_name}: {exc}")
                continue
            apply_network_expansion(sig, candidates)
            st.session_state.network_additions[sig.signature_id] = candidates
            invalidate_approval(sig.signature_id)
            progress.progress((i + 1) / len(signatures))
        progress.empty()
        st.session_state.expansion_settings = {
            'strength': strength, 'min_link_probability': strength_to_min_prob(strength),
            'max_additions': max_additions, 'min_supporting_signature_genes': 2}
        record_timing("Layer 3: network expansion", started)
        for failure in failures:
            st.warning(failure)
        if not failures:
            st.rerun()

    additions = st.session_state.network_additions
    if additions:
        total = sum(len(v) for v in additions.values())
        st.success(f"{total} genes added across {sum(1 for v in additions.values() if v)} signatures")
        with st.expander("Added genes"):
            for sid, candidates in additions.items():
                if candidates and sid in st.session_state.signatures:
                    st.caption(f"**{st.session_state.signatures[sid].signature_name}**: " + "; ".join(
                        f"{c['gene']} (linked to {', '.join(c['supporters'][:4])})" for c in candidates))


# ============================================================
# LAYER 4 - REVIEW / DEBATE
# ============================================================

def deselect(signature_id: str, gene: str, reason: str):
    """Mark a candidate gene as not selected, remembering why."""
    st.session_state.deselected.setdefault(signature_id, {})[gene] = reason
    invalidate_approval(signature_id)


def clear_deselections(signature_id: str, prefix: str):
    """Forget earlier deselections made by one layer (e.g. a previous debate)."""
    current = st.session_state.deselected.get(signature_id, {})
    st.session_state.deselected[signature_id] = {
        gene: reason for gene, reason in current.items() if not reason.startswith(prefix)}


def render_layer4(resources: Dict):
    st.markdown("---")
    st.markdown("### Layer 4 - Verification (optional)")
    method = st.radio("Method", ["Single-model review", "Three-agent debate"], horizontal=True,
                      key="verification_method")
    if method == "Single-model review":
        render_review(resources)
    else:
        render_debate()


def render_review(resources: Dict):
    st.caption("One model reviews each signature and proposes removals and additions. "
               "Proposed removals are deselected by default; you decide in Layer 5.")
    col1, col2, col3 = st.columns(3)
    model = model_selectbox(col1, "Model", reasoning_model(), "review_model_choice")
    additions = col2.checkbox("Suggest additions", value=True)
    batch_size = col3.slider("Signatures per call", 1, 8, 4)

    if st.button("Run review", type="primary", width="stretch"):
        signatures = ordered_signatures()
        started = time.time()
        status = st.empty()
        suggestions, calls = verify_signatures(
            get_router(), model, signatures, st.session_state.query_used,
            suggest_additions=additions, context=st.session_state.bio_context,
            publication_mode=st.session_state.generation_mode == 'publication',
            batch_size=batch_size, fallbacks=fallbacks_for(model), index=resources['index'],
            progress=lambda done, total: status.info(f"Reviewing batch {done}/{total}..."))
        status.empty()

        st.session_state.suggestions = suggestions
        st.session_state.review_calls = calls
        st.session_state.review_model = model
        for sig in signatures:
            result = suggestions.get(sig.signature_id, {})
            clear_deselections(sig.signature_id, "llm review")
            if result.get('status') == 'ok':
                sig.llm2_verified = True
                for gene in result['genes_to_remove']:
                    deselect(sig.signature_id, gene,
                             f"llm review: {result['reasoning'].get(gene) or 'no reason given'}")
        record_timing("Layer 4: review", started)
        st.rerun()

    suggestions = st.session_state.suggestions
    if suggestions:
        ok = sum(1 for s in suggestions.values() if s['status'] == 'ok')
        failed = len(suggestions) - ok
        if failed:
            errors = sorted({s['error'] for s in suggestions.values() if s['status'] == 'failed'})
            st.error(f"Review failed for {failed} of {len(suggestions)} signatures - they were "
                     f"NOT reviewed.\n\n" + "\n\n".join(f"`{e}`" for e in errors[:3]))
        if ok:
            removals = sum(len(s['genes_to_remove']) for s in suggestions.values())
            adds = sum(len(s['genes_to_add']) for s in suggestions.values())
            calls = st.session_state.review_calls
            used = sorted({c.get('model') for c in calls if c.get('model')})
            st.success(f"Reviewed {ok} signatures with `{', '.join(used) or st.session_state.review_model}`: "
                       f"{removals} removals and {adds} additions proposed (see Layer 5).")
            for note in sorted({n for c in calls for n in c.get('notes', [])}):
                st.caption(f"Note: {note}")


def render_debate():
    st.caption("Three models with opposed burdens of proof vote keep/remove on every gene, "
               "seeing per-gene database evidence. Contested genes are re-argued. A gene is "
               "voted out only on a confidence-weighted majority.")
    available = available_models()
    defaults = resolve_debate_models(available)

    cols = st.columns(3)
    models = {}
    for col, role in zip(cols, DEBATE_ROLES):
        models[role] = model_selectbox(col, ROLE_LABELS[role], defaults.get(role),
                                       f"debate_model_{role}")
    diversity = debate_diversity(models)
    if diversity['distinct_models'] < 3:
        st.warning(f"Only {diversity['distinct_models']} distinct model(s) across the three roles. "
                   "Agreement between copies of one model is weak evidence - connect another "
                   "provider in the sidebar for an independent voice.")
    else:
        st.caption(f"{diversity['distinct_models']} distinct models from "
                   f"{diversity['distinct_families']} model families. Connecting more providers "
                   "(or OpenRouter) lets the three roles come from three different families.")

    col1, col2, col3 = st.columns(3)
    max_rounds = col1.slider("Max rounds", 1, 4, 3)
    agreement_threshold = col2.slider("Stop at agreement", 0.5, 1.0, 1.0, 0.05,
                                      help="Fraction of genes with unanimous votes.")
    removal_threshold = col3.slider("Removal threshold", 0.5, 0.9, DEFAULT_REMOVAL_THRESHOLD, 0.05,
                                    help="Pooled probability of removal required (plus a majority). "
                                         "0.6 means clearly above a coin flip.")
    auto_apply = st.checkbox("Deselect the genes the agents vote out (you can re-select them in Layer 5)",
                             value=True)

    backend_up = backend_health()
    if not backend_up:
        st.warning("Pathway API offline: the debate will run WITHOUT database evidence and will "
                   "be labelled as such in the export.\n\n" + WAKE_BACKEND_HINT)

    pending = ordered_signatures()
    estimate = sum(estimate_tokens_per_model(len(s.genes), max_rounds, backend_up) for s in pending)
    st.caption(
        f"Up to about {estimate:,} tokens per model for {len(pending)} signatures (usually less: "
        "most debates stop early). Free-tier keys cap tokens per minute and per day, so large "
        "runs pause between calls. A model that runs out of quota is replaced automatically "
        "and the substitution is recorded in the export.")

    if st.button("Start debate", type="primary", width="stretch"):
        engine = DebateEngine(get_router(), models, fallbacks=available,
                              removal_threshold=removal_threshold)
        db = get_db_client()
        context = st.session_state.bio_context
        signatures = ordered_signatures()
        started = time.time()
        progress = st.progress(0.0)
        status = st.empty()
        problems = []

        for i, sig in enumerate(signatures):
            status.info(f"Debating {i + 1}/{len(signatures)}: {sig.signature_name}")
            try:
                result, evidence_error = debate_signature(
                    engine, sig, db=db if backend_up else None,
                    query=st.session_state.query_used, context=context,
                    max_rounds=max_rounds, agreement_threshold=agreement_threshold)
            except Exception as exc:
                problems.append(f"{sig.signature_name}: debate failed ({str(exc)[:200]})")
                progress.progress((i + 1) / len(signatures))
                continue

            if evidence_error:
                problems.append(f"{sig.signature_name}: no database evidence ({evidence_error})")
            st.session_state.debates[sig.signature_id] = result
            clear_deselections(sig.signature_id, "debate")
            if result.failed:
                problems.append(f"{sig.signature_name}: fewer than two agents answered - "
                                "no decision was taken.")
            else:
                # Metrics go onto the signature; the votes become default
                # deselections rather than deletions, so the user has the last word.
                apply_debate_result(sig, result, apply_removals=False)
                if auto_apply:
                    for gene in result.genes_to_remove:
                        decision = result.decisions[gene]
                        deselect(sig.signature_id, gene,
                                 f"debate: P(remove)={decision.p_remove:.2f}, "
                                 f"{decision.n_remove}/{decision.n_voters} agents")
                invalidate_approval(sig.signature_id)
            progress.progress((i + 1) / len(signatures))

        progress.empty()
        status.empty()
        st.session_state.debate_models = models
        st.session_state.debate_problems = problems
        record_timing("Layer 4: debate", started)
        st.rerun()

    for problem in st.session_state.debate_problems:
        st.warning(problem)

    debates = st.session_state.debates
    if not debates:
        return

    st.markdown("#### Debate results")
    rows = []
    for sid, result in debates.items():
        sig = st.session_state.signatures.get(sid)
        if not sig:
            continue
        m = result.metrics
        rows.append({
            'Signature': sig.signature_name, 'Agents': m['agents_responding'],
            'Rounds': m['rounds'], 'Stop': m['stop_reason'], 'Voted out': m['genes_removed'],
            'Stability': m['stability'], 'Agreement': m['agreement'],
            'Fleiss κ': m['fleiss_kappa'], 'Entropy': m['entropy'], 'Conflict': m['conflict'],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    by_name = {st.session_state.signatures[sid].signature_name: result
               for sid, result in debates.items() if sid in st.session_state.signatures}
    if not by_name:
        return
    result = by_name[st.selectbox("Inspect a debate", list(by_name),
                                  key=f"debate_view_{st.session_state.run_id}")]
    st.caption("Models that answered: " + " · ".join(
        f"{ROLE_LABELS[r]} = `{m}`" for r, m in result.models.items()))
    concordance = result.metrics['database_concordance']
    st.caption(f"Database concordance: {concordance['flagged_and_removed']} of "
               f"{concordance['flagged_genes']} database-flagged genes voted out, "
               f"{concordance['flagged_but_kept']} kept; "
               f"{concordance['removed_without_flag']} voted out without any database flag.")

    st.dataframe(pd.DataFrame([{
        'Gene': d.gene, 'Decision': d.decision,
        'P(remove)': None if d.p_remove is None else round(d.p_remove, 2),
        'Entropy': None if d.entropy is None else round(d.entropy, 2),
        'Conflict': None if d.conflict is None else round(d.conflict, 2),
        **{ROLE_LABELS[role]: f"{v['vote']} ({v['confidence']:.2f})" for role, v in d.votes.items()},
        'DB flags': ", ".join(d.db_flags),
    } for d in result.decisions.values()]), hide_index=True, width="stretch")

    with st.expander("Arguments by round"):
        for debate_round in result.rounds:
            st.markdown(f"**Round {debate_round.round_num}** - agreement "
                        f"{debate_round.agreement:.0%}, {debate_round.flips} vote(s) changed")
            for turn in debate_round.turns:
                label = f"{ROLE_LABELS.get(turn.role, turn.role)} (`{turn.model}`)"
                if turn.error:
                    st.error(f"{label} did not answer: {turn.error}")
                    continue
                reasons = [f"{g}: {turn.votes[g].vote} - {turn.votes[g].reason}"
                           for g in turn.revoted if turn.votes[g].reason]
                # st.text never interprets model output as HTML or markdown.
                st.caption(label)
                st.text("\n".join(reasons) if reasons else "(no reasons given)")
                for note in turn.llm.get('notes', []):
                    st.caption(f"Note: {note}")


# ============================================================
# LAYER 5 - APPROVAL
# ============================================================

def selected_genes(sig: GeneSignature) -> List[str]:
    """Candidate genes minus the current deselections, in ranked order."""
    off = st.session_state.deselected.get(sig.signature_id, {})
    return [g for g in sig.genes if g not in off]


def render_layer5():
    st.markdown("---")
    st.markdown("### Layer 5 - Review & approve")
    signatures = ordered_signatures()
    approved = st.session_state.approved

    col1, col2 = st.columns(2)
    if col1.button("Approve all", width="stretch"):
        for sig in signatures:
            genes = selected_genes(sig)
            if genes:
                approved[sig.signature_id] = genes
        st.rerun()
    if col2.button("Clear approvals", width="stretch"):
        st.session_state.approved = {}
        st.rerun()
    st.caption(f"{len(approved)} of {len(signatures)} signatures approved")

    minimum = (st.session_state.retrieval_params or {}).get('min_genes', 0)
    for sig in signatures:
        sid = sig.signature_id
        current = selected_genes(sig)
        mark = "✅ " if sid in approved else ""
        with st.expander(f"{mark}{sig.signature_name} ({len(current)} of {len(sig.genes)} genes)"):
            tags = [f"support {sig.support():.2f}", f"retrieval: {sig.retrieval_mode}"]
            if sig.dam_expanded:
                tags.append("network-expanded")
            if sig.debate_verified and sig.stability_score is not None:
                tags.append(f"debate stability {sig.stability_score:.2f} ({sig.quality_label()})")
            st.caption(" · ".join(tags))
            if sig.description:
                st.caption(sig.description)
            if len(current) < minimum:
                st.warning(f"Only {len(current)} genes are selected - below the minimum of "
                           f"{minimum}. The retrieved set may have been too broad for this "
                           "mechanism; re-select genes below if the votes were too strict.")

            suggestion = st.session_state.suggestions.get(sid, {})
            if suggestion.get('status') == 'failed':
                st.warning("LLM review failed for this signature - it has not been reviewed.")
            off = st.session_state.deselected.get(sid, {})
            voted_out = {g: why for g, why in off.items() if g in sig.genes and why != "user"}
            if voted_out:
                st.caption("Deselected by verification (re-select to overrule): " + "; ".join(
                    f"{g} [{why}]" for g, why in voted_out.items()))
            if sig.housekeeping_flagged and st.session_state.generation_mode != 'publication':
                present = [g for g in current if is_housekeeping(g)]
                if present:
                    st.caption("Housekeeping genes present: " + ", ".join(present))
            if sig.removed_genes:
                st.caption("Removed at build time: " + "; ".join(
                    f"{g} ({why})" for g, why in sig.removed_genes.items()))

            # The widget key includes the candidate list and the deselections,
            # so the picker is rebuilt whenever an upstream layer changes either.
            chosen = st.multiselect(
                "Genes (ranked; remove any you reject)", options=sig.genes, default=current,
                key=f"genes_{sid}_{st.session_state.run_id}_{hash(tuple(sig.genes))}_"
                    f"{hash(frozenset(off))}")
            now_off = set(sig.genes) - set(chosen)
            if now_off != set(g for g in off if g in sig.genes):
                st.session_state.deselected[sid] = {g: off.get(g, "user") for g in sig.genes
                                                    if g in now_off}
                invalidate_approval(sid)
                st.rerun()

            proposals = [g for g in suggestion.get('genes_to_add', []) if g not in sig.genes]
            if proposals:
                st.markdown("**LLM-suggested additions**")
                for col, gene in zip(st.columns(len(proposals)), proposals):
                    if col.button(f"Add {gene}", key=f"add_{sid}_{gene}",
                                  help=suggestion['reasoning'].get(gene) or None):
                        sig.genes.append(gene)
                        sig.gene_sources[gene] = 'llm'
                        invalidate_approval(sid)
                        st.rerun()

            col1, col2 = st.columns(2)
            if col1.button("Approve", key=f"approve_{sid}", width="stretch", disabled=not current):
                approved[sid] = current
                st.rerun()
            if sid in approved and col2.button("Withdraw approval", key=f"withdraw_{sid}",
                                               width="stretch"):
                approved.pop(sid)
                st.rerun()


# ============================================================
# EXPORT
# ============================================================

def approved_signatures() -> List[GeneSignature]:
    """Frozen copies of approved signatures: exactly the approved genes, with every
    gene left out recorded alongside the reason."""
    out = []
    for sid, genes in st.session_state.approved.items():
        sig = st.session_state.signatures.get(sid)
        if not sig:
            continue
        kept = set(genes)
        reasons = st.session_state.deselected.get(sid, {})
        removed = {**sig.removed_genes,
                   **{g: reasons.get(g, "user") for g in sig.genes if g not in kept}}
        out.append(dataclasses.replace(sig, genes=list(genes), removed_genes=removed))
    return out


def render_export(resources: Dict):
    signatures = approved_signatures()
    if not signatures:
        return
    st.markdown("---")
    st.markdown("### Export")
    st.success(f"{len(signatures)} approved signatures")

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    router = get_router()
    provenance = build_provenance(
        resources, pipeline_version=PIPELINE_VERSION, debate_version=DEBATE_VERSION,
        llm_providers=router.public(), reasoning_effort=router.effort or EFFORT_DEFAULT,
        pathway_api_url=MSIGDB_API_URL,
        decomposition=st.session_state.l1_provenance, mechanisms=st.session_state.facets,
        mechanisms_selected=st.session_state.selected_facet_ids,
        retrieval_parameters=st.session_state.retrieval_params,
        network_expansion=st.session_state.expansion_settings,
        debate_models=st.session_state.debate_models)
    debates = {sid: result.to_dict() for sid, result in st.session_state.debates.items()
               if sid in st.session_state.approved}
    review = {
        'model': st.session_state.review_model, 'calls': st.session_state.review_calls,
        'suggestions': {sid: s for sid, s in st.session_state.suggestions.items()
                        if sid in st.session_state.approved},
    } if st.session_state.suggestions else {}

    col1, col2, col3 = st.columns(3)
    col1.download_button("Download GMT", build_gmt(signatures),
                         file_name=f"signatures_{stamp}.gmt", mime="text/plain", width="stretch")
    col2.download_button("Download CSV", build_csv(signatures),
                         file_name=f"signatures_{stamp}.csv", mime="text/csv", width="stretch")
    col3.download_button(
        "Download JSON (full provenance)",
        build_json(signatures, query=st.session_state.query_used,
                   generation_mode=st.session_state.generation_mode,
                   biological_context=st.session_state.bio_context, provenance=provenance,
                   debates=debates, verification=review,
                   timings=st.session_state.layer_timings),
        file_name=f"signatures_{stamp}.json", mime="application/json", width="stretch")
    st.caption("The JSON records every parameter, provider and model, knowledge-base hash, "
               "alias mapping and debate vote needed to reproduce and audit this run. "
               "API keys are never exported.")


# ============================================================
# MAIN
# ============================================================

def main():
    st.set_page_config(page_title="MSigDB Signature Generator", page_icon="🧬", layout="wide")
    initialize_session_state()
    connect_from_environment()

    st.title("🧬 MSigDB Signature Generator")
    st.caption("LLM-proposed mechanisms · database-decided genes · evidence-grounded debate")

    try:
        resources = load_index()
    except Exception as exc:
        resources = None
        st.error(f"Failed to load the knowledge base: {exc}")

    render_sidebar()
    tab_generate, tab_kb = st.tabs(["Generate", "Knowledge base"])
    with tab_generate:
        render_generation_tab(resources)
    with tab_kb:
        render_kb_tab(resources)


if __name__ == "__main__":
    main()

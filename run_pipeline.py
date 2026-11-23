#!/usr/bin/env python
"""
Headless runner: research question in, provenanced gene signatures out
======================================================================

Runs exactly the same code as the Streamlit app, without a browser, so a
result can be regenerated from a single command line - which is what a
methods section needs.

    export GROQ_API_KEY=gsk_...            # and/or OPENAI_API_KEY, ANTHROPIC_API_KEY,
                                           # GEMINI_API_KEY, DEEPSEEK_API_KEY, ...
    python run_pipeline.py "T cell exhaustion in the tumor microenvironment" \\
        --mechanisms 8 --tissue "Blood/PBMC" --disease melanoma \\
        --expand --debate --out results/exhaustion

    python run_pipeline.py --check          # providers, models, debate roles, backend, data

Every provider whose key is in the environment is connected (see providers.py
for the variable names). Models are written ``provider:model``, e.g.
``--model anthropic:claude-opus-5-5``. With several providers connected the
three debate roles are drawn from different model families automatically.

Writes <out>.json (full provenance), <out>.gmt and <out>.csv.
Every signature that survives is exported; nothing is hand-curated here.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

from db_client import BackendUnavailable, DatabaseClient, strength_to_min_prob
from debate_engine import DEBATE_VERSION, DEFAULT_REMOVAL_THRESHOLD, DebateEngine
from exporters import build_csv, build_gmt, build_json, build_provenance
from llm_client import EFFORT_LEVELS, LLMRouter, connect_provider
from model_registry import (
    DEBATE_ROLES, MSIGDB_API_URL, debate_diversity, fallback_chain, pick_default,
    resolve_debate_models,
)
from ncbi_client import NCBIGene
from pipeline import (
    MAX_MECHANISMS, PIPELINE_VERSION, apply_debate_result, apply_network_expansion,
    apply_review, build_signatures, debate_signature, decompose_query, load_resources,
    resolve_seed_aliases, retrieved_genes, sentence_encoder, verify_signatures,
)
from providers import PROVIDERS, Connection, providers_in_environment
from signature_engine import ENGINE_VERSION, RetrievalParams, overlap_summary


def log(message: str = "") -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate evidence-grounded gene signatures for a research question.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("query", nargs="?", help="research question")
    parser.add_argument("--check", action="store_true",
                        help="report live models, debate roles, backend and data, then exit")
    parser.add_argument("--out", default="signatures", help="output path prefix")
    parser.add_argument("--provider", default=None, choices=sorted(PROVIDERS),
                        help="connect this provider with --api-key (default: every provider "
                             "whose key is in the environment)")
    parser.add_argument("--api-key", default=None, help="API key for --provider")
    parser.add_argument("--base-url", default=None,
                        help="endpoint for --provider (custom / ollama / self-hosted)")
    parser.add_argument("--model", default=None,
                        help="reasoning model as provider:model (default: best live model)")
    parser.add_argument("--effort", default=None, choices=EFFORT_LEVELS,
                        help="reasoning effort (default: each provider's own default)")

    layer1 = parser.add_argument_group("Layer 1 - mechanisms")
    layer1.add_argument("--mechanisms", type=int, default=10, help=f"3-{MAX_MECHANISMS}")
    layer1.add_argument("--publication", action="store_true",
                        help="publication mode: drop housekeeping genes, strict review")
    layer1.add_argument("--species", default="Homo sapiens")
    layer1.add_argument("--tissue", default=None)
    layer1.add_argument("--disease", default=None)
    layer1.add_argument("--cell-type", default=None)
    layer1.add_argument("--treatment", default=None)
    layer1.add_argument("--no-ncbi", action="store_true",
                        help="do not resolve gene aliases through NCBI Gene")

    layer2 = parser.add_argument_group("Layer 2 - retrieval")
    layer2.add_argument("--per-mechanism", type=int, default=1,
                        help="signatures per mechanism (>1 adds sub-modules)")
    layer2.add_argument("--min-genes", type=int, default=10)
    layer2.add_argument("--max-genes", type=int, default=20)
    layer2.add_argument("--top-k", type=int, default=RetrievalParams().top_k,
                        help="gene sets kept per mechanism")
    layer2.add_argument("--fdr", type=float, default=0.05, help="enrichment FDR threshold")
    layer2.add_argument("--min-support", type=int, default=2)
    layer2.add_argument("--max-jaccard", type=float, default=0.5,
                        help="signatures more similar than this are dropped")
    layer2.add_argument("--no-semantic", action="store_true",
                        help="seed enrichment only (no embedding model needed)")

    layer3 = parser.add_argument_group("Layer 3 - network expansion")
    layer3.add_argument("--expand", action="store_true")
    layer3.add_argument("--strength", type=float, default=0.5)
    layer3.add_argument("--max-additions", type=int, default=3)

    layer4 = parser.add_argument_group("Layer 4 - verification")
    layer4.add_argument("--review", action="store_true", help="single-model review")
    layer4.add_argument("--apply-review", action="store_true",
                        help="remove genes the reviewer flags (default: record only)")
    layer4.add_argument("--debate", action="store_true", help="three-agent debate")
    layer4.add_argument("--debate-models", default=None,
                        help="override roles, e.g. "
                             "skeptic=openai:gpt-5,discoverer=anthropic:claude-opus-5-5,"
                             "mediator=google:gemini-2.5-pro")
    layer4.add_argument("--rounds", type=int, default=3)
    layer4.add_argument("--agreement", type=float, default=1.0,
                        help="stop the debate at this fraction of unanimous genes")
    layer4.add_argument("--removal-threshold", type=float, default=DEFAULT_REMOVAL_THRESHOLD)
    layer4.add_argument("--no-apply-debate", action="store_true",
                        help="record debate decisions without removing genes")

    args = parser.parse_args(argv)
    if not args.check and not args.query:
        parser.error("a research question is required (or use --check)")
    return args


def backend_status(db: DatabaseClient) -> Optional[Dict]:
    try:
        return db.health_check()
    except Exception:
        return None


def connect(args: argparse.Namespace) -> Dict[str, Connection]:
    """Connect --provider/--api-key if given, else every provider with a key in the environment."""
    wanted: Dict[str, Dict] = {}
    if args.provider:
        wanted[args.provider] = {'api_key': args.api_key or '', 'base_url': args.base_url}
    else:
        wanted = {pid: {'api_key': key, 'base_url': None}
                  for pid, key in providers_in_environment().items()}
    connections: Dict[str, Connection] = {}
    for provider_id, settings in wanted.items():
        try:
            connections[provider_id] = connect_provider(provider_id, **settings)
        except Exception as exc:
            log(f"  {PROVIDERS[provider_id].label}: could not connect ({str(exc)[:200]})")
    return connections


def parse_debate_models(text: Optional[str]) -> Dict[str, str]:
    roles: Dict[str, str] = {}
    for part in (text or "").split(","):
        role, sep, model = part.strip().partition("=")
        if sep and role in DEBATE_ROLES and model:
            roles[role] = model
    return roles


def run_check(connections: Dict[str, Connection]) -> int:
    """Diagnostics: is everything the pipeline needs actually available?"""
    ok = True
    try:
        resources = load_resources()
        index = resources['index']
        print(f"Knowledge base : {index.n_pathways:,} gene sets, {index.n_genes:,} genes, "
              f"version {resources['kb_metadata'].get('version')}")
        print(f"                 sha256 {resources['kb_sha256']}")
        print(f"Embeddings     : {resources['embedding_model']}, sha256 {resources['embeddings_sha256']}")
    except Exception as exc:
        ok = False
        print(f"Knowledge base : FAILED - {exc}")

    health = backend_status(DatabaseClient(MSIGDB_API_URL))
    if health:
        print(f"Pathway API    : online (v{health.get('version')}), "
              f"databases: {', '.join(health.get('sqlite', []))}")
    else:
        print(f"Pathway API    : OFFLINE ({MSIGDB_API_URL}) - expansion unavailable, "
              "debate would run without database evidence")

    try:
        hits = NCBIGene().lookup(["GLUT1"])
        print(f"NCBI Gene      : online (GLUT1 -> {', '.join(h['symbol'] for h in hits['GLUT1']) or '?'})")
    except Exception as exc:
        print(f"NCBI Gene      : unavailable ({str(exc)[:80]}) - aliases would not be resolved")

    if not connections:
        names = ", ".join(sorted({name for p in PROVIDERS.values() for name in p.env_keys}))
        print("LLM providers  : none connected. Set one of: " + names)
        return 1
    models: List[str] = []
    for connection in connections.values():
        refs = connection.refs()
        models.extend(refs)
        print(f"{connection.provider.label:15s}: {len(refs)} chat models"
              + (f"  ({connection.warning})" if connection.warning else ""))
        for ref in refs[:8]:
            print(f"                 {ref}")
        if len(refs) > 8:
            print(f"                 ... and {len(refs) - 8} more")
    if not models:
        print("No usable chat model is available.")
        return 1
    print(f"Reasoning model: {pick_default(models)}")
    roles = resolve_debate_models(models)
    diversity = debate_diversity(roles)
    print(f"Debate roles   : {roles}")
    print(f"                 {diversity['distinct_models']} distinct models, "
          f"{diversity['distinct_families']} model families")
    if diversity['distinct_models'] < 3:
        print("                 WARNING: fewer than three distinct models are available")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    connections = connect(args)
    if args.check:
        return run_check(connections)
    if not connections:
        log("No LLM provider connected. Set a provider key in the environment "
            "(e.g. GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, "
            "DEEPSEEK_API_KEY) or pass --provider and --api-key.")
        return 2
    llm = LLMRouter(connections, effort=args.effort)

    timings: Dict[str, str] = {}

    def timed(name: str, started: float) -> None:
        timings[name] = f"{time.time() - started:.1f}s"

    context = {'species': args.species, 'tissue': args.tissue, 'disease': args.disease,
               'cell_type': args.cell_type, 'treatment': args.treatment}
    if not any(context[k] for k in ('tissue', 'disease', 'cell_type', 'treatment')):
        context = None
    if args.publication and not (context and context['tissue'] and context['disease']):
        log("Publication mode needs --tissue and --disease.")
        return 2

    log("Loading knowledge base...")
    resources = load_resources()
    index = resources['index']

    models = llm.models()
    if not models:
        log("No usable chat model is available on the connected providers.")
        return 2
    model = args.model or pick_default(models)
    fallbacks = fallback_chain(model, models)

    # ---------------------------------------------------------------- Layer 1
    started = time.time()
    log(f"Layer 1: {model} proposing {args.mechanisms} mechanisms...")
    facets, decomposition = decompose_query(llm, model, args.query, args.mechanisms,
                                            context=context, fallbacks=fallbacks, index=index)
    for note in decomposition.notes:
        log(f"  note: {note}")
    aliases: Dict = {'enabled': not args.no_ncbi}
    if not args.no_ncbi:
        try:
            aliases.update(resolve_seed_aliases(facets, index, NCBIGene().lookup))
            if aliases.get('looked_up'):
                log(f"  NCBI Gene: {aliases['resolved']} alias(es) resolved, "
                    f"{aliases['ambiguous']} ambiguous, {aliases['unresolved']} not found")
        except Exception as exc:
            aliases['error'] = f"NCBI unavailable, aliases not resolved: {str(exc)[:160]}"
            log(f"  {aliases['error']}")
    timed("Layer 1: mechanisms", started)
    l1_provenance = {'requested': args.mechanisms, 'returned': len(facets),
                     'alias_resolution': aliases, **decomposition.provenance()}

    # ---------------------------------------------------------------- Layer 2
    started = time.time()
    params = RetrievalParams(
        top_k=args.top_k, fdr_threshold=args.fdr, min_support=args.min_support,
        min_genes=args.min_genes, max_genes=args.max_genes, dedup_jaccard=args.max_jaccard,
        use_semantic=not args.no_semantic, drop_housekeeping=args.publication)
    encode = None
    if not args.no_semantic:
        try:
            encode = sentence_encoder(model_name=resources['embedding_model'])
        except Exception as exc:
            log(f"  embedding model unavailable ({exc}); using seed enrichment only")
    signatures, retrieval_info = build_signatures(index, facets, encode, params,
                                                  n_variants=args.per_mechanism)
    timed("Layer 2: retrieval", started)
    log(f"Layer 2: {len(signatures)} signatures from {len(facets)} mechanisms")
    if not signatures:
        log("No signature could be built. Try more mechanisms or a lower --min-genes.")
        return 1

    db = DatabaseClient(MSIGDB_API_URL)
    backend_up = backend_status(db) is not None if (args.expand or args.debate) else False

    # ---------------------------------------------------------------- Layer 3
    expansion_settings = None
    if args.expand:
        if not backend_up:
            log("Layer 3: skipped - pathway API offline")
        else:
            started = time.time()
            added_total = 0
            for sig in signatures:
                try:
                    candidates = db.network_expansion(
                        retrieved_genes(sig), strength=args.strength,
                        max_additions=args.max_additions, drop_housekeeping=args.publication)
                except BackendUnavailable as exc:
                    log(f"  {sig.signature_id}: expansion failed ({exc})")
                    continue
                added_total += len(apply_network_expansion(sig, candidates))
            expansion_settings = {
                'strength': args.strength,
                'min_link_probability': strength_to_min_prob(args.strength),
                'max_additions': args.max_additions, 'min_supporting_signature_genes': 2}
            timed("Layer 3: network expansion", started)
            log(f"Layer 3: {added_total} genes added")

    # ---------------------------------------------------------------- Layer 4
    review_record: Dict = {}
    if args.review:
        started = time.time()
        log(f"Layer 4: {model} reviewing {len(signatures)} signatures...")
        suggestions, calls = verify_signatures(
            llm, model, signatures, args.query, context=context,
            publication_mode=args.publication, fallbacks=fallbacks, index=index)
        failed = sum(1 for s in suggestions.values() if s['status'] == 'failed')
        if failed:
            log(f"  WARNING: review FAILED for {failed} signatures - they were not reviewed")
        for sig in signatures:
            apply_review(sig, suggestions[sig.signature_id], remove=args.apply_review)
        review_record = {'model': model, 'calls': calls, 'suggestions': suggestions,
                         'removals_applied': args.apply_review}
        timed("Layer 4: review", started)

    debates: Dict[str, Dict] = {}
    debate_models: Dict[str, str] = {}
    if args.debate:
        started = time.time()
        debate_models = {**resolve_debate_models(models), **parse_debate_models(args.debate_models)}
        diversity = debate_diversity(debate_models)
        log(f"Layer 4: debate with {debate_models} "
            f"({diversity['distinct_models']} models, {diversity['distinct_families']} families)")
        if not backend_up:
            log("  WARNING: pathway API offline - debating WITHOUT database evidence")
        engine = DebateEngine(llm, debate_models, fallbacks=models,
                              removal_threshold=args.removal_threshold)
        for i, sig in enumerate(signatures, start=1):
            try:
                result, evidence_error = debate_signature(
                    engine, sig, db=db if backend_up else None, query=args.query,
                    context=context, max_rounds=args.rounds,
                    agreement_threshold=args.agreement)
            except Exception as exc:
                log(f"  [{i}/{len(signatures)}] {sig.signature_id}: debate failed ({str(exc)[:160]})")
                continue
            if evidence_error:
                log(f"  {sig.signature_id}: no database evidence ({evidence_error})")
            debates[sig.signature_id] = result.to_dict()
            if result.failed:
                log(f"  [{i}/{len(signatures)}] {sig.signature_id}: fewer than two agents "
                    "answered - no decision taken")
                continue
            removed = apply_debate_result(sig, result, apply_removals=not args.no_apply_debate)
            metrics = result.metrics
            log(f"  [{i}/{len(signatures)}] {sig.signature_id}: {metrics['rounds']} round(s), "
                f"{metrics['stop_reason']}, removed {len(removed)}, "
                f"stability {metrics['stability']}, kappa {metrics['fleiss_kappa']}")
        timed("Layer 4: debate", started)

    # ----------------------------------------------------------------- export
    kept = [s for s in signatures if s.genes]
    provenance = build_provenance(
        resources, pipeline_version=PIPELINE_VERSION, debate_version=DEBATE_VERSION,
        llm_providers=llm.public(), reasoning_effort=args.effort or "provider default",
        pathway_api_url=MSIGDB_API_URL,
        decomposition=l1_provenance, mechanisms=facets,
        mechanisms_selected=[f['facet_id'] for f in facets],
        retrieval_parameters=retrieval_info, network_expansion=expansion_settings,
        debate_models=debate_models)
    provenance['command_line'] = {k: v for k, v in vars(args).items() if k != 'api_key'}

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as handle:
        handle.write(build_json(
            kept, query=args.query,
            generation_mode='publication' if args.publication else 'exploratory',
            biological_context=context, provenance=provenance, debates=debates,
            verification=review_record, timings=timings))
    with open(args.out + ".gmt", "w", encoding="utf-8") as handle:
        handle.write(build_gmt(kept))
    with open(args.out + ".csv", "w", encoding="utf-8") as handle:
        handle.write(build_csv(kept))

    overlap = overlap_summary(kept)
    print(f"\nEngine {ENGINE_VERSION} · {len(kept)} signatures · mean pairwise Jaccard "
          f"{overlap['mean_jaccard']:.3f} (max {overlap['max_jaccard']:.2f})")
    print(f"{'signature':46s} {'genes':>5s} {'support':>7s} {'seeds':>6s} {'stability':>9s}  top genes")
    for sig in kept:
        stability = "   n/a" if sig.stability_score is None else f"{sig.stability_score:6.2f}"
        print(f"{sig.signature_id[:46]:46s} {len(sig.genes):5d} {sig.support():7.2f} "
              f"{sig.current_seed_recovery():6.0%} {stability:>9s}  {', '.join(sig.genes[:8])}")
    print(f"\nWrote {args.out}.json, {args.out}.gmt, {args.out}.csv")
    for name, duration in timings.items():
        print(f"  {name}: {duration}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

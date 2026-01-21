"""
Biological Signature Generator - PUBLICATION-GRADE VERSION WITH DEBATE SYSTEM
=============================================================================

FIXES APPLIED:
✅ Fix A: DAM loader diagnostics
✅ Fix B: Token UI race condition  
✅ Fix C: Signature deduplication
✅ Fix D: Signature ID mapping in verification
✅ Fix E: LLM logging
✅ Root Cause #1: Use ALL mechanism queries (removed [:1])
✅ Root Cause #2: Dynamic variant allocation
✅ DAM: Remote API integration
✅ UI: Text-based gene selection

PUBLICATION-GRADE ENHANCEMENTS:
✅ Mode toggle (Exploratory vs Publication)
✅ Biological context collection
✅ Housekeeping gene filter
✅ Context-aware verification
✅ Mode + context in exports

NEW - DEBATE SYSTEM:
✅ Multi-round AI debate for signature validation
✅ Database-grounded evidence injection
✅ Material UI conversational chat interface
✅ 3 LLMs: Qwen, Zephyr, Phi-3
✅ Weighted voting & convergence tracking

PERFORMANCE:
- Layer 2: 10 separate searches (true diversity)
- Layer 3: Remote API (no local 1.3GB file)
- Layer 4: Batch verification OR Multi-round debate
- Total: ~2-5 minutes for 35 signatures
"""

import streamlit as st
import pandas as pd
import json
import gzip
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
import time
import numpy as np
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import hashlib
import pickle
import asyncio
import nest_asyncio
from openai import OpenAI  # NEW: For Groq API

# Original imports
from db_client import DatabaseClient
from cache_client import SearchCacheClient

def inject_minimal_styles():
    """Minimal styling that respects Streamlit's default UI"""
    st.markdown("""
    <style>
    /* Debate message styles only - does not override Streamlit defaults */
    .debate-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid;
    }

    .debate-qwen {
        background: rgba(255, 107, 157, 0.1);
        border-left-color: #FF6B9D;
    }
    .debate-zephyr {
        background: rgba(78, 205, 196, 0.1);
        border-left-color: #4ECDC4;
    }
    .debate-phi {
        background: rgba(255, 217, 61, 0.1);
        border-left-color: #FFD93D;
    }
    .debate-injector {
        background: rgba(155, 89, 182, 0.15);
        border-left-color: #9B59B6;
    }
    .debate-consensus {
        background: rgba(46, 204, 113, 0.15);
        border-left-color: #2ECC71;
    }
    </style>
    """, unsafe_allow_html=True)


def render_debate_message_simple(speaker: str, message: str, db_sources: list = None):
    """
    Simple debate message display with error detection.

    ✅ FIX BUG #3: Detects error messages and displays them distinctly
    """
    speaker_map = {
        'qwen': ('🤖 Qwen 2.5', 'qwen'),
        'zephyr': ('🤖 Zephyr', 'zephyr'),
        'phi': ('🤖 Phi-3', 'phi'),
        'injector': ('💉 Database', 'injector'),
        'consensus': ('🎯 Consensus', 'consensus')
    }

    label, css_class = speaker_map.get(speaker, ('💬 Unknown', 'qwen'))

    # ✅ Detect if this is an error message
    is_error = (
        message.startswith("Error (") or
        "failed:" in message.lower() or
        "exception" in message.lower() or
        "not found" in message.lower()
    )

    if is_error:
        # Display as error with red styling
        st.markdown(f"""
        <div style='background: rgba(235, 0, 20, 0.1); border-left: 4px solid #EB0014;
                    padding: 1rem; border-radius: 8px; margin: 0.5rem 0;'>
            <strong>❌ {label} (FAILED)</strong><br>
            <code style='color: #EB0014; background: rgba(0,0,0,0.05);
                        padding: 0.5rem; display: block; margin-top: 0.5rem;
                        border-radius: 4px; white-space: pre-wrap;'>{message}</code>
        </div>
        """, unsafe_allow_html=True)
        return

    # Normal message display
    sources_html = ""
    if db_sources:
        sources_html = f'<small style="color: #666;">📊 {", ".join(db_sources)}</small><br>'

    display_message = message if len(message) <= 500 else message[:500] + "..."

    st.markdown(f"""
    <div class="debate-message debate-{css_class}">
        <strong>{label}</strong><br>
        {sources_html}
        <div style="margin-top: 0.5rem; color: #666;">{display_message}</div>
    </div>
    """, unsafe_allow_html=True)
# NEW: Debate system imports
try:
    import importlib
    import sys

    # Force reload of debate system modules to avoid caching issues
    if 'debate_system_with_injector' in sys.modules:
        importlib.reload(sys.modules['debate_system_with_injector'])
    if 'db_client_enhanced' in sys.modules:
        importlib.reload(sys.modules['db_client_enhanced'])

    from db_client_enhanced import DatabaseClientEnhanced
    from debate_system_with_injector import (
        MultiRoundDebateEngine,
        DebateMode,
        DebateResult
    )

    # Validate that we have the correct version (Groq API with api_key parameter)
    import inspect
    sig = inspect.signature(MultiRoundDebateEngine.__init__)
    params = list(sig.parameters.keys())
    if 'api_key' not in params:
        print(f"⚠️  WARNING: MultiRoundDebateEngine has old signature: {params}")
        print("Expected 'api_key' parameter but found old 'hf_token' parameter.")
        print("This indicates a module caching issue. Please restart the Streamlit app.")
        DEBATE_SYSTEM_AVAILABLE = False
    else:
        DEBATE_SYSTEM_AVAILABLE = True
except ImportError as e:
    DEBATE_SYSTEM_AVAILABLE = False
    print(f"Debate system not available: {e}")

# Apply nest_asyncio for Streamlit compatibility
nest_asyncio.apply()


# ============================================================
# HOUSEKEEPING GENES FILTER (NEW: Publication-Grade)
# ============================================================

HOUSEKEEPING_GENES = {
    # Glycolysis/Metabolism
    'GAPDH', 'GAPDHS', 'PGK1', 'ENO1', 'PKM', 'ALDOA', 'TPI1',
    'LDHA', 'LDHB', 'G6PD', 'PFKM', 'PFKL',
    
    # Cytoskeleton
    'ACTB', 'ACTG1', 'TUBB', 'TUBA1A', 'TUBA1B', 'TUBA1C',
    'TUBB4B', 'TUBA4A', 'TUBB2A', 'TUBB3',
    
    # Translation/Protein Synthesis
    'B2M', 'PPIA', 'PPIB', 'RPLP0', 'RPL13A', 'RPS18',
    'HPRT1', 'TBP', 'YWHAZ', 'UBC', 'EEF1A1',
    
    # Heat Shock
    'HSP90AB1', 'HSPA8', 'HSPD1', 'HSPA1A', 'HSP90AA1',
}


def filter_housekeeping_genes(
    genes: List[str], 
    mode: str = 'exploratory'
) -> Tuple[List[str], List[str]]:
    """
    Filter housekeeping genes based on generation mode.
    
    Args:
        genes: List of gene symbols
        mode: 'exploratory' or 'publication'
    
    Returns:
        (filtered_genes, removed_genes)
    """
    filtered = []
    removed = []
    
    for gene in genes:
        is_housekeeping = False
        
        # Check exact match
        if gene in HOUSEKEEPING_GENES:
            is_housekeeping = True
        
        # Check ribosomal proteins (but exclude disease-relevant ones)
        elif gene.startswith(('RPL', 'RPS', 'MRPL', 'MRPS')):
            if gene not in ['RPL22', 'RPS19', 'RPS24']:
                is_housekeeping = True
        
        # Check mitochondrial
        elif gene.startswith('MT-'):
            is_housekeeping = True
        
        if is_housekeeping:
            removed.append(gene)
            # In publication mode, actually remove them
            if mode == 'publication':
                continue
        
        filtered.append(gene)
    
    return filtered, removed


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class GeneSignature:
    """A biologically meaningful gene signature"""
    signature_id: str
    signature_name: str
    genes: List[str]
    facet: str
    mechanism: str
    confidence: float
    gene_scores: Dict[str, float] = field(default_factory=dict)
    source_pathways: List[str] = field(default_factory=list)
    dam_expanded: bool = False
    llm2_verified: bool = False
    debate_verified: bool = False  # NEW
    
    def to_dict(self):
        return {
            'signature_id': self.signature_id,
            'signature_name': self.signature_name,
            'genes': self.genes,
            'gene_count': len(self.genes),
            'facet': self.facet,
            'mechanism': self.mechanism,
            'confidence': self.confidence,
            'top_genes': self.genes[:5] if len(self.genes) > 5 else self.genes,
            'dam_expanded': self.dam_expanded,
            'llm2_verified': self.llm2_verified,
            'debate_verified': self.debate_verified  # NEW
        }
    
    def gene_set_hash(self) -> str:
        """Hash of gene set for deduplication"""
        return hashlib.md5('|'.join(sorted(self.genes)).encode()).hexdigest()


@dataclass
class DecompositionResult:
    """Result from LLM decomposition"""
    facets: List[Dict[str, str]]
    granularity_level: int
    timestamp: str
    llm_model: str


@dataclass
class LayerTiming:
    """Track timing per layer"""
    layer_name: str
    start_time: float
    end_time: Optional[float] = None
    
    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def duration_str(self) -> str:
        dur = self.duration
        if dur < 60:
            return f"{dur:.1f}s"
        else:
            return f"{dur/60:.1f}m"


# ============================================================
# PRECOMPUTED LOADERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_precomputed_embeddings_or_fail(path="data/pathway_embeddings.pkl"):
    """Load precomputed pathway embeddings with diagnostics"""
    if not os.path.exists(path):
        st.error(f"❌ Embeddings file not found: {path}")
        st.info("💡 Make sure pathway_embeddings.pkl is in the data/ folder")
        st.stop()

    try:
        with open(path, "rb") as f:
            package = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load embeddings: {e}")
        st.stop()

    # FIX A: Better diagnostics
    required_keys = {"embeddings"}
    found_keys = set(package.keys())
    
    if not required_keys.issubset(found_keys):
        st.error(
            f"❌ Invalid embeddings file!\n\n"
            f"Expected keys: {required_keys}\n"
            f"Found keys: {found_keys}\n\n"
            f"💡 This looks like a DAM file, not embeddings. "
            f"Make sure you're loading pathway_embeddings.pkl"
        )
        st.stop()

    return package["embeddings"]


# ============================================================
# SIGNATURE BUILDER - WITH DEDUPLICATION & HOUSEKEEPING FILTER
# ============================================================

class SignatureBuilder:
    """Builds gene signatures with deduplication and dynamic variants"""
    
    def __init__(self, min_genes: int = 10, max_genes: int = 20):
        self.min_genes = min_genes
        self.max_genes = max_genes
        self.seen_signatures: List[Set[str]] = []  # For deduplication using Jaccard
    
    def build_signature_from_pathways(
        self,
        facet_name: str,
        mechanism_name: str,
        pathways_dict: Dict[str, List[str]],
        pathway_similarities: Dict[str, float]
    ) -> Optional[GeneSignature]:
        """Build ONE signature using relevance scoring"""
        
        if not pathways_dict:
            return None
        
        gene_scores = defaultdict(float)
        source_pathways = []
        
        for pathway_id, genes in pathways_dict.items():
            similarity = pathway_similarities.get(pathway_id, 0.0)
            source_pathways.append(pathway_id)
            
            for gene in genes:
                gene_scores[gene] += similarity
        
        if not gene_scores:
            return None
        
        ranked_genes = sorted(gene_scores.items(), key=lambda x: x[1], reverse=True)
        selected_pairs = ranked_genes[:self.max_genes]
        
        if len(selected_pairs) < self.min_genes:
            return None
        
        selected_genes = [gene for gene, score in selected_pairs]
        
        # NEW: Apply housekeeping filter
        mode = st.session_state.get('generation_mode', 'exploratory')
        selected_genes, removed_hk = filter_housekeeping_genes(selected_genes, mode)
        
        # NEW: Log what was filtered
        if removed_hk and len(removed_hk) > 0:
            if mode == 'publication':
                st.caption(f"🔬 Filtered {len(removed_hk)} housekeeping genes: {', '.join(removed_hk[:5])}{'...' if len(removed_hk) > 5 else ''}")
            else:
                st.caption(f"ℹ️ Contains {len(removed_hk)} housekeeping genes (flagged but not removed in exploratory mode)")
        
        # NEW: Check minimum AFTER filtering
        if len(selected_genes) < self.min_genes:
            # FIX: Backfill if filtering dropped below minimum
            current_set = set(selected_genes)
            removed_set = set(removed_hk)
            
            for gene, score in ranked_genes:
                if len(selected_genes) >= self.min_genes:
                    break
                
                if gene not in current_set and gene not in removed_set:
                    # Double check it isn't a housekeeping gene we missed (redundant but safe)
                    is_hk = False
                    if gene in HOUSEKEEPING_GENES: 
                        is_hk = True
                    
                    if not is_hk:
                        selected_genes.append(gene)
                        current_set.add(gene)
            
            # If still not enough, return None
            if len(selected_genes) < self.min_genes:
                return None
        
        # Rebuild scores dict with filtered genes
        scores_dict = {gene: score for gene, score in selected_pairs if gene in selected_genes}
        
        avg_score = np.mean([score for gene, score in scores_dict.items()])
        confidence = min(0.99, avg_score)
        
        sig_id = f"{self._make_id(facet_name)}_{self._make_id(mechanism_name)}"
        
        return GeneSignature(
            signature_id=sig_id,
            signature_name=f"{facet_name} - {mechanism_name}",
            genes=selected_genes,
            facet=facet_name,
            mechanism=mechanism_name,
            confidence=confidence,
            gene_scores=scores_dict,
            source_pathways=source_pathways[:5]
        )
    
    def build_multiple_mechanisms_dynamic(
        self,
        facet_name: str,
        pathways_dict: Dict[str, List[str]],
        pathway_similarities: Dict[str, float],
        num_variants: int = 3
    ) -> List[GeneSignature]:
        """
        FIX ROOT CAUSE #2: Dynamic variant generation
        Generate exactly num_variants unique signatures
        """
        
        signatures = []
        
        # Sort pathways by similarity
        sorted_pathways = sorted(
            pathway_similarities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        if len(pathways_dict) < 3:
            # Not enough pathways for variants - just do core
            core_sig = self.build_signature_from_pathways(
                facet_name, "Core Markers", pathways_dict, pathway_similarities
            )
            if core_sig:
                signatures.append(core_sig)
            return signatures
        
        # Strategy 1: Core (top pathways)
        top_count = max(3, len(sorted_pathways) // 3)
        top_ids = [pid for pid, _ in sorted_pathways[:top_count]]
        top_pathways = {pid: pathways_dict[pid] for pid in top_ids if pid in pathways_dict}
        top_sims = {pid: pathway_similarities[pid] for pid in top_ids if pid in pathway_similarities}
        
        core_sig = self.build_signature_from_pathways(
            facet_name, "Core Markers", top_pathways, top_sims
        )
        if core_sig and self._is_unique_signature(core_sig):
            signatures.append(core_sig)
        
        # Strategy 2: Extended (middle pathways)
        if len(signatures) < num_variants and len(sorted_pathways) >= 6:
            mid_start = len(sorted_pathways) // 3
            mid_end = 2 * len(sorted_pathways) // 3
            mid_ids = [pid for pid, _ in sorted_pathways[mid_start:mid_end]]
            mid_pathways = {pid: pathways_dict[pid] for pid in mid_ids if pid in pathways_dict}
            mid_sims = {pid: pathway_similarities[pid] for pid in mid_ids if pid in pathway_similarities}
            
            extended_sig = self.build_signature_from_pathways(
                facet_name, "Extended Network", mid_pathways, mid_sims
            )
            if extended_sig and self._is_unique_signature(extended_sig):
                signatures.append(extended_sig)
        
        # Strategy 3: Regulatory (bottom pathways)
        if len(signatures) < num_variants and len(sorted_pathways) >= 6:
            bottom_start = 2 * len(sorted_pathways) // 3
            bottom_ids = [pid for pid, _ in sorted_pathways[bottom_start:]]
            bottom_pathways = {pid: pathways_dict[pid] for pid in bottom_ids if pid in pathways_dict}
            bottom_sims = {pid: pathway_similarities[pid] for pid in bottom_ids if pid in pathway_similarities}
            
            regulatory_sig = self.build_signature_from_pathways(
                facet_name, "Regulatory Context", bottom_pathways, bottom_sims
            )
            if regulatory_sig and self._is_unique_signature(regulatory_sig):
                signatures.append(regulatory_sig)
        
        # If we still need more variants, try different sizes
        while len(signatures) < num_variants and len(pathways_dict) >= 3:
            # Random subset strategy
            import random
            sample_size = random.randint(3, min(10, len(pathways_dict)))
            sample_ids = random.sample(list(pathways_dict.keys()), sample_size)
            sample_pathways = {pid: pathways_dict[pid] for pid in sample_ids}
            sample_sims = {pid: pathway_similarities[pid] for pid in sample_ids if pid in pathway_similarities}
            
            variant_sig = self.build_signature_from_pathways(
                facet_name, f"Variant {len(signatures)+1}", sample_pathways, sample_sims
            )
            if variant_sig and self._is_unique_signature(variant_sig):
                signatures.append(variant_sig)
            else:
                break  # Can't generate more unique variants
        
        return signatures
    
    def _is_unique_signature(self, sig: GeneSignature) -> bool:
        """
        FIX C: Signature deduplication using Jaccard Similarity
        Allow up to 80% overlap (Jaccard similarity < 0.8)
        """
        new_gene_set = set(sig.genes)
        
        for seen_set in self.seen_signatures:
            intersection = len(new_gene_set.intersection(seen_set))
            union = len(new_gene_set.union(seen_set))
            
            if union == 0:
                continue
                
            jaccard = intersection / union
            
            # If too similar (high overlap), reject as duplicate
            if jaccard > 0.8:  # 80% similarity threshold
                return False
        
        self.seen_signatures.append(new_gene_set)
        return True

    def _make_id(self, name: str) -> str:
        return name.upper().replace(' ', '_').replace('-', '_').replace('/', '_')


# ============================================================
# OPTIMIZED SEMANTIC SEARCH
# ============================================================

@st.cache_resource
def load_embedding_model():
    """Load sentence transformer (cached as resource)"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.warning(f"Could not load embedding model: {e}")
        return None


def fast_semantic_search(
    query: str,
    pathway_embeddings: Dict[str, np.ndarray],
    pathways_dict: Dict[str, List[str]],
    model,
    top_k: int = 50
) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    """Fast semantic search using precomputed embeddings"""
    
    if not pathway_embeddings:
        return {}, {}
    
    # Encode and normalize query
    query_emb = model.encode(query, convert_to_numpy=True)
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    
    # Vectorized cosine similarity
    similarities = []
    for pid, pathway_emb in pathway_embeddings.items():
        similarity = float(np.dot(query_emb, pathway_emb))
        similarities.append((pid, similarity))
    
    # Sort and select top K
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_pathways = similarities[:top_k]
    
    # Return results
    selected_pathways = {pid: pathways_dict[pid] for pid, _ in top_pathways if pid in pathways_dict}
    similarity_dict = {pid: score for pid, score in top_pathways}
    
    return selected_pathways, similarity_dict


# ============================================================
# LLM FUNCTIONS
# ============================================================

def decompose_with_granularity(
    query: str,
    granularity_count: int,
    groq_api_key: str
) -> Optional[DecompositionResult]:
    """Use Groq's Llama 3.3 70B to decompose query into EXACTLY N mechanisms"""

    try:
        # Use OpenAI client with Groq base URL
        client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        # Use Llama 3.3 70B - Groq's most capable model
        model = "llama-3.3-70b-versatile"

        prompt = f"""Decompose this biological query into EXACTLY {granularity_count} SPECIFIC, NON-OVERLAPPING molecular mechanisms.

Query: "{query}"

REQUIREMENTS:
1. Generate EXACTLY {granularity_count} mechanisms (no more, no less)
2. Each mechanism must be maximally granular (10-20 genes per mechanism)
3. No overlap between mechanisms
4. Each mechanism must be biologically distinct and specific

Output JSON ONLY (no preamble, no markdown):
{{
  "facets": [
    {{
      "facet_id": "F1",
      "facet_name": "Mechanism Name",
      "mechanism_queries": ["search query terms"]
    }},
    ...
  ]
}}"""

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=3000,
            temperature=0.3
        )

        raw = response.choices[0].message.content
        
        # Parse JSON
        cleaned = raw.strip()
        if '```json' in cleaned:
            cleaned = cleaned.split('```json')[1].split('```')[0].strip()
        elif '```' in cleaned:
            parts = cleaned.split('```')
            if len(parts) >= 3:
                cleaned = parts[1].strip()
        
        if not cleaned.startswith('{'):
            start = cleaned.find('{')
            if start != -1:
                cleaned = cleaned[start:]
        
        data = json.loads(cleaned)
        facets = data.get('facets', [])

        return DecompositionResult(
            facets=facets,
            granularity_level=granularity_count,
            timestamp=datetime.now().isoformat(),
            llm_model=model  # Use the Groq model name
        )

    except Exception as e:
        st.error(f"Groq decomposition failed: {e}")
        return None


def verify_signatures_batch(
    signatures: List[GeneSignature],
    original_query: str,
    mode: str,
    llm2_model: str,
    groq_api_key: str,
    context_options: List[str],
    batch_size: int = 5
) -> Dict[str, Any]:
    """
    FIX D: Batch verification with proper signature ID mapping
    NEW: Context-aware verification for publication mode

    Args:
        batch_size: Number of signatures per API call

    Returns:
        Dict[signature_id, verification_result]
    """

    all_suggestions = {}

    try:
        # Use OpenAI client with Groq
        client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        # Use Llama 3.3 70B for verification (best quality)
        model_id = "llama-3.3-70b-versatile"
        
        # NEW: Get biological context if available
        bio_context = st.session_state.get('bio_context')
        generation_mode = st.session_state.get('generation_mode', 'exploratory')
        
        # NEW: Build context string for LLM
        if bio_context:
            context_str = f"""
BIOLOGICAL CONTEXT:
- Species: {bio_context['species']}
- Tissue: {bio_context['tissue']}
- Disease: {bio_context['disease']}
"""
            if bio_context.get('cell_type'):
                context_str += f"- Cell Type: {bio_context['cell_type']}\n"
            if bio_context.get('treatment'):
                context_str += f"- Treatment: {bio_context['treatment']}\n"
            
            if generation_mode == 'publication':
                context_str += "\n⚠️ PUBLICATION MODE: Apply strict biological validation standards.\n"
        else:
            context_str = "BIOLOGICAL CONTEXT: Not specified (exploratory mode)\n"
        
        # Process in batches
        num_batches = (len(signatures) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(signatures))
            batch = signatures[start_idx:end_idx]
            
            # FIX D: Use actual signature IDs in prompt
            signatures_text = "\n\n".join([
                f"{sig.signature_id}:\n"  # Use actual ID, not position
                f"Name: {sig.signature_name}\n"
                f"Mechanism: {sig.mechanism}\n"
                f"Genes ({len(sig.genes)}): {', '.join(sig.genes)}"
                for sig in batch
            ])
            
            if mode == "verify_only":
                task = "Identify genes that do NOT belong (if any)."
            else:
                task = "Identify genes to remove (if any) AND suggest genes to add (max 3 per signature)."
            
            # NEW: Include context in prompt
            prompt = f"""Review these {len(batch)} gene signatures for biological correctness.

{context_str}

Original Query: {original_query}

{signatures_text}

Validation Criteria ({generation_mode} mode):
1. Remove housekeeping genes (GAPDH, ACTB, B2M, ribosomal proteins)
2. {"Check tissue specificity for " + bio_context['tissue'] if bio_context else "Check general biological relevance"}
3. Verify pathway coherence (no contradictory mechanisms)
4. {"Require strong evidence for publication" if generation_mode == 'publication' else "Flag uncertain genes"}

Task: {task}

Output JSON ONLY using the EXACT signature IDs shown above:
{{
  "SIGNATURE_ID_1": {{
    "genes_to_remove": ["GENE1"],
    "genes_to_add": ["GENE2", "GENE3"],
    "reasoning": {{"GENE1": "why remove", "GENE2": "why add"}}
  }},
  "SIGNATURE_ID_2": {{...}},
  ...
}}

If no changes needed, return empty lists."""

            try:
                # FIX E: Add LLM logging
                st.caption(f"🔄 Batch {batch_idx+1}/{num_batches}: Verifying {len(batch)} signatures...")

                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_id,
                    max_tokens=2000,
                    temperature=0.2
                )

                raw = response.choices[0].message.content
                
                # Parse JSON
                cleaned = raw.strip()
                if '```json' in cleaned:
                    cleaned = cleaned.split('```json')[1].split('```')[0].strip()
                elif '```' in cleaned:
                    parts = cleaned.split('```')
                    if len(parts) >= 3:
                        cleaned = parts[1].strip()
                
                if not cleaned.startswith('{'):
                    start = cleaned.find('{')
                    if start != -1:
                        cleaned = cleaned[start:]
                
                batch_results = json.loads(cleaned)
                
                # FIX D: Map using actual signature IDs
                for sig in batch:
                    if sig.signature_id in batch_results:
                        all_suggestions[sig.signature_id] = batch_results[sig.signature_id]
                    else:
                        # No suggestions for this signature
                        all_suggestions[sig.signature_id] = {
                            "genes_to_remove": [],
                            "genes_to_add": [],
                            "reasoning": {}
                        }
                        
            except Exception as e:
                st.warning(f"Batch {batch_idx + 1}/{num_batches} failed: {e}")
                # Add empty suggestions for failed batch
                for sig in batch:
                    all_suggestions[sig.signature_id] = {
                        "genes_to_remove": [],
                        "genes_to_add": [],
                        "reasoning": {}
                    }
        
        return all_suggestions
        
    except Exception as e:
        st.error(f"Batch verification failed: {e}")
        return {}


# ============================================================
# NEW: DEBATE SYSTEM WRAPPERS
# ============================================================

def run_validation_debate_sync(
    genes: List[str],
    tissue_context: Optional[str] = None,
    max_rounds: int = 10
) -> Optional[DebateResult]:
    """
    Synchronous wrapper for validation debate with enhanced error handling.
    """
    if not DEBATE_SYSTEM_AVAILABLE:
        st.error("Debate system not available. Install required modules.")
        return None

    try:
        # Initialize debate engine with Groq API
        # validate_models=False to avoid blocking on init
        debate_engine = MultiRoundDebateEngine(
            api_key=st.session_state.groq_api_key,
            db_client=st.session_state.db_client_enhanced,
            base_url="https://api.groq.com/openai/v1",
            validate_models=False  # Skip validation for faster init
        )

        # Run async debate
        async def run():
            return await debate_engine.run_validation_debate(
                genes=genes,
                tissue_context=tissue_context,
                max_rounds=max_rounds,
                convergence_threshold=0.85
            )

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(run())

        return result

    except Exception as e:
        st.error(f"❌ Debate failed: {e}")

        # Provide helpful error messages
        error_str = str(e).lower()
        if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
            st.warning("""
            **Model Not Found Error**

            One or more Groq models are unavailable. This usually means:
            1. The model ID is incorrect
            2. The model was deprecated by Groq
            3. Your API key doesn't have access to that model

            **Solution**: Run the diagnostic tool to find working models:
            ```bash
            python groq_model_diagnostic.py <your_groq_api_key>
            ```
            """)
        elif "api" in error_str and "key" in error_str:
            st.warning("Check your Groq API key. It may be invalid or expired.")
        elif "rate" in error_str or "limit" in error_str:
            st.warning("Rate limit exceeded. Wait a few seconds and try again.")

        # Show full traceback in expander
        with st.expander("🔍 View Full Error Details"):
            import traceback
            st.code(traceback.format_exc())

        return None


def initialize_enhanced_db_client():
    """Initialize enhanced database client for debate system."""
    if not DEBATE_SYSTEM_AVAILABLE:
        return False
    
    if 'db_client_enhanced' not in st.session_state or st.session_state.db_client_enhanced is None:
        try:
            api_url = "https://arunviswanathan91-msigdb-api.hf.space"
            st.session_state.db_client_enhanced = DatabaseClientEnhanced(api_url)
            return True
        except Exception as e:
            st.error(f"Failed to initialize enhanced DB client: {e}")
            return False
    return True


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state():
    """Initialize session state with proper defaults"""
    defaults = {
        'groq_api_key': None,       # NEW: Groq API key
        'token_validated': False,
        'token_error': False,  # FIX B: Add explicit error flag
        'kb_loaded': False,

        # NEW: Publication-grade settings
        'generation_mode': 'exploratory',  # 'exploratory' or 'publication'
        'bio_context': None,
        
        # Layer 1
        'decomposition_result': None,
        'selected_mechanism_ids': [],
        'granularity_approved': False,
        
        # Layer 2
        'signature_ids': [],
        
        # Layer 3
        'dam_enabled': False,
        
        # Layer 4
        'llm2_suggestions': {},
        'verification_method': 'batch',  # NEW: 'batch' or 'debate'
        
        # NEW: Debate system
        'debate_enabled': False,
        'debate_num_rounds': 10,
        'current_debate_result': None,
        'debate_history': [],
        'db_client_enhanced': None,
        
        # Layer 5
        'final_approved_signature_ids': [],
        'gene_selections': {},  # For text-based UI
        
        # Timing
        'layer_timings': [],
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# KB LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_knowledge_base_cached() -> Optional[Dict[str, List[str]]]:
    """Load KB and cache"""
    possible_paths = [
        "data/knowledge_base.json.gz",
        "./data/knowledge_base.json.gz",
        "../data/knowledge_base.json.gz",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with gzip.open(path, 'rt', encoding='utf-8') as f:
                    kb_data = json.load(f)
                
                pathways = kb_data.get('pathways', {})
                return pathways
            except Exception as e:
                st.error(f"Error loading KB: {e}")
                return None
    
    return None


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    """Render sidebar with token input and timing info"""
    with st.sidebar:
        st.markdown("### 🔧 Configuration")

        with st.expander("🔑 Groq API Key", expanded=not st.session_state.token_validated):
            st.caption("Get your free API key at console.groq.com")

            key_input = st.text_input(
                "Groq API Key",
                type="password",
                value=st.session_state.groq_api_key or "",
                help="Free tier available at console.groq.com"
            )

            if st.button("Validate Key"):
                try:
                    # Test Groq connection
                    client = OpenAI(
                        api_key=key_input,
                        base_url="https://api.groq.com/openai/v1"
                    )
                    # Simple test call to verify the key works
                    client.models.list()
                    st.session_state.groq_api_key = key_input
                    st.session_state.token_validated = True
                    st.session_state.token_error = False
                    st.rerun()
                except Exception as e:
                    st.session_state.token_validated = False
                    st.session_state.token_error = True
                    st.session_state.groq_api_key = None
                    st.rerun()
        
        # FIX B: Proper state display with normalized flags
        if st.session_state.token_error and not st.session_state.token_validated:
            st.error("❌ Invalid API key")
        elif st.session_state.token_validated and not st.session_state.token_error:
            st.success("✅ Connected to Groq!")
        else:
            st.info("🔑 Enter API key and click Validate")
        
        st.markdown("---")
        
        # NEW: Mode indicator
        mode = st.session_state.get('generation_mode', 'exploratory')
        if mode == 'publication':
            st.info("📄 **Mode:** Publication")
        else:
            st.info("🔍 **Mode:** Exploratory")
        
        # NEW: Debate system status
        if DEBATE_SYSTEM_AVAILABLE:
            st.success("✅ Debate System: Available")
        else:
            st.warning("⚠️ Debate System: Not Available")
        
        st.markdown("---")
        
        # Timing diagnostics
        if st.session_state.layer_timings:
            st.markdown("### ⏱️ Performance")
            
            for timing in st.session_state.layer_timings:
                if timing.end_time:
                    st.caption(f"{timing.layer_name}: {timing.duration_str}")
            
            total = sum(t.duration for t in st.session_state.layer_timings if t.end_time)
            if total > 0:
                st.markdown(f"**Total: {total/60:.1f}m**")
        
        st.markdown("---")
        st.markdown("### ℹ️ Pipeline")
        st.caption("""
        **Layers:**
        
        1. 🧠 Granularity + Selection
        2. 🔍 Semantic (ALL queries)
        3. 🔬 DAM (Remote API)
        4. ✅ Verification (Batch OR Debate)
        5. 👤 Approval (Text-based)
        """)


# ============================================================
# KB TAB
# ============================================================

def render_kb_tab():
    st.markdown("## 📚 Knowledge Base")
    
    pathways = load_knowledge_base_cached()
    
    if pathways:
        st.session_state.kb_loaded = True
        st.success(f"✅ Loaded {len(pathways):,} pathways")
        
        sample_pid = list(pathways.keys())[0]
        sample_genes = pathways[sample_pid]
        
        with st.expander("Example signature"):
            st.caption(f"**{sample_pid}**")
            st.caption(f"Genes: {', '.join(sample_genes[:10])}... ({len(sample_genes)} total)")
    else:
        st.error("Knowledge base not found")


# ============================================================
# GENERATION TAB
# ============================================================

def render_generation_tab():
    st.markdown("## 🧬 Multi-Layer Signature Generation")
    
    if not st.session_state.kb_loaded:
        st.warning("⚠️ Please load knowledge base first")
        return
    
    if not st.session_state.token_validated:
        st.warning("⚠️ Please validate Groq API key in the sidebar")
        return
    
    # ============================================================
    # MODE SELECTION
    # ============================================================
    st.markdown("### 🎚️ Generation Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            "🔍 Exploratory Mode",
            type="primary" if st.session_state.get('generation_mode', 'exploratory') == 'exploratory' else "secondary",
            use_container_width=True,
            key="btn_exploratory"
        ):
            st.session_state['generation_mode'] = 'exploratory'
            st.rerun()
    
    with col2:
        if st.button(
            "📄 Publication Mode",
            type="primary" if st.session_state.get('generation_mode', 'exploratory') == 'publication' else "secondary",
            use_container_width=True,
            key="btn_publication"
        ):
            st.session_state['generation_mode'] = 'publication'
            st.rerun()
    
    # Show mode description
    mode = st.session_state.get('generation_mode', 'exploratory')
    
    if mode == 'publication':
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(33, 150, 243, 0.1), rgba(21, 101, 192, 0.1)); 
                    padding: 1.2rem; border-radius: 12px; margin: 1rem 0; 
                    border-left: 4px solid #2196F3;'>
        📄 <strong style='font-size: 1.1em;'>Publication Mode Active</strong><br><br>
        ✅ Requires biological context (species, tissue, disease)<br>
        ✅ Removes housekeeping genes automatically<br>
        ✅ Includes validation metadata in exports<br>
        ⚠️ Results are tagged for reproducibility
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(56, 142, 60, 0.1)); 
                    padding: 1.2rem; border-radius: 12px; margin: 1rem 0; 
                    border-left: 4px solid #4CAF50;'>
        🔍 <strong style='font-size: 1.1em;'>Exploratory Mode Active</strong><br><br>
        ⚡ Fast hypothesis generation<br>
        ⚡ Minimal validation (quick iteration)<br>
        ⚠️ <strong>Results NOT suitable for publication</strong>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================================
    # BIOLOGICAL CONTEXT
    # ============================================================
    
    if mode == 'publication':
        st.markdown("### 📋 Biological Context (Required)")
        st.caption("This information ensures reproducibility and biological validity")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            species = st.selectbox(
                "Species*",
                ["Homo sapiens", "Mus musculus"],
                index=0,
                key="species_select"
            )
        
        with col2:
            tissue_type = st.selectbox(
                "Primary Tissue*",
                [
                    "Blood/PBMC",
                    "Adipose Tissue",
                    "Liver",
                    "Brain",
                    "Lung",
                    "Intestine",
                    "Pancreas",
                    "Kidney",
                    "Heart",
                    "Muscle",
                    "Skin",
                    "Custom"
                ],
                index=0,
                key="tissue_select"
            )
            
            if tissue_type == "Custom":
                tissue_type = st.text_input(
                    "Specify tissue",
                    placeholder="e.g., Kidney cortex",
                    key="custom_tissue"
                )
        
        with col3:
            disease_context = st.text_input(
                "Disease/Condition*",
                placeholder="e.g., obesity, type 2 diabetes",
                key="disease_input"
            )
        
        # Optional fields
        with st.expander("➕ Additional Context (Optional but Recommended)"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                cell_type = st.text_input(
                    "Cell Type",
                    placeholder="e.g., CD4+ T cells, adipocytes",
                    key="cell_type_input"
                )
            
            with col_b:
                treatment = st.text_input(
                    "Treatment/Condition",
                    placeholder="e.g., LPS-stimulated, untreated",
                    key="treatment_input"
                )
        
        # Validation
        if not disease_context or not disease_context.strip():
            st.error("⚠️ Disease/Condition is required for Publication Mode")
            st.info("💡 Switch to Exploratory Mode if you want to skip this")
            return
        
        # Store context
        st.session_state['bio_context'] = {
            'species': species,
            'tissue': tissue_type,
            'disease': disease_context.strip(),
            'cell_type': cell_type.strip() if cell_type else None,
            'treatment': treatment.strip() if treatment else None,
            'timestamp': datetime.now().isoformat()
        }
        
        st.markdown("---")
    
    else:
        # Exploratory mode - context optional
        st.markdown("### 📋 Biological Context (Optional)")
        
        with st.expander("⚙️ Add Context (Recommended for Better Results)"):
            col1, col2 = st.columns(2)
            
            with col1:
                species = st.selectbox("Species", ["Homo sapiens", "Mus musculus"], key="species_exp")
                tissue_type = st.text_input("Tissue", placeholder="e.g., Blood", key="tissue_exp")
            
            with col2:
                disease_context = st.text_input("Disease", placeholder="e.g., obesity", key="disease_exp")
                cell_type = st.text_input("Cell Type", placeholder="e.g., T cells", key="cell_exp")
            
            if disease_context and disease_context.strip():
                st.session_state['bio_context'] = {
                    'species': species,
                    'tissue': tissue_type or 'unspecified',
                    'disease': disease_context.strip(),
                    'cell_type': cell_type.strip() if cell_type else None,
                    'treatment': None,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                st.session_state['bio_context'] = None
        
        st.markdown("---")
    
    # Query input
    st.markdown("### Research Question")
    query = st.text_area(
        "Biological question or mechanism",
        key="main_query",
        height=80,
        placeholder="Example: Th17 role in obesity"
    )
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_count = st.number_input("Total Signatures", 5, 100, 35, 5)
    
    with col2:
        min_genes = st.number_input("Min Genes", 5, 30, 10)
    
    with col3:
        max_genes = st.number_input("Max Genes", 10, 50, 20)
    
    if min_genes >= max_genes:
        st.error("Min genes must be < max genes")
        return
    
    st.markdown("---")
    
    # LAYERS
    render_layer1_granularity(query, target_count, min_genes, max_genes)
    
    if not st.session_state.granularity_approved:
        return
    
    render_layer2_semantic_fixed(query, target_count, min_genes, max_genes)
    
    if not st.session_state.signature_ids:
        return
    
    render_layer3_dam_remote()
    
    render_layer4_verification_with_debate(query)  # NEW: Enhanced with debate
    
    render_layer5_approval_text_based()


# ============================================================
# LAYER 1
# ============================================================

def render_layer1_granularity(query, target_count, min_genes, max_genes):
    """Layer 1: Granularity control + mechanism selection"""
    
    st.markdown("### 🧠 Layer 1: Granularity & Selection")
    
    st.markdown("""
    <div class="info-box">
    Control how many mechanisms LLM generates, then SELECT which ones to use.
    </div>
    """, unsafe_allow_html=True)
    
    granularity_level = st.slider(
        "Number of mechanisms:",
        min_value=3,
        max_value=60,
        value=10,
        step=1
    )
    
    st.caption(f"💡 Will generate {granularity_level} mechanisms")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 Generate Mechanisms", type="primary", use_container_width=True):
            if not query or not query.strip():
                st.error("Please enter a query")
                return
            
            timing = LayerTiming("Layer 1: Decomposition", time.time())
            
            with st.spinner(f"🧠 LLM generating {granularity_level} mechanisms..."):
                result = decompose_with_granularity(
                    query,
                    granularity_level,
                    st.session_state.groq_api_key
                )
                
                if result and result.facets:
                    timing.end_time = time.time()
                    st.session_state.layer_timings = [timing]
                    
                    st.session_state.decomposition_result = result
                    st.session_state.granularity_approved = False
                    st.session_state.selected_mechanism_ids = []
                    
                    st.success(f"✅ Generated {len(result.facets)} mechanisms!")
                    st.markdown(f'<span class="timing-badge">⏱️ {timing.duration_str}</span>', unsafe_allow_html=True)
                    time.sleep(0.5)
                    st.rerun()
    
    with col2:
        if st.session_state.decomposition_result:
            if st.button("🔄 Regenerate", use_container_width=True):
                st.session_state.decomposition_result = None
                st.session_state.granularity_approved = False
                st.rerun()
    
    # Show mechanisms with checkboxes
    if st.session_state.decomposition_result:
        st.markdown("---")
        st.markdown("#### 📋 Select Mechanisms to Use:")
        
        result = st.session_state.decomposition_result
        
        # Initialize selection (all selected by default)
        if not st.session_state.selected_mechanism_ids:
            st.session_state.selected_mechanism_ids = [f['facet_id'] for f in result.facets]
        
        # Render checkboxes
        for i, facet in enumerate(result.facets, 1):
            col1, col2 = st.columns([1, 20])
            
            with col1:
                is_selected = facet['facet_id'] in st.session_state.selected_mechanism_ids
                
                if st.checkbox(
                    f"select_{facet['facet_id']}",
                    key=f"checkbox_{facet['facet_id']}",
                    value=is_selected,
                    label_visibility="collapsed"
                ):
                    if facet['facet_id'] not in st.session_state.selected_mechanism_ids:
                        st.session_state.selected_mechanism_ids.append(facet['facet_id'])
                else:
                    if facet['facet_id'] in st.session_state.selected_mechanism_ids:
                        st.session_state.selected_mechanism_ids.remove(facet['facet_id'])
            
            with col2:
                with st.expander(f"{i}. {facet['facet_name']}", expanded=False):
                    st.caption(f"**Query:** {', '.join(facet.get('mechanism_queries', []))}")
                    st.caption(f"**ID:** {facet['facet_id']}")
        
        # Selection counter
        selected_count = len(st.session_state.selected_mechanism_ids)
        total_count = len(result.facets)
        
        st.markdown(f"""
        <div class="info-box">
        📊 <strong>Selected: {selected_count} of {total_count} mechanisms</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Approve button
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Approve Selected & Continue", type="primary", use_container_width=True):
                if selected_count == 0:
                    st.error("Please select at least 1 mechanism")
                else:
                    st.session_state.granularity_approved = True
                    st.session_state['target_count'] = target_count
                    st.success(f"✅ Approved {selected_count} mechanisms!")
                    time.sleep(0.5)
                    st.rerun()
        
        with col2:
            if st.button("❌ Reject All", use_container_width=True):
                st.session_state.decomposition_result = None
                st.session_state.granularity_approved = False
                st.rerun()


# ============================================================
# LAYER 2 - FIXED VERSION
# ============================================================

def render_layer2_semantic_fixed(query, target_count, min_genes, max_genes):
    """
    FIX ROOT CAUSE #1 & #2: 
    - Use ALL mechanism queries (removed [:1])
    - Dynamic variant allocation
    - Signature deduplication
    """
    
    st.markdown("---")
    st.markdown("### 🔍 Layer 2: Semantic Building (FIXED)")
    
    st.markdown(f"""
    <div class="info-box">
    ✅ <strong>FIXED:</strong> Uses ALL {len(st.session_state.selected_mechanism_ids)} mechanism queries for true diversity!
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Build Semantic Signatures", type="primary", use_container_width=True):
        
        timing = LayerTiming("Layer 2: Semantic Building", time.time())
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        try:
            # Load KB
            status.info("📚 Loading KB...")
            progress_bar.progress(5)
            
            pathways_dict = load_knowledge_base_cached()
            if not pathways_dict:
                st.error("Cannot load KB")
                return
            
            # Load model
            status.info("🔍 Loading model...")
            progress_bar.progress(10)
            
            embedding_model = load_embedding_model()
            if not embedding_model:
                st.error("Cannot load model")
                return
            
            # Load embeddings
            status.info("📂 Loading precomputed pathway embeddings...")
            progress_bar.progress(15)
            
            pathway_embeddings = load_precomputed_embeddings_or_fail(
                "data/pathway_embeddings.pkl"
            )
            
            status.success(f"✅ Loaded {len(pathway_embeddings):,} pathway embeddings")
            
            progress_bar.progress(40)
            status.info("🧬 Building signatures with TRUE diversity...")
            
            # Get selected facets
            all_facets = st.session_state.decomposition_result.facets
            selected_facets = [
                f for f in all_facets 
                if f['facet_id'] in st.session_state.selected_mechanism_ids
            ]
            
            # Calculate variants per mechanism
            num_mechanisms = len(selected_facets)
            variants_per_mechanism = max(1, target_count // num_mechanisms)
            remainder = target_count % num_mechanisms
            
            status.info(
                f"🎯 Target: {target_count} signatures from {num_mechanisms} mechanisms\n"
                f"   → {variants_per_mechanism} variants per mechanism (+{remainder} for top mechanisms)"
            )
            
            builder = SignatureBuilder(min_genes=min_genes, max_genes=max_genes)
            all_signatures = []
            
            for i, facet in enumerate(selected_facets):
                facet_name = facet['facet_name']
                mechanism_queries = facet.get('mechanism_queries', [facet_name])
                
                # FIX ROOT CAUSE #1: Use ALL queries, not just [:1]
                # Aggregate pathways from ALL queries first (Fix for Diversity Loss)
                facet_pathways_dict = {}
                facet_similarities = {}
                
                for query_idx, mech_query in enumerate(mechanism_queries):
                    status.info(f"   🔍 Searching: {facet_name} (Query {query_idx+1}/{len(mechanism_queries)})")
                    
                    rp, ps = fast_semantic_search(
                        mech_query,
                        pathway_embeddings,
                        pathways_dict,
                        embedding_model,
                        top_k=50
                    )
                    
                    if rp:
                        facet_pathways_dict.update(rp)
                        for pid, score in ps.items():
                            facet_similarities[pid] = max(facet_similarities.get(pid, 0.0), score)

                if facet_pathways_dict:
                    # Fix variants calculation
                    num_variants = variants_per_mechanism
                    if i < remainder:
                        num_variants += 1
                        
                    # Build diverse mechanisms from the aggregated pool
                    mechanism_sigs = builder.build_multiple_mechanisms_dynamic(
                        facet_name,
                        facet_pathways_dict,
                        facet_similarities,
                        num_variants=num_variants
                    )
                    
                    all_signatures.extend(mechanism_sigs)
                
                progress = 40 + int((i + 1) / len(selected_facets) * 50)
                progress_bar.progress(progress)
            
            # Trim to exact target (should be close already)
            if len(all_signatures) > target_count:
                all_signatures.sort(key=lambda s: s.confidence, reverse=True)
                all_signatures = all_signatures[:target_count]
                status.info(f"   ✂️ Trimmed to exactly {target_count} signatures")
            
            timing.end_time = time.time()
            st.session_state.layer_timings.append(timing)
            
            # Store signatures
            st.session_state.signature_ids = [sig.signature_id for sig in all_signatures]
            
            if 'signature_cache' not in st.session_state:
                st.session_state.signature_cache = {}
            
            for sig in all_signatures:
                st.session_state.signature_cache[sig.signature_id] = sig
            
            status.success(f"✅ Built {len(all_signatures)} unique signatures!")
            st.markdown(f'<span class="timing-badge">⏱️ {timing.duration_str}</span>', unsafe_allow_html=True)
            progress_bar.progress(100)
            
            # Show diversity stats
            facet_dist = Counter([sig.facet for sig in all_signatures])
            st.info(f"📊 Facet distribution: {dict(facet_dist)}")
            
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            status.error(f"❌ Failed: {e}")
            progress_bar.empty()
            import traceback
            st.code(traceback.format_exc())
    
    # Show results
    if st.session_state.signature_ids:
        signatures = [st.session_state.signature_cache[sid] for sid in st.session_state.signature_ids]
        
        st.markdown(f"""
        <div class="success-box">
        ✅ <strong>{len(signatures)} signatures built</strong>
        </div>
        """, unsafe_allow_html=True)
        
        gene_counts = [len(sig.genes) for sig in signatures]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Signatures", len(signatures))
        with col2:
            st.metric("Avg Genes", f"{np.mean(gene_counts):.1f}")
        with col3:
            st.metric("Range", f"{min(gene_counts)}-{max(gene_counts)}")


# ============================================================
# LAYER 3 - REMOTE API
# ============================================================

def render_layer3_dam_remote():
    """Layer 3: DAM expansion with REMOTE API"""
    
    st.markdown("---")
    st.markdown("### 🔬 Layer 3: DAM Expansion (Remote)")
    
    st.markdown("""
    <div class="info-box">
    ⚡ <strong>REMOTE MODE:</strong> Queries remote database - no local files needed!
    </div>
    """, unsafe_allow_html=True)
    
    enable_dam = st.checkbox("🔬 Enable DAM Expansion", value=False)
    
    if enable_dam:
        col1, col2 = st.columns(2)
        
        with col1:
            expansion_strength = st.slider("Strength", 0.1, 1.0, 0.5, 0.1)
        
        with col2:
            max_neighbors = st.slider("Max neighbors", 1, 10, 5, 1)
        
        # API URL
        api_url = "https://arunviswanathan91-msigdb-api.hf.space"
        
        if st.button("🔬 Expand with DAM", type="primary", use_container_width=True):
            
            timing = LayerTiming("Layer 3: DAM Expansion", time.time())
            
            try:
                status = st.empty()
                status.info("🌐 Connecting to remote database...")
                
                db_api = DatabaseClient(api_url=api_url)
                status.success("✅ Connected!")
                
                signatures = [
                    st.session_state.signature_cache[sid] 
                    for sid in st.session_state.signature_ids
                ]
                
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                total = len(signatures)
                expanded_count = 0
                
                for i, sig in enumerate(signatures):
                    progress_text.info(
                        f"🔍 Querying API... signature {i+1}/{total}: {sig.signature_name}"
                    )
                    
                    try:
                        expanded_genes = db_api.expand_signature_smart(
                            seed_genes=sig.genes,
                            strength=expansion_strength,
                            max_pathways_per_gene=max_neighbors
                        )
                        
                        sig.genes = list(expanded_genes)
                        sig.dam_expanded = True
                        st.session_state.signature_cache[sig.signature_id] = sig
                        expanded_count += 1
                        
                    except Exception as e:
                        st.warning(f"⚠️ Failed: {sig.signature_name}")
                    
                    progress_bar.progress((i + 1) / total)
                
                timing.end_time = time.time()
                st.session_state.layer_timings.append(timing)
                st.session_state.dam_enabled = True
                
                progress_text.success(
                    f"✅ Expanded {expanded_count}/{total} signatures!"
                )
                
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")
        
        if st.session_state.dam_enabled:
            st.success("✅ DAM expansion complete")
    else:
        st.info("DAM disabled")


# ============================================================
# LAYER 4 - NEW: VERIFICATION WITH DEBATE SYSTEM
# ============================================================

def render_layer4_verification_with_debate(query):
    """Layer 4: Enhanced verification with debate system option"""
    
    st.markdown("---")
    st.markdown("### ✅ Layer 4: Verification")
    
    if not DEBATE_SYSTEM_AVAILABLE:
        st.markdown("""
        <div class="warning-box">
        ⚠️ <strong>Debate System Not Available</strong><br>
        Using standard batch verification only. To enable debate system, install:<br>
        • db_client_enhanced.py<br>
        • debate_system_with_injector.py<br>
        • material_ui_builtin.py
        </div>
        """, unsafe_allow_html=True)
        
        # Fall back to standard verification
        render_layer4_standard_verification(query)
        return
    

    
    st.markdown("""
    <div class="info-box">
    🆕 <strong>Choose verification method:</strong> Traditional batch LLM or Multi-round AI Debate
    </div>
    """, unsafe_allow_html=True)
    
    # Method selection
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            "📋 Batch Verification",
            type="primary" if st.session_state.get('verification_method') == 'batch' else "secondary",
            use_container_width=True,
            key="btn_batch_verify"
        ):
            st.session_state.verification_method = 'batch'
            st.rerun()
    
    with col2:
        if st.button(
            "🗣️ Multi-Round Debate",
            type="primary" if st.session_state.get('verification_method') == 'debate' else "secondary",
            use_container_width=True,
            key="btn_debate_verify"
        ):
            st.session_state.verification_method = 'debate'
            st.rerun()
    
    method = st.session_state.get('verification_method', 'batch')
    
    if method == 'batch':
        render_layer4_standard_verification(query)
    else:
        render_layer4_debate_verification(query)


def render_layer4_standard_verification(query):
    """Standard batch LLM verification (original implementation)"""
    
    st.markdown("#### 📋 Batch LLM Verification")
    
    verification_mode = st.radio(
        "Mode:",
        ["None", "Verify Only", "Verify + Expand"],
        index=0,
        key="batch_mode"
    )
    
    if verification_mode != "None":

        llm2_model = st.selectbox(
            "Model:",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
            key="llm2_model",
            help="Select Groq model for verification"
        )
        
        context_options = st.multiselect(
            "Context:",
            ["Original query context", "Pathway sources used", "Mechanism description", "All of the above"],
            default=["All of the above"],
            key="context_opts"
        )
        
        batch_size = st.slider("Batch size:", 1, 10, 5, 1, help="Signatures per API call", key="batch_size_slider")
        
        if st.button("🧬 Run Batch Verification", type="primary", use_container_width=True, key="run_batch"):
            
            timing = LayerTiming("Layer 4: Batch Verification", time.time())
            
            signatures = [st.session_state.signature_cache[sid] for sid in st.session_state.signature_ids]
            
            mode = "verify_only" if "Only" in verification_mode else "verify_expand"
            
            progress_bar = st.progress(0)
            status = st.empty()
            
            status.info(f"🧬 Verifying {len(signatures)} signatures in batches of {batch_size}...")
            
            suggestions = verify_signatures_batch(
                signatures,
                query,
                mode,
                llm2_model,
                st.session_state.groq_api_key,
                context_options,
                batch_size=batch_size
            )
            
            timing.end_time = time.time()
            st.session_state.layer_timings.append(timing)
            st.session_state.llm2_suggestions = suggestions
            
            progress_bar.progress(1.0)
            status.success(f"✅ Verified {len(signatures)} signatures!")
            st.markdown(f'<span class="timing-badge">⏱️ {timing.duration_str}</span>', unsafe_allow_html=True)
            
            time.sleep(1)
            st.rerun()
    
    else:
        st.info("Batch verification disabled")


def render_layer4_debate_verification(query):
    """NEW: Multi-round debate verification"""
    
    st.markdown("#### 🗣️ Multi-Round AI Debate System")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(255, 107, 157, 0.1), rgba(255, 93, 143, 0.1)); 
                padding: 1.2rem; border-radius: 12px; margin: 1rem 0; 
                border-left: 4px solid #FF6B9D;'>
    <strong>✨ Database-Grounded Evidence + 3 Expert LLMs</strong><br><br>
    Three LLMs (Qwen, Zephyr, Phi-3) debate signature quality with evidence from:<br>
    • Gene-Gene network (2.1M probabilities)<br>
    • Gene-Pathway network (2.1M probabilities)<br>
    • GTEx expression (47K genes)<br>
    • Gene evidence database (10K genes)
    </div>
    """, unsafe_allow_html=True)
    
    # Debate configuration
    col1, col2 = st.columns(2)
    
    with col1:
        num_rounds = st.slider(
            "Number of rounds",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            help="More rounds = more thorough but slower",
            key="debate_rounds"
        )
        st.session_state.debate_num_rounds = num_rounds
    
    with col2:
        convergence_threshold = st.slider(
            "Convergence threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.85,
            step=0.05,
            help="Stop when models agree this much",
            key="debate_convergence"
        )
    
    # Initialize enhanced DB client
    if st.button("🔌 Initialize Database Connection", use_container_width=True, key="init_db"):
        with st.spinner("Connecting to database API..."):
            if initialize_enhanced_db_client():
                st.success("✅ Database client ready!")
            else:
                st.error("❌ Failed to initialize database client")
    
    # Run debate button
    if st.button("🗣️ Start Multi-Round Debate", type="primary", use_container_width=True, key="run_debate"):
        
        if not initialize_enhanced_db_client():
            st.error("Please initialize database connection first")
            return
        
        timing = LayerTiming("Layer 4: Multi-Round Debate", time.time())
        
        signatures = [st.session_state.signature_cache[sid] for sid in st.session_state.signature_ids]
        
        # Collect all genes
        all_genes = []
        for sig in signatures:
            all_genes.extend(sig.genes)
        all_genes = list(set(all_genes))
        
        st.info(f"🧬 Running debate on {len(all_genes)} unique genes from {len(signatures)} signatures...")
        
        # Get tissue context
        bio_context = st.session_state.get('bio_context')
        tissue_context = bio_context['tissue'] if bio_context else None
        
        # Run debate
        with st.spinner(f"Running {num_rounds}-round debate..."):
            result = run_validation_debate_sync(
                genes=all_genes,
                tissue_context=tissue_context,
                max_rounds=num_rounds
            )
        
        if result:
            timing.end_time = time.time()
            st.session_state.layer_timings.append(timing)
            st.session_state.current_debate_result = result
            st.session_state.debate_history.append(result)
            
            st.success(f"✅ Debate complete! ({result.total_rounds} rounds, {result.convergence_rate:.1%} convergence)")
            st.markdown(f'<span class="timing-badge">⏱️ {timing.duration_str}</span>', unsafe_allow_html=True)
            
            time.sleep(1)
            st.rerun()
    

    # Show debate results (SIMPLE UI)
    if st.session_state.current_debate_result:
        st.markdown("---")
        st.markdown("### 💬 Debate Conversation")
        
        result = st.session_state.current_debate_result
        
        # Show each round in expandable sections
        for debate_round in result.all_rounds:
            conv_pct = debate_round.convergence_rate * 100
            with st.expander(f"🔄 Round {debate_round.round_num} of {result.total_rounds} (Convergence: {conv_pct:.1f}%)", expanded=(debate_round.round_num == 1)):
                
                # Render messages with simple UI
                for msg in debate_round.messages:
                    render_debate_message_simple(
                        speaker=msg.speaker,
                        message=msg.message,
                        db_sources=msg.db_sources if msg.db_sources else None
                    )
        
        # Final consensus
        st.markdown("---")
        st.markdown("### 🎯 Final Consensus")
        
        consensus_text = f"""
**Decision:** {result.final_decision.upper()}

**Affected Genes:** {', '.join(result.affected_genes) if result.affected_genes else 'None'}

**Confidence:** {result.confidence:.2%}

**Convergence:** {result.convergence_rate:.2%}

**Total Rounds:** {result.total_rounds}
"""
        
        render_debate_message_simple(
            speaker="consensus",
            message=consensus_text,
            db_sources=[]
        )
        
        # Apply recommendations
        if result.final_decision == "remove" and result.affected_genes:
            st.markdown("---")
            
            if st.button("✂️ Apply Recommendations (Remove Flagged Genes)", type="primary", use_container_width=True, key="apply_debate"):
                genes_to_remove = set(result.affected_genes)
                
                # Update all signatures
                signatures = [st.session_state.signature_cache[sid] for sid in st.session_state.signature_ids]
                
                for sig in signatures:
                    sig.genes = [g for g in sig.genes if g not in genes_to_remove]
                    sig.debate_verified = True
                    st.session_state.signature_cache[sig.signature_id] = sig
                
                st.success(f"✅ Removed {len(genes_to_remove)} genes from all signatures")
                time.sleep(1)
                st.rerun()


# ============================================================
# LAYER 5 - TEXT-BASED GENE SELECTION
# ============================================================

def render_layer5_approval_text_based():
    """Layer 5: Text-based clickable gene selection"""
    
    st.markdown("---")
    st.markdown("### 👤 Layer 5: Review & Approve")
    
    if not st.session_state.signature_ids:
        return
    
    signatures = [st.session_state.signature_cache[sid] for sid in st.session_state.signature_ids]
    
    st.markdown(f"""
    <div class="warning-box">
    Review {len(signatures)} signatures. Click genes to toggle selection (white = selected, grey = unselected).
    </div>
    """, unsafe_allow_html=True)
    
    llm2_suggestions = st.session_state.llm2_suggestions
    
    # Initialize gene selections
    if 'gene_selections' not in st.session_state:
        st.session_state.gene_selections = {}
    
    for i, sig in enumerate(signatures):
        with st.expander(f"📋 {sig.signature_name} ({len(sig.genes)} genes)", expanded=(i == 0)):
            
            st.caption(f"**Facet:** {sig.facet}")
            st.caption(f"**Mechanism:** {sig.mechanism}")
            st.caption(f"**Confidence:** {sig.confidence:.3f}")
            
            if sig.dam_expanded:
                st.caption("🔬 DAM Expanded")
            
            if sig.debate_verified:
                st.caption("🗣️ Debate Verified")
            
            sig_suggestions = llm2_suggestions.get(sig.signature_id, {})
            
            if sig_suggestions.get('genes_to_remove'):
                st.warning(f"⚠️ LLM suggests removing: {', '.join(sig_suggestions['genes_to_remove'])}")
            
            if sig_suggestions.get('genes_to_add'):
                st.info(f"➕ LLM suggests adding: {', '.join(sig_suggestions['genes_to_add'])}")
            
            st.markdown("**Select Genes (click to toggle):**")
            
            # Initialize selection for this signature
            if sig.signature_id not in st.session_state.gene_selections:
                # Default: all genes selected except LLM-flagged removals
                flagged = set(sig_suggestions.get('genes_to_remove', []))
                st.session_state.gene_selections[sig.signature_id] = set(sig.genes) - flagged
            
            selected_genes = st.session_state.gene_selections[sig.signature_id]
            
            # Create buttons for each gene (styled to look like clickable text)
            num_cols = 5  # Genes per row
            gene_rows = [sig.genes[j:j+num_cols] for j in range(0, len(sig.genes), num_cols)]
            
            for row_idx, gene_row in enumerate(gene_rows):
                cols = st.columns(num_cols)
                
                for col_idx, gene in enumerate(gene_row):
                    is_selected = gene in selected_genes
                    is_flagged = gene in sig_suggestions.get('genes_to_remove', [])
                    
                    with cols[col_idx]:
                        # Button styled as text
                        button_key = f"gene_{sig.signature_id}_{gene}_{row_idx}_{col_idx}"
                        
                        if is_selected and not is_flagged:
                            label = f"**{gene}**"
                            button_type = "primary"
                        elif is_selected and is_flagged:
                            label = f"**{gene}** ⚠️"
                            button_type = "secondary"
                        else:
                            label = f"~~{gene}~~"
                            button_type = "secondary"
                        
                        if st.button(label, key=button_key, use_container_width=True, type=button_type):
                            # Toggle selection
                            if gene in selected_genes:
                                selected_genes.remove(gene)
                            else:
                                selected_genes.add(gene)
                            st.rerun()
            
            # Show selected genes as comma-separated text
            st.markdown("---")
            selected_list = sorted(list(selected_genes))
            st.caption(f"**Selected ({len(selected_list)} genes):** {', '.join(selected_list)}")
            
            # Add suggested genes
            if sig_suggestions.get('genes_to_add'):
                st.markdown("**Suggested Additions:**")
                
                suggested_cols = st.columns(len(sig_suggestions['genes_to_add']))
                for col_idx, gene in enumerate(sig_suggestions['genes_to_add']):
                    with suggested_cols[col_idx]:
                        add_key = f"add_{sig.signature_id}_{gene}"
                        if st.button(f"➕ {gene}", key=add_key, use_container_width=True):
                            selected_genes.add(gene)
                            st.rerun()
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Approve", key=f"approve_{sig.signature_id}", use_container_width=True):
                    
                    # Update signature with selected genes
                    sig.genes = list(selected_genes)
                    sig.llm2_verified = bool(sig_suggestions)
                    
                    # Update cache
                    st.session_state.signature_cache[sig.signature_id] = sig
                    
                    # Add to approved list
                    if sig.signature_id not in st.session_state.final_approved_signature_ids:
                        st.session_state.final_approved_signature_ids.append(sig.signature_id)
                    
                    st.success(f"✅ Approved with {len(selected_genes)} genes")
            
            with col2:
                if st.button("❌ Reject", key=f"reject_{sig.signature_id}", use_container_width=True):
                    # Remove from approved list
                    if sig.signature_id in st.session_state.final_approved_signature_ids:
                        st.session_state.final_approved_signature_ids.remove(sig.signature_id)
                    st.warning("Rejected")
    
    # Export
    if st.session_state.final_approved_signature_ids:
        st.markdown("---")
        st.markdown("### 💾 Export")
        
        approved_sigs = [
            st.session_state.signature_cache[sid] 
            for sid in st.session_state.final_approved_signature_ids
        ]
        
        st.success(f"✅ {len(approved_sigs)} signatures approved")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # GMT
            gmt_lines = []
            for sig in approved_sigs:
                sig_id = sig.signature_id
                desc = f"{sig.facet}|{sig.mechanism}|{len(sig.genes)}genes|conf:{sig.confidence:.3f}"
                if sig.dam_expanded:
                    desc += "|DAM"
                if sig.llm2_verified:
                    desc += "|LLM2"
                if sig.debate_verified:
                    desc += "|DEBATE"
                genes = '\t'.join(sig.genes)
                gmt_lines.append(f"{sig_id}\t{desc}\t{genes}")
            
            gmt_content = '\n'.join(gmt_lines)
            
            st.download_button(
                "📥 Download GMT",
                data=gmt_content,
                file_name=f"signatures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gmt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # Include mode + context + debate results in export
            results = {
                'query': st.session_state.get('main_query', ''),
                'generation_mode': st.session_state.get('generation_mode', 'exploratory'),
                'biological_context': st.session_state.get('bio_context'),
                'verification_method': st.session_state.get('verification_method', 'batch'),
                'signatures': [sig.to_dict() for sig in approved_sigs],
                'total_signatures': len(approved_sigs),
                'timestamp': datetime.now().isoformat(),
                'layer_timings': {t.layer_name: t.duration_str for t in st.session_state.layer_timings},
                'pipeline_version': '2.2-debate-system',
            }
            
            # Add debate results if available
            if st.session_state.current_debate_result:
                results['debate_summary'] = {
                    'total_rounds': st.session_state.current_debate_result.total_rounds,
                    'final_decision': st.session_state.current_debate_result.final_decision,
                    'affected_genes': st.session_state.current_debate_result.affected_genes,
                    'confidence': st.session_state.current_debate_result.confidence,
                    'convergence_rate': st.session_state.current_debate_result.convergence_rate
                }
            
            st.download_button(
                "📥 Download JSON",
                data=json.dumps(results, indent=2),
                file_name=f"signatures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )


# ============================================================
# MAIN APP
# ============================================================

def main():
    st.set_page_config(
        page_title="Signature Generator - With Debate System",
        page_icon="🧬",
        layout="wide"
    )
    
    initialize_session_state()
    inject_minimal_styles()

    st.markdown("""
    <div style='text-align: center; padding: 32px 0 16px 0;'>
        <h1>🧬 Signature Generator</h1>
        <p style='font-size: 1.1rem; color: #616161;'>
            Publication-Grade • Multi-Round Debate • Database-Grounded Evidence
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    render_sidebar()
    
    tab1, tab2 = st.tabs(["📚 Knowledge Base", "🧬 Generate"])
    
    with tab1:
        render_kb_tab()
    
    with tab2:
        render_generation_tab()


if __name__ == "__main__":
    main()

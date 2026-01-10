"""
Biological Signature Generator
===============================

Generates custom gene signatures from biological queries.

CORE PURPOSE:
- User asks for N signatures on a topic
- Decomposes topic into biological facets
- BUILDS gene signatures for each facet
- Uses pathways as source material (not end product)

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
import gzip
import os
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
import time
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, Counter

# Import pipeline modules
try:
    from complete_module_replacements import (
        PipelineConfig,
        KnowledgeBase,
    )
    from kb_builder import KBBuilder, validate_gmt_content
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    st.error(f"⚠️ Pipeline modules not found: {e}")


# ============================================================
# SIGNATURE BUILDER - THE CORE ENGINE
# ============================================================

@dataclass
class GeneSignature:
    """A custom-built gene signature"""
    signature_id: str
    signature_name: str
    facet: str
    genes: List[str]
    gene_count: int
    derivation_method: str  # 'semantic', 'core_extraction', 'neighbor_expansion'
    source_pathways: List[str]
    confidence: float
    
    def to_dict(self):
        return {
            'signature_id': self.signature_id,
            'signature_name': self.signature_name,
            'facet': self.facet,
            'genes': self.genes,
            'gene_count': self.gene_count,
            'derivation_method': self.derivation_method,
            'source_pathways': self.source_pathways,
            'confidence': self.confidence
        }


class SignatureBuilder:
    """
    Builds gene signatures from pathways and query context.
    
    This is the CORE of the application - it GENERATES signatures,
    not just selects pathways.
    """
    
    def __init__(self, mode: str = 'semantic'):
        """
        Args:
            mode: 'semantic' (pathway genes only) or 'neighbor_expansion' (use DAM)
        """
        self.mode = mode
        
    def build_core_signature(self,
                            facet_name: str,
                            pathways: List[Dict[str, Any]],
                            min_frequency: float = 0.5) -> Optional[GeneSignature]:
        """
        Build a CORE signature - genes appearing frequently across pathways.
        
        Args:
            facet_name: Name of biological facet (e.g., "Glycolysis")
            pathways: List of pathway dicts with 'genes' field
            min_frequency: Gene must appear in this % of pathways
            
        Returns:
            GeneSignature object or None
        """
        if not pathways:
            return None
        
        # Count gene frequencies
        gene_counts = Counter()
        all_pathway_ids = []
        
        for pathway in pathways:
            genes = pathway.get('genes', [])
            all_pathway_ids.append(pathway.get('pathway_id', 'Unknown'))
            gene_counts.update(genes)
        
        # Extract core genes
        n_pathways = len(pathways)
        threshold = int(n_pathways * min_frequency)
        
        core_genes = [
            gene for gene, count in gene_counts.items()
            if count >= threshold
        ]
        
        if len(core_genes) < 3:
            return None
        
        # Calculate confidence based on consistency
        avg_frequency = np.mean([gene_counts[g] / n_pathways for g in core_genes])
        
        signature_id = f"{facet_name.upper().replace(' ', '_')}_CORE"
        
        return GeneSignature(
            signature_id=signature_id,
            signature_name=f"{facet_name} (Core)",
            facet=facet_name,
            genes=sorted(core_genes),
            gene_count=len(core_genes),
            derivation_method='core_extraction',
            source_pathways=all_pathway_ids[:5],  # Top 5 for brevity
            confidence=float(avg_frequency)
        )
    
    def build_extended_signature(self,
                                 facet_name: str,
                                 pathways: List[Dict[str, Any]],
                                 min_frequency: float = 0.3) -> Optional[GeneSignature]:
        """
        Build an EXTENDED signature - includes less frequent but relevant genes.
        """
        if not pathways:
            return None
        
        gene_counts = Counter()
        all_pathway_ids = []
        
        for pathway in pathways:
            genes = pathway.get('genes', [])
            all_pathway_ids.append(pathway.get('pathway_id', 'Unknown'))
            gene_counts.update(genes)
        
        n_pathways = len(pathways)
        threshold = int(n_pathways * min_frequency)
        
        extended_genes = [
            gene for gene, count in gene_counts.items()
            if count >= threshold
        ]
        
        if len(extended_genes) < 5:
            return None
        
        avg_frequency = np.mean([gene_counts[g] / n_pathways for g in extended_genes])
        
        signature_id = f"{facet_name.upper().replace(' ', '_')}_EXTENDED"
        
        return GeneSignature(
            signature_id=signature_id,
            signature_name=f"{facet_name} (Extended)",
            facet=facet_name,
            genes=sorted(extended_genes),
            gene_count=len(extended_genes),
            derivation_method=(     
                "hybrid_expansion"     
                if self.mode in ["hybrid", "neighbor"]     
                else "semantic" 
            ),
            source_pathways=all_pathway_ids[:5],
            confidence=float(avg_frequency)
        )
    
    def build_unique_signature(self,
                              facet_name: str,
                              pathways: List[Dict[str, Any]],
                              all_genes_in_other_facets: Set[str]) -> Optional[GeneSignature]:
        """
        Build a UNIQUE signature - genes specific to this facet.
        """
        if not pathways:
            return None
        
        # Get all genes in this facet
        facet_genes = set()
        all_pathway_ids = []
        
        for pathway in pathways:
            genes = pathway.get('genes', [])
            all_pathway_ids.append(pathway.get('pathway_id', 'Unknown'))
            facet_genes.update(genes)
        
        # Find genes unique to this facet
        unique_genes = facet_genes - all_genes_in_other_facets
        
        if len(unique_genes) < 3:
            return None
        
        signature_id = f"{facet_name.upper().replace(' ', '_')}_UNIQUE"
        
        return GeneSignature(
            signature_id=signature_id,
            signature_name=f"{facet_name} (Unique)",
            facet=facet_name,
            genes=sorted(list(unique_genes)),
            gene_count=len(unique_genes),
            derivation_method='unique_extraction',
            source_pathways=all_pathway_ids[:5],
            confidence=0.75
        )
    
    def build_signatures_for_facet(self,
                                   facet_name: str,
                                   pathways: List[Dict[str, Any]],
                                   all_genes_in_other_facets: Set[str]) -> List[GeneSignature]:
        """
        Build ALL signature types for a single facet.
        
        Returns list of signatures (core, extended, unique as applicable)
        """
        signatures = []
        # Adjust thresholds based on expansion mode
        if self.mode in ["hybrid", "neighbor"]:
            core_freq = 0.5
            extended_freq = 0.25
        else:
            core_freq = 0.6
            extended_freq = 0.3


                                       
        # Core signature
        core_sig = self.build_core_signature(facet_name, pathways, min_frequency=core_freq)
        if core_sig:
            signatures.append(core_sig)
        
        # Extended signature
        extended_sig = self.build_extended_signature(     
            facet_name,     
            pathways,     
            min_frequency=extended_freq 
        )
        if extended_sig:
            signatures.append(extended_sig)
        
        # Unique signature
        unique_sig = self.build_unique_signature(facet_name, pathways, all_genes_in_other_facets)
        if unique_sig:
            signatures.append(unique_sig)
        
        return signatures


# ============================================================
# SEMANTIC RETRIEVAL (Simplified)
# ============================================================

@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except ImportError:
        st.error("⚠️ Install sentence-transformers: pip install sentence-transformers")
        return None
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None


def compute_pathway_embedding(pathway_id: str, genes: List[str], model) -> np.ndarray:
    """Create embedding for pathway"""
    pathway_name = pathway_id.replace('_', ' ').lower()
    gene_context = ' '.join(genes[:20])
    text = f"{pathway_name} {pathway_name} {gene_context}"
    return model.encode(text, convert_to_numpy=True)


@st.cache_data
def compute_pathway_embeddings_cached(_model, pathways: Dict[str, List[str]], kb_hash: str):
    """Pre-compute all pathway embeddings"""
    embeddings = {}
    for pathway_id in pathways.keys():
        genes = pathways[pathway_id]
        embeddings[pathway_id] = compute_pathway_embedding(pathway_id, genes, _model)
    return embeddings


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def semantic_retrieval(query: str,
                       pathways: Dict[str, List[str]],
                       model,
                       pathway_embeddings: Dict[str, np.ndarray],
                       top_k: int = 50) -> List[Dict[str, Any]]:
    """Find pathways semantically similar to query"""
    query_embedding = model.encode(query, convert_to_numpy=True)
    
    similarities = []
    for pathway_id, pathway_embedding in pathway_embeddings.items():
        similarity = cosine_similarity(query_embedding, pathway_embedding)
        
        similarities.append({
            'pathway_id': pathway_id,
            'genes': pathways[pathway_id],
            'similarity': similarity
        })
    
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:top_k]
                           
# ============================================================
# NEIGHBOR EXPANSION (PLACEHOLDER FOR DAM)
# ============================================================

def neighbor_expand_pathways(
    seed_pathways: List[Dict[str, Any]],
    all_pathways: Dict[str, List[str]],
    expansion_level: str = "balanced"
) -> List[Dict[str, Any]]:
    """
    Expand seed pathways into neighboring biology.
    This is a placeholder for DAM-based expansion.
    """

    level_threshold = {
        "core": 3,
        "balanced": 2,
        "broad": 1
    }.get(expansion_level, 2)

    expanded = list(seed_pathways)
    seed_ids = {p["pathway_id"] for p in seed_pathways}

    seed_genes = set()
    for p in seed_pathways:
        seed_genes.update(p["genes"])

    for pid, genes in all_pathways.items():
        if pid in seed_ids:
            continue

        overlap = len(seed_genes.intersection(genes))
        if overlap >= level_threshold:
            expanded.append({
                "pathway_id": pid,
                "genes": genes,
                "similarity": 0.0
            })

    return expanded


# ============================================================
# QUERY DECOMPOSER (LLM)
# ============================================================

def decompose_query_simple(query: str, target_count: int) -> List[Dict[str, str]]:
    """
    Simple decomposition without LLM (fallback).
    Returns generic facets.
    """
    # Generic biological categories
    facets = [
        {'facet_id': 'F1', 'facet_name': 'Core Mechanisms', 'query': query},
        {'facet_id': 'F2', 'facet_name': 'Regulatory Pathways', 'query': f"regulation {query}"},
        {'facet_id': 'F3', 'facet_name': 'Metabolic Context', 'query': f"metabolism {query}"},
        {'facet_id': 'F4', 'facet_name': 'Signaling Networks', 'query': f"signaling {query}"},
    ]
    
    return facets


def decompose_query_llm(query: str, target_count: int, hf_token: str) -> List[Dict[str, str]]:
    """
    Decompose query into biological facets using LLM.
    """
    try:
        from huggingface_hub import InferenceClient
        
        client = InferenceClient(token=hf_token)
        
        prompt = f"""Decompose this biological query into 5-8 distinct biological facets.

Query: "{query}"
Target signatures: {target_count}

For each facet, provide:
- A clear facet name
- A retrieval query for finding related pathways

Output JSON only:
{{
  "facets": [
    {{"facet_id": "F1", "facet_name": "Glycolysis", "query": "glycolysis glucose metabolism"}},
    {{"facet_id": "F2", "facet_name": "...", "query": "..."}}
  ]
}}"""

        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="Qwen/Qwen2.5-72B-Instruct",
            max_tokens=1500,
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
        return data.get('facets', [])
        
    except Exception as e:
        st.warning(f"LLM decomposition failed: {e}. Using simple decomposition.")
        return decompose_query_simple(query, target_count)


# ============================================================
# MODERN UI
# ============================================================

def inject_modern_css():
    """Inject modern styling"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
    }
    
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
        color: #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state():
    """Initialize session state"""
    defaults = {
        'hf_token': None,
        'token_validated': False,
        'kb_mode': 'builtin',
        'kb_path': None,
        'kb_uploaded': False,
        'results': None,
        'execution_complete': False,
        'generation_mode': 'semantic',  # 'semantic', 'hybrid', or 'neighbor'
        'expansion_level': 'core'  # 'core', 'balanced', 'broad'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_hf_token(token: str) -> bool:
    """Validate Hugging Face token"""
    if not token or len(token) < 20:
        return False
    
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=token)
        return True
    except:
        return False


def get_default_kb_path() -> str:
    """Get path to built-in KB"""
    possible_paths = [
        "data/knowledge_base.json.gz",
        "./data/knowledge_base.json.gz",
        "../data/knowledge_base.json.gz",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return "data/knowledge_base.json.gz"


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    """Render sidebar"""
    with st.sidebar:
        st.markdown("### 🔧 Setup")
        
        with st.expander("🔑 API Token (Optional)", expanded=False):
            st.caption("For advanced LLM-based decomposition")
            token_input = st.text_input(
                "Hugging Face Token",
                type="password",
                value=st.session_state.hf_token or "",
                help="Get token at huggingface.co/settings/tokens"
            )
            
            if st.button("Validate", use_container_width=True):
                if validate_hf_token(token_input):
                    st.session_state.hf_token = token_input
                    st.session_state.token_validated = True
                    st.success("✅ Valid!")
                else:
                    st.error("Invalid")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.caption("""
        **Biological Signature Generator**
        
        Generates custom gene signatures from biological queries.
        
        - Decomposes query into facets
        - Builds gene signatures for each
        - Uses pathway knowledge as source
        """)


# ============================================================
# KB MANAGEMENT TAB
# ============================================================

def render_kb_tab():
    """Render KB tab (simplified)"""
    st.markdown("## 📚 Pathway Knowledge Base")
    
    st.markdown("""
    <div class="info-box">
    The knowledge base contains pathway information used as SOURCE MATERIAL
    for building signatures. The app extracts and combines genes from pathways
    to create custom signatures matching your query.
    </div>
    """, unsafe_allow_html=True)
    
    kb_path = get_default_kb_path()
    
    if os.path.exists(kb_path):
        st.success(f"✅ Knowledge base ready: `{kb_path}`")
        
        try:
            with gzip.open(kb_path, 'rt', encoding='utf-8') as f:
                kb_data = json.load(f)
            
            metadata = kb_data.get('metadata', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pathways", f"{metadata.get('total_pathways', 'N/A'):,}")
            with col2:
                st.metric("Genes", f"{metadata.get('total_unique_genes', 'N/A'):,}")
            with col3:
                st.metric("Sources", len(metadata.get('sources', {})))
            
            st.session_state.kb_path = kb_path
            st.session_state.kb_uploaded = True
            
        except Exception as e:
            st.error(f"Error loading KB: {e}")
            st.session_state.kb_uploaded = False
    else:
        st.warning(f"⚠️ Knowledge base not found")
        st.session_state.kb_uploaded = False


# ============================================================
# SIGNATURE GENERATION TAB
# ============================================================

def render_generation_tab():
    """Render main signature generation tab"""
    
    if not st.session_state.kb_uploaded:
        st.warning("⚠️ Please load knowledge base first")
        return
    
    st.markdown("## 🧬 Generate Signatures")
    
    st.markdown("""
    <div class="info-box">
    <strong>How this works:</strong><br><br>
    
    1. <strong>Decomposition</strong> – Your query is broken into biological facets<br>
    2. <strong>Signature Building</strong> – For each facet, custom gene signatures are created<br>
    3. <strong>Gene Extraction</strong> – Pathways are used as source material to extract relevant genes<br><br>
    
    <em>You control how many signatures you want and how they're built.</em>
    </div>
    """, unsafe_allow_html=True)
    
    # Query input
    st.markdown("### Your Research Question")
    query = st.text_area(
        "What biological signatures do you need?",
        height=100,
        placeholder="Example: CD8+ T cell exhaustion in pancreatic cancer\nExample: Metabolic reprogramming in drug-resistant tumors\nExample: Immune checkpoint pathways in melanoma",
        help="Describe what you're researching"
    )
    
    # Target count
    target_count = st.slider(
        "How many gene signatures do you want?",
        min_value=5,
        max_value=100,
        value=25,
        step=5,
        help="Total number of custom gene signatures to generate"
    )
    
    # Generation mode
    st.markdown("### Signature Building Strategy")
    
    generation_mode = st.radio(
        "How should signatures be built?",
        options=[
            "Semantic only (precise, focused)",
            "Semantic + neighbor expansion (recommended)",
            "Neighbor expansion only (exploratory)"
        ],
        index=1,
        help=(
            "Semantic search finds biology related to your question. "
            "Neighbor expansion uses a mathematical model to include related biology. "
            "The recommended option combines both."
        )
    )
    
    # Normalize internal state
    if generation_mode.startswith("Semantic only"):
        st.session_state.generation_mode = "semantic"
    elif generation_mode.startswith("Semantic +"):
        st.session_state.generation_mode = "hybrid"
    else:
        st.session_state.generation_mode = "neighbor"
    
    st.markdown("""
    <div class="info-box">
    <strong>What this means:</strong><br><br>
    • <b>Semantic only</b>: build signatures from directly relevant pathways<br>
    • <b>Semantic + neighbor</b>: expand into related biology before building signatures<br>
    • <b>Neighbor only</b>: explore broader biological neighborhoods<br><br>
    Pathways are used as evidence — the final output is gene signatures.
    </div>
    """, unsafe_allow_html=True)
    # Advanced settings
    with st.expander("⚙️ Advanced Settings", expanded=False):
        st.warning("Default values work well for most cases")
        
        col1, col2 = st.columns(2)
        with col1:
            min_genes = st.number_input("Min genes per signature", 3, 20, 5)
        with col2:
            max_genes = st.number_input("Max genes per signature", 20, 200, 100)
        st.selectbox(
            "Biological expansion depth",
            options=["core", "balanced", "broad"],
            index=1,
            help=(
                "Controls how far signatures expand into related biology. "
                "Does not affect ranking, only breadth."
            ),
            key="expansion_level"
        )

    # Generate button
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Generate Signatures", type="primary", use_container_width=True):
        if not query:
            st.error("Please enter your research question")
            return
        
        generate_signatures(query, target_count)


def generate_signatures(query: str, target_count: int):
    """Main signature generation pipeline"""
    
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status = st.empty()
        
        try:
            # Load KB
            status.info("📚 Loading knowledge base...")
            progress_bar.progress(10)
            
            kb = KnowledgeBase(st.session_state.kb_path)
            pathways, metadata = kb.load()
            
            # Decompose query
            status.info("🧠 Decomposing query into biological facets...")
            progress_bar.progress(20)
            
            if st.session_state.token_validated and st.session_state.hf_token:
                facets = decompose_query_llm(query, target_count, st.session_state.hf_token)
            else:
                facets = decompose_query_simple(query, target_count)
            
            status.info(f"   Found {len(facets)} biological facets")
            progress_bar.progress(30)
            
            # Load embedding model
            status.info("🔍 Loading semantic search model...")
            embedding_model = load_embedding_model()
            if not embedding_model:
                st.error("Cannot proceed without embedding model")
                return
            
            # Compute pathway embeddings
            kb_hash = hash(str(sorted(pathways.keys())))
            pathway_embeddings = compute_pathway_embeddings_cached(
                embedding_model, pathways, str(kb_hash)
            )
            progress_bar.progress(40)
            
            # Build signatures for each facet
            status.info("🧬 Building gene signatures...")
            
            builder = SignatureBuilder(mode=st.session_state.generation_mode)
            all_signatures = []
            facet_genes_map = {}  # Track genes per facet
            
            for i, facet in enumerate(facets):
                facet_name = facet['facet_name']
                facet_query = facet['query']
                
                status.info(f"   Building signatures for: {facet_name}")
                
                # Retrieve relevant pathways
                # ------------------------------------------------------------
                # STEP 1: Semantic seed (if enabled)
                # ------------------------------------------------------------
                if st.session_state.generation_mode in ["semantic", "hybrid"]:
                    seed_pathways = semantic_retrieval(
                        facet_query,
                        pathways,
                        embedding_model,
                        pathway_embeddings,
                        top_k=30
                    )
                else:
                    # Neighbor-only mode starts from all pathways
                    seed_pathways = [
                        {"pathway_id": pid, "genes": genes, "similarity": 0.0}
                        for pid, genes in pathways.items()
                    ]
                
                # ------------------------------------------------------------
                # STEP 2: Neighbor expansion (if enabled)
                # ------------------------------------------------------------
                if st.session_state.generation_mode in ["hybrid", "neighbor"]:
                    relevant_pathways = neighbor_expand_pathways(
                        seed_pathways,
                        pathways,
                        expansion_level=st.session_state.expansion_level
                    )
                else:
                    relevant_pathways = seed_pathways

                
                # Get genes from other facets (for unique signatures)
                other_facet_genes = set()
                for other_facet_id, other_genes in facet_genes_map.items():
                    if other_facet_id != facet['facet_id']:
                        other_facet_genes.update(other_genes)
                
                # Build signatures
                facet_signatures = builder.build_signatures_for_facet(
                    facet_name,
                    relevant_pathways,
                    other_facet_genes
                )
                
                # Track genes in this facet
                facet_all_genes = set()
                for sig in facet_signatures:
                    facet_all_genes.update(sig.genes)
                facet_genes_map[facet['facet_id']] = facet_all_genes
                
                all_signatures.extend(facet_signatures)
                
                progress_pct = 40 + int((i + 1) / len(facets) * 50)
                progress_bar.progress(progress_pct)
            
            status.info(f"   Generated {len(all_signatures)} signatures")
            progress_bar.progress(90)
            
            # Enforce target count
            if len(all_signatures) > target_count:
                # Rank by confidence and diversity
                all_signatures.sort(key=lambda s: s.confidence, reverse=True)
                all_signatures = all_signatures[:target_count]
                status.info(f"   Trimmed to {target_count} signatures")
            
            # Store results
            st.session_state.results = {
                'query': query,
                'target_count': target_count,
                'facets': facets,
                'signatures': [sig.to_dict() for sig in all_signatures],
                'total_signatures': len(all_signatures),
                'generation_mode': (     
                    "neighbor_only_unranked"     
                    if st.session_state.generation_mode == "neighbor"     
                    else st.session_state.generation_mode 
                ),
                'timestamp': datetime.now().isoformat()
            }
            
            progress_bar.progress(100)
            status.success("✅ Signature generation complete!")
            
            st.session_state.execution_complete = True
            st.balloons()
            
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            status.error(f"❌ Generation failed: {e}")
            progress_bar.empty()
            import traceback
            st.code(traceback.format_exc())


# ============================================================
# RESULTS TAB
# ============================================================

def render_results_tab():
    """Render results tab"""
    
    if not st.session_state.execution_complete or not st.session_state.results:
        st.info("ℹ️ No results yet. Generate signatures to see results.")
        return
    
    results = st.session_state.results
    
    st.markdown("## 📊 Generated Signatures")
    
    st.markdown("""
    <div class="info-box">
    <strong>What you're seeing:</strong><br><br>
    
    • Each row is a CUSTOM GENE SIGNATURE built for your query<br>
    • Signatures are organized by biological facet<br>
    • Genes were extracted and combined from multiple pathways<br>
    • These are NEW gene sets, not existing pathways
    </div>
    """, unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Signatures Generated", results['total_signatures'])
    with col2:
        st.metric("Target Count", results['target_count'])
    with col3:
        st.metric("Biological Facets", len(results['facets']))
    with col4:
        total_unique_genes = len(set().union(*[set(sig['genes']) for sig in results['signatures']]))
        st.metric("Unique Genes", total_unique_genes)
    
    st.markdown("---")
    
    # Signatures table
    with st.expander("🧬 Signature Library", expanded=True):
        sig_data = []
        for sig in results['signatures']:
            sig_data.append({
                'Signature ID': sig['signature_id'],
                'Signature Name': sig['signature_name'],
                'Facet': sig['facet'],
                'Genes': sig['gene_count'],
                'Method': sig['derivation_method'],
                'Confidence': f"{sig['confidence']:.2f}",
                'Sample Genes': ', '.join(sig['genes'][:5]) + '...' if len(sig['genes']) > 5 else ', '.join(sig['genes'])
            })
        
        df = pd.DataFrame(sig_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        st.caption(f"Showing all {len(sig_data)} generated signatures")
    
    # Facet distribution
    with st.expander("📊 Facet Distribution"):
        facet_counts = Counter([sig['facet'] for sig in results['signatures']])
        facet_df = pd.DataFrame([
            {'Facet': k, 'Signatures': v}
            for k, v in facet_counts.items()
        ])
        st.bar_chart(facet_df.set_index('Facet'))
    
    st.markdown("---")
    
    # Downloads
    st.markdown("### 💾 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON download
        st.download_button(
            "📥 Full Results (JSON)",
            data=json.dumps(results, indent=2),
            file_name=f"signatures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # GMT download
        gmt_lines = []
        for sig in results['signatures']:
            sig_id = sig['signature_id']
            desc = f"{sig['facet']}|{sig['derivation_method']}|conf:{sig['confidence']:.2f}"
            genes = '\t'.join(sig['genes'])
            gmt_lines.append(f"{sig_id}\t{desc}\t{genes}")
        
        gmt_content = '\n'.join(gmt_lines)
        
        st.download_button(
            "📥 Signatures (GMT)",
            data=gmt_content,
            file_name=f"signatures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gmt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        # Gene list
        all_genes = set()
        for sig in results['signatures']:
            all_genes.update(sig['genes'])
        
        gene_list = '\n'.join(sorted(all_genes))
        
        st.download_button(
            "📥 All Genes (TXT)",
            data=gene_list,
            file_name=f"genes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """Main application entry point"""
    
    st.set_page_config(
        page_title="Signature Generator",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    inject_modern_css()
    
    if not PIPELINE_AVAILABLE:
        st.stop()
    
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 32px 0 16px 0;'>
        <h1>🧬 Biological Signature Generator</h1>
        <p style='font-size: 1.1rem; color: #94a3b8; margin-top: -8px;'>
            Generate custom gene signatures from biological queries
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📚 Knowledge Base", "🧬 Generate Signatures", "📊 Results"])
    
    with tab1:
        render_kb_tab()
    
    with tab2:
        render_generation_tab()
    
    with tab3:
        render_results_tab()


if __name__ == "__main__":
    main()

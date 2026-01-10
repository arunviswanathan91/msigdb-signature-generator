"""
MSigDB Pathway Discovery - Production Application
==================================================

Biological pathway discovery with mathematical importance scoring.

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
from typing import Dict, List, Any, Optional, Tuple
import time
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

# Import pipeline modules
try:
    from complete_module_replacements import (
        PipelineConfig,
        KnowledgeBase,
        CompletePipeline
    )
    from kb_builder import KBBuilder, validate_gmt_content
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    st.error(f"⚠️ Pipeline modules not found: {e}")


# ============================================================
# DECISION ANALYSIS MODEL (DAM) INTEGRATION
# ============================================================

@dataclass
class PathwayInput:
    """Input structure for Decision Analysis Model"""
    pathway_id: str
    genes: List[str]
    relevance_score: float  # Will be z-scored internally
    source: str = ""
    description: str = ""
    gene_count: int = 0
    
    def __post_init__(self):
        if self.gene_count == 0:
            self.gene_count = len(self.genes)


def build_correlation_matrix(pathways: List[PathwayInput]) -> np.ndarray:
    """Build pathway-pathway correlation matrix using gene overlap (Jaccard similarity)."""
    n = len(pathways)
    R_x = np.zeros((n, n))
    
    for i in range(n):
        set_i = set(pathways[i].genes)
        for j in range(i, n):
            set_j = set(pathways[j].genes)
            
            if i == j:
                R_x[i, j] = 1.0
            else:
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                
                if union > 0:
                    jaccard = intersection / union
                    R_x[i, j] = jaccard
                    R_x[j, i] = jaccard
    
    return R_x


def compute_path_coefficients(R_x: np.ndarray, r_y: np.ndarray) -> np.ndarray:
    """Compute path coefficients β using regression."""
    try:
        beta = np.linalg.solve(R_x, r_y)
        return beta
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(R_x) @ r_y
        return beta


def compute_decision_coefficients(R_x: np.ndarray, beta: np.ndarray, r_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute decision coefficients (direct + indirect determination)."""
    n = len(beta)
    
    direct = beta * r_y
    
    indirect = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                indirect[i] += beta[j] * R_x[i, j]
    
    decision_coef = direct + indirect
    
    return decision_coef, direct, indirect


class DAMRanker:
    """
    Decision Analysis Model-based pathway ranking.
    
    This is the mathematical engine that determines pathway importance.
    It analyzes how pathways relate to each other and ranks them accordingly.
    """
    
    def __init__(self):
        self.pathway_inputs = []
        self.R_x = None
        self.r_y = None
        self.beta = None
        self.decision_coefficients = None
        self.results = []
    
    def rank_pathways(self, pathways: List[PathwayInput], verbose: bool = True) -> List[Dict[str, Any]]:
        """Rank pathways using Decision Analysis Model."""
        if verbose:
            print(f"\n🔬 Running mathematical scoring model")
            print(f"   Analyzing {len(pathways)} pathways...")
        
        self.pathway_inputs = pathways
        
        # Standardize relevance scores
        scores = np.array([p.relevance_score for p in pathways])
        self.r_y = (scores - scores.mean()) / (scores.std() + 1e-10)
        
        # Build correlation matrix
        self.R_x = build_correlation_matrix(pathways)
        
        # Compute path coefficients
        self.beta = compute_path_coefficients(self.R_x, self.r_y)
        
        # Compute decision coefficients
        self.decision_coefficients, direct, indirect = compute_decision_coefficients(
            self.R_x, self.beta, self.r_y
        )
        
        # Create ranked results
        self.results = []
        for i, pathway in enumerate(pathways):
            self.results.append({
                'pathway_id': pathway.pathway_id,
                'genes': pathway.genes,
                'gene_count': pathway.gene_count,
                'source': pathway.source,
                'description': pathway.description,
                'decision_coefficient': float(self.decision_coefficients[i]),
                'abs_dc': float(np.abs(self.decision_coefficients[i])),
                'direct_determination': float(direct[i]),
                'indirect_determination': float(indirect[i]),
                'path_coefficient': float(self.beta[i]),
                'relevance_score_z': float(self.r_y[i]),
                'impact_direction': 'POSITIVE' if self.decision_coefficients[i] > 0 else 'NEGATIVE'
            })
        
        # Rank by importance (absolute decision coefficient)
        self.results.sort(key=lambda x: x['abs_dc'], reverse=True)
        
        for rank, result in enumerate(self.results, 1):
            result['rank'] = rank
        
        if verbose:
            print(f"   ✅ Pathways ranked by importance score")
        
        return self.results
    
    def apply_dc_expansion(self, mode: str, level: Any, verbose: bool = True) -> List[Dict[str, Any]]:
        """Apply network scope selection after ranking."""
        if not self.results:
            raise ValueError("Must run rank_pathways() first")
        
        if mode == 'top_k':
            k = int(level)
            selected = self.results[:k]
        
        elif mode == 'percentile':
            percentile = float(level)
            threshold_dc = np.percentile([r['abs_dc'] for r in self.results], 100 - percentile)
            selected = [r for r in self.results if r['abs_dc'] >= threshold_dc]
        
        elif mode == 'relative_decay':
            decay_map = {'core': 1, 'balanced': 2, 'broad': 3}
            decay_factor = decay_map.get(str(level).lower(), 2)
            
            dc_max = max(r['abs_dc'] for r in self.results)
            threshold = dc_max / decay_factor
            selected = [r for r in self.results if r['abs_dc'] >= threshold]
        
        else:
            raise ValueError(f"Unknown expansion mode: {mode}")
        
        return selected


# ============================================================
# SEMANTIC RETRIEVAL UTILITIES
# ============================================================

@st.cache_resource
def load_embedding_model():
    """Load language understanding model for semantic search."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except ImportError:
        st.error("⚠️ Semantic search requires sentence-transformers. Install with: pip install sentence-transformers")
        return None
    except Exception as e:
        st.error(f"Failed to load search model: {e}")
        return None


def compute_pathway_embedding(pathway_id: str, genes: List[str], model) -> np.ndarray:
    """Create searchable representation of a pathway."""
    pathway_name = pathway_id.replace('_', ' ').lower()
    gene_context = ' '.join(genes[:20])
    text = f"{pathway_name} {pathway_name} {gene_context}"
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding


@st.cache_data
def compute_pathway_embeddings_cached(_model, pathways: Dict[str, List[str]], kb_hash: str) -> Dict[str, np.ndarray]:
    """Pre-compute searchable representations for all pathways (cached)."""
    embeddings = {}
    for pathway_id in pathways.keys():
        genes = pathways[pathway_id]
        embeddings[pathway_id] = compute_pathway_embedding(pathway_id, genes, _model)
    return embeddings


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute similarity between two items."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def semantic_retrieval(
    query: str,
    pathways: Dict[str, List[str]],
    metadata: Dict[str, Any],
    pool_size: int,
    model,
    pathway_embeddings: Dict[str, np.ndarray]
) -> List[Dict[str, Any]]:
    """Find pathways using language understanding."""
    query_embedding = model.encode(query, convert_to_numpy=True)
    
    similarities = []
    for pathway_id, pathway_embedding in pathway_embeddings.items():
        similarity = cosine_similarity(query_embedding, pathway_embedding)
        relevance = similarity + 1.0
        
        similarities.append({
            'pathway_id': pathway_id,
            'genes': pathways[pathway_id],
            'relevance_score': relevance,
            'cosine_similarity': similarity,
            'source': metadata.get('sources', {}).get(pathway_id.split('_')[0], 'Unknown')
        })
    
    similarities.sort(key=lambda x: x['relevance_score'], reverse=True)
    return similarities[:pool_size]


def keyword_retrieval(
    query: str,
    pathways: Dict[str, List[str]],
    metadata: Dict[str, Any],
    pool_size: int
) -> List[Dict[str, Any]]:
    """Find pathways using exact word matching."""
    query_terms = set(query.lower().split())
    candidates = []
    
    for pathway_id, genes in pathways.items():
        pathway_terms = set(pathway_id.lower().split('_'))
        overlap = len(query_terms & pathway_terms)
        
        if overlap > 0 or len(candidates) < pool_size:
            relevance = overlap + 0.1
            candidates.append({
                'pathway_id': pathway_id,
                'genes': genes,
                'relevance_score': relevance,
                'source': metadata.get('sources', {}).get(pathway_id.split('_')[0], 'Unknown')
            })
    
    candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
    return candidates[:pool_size]


# ============================================================
# MODERN SURFACE UI
# ============================================================

def inject_modern_css():
    """Inject modern, clean UI styling"""
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
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        color: #e2e8f0;
        letter-spacing: -0.01em;
        margin-top: 2rem !important;
    }
    
    h3 {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        color: #cbd5e1;
        letter-spacing: -0.01em;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .stButton button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        transition: all 0.3s !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
    }
    
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
        color: #e2e8f0;
    }
    
    .help-text {
        color: #94a3b8;
        font-size: 0.9rem;
        font-style: italic;
        margin-top: 4px;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'hf_token': None,
        'token_validated': False,
        'kb_mode': 'builtin',
        'kb_path': None,
        'kb_uploaded': False,
        'custom_kb_data': None,
        'results': None,
        'dam_results': None,
        'execution_complete': False,
        'retrieval_mode': 'semantic',  # 'semantic' or 'keyword'
        'network_scope': 'balanced',   # 'core', 'balanced', 'broad'
        'pool_size': 100
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
    except Exception as e:
        st.error(f"Token validation failed: {str(e)}")
        return False


def get_default_kb_path() -> str:
    """Get path to built-in KB"""
    possible_paths = [
        "data/knowledge_base.json.gz",
        "./data/knowledge_base.json.gz",
        "../data/knowledge_base.json.gz",
        "/content/drive/MyDrive/siggen/knowledge_base.json.gz"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return "data/knowledge_base.json.gz"


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    """Render sidebar with configuration"""
    
    with st.sidebar:
        st.markdown("### 🔧 Setup")
        
        # HF Token
        with st.expander("🔑 API Token (Optional)", expanded=False):
            st.caption("Only needed for advanced LLM features")
            token_input = st.text_input(
                "Hugging Face Token",
                type="password",
                value=st.session_state.hf_token or "",
                help="Get a free token at huggingface.co/settings/tokens"
            )
            
            if st.button("Validate Token", use_container_width=True):
                if validate_hf_token(token_input):
                    st.session_state.hf_token = token_input
                    st.session_state.token_validated = True
                    st.success("✅ Valid!")
                else:
                    st.error("Invalid token")
        
        st.markdown("---")
        
        # About
        st.markdown("### ℹ️ About")
        st.caption("""
        **Pathway Discovery Tool**
        
        Finds biological pathways related to your research question using:
        - Language understanding for discovery
        - Mathematical scoring for importance
        - Network analysis for context
        """)
        
        st.caption("Built for biological research")


# ============================================================
# KB MANAGEMENT TAB
# ============================================================

def render_kb_management_tab():
    """Render Knowledge Base management tab"""
    
    st.markdown("## 📚 Pathway Database")
    
    st.markdown("""
    <div class="info-box">
    <strong>What is this?</strong><br>
    A pathway database contains information about biological pathways and the genes involved in them.
    You can use our curated database or build your own from GMT files.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        builtin_selected = st.button(
            "📦 Use Curated Database",
            use_container_width=True,
            type="primary" if st.session_state.kb_mode == 'builtin' else "secondary"
        )
        st.caption("Recommended - includes 19,000+ pathways from major databases")
        if builtin_selected:
            st.session_state.kb_mode = 'builtin'
            st.session_state.kb_uploaded = False
    
    with col2:
        custom_selected = st.button(
            "🔧 Build Custom Database", 
            use_container_width=True,
            type="primary" if st.session_state.kb_mode == 'custom' else "secondary"
        )
        st.caption("Upload your own GMT files for specialized research")
        if custom_selected:
            st.session_state.kb_mode = 'custom'
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.session_state.kb_mode == 'builtin':
        render_builtin_kb_interface()
    else:
        render_custom_kb_interface()


def render_builtin_kb_interface():
    """Interface for using built-in KB"""
    
    st.markdown("### 📦 Curated Pathway Database")
    
    kb_path = get_default_kb_path()
    
    if os.path.exists(kb_path):
        st.success(f"✅ Database ready: `{kb_path}`")
        
        try:
            with gzip.open(kb_path, 'rt', encoding='utf-8') as f:
                kb_data = json.load(f)
            
            metadata = kb_data.get('metadata', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pathways", f"{metadata.get('total_pathways', 'N/A'):,}")
            with col2:
                st.metric("Genes", f"{metadata.get('total_unique_genes', 'N/A'):,}")
            with col3:
                st.metric("Sources", len(metadata.get('sources', {})))
            with col4:
                st.metric("Version", metadata.get('version', 'N/A'))
            
            with st.expander("📊 Database Sources"):
                sources = metadata.get('sources', {})
                if sources:
                    source_df = pd.DataFrame([
                        {'Source': k, 'Pathways': v} 
                        for k, v in sorted(sources.items(), key=lambda x: x[1], reverse=True)
                    ])
                    st.dataframe(source_df, use_container_width=True)
            
            st.session_state.kb_path = kb_path
            st.session_state.kb_uploaded = True
            
        except Exception as e:
            st.error(f"Error loading database: {e}")
            st.session_state.kb_uploaded = False
    else:
        st.warning(f"⚠️ Database file not found at `{kb_path}`")
        st.info("Please ensure knowledge_base.json.gz is in the data/ directory, or use Custom Database mode.")
        st.session_state.kb_uploaded = False


def render_custom_kb_interface():
    """Interface for building custom KB from GMT files"""
    
    st.markdown("### 🔧 Build Custom Database")
    st.caption("Upload GMT format files to create your own pathway database")
    
    with st.expander("ℹ️ What are GMT files?", expanded=False):
        st.markdown("""
        GMT (Gene Matrix Transposed) files are a standard format for storing pathway data.
        Each line contains:
        - Pathway name
        - Description
        - List of genes (tab-separated)
        
        Download GMT files from:
        - MSigDB (https://www.gsea-msigdb.org)
        - KEGG (https://www.genome.jp/kegg)
        - Reactome (https://reactome.org)
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        min_genes = st.number_input(
            "Minimum genes per pathway",
            min_value=1,
            max_value=100,
            value=5,
            help="Pathways with fewer genes will be filtered out"
        )
    with col2:
        max_genes = st.number_input(
            "Maximum genes per pathway",
            min_value=10,
            max_value=1000,
            value=500,
            help="Pathways with more genes will be filtered out"
        )
    
    uploaded_files = st.file_uploader(
        "Upload GMT Files",
        type=['gmt', 'txt'],
        accept_multiple_files=True,
        help="You can upload multiple GMT files at once"
    )
    
    if uploaded_files:
        st.info(f"📋 {len(uploaded_files)} file(s) ready to process")
        
        with st.expander("📋 Files to Process"):
            for file in uploaded_files:
                st.text(f"• {file.name} ({file.size / 1024:.1f} KB)")
        
        if st.button("🔨 Build Database", type="primary", use_container_width=True):
            build_custom_kb(uploaded_files, min_genes, max_genes)


def build_custom_kb(uploaded_files, min_genes: int, max_genes: int):
    """Build custom KB from uploaded GMT files"""
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    try:
        builder = KBBuilder(min_genes=min_genes, max_genes=max_genes)
        
        status.info("🔍 Validating files...")
        progress_bar.progress(10)
        
        file_data = []
        for i, file in enumerate(uploaded_files):
            content = file.read().decode('utf-8', errors='ignore')
            is_valid, error_msg = validate_gmt_content(content)
            
            if not is_valid:
                st.error(f"❌ {file.name}: {error_msg}")
                return
            
            file_data.append((file.name, content))
            progress_bar.progress(10 + (i + 1) * 20 // len(uploaded_files))
        
        status.info("🔨 Building database...")
        progress_bar.progress(40)
        
        kb_data = builder.build_kb(file_data)
        progress_bar.progress(70)
        
        status.info("💾 Saving database...")
        temp_dir = tempfile.gettempdir()
        kb_path = os.path.join(temp_dir, 'custom_knowledge_base.json.gz')
        
        file_size = builder.save_kb(kb_data, kb_path)
        progress_bar.progress(90)
        
        status.success("✅ Database built successfully!")
        progress_bar.progress(100)
        
        summary = builder.get_summary()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Pathways", f"{summary['total_pathways_kept']:,}")
        with col2:
            st.metric("Sources", summary['total_sources'])
        with col3:
            st.metric("File Size", f"{file_size / (1024*1024):.2f} MB")
        
        with st.expander("📊 Build Statistics"):
            for source, stats in summary['sources'].items():
                st.write(f"**{source}**")
                st.write(f"- Kept: {stats['kept']} pathways")
                st.write(f"- Filtered: {stats['too_small']} too small, {stats['too_large']} too large")
                st.write(f"- Duplicates: {stats['duplicates']}")
                st.write("---")
        
        st.session_state.kb_path = kb_path
        st.session_state.kb_uploaded = True
        st.session_state.custom_kb_data = kb_data
        
        st.success("✅ Ready to use your custom database!")
        
    except Exception as e:
        status.error(f"❌ Error building database: {e}")
        progress_bar.empty()


# ============================================================
# PIPELINE TAB
# ============================================================

def render_pipeline_tab():
    """Render main pipeline execution tab"""
    
    if not st.session_state.kb_uploaded:
        st.warning("⚠️ Please select a pathway database first (see Pathway Database tab)")
        return
    
    st.markdown("## 🔬 Pathway Discovery")
    
    # HOW IT WORKS EXPLANATION
    st.markdown("""
    <div class="info-box">
    <strong>How this works (in simple terms):</strong><br><br>
    
    <strong>1. Pathway Discovery</strong> – The tool finds pathways related to your question<br>
    <strong>2. Importance Scoring</strong> – A mathematical model ranks pathways by importance<br>
    <strong>3. Network Scope</strong> – You choose how broad the results should be<br><br>
    
    <em>Note: Ranking is done mathematically, not by AI opinion. Language understanding is only used for discovery.</em>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Your Research Question")
    
    query = st.text_area(
        "What biological pathways are you interested in?",
        height=100,
        placeholder="Example: pathways involved in cancer metastasis\nExample: immune response to viral infection\nExample: metabolic changes in diabetes",
        help="Describe your research question in plain language"
    )
    
    # RETRIEVAL MODE SELECTION
    st.markdown("### How should pathways be found?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        semantic_selected = st.button(
            "🧠 Semantic Search (Recommended)",
            use_container_width=True,
            type="primary" if st.session_state.retrieval_mode == 'semantic' else "secondary"
        )
        st.markdown('<p class="help-text">Uses language understanding to find related pathways, even if wording differs. Finds more relevant results.</p>', unsafe_allow_html=True)
        if semantic_selected:
            st.session_state.retrieval_mode = 'semantic'
    
    with col2:
        keyword_selected = st.button(
            "🔤 Keyword Search",
            use_container_width=True,
            type="primary" if st.session_state.retrieval_mode == 'keyword' else "secondary"
        )
        st.markdown('<p class="help-text">Uses exact word matching. Faster, but may miss related biology. Good for specific pathway names.</p>', unsafe_allow_html=True)
        if keyword_selected:
            st.session_state.retrieval_mode = 'keyword'
    
    st.markdown(f"**Selected:** {st.session_state.retrieval_mode.title()} search")
    st.caption("This controls WHICH pathways are considered. It does NOT affect how they are ranked.")
    
    # NETWORK SCOPE CONTROLS
    st.markdown("### Importance & Network Scope")
    
    st.caption("How many pathways should be shown after importance is calculated?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        core_selected = st.button(
            "🎯 Core Pathways",
            use_container_width=True,
            type="primary" if st.session_state.network_scope == 'core' else "secondary"
        )
        st.caption("Most important only")
        if core_selected:
            st.session_state.network_scope = 'core'
    
    with col2:
        balanced_selected = st.button(
            "⚖️ Balanced Network",
            use_container_width=True,
            type="primary" if st.session_state.network_scope == 'balanced' else "secondary"
        )
        st.caption("Good balance (default)")
        if balanced_selected:
            st.session_state.network_scope = 'balanced'
    
    with col3:
        broad_selected = st.button(
            "🌐 Broad Context",
            use_container_width=True,
            type="primary" if st.session_state.network_scope == 'broad' else "secondary"
        )
        st.caption("Comprehensive view")
        if broad_selected:
            st.session_state.network_scope = 'broad'
    
    st.info(f"**Selected:** {st.session_state.network_scope.title()} scope - This filters results AFTER ranking. It does not change the order.")
    
    # ADVANCED SETTINGS (HIDDEN)
    with st.expander("⚙️ Advanced Settings (Optional)", expanded=False):
        st.warning("**For advanced users only.** Default values are suitable for most analyses.")
        
        st.markdown("#### Search Parameters")
        col1, col2 = st.columns(2)
        with col1:
            pool_size = st.slider(
                "Candidate pool size",
                min_value=50,
                max_value=200,
                value=100,
                step=25,
                help="How many pathways to analyze before ranking"
            )
            st.session_state.pool_size = pool_size
        
        with col2:
            st.caption("Number of pathways to retrieve and analyze")
        
        st.markdown("#### Gene Constraints")
        col1, col2 = st.columns(2)
        with col1:
            min_genes = st.number_input("Min genes per pathway", 1, 50, 5)
        with col2:
            max_genes = st.number_input("Max genes per pathway", 10, 500, 300)
        
        st.markdown("#### Overlap Thresholds")
        st.caption("Controls how similar pathways can be")
        col1, col2 = st.columns(2)
        with col1:
            within_facet_overlap = st.slider("Within-group similarity", 0.0, 1.0, 0.50, 0.05)
        with col2:
            cross_facet_overlap = st.slider("Between-group similarity", 0.0, 1.0, 0.25, 0.05)
    
    # EXECUTE BUTTON
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Discover Pathways", type="primary", use_container_width=True):
        if not query:
            st.error("Please enter your research question")
            return
        
        execute_discovery_pipeline(query)


def execute_discovery_pipeline(query: str):
    """Execute the pathway discovery pipeline"""
    
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status = st.empty()
        
        try:
            # Map network scope to technical parameters
            scope_map = {
                'core': ('relative_decay', 'core'),
                'balanced': ('relative_decay', 'balanced'),
                'broad': ('relative_decay', 'broad')
            }
            expansion_mode, expansion_level = scope_map[st.session_state.network_scope]
            
            status.info("⚙️ Initializing discovery pipeline...")
            progress_bar.progress(5)
            
            # Load knowledge base
            status.info("📚 Loading pathway database...")
            kb = KnowledgeBase(st.session_state.kb_path)
            pathways, metadata = kb.load()
            progress_bar.progress(20)
            
            # RETRIEVAL STEP
            pool_size = st.session_state.pool_size
            
            if st.session_state.retrieval_mode == 'semantic':
                status.info(f"🧠 Finding pathways using semantic search...")
                
                embedding_model = load_embedding_model()
                if embedding_model is None:
                    st.error("Semantic search is not available. Falling back to keyword search.")
                    st.session_state.retrieval_mode = 'keyword'
                    candidate_pathways = keyword_retrieval(query, pathways, metadata, pool_size)
                else:
                    kb_hash = hash(str(sorted(pathways.keys())))
                    pathway_embeddings = compute_pathway_embeddings_cached(
                        embedding_model, pathways, str(kb_hash)
                    )
                    candidate_pathways = semantic_retrieval(
                        query, pathways, metadata, pool_size,
                        embedding_model, pathway_embeddings
                    )
            else:
                status.info(f"🔤 Finding pathways using keyword search...")
                candidate_pathways = keyword_retrieval(query, pathways, metadata, pool_size)
            
            status.info(f"   Found {len(candidate_pathways)} candidate pathways")
            progress_bar.progress(40)
            
            # Convert to PathwayInput objects
            status.info("🔄 Preparing pathways for analysis...")
            pathway_inputs = []
            for cp in candidate_pathways:
                pathway_inputs.append(PathwayInput(
                    pathway_id=cp['pathway_id'],
                    genes=cp['genes'],
                    relevance_score=cp['relevance_score'],
                    source=cp['source'],
                    description=cp['pathway_id'],
                    gene_count=len(cp['genes'])
                ))
            
            progress_bar.progress(50)
            
            # Run importance scoring (DAM)
            status.info("📊 Computing pathway importance scores...")
            status.info("   This uses a mathematical model, not AI opinion...")
            
            ranker = DAMRanker()
            ranked_results = ranker.rank_pathways(pathway_inputs, verbose=True)
            
            status.info(f"   ✅ Ranked {len(ranked_results)} pathways by importance")
            progress_bar.progress(70)
            
            # Apply network scope
            status.info("🎯 Applying network scope filter...")
            
            expanded_results = ranker.apply_dc_expansion(
                mode=expansion_mode,
                level=expansion_level,
                verbose=True
            )
            
            status.info(f"   Selected {len(expanded_results)} pathways for final results")
            progress_bar.progress(90)
            
            # Store results
            st.session_state.dam_results = {
                'query': query,
                'retrieval_mode': st.session_state.retrieval_mode,
                'network_scope': st.session_state.network_scope,
                'pool_size': pool_size,
                'total_retrieved': len(ranked_results),
                'total_selected': len(expanded_results),
                'all_ranked': ranked_results,
                'selected': expanded_results,
                'dc_max': max(r['abs_dc'] for r in ranked_results) if ranked_results else 0
            }
            
            # Backward compatibility
            st.session_state.results = {
                'total_signatures': len(expanded_results),
                'total_pathways_selected': len(expanded_results),
                'derived_signatures': [
                    {
                        'signature_id': r['pathway_id'],
                        'facet_id': 'Discovery',
                        'genes': r['genes'],
                        'gene_count': r['gene_count'],
                        'source': r['source'],
                        'derivation_method': 'Mathematical Scoring',
                        'confidence': r['abs_dc'],
                        'decision_coefficient': r['decision_coefficient']
                    }
                    for r in expanded_results
                ]
            }
            
            progress_bar.progress(100)
            status.success("✅ Pathway discovery completed!")
            
            st.session_state.execution_complete = True
            st.balloons()
            
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            status.error(f"❌ Discovery failed: {e}")
            st.error(f"Error details: {str(e)}")
            progress_bar.empty()
            import traceback
            st.code(traceback.format_exc())


# ============================================================
# RESULTS TAB
# ============================================================

def render_results_tab():
    """Render results and download tab"""
    
    if not st.session_state.execution_complete:
        st.info("ℹ️ No results yet. Run the pathway discovery to see results.")
        return
    
    # Check which results are available
    is_dam_mode = st.session_state.dam_results is not None
    
    if is_dam_mode:
        render_dam_results()
    else:
        st.info("Results from standard pipeline mode")


def render_dam_results():
    """Render DAM-specific results"""
    
    dam_results = st.session_state.dam_results
    
    st.markdown("## 📊 Discovery Results")
    
    # EXPLANATION
    st.markdown("""
    <div class="info-box">
    <strong>What you are seeing:</strong><br><br>
    
    • Pathways are <strong>ordered by importance</strong> (calculated mathematically)<br>
    • Importance reflects both <strong>direct relevance</strong> to your question and <strong>network interactions</strong><br>
    • Filters and scope settings do <strong>not change the ranking</strong> – they only control what's shown
    </div>
    """, unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pathways Analyzed", dam_results['total_retrieved'])
    with col2:
        st.metric("Pathways Selected", dam_results['total_selected'])
    with col3:
        st.metric("Max Importance", f"{dam_results['dc_max']:.3f}")
    with col4:
        selection_rate = (dam_results['total_selected'] / dam_results['total_retrieved'] * 100) if dam_results['total_retrieved'] > 0 else 0
        st.metric("Selection Rate", f"{selection_rate:.1f}%")
    
    st.markdown("---")
    
    # Query summary
    with st.expander("📋 Discovery Parameters", expanded=False):
        st.write(f"**Research Question:** {dam_results['query']}")
        st.write(f"**Search Method:** {dam_results['retrieval_mode'].title()}")
        st.write(f"**Network Scope:** {dam_results['network_scope'].title()}")
        st.write(f"**Candidate Pool:** {dam_results['pool_size']} pathways")
    
    # Selected pathways table
    with st.expander("✅ Selected Pathways", expanded=True):
        if dam_results['selected']:
            selected_data = []
            for i, result in enumerate(dam_results['selected'], 1):
                selected_data.append({
                    'Rank': i,
                    'Pathway Name': result['pathway_id'][:60] + '...' if len(result['pathway_id']) > 60 else result['pathway_id'],
                    'Genes': result['gene_count'],
                    'Importance': f"{result['abs_dc']:.3f}",
                    'Direction': result['impact_direction'],
                    'Source': result['source']
                })
            
            df_selected = pd.DataFrame(selected_data)
            st.dataframe(df_selected, use_container_width=True, height=400)
            
            st.caption(f"Showing {len(selected_data)} pathways selected by importance and network scope")
    
    # Full ranking table
    with st.expander("📊 All Analyzed Pathways (Full Ranking)", expanded=False):
        if dam_results['all_ranked']:
            all_data = []
            for result in dam_results['all_ranked'][:100]:
                all_data.append({
                    'Rank': result['rank'],
                    'Pathway Name': result['pathway_id'][:60] + '...' if len(result['pathway_id']) > 60 else result['pathway_id'],
                    'Genes': result['gene_count'],
                    'Importance Score': f"{result['abs_dc']:.4f}",
                    'Direction': result['impact_direction'],
                    'Source': result['source']
                })
            
            df_all = pd.DataFrame(all_data)
            st.dataframe(df_all, use_container_width=True, height=400)
            
            st.caption(f"Showing top 100 of {len(dam_results['all_ranked'])} analyzed pathways")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📈 Importance Distribution"):
            importance_values = [r['abs_dc'] for r in dam_results['all_ranked']]
            hist_data = pd.DataFrame({'Importance Score': importance_values})
            st.bar_chart(hist_data['Importance Score'].value_counts().sort_index())
            st.caption(f"Distribution of {len(importance_values)} importance scores")
    
    with col2:
        with st.expander("🎯 Impact Direction"):
            impact_counts = pd.DataFrame(
                dam_results['all_ranked']
            )['impact_direction'].value_counts()
            st.bar_chart(impact_counts)
            st.caption("Positive vs. negative impact pathways")
    
    st.markdown("---")
    
    # Download section
    st.markdown("### 💾 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Full results as JSON
        full_export = {
            'metadata': {
                'query': dam_results['query'],
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'retrieval_mode': dam_results['retrieval_mode'],
                    'network_scope': dam_results['network_scope'],
                    'pool_size': dam_results['pool_size']
                }
            },
            'statistics': {
                'total_analyzed': dam_results['total_retrieved'],
                'total_selected': dam_results['total_selected'],
                'max_importance': dam_results['dc_max']
            },
            'selected_pathways': dam_results['selected'],
            'all_ranked_pathways': dam_results['all_ranked']
        }
        
        st.download_button(
            "📥 Full Results (JSON)",
            data=json.dumps(full_export, indent=2),
            file_name=f"pathway_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Selected pathways as CSV
        if dam_results['selected']:
            csv_data = pd.DataFrame([
                {
                    'rank': i,
                    'pathway_id': r['pathway_id'],
                    'gene_count': r['gene_count'],
                    'importance_score': r['abs_dc'],
                    'decision_coefficient': r['decision_coefficient'],
                    'impact_direction': r['impact_direction'],
                    'source': r['source']
                }
                for i, r in enumerate(dam_results['selected'], 1)
            ])
            
            st.download_button(
                "📥 Selected Pathways (CSV)",
                data=csv_data.to_csv(index=False),
                file_name=f"selected_pathways_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col3:
        # Gene list
        all_genes = set()
        for r in dam_results['selected']:
            all_genes.update(r['genes'])
        
        gene_list = '\n'.join(sorted(all_genes))
        
        st.download_button(
            "📥 Gene List (TXT)",
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
        page_title="Pathway Discovery Tool",
        page_icon="🔬",
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
        <h1>🔬 Pathway Discovery Tool</h1>
        <p style='font-size: 1.1rem; color: #94a3b8; margin-top: -8px;'>
            Find biological pathways using language understanding and mathematical scoring
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📚 Pathway Database", "🔬 Discovery", "📊 Results"])
    
    with tab1:
        render_kb_management_tab()
    
    with tab2:
        render_pipeline_tab()
    
    with tab3:
        render_results_tab()


if __name__ == "__main__":
    main()

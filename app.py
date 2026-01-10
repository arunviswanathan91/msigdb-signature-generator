"""
Biological Signature Generator - COMPLETE MULTI-LAYER PIPELINE
===============================================================

LAYERS IMPLEMENTED:
1. LLM-based Granularity Control (Qwen/Qwen2.5-72B-Instruct)
2. Semantic Pathway Selection
3. Mathematical Neighbor Expansion (DAM) - Optional
4. LLM Verification & Expansion (BioMedLM/BioGPT) - Optional
5. Interactive Approval Interface

User controls EVERY decision point.
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
            'llm2_verified': self.llm2_verified
        }


@dataclass
class DecompositionResult:
    """Result from LLM decomposition"""
    facets: List[Dict[str, str]]
    granularity_level: int
    timestamp: str
    llm_model: str


# ============================================================
# SIGNATURE BUILDER
# ============================================================

class SignatureBuilder:
    """Builds gene signatures with relevance scoring and DAM expansion"""
    
    def __init__(self, min_genes: int = 10, max_genes: int = 20):
        self.min_genes = min_genes
        self.max_genes = max_genes
    
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
        scores_dict = dict(selected_pairs)
        
        avg_score = np.mean([score for gene, score in selected_pairs])
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
    
    def build_multiple_mechanisms(
        self,
        facet_name: str,
        pathways_dict: Dict[str, List[str]],
        pathway_similarities: Dict[str, float],
        num_variants: int = 3
    ) -> List[GeneSignature]:
        """Build multiple mechanism variants per facet"""
        
        signatures = []
        
        # Core
        core_sig = self.build_signature_from_pathways(
            facet_name, "Core Markers", pathways_dict, pathway_similarities
        )
        if core_sig:
            signatures.append(core_sig)
        
        # Extended (top pathways)
        if len(pathways_dict) >= 4:
            sorted_pathways = sorted(pathway_similarities.items(), key=lambda x: x[1], reverse=True)
            top_half_ids = [pid for pid, _ in sorted_pathways[:len(sorted_pathways)//2]]
            top_pathways = {pid: pathways_dict[pid] for pid in top_half_ids if pid in pathways_dict}
            top_sims = {pid: pathway_similarities[pid] for pid in top_half_ids if pid in pathway_similarities}
            
            extended_sig = self.build_signature_from_pathways(
                facet_name, "Extended Network", top_pathways, top_sims
            )
            if extended_sig and extended_sig.genes != core_sig.genes:
                signatures.append(extended_sig)
        
        # Regulatory (bottom pathways)
        if len(pathways_dict) >= 4 and len(signatures) < num_variants:
            sorted_pathways = sorted(pathway_similarities.items(), key=lambda x: x[1], reverse=True)
            bottom_half_ids = [pid for pid, _ in sorted_pathways[len(sorted_pathways)//2:]]
            bottom_pathways = {pid: pathways_dict[pid] for pid in bottom_half_ids if pid in pathways_dict}
            bottom_sims = {pid: pathway_similarities[pid] for pid in bottom_half_ids if pid in pathway_similarities}
            
            regulatory_sig = self.build_signature_from_pathways(
                facet_name, "Regulatory Context", bottom_pathways, bottom_sims
            )
            if regulatory_sig:
                signatures.append(regulatory_sig)
        
        return signatures
    
    def expand_with_dam(
        self,
        signature: GeneSignature,
        pathways_kb: Dict[str, List[str]],
        expansion_strength: float = 0.5,
        max_additional: int = 5
    ) -> GeneSignature:
        """Expand signature using DAM"""
        
        gene_cooccurrence = self._build_cooccurrence_matrix(pathways_kb, signature.genes)
        
        neighbor_scores = {}
        
        for seed_gene in signature.genes:
            neighbors = gene_cooccurrence.get(seed_gene, {})
            
            for neighbor, score in neighbors.items():
                if neighbor not in signature.genes:
                    neighbor_scores[neighbor] = max(
                        neighbor_scores.get(neighbor, 0),
                        score * expansion_strength
                    )
        
        top_neighbors = sorted(neighbor_scores.items(), key=lambda x: x[1], reverse=True)
        top_neighbors = top_neighbors[:max_additional]
        
        expanded_genes = signature.genes + [g for g, _ in top_neighbors]
        
        if len(expanded_genes) > self.max_genes:
            expanded_genes = expanded_genes[:self.max_genes]
        
        signature.genes = expanded_genes
        signature.dam_expanded = True
        
        return signature
    
    def _build_cooccurrence_matrix(
        self,
        pathways_kb: Dict[str, List[str]],
        seed_genes: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Build gene co-occurrence matrix from KB"""
        
        cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for pathway_id, genes in pathways_kb.items():
            gene_set = set(genes)
            
            for seed_gene in seed_genes:
                if seed_gene in gene_set:
                    for other_gene in gene_set:
                        if other_gene != seed_gene:
                            cooccurrence[seed_gene][other_gene] += 1
        
        normalized = {}
        for seed_gene, neighbors in cooccurrence.items():
            max_count = max(neighbors.values()) if neighbors else 1
            normalized[seed_gene] = {
                gene: count / max_count
                for gene, count in neighbors.items()
            }
        
        return normalized
    
    def _make_id(self, name: str) -> str:
        return name.upper().replace(' ', '_').replace('-', '_').replace('/', '_')


# ============================================================
# SEMANTIC SEARCH
# ============================================================

@st.cache_resource
def load_embedding_model():
    """Load sentence transformer"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.warning(f"Could not load embedding model: {e}")
        return None


def semantic_search_with_scores(
    query: str,
    pathways_dict: Dict[str, List[str]],
    model,
    top_k: int = 50
) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    """Find relevant pathways with similarity scores"""
    
    if model is None:
        query_terms = set(query.lower().split())
        matches = {}
        scores = {}
        
        for pid, genes in pathways_dict.items():
            pid_terms = set(pid.lower().replace('_', ' ').split())
            overlap = len(query_terms & pid_terms)
            if overlap > 0:
                matches[pid] = genes
                scores[pid] = overlap / max(len(query_terms), len(pid_terms))
                if len(matches) >= top_k:
                    break
        
        return matches, scores
    
    query_emb = model.encode(query, convert_to_numpy=True)
    
    pathway_scores = []
    for pid, genes in pathways_dict.items():
        text = f"{pid.replace('_', ' ')} {' '.join(genes[:10])}"
        pathway_emb = model.encode(text, convert_to_numpy=True)
        
        similarity = float(np.dot(query_emb, pathway_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(pathway_emb) + 1e-8
        ))
        
        pathway_scores.append((pid, similarity))
    
    pathway_scores.sort(key=lambda x: x[1], reverse=True)
    top_pathways = pathway_scores[:top_k]
    
    selected_pathways = {pid: pathways_dict[pid] for pid, _ in top_pathways}
    similarity_dict = {pid: score for pid, score in top_pathways}
    
    return selected_pathways, similarity_dict


# ============================================================
# LLM FUNCTIONS
# ============================================================

def decompose_with_granularity(
    query: str,
    granularity_count: int,
    hf_token: str
) -> Optional[DecompositionResult]:
    """Use Qwen2.5-72B to decompose query into EXACTLY N mechanisms"""
    
    try:
        from huggingface_hub import InferenceClient
        
        client = InferenceClient(token=hf_token)
        
        prompt = f"""Decompose this biological query into EXACTLY {granularity_count} SPECIFIC, NON-OVERLAPPING molecular mechanisms.

Query: "{query}"

REQUIREMENTS:
1. Generate EXACTLY {granularity_count} mechanisms (no more, no less)
2. Each mechanism must be maximally granular (10-20 genes per mechanism)
3. No overlap between mechanisms
4. Each mechanism must be biologically distinct and specific

EXAMPLE for "glycolysis in macrophage in colorectal cancer" with count={granularity_count}:
If {granularity_count}=9, generate:
1. Macrophage metabolic changes in cancer
2. Macrophage stress responses in cancer
3. Macrophage effects in colorectal cancer tissue
4. Mitochondrial expression changes in macrophages
5. Fatty acid metabolism alterations
6. ROS production changes in cancer-associated macrophages
7. OXPHOS pathway changes in tumor macrophages
8. Other metabolic pathway alterations
9. Colorectal-specific metabolism markers in macrophages

Output JSON ONLY (no preamble, no markdown):
{{
  "facets": [
    {{
      "facet_id": "F1",
      "facet_name": "Macrophage Metabolic Changes in Cancer",
      "mechanism_queries": ["macrophage metabolism cancer glycolysis glucose"]
    }},
    {{
      "facet_id": "F2",
      "facet_name": "...",
      "mechanism_queries": ["..."]
    }}
  ]
}}"""

        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="Qwen/Qwen2.5-72B-Instruct",
            max_tokens=3000,
            temperature=0.3
        )
        
        raw = response.choices[0].message.content
        
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
            llm_model="Qwen/Qwen2.5-72B-Instruct"
        )
        
    except Exception as e:
        st.error(f"LLM decomposition failed: {e}")
        return None


def verify_signature_with_llm2(
    signature: GeneSignature,
    original_query: str,
    mode: str,
    llm2_model: str,
    hf_token: str,
    context_options: List[str]
) -> Dict[str, Any]:
    """Use biomedical LLM to verify/expand signature"""
    
    try:
        from huggingface_hub import InferenceClient
        
        client = InferenceClient(token=hf_token)
        
        context_parts = []
        
        if "Original query context" in context_options or "All of the above" in context_options:
            context_parts.append(f"Original Query: {original_query}")
        
        if "Mechanism description" in context_options or "All of the above" in context_options:
            context_parts.append(f"Mechanism: {signature.mechanism}")
        
        if "Pathway sources used" in context_options or "All of the above" in context_options:
            context_parts.append(f"Source Pathways: {', '.join(signature.source_pathways)}")
        
        context_parts.append(f"Current Genes ({len(signature.genes)}): {', '.join(signature.genes)}")
        
        context = "\n".join(context_parts)
        
        if mode == "verify_only":
            prompt = f"""Review this gene signature for biological correctness.

{context}

Task: Identify genes that do NOT belong in this signature based on biological relevance to the mechanism.

Output JSON only:
{{
  "genes_to_remove": ["GENE1", "GENE2"],
  "reasoning": {{
    "GENE1": "Explanation why this gene should be removed",
    "GENE2": "Explanation..."
  }}
}}

If all genes are correct, return empty list for genes_to_remove."""

        else:
            prompt = f"""Review this gene signature and suggest improvements.

{context}

Task:
1. Identify genes that do NOT belong (if any)
2. Suggest genes that SHOULD be added to better represent this mechanism (limit to 5 suggestions)

Output JSON only:
{{
  "genes_to_remove": ["GENE1"],
  "genes_to_add": ["GENE3", "GENE4"],
  "reasoning": {{
    "GENE1": "Why remove",
    "GENE3": "Why add",
    "GENE4": "Why add"
  }}
}}"""

        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=llm2_model,
            max_tokens=1500,
            temperature=0.2
        )
        
        raw = response.choices[0].message.content
        
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
        
        result = json.loads(cleaned)
        
        return result
        
    except Exception as e:
        st.error(f"LLM #2 verification failed: {e}")
        return {"genes_to_remove": [], "genes_to_add": [], "reasoning": {}}


# ============================================================
# UI STYLING
# ============================================================

def inject_modern_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
    h1 { 
        font-size: 2.5rem !important;
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
    .warning-box {
        background: rgba(251, 191, 36, 0.1);
        border-left: 4px solid #fbbf24;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
        color: #e2e8f0;
    }
    .success-box {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
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
    defaults = {
        'hf_token': None,
        'token_validated': False,
        'kb_path': None,
        'kb_loaded': False,
        'decomposition_result': None,
        'granularity_approved': False,
        'semantic_signatures': None,
        'dam_expanded_signatures': None,
        'llm2_suggestions': {},
        'verified_signatures': None,
        'final_approved_signatures': [],
        'execution_complete': False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# KB LOADING
# ============================================================

def load_knowledge_base() -> Optional[Dict[str, List[str]]]:
    """Load KB and return pathways dict"""
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
                st.session_state.kb_path = path
                st.session_state.kb_loaded = True
                return pathways
            except Exception as e:
                st.error(f"Error loading KB: {e}")
                return None
    
    st.error("Knowledge base not found in data/ directory")
    return None


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    """Render sidebar with token input"""
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        
        with st.expander("🔑 HuggingFace Token (Required)", expanded=not st.session_state.token_validated):
            st.caption("One token accesses both LLMs")
            st.caption("• Qwen2.5-72B (decomposition)")
            st.caption("• BioMedLM (verification)")
            
            token_input = st.text_input(
                "HuggingFace Token",
                type="password",
                value=st.session_state.hf_token or "",
                help="Get token at huggingface.co/settings/tokens"
            )
            
            if st.button("Validate Token"):
                try:
                    from huggingface_hub import InferenceClient
                    client = InferenceClient(token=token_input)
                    st.session_state.hf_token = token_input
                    st.session_state.token_validated = True
                    st.success("✅ Valid!")
                    time.sleep(0.5)
                    st.rerun()
                except:
                    st.error("❌ Invalid token")
        
        if st.session_state.token_validated:
            st.success("🔓 Token Active")
        
        st.markdown("---")
        st.markdown("### ℹ️ Pipeline Layers")
        st.caption("""
        **1. 🧠 Granularity Control**
        LLM decomposes query
        
        **2. 🔍 Semantic Selection**
        Build signatures from pathways
        
        **3. 🔬 DAM Expansion** *(Optional)*
        Add mathematical neighbors
        
        **4. ✅ LLM Verification** *(Optional)*
        Verify & expand genes
        
        **5. 👤 User Approval**
        Review & approve each signature
        """)


# ============================================================
# KB TAB
# ============================================================

def render_kb_tab():
    st.markdown("## 📚 Knowledge Base")
    
    st.markdown("""
    <div class="info-box">
    Pathways are SOURCE MATERIAL for building signatures.
    </div>
    """, unsafe_allow_html=True)
    
    pathways = load_knowledge_base()
    
    if pathways:
        st.success(f"✅ Loaded {len(pathways):,} pathways")
        
        sample_pid = list(pathways.keys())[0]
        sample_genes = pathways[sample_pid]
        
        with st.expander("📋 Sample Pathway"):
            st.caption(f"**{sample_pid}**")
            st.caption(f"Genes: {', '.join(sample_genes[:10])}... ({len(sample_genes)} total)")


# ============================================================
# GENERATION TAB - COMPLETE
# ============================================================

def render_generation_tab():
    st.markdown("## 🧬 Multi-Layer Signature Generation")
    
    if not st.session_state.kb_loaded:
        st.warning("⚠️ Please load knowledge base first")
        return
    
    if not st.session_state.token_validated:
        st.warning("⚠️ Please validate HuggingFace token in sidebar")
        return
    
    # Query input
    st.markdown("### Research Question")
    query = st.text_area(
        "",
        height=80,
        placeholder="Example: glycolysis metabolism changes in Macrophage in colorectal cancer",
        help="Describe your biological question",
        key="main_query"
    )
    
    # Gene constraints
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_count = st.number_input(
            "Total Signatures",
            min_value=5,
            max_value=100,
            value=35,
            step=5
        )
    
    with col2:
        min_genes = st.number_input(
            "Min Genes/Signature",
            min_value=5,
            max_value=30,
            value=10
        )
    
    with col3:
        max_genes = st.number_input(
            "Max Genes/Signature",
            min_value=10,
            max_value=50,
            value=20
        )
    
    if min_genes >= max_genes:
        st.error("❌ Min genes must be less than max genes!")
        return
    
    st.markdown("---")
    
    # LAYER 1
    render_layer1_granularity(query, target_count, min_genes, max_genes)
    
    if not st.session_state.granularity_approved:
        return
    
    # LAYER 2
    render_layer2_semantic(query, min_genes, max_genes)
    
    if not st.session_state.semantic_signatures:
        return
    
    # LAYER 3
    render_layer3_dam()
    
    # LAYER 4
    render_layer4_verification(query)
    
    # LAYER 5
    render_layer5_approval()


def render_layer1_granularity(query, target_count, min_genes, max_genes):
    """Layer 1: Granularity control"""
    
    st.markdown("### 🧠 Layer 1: Granularity Control")
    
    st.markdown("""
    <div class="info-box">
    <strong>LLM (Qwen2.5-72B)</strong> decomposes your query into granular mechanisms.
    Use the slider to control how many mechanisms.
    </div>
    """, unsafe_allow_html=True)
    
    granularity_level = st.slider(
        "Number of granular mechanisms:",
        min_value=3,
        max_value=60,
        value=9,
        step=1,
        help="Higher = more specific mechanisms"
    )
    
    st.caption(f"💡 **{granularity_level} mechanisms** × 3 variants = ~{granularity_level * 3} signatures")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 Generate Decomposition", type="primary", use_container_width=True):
            if not query or not query.strip():
                st.error("Please enter a query first")
                return
            
            with st.spinner(f"🧠 LLM generating {granularity_level} mechanisms..."):
                result = decompose_with_granularity(
                    query,
                    granularity_level,
                    st.session_state.hf_token
                )
                
                if result and result.facets:
                    st.session_state.decomposition_result = result
                    st.session_state.granularity_approved = False
                    st.success(f"✅ Generated {len(result.facets)} mechanisms!")
                    time.sleep(0.5)
                    st.rerun()
    
    with col2:
        if st.session_state.decomposition_result:
            if st.button("🔄 Regenerate", use_container_width=True):
                st.session_state.decomposition_result = None
                st.session_state.granularity_approved = False
                st.rerun()
    
    if st.session_state.decomposition_result:
        st.markdown("---")
        st.markdown("#### 📋 Generated Mechanisms:")
        
        result = st.session_state.decomposition_result
        
        for i, facet in enumerate(result.facets, 1):
            with st.expander(f"{i}. {facet['facet_name']}", expanded=(i <= 3)):
                st.caption(f"**Query:** {', '.join(facet.get('mechanism_queries', []))}")
        
        st.markdown(f"""
        <div class="warning-box">
        <strong>⚠️ Review {len(result.facets)} mechanisms above.</strong><br>
        Satisfied? Click "Approve". Need more granularity? Adjust slider & regenerate.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Approve & Continue", type="primary", use_container_width=True):
                st.session_state.granularity_approved = True
                st.session_state['target_count'] = target_count
                st.success("✅ Mechanisms approved!")
                time.sleep(0.5)
                st.rerun()
        
        with col2:
            if st.button("❌ Reject", use_container_width=True):
                st.session_state.decomposition_result = None
                st.session_state.granularity_approved = False
                st.rerun()


def render_layer2_semantic(query, min_genes, max_genes):
    """Layer 2: Semantic selection"""
    
    st.markdown("---")
    st.markdown("### 🔍 Layer 2: Semantic Signature Building")
    
    st.markdown(f"""
    <div class="info-box">
    Building signatures: Semantic search → Gene scoring → Size limits ({min_genes}-{max_genes} genes)
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Build Semantic Signatures", type="primary", use_container_width=True):
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        try:
            status.info("📚 Loading knowledge base...")
            progress_bar.progress(10)
            
            pathways_dict = load_knowledge_base()
            if not pathways_dict:
                st.error("Cannot load KB")
                return
            
            status.info("🔍 Loading semantic search...")
            progress_bar.progress(20)
            
            embedding_model = load_embedding_model()
            
            status.info("🧬 Building signatures...")
            progress_bar.progress(30)
            
            builder = SignatureBuilder(min_genes=min_genes, max_genes=max_genes)
            all_signatures = []
            
            facets = st.session_state.decomposition_result.facets
            
            for i, facet in enumerate(facets):
                facet_name = facet['facet_name']
                mechanism_queries = facet.get('mechanism_queries', [facet_name])
                
                status.info(f"   {facet_name}...")
                
                for mech_query in mechanism_queries[:1]:
                    relevant_pathways, pathway_similarities = semantic_search_with_scores(
                        mech_query,
                        pathways_dict,
                        embedding_model,
                        top_k=50
                    )
                    
                    if not relevant_pathways:
                        continue
                    
                    mechanism_sigs = builder.build_multiple_mechanisms(
                        facet_name,
                        relevant_pathways,
                        pathway_similarities,
                        num_variants=3
                    )
                    
                    all_signatures.extend(mechanism_sigs)
                
                progress = 30 + int((i + 1) / len(facets) * 50)
                progress_bar.progress(progress)
            
            target_count = st.session_state.get('target_count', 35)
            if len(all_signatures) > target_count:
                all_signatures.sort(key=lambda s: s.confidence, reverse=True)
                all_signatures = all_signatures[:target_count]
            
            status.success(f"✅ Built {len(all_signatures)} signatures!")
            progress_bar.progress(100)
            
            st.session_state.semantic_signatures = all_signatures
            
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            status.error(f"❌ Failed: {e}")
            progress_bar.empty()
    
    if st.session_state.semantic_signatures:
        st.markdown(f"""
        <div class="success-box">
        ✅ <strong>{len(st.session_state.semantic_signatures)} semantic signatures built</strong>
        </div>
        """, unsafe_allow_html=True)
        
        gene_counts = [len(sig.genes) for sig in st.session_state.semantic_signatures]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Signatures", len(st.session_state.semantic_signatures))
        with col2:
            st.metric("Avg Genes", f"{np.mean(gene_counts):.1f}")
        with col3:
            st.metric("Range", f"{min(gene_counts)}-{max(gene_counts)}")


def render_layer3_dam():
    """Layer 3: DAM expansion"""
    
    st.markdown("---")
    st.markdown("### 🔬 Layer 3: DAM Expansion (Optional)")
    
    st.markdown("""
    <div class="info-box">
    <strong>Optional:</strong> Add mathematically related genes based on co-occurrence.
    </div>
    """, unsafe_allow_html=True)
    
    enable_dam = st.checkbox(
        "🔬 Enable DAM Neighbor Expansion",
        value=False,
        help="Add related genes"
    )
    
    if enable_dam:
        col1, col2 = st.columns(2)
        
        with col1:
            expansion_strength = st.slider(
                "Expansion strength",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
        
        with col2:
            max_neighbors = st.slider(
                "Max neighbors",
                min_value=1,
                max_value=10,
                value=5,
                step=1
            )
        
        if st.button("🔬 Expand with DAM", type="primary", use_container_width=True):
            
            if not st.session_state.semantic_signatures:
                st.error("Build semantic signatures first")
                return
            
            pathways_dict = load_knowledge_base()
            builder = SignatureBuilder()
            expanded_signatures = []
            
            progress_bar = st.progress(0)
            
            for i, sig in enumerate(st.session_state.semantic_signatures):
                expanded_sig = GeneSignature(
                    signature_id=sig.signature_id,
                    signature_name=sig.signature_name,
                    genes=sig.genes.copy(),
                    facet=sig.facet,
                    mechanism=sig.mechanism,
                    confidence=sig.confidence,
                    gene_scores=sig.gene_scores.copy(),
                    source_pathways=sig.source_pathways.copy()
                )
                
                expanded_sig = builder.expand_with_dam(
                    expanded_sig,
                    pathways_dict,
                    expansion_strength=expansion_strength,
                    max_additional=max_neighbors
                )
                
                expanded_signatures.append(expanded_sig)
                progress_bar.progress((i + 1) / len(st.session_state.semantic_signatures))
            
            st.session_state.dam_expanded_signatures = expanded_signatures
            st.success(f"✅ Expanded {len(expanded_signatures)} signatures!")
            
            time.sleep(1)
            st.rerun()
        
        if st.session_state.dam_expanded_signatures:
            st.success("✅ DAM expansion complete")
    
    else:
        st.info("DAM disabled")
        st.session_state.dam_expanded_signatures = None


def render_layer4_verification(query):
    """Layer 4: LLM verification"""
    
    st.markdown("---")
    st.markdown("### ✅ Layer 4: LLM Verification (Optional)")
    
    st.markdown("""
    <div class="info-box">
    <strong>Optional:</strong> Use BioMedLM to verify genes and suggest additions.
    </div>
    """, unsafe_allow_html=True)
    
    verification_mode = st.radio(
        "Verification mode:",
        options=[
            "None (Skip)",
            "Verify Only",
            "Verify + Expand"
        ],
        index=0
    )
    
    if verification_mode != "None (Skip)":
        
        llm2_model = st.selectbox(
            "Select model:",
            options=[
                "stanford-crfm/BioMedLM",
                "microsoft/BioGPT-Large"
            ]
        )
        
        context_options = st.multiselect(
            "Context:",
            options=[
                "Original query context",
                "Pathway sources used",
                "Mechanism description",
                "All of the above"
            ],
            default=["All of the above"]
        )
        
        if st.button("🧬 Run Verification", type="primary", use_container_width=True):
            
            signatures_to_verify = (
                st.session_state.dam_expanded_signatures 
                if st.session_state.dam_expanded_signatures 
                else st.session_state.semantic_signatures
            )
            
            if not signatures_to_verify:
                st.error("No signatures")
                return
            
            mode = "verify_only" if "Only" in verification_mode else "verify_expand"
            
            suggestions = {}
            progress_bar = st.progress(0)
            status = st.empty()
            
            for i, sig in enumerate(signatures_to_verify):
                status.info(f"Verifying: {sig.signature_name}...")
                
                result = verify_signature_with_llm2(
                    sig,
                    query,
                    mode,
                    llm2_model,
                    st.session_state.hf_token,
                    context_options
                )
                
                suggestions[sig.signature_id] = result
                progress_bar.progress((i + 1) / len(signatures_to_verify))
            
            st.session_state.llm2_suggestions = suggestions
            st.success(f"✅ Verified {len(signatures_to_verify)} signatures!")
            
            time.sleep(1)
            st.rerun()
    
    else:
        st.info("Verification disabled")
        st.session_state.llm2_suggestions = {}


def render_layer5_approval():
    """Layer 5: Approval interface"""
    
    st.markdown("---")
    st.markdown("### 👤 Layer 5: Review & Approve")
    
    signatures_to_approve = (
        st.session_state.dam_expanded_signatures 
        if st.session_state.dam_expanded_signatures 
        else st.session_state.semantic_signatures
    )
    
    if not signatures_to_approve:
        return
    
    st.markdown(f"""
    <div class="warning-box">
    <strong>Review {len(signatures_to_approve)} signatures</strong><br>
    Select genes to keep for each signature.
    </div>
    """, unsafe_allow_html=True)
    
    llm2_suggestions = st.session_state.llm2_suggestions
    
    for i, sig in enumerate(signatures_to_approve):
        with st.expander(
            f"📋 {sig.signature_name} ({len(sig.genes)} genes)",
            expanded=(i == 0)
        ):
            
            st.caption(f"**Facet:** {sig.facet}")
            st.caption(f"**Mechanism:** {sig.mechanism}")
            st.caption(f"**Confidence:** {sig.confidence:.3f}")
            
            if sig.dam_expanded:
                st.caption("🔬 DAM Expanded")
            
            sig_suggestions = llm2_suggestions.get(sig.signature_id, {})
            
            if sig_suggestions.get('genes_to_remove'):
                st.warning(f"⚠️ LLM suggests removing: {', '.join(sig_suggestions['genes_to_remove'])}")
            
            if sig_suggestions.get('genes_to_add'):
                st.info(f"➕ LLM suggests adding: {', '.join(sig_suggestions['genes_to_add'])}")
            
            st.markdown("**Gene Selection:**")
            
            select_all_key = f"select_all_{sig.signature_id}"
            select_all = st.checkbox("Select All", key=select_all_key, value=True)
            
            selected_genes = []
            
            for gene in sig.genes:
                is_flagged = gene in sig_suggestions.get('genes_to_remove', [])
                
                col1, col2, col3 = st.columns([1, 5, 2])
                
                with col1:
                    gene_key = f"gene_{sig.signature_id}_{gene}"
                    is_selected = st.checkbox(
                        "",
                        key=gene_key,
                        value=select_all and not is_flagged,
                        label_visibility="collapsed"
                    )
                
                with col2:
                    label = f"**{gene}**"
                    if is_flagged:
                        label += " ⚠️"
                    st.markdown(label)
                
                with col3:
                    if is_flagged:
                        reasoning = sig_suggestions.get('reasoning', {}).get(gene, "")
                        if reasoning and st.button("Why?", key=f"why_{sig.signature_id}_{gene}"):
                            st.caption(reasoning)
                
                if is_selected:
                    selected_genes.append(gene)
            
            if sig_suggestions.get('genes_to_add'):
                st.markdown("**Suggested Additions:**")
                
                for gene in sig_suggestions['genes_to_add']:
                    col1, col2, col3 = st.columns([1, 5, 2])
                    
                    with col1:
                        add_key = f"add_{sig.signature_id}_{gene}"
                        should_add = st.checkbox("", key=add_key, value=False)
                    
                    with col2:
                        st.markdown(f"**{gene}** ➕")
                    
                    with col3:
                        reasoning = sig_suggestions.get('reasoning', {}).get(gene, "")
                        if reasoning and st.button("Why?", key=f"why_add_{sig.signature_id}_{gene}"):
                            st.caption(reasoning)
                    
                    if should_add:
                        selected_genes.append(gene)
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Approve", key=f"approve_{sig.signature_id}", use_container_width=True):
                    
                    approved_sig = GeneSignature(
                        signature_id=sig.signature_id,
                        signature_name=sig.signature_name,
                        genes=selected_genes,
                        facet=sig.facet,
                        mechanism=sig.mechanism,
                        confidence=sig.confidence,
                        gene_scores=sig.gene_scores,
                        source_pathways=sig.source_pathways,
                        dam_expanded=sig.dam_expanded,
                        llm2_verified=bool(sig_suggestions)
                    )
                    
                    existing_ids = [s.signature_id for s in st.session_state.final_approved_signatures]
                    if sig.signature_id in existing_ids:
                        idx = existing_ids.index(sig.signature_id)
                        st.session_state.final_approved_signatures[idx] = approved_sig
                    else:
                        st.session_state.final_approved_signatures.append(approved_sig)
                    
                    st.success(f"✅ Approved with {len(selected_genes)} genes")
            
            with col2:
                if st.button("❌ Reject", key=f"reject_{sig.signature_id}", use_container_width=True):
                    st.warning("Signature rejected")
    
    if st.session_state.final_approved_signatures:
        st.markdown("---")
        st.markdown("### 💾 Export Approved Signatures")
        
        st.success(f"✅ {len(st.session_state.final_approved_signatures)} signatures approved")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gmt_lines = []
            for sig in st.session_state.final_approved_signatures:
                sig_id = sig.signature_id
                desc = f"{sig.facet}|{sig.mechanism}|{len(sig.genes)}genes|conf:{sig.confidence:.3f}"
                if sig.dam_expanded:
                    desc += "|DAM"
                if sig.llm2_verified:
                    desc += "|LLM2"
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
            results = {
                'query': st.session_state.get('main_query', ''),
                'decomposition': st.session_state.decomposition_result.facets if st.session_state.decomposition_result else [],
                'signatures': [sig.to_dict() for sig in st.session_state.final_approved_signatures],
                'total_signatures': len(st.session_state.final_approved_signatures),
                'timestamp': datetime.now().isoformat()
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
        page_title="Signature Generator",
        page_icon="🧬",
        layout="wide"
    )
    
    initialize_session_state()
    inject_modern_css()
    
    st.markdown("""
    <div style='text-align: center; padding: 32px 0 16px 0;'>
        <h1>🧬 Multi-Layer Signature Generator</h1>
        <p style='font-size: 1.1rem; color: #94a3b8;'>
            LLM Granularity → Semantic Selection → DAM Expansion → LLM Verification → Approval
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    render_sidebar()
    
    tab1, tab2 = st.tabs(["📚 Knowledge Base", "🧬 Generate Signatures"])
    
    with tab1:
        render_kb_tab()
    
    with tab2:
        render_generation_tab()


if __name__ == "__main__":
    main()

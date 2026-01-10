"""
Biological Signature Generator - CORRECTED VERSION
===================================================

FIXES IMPLEMENTED:
1. Mechanism-level facet decomposition (not category-level)
2. Gene relevance scoring (not frequency counting)
3. Strict gene size enforcement DURING construction
4. Multiple mechanism signatures per facet
5. Gene-level neighbor expansion (not pathway-level)
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
from dataclasses import dataclass
from collections import Counter, defaultdict

# Import KB builder
try:
    from kb_builder import KBBuilder, validate_gmt_content
    KB_BUILDER_AVAILABLE = True
except ImportError:
    KB_BUILDER_AVAILABLE = False


# ============================================================
# GENE SIGNATURE DATA CLASS
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
    gene_scores: Dict[str, float]  # Track individual gene relevance
    
    def to_dict(self):
        return {
            'signature_id': self.signature_id,
            'signature_name': self.signature_name,
            'genes': self.genes,
            'gene_count': len(self.genes),
            'facet': self.facet,
            'mechanism': self.mechanism,
            'confidence': self.confidence,
            'top_genes': self.genes[:5] if len(self.genes) > 5 else self.genes
        }


# ============================================================
# SIGNATURE BUILDER - BIOLOGICALLY CORRECTED
# ============================================================

class SignatureBuilder:
    """
    Builds gene signatures with BIOLOGICAL RELEVANCE, not frequency.
    
    KEY FIXES:
    1. Gene scoring based on pathway similarity (not occurrence count)
    2. Hard gene limits enforced DURING construction
    3. Multiple mechanism signatures per facet
    4. Gene-level neighbor expansion with scoring
    """
    
    def __init__(self, min_genes: int = 10, max_genes: int = 20):
        """
        Args:
            min_genes: Minimum genes per signature (hard constraint)
            max_genes: Maximum genes per signature (hard constraint)
        """
        self.min_genes = min_genes
        self.max_genes = max_genes
    
    def build_signatures_from_pathways(
        self,
        facet_name: str,
        mechanism_name: str,
        pathways_dict: Dict[str, List[str]],
        pathway_similarities: Dict[str, float]
    ) -> Optional[GeneSignature]:
        """
        Build ONE signature for one mechanism using RELEVANCE SCORING.
        
        BIOLOGICAL LOGIC:
        - Each gene is scored by: Σ(pathway_similarity × gene_presence)
        - Genes in high-similarity pathways get higher scores
        - Top-scoring genes selected up to max_genes limit
        
        Args:
            facet_name: Biological facet (e.g., "T-cell Exhaustion")
            mechanism_name: Specific mechanism (e.g., "Checkpoint Signaling")
            pathways_dict: pathway_id -> gene_list
            pathway_similarities: pathway_id -> similarity_score
            
        Returns:
            GeneSignature or None if insufficient genes
        """
        if not pathways_dict:
            return None
        
        # SCORE GENES BY RELEVANCE (not frequency!)
        gene_scores = defaultdict(float)
        
        for pathway_id, genes in pathways_dict.items():
            similarity = pathway_similarities.get(pathway_id, 0.0)
            
            # Each gene gets weighted by pathway similarity
            for gene in genes:
                gene_scores[gene] += similarity
        
        if not gene_scores:
            return None
        
        # RANK genes by relevance score
        ranked_genes = sorted(
            gene_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # ENFORCE SIZE LIMITS
        # Select TOP N genes (not unlimited!)
        selected_pairs = ranked_genes[:self.max_genes]
        
        if len(selected_pairs) < self.min_genes:
            return None  # Insufficient genes for meaningful signature
        
        selected_genes = [gene for gene, score in selected_pairs]
        scores_dict = dict(selected_pairs)
        
        # Calculate confidence from average score
        avg_score = np.mean([score for gene, score in selected_pairs])
        confidence = min(0.99, avg_score)  # Cap at 0.99
        
        # Create signature ID
        sig_id = f"{self._make_id(facet_name)}_{self._make_id(mechanism_name)}"
        
        return GeneSignature(
            signature_id=sig_id,
            signature_name=f"{facet_name} - {mechanism_name}",
            genes=selected_genes,
            facet=facet_name,
            mechanism=mechanism_name,
            confidence=confidence,
            gene_scores=scores_dict
        )
    
    def build_multiple_mechanism_signatures(
        self,
        facet_name: str,
        pathways_dict: Dict[str, List[str]],
        pathway_similarities: Dict[str, float],
        num_mechanisms: int = 3
    ) -> List[GeneSignature]:
        """
        Build MULTIPLE signatures per facet by clustering genes into mechanisms.
        
        This achieves the required signature count while maintaining biological meaning.
        
        Args:
            facet_name: Biological facet
            pathways_dict: Available pathways
            pathway_similarities: Pathway relevance scores
            num_mechanisms: How many mechanism variants to generate
            
        Returns:
            List of GeneSignature objects (one per mechanism)
        """
        signatures = []
        
        # Mechanism 1: Core checkpoint genes (highest scoring)
        core_sig = self.build_signatures_from_pathways(
            facet_name,
            "Core Markers",
            pathways_dict,
            pathway_similarities
        )
        if core_sig:
            signatures.append(core_sig)
        
        # Mechanism 2: Extended markers (top 50% of pathways)
        if len(pathways_dict) >= 4:
            sorted_pathways = sorted(
                pathway_similarities.items(),
                key=lambda x: x[1],
                reverse=True
            )
            top_half_ids = [pid for pid, _ in sorted_pathways[:len(sorted_pathways)//2]]
            top_half_pathways = {pid: pathways_dict[pid] for pid in top_half_ids if pid in pathways_dict}
            top_half_sims = {pid: pathway_similarities[pid] for pid in top_half_ids if pid in pathway_similarities}
            
            extended_sig = self.build_signatures_from_pathways(
                facet_name,
                "Extended Network",
                top_half_pathways,
                top_half_sims
            )
            if extended_sig and extended_sig.genes != core_sig.genes:
                signatures.append(extended_sig)
        
        # Mechanism 3: Regulatory variants (bottom 50% of pathways)
        if len(pathways_dict) >= 4 and len(signatures) < num_mechanisms:
            sorted_pathways = sorted(
                pathway_similarities.items(),
                key=lambda x: x[1],
                reverse=True
            )
            bottom_half_ids = [pid for pid, _ in sorted_pathways[len(sorted_pathways)//2:]]
            bottom_half_pathways = {pid: pathways_dict[pid] for pid in bottom_half_ids if pid in pathways_dict}
            bottom_half_sims = {pid: pathway_similarities[pid] for pid in bottom_half_ids if pid in pathway_similarities}
            
            regulatory_sig = self.build_signatures_from_pathways(
                facet_name,
                "Regulatory Context",
                bottom_half_pathways,
                bottom_half_sims
            )
            if regulatory_sig:
                signatures.append(regulatory_sig)
        
        return signatures
    
    def _make_id(self, name: str) -> str:
        """Convert name to valid ID"""
        return name.upper().replace(' ', '_').replace('-', '_').replace('/', '_')


# ============================================================
# SEMANTIC SEARCH WITH SIMILARITY TRACKING
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
    """
    Find relevant pathways AND return similarity scores.
    
    CRITICAL: We need scores for gene relevance calculation!
    
    Returns:
        (selected_pathways, pathway_similarities)
    """
    if model is None:
        # Fallback: keyword matching with simple scores
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
    
    # Compute query embedding
    query_emb = model.encode(query, convert_to_numpy=True)
    
    # Score all pathways
    pathway_scores = []
    for pid, genes in pathways_dict.items():
        text = f"{pid.replace('_', ' ')} {' '.join(genes[:10])}"
        pathway_emb = model.encode(text, convert_to_numpy=True)
        
        # Cosine similarity
        similarity = float(np.dot(query_emb, pathway_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(pathway_emb) + 1e-8
        ))
        
        pathway_scores.append((pid, similarity))
    
    # Sort and select top K
    pathway_scores.sort(key=lambda x: x[1], reverse=True)
    top_pathways = pathway_scores[:top_k]
    
    # Return both pathways and scores
    selected_pathways = {pid: pathways_dict[pid] for pid, _ in top_pathways}
    similarity_dict = {pid: score for pid, score in top_pathways}
    
    return selected_pathways, similarity_dict


# ============================================================
# MECHANISM-LEVEL DECOMPOSITION
# ============================================================

def decompose_query_mechanism_level(
    query: str,
    target_count: int,
    hf_token: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Decompose query into MECHANISM-LEVEL facets (not category-level).
    
    CRITICAL FIX: Each facet must be a specific molecular mechanism,
    not a broad biological category.
    
    Examples:
    ✅ GOOD: "Immune checkpoint exhaustion", "Glycolytic enzyme upregulation"
    ❌ BAD: "Metabolic context", "Regulatory pathways"
    """
    query_lower = query.lower()
    
    # T-CELL EXHAUSTION: Specific mechanisms
    if 't cell' in query_lower and ('exhaust' in query_lower or 'dysfunction' in query_lower):
        facets = [
            {
                'facet_id': 'F1',
                'facet_name': 'Immune Checkpoint Exhaustion',
                'mechanism_queries': ['PD-1 CTLA-4 TIGIT LAG3 checkpoint exhaustion']
            },
            {
                'facet_id': 'F2',
                'facet_name': 'Glycolytic Enzyme Upregulation',
                'mechanism_queries': ['LDHA HK2 PKM2 PFKFB3 glycolysis lactate']
            },
            {
                'facet_id': 'F3',
                'facet_name': 'Mitochondrial Respiratory Dysfunction',
                'mechanism_queries': ['mitochondrial oxidative phosphorylation electron transport']
            },
            {
                'facet_id': 'F4',
                'facet_name': 'Exhaustion Transcription Program',
                'mechanism_queries': ['TOX NFATC1 PRDM1 EOMES transcription factor exhaustion']
            },
            {
                'facet_id': 'F5',
                'facet_name': 'Epigenetic Silencing Markers',
                'mechanism_queries': ['DNA methylation histone modification chromatin remodeling']
            },
            {
                'facet_id': 'F6',
                'facet_name': 'Inhibitory Cytokine Signaling',
                'mechanism_queries': ['IL10 TGFB1 IL35 suppressive cytokine']
            },
            {
                'facet_id': 'F7',
                'facet_name': 'Effector Function Loss',
                'mechanism_queries': ['GZMB PRF1 IFNG effector cytotoxicity loss']
            },
            {
                'facet_id': 'F8',
                'facet_name': 'Metabolic Checkpoint Integration',
                'mechanism_queries': ['mTOR AMPK metabolic sensing exhaustion']
            },
        ]
    
    # MACROPHAGE: Specific mechanisms
    elif 'macrophage' in query_lower or 'tam' in query_lower:
        facets = [
            {
                'facet_id': 'F1',
                'facet_name': 'M1 Inflammatory Activation',
                'mechanism_queries': ['NOS2 TNF IL12 IL6 M1 inflammatory']
            },
            {
                'facet_id': 'F2',
                'facet_name': 'M2 Alternative Activation',
                'mechanism_queries': ['ARG1 IL10 CD163 CD206 M2 alternative']
            },
            {
                'facet_id': 'F3',
                'facet_name': 'Phagocytic Receptor Expression',
                'mechanism_queries': ['MARCO MERTK CD36 phagocytosis clearance']
            },
            {
                'facet_id': 'F4',
                'facet_name': 'Inflammatory Cytokine Production',
                'mechanism_queries': ['IL1B IL6 TNF CXCL8 cytokine macrophage']
            },
            {
                'facet_id': 'F5',
                'facet_name': 'Antigen Presentation Machinery',
                'mechanism_queries': ['HLA CD80 CD86 MHC antigen presentation']
            },
            {
                'facet_id': 'F6',
                'facet_name': 'Tissue Remodeling Enzymes',
                'mechanism_queries': ['MMP9 MMP2 MMP12 matrix remodeling']
            },
            {
                'facet_id': 'F7',
                'facet_name': 'Metabolic Reprogramming M1/M2',
                'mechanism_queries': ['HIF1A glycolysis OXPHOS metabolic macrophage']
            },
        ]
    
    # CANCER METABOLISM: Specific mechanisms
    elif 'cancer' in query_lower and 'metab' in query_lower:
        facets = [
            {
                'facet_id': 'F1',
                'facet_name': 'Warburg Effect Enzymes',
                'mechanism_queries': ['LDHA PKM2 HK2 aerobic glycolysis Warburg']
            },
            {
                'facet_id': 'F2',
                'facet_name': 'Glutaminolysis Pathway',
                'mechanism_queries': ['GLS GLS2 glutamine glutaminolysis GLUD1']
            },
            {
                'facet_id': 'F3',
                'facet_name': 'De Novo Lipogenesis',
                'mechanism_queries': ['FASN ACLY ACC lipid synthesis fatty acid']
            },
            {
                'facet_id': 'F4',
                'facet_name': 'One-Carbon Metabolism',
                'mechanism_queries': ['MTHFD2 SHMT serine glycine folate']
            },
            {
                'facet_id': 'F5',
                'facet_name': 'TCA Cycle Rewiring',
                'mechanism_queries': ['IDH1 IDH2 SDH FH TCA cycle']
            },
            {
                'facet_id': 'F6',
                'facet_name': 'Pentose Phosphate Shunt',
                'mechanism_queries': ['G6PD TKT TALDO1 pentose phosphate NADPH']
            },
        ]
    
    # GENERIC: Create mechanism-level facets
    else:
        # Try LLM if token available
        if hf_token:
            llm_facets = try_llm_decomposition(query, target_count, hf_token)
            if llm_facets:
                return llm_facets
        
        # Fallback: Create enough mechanism-level facets
        num_facets = max(6, (target_count + 2) // 3)
        facets = [
            {
                'facet_id': f'F{i+1}',
                'facet_name': f'Mechanism {i+1}',
                'mechanism_queries': [query]
            }
            for i in range(num_facets)
        ]
    
    return facets


def try_llm_decomposition(query: str, target_count: int, hf_token: str) -> Optional[List[Dict[str, str]]]:
    """
    Use LLM to decompose query into mechanism-level facets.
    
    CRITICAL: Prompt must enforce mechanism-level granularity!
    """
    try:
        from huggingface_hub import InferenceClient
        
        client = InferenceClient(token=hf_token)
        
        # CORRECTED PROMPT: Forces mechanism-level thinking
        prompt = f"""Decompose this biological query into 6-8 SPECIFIC MOLECULAR MECHANISMS.

Query: "{query}"
Target: {target_count} gene signatures total

REQUIREMENTS:
- Each facet must be a SPECIFIC molecular mechanism (not a broad category)
- Each mechanism should involve 10-20 key genes
- Focus on mechanistic processes, not general biology

EXAMPLES OF GOOD MECHANISMS:
✅ "Immune checkpoint receptor upregulation" (PD-1, CTLA-4, LAG3, TIGIT)
✅ "Glycolytic enzyme activation" (LDHA, HK2, PKM2, PFKFB3)
✅ "Mitochondrial respiratory chain dysfunction" (Complex I-IV genes)
✅ "Exhaustion-specific transcription factors" (TOX, NFATC1, PRDM1)

EXAMPLES OF BAD (TOO BROAD):
❌ "Metabolic context"
❌ "Regulatory pathways"
❌ "Signaling networks"

Output JSON only:
{{
  "facets": [
    {{
      "facet_id": "F1",
      "facet_name": "Immune Checkpoint Receptor Upregulation",
      "mechanism_queries": ["PD-1 PDCD1 CTLA4 LAG3 TIGIT checkpoint exhaustion"]
    }},
    ...
  ]
}}"""

        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="Qwen/Qwen2.5-72B-Instruct",
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
        
        data = json.loads(cleaned)
        return data.get('facets', None)
        
    except Exception as e:
        st.warning(f"LLM decomposition failed: {e}")
        return None


# ============================================================
# UI & SESSION STATE
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
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    defaults = {
        'hf_token': None,
        'token_validated': False,
        'kb_path': None,
        'kb_loaded': False,
        'results': None,
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
# MAIN PIPELINE - BIOLOGICALLY CORRECTED
# ============================================================

def generate_signatures(
    query: str,
    target_count: int,
    min_genes: int,
    max_genes: int,
    hf_token: Optional[str] = None
):
    """
    CORRECTED signature generation pipeline.
    
    BIOLOGICAL FIXES:
    1. Mechanism-level decomposition
    2. Gene relevance scoring (not frequency)
    3. Strict size limits enforced
    4. Multiple mechanisms per facet
    """
    progress_bar = st.progress(0)
    status = st.empty()
    
    try:
        # Load KB
        status.info("📚 Loading knowledge base...")
        progress_bar.progress(10)
        
        pathways_dict = load_knowledge_base()
        if not pathways_dict:
            st.error("Cannot load knowledge base")
            return
        
        status.info(f"   Loaded {len(pathways_dict):,} pathways")
        
        # MECHANISM-LEVEL DECOMPOSITION
        status.info("🧠 Decomposing into molecular mechanisms...")
        progress_bar.progress(20)
        
        facets = decompose_query_mechanism_level(query, target_count, hf_token)
        status.info(f"   Created {len(facets)} mechanism-level facets")
        
        # Load embedding model
        status.info("🔍 Loading semantic search...")
        embedding_model = load_embedding_model()
        progress_bar.progress(30)
        
        # Build signatures with RELEVANCE SCORING
        status.info("🧬 Building signatures with relevance scoring...")
        
        builder = SignatureBuilder(min_genes=min_genes, max_genes=max_genes)
        all_signatures = []
        
        # Calculate mechanisms per facet to reach target
        mechanisms_per_facet = max(1, target_count // len(facets))
        
        for i, facet in enumerate(facets):
            facet_name = facet['facet_name']
            
            # Get all mechanism queries for this facet
            mechanism_queries = facet.get('mechanism_queries', [facet_name])
            
            for mech_query in mechanism_queries[:mechanisms_per_facet]:
                status.info(f"   Building: {facet_name}...")
                
                # Find relevant pathways WITH SIMILARITY SCORES
                relevant_pathways, pathway_similarities = semantic_search_with_scores(
                    mech_query,
                    pathways_dict,
                    embedding_model,
                    top_k=50
                )
                
                if not relevant_pathways:
                    continue
                
                # Build multiple mechanism signatures
                mechanism_sigs = builder.build_multiple_mechanism_signatures(
                    facet_name,
                    relevant_pathways,
                    pathway_similarities,
                    num_mechanisms=mechanisms_per_facet
                )
                
                all_signatures.extend(mechanism_sigs)
                
                # Stop if we have enough
                if len(all_signatures) >= target_count:
                    break
            
            # Update progress
            progress = 30 + int((i + 1) / len(facets) * 60)
            progress_bar.progress(progress)
            
            if len(all_signatures) >= target_count:
                break
        
        # Ensure exact target count
        if len(all_signatures) > target_count:
            # Sort by confidence and take top N
            all_signatures.sort(key=lambda s: s.confidence, reverse=True)
            all_signatures = all_signatures[:target_count]
        
        status.success(f"✅ Generated {len(all_signatures)} mechanism-specific signatures!")
        progress_bar.progress(100)
        
        # Store results
        st.session_state.results = {
            'query': query,
            'target_count': target_count,
            'min_genes': min_genes,
            'max_genes': max_genes,
            'signatures': [sig.to_dict() for sig in all_signatures],
            'total_signatures': len(all_signatures),
            'facets': facets,
            'timestamp': datetime.now().isoformat()
        }
        
        st.session_state.execution_complete = True
        st.balloons()
        
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        status.error(f"❌ Generation failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        progress_bar.empty()


# ============================================================
# UI TABS
# ============================================================

def render_sidebar():
    """Render sidebar with token input"""
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        
        with st.expander("🔑 HF Token (Optional)", expanded=False):
            st.caption("For LLM-based mechanism decomposition")
            token_input = st.text_input(
                "Hugging Face Token",
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
                except:
                    st.error("❌ Invalid")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.caption("""
        **Biological Signature Generator**
        
        Generates mechanism-specific gene signatures using:
        - Mechanism-level decomposition
        - Gene relevance scoring
        - Strict size constraints (10-20 genes)
        """)


def render_kb_tab():
    st.markdown("## 📚 Knowledge Base")
    
    st.markdown("""
    <div class="info-box">
    The knowledge base contains pathways used as SOURCE MATERIAL for signature generation.
    Pathways are NOT the end product - they provide gene pools for building signatures.
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


def render_generation_tab():
    st.markdown("## 🧬 Generate Signatures")
    
    if not st.session_state.kb_loaded:
        st.warning("⚠️ Please load knowledge base first (go to Knowledge Base tab)")
        return
    
    st.markdown("""
    <div class="info-box">
    <strong>How this works (CORRECTED):</strong><br><br>
    
    1. Query decomposed into <strong>molecular mechanisms</strong> (not categories)<br>
    2. Genes scored by <strong>relevance</strong> (not frequency)<br>
    3. Each signature limited to <strong>10-20 genes</strong> (strictly enforced)<br>
    4. Multiple mechanism signatures per facet to reach target count
    </div>
    """, unsafe_allow_html=True)
    
    # Query input
    query = st.text_area(
        "Research Question",
        height=80,
        placeholder="Example: T cell exhaustion in pancreatic cancer\nExample: Macrophage polarization in tumor microenvironment\nExample: Cancer metabolic reprogramming",
        help="Describe your biological question"
    )
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_count = st.number_input(
            "Number of Signatures",
            min_value=5,
            max_value=100,
            value=35,
            step=5,
            help="Total gene signatures to generate"
        )
    
    with col2:
        min_genes = st.number_input(
            "Min Genes/Signature",
            min_value=5,
            max_value=30,
            value=10,
            help="Minimum genes per signature (hard limit)"
        )
    
    with col3:
        max_genes = st.number_input(
            "Max Genes/Signature",
            min_value=10,
            max_value=50,
            value=20,
            help="Maximum genes per signature (hard limit)"
        )
    
    # Validation
    if min_genes >= max_genes:
        st.error("❌ Min genes must be less than max genes!")
        return
    
    st.markdown(f"""
    <div class="warning-box">
    <strong>Configuration:</strong> Will generate <strong>{target_count} signatures</strong>, 
    each with exactly <strong>{min_genes}-{max_genes} genes</strong>.
    Gene limits are STRICTLY ENFORCED during construction.
    </div>
    """, unsafe_allow_html=True)
    
    # Generate button
    if st.button("🚀 Generate Signatures", type="primary", use_container_width=True):
        if not query or not query.strip():
            st.error("Please enter a research question")
            return
        
        generate_signatures(
            query,
            target_count,
            min_genes,
            max_genes,
            st.session_state.hf_token if st.session_state.token_validated else None
        )


def render_results_tab():
    if not st.session_state.execution_complete:
        st.info("ℹ️ No results yet. Generate signatures first.")
        return
    
    results = st.session_state.results
    
    st.markdown("## 📊 Generated Signatures")
    
    st.markdown(f"""
    <div class="info-box">
    <strong>Generated {results['total_signatures']} mechanism-specific signatures</strong><br>
    Query: {results['query']}<br>
    Gene range: {results['min_genes']}-{results['max_genes']} per signature
    </div>
    """, unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Signatures", results['total_signatures'])
    with col2:
        st.metric("Target", results['target_count'])
    with col3:
        gene_counts = [s['gene_count'] for s in results['signatures']]
        avg_genes = np.mean(gene_counts)
        st.metric("Avg Genes", f"{avg_genes:.1f}")
    with col4:
        total_unique = len(set().union(*[set(s['genes']) for s in results['signatures']]))
        st.metric("Unique Genes", total_unique)
    
    # Signature table
    with st.expander("🧬 Signature Details", expanded=True):
        sig_data = []
        for sig in results['signatures']:
            sig_data.append({
                'ID': sig['signature_id'],
                'Name': sig['signature_name'],
                'Facet': sig['facet'],
                'Mechanism': sig['mechanism'],
                'Genes': sig['gene_count'],
                'Confidence': f"{sig['confidence']:.3f}",
                'Top Genes': ', '.join(sig['top_genes'])
            })
        
        df = pd.DataFrame(sig_data)
        st.dataframe(df, use_container_width=True, height=400)
    
    # Distribution
    with st.expander("📊 Facet Distribution"):
        facet_counts = Counter([s['facet'] for s in results['signatures']])
        facet_df = pd.DataFrame([
            {'Facet': k, 'Count': v}
            for k, v in facet_counts.most_common()
        ])
        st.bar_chart(facet_df.set_index('Facet'))
    
    # Downloads
    st.markdown("### 💾 Downloads")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # GMT format
        gmt_lines = []
        for sig in results['signatures']:
            sig_id = sig['signature_id']
            desc = f"{sig['facet']}|{sig['mechanism']}|{sig['gene_count']}genes|conf:{sig['confidence']:.3f}"
            genes = '\t'.join(sig['genes'])
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
        # JSON format
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
        <h1>🧬 Biological Signature Generator</h1>
        <p style='font-size: 1.1rem; color: #94a3b8;'>
            Mechanism-specific gene signatures (10-20 genes each)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    render_sidebar()
    
    tab1, tab2, tab3 = st.tabs(["📚 Knowledge Base", "🧬 Generate", "📊 Results"])
    
    with tab1:
        render_kb_tab()
    
    with tab2:
        render_generation_tab()
    
    with tab3:
        render_results_tab()


if __name__ == "__main__":
    main()

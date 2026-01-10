"""
Biological Signature Generator (FIXED)
========================================

FIXES:
1. Hard limit: 10-35 genes per signature
2. Generates MULTIPLE signatures (not one giant one)
3. Proper signature count control
4. Better biological decomposition
"""

import streamlit as st
import pandas as pd
import json
import gzip
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
import time
import numpy as np
from dataclasses import dataclass
from collections import Counter

# ============================================================
# GENE SIGNATURE DATA CLASS
# ============================================================

@dataclass
class GeneSignature:
    """A gene signature with controlled size"""
    signature_id: str
    signature_name: str
    genes: List[str]
    facet: str
    confidence: float
    
    def to_dict(self):
        return {
            'signature_id': self.signature_id,
            'signature_name': self.signature_name,
            'genes': self.genes,
            'gene_count': len(self.genes),
            'facet': self.facet,
            'confidence': self.confidence
        }


# ============================================================
# SIGNATURE BUILDER (FIXED)
# ============================================================

class SignatureBuilder:
    """
    Builds gene signatures with STRICT gene count limits.
    
    KEY FIX: Each signature has 10-35 genes (not 3000+!)
    """
    
    def __init__(self, min_genes: int = 10, max_genes: int = 35):
        self.min_genes = min_genes
        self.max_genes = max_genes
    
    def build_signatures_from_pathways(
        self,
        facet_name: str,
        pathways_dict: Dict[str, List[str]],
        signatures_needed: int
    ) -> List[GeneSignature]:
        """
        Build multiple signatures from pathways.
        
        Args:
            facet_name: Name of facet (e.g., "T-cell Exhaustion")
            pathways_dict: Dict of pathway_id -> gene_list
            signatures_needed: How many signatures to generate
            
        Returns:
            List of GeneSignature objects
        """
        if not pathways_dict:
            return []
        
<<<<<<< HEAD
        signatures = []
        
        # Collect all genes with frequencies
        gene_freq = Counter()
        for genes in pathways_dict.values():
            gene_freq.update(genes)
        
        total_pathways = len(pathways_dict)
=======
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
>>>>>>> 629ba36d2ebb51a7d671729affe9c83b6e31c6dd
        
        # Strategy 1: High-frequency signature (genes in 60%+ pathways)
        high_freq_genes = [
            gene for gene, count in gene_freq.most_common()
            if count / total_pathways >= 0.6
        ][:self.max_genes]  # ⭐ CAP AT MAX GENES
        
        if len(high_freq_genes) >= self.min_genes:
            signatures.append(GeneSignature(
                signature_id=f"{self._make_id(facet_name)}_HIGH_FREQ",
                signature_name=f"{facet_name} (Core Markers)",
                genes=high_freq_genes,
                facet=facet_name,
                confidence=0.90
            ))
        
        # Strategy 2: Medium-frequency signature (genes in 30-60% pathways)
        med_freq_genes = [
            gene for gene, count in gene_freq.most_common()
            if 0.3 <= count / total_pathways < 0.6
        ][:self.max_genes]
        
        if len(med_freq_genes) >= self.min_genes:
            signatures.append(GeneSignature(
                signature_id=f"{self._make_id(facet_name)}_MED_FREQ",
                signature_name=f"{facet_name} (Extended)",
                genes=med_freq_genes,
                facet=facet_name,
                confidence=0.75
            ))
        
        # Strategy 3: Top genes overall (most frequent)
        top_genes = [gene for gene, _ in gene_freq.most_common(self.max_genes)]
        
        if len(top_genes) >= self.min_genes:
            signatures.append(GeneSignature(
                signature_id=f"{self._make_id(facet_name)}_TOP",
                signature_name=f"{facet_name} (Top Genes)",
                genes=top_genes,
                facet=facet_name,
                confidence=0.85
            ))
        
        # Strategy 4-N: Create more signatures if needed
        # Sample different subsets of pathways
        pathway_ids = list(pathways_dict.keys())
        
        if len(pathway_ids) > 10 and len(signatures) < signatures_needed:
            # Create signatures from pathway subsets
            subsets = [
                pathway_ids[:len(pathway_ids)//2],  # First half
                pathway_ids[len(pathway_ids)//2:],  # Second half
            ]
            
            for i, subset_ids in enumerate(subsets):
                subset_genes = Counter()
                for pid in subset_ids:
                    subset_genes.update(pathways_dict[pid])
                
                top_subset = [g for g, _ in subset_genes.most_common(self.max_genes)]
                
                if len(top_subset) >= self.min_genes:
                    signatures.append(GeneSignature(
                        signature_id=f"{self._make_id(facet_name)}_SUBSET_{i+1}",
                        signature_name=f"{facet_name} (Subset {i+1})",
                        genes=top_subset,
                        facet=facet_name,
                        confidence=0.70
                    ))
                
                if len(signatures) >= signatures_needed:
                    break
        
        return signatures[:signatures_needed]  # Enforce limit
    
    def _make_id(self, name: str) -> str:
        """Convert facet name to ID"""
        return name.upper().replace(' ', '_').replace('-', '_')


# ============================================================
# SEMANTIC SEARCH (Simplified)
# ============================================================

@st.cache_resource
def load_embedding_model():
    """Load sentence transformer"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except:
        return None


def semantic_search_pathways(
    query: str,
    pathways_dict: Dict[str, List[str]],
    model,
    top_k: int = 50
) -> Dict[str, List[str]]:
    """
    Find relevant pathways for query.
    
    Returns:
        Dict of pathway_id -> genes (top K most relevant)
    """
    if model is None:
        # Fallback: keyword matching
        query_terms = set(query.lower().split())
        matches = {}
        
        for pid, genes in pathways_dict.items():
            pid_terms = set(pid.lower().replace('_', ' ').split())
            if query_terms & pid_terms:
                matches[pid] = genes
                if len(matches) >= top_k:
                    break
        
        return matches
    
<<<<<<< HEAD
    # Compute embeddings
    query_emb = model.encode(query, convert_to_numpy=True)
    
    pathway_scores = []
    for pid, genes in pathways_dict.items():
        # Simple embedding: pathway name + sample genes
        text = f"{pid.replace('_', ' ')} {' '.join(genes[:10])}"
        pathway_emb = model.encode(text, convert_to_numpy=True)
        
        # Cosine similarity
        similarity = np.dot(query_emb, pathway_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(pathway_emb)
        )
        
        pathway_scores.append((pid, similarity))
    
    # Sort by similarity
    pathway_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top K
    top_pathways = {}
    for pid, score in pathway_scores[:top_k]:
        top_pathways[pid] = pathways_dict[pid]
    
    return top_pathways
=======
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
>>>>>>> 629ba36d2ebb51a7d671729affe9c83b6e31c6dd


# ============================================================
# BIOLOGICAL DECOMPOSITION (FIXED)
# ============================================================

def decompose_query_biological(query: str, target_count: int) -> List[Dict[str, str]]:
    """
    Decompose query into biological facets.
    
    KEY FIX: Generate ENOUGH facets to reach target signature count
    """
    query_lower = query.lower()
    
    # Detect query type and create specific facets
    if 't cell' in query_lower and ('exhaust' in query_lower or 'dysfunction' in query_lower):
        facets = [
            {'facet_id': 'F1', 'facet_name': 'T-cell Exhaustion Markers', 
             'query': 'PDCD1 LAG3 HAVCR2 CTLA4 TIGIT exhaustion'},
            {'facet_id': 'F2', 'facet_name': 'Glycolysis Metabolic Shift',
             'query': 'LDHA HK2 PKM2 glycolysis glucose'},
            {'facet_id': 'F3', 'facet_name': 'Mitochondrial Dysfunction',
             'query': 'mitochondrial respiration oxidative'},
            {'facet_id': 'F4', 'facet_name': 'Transcription Factors',
             'query': 'TOX NFATC1 PRDM1 EOMES transcription'},
            {'facet_id': 'F5', 'facet_name': 'Cytokine Signaling',
             'query': 'IL10 TGFB1 IFNG TNF cytokine'},
            {'facet_id': 'F6', 'facet_name': 'Checkpoint Pathways',
             'query': 'CD274 PDCD1LG2 checkpoint PD-L1'},
            {'facet_id': 'F7', 'facet_name': 'T-cell Activation',
             'query': 'CD28 ICOS CD3 activation'},
            {'facet_id': 'F8', 'facet_name': 'Effector Function',
             'query': 'GZMB PRF1 IFNG effector cytotoxic'},
        ]
    
    elif 'macrophage' in query_lower or 'tam' in query_lower:
        facets = [
            {'facet_id': 'F1', 'facet_name': 'M1 Polarization',
             'query': 'M1 macrophage NOS2 TNF IL12 inflammatory'},
            {'facet_id': 'F2', 'facet_name': 'M2 Polarization',
             'query': 'M2 macrophage ARG1 IL10 CD163'},
            {'facet_id': 'F3', 'facet_name': 'Phagocytosis',
             'query': 'phagocytosis MARCO MERTK CD36'},
            {'facet_id': 'F4', 'facet_name': 'Cytokine Production',
             'query': 'cytokine IL6 IL1B TNF macrophage'},
            {'facet_id': 'F5', 'facet_name': 'Antigen Presentation',
             'query': 'HLA CD80 CD86 antigen presentation'},
            {'facet_id': 'F6', 'facet_name': 'Tissue Remodeling',
             'query': 'MMP9 MMP2 remodeling fibrosis'},
            {'facet_id': 'F7', 'facet_name': 'Metabolic Programming',
             'query': 'glycolysis oxidative phosphorylation macrophage'},
        ]
    
    elif 'cancer' in query_lower and 'metab' in query_lower:
        facets = [
            {'facet_id': 'F1', 'facet_name': 'Warburg Effect',
             'query': 'LDHA PKM2 glycolysis lactate Warburg'},
            {'facet_id': 'F2', 'facet_name': 'Glutamine Metabolism',
             'query': 'GLS glutamine glutaminolysis'},
            {'facet_id': 'F3', 'facet_name': 'Lipid Metabolism',
             'query': 'FASN ACLY lipid fatty acid'},
            {'facet_id': 'F4', 'facet_name': 'One-Carbon Metabolism',
             'query': 'MTHFD2 SHMT serine glycine'},
            {'facet_id': 'F5', 'facet_name': 'TCA Cycle',
             'query': 'IDH1 IDH2 succinate TCA'},
            {'facet_id': 'F6', 'facet_name': 'Pentose Phosphate',
             'query': 'G6PD pentose phosphate NADPH'},
        ]
    
    else:
        # Generic biological categories
        num_facets = max(8, (target_count + 2) // 3)  # Ensure enough facets
        facets = [
            {'facet_id': f'F{i+1}', 'facet_name': f'Aspect {i+1}', 'query': query}
            for i in range(num_facets)
        ]
    
    return facets


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
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    defaults = {
        'kb_path': None,
        'kb_loaded': False,
        'results': None,
        'execution_complete': False,
<<<<<<< HEAD
=======
        'generation_mode': 'semantic',  # 'semantic', 'hybrid', or 'neighbor'
        'expansion_level': 'core'  # 'core', 'balanced', 'broad'
>>>>>>> 629ba36d2ebb51a7d671729affe9c83b6e31c6dd
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
    
    st.error("Knowledge base not found")
    return None


# ============================================================
# MAIN PIPELINE
# ============================================================

def generate_signatures(query: str, target_count: int, min_genes: int, max_genes: int):
    """
    Main signature generation pipeline.
    
    FIXED: Generates target_count signatures, each with min_genes to max_genes
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
        
        status.info(f"   Loaded {len(pathways_dict)} pathways")
        
        # Decompose query
        status.info("🧠 Decomposing query...")
        progress_bar.progress(20)
        
        facets = decompose_query_biological(query, target_count)
        status.info(f"   Created {len(facets)} biological facets")
        
        # Load embedding model
        status.info("🔍 Loading search model...")
        embedding_model = load_embedding_model()
        progress_bar.progress(30)
        
        # Build signatures
        status.info("🧬 Building gene signatures...")
        
        builder = SignatureBuilder(min_genes=min_genes, max_genes=max_genes)
        all_signatures = []
        
        # Calculate signatures per facet
        signatures_per_facet = max(1, target_count // len(facets))
        
        for i, facet in enumerate(facets):
            facet_name = facet['facet_name']
            facet_query = facet['query']
            
            status.info(f"   Building signatures for: {facet_name}")
            
            # Find relevant pathways
            relevant_pathways = semantic_search_pathways(
                facet_query,
                pathways_dict,
                embedding_model,
                top_k=50
            )
            
            if not relevant_pathways:
                continue
            
            # Build signatures for this facet
            facet_signatures = builder.build_signatures_from_pathways(
                facet_name,
                relevant_pathways,
                signatures_needed=signatures_per_facet
            )
            
            all_signatures.extend(facet_signatures)
            
            # Update progress
            progress = 30 + int((i + 1) / len(facets) * 60)
            progress_bar.progress(progress)
            
            # Stop if we have enough
            if len(all_signatures) >= target_count:
                break
        
        # Ensure we have exactly target_count
        if len(all_signatures) > target_count:
            all_signatures = all_signatures[:target_count]
        
        status.success(f"✅ Generated {len(all_signatures)} signatures!")
        progress_bar.progress(100)
        
        # Store results
        st.session_state.results = {
            'query': query,
            'target_count': target_count,
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

def render_kb_tab():
    st.markdown("## 📚 Knowledge Base")
    
    st.markdown("""
    <div class="info-box">
    The knowledge base contains pathways used as source material for building signatures.
    </div>
    """, unsafe_allow_html=True)
    
    pathways = load_knowledge_base()
    
    if pathways:
        st.success(f"✅ Loaded {len(pathways):,} pathways")
        
        # Show sample
        sample_pid = list(pathways.keys())[0]
        sample_genes = pathways[sample_pid]
        
        st.caption(f"Sample: {sample_pid}")
        st.caption(f"Genes: {', '.join(sample_genes[:10])}... ({len(sample_genes)} total)")


def render_generation_tab():
    st.markdown("## 🧬 Generate Signatures")
    
    if not st.session_state.kb_loaded:
        st.warning("⚠️ Please load knowledge base first")
        return
    
    st.markdown("""
    <div class="info-box">
    <strong>How this works:</strong><br><br>
    1. Your query is decomposed into biological facets<br>
    2. For each facet, custom gene signatures are built<br>
    3. Each signature has 10-35 genes (controllable)<br>
    4. You get exactly the number of signatures you request
    </div>
    """, unsafe_allow_html=True)
    
    # Query input
    query = st.text_area(
        "Research Question",
        height=80,
        placeholder="Example: T cell exhaustion in cancer\nExample: Macrophage polarization\nExample: Metabolic reprogramming",
    )
    
<<<<<<< HEAD
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_count = st.number_input(
            "Number of Signatures",
            min_value=5,
            max_value=100,
            value=35,
            step=5,
            help="Total signatures to generate"
        )
    
    with col2:
        min_genes = st.number_input(
            "Min Genes/Signature",
            min_value=5,
            max_value=30,
            value=10,
            help="Minimum genes per signature"
        )
    
    with col3:
        max_genes = st.number_input(
            "Max Genes/Signature",
            min_value=10,
            max_value=100,
            value=35,
            help="Maximum genes per signature"
        )
    
    st.info(f"Will generate **{target_count} signatures**, each with **{min_genes}-{max_genes} genes**")
    
=======
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

>>>>>>> 629ba36d2ebb51a7d671729affe9c83b6e31c6dd
    # Generate button
    if st.button("🚀 Generate Signatures", type="primary", use_container_width=True):
        if not query:
            st.error("Please enter a research question")
            return
        
        generate_signatures(query, target_count, min_genes, max_genes)


<<<<<<< HEAD
=======
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

>>>>>>> 629ba36d2ebb51a7d671729affe9c83b6e31c6dd
def render_results_tab():
    if not st.session_state.execution_complete:
        st.info("ℹ️ No results yet. Generate signatures first.")
        return
    
    results = st.session_state.results
    
    st.markdown("## 📊 Generated Signatures")
    
    st.markdown(f"""
    <div class="info-box">
    Generated <strong>{results['total_signatures']} custom gene signatures</strong><br>
    Each signature contains specific genes for a biological facet
    </div>
    """, unsafe_allow_html=True)
    
    # Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Signatures", results['total_signatures'])
    with col2:
        st.metric("Target", results['target_count'])
    with col3:
        avg_genes = np.mean([s['gene_count'] for s in results['signatures']])
        st.metric("Avg Genes/Sig", f"{avg_genes:.1f}")
    with col4:
        total_unique = len(set().union(*[set(s['genes']) for s in results['signatures']]))
        st.metric("Unique Genes", total_unique)
    
    # Table
    with st.expander("🧬 Signature Details", expanded=True):
        sig_data = []
        for sig in results['signatures']:
            sig_data.append({
                'ID': sig['signature_id'],
                'Name': sig['signature_name'],
                'Facet': sig['facet'],
                'Genes': sig['gene_count'],
                'Confidence': f"{sig['confidence']:.2f}",
                'Sample': ', '.join(sig['genes'][:5]) + '...'
            })
        
        df = pd.DataFrame(sig_data)
        st.dataframe(df, use_container_width=True, height=400)
    
    # Downloads
    st.markdown("### 💾 Downloads")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # GMT format
        gmt_lines = []
        for sig in results['signatures']:
            sig_id = sig['signature_id']
            desc = f"{sig['facet']}|{sig['gene_count']}genes|conf:{sig['confidence']:.2f}"
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
        <h1>🧬 Gene Signature Generator</h1>
        <p style='font-size: 1.1rem; color: #94a3b8;'>
            Generate custom gene signatures (10-35 genes each)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📚 Knowledge Base", "🧬 Generate", "📊 Results"])
    
    with tab1:
        render_kb_tab()
    
    with tab2:
        render_generation_tab()
    
    with tab3:
        render_results_tab()


if __name__ == "__main__":
    main()

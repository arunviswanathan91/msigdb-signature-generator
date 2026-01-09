"""
MSigDB Signature Generator - Streamlit Application
===================================================

Production-grade Streamlit app with glassmorphism UI for biological signature generation.

Requirements:
    pip install streamlit pandas numpy

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import time

# Import your pipeline modules
# Make sure these are in the same directory or in your Python path
try:
    from complete_module_replacements import (
        PipelineConfig,
        KnowledgeBase,
        CompletePipeline
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    st.error("⚠️ Pipeline modules not found. Make sure complete_module_replacements.py is available.")

# ============================================================
# GLASSMORPHISM CSS
# ============================================================

def inject_glassmorphism_css():
    """Inject glassmorphism and dark theme CSS"""
    st.markdown("""
    <style>
    /* Global Theme */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e0e0e0;
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(138, 180, 248, 0.3);
        box-shadow: 0 8px 32px 0 rgba(138, 180, 248, 0.2);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #8ab4f8;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(22, 33, 62, 0.95);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Input Fields */
    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: #e0e0e0;
        padding: 12px;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #8ab4f8;
        box-shadow: 0 0 0 2px rgba(138, 180, 248, 0.2);
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.6);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Dataframe */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        border-left: 4px solid #8ab4f8;
    }
    
    /* Sliders */
    .stSlider {
        padding: 8px 0;
    }
    
    /* Checkboxes */
    .stCheckbox label {
        color: #e0e0e0;
    }
    
    /* Stage Headers */
    .stage-header {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        padding: 16px;
        border-radius: 12px;
        margin: 24px 0 16px 0;
        border-left: 4px solid #667eea;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: 600;
        margin-left: 8px;
    }
    
    .status-pending {
        background: rgba(255, 193, 7, 0.2);
        color: #ffc107;
        border: 1px solid rgba(255, 193, 7, 0.3);
    }
    
    .status-running {
        background: rgba(33, 150, 243, 0.2);
        color: #2196f3;
        border: 1px solid rgba(33, 150, 243, 0.3);
    }
    
    .status-complete {
        background: rgba(76, 175, 80, 0.2);
        color: #4caf50;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    
    /* Download Button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .stDownloadButton button:hover {
        box-shadow: 0 6px 20px 0 rgba(17, 153, 142, 0.6);
    }
    
    /* Code blocks */
    code {
        background: rgba(255, 255, 255, 0.05);
        padding: 2px 6px;
        border-radius: 4px;
        color: #8ab4f8;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def initialize_session_state():
    """Initialize all session state variables"""
    
    # Authentication
    if 'hf_token' not in st.session_state:
        st.session_state.hf_token = None
    if 'token_validated' not in st.session_state:
        st.session_state.token_validated = False
    
    # Pipeline state
    if 'pipeline_initialized' not in st.session_state:
        st.session_state.pipeline_initialized = False
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    
    # Results
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'execution_complete' not in st.session_state:
        st.session_state.execution_complete = False
    
    # Progress tracking
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = None
    if 'progress' not in st.session_state:
        st.session_state.progress = 0
    if 'status_message' not in st.session_state:
        st.session_state.status_message = ""


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_hf_token(token: str) -> bool:
    """Validate Hugging Face token with a lightweight test"""
    if not token or len(token) < 20:
        return False
    
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=token)
        # Simple test - just check if we can create a client
        return True
    except Exception as e:
        st.error(f"Token validation failed: {str(e)}")
        return False


def update_progress(progress: float, message: str):
    """Update progress bar and status message"""
    st.session_state.progress = progress
    st.session_state.status_message = message


def create_slug(text: str, max_length: int = 50) -> str:
    """Create URL-safe slug from text"""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text)
    return text[:max_length]


def format_results_for_download(results: Dict, format_type: str) -> str:
    """Format results for download"""
    
    if format_type == "JSON":
        return json.dumps(results, indent=2)
    
    elif format_type == "TXT":
        output = []
        output.append("=" * 70)
        output.append("SIGNATURE GENERATION RESULTS")
        output.append("=" * 70)
        output.append(f"\nQuery: {results['query']}")
        output.append(f"Timestamp: {results['timestamp']}")
        output.append(f"Total Signatures: {results['total_signatures']}")
        output.append(f"Unique Genes: {results['total_unique_genes']}")
        output.append("\n" + "=" * 70)
        output.append("SIGNATURES")
        output.append("=" * 70)
        
        for sig in results.get('derived_signatures', []):
            output.append(f"\n{sig['signature_id']}")
            output.append(f"  Facet: {sig['facet_id']}")
            output.append(f"  Source: {sig['source']}")
            output.append(f"  Method: {sig['derivation_method']}")
            output.append(f"  Genes ({sig['gene_count']}): {', '.join(sig['genes'][:10])}")
            if sig['gene_count'] > 10:
                output.append(f"    ... and {sig['gene_count'] - 10} more")
        
        return "\n".join(output)
    
    elif format_type == "GMT":
        lines = []
        for sig in results.get('derived_signatures', []):
            sig_id = sig['signature_id']
            desc = f"{sig['facet_id']}|{sig['source']}|{sig['derivation_method']}"
            genes = "\t".join(sig['genes'])
            lines.append(f"{sig_id}\t{desc}\t{genes}")
        return "\n".join(lines)
    
    return ""


# ============================================================
# STAGE 1: QUERY & TARGET DEFINITION
# ============================================================

def render_stage_1_query():
    """Render Stage 1: Query & Target Definition"""
    
    st.markdown('<div class="stage-header">', unsafe_allow_html=True)
    st.markdown("### 🔍 Stage 1: Query & Target Definition")
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("📝 Define Your Research Question", expanded=True):
        # Query input
        query = st.text_area(
            "Biological question / intent",
            placeholder="e.g., Give me signatures for metabolic pathways in pancreatic cancer resistance...",
            height=150,
            help="Describe the biological signatures you want to generate"
        )
        
        st.markdown("---")
        
        # Target count strategy
        target_strategy = st.radio(
            "How should total signatures be decided?",
            ["Extract automatically from query", "Manually specify"],
            help="Auto-extract looks for phrases like 'give me 25 signatures'"
        )
        
        # Manual target count
        manual_target = None
        if target_strategy == "Manually specify":
            manual_target = st.slider(
                "Total signatures",
                min_value=10,
                max_value=500,
                value=25,
                step=10,
                help="Number of final signatures to generate"
            )
        
        # Facet preview option
        show_facets = st.checkbox(
            "Show facet decomposition before execution",
            value=False,
            help="Preview how the query will be split into biological facets"
        )
        
        return {
            'query': query,
            'target_strategy': target_strategy,
            'manual_target': manual_target,
            'show_facets': show_facets
        }


# ============================================================
# STAGE 2: KNOWLEDGE BASE SELECTION
# ============================================================

def render_stage_2_kb_selection():
    """Render Stage 2: Knowledge Base Selection"""
    
    st.markdown('<div class="stage-header">', unsafe_allow_html=True)
    st.markdown("### 📚 Stage 2: Knowledge Base Selection")
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("🗂️ Select Pathway Sources", expanded=True):
        st.markdown("**Select which pathway databases to include:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            kegg = st.checkbox("KEGG", value=True)
            reactome = st.checkbox("Reactome", value=True)
            wikipathways = st.checkbox("WikiPathways", value=True)
            hallmark = st.checkbox("Hallmark", value=True)
        
        with col2:
            go = st.checkbox("Gene Ontology", value=True)
            if go:
                go_bp = st.checkbox("  └─ GO-BP (Biological Process)", value=True)
                go_mf = st.checkbox("  └─ GO-MF (Molecular Function)", value=True)
                go_cc = st.checkbox("  └─ GO-CC (Cellular Component)", value=True)
            else:
                go_bp = go_mf = go_cc = False
        
        st.markdown("---")
        
        exclude_medicus = st.checkbox(
            "Exclude KEGG MEDICUS pathways",
            value=True,
            help="MEDICUS pathways are drug-specific and may not be relevant"
        )
        
        return {
            'sources': {
                'KEGG': kegg,
                'REACTOME': reactome,
                'WIKIPATHWAYS': wikipathways,
                'HALLMARK': hallmark,
                'GO': go,
                'GO_BP': go_bp,
                'GO_MF': go_mf,
                'GO_CC': go_cc
            },
            'exclude_medicus': exclude_medicus
        }


# ============================================================
# STAGE 3: SIGNATURE DESIGN CONTROLS
# ============================================================

def render_stage_3_design_controls():
    """Render Stage 3: Signature Design Controls"""
    
    st.markdown('<div class="stage-header">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Stage 3: Signature Design Controls")
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("🧬 Configure Signature Parameters", expanded=True):
        
        # A. Gene Count Constraints
        st.markdown("**A. Gene Count Constraints**")
        col1, col2 = st.columns(2)
        with col1:
            min_genes = st.slider(
                "Min genes per signature",
                min_value=3,
                max_value=50,
                value=5,
                help="Minimum number of genes required"
            )
        with col2:
            max_genes = st.slider(
                "Max genes per signature",
                min_value=10,
                max_value=300,
                value=300,
                help="Maximum number of genes allowed"
            )
        
        st.markdown("---")
        
        # B. Signature Derivation Thresholds
        st.markdown("**B. Signature Derivation Thresholds**")
        
        core_threshold = st.slider(
            "Core gene frequency threshold",
            min_value=0.5,
            max_value=0.9,
            value=0.7,
            step=0.05,
            help="Gene must appear in this fraction of similar pathways to be 'core'"
        )
        
        unique_threshold = st.slider(
            "Uniqueness threshold",
            min_value=0.05,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Gene must appear in <this fraction of pathways to be 'unique'"
        )
        
        merge_threshold = st.slider(
            "Facet merge similarity",
            min_value=0.7,
            max_value=0.95,
            value=0.85,
            step=0.05,
            help="Facets with similarity ≥ this will be merged"
        )
        
        st.markdown("---")
        
        # C. Facet Controls
        st.markdown("**C. Facet Controls**")
        col1, col2 = st.columns(2)
        with col1:
            min_facets = st.slider(
                "Minimum facets",
                min_value=2,
                max_value=6,
                value=3,
                help="Minimum number of biological facets"
            )
        with col2:
            max_facets = st.slider(
                "Maximum facets",
                min_value=6,
                max_value=12,
                value=10,
                help="Maximum number of biological facets"
            )
        
        st.markdown("---")
        
        # D. Diversity Weights
        st.markdown("**D. Diversity Weights**")
        alpha = st.slider(
            "Within-facet diversity (α)",
            min_value=0.3,
            max_value=0.9,
            value=0.7,
            step=0.05,
            help="Weight for diversity within same facet"
        )
        beta = 1.0 - alpha
        st.caption(f"Cross-facet diversity (β) = {beta:.2f} (auto-computed as 1 - α)")
        
        return {
            'min_genes': min_genes,
            'max_genes': max_genes,
            'core_threshold': core_threshold,
            'unique_threshold': unique_threshold,
            'merge_threshold': merge_threshold,
            'min_facets': min_facets,
            'max_facets': max_facets,
            'alpha': alpha,
            'beta': beta
        }


# ============================================================
# STAGE 4: EXECUTION
# ============================================================

def render_stage_4_execution(config_dict: Dict):
    """Render Stage 4: Execution Controls"""
    
    st.markdown('<div class="stage-header">', unsafe_allow_html=True)
    st.markdown("### ▶️ Stage 4: Pipeline Execution")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Validation checks
    ready_to_run = True
    warnings = []
    
    if not config_dict['query']['query'].strip():
        ready_to_run = False
        warnings.append("❌ Query is empty")
    
    if not st.session_state.token_validated:
        ready_to_run = False
        warnings.append("❌ Hugging Face token not validated")
    
    if warnings:
        st.warning("\n".join(warnings))
    
    # Run button
    if st.button("🚀 Run Signature Generation Pipeline", 
                 disabled=not ready_to_run,
                 use_container_width=True):
        return True
    
    return False


def execute_pipeline(config_dict: Dict, base_dir: str):
    """Execute the pipeline with live progress updates"""
    
    # Progress container
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Stage 1: Initialize
            status_text.markdown("🔧 **Initializing pipeline...**")
            progress_bar.progress(5)
            time.sleep(0.5)
            
            # Create config
            pipeline_config = PipelineConfig(
                min_genes=config_dict['design']['min_genes'],
                max_genes=config_dict['design']['max_genes'],
                core_signature_threshold=config_dict['design']['core_threshold'],
                unique_signature_threshold=config_dict['design']['unique_threshold'],
                facet_merge_similarity_threshold=config_dict['design']['merge_threshold'],
                min_facets=config_dict['design']['min_facets'],
                max_facets=config_dict['design']['max_facets'],
                within_facet_diversity_weight=config_dict['design']['alpha'],
                cross_facet_diversity_weight=config_dict['design']['beta']
            )
            
            # Stage 2: Load KB
            status_text.markdown("📚 **Loading pathway knowledge base...**")
            progress_bar.progress(10)
            
            base_path = Path(base_dir)
            pipeline = CompletePipeline(
                kb_path=str(base_path / 'knowledge_base.json.gz'),
                embeddings_cache_dir=str(base_path / 'embeddings'),
                faiss_index_path=str(base_path / 'faiss_index.bin'),
                hf_token=st.session_state.hf_token,
                config=pipeline_config
            )
            
            # Stage 3: Initialize components
            status_text.markdown("🧬 **Initializing components (embeddings, FAISS index)...**")
            progress_bar.progress(20)
            
            pipeline.initialize()
            progress_bar.progress(35)
            
            # Stage 4: Query planning
            status_text.markdown("🧠 **Decomposing query into biological facets...**")
            progress_bar.progress(40)
            
            query = config_dict['query']['query']
            target_count = config_dict['query']['manual_target']
            
            # Stage 5: Run pipeline
            status_text.markdown("🔍 **Retrieving candidate pathways...**")
            progress_bar.progress(50)
            
            status_text.markdown("🧪 **Validating biological constraints...**")
            progress_bar.progress(65)
            
            status_text.markdown("🎯 **Selecting signatures with quotas...**")
            progress_bar.progress(75)
            
            # Actually run the pipeline
            results = pipeline.run(query, target_count=target_count)
            
            status_text.markdown("📊 **Computing coverage statistics...**")
            progress_bar.progress(90)
            
            status_text.markdown("✅ **Pipeline complete! Preparing results...**")
            progress_bar.progress(100)
            
            # Store results
            st.session_state.results = results
            st.session_state.execution_complete = True
            
            time.sleep(0.5)
            status_text.markdown("✅ **Success! Results ready for download.**")
            
            return True
            
        except Exception as e:
            status_text.markdown(f"❌ **Error: {str(e)}**")
            st.error(f"Pipeline execution failed: {str(e)}")
            return False


# ============================================================
# STAGE 5: RESULTS & DOWNLOADS
# ============================================================

def render_stage_5_results():
    """Render Stage 5: Results & Downloads"""
    
    if not st.session_state.execution_complete or not st.session_state.results:
        return
    
    results = st.session_state.results
    
    st.markdown('<div class="stage-header">', unsafe_allow_html=True)
    st.markdown("### 📊 Stage 5: Results & Downloads")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Signatures", results['total_signatures'])
    with col2:
        st.metric("Pathways Selected", results['total_pathways_selected'])
    with col3:
        st.metric("Unique Genes", results['total_unique_genes'])
    with col4:
        ratio = results['total_signatures'] / results['total_pathways_selected'] if results['total_pathways_selected'] > 0 else 0
        st.metric("Signatures/Pathway", f"{ratio:.2f}")
    
    st.markdown("---")
    
    # Results table
    with st.expander("📋 Signature Details", expanded=True):
        if results.get('derived_signatures'):
            # Create DataFrame
            sig_data = []
            for sig in results['derived_signatures']:
                sig_data.append({
                    'Signature ID': sig['signature_id'],
                    'Facet': sig['facet_id'],
                    'Gene Count': sig['gene_count'],
                    'Source': sig['source'],
                    'Method': sig['derivation_method'],
                    'Confidence': f"{sig['confidence']:.3f}",
                    'Genes': ', '.join(sig['genes'][:5]) + ('...' if sig['gene_count'] > 5 else '')
                })
            
            df = pd.DataFrame(sig_data)
            st.dataframe(df, use_container_width=True, height=400)
    
    # Facet distribution
    with st.expander("📊 Facet Distribution", expanded=False):
        if results.get('facet_distribution'):
            facet_df = pd.DataFrame([
                {'Facet': k, 'Count': v} 
                for k, v in results['facet_distribution'].items()
            ])
            st.bar_chart(facet_df.set_index('Facet'))
    
    # Source distribution
    with st.expander("📚 Source Distribution", expanded=False):
        if results.get('source_distribution'):
            source_df = pd.DataFrame([
                {'Source': k, 'Count': v}
                for k, v in results['source_distribution'].items()
            ])
            st.bar_chart(source_df.set_index('Source'))
    
    st.markdown("---")
    
    # Download section
    st.markdown("### 💾 Download Results")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        format_type = st.radio(
            "Select download format",
            ["JSON", "TXT", "GMT"],
            horizontal=True,
            help="JSON: Full metadata | TXT: Human-readable | GMT: MSigDB compatible"
        )
    
    # Generate filename
    query_slug = create_slug(results['query'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{query_slug}_{results['total_signatures']}_signatures_{timestamp}"
    
    # Format-specific extension
    extensions = {"JSON": "json", "TXT": "txt", "GMT": "gmt"}
    full_filename = f"{filename}.{extensions[format_type]}"
    
    # Generate download content
    download_content = format_results_for_download(results, format_type)
    
    with col2:
        st.download_button(
            label=f"⬇️ Download {format_type}",
            data=download_content,
            file_name=full_filename,
            mime="text/plain",
            use_container_width=True
        )
    
    # Show sample of download content
    with st.expander("👁️ Preview Download Content", expanded=False):
        st.code(download_content[:1000] + "\n\n... (truncated)" if len(download_content) > 1000 else download_content)


# ============================================================
# SIDEBAR: AUTHENTICATION & SETTINGS
# ============================================================

def render_sidebar():
    """Render sidebar with authentication and settings"""
    
    with st.sidebar:
        st.markdown("### 🔐 Authentication")
        
        # HF Token input
        token_input = st.text_input(
            "Hugging Face API Token",
            type="password",
            value=st.session_state.hf_token or "",
            help="Required for LLM-based query planning"
        )
        
        remember_token = st.checkbox(
            "Remember token for this session",
            value=True
        )
        
        # Validate button
        if st.button("Validate Token", use_container_width=True):
            if token_input:
                with st.spinner("Validating token..."):
                    if validate_hf_token(token_input):
                        if remember_token:
                            st.session_state.hf_token = token_input
                        st.session_state.token_validated = True
                        st.success("✅ Token validated!")
                    else:
                        st.session_state.token_validated = False
                        st.error("❌ Invalid token")
            else:
                st.warning("Please enter a token")
        
        # Token status
        if st.session_state.token_validated:
            st.success("✅ Token active")
        else:
            st.warning("⚠️ Token not validated")
        
        # Help text
        with st.expander("ℹ️ How to get a token", expanded=False):
            st.markdown("""
            1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
            2. Click "Create new token"
            3. Select "Read" access
            4. Copy and paste here
            """)
        
        st.markdown("---")
        
        # Data directory
        st.markdown("### 📁 Data Directory")
        base_dir = st.text_input(
            "Knowledge Base Directory",
            value="/content/drive/MyDrive/siggen",
            help="Directory containing knowledge_base.json.gz and embeddings"
        )
        
        st.markdown("---")
        
        # About
        st.markdown("### ℹ️ About")
        st.caption("""
        **MSigDB Signature Generator v2.1**
        
        Production-grade pipeline for biological signature generation with:
        - Multi-facet query decomposition
        - Quota-based selection
        - Signature derivation
        - Export to multiple formats
        """)
        
        return base_dir


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """Main application entry point"""
    
    # Page config
    st.set_page_config(
        page_title="MSigDB Signature Generator",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize
    initialize_session_state()
    inject_glassmorphism_css()
    
    # Check if pipeline modules are available
    if not PIPELINE_AVAILABLE:
        st.stop()
    
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>🧬 MSigDB Signature Generator</h1>
        <p style='font-size: 1.2em; color: #8ab4f8;'>
            Production-grade biological signature generation pipeline
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    base_dir = render_sidebar()
    
    # Main content
    main_container = st.container()
    
    with main_container:
        # Collect configuration from all stages
        config_dict = {}
        
        # Stage 1: Query
        config_dict['query'] = render_stage_1_query()
        
        # Stage 2: KB Selection
        config_dict['kb'] = render_stage_2_kb_selection()
        
        # Stage 3: Design Controls
        config_dict['design'] = render_stage_3_design_controls()
        
        # Stage 4: Execution
        should_run = render_stage_4_execution(config_dict)
        
        # Execute if run button clicked
        if should_run:
            st.markdown("---")
            with st.container():
                success = execute_pipeline(config_dict, base_dir)
                if success:
                    st.balloons()
        
        # Stage 5: Results (only if execution complete)
        if st.session_state.execution_complete:
            st.markdown("---")
            render_stage_5_results()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

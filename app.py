"""
MSigDB Signature Generator - Production Application
===================================================

Modern Streamlit application with dual KB management options.

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
from typing import Dict, List, Any, Optional
import time

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
# MODERN SURFACE UI - TAILWIND-INSPIRED CSS
# ============================================================

def inject_modern_css():
    """Inject modern, clean UI styling"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Reset */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Custom Containers */
    .surface-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 28px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .surface-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
        transform: translateY(-2px);
    }
    
    /* Headers */
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
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #e2e8f0;
    }
    
    /* Input Fields */
    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input,
    .stSelectbox select {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1.5px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        padding: 12px 16px !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease;
    }
    
    .stTextInput input:focus,
    .stTextArea textarea:focus,
    .stNumberInput input:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.15) !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.01em !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
    }
    
    .stButton button:active {
        transform: translateY(0) !important;
    }
    
    /* Download Button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4) !important;
    }
    
    .stDownloadButton button:hover {
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.6) !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        border-radius: 8px;
        height: 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.02);
        padding: 6px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.04);
        color: #cbd5e1;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #e2e8f0 !important;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(129, 140, 248, 0.3);
    }
    
    /* Dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Alert Boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        border-radius: 10px;
        border-left: 4px solid #6366f1;
        padding: 16px;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 24px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(129, 140, 248, 0.4);
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* Radio Buttons */
    .stRadio > label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }
    
    .stRadio [role="radiogroup"] {
        gap: 12px;
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: #cbd5e1 !important;
        font-weight: 400 !important;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-left: 10px;
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-warning {
        background: rgba(251, 191, 36, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .status-info {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* Section Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.1) 50%, transparent 100%);
        margin: 32px 0;
    }
    
    /* Code Blocks */
    code {
        background: rgba(255, 255, 255, 0.05);
        padding: 3px 8px;
        border-radius: 6px;
        color: #818cf8;
        font-size: 0.9em;
        font-family: 'Monaco', 'Menlo', monospace;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.15);
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
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
        'kb_mode': 'builtin',  # 'builtin' or 'custom'
        'kb_path': None,
        'kb_uploaded': False,
        'custom_kb_data': None,
        'pipeline': None,
        'results': None,
        'execution_complete': False,
        'current_stage': None,
        'progress': 0,
        'status_message': ""
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
    # Try multiple locations
    possible_paths = [
        "data/knowledge_base.json.gz",
        "./data/knowledge_base.json.gz",
        "../data/knowledge_base.json.gz",
        "/content/drive/MyDrive/siggen/knowledge_base.json.gz"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return "data/knowledge_base.json.gz"  # Default


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    """Render sidebar with configuration"""
    
    with st.sidebar:
        # Header
        st.markdown("### 🔧 Configuration")
        
        # HF Token
        with st.expander("🔐 Hugging Face Token", expanded=not st.session_state.token_validated):
            token_input = st.text_input(
                "API Token",
                type="password",
                value=st.session_state.hf_token or "",
                help="Required for LLM-powered query decomposition"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Validate", use_container_width=True):
                    if validate_hf_token(token_input):
                        st.session_state.hf_token = token_input
                        st.session_state.token_validated = True
                        st.success("✅ Valid!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("❌ Invalid")
            
            with col2:
                if st.session_state.token_validated:
                    st.markdown('<span class="status-badge status-success">Active</span>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown('<span class="status-badge status-warning">Required</span>', 
                              unsafe_allow_html=True)
            
            st.caption("[Get token →](https://huggingface.co/settings/tokens)")
        
        st.markdown("---")
        
        # About
        st.markdown("### ℹ️ About")
        st.caption("""
        **MSigDB Signature Generator v3.0**
        
        Production-grade biological signature generation with:
        - Dual KB management
        - LLM-powered query decomposition  
        - Multi-source integration
        - Advanced signature derivation
        """)
        
        st.caption("Built with ❤️ for biological research")


# ============================================================
# KB MANAGEMENT TAB
# ============================================================

def render_kb_management_tab():
    """Render Knowledge Base management tab"""
    
    st.markdown("## 📚 Knowledge Base Management")
    st.caption("Choose your knowledge base source")
    
    # KB Mode Selection
    col1, col2 = st.columns(2)
    
    with col1:
        builtin_selected = st.button(
            "🗄️ Use Built-in KB",
            use_container_width=True,
            type="primary" if st.session_state.kb_mode == 'builtin' else "secondary"
        )
        if builtin_selected:
            st.session_state.kb_mode = 'builtin'
            st.session_state.kb_uploaded = False
    
    with col2:
        custom_selected = st.button(
            "⚙️ Build Custom KB", 
            use_container_width=True,
            type="primary" if st.session_state.kb_mode == 'custom' else "secondary"
        )
        if custom_selected:
            st.session_state.kb_mode = 'custom'
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Show appropriate interface based on mode
    if st.session_state.kb_mode == 'builtin':
        render_builtin_kb_interface()
    else:
        render_custom_kb_interface()


def render_builtin_kb_interface():
    """Interface for using built-in KB"""
    
    st.markdown("### 🗄️ Built-in Knowledge Base")
    
    kb_path = get_default_kb_path()
    
    if os.path.exists(kb_path):
        st.success(f"✅ Knowledge base found: `{kb_path}`")
        
        # Load and show KB stats
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
            
            # Source breakdown
            with st.expander("📊 Source Breakdown"):
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
            st.error(f"Error loading KB: {e}")
            st.session_state.kb_uploaded = False
    else:
        st.warning(f"⚠️ Knowledge base not found at `{kb_path}`")
        st.info("Please ensure `knowledge_base.json.gz` is in the `data/` directory or switch to Custom KB mode.")
        st.session_state.kb_uploaded = False


def render_custom_kb_interface():
    """Interface for building custom KB from GMT files"""
    
    st.markdown("### ⚙️ Build Custom Knowledge Base")
    st.caption("Upload GMT files to create a custom knowledge base")
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        min_genes = st.number_input("Min genes per pathway", 1, 100, 5)
    with col2:
        max_genes = st.number_input("Max genes per pathway", 10, 1000, 500)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload GMT Files",
        type=['gmt', 'txt'],
        accept_multiple_files=True,
        help="Upload one or more GMT format files"
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} file(s) uploaded")
        
        # Show file list
        with st.expander("📋 Uploaded Files"):
            for file in uploaded_files:
                st.text(f"• {file.name} ({file.size / 1024:.1f} KB)")
        
        # Build button
        if st.button("🔨 Build Knowledge Base", type="primary", use_container_width=True):
            build_custom_kb(uploaded_files, min_genes, max_genes)


def build_custom_kb(uploaded_files, min_genes: int, max_genes: int):
    """Build custom KB from uploaded GMT files"""
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    try:
        # Initialize builder
        builder = KBBuilder(min_genes=min_genes, max_genes=max_genes)
        
        # Validate files
        status.info("🔍 Validating GMT files...")
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
        
        # Build KB
        status.info("🔨 Building knowledge base...")
        progress_bar.progress(40)
        
        kb_data = builder.build_kb(file_data)
        progress_bar.progress(70)
        
        # Save to temp file
        status.info("💾 Saving knowledge base...")
        temp_dir = tempfile.gettempdir()
        kb_path = os.path.join(temp_dir, 'custom_knowledge_base.json.gz')
        
        file_size = builder.save_kb(kb_data, kb_path)
        progress_bar.progress(90)
        
        # Display results
        status.success("✅ Knowledge base built successfully!")
        progress_bar.progress(100)
        
        summary = builder.get_summary()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Pathways", f"{summary['total_pathways_kept']:,}")
        with col2:
            st.metric("Sources", summary['total_sources'])
        with col3:
            st.metric("File Size", f"{file_size / (1024*1024):.2f} MB")
        
        # Source stats
        with st.expander("📊 Build Statistics"):
            for source, stats in summary['sources'].items():
                st.write(f"**{source}**")
                st.write(f"- Kept: {stats['kept']} pathways")
                st.write(f"- Filtered: {stats['too_small']} too small, {stats['too_large']} too large")
                st.write(f"- Duplicates: {stats['duplicates']}")
                st.write("---")
        
        # Update session state
        st.session_state.kb_path = kb_path
        st.session_state.kb_uploaded = True
        st.session_state.custom_kb_data = kb_data
        
        st.success("✅ Ready to use custom knowledge base!")
        
    except Exception as e:
        status.error(f"❌ Error building KB: {e}")
        progress_bar.empty()


# ============================================================
# PIPELINE TAB
# ============================================================

def render_pipeline_tab():
    """Render main pipeline execution tab"""
    
    # Check prerequisites
    if not st.session_state.token_validated:
        st.warning("⚠️ Please validate your Hugging Face token in the sidebar first")
        return
    
    if not st.session_state.kb_uploaded:
        st.warning("⚠️ Please select or build a knowledge base first")
        return
    
    st.markdown("## 🧬 Signature Generation Pipeline")
    
    # Stage 1: Query Input
    with st.expander("📝 Stage 1: Query & Target Definition", expanded=True):
        query = st.text_area(
            "Biological Query",
            height=100,
            placeholder="e.g., Pathways involved in pancreatic cancer progression and metastasis",
            help="Describe your biological question or research focus"
        )
        
        total_signatures = st.slider(
            "Target Signature Count",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Total number of signatures to generate"
        )
    
    # Stage 2: Design Controls
    with st.expander("⚙️ Stage 2: Design Controls", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Gene Size Constraints**")
            min_genes = st.number_input("Min genes per signature", 1, 50, 5)
            max_genes = st.number_input("Max genes per signature", 10, 500, 300)
        
        with col2:
            st.markdown("**Diversity Settings**")
            within_facet_overlap = st.slider("Within-facet overlap threshold", 0.0, 1.0, 0.50, 0.05)
            cross_facet_overlap = st.slider("Cross-facet overlap threshold", 0.0, 1.0, 0.25, 0.05)
    
    # Execute button
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Generate Signatures", type="primary", use_container_width=True):
        if not query:
            st.error("Please enter a biological query")
            return
        
        execute_pipeline(
            query=query,
            total_signatures=total_signatures,
            min_genes=min_genes,
            max_genes=max_genes,
            within_facet_overlap=within_facet_overlap,
            cross_facet_overlap=cross_facet_overlap
        )


def execute_pipeline(query: str, total_signatures: int, min_genes: int, max_genes: int,
                     within_facet_overlap: float, cross_facet_overlap: float):
    """Execute the signature generation pipeline"""
    
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status = st.empty()
        
        try:
            # Initialize config
            status.info("⚙️ Initializing pipeline...")
            progress_bar.progress(5)
            
            config = PipelineConfig(
                min_genes=min_genes,
                max_genes=max_genes,
                within_facet_overlap_threshold=within_facet_overlap,
                cross_facet_overlap_threshold=cross_facet_overlap
            )
            
            # Initialize pipeline
            status.info("🔧 Loading knowledge base...")
            progress_bar.progress(10)
            
            pipeline = CompletePipeline(
                kb_path=st.session_state.kb_path,
                hf_token=st.session_state.hf_token,
                config=config
            )
            
            # Run pipeline
            status.info("🧠 Running signature generation...")
            progress_bar.progress(20)
            
            results = pipeline.run(
                query=query,
                target_signature_count=total_signatures
            )
            
            progress_bar.progress(100)
            status.success("✅ Pipeline completed successfully!")
            
            # Store results
            st.session_state.results = results
            st.session_state.execution_complete = True
            
            st.balloons()
            
            # Auto-scroll to results
            st.rerun()
            
        except Exception as e:
            status.error(f"❌ Pipeline failed: {e}")
            progress_bar.empty()


# ============================================================
# RESULTS TAB
# ============================================================

def render_results_tab():
    """Render results and download tab"""
    
    if not st.session_state.execution_complete or not st.session_state.results:
        st.info("ℹ️ No results yet. Run the pipeline to generate signatures.")
        return
    
    results = st.session_state.results
    
    st.markdown("## 📊 Results & Downloads")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Signatures", results.get('total_signatures', 0))
    with col2:
        st.metric("Pathways Selected", results.get('total_pathways_selected', 0))
    with col3:
        st.metric("Unique Genes", results.get('total_unique_genes', 0))
    with col4:
        ratio = (results.get('total_signatures', 0) / results.get('total_pathways_selected', 1))
        st.metric("Signatures/Pathway", f"{ratio:.2f}")
    
    st.markdown("---")
    
    # Signature table
    with st.expander("📋 Signature Details", expanded=True):
        if results.get('derived_signatures'):
            sig_data = []
            for sig in results['derived_signatures']:
                sig_data.append({
                    'ID': sig['signature_id'],
                    'Facet': sig['facet_id'],
                    'Genes': sig['gene_count'],
                    'Source': sig['source'],
                    'Method': sig['derivation_method'],
                    'Confidence': f"{sig['confidence']:.3f}",
                    'Sample': ', '.join(sig['genes'][:3]) + '...'
                })
            
            df = pd.DataFrame(sig_data)
            st.dataframe(df, use_container_width=True, height=400)
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📊 Facet Distribution"):
            if results.get('facet_distribution'):
                facet_df = pd.DataFrame([
                    {'Facet': k, 'Count': v}
                    for k, v in results['facet_distribution'].items()
                ])
                st.bar_chart(facet_df.set_index('Facet'))
    
    with col2:
        with st.expander("📚 Source Distribution"):
            if results.get('source_distribution'):
                source_df = pd.DataFrame([
                    {'Source': k, 'Count': v}
                    for k, v in results['source_distribution'].items()
                ])
                st.bar_chart(source_df.set_index('Source'))
    
    st.markdown("---")
    
    # Download section
    st.markdown("### 💾 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(results, indent=2),
            file_name=f"signatures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Additional download formats would go in col2 and col3


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
    inject_modern_css()
    
    # Check pipeline availability
    if not PIPELINE_AVAILABLE:
        st.stop()
    
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 32px 0 16px 0;'>
        <h1>🧬 MSigDB Signature Generator</h1>
        <p style='font-size: 1.1rem; color: #94a3b8; margin-top: -8px;'>
            Production-grade biological signature generation with dual KB management
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📚 Knowledge Base", "🧬 Pipeline", "📊 Results"])
    
    with tab1:
        render_kb_management_tab()
    
    with tab2:
        render_pipeline_tab()
    
    with tab3:
        render_results_tab()


if __name__ == "__main__":
    main()

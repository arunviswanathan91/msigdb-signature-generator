"""
Built-in Material UI CSS - Dark Theme with Rounded Corners
===========================================================
No external dependencies - all CSS injected directly into Streamlit
"""

import streamlit as st


def inject_material_ui_css():
    """
    Inject comprehensive Material UI dark theme CSS directly into Streamlit.
    Features:
    - Dark navy blue theme (inspired by mui.com documentation)
    - Rounded corners throughout
    - Smooth animations
    - Conversation-style chat boxes
    - Database attribution badges
    """
    
    st.markdown("""
    <style>
    /* ============================================================
       GOOGLE FONTS
       ============================================================ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* ============================================================
       MATERIAL DARK THEME - DESIGN TOKENS
       ============================================================ */
    :root {
        /* Navy Dark Theme (mui.com style) */
        --mui-bg-default: #0A1929;      /* Deep Navy */
        --mui-bg-paper:   #001E3C;      /* Card/Sidebar */
        --mui-bg-subtle:  #132F4C;      /* Inputs/Hover */
        --mui-bg-elevated: #1A2F4C;     /* Elevated surfaces */
        
        /* Primary Brand Colors */
        --mui-primary-main: #007FFF;    /* MUI Blue */
        --mui-primary-dark: #0059B2;
        --mui-primary-light: #3399FF;
        
        /* Text Colors */
        --mui-text-primary:   #FFFFFF;
        --mui-text-secondary: #B2BAC2;
        --mui-text-disabled:  #5A6A7A;
        
        /* Borders & Dividers */
        --mui-divider: rgba(194, 224, 255, 0.08);
        --mui-border:  rgba(194, 224, 255, 0.12);
        
        /* Functional Colors */
        --mui-success: #1AA251;
        --mui-warning: #DEA500;
        --mui-error:   #EB0014;
        --mui-info:    #007FFF;
        
        /* AI Debate Colors */
        --ai-qwen: #FF6B9D;        /* Pink for Qwen */
        --ai-zephyr: #4ECDC4;      /* Teal for Zephyr */
        --ai-phi: #FFD93D;         /* Yellow for Phi-3 */
        --injector: #9B59B6;       /* Purple for DB Injector */
        --consensus: #2ECC71;      /* Green for Meta-synthesizer */
        
        /* Database Source Colors */
        --db-gtex: #E74C3C;        /* Red for GTEx */
        --db-gsea: #3498DB;        /* Blue for GSEA */
        --db-evidence: #F39C12;    /* Orange for Evidence */
        --db-gene-gene: #1ABC9C;   /* Turquoise for Gene-Gene */
        --db-pathway: #9B59B6;     /* Purple for Pathways */
        
        /* Shape & Typography */
        --mui-radius-base: 12px;   /* Rounded corners */
        --mui-radius-lg:   16px;
        --mui-font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        
        /* Shadows */
        --shadow-sm: 0 2px 4px rgba(0,0,0,0.2);
        --shadow-md: 0 4px 8px rgba(0,0,0,0.3);
        --shadow-lg: 0 8px 16px rgba(0,0,0,0.4);
    }
    
    /* ============================================================
       GLOBAL RESETS
       ============================================================ */
    html, body, [class*="css"] {
        font-family: var(--mui-font-family);
        -webkit-font-smoothing: antialiased;
        background-color: var(--mui-bg-default);
        color: var(--mui-text-primary);
    }
    
    .stApp {
        background-color: var(--mui-bg-default) !important;
        color: var(--mui-text-primary) !important;
    }
    
    /* Header */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--mui-bg-paper) !important;
        border-right: 1px solid var(--mui-divider) !important;
        border-radius: 0 !important;
    }
    
    /* Main Container */
    .block-container {
        padding-top: 2rem !important;
        max-width: 100% !important;
    }
    
    /* ============================================================
       TYPOGRAPHY
       ============================================================ */
    h1, h2, h3, h4, h5, h6 {
        color: var(--mui-text-primary) !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        margin-bottom: 1rem !important;
    }
    
    h1 {
        background: linear-gradient(90deg, #FFF 0%, var(--mui-primary-light) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
    }
    
    h2 {
        font-size: 2rem !important;
        color: var(--mui-text-primary) !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
    }
    
    p, li, .stMarkdown {
        color: var(--mui-text-secondary) !important;
        line-height: 1.7 !important;
    }
    
    /* ============================================================
       BUTTONS
       ============================================================ */
    .stButton button {
        border-radius: var(--mui-radius-base) !important;
        font-weight: 600 !important;
        text-transform: none !important;
        transition: all 0.2s ease-in-out !important;
        border: 1px solid var(--mui-border) !important;
        background-color: var(--mui-bg-paper) !important;
        color: var(--mui-primary-light) !important;
        box-shadow: var(--shadow-sm) !important;
        padding: 0.5rem 1.5rem !important;
    }
    
    .stButton button:hover {
        border-color: var(--mui-primary-main) !important;
        background-color: rgba(0, 127, 255, 0.08) !important;
        box-shadow: 0 0 16px rgba(0, 127, 255, 0.2) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Primary Button */
    .stButton button[kind="primary"] {
        background-color: var(--mui-primary-main) !important;
        color: white !important;
        border-color: var(--mui-primary-main) !important;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: var(--mui-primary-dark) !important;
        box-shadow: 0 0 20px rgba(0, 127, 255, 0.4) !important;
    }
    
    /* ============================================================
       INPUTS & TEXT AREAS
       ============================================================ */
    .stTextInput input, 
    .stTextArea textarea, 
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: var(--mui-bg-subtle) !important;
        border: 1px solid var(--mui-border) !important;
        border-radius: var(--mui-radius-base) !important;
        color: var(--mui-text-primary) !important;
        padding: 0.75rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    .stTextInput input:focus, 
    .stTextArea textarea:focus, 
    .stSelectbox div[data-baseweb="select"] > div:focus-within {
        border-color: var(--mui-primary-main) !important;
        box-shadow: 0 0 0 2px rgba(0, 127, 255, 0.25) !important;
        background-color: var(--mui-bg-elevated) !important;
    }
    
    /* Placeholder Text */
    ::placeholder {
        color: var(--mui-text-disabled) !important;
        opacity: 0.7 !important;
    }
    
    /* ============================================================
       SLIDERS
       ============================================================ */
    .stSlider {
        padding: 1rem 0 !important;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, var(--mui-primary-dark), var(--mui-primary-main)) !important;
        height: 6px !important;
        border-radius: 3px !important;
    }
    
    .stSlider > div > div > div > div {
        background: white !important;
        width: 24px !important;
        height: 24px !important;
        border: 3px solid var(--mui-primary-main) !important;
        box-shadow: var(--shadow-md) !important;
        border-radius: 50% !important;
    }
    
    .stSlider > div > div > div > div:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 0 16px rgba(0, 127, 255, 0.4) !important;
    }
    
    /* ============================================================
       EXPANDERS
       ============================================================ */
    .streamlit-expanderHeader {
        background-color: var(--mui-bg-paper) !important;
        border: 1px solid var(--mui-border) !important;
        border-radius: var(--mui-radius-base) !important;
        color: var(--mui-text-primary) !important;
        padding: 1rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: var(--mui-bg-subtle) !important;
        border-color: var(--mui-primary-main) !important;
    }
    
    .streamlit-expanderContent {
        background-color: var(--mui-bg-default) !important;
        border: 1px solid var(--mui-border) !important;
        border-top: none !important;
        border-radius: 0 0 var(--mui-radius-base) var(--mui-radius-base) !important;
        padding: 1rem !important;
    }
    
    /* ============================================================
       TABS
       ============================================================ */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 2px solid var(--mui-divider) !important;
        gap: 0.5rem !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--mui-text-secondary) !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: var(--mui-radius-base) var(--mui-radius-base) 0 0 !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--mui-bg-subtle) !important;
        color: var(--mui-text-primary) !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--mui-primary-light) !important;
        background-color: var(--mui-bg-paper) !important;
        border-bottom: 3px solid var(--mui-primary-main) !important;
    }
    
    /* ============================================================
       METRICS
       ============================================================ */
    [data-testid="stMetricValue"] {
        color: var(--mui-primary-light) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--mui-text-disabled) !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.05em !important;
    }
    
    /* ============================================================
       ALERTS & MESSAGES
       ============================================================ */
    .stAlert {
        border-radius: var(--mui-radius-base) !important;
        border-left: 4px solid !important;
        padding: 1rem 1.5rem !important;
        margin: 1rem 0 !important;
    }
    
    [data-testid="stAlert-info"] {
        background-color: rgba(0, 127, 255, 0.1) !important;
        border-left-color: var(--mui-info) !important;
        color: var(--mui-text-primary) !important;
    }
    
    [data-testid="stAlert-success"] {
        background-color: rgba(26, 162, 81, 0.1) !important;
        border-left-color: var(--mui-success) !important;
        color: var(--mui-text-primary) !important;
    }
    
    [data-testid="stAlert-warning"] {
        background-color: rgba(222, 165, 0, 0.1) !important;
        border-left-color: var(--mui-warning) !important;
        color: var(--mui-text-primary) !important;
    }
    
    [data-testid="stAlert-error"] {
        background-color: rgba(235, 0, 20, 0.1) !important;
        border-left-color: var(--mui-error) !important;
        color: var(--mui-text-primary) !important;
    }
    
    /* ============================================================
       CONVERSATIONAL CHAT BOXES (NEW)
       ============================================================ */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .chat-message {
        padding: 1rem 1.5rem;
        border-radius: var(--mui-radius-base);
        max-width: 85%;
        position: relative;
        animation: slideIn 0.3s ease-out;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--mui-border);
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* AI Model Messages */
    .chat-qwen {
        background: linear-gradient(135deg, rgba(255, 107, 157, 0.15), rgba(255, 107, 157, 0.05));
        border-left: 4px solid var(--ai-qwen);
        margin-left: 0;
    }
    
    .chat-zephyr {
        background: linear-gradient(135deg, rgba(78, 205, 196, 0.15), rgba(78, 205, 196, 0.05));
        border-left: 4px solid var(--ai-zephyr);
        margin-left: 0;
    }
    
    .chat-phi {
        background: linear-gradient(135deg, rgba(255, 217, 61, 0.15), rgba(255, 217, 61, 0.05));
        border-left: 4px solid var(--ai-phi);
        margin-left: 0;
    }
    
    /* Database Injector */
    .chat-injector {
        background: linear-gradient(135deg, rgba(155, 89, 182, 0.2), rgba(155, 89, 182, 0.05));
        border-left: 4px solid var(--injector);
        margin-left: 2rem;
        border-radius: 0 var(--mui-radius-base) var(--mui-radius-base) 0;
    }
    
    /* Meta-Synthesizer */
    .chat-consensus {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.2), rgba(46, 204, 113, 0.05));
        border: 2px solid var(--consensus);
        margin: 0 auto;
        max-width: 95%;
        text-align: center;
    }
    
    /* Chat Headers */
    .chat-header {
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .chat-header-qwen { color: var(--ai-qwen); }
    .chat-header-zephyr { color: var(--ai-zephyr); }
    .chat-header-phi { color: var(--ai-phi); }
    .chat-header-injector { color: var(--injector); }
    .chat-header-consensus { color: var(--consensus); }
    
    /* ============================================================
       DATABASE SOURCE BADGES (NEW)
       ============================================================ */
    .db-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        box-shadow: var(--shadow-sm);
    }
    
    .db-badge-gtex {
        background-color: rgba(231, 76, 60, 0.2);
        color: var(--db-gtex);
        border: 1px solid var(--db-gtex);
    }
    
    .db-badge-gsea {
        background-color: rgba(52, 152, 219, 0.2);
        color: var(--db-gsea);
        border: 1px solid var(--db-gsea);
    }
    
    .db-badge-evidence {
        background-color: rgba(243, 156, 18, 0.2);
        color: var(--db-evidence);
        border: 1px solid var(--db-evidence);
    }
    
    .db-badge-gene-gene {
        background-color: rgba(26, 188, 156, 0.2);
        color: var(--db-gene-gene);
        border: 1px solid var(--db-gene-gene);
    }
    
    .db-badge-pathway {
        background-color: rgba(155, 89, 182, 0.2);
        color: var(--db-pathway);
        border: 1px solid var(--db-pathway);
    }
    
    /* ============================================================
       PROGRESS BARS
       ============================================================ */
    .stProgress > div > div {
        background-color: var(--mui-bg-subtle);
        border-radius: var(--mui-radius-base);
        height: 8px;
    }
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--mui-primary-main), var(--mui-primary-light));
        border-radius: var(--mui-radius-base);
    }
    
    /* ============================================================
       SCROLLBARS
       ============================================================ */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--mui-bg-default);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--mui-bg-subtle);
        border-radius: 5px;
        border: 2px solid var(--mui-bg-default);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--mui-primary-main);
    }
    
    /* ============================================================
       DATAFRAMES & TABLES
       ============================================================ */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--mui-border) !important;
        border-radius: var(--mui-radius-lg) !important;
        overflow: hidden !important;
    }
    
    [data-testid="stDataFrame"] th {
        background-color: var(--mui-bg-paper) !important;
        color: var(--mui-text-primary) !important;
        border-bottom: 2px solid var(--mui-border) !important;
        padding: 1rem !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stDataFrame"] td {
        background-color: var(--mui-bg-default) !important;
        color: var(--mui-text-secondary) !important;
        border-bottom: 1px solid var(--mui-divider) !important;
        padding: 0.75rem 1rem !important;
    }
    
    [data-testid="stDataFrame"] tr:hover td {
        background-color: var(--mui-bg-subtle) !important;
    }
    
    /* ============================================================
       CHECKBOXES & RADIO
       ============================================================ */
    .stCheckbox input[type="checkbox"],
    .stRadio input[type="radio"] {
        accent-color: var(--mui-primary-main) !important;
    }
    
    /* ============================================================
       LOADING SPINNERS
       ============================================================ */
    .stSpinner > div {
        border-color: var(--mui-primary-main) !important;
    }
    
    /* ============================================================
       HIDE STREAMLIT BRANDING
       ============================================================ */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* ============================================================
       CUSTOM UTILITY CLASSES
       ============================================================ */
    .surface-card {
        background-color: var(--mui-bg-paper);
        border: 1px solid var(--mui-border);
        border-radius: var(--mui-radius-lg);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-md);
    }
    
    .elevated-card {
        background-color: var(--mui-bg-elevated);
        border: 1px solid var(--mui-border);
        border-radius: var(--mui-radius-lg);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-lg);
    }
    
    .divider {
        height: 1px;
        background-color: var(--mui-divider);
        margin: 2rem 0;
    }
    
    </style>
    """, unsafe_allow_html=True)


def render_chat_message(speaker: str, message: str, db_sources: list = None):
    """
    Render a chat message with proper styling and database attribution.
    
    Args:
        speaker: One of ['qwen', 'zephyr', 'phi', 'injector', 'consensus']
        message: The message content
        db_sources: List of database sources (e.g., ['GTEx', 'GSEA'])
    """
    speaker_labels = {
        'qwen': ('🤖 Qwen 2.5', 'qwen'),
        'zephyr': ('🤖 Zephyr', 'zephyr'),
        'phi': ('🤖 Phi-3', 'phi'),
        'injector': ('💉 Database Injector', 'injector'),
        'consensus': ('🎯 Meta-Synthesizer', 'consensus')
    }
    
    label, css_class = speaker_labels.get(speaker, ('Unknown', 'qwen'))
    
    # Render database badges if provided
    badges_html = ""
    if db_sources:
        badge_map = {
            'GTEx': 'gtex',
            'GSEA': 'gsea',
            'Evidence': 'evidence',
            'Gene-Gene': 'gene-gene',
            'Pathway': 'pathway'
        }
        for source in db_sources:
            badge_class = badge_map.get(source, 'gtex')
            badges_html += f'<span class="db-badge db-badge-{badge_class}">📊 {source}</span>'
    
    st.markdown(f"""
    <div class="chat-message chat-{css_class}">
        <div class="chat-header chat-header-{css_class}">
            {label}
        </div>
        {badges_html}
        <div style="color: var(--mui-text-secondary); line-height: 1.6;">
            {message}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_round_separator(round_num: int, total_rounds: int):
    """Render a visual separator between debate rounds"""
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 1.5rem;
        margin: 2rem 0;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(0, 127, 255, 0.2), 
            transparent);
        border-radius: var(--mui-radius-base);
        color: var(--mui-primary-light);
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 0.05em;
    ">
        🔄 Round {round_num} of {total_rounds}
    </div>
    """, unsafe_allow_html=True)


def render_convergence_indicator(convergence_rate: float):
    """Render a visual convergence indicator"""
    color = (
        "var(--mui-success)" if convergence_rate >= 0.85 
        else "var(--mui-warning)" if convergence_rate >= 0.60
        else "var(--mui-error)"
    )
    
    st.markdown(f"""
    <div style="
        background-color: var(--mui-bg-paper);
        border: 2px solid {color};
        border-radius: var(--mui-radius-base);
        padding: 1rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    ">
        <div style="
            font-size: 2rem;
            color: {color};
        ">
            {'✅' if convergence_rate >= 0.85 else '⚠️' if convergence_rate >= 0.60 else '🔄'}
        </div>
        <div style="flex: 1;">
            <div style="color: var(--mui-text-primary); font-weight: 600; margin-bottom: 0.25rem;">
                Convergence Rate
            </div>
            <div style="
                background-color: var(--mui-bg-subtle);
                height: 8px;
                border-radius: 4px;
                overflow: hidden;
            ">
                <div style="
                    background-color: {color};
                    height: 100%;
                    width: {convergence_rate * 100}%;
                    transition: width 0.3s ease-in-out;
                "></div>
            </div>
        </div>
        <div style="
            font-size: 1.5rem;
            font-weight: 700;
            color: {color};
            min-width: 4rem;
            text-align: right;
        ">
            {convergence_rate * 100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

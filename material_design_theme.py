# material_design_theme.py
import streamlit as st

def inject_material_theme():
    st.markdown("""
    <style>
    /* =========================================================
       GOOGLE FONTS
       ========================================================= */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    /* =========================================================
       MATERIAL COLOR TOKENS (DARK)
       ========================================================= */
    :root {
        --md-primary: #6DBAFF;
        --md-primary-hover: #82C7FF;

        --md-bg: #0F1419;
        --md-surface: #1B2025;
        --md-surface-high: #262B30;

        --md-text: #E1E2E5;
        --md-text-muted: #AAB2BD;

        --md-border: #3A4248;
        --md-radius: 14px;
    }

    /* =========================================================
       BASE APP (⚠️ NO GLOBAL * SELECTOR)
       ========================================================= */
    .stApp {
        background-color: var(--md-bg);
        color: var(--md-text);
        font-family: 'Roboto', sans-serif;
    }

    /* Streamlit main container */
    .block-container {
        padding-top: 2rem;
        max-width: 100%;
    }

    /* =========================================================
       TEXT & TYPOGRAPHY
       ========================================================= */
    p, span, div {
        line-height: 1.5;
    }

    label {
        color: var(--md-text-muted) !important;
        font-size: 0.85rem !important;
        font-weight: 500;
    }

    /* =========================================================
       INPUTS (TEXT / AREA / SELECT) – FIXED
       ========================================================= */
    input, textarea, select {
        background-color: var(--md-surface-high) !important;
        color: var(--md-text) !important;

        border-radius: var(--md-radius) !important;
        border: 1px solid var(--md-border) !important;

        padding: 0.6rem 0.8rem !important;
        line-height: 1.5 !important;

        height: auto !important;
        box-sizing: border-box !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: var(--md-text-muted) !important;
        opacity: 0.8;
    }

    /* =========================================================
       BUTTONS (MATERIAL FILLED)
       ========================================================= */
    .stButton > button {
        background-color: var(--md-primary);
        color: #002A45;

        border-radius: 999px;
        border: none;

        padding: 0.6rem 1.6rem;
        font-weight: 500;
        line-height: 1.2;

        transition: background-color 0.15s ease, transform 0.15s ease;
    }

    .stButton > button:hover {
        background-color: var(--md-primary-hover);
        transform: translateY(-1px);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* =========================================================
       SIDEBAR
       ========================================================= */
    section[data-testid="stSidebar"] {
        background-color: var(--md-surface);
        border-right: 1px solid var(--md-border);
    }

    /* =========================================================
       EXPANDERS, MARKDOWN, CONTAINERS (CRITICAL SAFETY)
       ========================================================= */
    .element-container,
    .stMarkdown,
    .stExpander,
    .stTextInput,
    .stTextArea,
    .stSelectbox {
        overflow: visible !important;
        height: auto !important;
        min-height: unset !important;
    }

    /* =========================================================
       TABS
       ========================================================= */
    .stTabs [data-baseweb="tab"] {
        color: var(--md-text-muted);
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        color: var(--md-primary) !important;
    }

    /* =========================================================
       PROGRESS BAR
       ========================================================= */
    .stProgress > div > div {
        background-color: var(--md-surface-high);
        border-radius: 4px;
    }

    .stProgress > div > div > div {
        background-color: var(--md-primary);
        border-radius: 4px;
    }

    /* =========================================================
       SCROLLBAR
       ========================================================= */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--md-surface);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--md-border);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--md-text-muted);
    }

    /* =========================================================
       STREAMLIT BRANDING HIDE (OPTIONAL)
       ========================================================= */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }

    </style>
    """, unsafe_allow_html=True)

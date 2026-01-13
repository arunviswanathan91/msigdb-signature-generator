import streamlit as st

# ==============================================================================
# 1. WRAPPER FUNCTIONS
#    (These map your custom names to standard Streamlit widgets)
# ==============================================================================


def material_text_field(label, value="", type="default", help=None, key=None, placeholder=None):
    return st.text_input(label, value=value, type=type, help=help, key=key, placeholder=placeholder)


def material_text_area(label, value="", height=None, placeholder=None, key=None):
    return st.text_area(label, value=value, height=height, placeholder=placeholder if placeholder else "", key=key)

def material_slider(label, min_value, max_value, value, step=None, help=None, key=None):
    return st.slider(label, min_value, max_value, value, step=step, help=help, key=key)

def material_select(label, options, index=0, key=None, help=None, on_change=None):
    return st.selectbox(label, options, index=index, key=key, help=help, on_change=on_change)

def material_multiselect(label, options, default=None, key=None, help=None, on_change=None):
    return st.multiselect(label, options, default=default, key=key, help=help, on_change=on_change)

def material_checkbox(label, value=False, key=None, help=None, label_visibility="visible"):
    return st.checkbox(label, value=value, key=key, help=help, label_visibility=label_visibility)

def material_button(label, key=None, type="secondary", use_container_width=False, on_click=None):
    return st.button(label, key=key, type=type, use_container_width=use_container_width, on_click=on_click)

def material_download_button(label, data, file_name=None, mime=None, key=None, use_container_width=False):
    return st.download_button(label, data, file_name=file_name, mime=mime, key=key, use_container_width=use_container_width)

# ==============================================================================
# 2. CSS INJECTION
#    (Fixes the UI and height issues)
# ==============================================================================

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
       BASE APP
       ========================================================= */
    .stApp {
        background-color: var(--md-bg);
        color: var(--md-text);
        font-family: 'Roboto', sans-serif;
    }

    /* Reduce top padding */
    .block-container {
        padding-top: 2rem;
        max-width: 100%;
    }

    p, span, div {
        line-height: 1.5;
    }

    label {
        color: var(--md-text-muted) !important;
        font-size: 0.85rem !important;
        font-weight: 500;
    }

    /* =========================================================
       INPUTS & SELECTS (Fixed Height)
       ========================================================= */
    /* Target inputs and selects, but exclude textareas so they can resize */
    input, select {
        background-color: var(--md-surface-high) !important;
        color: var(--md-text) !important;
        border-radius: var(--md-radius) !important;
        border: 1px solid var(--md-border) !important;
        padding: 0.6rem 0.8rem !important;
        line-height: 1.5 !important;
        box-sizing: border-box !important;
    }

    /* =========================================================
       TEXT AREA (Variable Height)
       ========================================================= */
    /* We do NOT set height: auto !important here, so Streamlit's 'height' param works */
    textarea {
        background-color: var(--md-surface-high) !important;
        color: var(--md-text) !important;
        border-radius: var(--md-radius) !important;
        border: 1px solid var(--md-border) !important;
        padding: 0.6rem 0.8rem !important;
        line-height: 1.5 !important;
        box-sizing: border-box !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: var(--md-text-muted) !important;
        opacity: 0.8;
    }

    /* =========================================================
       BUTTONS (Material Style)
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

    /* Secondary / Outlined Buttons */
    button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid var(--md-border) !important;
        color: var(--md-text) !important;
    }

    /* =========================================================
       SIDEBAR
       ========================================================= */
    section[data-testid="stSidebar"] {
        background-color: var(--md-surface);
        border-right: 1px solid var(--md-border);
    }

    /* =========================================================
       FIXES FOR UI CONTAINERS
       ========================================================= */
    /* These prevent elements from collapsing unexpectedly */
    .element-container,
    .stMarkdown,
    .stExpander,
    .stTextInput,
    .stSelectbox {
        overflow: visible !important;
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
       UTILITIES
       ========================================================= */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }

    </style>
    """, unsafe_allow_html=True)

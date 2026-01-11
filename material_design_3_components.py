import streamlit as st

# ============================================================
# MATERIAL-INSPIRED THEME (STREAMLIT-SAFE)
# ============================================================

def inject_material_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

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

    * {
        font-family: 'Roboto', sans-serif !important;
    }

    .stApp {
        background-color: var(--md-bg);
        color: var(--md-text);
    }

    /* Main container */
    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Inputs */
    input, textarea, select {
        background-color: var(--md-surface-high) !important;
        color: var(--md-text) !important;
        border-radius: var(--md-radius) !important;
        border: 1px solid var(--md-border) !important;
    }

    label {
        color: var(--md-text-muted) !important;
        font-size: 0.85rem !important;
        font-weight: 500;
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--md-primary);
        color: #002A45;
        border-radius: 999px;
        border: none;
        padding: 0.6rem 1.6rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: var(--md-primary-hover);
        transform: translateY(-1px);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--md-surface);
        border-right: 1px solid var(--md-border);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
    }

    </style>
    """, unsafe_allow_html=True)


# ============================================================
# SAFE UI WRAPPERS (ACCESSIBLE)
# ============================================================

def material_text(label, key, **kwargs):
    return st.text_input(label=label, key=key, **kwargs)

def material_text_area(label, key, **kwargs):
    return st.text_area(label=label, key=key, **kwargs)

def material_slider(label, key, min_value, max_value, value):
    return st.slider(label, min_value, max_value, value, key=key)

def material_select(label, options, key):
    return st.selectbox(label, options, key=key)

def material_checkbox(label, key, value=False):
    return st.checkbox(label, value=value, key=key)

def material_button(label, key):
    return st.button(label, key=key)

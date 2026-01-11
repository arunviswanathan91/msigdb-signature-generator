# material_design_theme.py
import streamlit as st

def inject_material_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    :root{
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

    *{ font-family: 'Roboto', sans-serif !important; }

    .stApp { background-color: var(--md-bg); color: var(--md-text); }
    .block-container { padding-top: 2rem; max-width: 1200px; }

    input, textarea, select {
        background-color: var(--md-surface-high) !important;
        color: var(--md-text) !important;
        border-radius: var(--md-radius) !important;
        border: 1px solid var(--md-border) !important;
    }

    label { color: var(--md-text-muted) !important; font-size: 0.85rem !important; font-weight: 500; }

    .stButton > button {
        background-color: var(--md-primary);
        color: #002A45;
        border-radius: 999px;
        border: none;
        padding: 0.6rem 1.6rem;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    .stButton > button:hover { background-color: var(--md-primary-hover); transform: translateY(-1px); }

    section[data-testid="stSidebar"] { background-color: var(--md-surface); border-right: 1px solid var(--md-border); }
    </style>
    """, unsafe_allow_html=True)


# -------------------------
# Flexible wrappers
# -------------------------
# All wrappers accept **kwargs and forward them to Streamlit,
# so your app can call with parameters like type/use_container_width/label_visibility.

def material_text_field(label, key=None, value=None, type:str="default", help=None, **kwargs):
    if type == "password":
        return st.text_input(label=label, value=value or "", key=key, type="password", help=help, **kwargs)
    return st.text_input(label=label, value=value or "", key=key, help=help, **kwargs)

def material_text_area(label, key=None, height=140, placeholder=None, value=None, **kwargs):
    return st.text_area(label=label, key=key, height=height, placeholder=placeholder or "", value=value or "", **kwargs)

def material_slider(label, key=None, min_value=0, max_value=10, value=None, step=None, **kwargs):
    # streamlit.slider expects value default; choose sensible default if None
    default_value = value if value is not None else min_value
    return st.slider(label=label, min_value=min_value, max_value=max_value, value=default_value, step=step, key=key, **kwargs)

def material_select(label, options=None, key=None, index=0, **kwargs):
    options = options or []
    return st.selectbox(label=label, options=options, index=index if index < len(options) else 0, key=key, **kwargs)

def material_multiselect(label, options=None, key=None, default=None, **kwargs):
    options = options or []
    return st.multiselect(label=label, options=options, default=default or [], key=key, **kwargs)

def material_checkbox(label, key=None, value=False, label_visibility=None, **kwargs):
    # label must not be an empty string to avoid Streamlit warning.
    if label is None or label == "":
        # use a minimally visible non-empty label to avoid accessibility warning.
        label = "select"
    return st.checkbox(label=label, value=value, key=key, label_visibility=label_visibility or "visible", **kwargs)

def material_button(label, key=None, **kwargs):
    # Forward all kwargs — st.button supports use_container_width.
    return st.button(label, key=key, **kwargs)

def material_download_button(label, data, file_name, mime="application/octet-stream", key=None, **kwargs):
    return st.download_button(label, data=data, file_name=file_name, mime=mime, key=key, **kwargs)

"""
MUI.COM NAVY DARK THEME - CLEAN VERSION
Based on the user-provided clean CSS structure
Official mui.com documentation site palette
"""

import streamlit as st


def inject_mui_css():
    """
    Injects the official MUI.com "Navy Blue" Dark Theme CSS.
    Verified against mui.com documentation styles.
    """
    st.markdown("""
    <style>
    /* ============================================================
       MUI.COM DARK THEME - DESIGN TOKENS
       Based on the official mui.com documentation site palette.
       ============================================================ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        /* ------------------- PALETTE ------------------- */
        /* Backgrounds */
        --mui-bg-default: #0A1929;      /* The signature Deep Navy */
        --mui-bg-paper:   #001E3C;      /* Slightly lighter navy for cards/sidebars */
        --mui-bg-subtle:  #132F4C;      /* For inputs/hover states */
        
        /* Primary Brand Colors */
        --mui-primary-main: #007FFF;    /* MUI Blue */
        --mui-primary-dark: #0059B2;
        --mui-primary-light:#3399FF;    /* Lighter blue for dark mode text */
        
        /* Text Colors */
        --mui-text-primary:   #FFFFFF;
        --mui-text-secondary: #B2BAC2;  /* Muted blue-grey */
        --mui-text-disabled:  #5A6A7A;
        
        /* Borders & Dividers */
        --mui-divider: rgba(194, 224, 255, 0.08);
        --mui-border:  rgba(194, 224, 255, 0.12);
        
        /* Functional Colors */
        --mui-success: #1AA251;
        --mui-warning: #DEA500;
        --mui-error:   #EB0014;
        --mui-info:    #007FFF;
        
        /* ------------------- SHAPE & TYPOGRAPHY ------------------- */
        --mui-radius-base: 10px;
        --mui-radius-lg:   12px;
        --mui-font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ============================================================
       GLOBAL RESETS & BACKGROUNDS
       ============================================================ */
    html, body, [class*="css"] {
        font-family: var(--mui-font-family);
        -webkit-font-smoothing: antialiased;
    }
    
    /* App Container */
    .stApp {
        background-color: var(--mui-bg-default) !important;
        color: var(--mui-text-primary) !important;
    }

    /* Header (remove default streamlit white bar) */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--mui-bg-default) !important;
        border-right: 1px solid var(--mui-divider) !important;
    }

    /* ============================================================
       TYPOGRAPHY
       ============================================================ */
    h1, h2, h3, h4, h5, h6 {
        color: var(--mui-text-primary) !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
    }
    
    h1 {
        background: linear-gradient(90deg, #FFF 0%, var(--mui-primary-light) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem !important;
    }

    p, li, .stMarkdown {
        color: var(--mui-text-secondary) !important;
        line-height: 1.7 !important;
    }

    /* ============================================================
       COMPONENTS
       ============================================================ */
    
    /* --- BUTTONS --- */
    .stButton button {
        border-radius: var(--mui-radius-base) !important;
        font-weight: 600 !important;
        text-transform: none !important;
        transition: all 0.2s ease-in-out !important;
        border: 1px solid var(--mui-border) !important;
        background-color: var(--mui-bg-paper) !important;
        color: var(--mui-primary-light) !important;
    }

    /* Primary Button Hover */
    .stButton button:hover {
        border-color: var(--mui-primary-main) !important;
        background-color: rgba(0, 127, 255, 0.08) !important;
        box-shadow: 0 0 16px rgba(0, 127, 255, 0.2) !important;
    }

    /* --- INPUTS & TEXT AREAS --- */
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
        background-color: var(--mui-bg-subtle) !important;
        border: 1px solid var(--mui-border) !important;
        border-radius: var(--mui-radius-base) !important;
        color: var(--mui-text-primary) !important;
    }

    .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox div[data-baseweb="select"] > div:focus-within {
        border-color: var(--mui-primary-main) !important;
        box-shadow: 0 0 0 2px rgba(0, 127, 255, 0.25) !important;
    }

    /* --- SLIDERS --- */
    .stSlider > div > div > div {
        background: var(--mui-primary-main) !important;
        height: 4px !important;
    }
    
    .stSlider > div > div > div > div {
        background: var(--mui-primary-light) !important;
        width: 20px !important;
        height: 20px !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(0, 127, 255, 0.4) !important;
    }

    /* --- DATAFRAMES / TABLES --- */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--mui-border) !important;
        border-radius: var(--mui-radius-lg) !important;
        overflow: hidden !important;
    }
    
    [data-testid="stDataFrame"] th {
        background-color: var(--mui-bg-paper) !important;
        color: var(--mui-text-primary) !important;
        border-bottom: 1px solid var(--mui-border) !important;
    }
    
    [data-testid="stDataFrame"] td {
        background-color: var(--mui-bg-default) !important;
        color: var(--mui-text-secondary) !important;
        border-bottom: 1px solid var(--mui-divider) !important;
    }

    /* --- METRICS --- */
    [data-testid="stMetricValue"] {
        color: var(--mui-primary-light) !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--mui-text-disabled) !important;
    }

    /* --- EXPANDERS --- */
    .streamlit-expanderHeader {
        background-color: var(--mui-bg-paper) !important;
        border: 1px solid var(--mui-border) !important;
        border-radius: var(--mui-radius-base) !important;
        color: var(--mui-text-primary) !important;
    }
    
    .streamlit-expanderContent {
        background-color: var(--mui-bg-default) !important;
        border: 1px solid var(--mui-border) !important;
        border-top: none !important;
        color: var(--mui-text-secondary) !important;
    }
    
    /* --- TABS --- */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--mui-border) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--mui-text-secondary) !important;
        border: none !important;
        padding: 12px 16px !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--mui-primary-light) !important;
        border-bottom: 2px solid var(--mui-primary-main) !important;
    }

    /* --- CHECKBOXES --- */
    .stCheckbox input[type="checkbox"] {
        accent-color: var(--mui-primary-main) !important;
    }

    /* ============================================================
       SCROLLBARS (Webkit)
       ============================================================ */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
        background: var(--mui-bg-default);
    }
    ::-webkit-scrollbar-thumb {
        background: #1e4976;
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--mui-primary-main);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    </style>
    """, unsafe_allow_html=True)

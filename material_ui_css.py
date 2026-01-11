"""
REAL MUI.COM DARK THEME
Exact colors and design from mui.com website
Dark navy background with rich blue accents
"""

import streamlit as st


def inject_material_ui_css():
    st.markdown("""
    <style>
    /* ============================================================
       REAL MUI.COM DARK THEME
       Dark navy background with rich blues - exactly like mui.com
       ============================================================ */
    
    /* Import Inter font (used by mui.com) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* ============================================================
       COLOR VARIABLES - Exact mui.com colors
       ============================================================ */
    :root {
        /* Dark navy backgrounds */
        --mui-bg-primary: #0A1929;
        --mui-bg-secondary: #132F4C;
        --mui-bg-elevated: #1A2027;
        
        /* Blues */
        --mui-blue-light: #66B2FF;
        --mui-blue-main: #007FFF;
        --mui-blue-dark: #0059B2;
        
        /* Text colors */
        --mui-text-primary: #FFFFFF;
        --mui-text-secondary: #B2BAC2;
        --mui-text-disabled: #5A6A7A;
        
        /* Borders */
        --mui-border-subtle: rgba(194, 224, 255, 0.08);
        --mui-border-medium: rgba(194, 224, 255, 0.12);
        
        /* Status colors */
        --mui-success: #1AA251;
        --mui-warning: #FFB400;
        --mui-error: #E61E50;
        --mui-info: #007FFF;
    }
    
    /* ============================================================
       BASE STYLING - Dark navy background
       ============================================================ */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Main app background - dark navy */
    .stApp {
        background: var(--mui-bg-primary) !important;
        color: var(--mui-text-primary) !important;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }
    
    /* Remove white backgrounds everywhere */
    .stApp > header {
        background: transparent !important;
    }
    
    /* ============================================================
       TYPOGRAPHY - Clean hierarchy on dark
       ============================================================ */
    h1 {
        font-size: 3rem !important;
        font-weight: 800 !important;
        color: var(--mui-text-primary) !important;
        letter-spacing: -0.03em !important;
        margin-bottom: 1rem !important;
        line-height: 1.1 !important;
        background: linear-gradient(135deg, #FFFFFF 0%, var(--mui-blue-light) 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }
    
    h2 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--mui-text-primary) !important;
        margin-top: 3rem !important;
        margin-bottom: 1.5rem !important;
        letter-spacing: -0.02em !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: var(--mui-text-primary) !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    p, .stMarkdown {
        color: var(--mui-text-secondary) !important;
        font-size: 1rem !important;
        line-height: 1.7 !important;
    }
    
    /* Subtle text */
    .stCaption {
        color: var(--mui-text-disabled) !important;
        font-size: 0.875rem !important;
    }
    
    /* ============================================================
       BUTTONS - Glowing blue buttons like mui.com
       ============================================================ */
    
    /* Primary Button - Blue glow */
    .stButton button[kind="primary"],
    .stButton button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, var(--mui-blue-main) 0%, var(--mui-blue-dark) 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.875rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.9375rem !important;
        letter-spacing: 0.01em !important;
        text-transform: none !important;
        box-shadow: 0 0 20px rgba(0, 127, 255, 0.3),
                    0 4px 12px rgba(0, 127, 255, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        position: relative !important;
    }
    
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, var(--mui-blue-light) 0%, var(--mui-blue-main) 100%) !important;
        box-shadow: 0 0 30px rgba(0, 127, 255, 0.5),
                    0 8px 20px rgba(0, 127, 255, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    .stButton button[kind="primary"]:active {
        transform: translateY(0) !important;
        box-shadow: 0 0 15px rgba(0, 127, 255, 0.4) !important;
    }
    
    /* Secondary Button - Dark with blue border */
    .stButton button[kind="secondary"],
    .stButton button {
        background: var(--mui-bg-secondary) !important;
        color: var(--mui-blue-light) !important;
        border: 1.5px solid var(--mui-border-medium) !important;
        border-radius: 10px !important;
        padding: 0.875rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.9375rem !important;
        text-transform: none !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }
    
    .stButton button[kind="secondary"]:hover,
    .stButton button:hover {
        background: var(--mui-bg-elevated) !important;
        border-color: var(--mui-blue-main) !important;
        box-shadow: 0 0 15px rgba(0, 127, 255, 0.2),
                    0 4px 12px rgba(0, 0, 0, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Disabled state */
    .stButton button:disabled {
        background: var(--mui-bg-secondary) !important;
        color: var(--mui-text-disabled) !important;
        border-color: var(--mui-border-subtle) !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
        opacity: 0.5 !important;
    }
    
    /* ============================================================
       CARDS & CONTAINERS - Elevated dark cards
       ============================================================ */
    
    /* Info boxes - Dark with blue glow */
    .info-box {
        background: var(--mui-bg-secondary) !important;
        border-left: 4px solid var(--mui-blue-main) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1.5rem 0 !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4),
                    0 0 15px rgba(0, 127, 255, 0.1) !important;
        color: var(--mui-text-primary) !important;
    }
    
    .info-box strong {
        color: var(--mui-blue-light) !important;
        font-weight: 600 !important;
    }
    
    /* Success */
    .stSuccess, .success-box {
        background: var(--mui-bg-secondary) !important;
        border-left: 4px solid var(--mui-success) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        color: var(--mui-text-primary) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4),
                    0 0 15px rgba(26, 162, 81, 0.15) !important;
    }
    
    /* Warning */
    .stWarning, .warning-box {
        background: var(--mui-bg-secondary) !important;
        border-left: 4px solid var(--mui-warning) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        color: var(--mui-text-primary) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4),
                    0 0 15px rgba(255, 180, 0, 0.15) !important;
    }
    
    /* Error */
    .stError, .error-box {
        background: var(--mui-bg-secondary) !important;
        border-left: 4px solid var(--mui-error) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        color: var(--mui-text-primary) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4),
                    0 0 15px rgba(230, 30, 80, 0.15) !important;
    }
    
    /* Info */
    .stInfo {
        background: var(--mui-bg-secondary) !important;
        border-left: 4px solid var(--mui-info) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        color: var(--mui-text-primary) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4),
                    0 0 15px rgba(0, 127, 255, 0.15) !important;
    }
    
    /* ============================================================
       INPUTS - Dark inputs with blue focus
       ============================================================ */
    
    /* Text inputs */
    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input {
        background: var(--mui-bg-secondary) !important;
        border: 1.5px solid var(--mui-border-medium) !important;
        border-radius: 10px !important;
        padding: 0.875rem 1rem !important;
        font-size: 1rem !important;
        color: var(--mui-text-primary) !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stTextInput input:hover,
    .stTextArea textarea:hover,
    .stNumberInput input:hover {
        border-color: var(--mui-blue-main) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3),
                    0 0 10px rgba(0, 127, 255, 0.2) !important;
    }
    
    .stTextInput input:focus,
    .stTextArea textarea:focus,
    .stNumberInput input:focus {
        border-color: var(--mui-blue-main) !important;
        box-shadow: 0 0 0 3px rgba(0, 127, 255, 0.2),
                    0 4px 12px rgba(0, 0, 0, 0.3) !important;
        outline: none !important;
        background: var(--mui-bg-elevated) !important;
    }
    
    /* Text area specific */
    .stTextArea textarea {
        min-height: 120px !important;
    }
    
    /* Input labels */
    .stTextInput label,
    .stTextArea label,
    .stNumberInput label {
        color: var(--mui-text-secondary) !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Placeholder text */
    ::placeholder {
        color: var(--mui-text-disabled) !important;
        opacity: 0.6 !important;
    }
    
    /* ============================================================
       CHECKBOXES - Modern dark checkboxes
       ============================================================ */
    
    .stCheckbox {
        padding: 0.75rem 0 !important;
    }
    
    .stCheckbox > label {
        display: flex !important;
        align-items: center !important;
        cursor: pointer !important;
        padding: 0.875rem 1rem !important;
        border-radius: 10px !important;
        transition: background 0.2s ease !important;
        background: var(--mui-bg-secondary) !important;
        border: 1px solid var(--mui-border-subtle) !important;
    }
    
    .stCheckbox > label:hover {
        background: var(--mui-bg-elevated) !important;
        border-color: var(--mui-border-medium) !important;
    }
    
    .stCheckbox input[type="checkbox"] {
        width: 22px !important;
        height: 22px !important;
        cursor: pointer !important;
        accent-color: var(--mui-blue-main) !important;
        border-radius: 6px !important;
    }
    
    .stCheckbox span {
        color: var(--mui-text-secondary) !important;
        font-weight: 500 !important;
    }
    
    /* ============================================================
       SLIDERS - Blue gradient sliders
       ============================================================ */
    
    .stSlider {
        padding: 1rem 0 !important;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, var(--mui-blue-main) 0%, var(--mui-blue-light) 100%) !important;
        height: 8px !important;
        border-radius: 4px !important;
        box-shadow: 0 0 10px rgba(0, 127, 255, 0.3) !important;
    }
    
    .stSlider > div > div > div > div {
        background: #FFFFFF !important;
        border: 3px solid var(--mui-blue-main) !important;
        width: 24px !important;
        height: 24px !important;
        box-shadow: 0 0 10px rgba(0, 127, 255, 0.4),
                    0 4px 12px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    
    .stSlider > div > div > div > div:hover {
        transform: scale(1.3) !important;
        box-shadow: 0 0 20px rgba(0, 127, 255, 0.6),
                    0 6px 16px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* ============================================================
       SELECT BOXES - Dark dropdowns
       ============================================================ */
    
    .stSelectbox > div > div {
        background: var(--mui-bg-secondary) !important;
        border: 1.5px solid var(--mui-border-medium) !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
        padding: 0.5rem !important;
        color: var(--mui-text-primary) !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--mui-blue-main) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3),
                    0 0 10px rgba(0, 127, 255, 0.2) !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--mui-blue-main) !important;
        box-shadow: 0 0 0 3px rgba(0, 127, 255, 0.2),
                    0 4px 12px rgba(0, 0, 0, 0.3) !important;
        background: var(--mui-bg-elevated) !important;
    }
    
    /* Dropdown options */
    [data-baseweb="popover"] {
        background: var(--mui-bg-elevated) !important;
        border: 1px solid var(--mui-border-medium) !important;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.5) !important;
    }
    
    /* ============================================================
       RADIO BUTTONS - Card-style selection on dark
       ============================================================ */
    
    .stRadio > div {
        gap: 0.75rem !important;
    }
    
    .stRadio > div > label {
        background: var(--mui-bg-secondary) !important;
        border: 1.5px solid var(--mui-border-medium) !important;
        border-radius: 10px !important;
        padding: 1rem 1.5rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
        color: var(--mui-text-secondary) !important;
    }
    
    .stRadio > div > label:hover {
        background: var(--mui-bg-elevated) !important;
        border-color: var(--mui-blue-main) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3),
                    0 0 10px rgba(0, 127, 255, 0.2) !important;
    }
    
    .stRadio > div > label[data-checked="true"] {
        background: rgba(0, 127, 255, 0.1) !important;
        border-color: var(--mui-blue-main) !important;
        border-width: 2px !important;
        box-shadow: 0 0 20px rgba(0, 127, 255, 0.3),
                    0 4px 12px rgba(0, 0, 0, 0.3) !important;
        font-weight: 600 !important;
        color: var(--mui-blue-light) !important;
    }
    
    /* ============================================================
       EXPANDERS - Dark accordion
       ============================================================ */
    
    .streamlit-expanderHeader {
        background: var(--mui-bg-secondary) !important;
        border: 1.5px solid var(--mui-border-medium) !important;
        border-radius: 12px !important;
        padding: 1rem 1.5rem !important;
        font-weight: 500 !important;
        color: var(--mui-text-primary) !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--mui-bg-elevated) !important;
        border-color: var(--mui-blue-main) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3),
                    0 0 10px rgba(0, 127, 255, 0.2) !important;
    }
    
    .streamlit-expanderContent {
        border: 1.5px solid var(--mui-border-medium) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        padding: 1.5rem !important;
        background: var(--mui-bg-secondary) !important;
    }
    
    /* ============================================================
       TABS - Clean dark tabs
       ============================================================ */
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem !important;
        background: transparent !important;
        border-bottom: 2px solid var(--mui-border-medium) !important;
        padding-bottom: 0 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 1.5rem !important;
        color: var(--mui-text-secondary) !important;
        font-weight: 500 !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
        border-radius: 8px 8px 0 0 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 127, 255, 0.05) !important;
        color: var(--mui-blue-light) !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--mui-blue-light) !important;
        border-bottom: 3px solid var(--mui-blue-main) !important;
        background: transparent !important;
        font-weight: 600 !important;
    }
    
    /* Tab content */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 2rem !important;
    }
    
    /* ============================================================
       METRICS - Glowing stats
       ============================================================ */
    
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, var(--mui-blue-light) 0%, var(--mui-blue-main) 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        filter: drop-shadow(0 0 20px rgba(0, 127, 255, 0.5));
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        color: var(--mui-text-secondary) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }
    
    /* ============================================================
       PROGRESS BARS - Blue glow animation
       ============================================================ */
    
    .stProgress > div > div {
        background: var(--mui-bg-secondary) !important;
        border-radius: 10px !important;
        height: 12px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) inset !important;
    }
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--mui-blue-main) 0%, var(--mui-blue-light) 100%) !important;
        border-radius: 10px !important;
        box-shadow: 0 0 15px rgba(0, 127, 255, 0.6) !important;
        animation: progressPulse 2s ease infinite !important;
    }
    
    @keyframes progressPulse {
        0%, 100% {
            box-shadow: 0 0 15px rgba(0, 127, 255, 0.6);
        }
        50% {
            box-shadow: 0 0 25px rgba(0, 127, 255, 0.9);
        }
    }
    
    /* ============================================================
       SIDEBAR - Dark navigation
       ============================================================ */
    
    [data-testid="stSidebar"] {
        background: var(--mui-bg-primary) !important;
        border-right: 1px solid var(--mui-border-medium) !important;
        padding: 2rem 1.5rem !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--mui-text-secondary) !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: var(--mui-text-primary) !important;
        font-size: 1.125rem !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton button {
        width: 100% !important;
        justify-content: center !important;
    }
    
    /* ============================================================
       DOWNLOAD BUTTON - Success green glow
       ============================================================ */
    
    .stDownloadButton button {
        background: linear-gradient(135deg, var(--mui-success) 0%, #0D7D3A 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.875rem 2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 0 20px rgba(26, 162, 81, 0.3),
                    0 4px 12px rgba(26, 162, 81, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    
    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #22C55E 0%, var(--mui-success) 100%) !important;
        box-shadow: 0 0 30px rgba(26, 162, 81, 0.5),
                    0 8px 20px rgba(26, 162, 81, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    /* ============================================================
       SPINNER - Blue rotating
       ============================================================ */
    
    .stSpinner > div {
        border-color: var(--mui-blue-main) var(--mui-bg-secondary) var(--mui-bg-secondary) var(--mui-bg-secondary) !important;
        border-width: 3px !important;
    }
    
    /* ============================================================
       DATAFRAME / TABLE - Dark table
       ============================================================ */
    
    .stDataFrame {
        border: 1.5px solid var(--mui-border-medium) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4) !important;
    }
    
    [data-testid="stDataFrame"] th {
        background: var(--mui-bg-elevated) !important;
        color: var(--mui-text-primary) !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        border-bottom: 2px solid var(--mui-border-medium) !important;
    }
    
    [data-testid="stDataFrame"] td {
        padding: 0.875rem 1rem !important;
        border-bottom: 1px solid var(--mui-border-subtle) !important;
        background: var(--mui-bg-secondary) !important;
        color: var(--mui-text-secondary) !important;
    }
    
    [data-testid="stDataFrame"] tr:hover {
        background: var(--mui-bg-elevated) !important;
    }
    
    /* ============================================================
       CUSTOM BADGES - Glowing status
       ============================================================ */
    
    .timing-badge {
        display: inline-block;
        background: rgba(0, 127, 255, 0.15);
        color: var(--mui-blue-light);
        padding: 0.375rem 0.875rem;
        border-radius: 20px;
        font-size: 0.8125rem;
        font-weight: 600;
        margin-left: 0.75rem;
        border: 1px solid rgba(0, 127, 255, 0.3);
        box-shadow: 0 0 10px rgba(0, 127, 255, 0.3);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.375rem 0.875rem;
        border-radius: 20px;
        font-size: 0.8125rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border: 1px solid;
    }
    
    .status-success {
        background: rgba(26, 162, 81, 0.15);
        color: #4ADE80;
        border-color: rgba(26, 162, 81, 0.3);
    }
    
    .status-warning {
        background: rgba(255, 180, 0, 0.15);
        color: #FCD34D;
        border-color: rgba(255, 180, 0, 0.3);
    }
    
    .status-error {
        background: rgba(230, 30, 80, 0.15);
        color: #FF5C8D;
        border-color: rgba(230, 30, 80, 0.3);
    }
    
    .status-info {
        background: rgba(0, 127, 255, 0.15);
        color: var(--mui-blue-light);
        border-color: rgba(0, 127, 255, 0.3);
    }
    
    /* ============================================================
       CODE BLOCKS - Dark syntax highlighting
       ============================================================ */
    
    code {
        background: var(--mui-bg-elevated) !important;
        color: var(--mui-blue-light) !important;
        padding: 0.2em 0.4em !important;
        border-radius: 6px !important;
        font-size: 0.875em !important;
        border: 1px solid var(--mui-border-subtle) !important;
    }
    
    pre {
        background: var(--mui-bg-elevated) !important;
        border: 1px solid var(--mui-border-medium) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* ============================================================
       SCROLLBAR - Dark blue scrollbar
       ============================================================ */
    
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--mui-bg-secondary);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--mui-blue-dark) 0%, var(--mui-blue-main) 100%);
        border-radius: 10px;
        border: 2px solid var(--mui-bg-secondary);
        box-shadow: 0 0 10px rgba(0, 127, 255, 0.3);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--mui-blue-main) 0%, var(--mui-blue-light) 100%);
        box-shadow: 0 0 15px rgba(0, 127, 255, 0.5);
    }
    
    /* ============================================================
       ANIMATIONS - Smooth micro-interactions
       ============================================================ */
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .stMarkdown, .stButton, .stTextInput {
        animation: fadeInUp 0.4s ease-out;
    }
    
    /* Glow pulse for focused elements */
    @keyframes glowPulse {
        0%, 100% {
            box-shadow: 0 0 15px rgba(0, 127, 255, 0.3);
        }
        50% {
            box-shadow: 0 0 25px rgba(0, 127, 255, 0.5);
        }
    }
    
    /* ============================================================
       HIDE STREAMLIT BRANDING
       ============================================================ */
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* ============================================================
       RESPONSIVE DESIGN
       ============================================================ */
    
    @media (max-width: 768px) {
        h1 {
            font-size: 2rem !important;
        }
        
        h2 {
            font-size: 1.5rem !important;
        }
        
        .stButton button {
            width: 100% !important;
        }
        
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
    
    /* ============================================================
       SELECTION COLORS - Blue highlights
       ============================================================ */
    
    ::selection {
        background: rgba(0, 127, 255, 0.3);
        color: #FFFFFF;
    }
    
    ::-moz-selection {
        background: rgba(0, 127, 255, 0.3);
        color: #FFFFFF;
    }
    
    </style>
    """, unsafe_allow_html=True)

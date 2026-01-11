"""
REAL Material UI CSS - Beautiful Modern Design
Inspired by mui.com - Clean, Professional, Modern
"""

import streamlit as st


def inject_material_ui_css():
    st.markdown("""
    <style>
    /* ============================================================
       REAL MATERIAL UI DESIGN - Clean & Modern
       Based on mui.com design system
       ============================================================ */
    
    /* Import Inter font (modern, clean) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* ============================================================
       GLOBAL RESET & BASE
       ============================================================ */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Main app - Clean white background */
    .stApp {
        background: #ffffff;
    }
    
    /* Main container - Add breathing room */
    .main .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }
    
    /* ============================================================
       TYPOGRAPHY - Clean hierarchy
       ============================================================ */
    h1 {
        font-size: 3rem !important;
        font-weight: 700 !important;
        color: #0A1929 !important;
        letter-spacing: -0.02em !important;
        margin-bottom: 0.5rem !important;
        line-height: 1.2 !important;
    }
    
    h2 {
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #0A1929 !important;
        margin-top: 3rem !important;
        margin-bottom: 1rem !important;
        letter-spacing: -0.01em !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #1E3A5F !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    p, .stMarkdown {
        color: #3E5060 !important;
        font-size: 1rem !important;
        line-height: 1.7 !important;
    }
    
    /* Subtle text */
    .stCaption {
        color: #5B7083 !important;
        font-size: 0.875rem !important;
    }
    
    /* ============================================================
       BUTTONS - Beautiful Material Design
       ============================================================ */
    
    /* Primary Button - Blue gradient */
    .stButton button[kind="primary"],
    .stButton button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #007FFF 0%, #0059B2 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.9375rem !important;
        letter-spacing: 0.02em !important;
        text-transform: none !important;
        box-shadow: 0 1px 3px rgba(0, 127, 255, 0.3), 
                    0 4px 8px rgba(0, 127, 255, 0.15) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }
    
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0059B2 0%, #003D7A 100%) !important;
        box-shadow: 0 4px 12px rgba(0, 127, 255, 0.35), 
                    0 8px 20px rgba(0, 127, 255, 0.2) !important;
        transform: translateY(-2px) !important;
    }
    
    .stButton button[kind="primary"]:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 4px rgba(0, 127, 255, 0.3) !important;
    }
    
    /* Secondary Button - Outlined */
    .stButton button[kind="secondary"],
    .stButton button {
        background: #ffffff !important;
        color: #007FFF !important;
        border: 1.5px solid #E3EFFB !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.9375rem !important;
        text-transform: none !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }
    
    .stButton button[kind="secondary"]:hover,
    .stButton button:hover {
        background: #F0F7FF !important;
        border-color: #B3D7FF !important;
        box-shadow: 0 2px 8px rgba(0, 127, 255, 0.15) !important;
    }
    
    /* Disabled state */
    .stButton button:disabled {
        background: #F3F6F9 !important;
        color: #B0B8C4 !important;
        border-color: #E7EBF0 !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
        opacity: 0.6 !important;
    }
    
    /* ============================================================
       CARDS & CONTAINERS - Elevation system
       ============================================================ */
    
    /* Info boxes - Clean cards */
    .info-box {
        background: #F0F7FF !important;
        border-left: 4px solid #007FFF !important;
        border-radius: 12px !important;
        padding: 1.25rem 1.5rem !important;
        margin: 1.5rem 0 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        color: #1E3A5F !important;
    }
    
    .info-box strong {
        color: #0A1929 !important;
        font-weight: 600 !important;
    }
    
    /* Success */
    .stSuccess, .success-box {
        background: #F1FAF5 !important;
        border-left: 4px solid #1AA251 !important;
        border-radius: 12px !important;
        padding: 1.25rem 1.5rem !important;
        color: #1E4620 !important;
        box-shadow: 0 2px 8px rgba(26, 162, 81, 0.1) !important;
    }
    
    /* Warning */
    .stWarning, .warning-box {
        background: #FFF9E6 !important;
        border-left: 4px solid #FFB400 !important;
        border-radius: 12px !important;
        padding: 1.25rem 1.5rem !important;
        color: #5F3A00 !important;
        box-shadow: 0 2px 8px rgba(255, 180, 0, 0.1) !important;
    }
    
    /* Error */
    .stError, .error-box {
        background: #FFF0F1 !important;
        border-left: 4px solid #E61E50 !important;
        border-radius: 12px !important;
        padding: 1.25rem 1.5rem !important;
        color: #5F1A20 !important;
        box-shadow: 0 2px 8px rgba(230, 30, 80, 0.1) !important;
    }
    
    /* Info */
    .stInfo {
        background: #F0F7FF !important;
        border-left: 4px solid #007FFF !important;
        border-radius: 12px !important;
        padding: 1.25rem 1.5rem !important;
        color: #1E3A5F !important;
        box-shadow: 0 2px 8px rgba(0, 127, 255, 0.1) !important;
    }
    
    /* ============================================================
       INPUTS - Modern form elements
       ============================================================ */
    
    /* Text inputs */
    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input {
        border: 1.5px solid #E3EFFB !important;
        border-radius: 10px !important;
        padding: 0.875rem 1rem !important;
        font-size: 1rem !important;
        color: #0A1929 !important;
        background: #ffffff !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    }
    
    .stTextInput input:hover,
    .stTextArea textarea:hover,
    .stNumberInput input:hover {
        border-color: #B3D7FF !important;
        box-shadow: 0 2px 8px rgba(0, 127, 255, 0.08) !important;
    }
    
    .stTextInput input:focus,
    .stTextArea textarea:focus,
    .stNumberInput input:focus {
        border-color: #007FFF !important;
        box-shadow: 0 0 0 3px rgba(0, 127, 255, 0.1) !important;
        outline: none !important;
    }
    
    /* Text area specific */
    .stTextArea textarea {
        min-height: 120px !important;
    }
    
    /* Input labels */
    .stTextInput label,
    .stTextArea label,
    .stNumberInput label {
        color: #1E3A5F !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* ============================================================
       CHECKBOXES - Modern toggle style
       ============================================================ */
    
    .stCheckbox {
        padding: 0.75rem 0 !important;
    }
    
    .stCheckbox > label {
        display: flex !important;
        align-items: center !important;
        cursor: pointer !important;
        padding: 0.75rem 1rem !important;
        border-radius: 10px !important;
        transition: background 0.2s ease !important;
    }
    
    .stCheckbox > label:hover {
        background: #F0F7FF !important;
    }
    
    .stCheckbox input[type="checkbox"] {
        width: 22px !important;
        height: 22px !important;
        cursor: pointer !important;
        accent-color: #007FFF !important;
        border-radius: 6px !important;
    }
    
    .stCheckbox span {
        color: #1E3A5F !important;
        font-weight: 500 !important;
    }
    
    /* ============================================================
       SLIDERS - Beautiful range inputs
       ============================================================ */
    
    .stSlider {
        padding: 1rem 0 !important;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #007FFF 0%, #0059B2 100%) !important;
        height: 6px !important;
        border-radius: 3px !important;
    }
    
    .stSlider > div > div > div > div {
        background: #ffffff !important;
        border: 3px solid #007FFF !important;
        width: 20px !important;
        height: 20px !important;
        box-shadow: 0 2px 8px rgba(0, 127, 255, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    
    .stSlider > div > div > div > div:hover {
        transform: scale(1.2) !important;
        box-shadow: 0 4px 12px rgba(0, 127, 255, 0.4) !important;
    }
    
    /* ============================================================
       SELECT BOXES - Dropdown styling
       ============================================================ */
    
    .stSelectbox > div > div {
        border: 1.5px solid #E3EFFB !important;
        border-radius: 10px !important;
        background: #ffffff !important;
        transition: all 0.2s ease !important;
        padding: 0.5rem !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #B3D7FF !important;
        box-shadow: 0 2px 8px rgba(0, 127, 255, 0.08) !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #007FFF !important;
        box-shadow: 0 0 0 3px rgba(0, 127, 255, 0.1) !important;
    }
    
    /* ============================================================
       RADIO BUTTONS - Card style selection
       ============================================================ */
    
    .stRadio > div {
        gap: 0.75rem !important;
    }
    
    .stRadio > div > label {
        background: #ffffff !important;
        border: 1.5px solid #E3EFFB !important;
        border-radius: 10px !important;
        padding: 1rem 1.5rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    }
    
    .stRadio > div > label:hover {
        background: #F0F7FF !important;
        border-color: #B3D7FF !important;
        box-shadow: 0 2px 8px rgba(0, 127, 255, 0.08) !important;
    }
    
    .stRadio > div > label[data-checked="true"] {
        background: #E6F4FF !important;
        border-color: #007FFF !important;
        border-width: 2px !important;
        box-shadow: 0 2px 8px rgba(0, 127, 255, 0.15) !important;
        font-weight: 600 !important;
        color: #0A1929 !important;
    }
    
    /* ============================================================
       EXPANDERS - Accordion style
       ============================================================ */
    
    .streamlit-expanderHeader {
        background: #ffffff !important;
        border: 1.5px solid #E7EBF0 !important;
        border-radius: 12px !important;
        padding: 1rem 1.5rem !important;
        font-weight: 500 !important;
        color: #1E3A5F !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: #F3F6F9 !important;
        border-color: #B3D7FF !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    }
    
    .streamlit-expanderContent {
        border: 1.5px solid #E7EBF0 !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        padding: 1.5rem !important;
        background: #FAFBFC !important;
    }
    
    /* ============================================================
       TABS - Clean navigation
       ============================================================ */
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem !important;
        background: transparent !important;
        border-bottom: 2px solid #E7EBF0 !important;
        padding-bottom: 0 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 1.5rem !important;
        color: #5B7083 !important;
        font-weight: 500 !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
        border-radius: 8px 8px 0 0 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #F0F7FF !important;
        color: #007FFF !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #007FFF !important;
        border-bottom: 3px solid #007FFF !important;
        background: transparent !important;
        font-weight: 600 !important;
    }
    
    /* Tab content */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 2rem !important;
    }
    
    /* ============================================================
       METRICS - Stats cards
       ============================================================ */
    
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #0A1929 !important;
        background: linear-gradient(135deg, #007FFF 0%, #0059B2 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        color: #5B7083 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }
    
    /* ============================================================
       PROGRESS BARS - Smooth animation
       ============================================================ */
    
    .stProgress > div > div {
        background: #E7EBF0 !important;
        border-radius: 10px !important;
        height: 10px !important;
    }
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #007FFF 0%, #0059B2 100%) !important;
        border-radius: 10px !important;
        animation: progressShine 2s ease infinite !important;
    }
    
    @keyframes progressShine {
        0% { opacity: 1; }
        50% { opacity: 0.85; }
        100% { opacity: 1; }
    }
    
    /* ============================================================
       SIDEBAR - Clean navigation panel
       ============================================================ */
    
    [data-testid="stSidebar"] {
        background: #FAFBFC !important;
        border-right: 1px solid #E7EBF0 !important;
        padding: 2rem 1.5rem !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #3E5060 !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #0A1929 !important;
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
       DOWNLOAD BUTTON - Special styling
       ============================================================ */
    
    .stDownloadButton button {
        background: linear-gradient(135deg, #1AA251 0%, #0D7D3A 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(26, 162, 81, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    
    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #0D7D3A 0%, #0A5A2A 100%) !important;
        box-shadow: 0 4px 12px rgba(26, 162, 81, 0.35) !important;
        transform: translateY(-2px) !important;
    }
    
    /* ============================================================
       SPINNER - Loading animation
       ============================================================ */
    
    .stSpinner > div {
        border-color: #007FFF #E7EBF0 #E7EBF0 #E7EBF0 !important;
        border-width: 3px !important;
    }
    
    /* ============================================================
       DATAFRAME / TABLE - Clean table style
       ============================================================ */
    
    .stDataFrame {
        border: 1.5px solid #E7EBF0 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    [data-testid="stDataFrame"] th {
        background: #F3F6F9 !important;
        color: #1E3A5F !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        border-bottom: 2px solid #E7EBF0 !important;
    }
    
    [data-testid="stDataFrame"] td {
        padding: 0.875rem 1rem !important;
        border-bottom: 1px solid #F3F6F9 !important;
    }
    
    [data-testid="stDataFrame"] tr:hover {
        background: #F0F7FF !important;
    }
    
    /* ============================================================
       CUSTOM CLASSES - Special elements
       ============================================================ */
    
    /* Timing badge */
    .timing-badge {
        display: inline-block;
        background: linear-gradient(135deg, #E6F4FF 0%, #D6EBFF 100%);
        color: #0059B2;
        padding: 0.375rem 0.875rem;
        border-radius: 20px;
        font-size: 0.8125rem;
        font-weight: 600;
        margin-left: 0.75rem;
        box-shadow: 0 2px 4px rgba(0, 127, 255, 0.15);
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.375rem 0.875rem;
        border-radius: 20px;
        font-size: 0.8125rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-success {
        background: #E8F5E9;
        color: #1AA251;
    }
    
    .status-warning {
        background: #FFF9E6;
        color: #BF7A00;
    }
    
    .status-error {
        background: #FFF0F1;
        color: #E61E50;
    }
    
    .status-info {
        background: #E6F4FF;
        color: #0059B2;
    }
    
    /* ============================================================
       HIDE STREAMLIT BRANDING - Optional
       ============================================================ */
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Hide "Deploy" button */
    .stDeployButton {display: none;}
    
    /* ============================================================
       ANIMATIONS - Smooth micro-interactions
       ============================================================ */
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .stMarkdown, .stButton, .stTextInput {
        animation: fadeIn 0.3s ease-out;
    }
    
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
       SCROLLBAR - Custom styling
       ============================================================ */
    
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F3F6F9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #B3D7FF 0%, #91C5FF 100%);
        border-radius: 10px;
        border: 2px solid #F3F6F9;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #91C5FF 0%, #007FFF 100%);
    }
    
    </style>
    """, unsafe_allow_html=True)

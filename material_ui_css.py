"""
Material UI Inspired CSS
Replace inject_modern_css() function in your app with this
"""

def inject_material_ui_css():
    st.markdown("""
    <style>
    /* ============================================================
       MATERIAL UI INSPIRED DESIGN
       Clean, minimal, with clear visual feedback
       ============================================================ */
    
    /* Import Roboto font (Material UI default) */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    /* ============================================================
       GLOBAL STYLES
       ============================================================ */
    * {
        font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main app background - clean light grey */
    .stApp {
        background: #fafafa;
    }
    
    /* ============================================================
       TYPOGRAPHY
       ============================================================ */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 400 !important;
        color: #212121 !important;
        margin-bottom: 8px !important;
        letter-spacing: -0.5px !important;
    }
    
    h2 {
        font-size: 2rem !important;
        font-weight: 400 !important;
        color: #424242 !important;
        margin-top: 32px !important;
        margin-bottom: 16px !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
        font-weight: 500 !important;
        color: #616161 !important;
        margin-top: 24px !important;
        margin-bottom: 12px !important;
    }
    
    p, .stMarkdown {
        color: #616161 !important;
        line-height: 1.6 !important;
    }
    
    /* ============================================================
       BUTTONS - Clear states with Material Design
       ============================================================ */
    
    /* Primary buttons */
    .stButton button[kind="primary"],
    .stButton button[data-testid="baseButton-primary"] {
        background: #1976d2 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 10px 24px !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08) !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton button[kind="primary"]:hover,
    .stButton button[data-testid="baseButton-primary"]:hover {
        background: #1565c0 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.16), 0 2px 4px rgba(0,0,0,0.12) !important;
        transform: translateY(-1px) !important;
    }
    
    .stButton button[kind="primary"]:active,
    .stButton button[data-testid="baseButton-primary"]:active {
        background: #0d47a1 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.12) !important;
        transform: translateY(0) !important;
    }
    
    /* Secondary buttons */
    .stButton button[kind="secondary"],
    .stButton button {
        background: white !important;
        color: #1976d2 !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 4px !important;
        padding: 10px 24px !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.08) !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton button[kind="secondary"]:hover,
    .stButton button:hover {
        background: #f5f5f5 !important;
        border-color: #1976d2 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.12) !important;
    }
    
    .stButton button[kind="secondary"]:active,
    .stButton button:active {
        background: #eeeeee !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.08) !important;
    }
    
    /* Selected button state (for gene selection) */
    .stButton button.selected {
        background: #e3f2fd !important;
        color: #1976d2 !important;
        border: 2px solid #1976d2 !important;
        font-weight: 600 !important;
    }
    
    /* Disabled buttons */
    .stButton button:disabled {
        background: #f5f5f5 !important;
        color: #bdbdbd !important;
        border-color: #e0e0e0 !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
    }
    
    /* ============================================================
       CHECKBOXES - Clear visual feedback
       ============================================================ */
    .stCheckbox {
        padding: 8px 0 !important;
    }
    
    .stCheckbox > label {
        display: flex !important;
        align-items: center !important;
        cursor: pointer !important;
        padding: 12px !important;
        border-radius: 4px !important;
        transition: background 0.2s ease !important;
    }
    
    .stCheckbox > label:hover {
        background: #f5f5f5 !important;
    }
    
    /* Checkbox input */
    .stCheckbox input[type="checkbox"] {
        width: 20px !important;
        height: 20px !important;
        cursor: pointer !important;
        accent-color: #1976d2 !important;
    }
    
    /* ============================================================
       SLIDERS - Material Design style
       ============================================================ */
    .stSlider > div > div > div {
        background: #1976d2 !important;
    }
    
    .stSlider > div > div > div > div {
        background: #1976d2 !important;
        border: 2px solid white !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    
    .stSlider > div > div > div > div:hover {
        box-shadow: 0 0 0 8px rgba(25, 118, 210, 0.16) !important;
    }
    
    /* ============================================================
       TEXT INPUTS - Clear borders and focus states
       ============================================================ */
    .stTextInput input,
    .stTextArea textarea {
        border: 1px solid #e0e0e0 !important;
        border-radius: 4px !important;
        padding: 12px !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
        background: white !important;
    }
    
    .stTextInput input:hover,
    .stTextArea textarea:hover {
        border-color: #bdbdbd !important;
    }
    
    .stTextInput input:focus,
    .stTextArea textarea:focus {
        border-color: #1976d2 !important;
        box-shadow: 0 0 0 2px rgba(25, 118, 210, 0.2) !important;
        outline: none !important;
    }
    
    /* ============================================================
       CARDS / INFO BOXES - Material elevation
       ============================================================ */
    .info-box {
        background: white !important;
        border-left: 4px solid #1976d2 !important;
        padding: 16px 20px !important;
        border-radius: 4px !important;
        margin: 16px 0 !important;
        color: #424242 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08) !important;
    }
    
    .info-box strong {
        color: #212121 !important;
    }
    
    /* Success boxes */
    .stSuccess {
        background: #e8f5e9 !important;
        border-left: 4px solid #4caf50 !important;
        padding: 16px 20px !important;
        border-radius: 4px !important;
        color: #2e7d32 !important;
    }
    
    /* Warning boxes */
    .stWarning {
        background: #fff3e0 !important;
        border-left: 4px solid #ff9800 !important;
        padding: 16px 20px !important;
        border-radius: 4px !important;
        color: #e65100 !important;
    }
    
    /* Error boxes */
    .stError {
        background: #ffebee !important;
        border-left: 4px solid #f44336 !important;
        padding: 16px 20px !important;
        border-radius: 4px !important;
        color: #c62828 !important;
    }
    
    /* Info boxes */
    .stInfo {
        background: #e3f2fd !important;
        border-left: 4px solid #2196f3 !important;
        padding: 16px 20px !important;
        border-radius: 4px !important;
        color: #1565c0 !important;
    }
    
    /* ============================================================
       EXPANDERS - Card style with clear expand indicator
       ============================================================ */
    .streamlit-expanderHeader {
        background: white !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 4px !important;
        padding: 12px 16px !important;
        font-weight: 500 !important;
        color: #424242 !important;
        transition: all 0.2s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: #f5f5f5 !important;
        border-color: #bdbdbd !important;
    }
    
    .streamlit-expanderContent {
        border: 1px solid #e0e0e0 !important;
        border-top: none !important;
        border-radius: 0 0 4px 4px !important;
        padding: 16px !important;
        background: white !important;
    }
    
    /* ============================================================
       TABS - Material Design style
       ============================================================ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        background: white !important;
        border-bottom: 2px solid #e0e0e0 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px !important;
        color: #757575 !important;
        font-weight: 500 !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f5f5f5 !important;
        color: #424242 !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #1976d2 !important;
        border-bottom: 2px solid #1976d2 !important;
        background: transparent !important;
    }
    
    /* ============================================================
       METRICS - Card style
       ============================================================ */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 500 !important;
        color: #212121 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: #757575 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    /* ============================================================
       PROGRESS BARS - Material Design
       ============================================================ */
    .stProgress > div > div {
        background: #e0e0e0 !important;
        border-radius: 4px !important;
        height: 8px !important;
    }
    
    .stProgress > div > div > div {
        background: #1976d2 !important;
        border-radius: 4px !important;
    }
    
    /* ============================================================
       SIDEBAR - Clean material style
       ============================================================ */
    [data-testid="stSidebar"] {
        background: white !important;
        border-right: 1px solid #e0e0e0 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #424242 !important;
    }
    
    /* ============================================================
       SELECT BOXES - Material style
       ============================================================ */
    .stSelectbox > div > div {
        border: 1px solid #e0e0e0 !important;
        border-radius: 4px !important;
        background: white !important;
        transition: all 0.2s ease !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #bdbdbd !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #1976d2 !important;
        box-shadow: 0 0 0 2px rgba(25, 118, 210, 0.2) !important;
    }
    
    /* ============================================================
       RADIO BUTTONS - Clear selection
       ============================================================ */
    .stRadio > div {
        gap: 8px !important;
    }
    
    .stRadio > div > label {
        background: white !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 4px !important;
        padding: 12px 16px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    
    .stRadio > div > label:hover {
        background: #f5f5f5 !important;
        border-color: #bdbdbd !important;
    }
    
    .stRadio > div > label[data-checked="true"] {
        background: #e3f2fd !important;
        border-color: #1976d2 !important;
        border-width: 2px !important;
        font-weight: 500 !important;
    }
    
    /* ============================================================
       NUMBER INPUT - Material style
       ============================================================ */
    .stNumberInput input {
        border: 1px solid #e0e0e0 !important;
        border-radius: 4px !important;
        padding: 12px !important;
        background: white !important;
        transition: all 0.2s ease !important;
    }
    
    .stNumberInput input:hover {
        border-color: #bdbdbd !important;
    }
    
    .stNumberInput input:focus {
        border-color: #1976d2 !important;
        box-shadow: 0 0 0 2px rgba(25, 118, 210, 0.2) !important;
        outline: none !important;
    }
    
    /* ============================================================
       DOWNLOAD BUTTON - Material style
       ============================================================ */
    .stDownloadButton button {
        background: #4caf50 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 10px 24px !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.12) !important;
        transition: all 0.2s ease !important;
    }
    
    .stDownloadButton button:hover {
        background: #43a047 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.16) !important;
        transform: translateY(-1px) !important;
    }
    
    /* ============================================================
       SPINNER - Material Design
       ============================================================ */
    .stSpinner > div {
        border-color: #1976d2 transparent transparent transparent !important;
    }
    
    /* ============================================================
       REMOVE UNNECESSARY ELEMENTS
       ============================================================ */
    
    /* Remove Streamlit branding if you want */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ============================================================
       CUSTOM CLASSES FOR YOUR APP
       ============================================================ */
    
    /* Timing badge */
    .timing-badge {
        display: inline-block;
        background: #e3f2fd;
        color: #1565c0;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.875rem;
        font-weight: 500;
        margin-left: 8px;
    }
    
    /* Gene selection buttons - selected state */
    .gene-button-selected {
        background: #e3f2fd !important;
        color: #1976d2 !important;
        border: 2px solid #1976d2 !important;
        font-weight: 600 !important;
    }
    
    /* Gene selection buttons - unselected state */
    .gene-button-unselected {
        background: white !important;
        color: #9e9e9e !important;
        border: 1px solid #e0e0e0 !important;
        opacity: 0.6 !important;
        text-decoration: line-through !important;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-success {
        background: #e8f5e9;
        color: #2e7d32;
    }
    
    .status-warning {
        background: #fff3e0;
        color: #e65100;
    }
    
    .status-error {
        background: #ffebee;
        color: #c62828;
    }
    
    .status-info {
        background: #e3f2fd;
        color: #1565c0;
    }
    
    /* ============================================================
       RESPONSIVE DESIGN
       ============================================================ */
    @media (max-width: 768px) {
        h1 {
            font-size: 2rem !important;
        }
        
        .stButton button {
            width: 100% !important;
        }
    }
    
    </style>
    """, unsafe_allow_html=True)

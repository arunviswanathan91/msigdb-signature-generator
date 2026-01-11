"""
REAL MATERIAL DESIGN WEB COMPONENTS FOR STREAMLIT
Based on Material Design 3 (M3) - Google's official design system
Interactive components with proper Material theming
"""

import streamlit as st
import streamlit.components.v1 as components


def inject_material_design_3():
    """
    Inject Material Design 3 CSS + Web Components support
    Based on material-components/material-web
    """
    st.markdown("""
    <style>
    /* ============================================================
       MATERIAL DESIGN 3 (M3) - DARK THEME
       Based on material-components/material-web
       ============================================================ */
    
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    :root {
        /* M3 Dark Theme Colors */
        --md-sys-color-primary: #6DBAFF;
        --md-sys-color-on-primary: #003353;
        --md-sys-color-primary-container: #004975;
        --md-sys-color-on-primary-container: #C7E7FF;
        
        --md-sys-color-secondary: #B8C8DA;
        --md-sys-color-on-secondary: #23323F;
        --md-sys-color-secondary-container: #394857;
        --md-sys-color-on-secondary-container: #D4E4F6;
        
        --md-sys-color-surface: #0F1419;
        --md-sys-color-surface-dim: #0F1419;
        --md-sys-color-surface-bright: #353A40;
        --md-sys-color-surface-container-lowest: #0A0F13;
        --md-sys-color-surface-container-low: #171C21;
        --md-sys-color-surface-container: #1B2025;
        --md-sys-color-surface-container-high: #262B30;
        --md-sys-color-surface-container-highest: #30353B;
        
        --md-sys-color-on-surface: #E1E2E5;
        --md-sys-color-on-surface-variant: #C0C8D3;
        --md-sys-color-outline: #8A9399;
        --md-sys-color-outline-variant: #40484D;
        
        --md-sys-color-error: #FFB4AB;
        --md-sys-color-on-error: #690005;
        
        /* Typography */
        --md-sys-typescale-body-large-font: 'Roboto', sans-serif;
        --md-sys-typescale-body-large-size: 16px;
        --md-sys-typescale-body-large-weight: 400;
        
        /* Shape */
        --md-sys-shape-corner-small: 8px;
        --md-sys-shape-corner-medium: 12px;
        --md-sys-shape-corner-large: 16px;
        
        /* Elevation */
        --md-sys-elevation-1: 0 1px 3px rgba(0,0,0,0.3), 0 4px 8px rgba(0,0,0,0.15);
        --md-sys-elevation-2: 0 2px 6px rgba(0,0,0,0.3), 0 8px 16px rgba(0,0,0,0.15);
        --md-sys-elevation-3: 0 4px 12px rgba(0,0,0,0.3), 0 16px 24px rgba(0,0,0,0.15);
    }
    
    /* ============================================================
       BASE STYLING
       ============================================================ */
    * {
        font-family: 'Roboto', sans-serif;
    }
    
    .stApp {
        background-color: var(--md-sys-color-surface) !important;
        color: var(--md-sys-color-on-surface) !important;
    }
    
    /* ============================================================
       STREAMLIT COMPONENT OVERRIDES - M3 STYLE
       ============================================================ */
    
    /* Slider - Material Design 3 Style */
    .stSlider {
        padding: 1.5rem 0 !important;
    }
    
    /* Hide only the value tooltip, not the slider itself */
    .stSlider [data-baseweb="slider"] [role="slider"]::before {
        display: none !important;
    }
    
    /* Slider track */
    .stSlider [data-baseweb="slider"] {
        background: transparent !important;
    }
    
    .stSlider [data-baseweb="slider"] > div {
        background: transparent !important;
    }
    
    .stSlider [data-baseweb="slider"] [data-testid="stTickBar"] {
        background: var(--md-sys-color-outline-variant) !important;
        height: 4px !important;
        border-radius: 2px !important;
    }
    
    /* Slider filled track */
    .stSlider [data-baseweb="slider"] [data-testid="stTickBar"] > div {
        background: var(--md-sys-color-primary) !important;
    }
    
    /* Slider thumb */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: var(--md-sys-color-primary) !important;
        width: 20px !important;
        height: 20px !important;
        border-radius: 50% !important;
        border: none !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.4), 0 2px 8px rgba(0,0,0,0.2) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
    }
    
    .stSlider [data-baseweb="slider"] [role="slider"]:hover {
        transform: translateY(-50%) scale(1.2) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4), 0 4px 16px rgba(0,0,0,0.2) !important;
    }
    
    .stSlider [data-baseweb="slider"] [role="slider"]:active {
        transform: translateY(-50%) scale(1.3) !important;
    }
    
    /* Labels */
    .stSlider label {
        color: var(--md-sys-color-on-surface-variant) !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        letter-spacing: 0.1px !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Buttons - M3 Filled Button */
    .stButton > button,
    .stButton button,
    button[kind="primary"],
    button[kind="secondary"] {
        background: var(--md-sys-color-primary) !important;
        color: #000000 !important;  /* Pure black for maximum contrast */
        border: none !important;
        border-radius: var(--md-sys-shape-corner-large) !important;
        padding: 10px 24px !important;
        font-size: 14px !important;
        font-weight: 600 !important;  /* Increased weight for better visibility */
        letter-spacing: 0.1px !important;
        text-transform: none !important;
        box-shadow: var(--md-sys-elevation-1) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stButton > button:hover,
    .stButton button:hover {
        box-shadow: var(--md-sys-elevation-2) !important;
        background: color-mix(in srgb, var(--md-sys-color-primary) 90%, white) !important;
        color: #000000 !important;
    }
    
    .stButton > button:active,
    .stButton button:active {
        box-shadow: var(--md-sys-elevation-1) !important;
        color: #000000 !important;
    }
    
    /* Force text color in button children */
    .stButton button p,
    .stButton button span,
    .stButton button div {
        color: #000000 !important;
    }
    
    /* Text Input - M3 Filled Text Field */
    .stTextInput input {
        background: var(--md-sys-color-surface-container-highest) !important;
        border: none !important;
        border-bottom: 1px solid var(--md-sys-color-outline-variant) !important;
        border-radius: var(--md-sys-shape-corner-small) var(--md-sys-shape-corner-small) 0 0 !important;
        color: var(--md-sys-color-on-surface) !important;
        padding: 16px 12px 8px !important;
        font-size: 16px !important;
        transition: all 0.2s ease !important;
    }
    
    .stTextInput input:hover {
        background: color-mix(in srgb, var(--md-sys-color-surface-container-highest) 95%, white) !important;
    }
    
    .stTextInput input:focus {
        border-bottom-color: var(--md-sys-color-primary) !important;
        border-bottom-width: 2px !important;
        outline: none !important;
    }
    
    .stTextInput label {
        color: var(--md-sys-color-on-surface-variant) !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }
    
    /* Select Box - M3 Filled Dropdown */
    .stSelectbox > div > div {
        background: var(--md-sys-color-surface-container-highest) !important;
        border: none !important;
        border-bottom: 1px solid var(--md-sys-color-outline-variant) !important;
        border-radius: var(--md-sys-shape-corner-small) var(--md-sys-shape-corner-small) 0 0 !important;
        color: var(--md-sys-color-on-surface) !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-bottom-color: var(--md-sys-color-primary) !important;
        border-bottom-width: 2px !important;
    }
    
    /* Checkbox - M3 Style */
    .stCheckbox {
        color: var(--md-sys-color-on-surface) !important;
    }
    
    .stCheckbox input[type="checkbox"] {
        width: 18px !important;
        height: 18px !important;
        accent-color: var(--md-sys-color-primary) !important;
        border-radius: var(--md-sys-shape-corner-small) !important;
    }
    
    /* Cards - M3 Elevated Card */
    .element-container {
        transition: all 0.2s ease !important;
    }
    
    /* Info/Success/Warning/Error - M3 Snackbar style */
    .stSuccess, .stInfo, .stWarning, .stError {
        background: var(--md-sys-color-surface-container-high) !important;
        border-left: 4px solid var(--md-sys-color-primary) !important;
        border-radius: var(--md-sys-shape-corner-medium) !important;
        padding: 16px !important;
        box-shadow: var(--md-sys-elevation-1) !important;
    }
    
    .stSuccess {
        border-left-color: #4CAF50 !important;
    }
    
    .stWarning {
        border-left-color: #FF9800 !important;
    }
    
    .stError {
        border-left-color: var(--md-sys-color-error) !important;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: var(--md-sys-color-on-surface) !important;
        font-weight: 400 !important;
    }
    
    h1 {
        font-size: 45px !important;
        letter-spacing: -0.25px !important;
    }
    
    h2 {
        font-size: 36px !important;
        letter-spacing: 0px !important;
    }
    
    h3 {
        font-size: 24px !important;
        letter-spacing: 0px !important;
    }
    
    p {
        color: var(--md-sys-color-on-surface-variant) !important;
        font-size: 16px !important;
        line-height: 24px !important;
        letter-spacing: 0.5px !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--md-sys-color-surface-container-low) !important;
        border-right: 1px solid var(--md-sys-color-outline-variant) !important;
    }
    
    /* Tabs - M3 Primary Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--md-sys-color-outline-variant) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--md-sys-color-on-surface-variant) !important;
        border: none !important;
        padding: 14px 16px !important;
        font-weight: 500 !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--md-sys-color-primary) !important;
        border-bottom: 3px solid var(--md-sys-color-primary) !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: var(--md-sys-color-surface-container-highest) !important;
        height: 4px !important;
        border-radius: 2px !important;
    }
    
    .stProgress > div > div > div {
        background: var(--md-sys-color-primary) !important;
        border-radius: 2px !important;
    }
    
    /* Scrollbar - M3 Style */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--md-sys-color-surface-container-low);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--md-sys-color-outline-variant);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--md-sys-color-outline);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    </style>
    """, unsafe_allow_html=True)


def material_slider(label, min_value=0, max_value=100, value=50, step=1, key=None, help=None, disabled=False, format=None):
    """
    Material Design 3 Slider with custom styling
    Returns the slider value
    """
    # Use Streamlit's native slider but with M3 styling from CSS
    return st.slider(
        label=label,
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=step,
        key=key,
        help=help,
        disabled=disabled,
        format=format
    )


def material_button(label, key=None, on_click=None, type="primary", use_container_width=False, disabled=False, help=None):
    """
    Material Design 3 Button
    Returns True if clicked
    """
    return st.button(
        label, 
        key=key, 
        on_click=on_click, 
        type=type,
        use_container_width=use_container_width,
        disabled=disabled,
        help=help
    )


def material_text_field(label, value="", key=None, placeholder="", help=None, type="default", max_chars=None, disabled=False):
    """
    Material Design 3 Filled Text Field
    Returns the text value
    """
    return st.text_input(
        label=label,
        value=value,
        key=key,
        placeholder=placeholder,
        help=help,
        type=type,
        max_chars=max_chars,
        disabled=disabled
    )


def material_select(label, options, index=0, key=None, help=None, disabled=False, format_func=None):
    """
    Material Design 3 Filled Dropdown
    Returns the selected value
    """
    return st.selectbox(
        label=label,
        options=options,
        index=index,
        key=key,
        help=help,
        disabled=disabled,
        format_func=format_func
    )


def material_checkbox(label, value=False, key=None, help=None, disabled=False, label_visibility="visible"):
    """
    Material Design 3 Checkbox
    Returns the checkbox state
    """
    return st.checkbox(
        label=label,
        value=value,
        key=key,
        help=help,
        disabled=disabled,
        label_visibility=label_visibility
    )


def material_multiselect(label, options, default=None, key=None, help=None, disabled=False, format_func=None, max_selections=None):
    """
    Material Design 3 Multi-Select
    Returns the selected values
    """
    return st.multiselect(
        label=label,
        options=options,
        default=default,
        key=key,
        help=help,
        disabled=disabled,
        format_func=format_func,
        max_selections=max_selections
    )


def material_card(content, elevation=1):
    """
    Material Design 3 Card with elevation
    Wraps content in a card container
    """
    elevation_shadows = {
        1: "0 1px 3px rgba(0,0,0,0.3), 0 4px 8px rgba(0,0,0,0.15)",
        2: "0 2px 6px rgba(0,0,0,0.3), 0 8px 16px rgba(0,0,0,0.15)",
        3: "0 4px 12px rgba(0,0,0,0.3), 0 16px 24px rgba(0,0,0,0.15)"
    }
    
    st.markdown(f"""
    <div style="
        background: var(--md-sys-color-surface-container-low);
        border-radius: var(--md-sys-shape-corner-medium);
        padding: 16px;
        box-shadow: {elevation_shadows.get(elevation, elevation_shadows[1])};
        margin: 16px 0;
    ">
        {content}
    </div>
    """, unsafe_allow_html=True)


def material_chip(label, selected=False, icon=None):
    """
    Material Design 3 Chip (Filter/Assist chip)
    """
    bg_color = "var(--md-sys-color-primary-container)" if selected else "var(--md-sys-color-surface-container-high)"
    text_color = "var(--md-sys-color-on-primary-container)" if selected else "var(--md-sys-color-on-surface-variant)"
    
    icon_html = f"<span style='margin-right: 8px;'>{icon}</span>" if icon else ""
    
    st.markdown(f"""
    <div style="
        display: inline-block;
        background: {bg_color};
        color: {text_color};
        border-radius: 8px;
        padding: 6px 16px;
        font-size: 14px;
        font-weight: 500;
        margin: 4px;
        cursor: pointer;
        transition: all 0.2s ease;
    " onmouseover="this.style.boxShadow='var(--md-sys-elevation-1)'" 
       onmouseout="this.style.boxShadow='none'">
        {icon_html}{label}
    </div>
    """, unsafe_allow_html=True)


def material_snackbar(message, type="info"):
    """
    Material Design 3 Snackbar notification
    """
    if type == "success":
        st.success(message)
    elif type == "warning":
        st.warning(message)
    elif type == "error":
        st.error(message)
    else:
        st.info(message)


# Example usage function
def show_material_components_demo():
    """
    Demo of Material Design 3 components
    """
    inject_material_design_3()
    
    st.title("🎨 Material Design 3 Components")
    st.caption("Based on material-components/material-web")
    
    # Slider Demo
    st.subheader("Slider")
    slider_value = material_slider("Number of mechanisms", 0, 60, 10)
    st.caption(f"💡 Will generate {slider_value} mechanisms")
    
    # Text Field Demo
    st.subheader("Text Field")
    text_value = material_text_field("Enter query", placeholder="e.g., Th17 role in obesity")
    
    # Select Demo
    st.subheader("Dropdown")
    select_value = material_select(
        "Select model",
        options=["Claude Sonnet", "Claude Opus", "GPT-4"],
        index=0
    )
    
    # Buttons Demo
    st.subheader("Buttons")
    col1, col2, col3 = st.columns(3)
    with col1:
        if material_button("Primary"):
            material_snackbar("Primary button clicked!", "success")
    with col2:
        if material_button("Secondary"):
            material_snackbar("Secondary button clicked!", "info")
    with col3:
        if material_button("Tertiary"):
            material_snackbar("Tertiary button clicked!", "warning")
    
    # Chips Demo
    st.subheader("Chips")
    material_chip("Selected", selected=True, icon="✓")
    material_chip("Option 1", selected=False)
    material_chip("Option 2", selected=False)
    material_chip("Option 3", selected=False)
    
    # Card Demo
    st.subheader("Card")
    material_card("""
        <h3 style='margin: 0 0 8px 0;'>Card Title</h3>
        <p style='margin: 0;'>This is a Material Design 3 elevated card with shadow and rounded corners.</p>
    """, elevation=2)


if __name__ == "__main__":
    show_material_components_demo()

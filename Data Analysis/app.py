import streamlit as st
from utils.constants import beatbox_sounds, participants
from spectral_and_temporal_analysis_page import render_spectral_and_temporal_analysis_page as spectral_and_temporal_analysis_page
from comparison_page import comparison_page_render

def configure_page() -> None:
    """Configure the main page settings"""
    st.set_page_config(
        page_title="Spectral and Temporal Analysis of Beatbox Sounds",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def configure_sidebar() -> None:
    """Configure the sidebar navigation"""
    st.sidebar.markdown("### 🎵 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Choose Analysis Phase:",
        ["Overview", "Spectral and Temporal Analysis", "Comparison Analysis"],
        key="page_selector"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Stats")
    st.sidebar.info(f"**Participants:** {len(participants)}")
    st.sidebar.info(f"**Beatbox Sounds:** {len(beatbox_sounds)}")
    
    return page

def render_overview_page() -> None:
    """Render the overview page"""
    st.title("🎵 Spectral and Temporal Analysis of Beatbox Sounds")
    
    # Main overview section
    st.markdown("## 📋 Overview")
    st.markdown("""
    This application provides comprehensive analysis tools for beatbox sounds using advanced audio processing techniques. 
    The app allows you to analyze spectral and temporal characteristics, compare participant recordings with reference samples, 
    and visualize audio features through various methods.
    """)
    
    # Features section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔬 Spectra Analysis")
        st.markdown("""
        - **Audio Upload & Processing**
        - **Waveform Visualization** 
        - **Spectral Feature Extraction**
        - **Spectrogram Generation**
        - **Basic Audio Statistics**
        """)
        
    with col2:
        st.markdown("### 🎯 Phase 3 Analysis")
        st.markdown("""
        - **Participant vs Reference Matching**
        - **Similarity Scoring**
        """)
    
    # Technical details
    st.markdown("## 🛠️ Technical Features")
    
    feature_cols = st.columns(3)
    
    with feature_cols[0]:
        st.markdown("#### Audio Features")
        st.markdown("""
        - MFCC (Mel-frequency Cepstral Coefficients)
        - Chroma CQT
        - Spectral Centroid
        - Spectral Bandwidth
        - Spectral Contrast
        """)
    
    with feature_cols[1]:
        st.markdown("#### Visualization")
        st.markdown("""
        - Waveform Plots
        - Mel-Spectrograms
        - Feature Distribution Charts
        - Comparison Matrices
        - Interactive Plots
        """)
    
    with feature_cols[2]:
        st.markdown("#### Analysis Methods")
        st.markdown("""
        - Dynamic Time Warping (DTW)
        - Statistical Feature Analysis
        - Audio Similarity Metrics
        - Cross-Correlation Analysis
        - Machine Learning Ready Data
        """)

def render_spectral_and_temporal_analysis_page() -> None:
    """Render Spectral and Temporal Analysis page"""
    spectral_and_temporal_analysis_page()

def render_comparison_page() -> None:
    """Render Comparison page"""
    comparison_page_render()

def main() -> None:
    """Main application function"""
    # Configure page
    configure_page()
    
    # Configure sidebar and get selected page
    selected_page = configure_sidebar()
    
    # Render appropriate page based on selection
    if selected_page == "Overview":
        render_overview_page()
    elif selected_page == "Spectral and Temporal Analysis":
        render_spectral_and_temporal_analysis_page()
    elif selected_page == "Comparison Analysis":
        render_comparison_page()

if __name__ == "__main__":
    main()
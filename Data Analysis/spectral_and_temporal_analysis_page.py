import streamlit as st
import pandas as pd
import numpy as np
import os
from utils.analyze_audio_spectral import get_all_spectral_features
from utils.import_audio import get_participant_audio, get_all_audios_from_participant_number_and_sound_name, get_all_audios_from_sound_name
from utils.constants import beatbox_sounds
from utils.publish_graphs import visualize_multiple_waveforms, visualize_spectrogram, visualize_waveform, visualize_melspectrogram
import matplotlib.pyplot as plt

def render_spectral_and_temporal_analysis_page() -> None:
    """Render Spectral and Temporal Analysis page"""
    st.title("🔬 Spectral and Temporal Analysis - Basic Audio Processing")
    
    st.markdown("### Batch Process Multiple Files")
    
    render_batch_processing_section()

def render_batch_processing_section() -> None:
    """Render batch processing section"""
    from utils.constants import participants
        
    st.info("This section allows you to process multiple audio files at once and generate a comprehensive analysis report.")
    
    # Processing options
    col1, col2 = st.columns(2)
    
    with col1:
        # Add "Select All Participants" button
        col1_1, col1_2 = st.columns([3, 1])
        
        with col1_1:
            st.markdown("**Select Participants:**")
        with col1_2:
            if st.button("Select All", key="select_all_participants"):
                st.session_state.batch_participants = participants
        
        selected_participants = st.multiselect(
            "Choose participants",
            participants,
            default=participants,
            key="batch_participants"
        )
    
    with col2:
        # Add "Select All Sounds" button
        col2_1, col2_2 = st.columns([3, 1])
        
        with col2_1:
            st.markdown("**Select Sounds:**")
        with col2_2:
            if st.button("Select All", key="select_all_sounds"):
                st.session_state.batch_sounds = list(beatbox_sounds.keys())
        
        selected_sounds = st.multiselect(
            "Choose sounds",
            list(beatbox_sounds.keys()),
            default=list(beatbox_sounds.keys()),
            key="batch_sounds"
        )
    
    st.markdown("---")
    
    # Checkbox for graph generation    
    if st.button("Show sounds", type="primary"):
        if selected_participants and selected_sounds:
            show_available_sounds(selected_participants, selected_sounds)
        else:

            st.warning("Please select at least one participant and one sound to show available audio files.")

    st.markdown("---")
    generate_graphs = st.checkbox("Generate visualization graphs", value=False, key="generate_graphs_checkbox")
    if st.button("Start Batch Processing", type="primary"):
        if selected_participants and selected_sounds:
            run_batch_processing(selected_participants, selected_sounds, generate_graphs)
        else:
            st.warning("Please select at least one participant and one sound.")
    
def show_available_sounds(participants_list: list, sounds_list: list) -> None:
    """Show all available audio files for selected participants and sounds"""
    st.markdown("#### 🎵 Available Audio Files")
    
    for participant in participants_list:
        st.markdown(f"### Participant {participant}")
        
        for sound in sounds_list:
            try:
                # Get all audio files for this participant and sound
                audio_files = get_all_audios_from_participant_number_and_sound_name(
                    participant, sound, beatbox_sounds
                )
                
                if audio_files:
                    st.markdown(f"**{sound}** - Found {len(audio_files)} file(s):")
                    
                    # Display each audio file with a player
                    for i, audio_path in enumerate(audio_files, 1):
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            st.audio(audio_path)
                        
                            # Show waveform using the imported function
                            try:
                                fig = visualize_waveform(audio_path, figsize=(12,3) ,title=f"Waveform - Attempt {i}")
                                if fig:
                                    st.pyplot(fig)
                                    plt.close(fig)
                            except Exception as e:
                                st.error(f"Error generating waveform: {str(e)}")

                        with col2:
                            # Show spectrogram using the imported function
                            try:
                                fig, ax = visualize_spectrogram(audio_path, title=f"Spectrogram - Attempt {i}")
                                if fig:
                                    st.pyplot(fig)
                                    plt.close(fig)
                            except Exception as e:
                                st.error(f"Error generating spectrogram: {str(e)}")

                        with col3:
                            # Extract attempt number from filename
                            filename = os.path.basename(audio_path)
                            st.caption(f"Attempt {i}")
                            st.caption(f"File: {filename}")
                else:
                    st.warning(f"**{sound}** - No files found")
                    
            except FileNotFoundError as e:
                st.error(f"**{sound}** - {str(e)}")
            except Exception as e:
                st.error(f"**{sound}** - Error: {str(e)}")
        
        st.markdown("---") 


def run_batch_processing(participants_list: list, sounds_list: list, generate_graphs: bool) -> None:
    """Run batch processing for selected participants and sounds"""
    st.markdown("#### 📊 Batch Processing Results")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Feature keys in order
    feature_keys = ['chroma_cqt', 'spectrogram', 'melspectrogram', 'mfcc', 'rms', 
                   'spectral_centroid', 'spectral_bandwidth', 'spectral_contrast', 
                   'spectral_flatness', 'spectral_rolloff', 'zero_crossing_rate']
    
    stat_names = ['mean', 'std', 'median', 'min', 'max', 'p25', 'p75']
    
    # Build column names
    columns = ['participant', 'sound', 'attempt', 'filename']
    for feature in feature_keys:
        # Add 7 stat columns for each feature
        for stat in stat_names:
            columns.append(f"{feature}_{stat}")
        # Add graph column if enabled
        if generate_graphs:
            columns.append(f"{feature}_graph")
    
    all_data = []
    summary_data = []
    
    total_combinations = len(participants_list) * len(sounds_list)
    current_progress = 0
    
    for participant in participants_list:
        for sound in sounds_list:
            status_text.text(f"Processing Participant {participant} - {sound}...")
            
            try:
                audio_paths = get_all_audios_from_participant_number_and_sound_name(
                    participant, sound, beatbox_sounds
                )
                
                if not audio_paths:
                    continue
                
                participant_sound_stats = {feature: {stat: [] for stat in stat_names} for feature in feature_keys}
                
                for i, audio_path in enumerate(audio_paths, 1):
                    try:
                        features, summary_stats, graphs = get_all_spectral_features(
                            audio_path, generate_graphs=generate_graphs
                        )
                        
                        # Build row data
                        row_data = {
                            'participant': participant,
                            'sound': sound,
                            'attempt': i,
                            'filename': os.path.basename(audio_path)
                        }
                        
                        # Add feature statistics
                        for feature in feature_keys:
                            if feature in summary_stats:
                                stats = summary_stats[feature]
                                for stat in stat_names:
                                    if stat in stats:
                                        stat_value = stats[stat]
                                        # Convert to scalar for display
                                        if isinstance(stat_value, np.ndarray):
                                            display_val = f"{np.mean(stat_value):.4f}" if stat_value.size > 1 else f"{float(stat_value):.4f}"
                                        else:
                                            display_val = f"{stat_value:.4f}"
                                        row_data[f"{feature}_{stat}"] = display_val
                                        
                                        # Store for summary calculation
                                        participant_sound_stats[feature][stat].append(stat_value)
                                    else:
                                        row_data[f"{feature}_{stat}"] = "N/A"
                                
                                # Add graph link if enabled
                                if generate_graphs and graphs and feature in graphs:
                                    # Save graph and create link
                                    graph_filename = f"graph_{participant}_{sound.replace(' ', '_')}_{i}_{feature}.png"
                                    if graphs[feature]:
                                        graphs[feature].savefig(f"temp_graphs/{graph_filename}", dpi=100, bbox_inches='tight')
                                        plt.close(graphs[feature])
                                        row_data[f"{feature}_graph"] = f"/temp_graphs/{graph_filename}"
                                    else:
                                        row_data[f"{feature}_graph"] = "❌ Error"
                            else:
                                # Fill with N/A if feature missing
                                for stat in stat_names:
                                    row_data[f"{feature}_{stat}"] = "N/A"
                                if generate_graphs:
                                    row_data[f"{feature}_graph"] = "❌ Missing"
                        
                        all_data.append(row_data)
                        
                    except Exception as e:
                        st.error(f"Error processing {os.path.basename(audio_path)}: {str(e)}")
                        continue
                
                # Generate summary for this participant-sound combination
                summary_row = {
                    'participant': participant,
                    'sound': sound,
                    'num_attempts': len(audio_paths),
                }
                
                for feature in feature_keys:
                    for stat in stat_names:
                        if participant_sound_stats[feature][stat]:
                            values = participant_sound_stats[feature][stat]
                            if all(isinstance(val, np.ndarray) for val in values):
                                avg_value = np.mean([np.mean(val) for val in values])
                            else:
                                avg_value = np.mean(values)
                            summary_row[f"{feature}_{stat}_avg"] = f"{avg_value:.4f}"
                        else:
                            summary_row[f"{feature}_{stat}_avg"] = "N/A"
                
                summary_data.append(summary_row)
                
            except Exception as e:
                st.error(f"Error processing Participant {participant} - {sound}: {str(e)}")
            
            current_progress += 1
            progress_bar.progress(current_progress / total_combinations)
    
    status_text.text("✅ Processing completed!")
    
    # Display results
    if all_data:
        st.markdown("**📋 Detailed Results**")
        df = pd.DataFrame(all_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Detailed Results as CSV",
            data=csv,
            file_name="audio_analysis_detailed.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        st.markdown("**📊 Summary Statistics**")
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Download summary
        summary_csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Summary as CSV",
            data=summary_csv,
            file_name="audio_analysis_summary.csv",
            mime="text/csv"
        )
        
        # Create temp_graphs directory if it doesn't exist
        os.makedirs("temp_graphs", exist_ok=True)
        
        st.markdown("**🎯 Processing Complete**")
        st.success(f"Processed {len(participants_list)} participants and {len(sounds_list)} sounds successfully!")
        
        if generate_graphs:
            st.info("📊 Graphs are saved in the temp_graphs folder and linked in the table. For deployment, consider using cloud storage URLs.")
    else:
        st.warning("No data was processed successfully.")

# For backwards compatibility, create an alias
spectral_and_temporal_analysis_page = render_spectral_and_temporal_analysis_page
import streamlit as st
import pandas as pd
import numpy as np
import os
from utils.analyze_audio_spectral import get_all_spectral_features, compare_audio_with_reference, get_reference_audio_path, compare_audios_with_each_other
from utils.import_audio import get_participant_audio, get_all_audios_from_participant_number_and_sound_name
from utils.constants import beatbox_sounds, participants

def comparison_page_render() -> None:
    st.title("🎯 MFCC DTW Comparison")
    
    st.markdown("### DTW Comparison and Advanced Analysis")
    
    # Create tabs for different analysis types
    tab1, tab2, tab3 = st.tabs(["Intra-participant Comparison", "Inter-participant Comparison", "Beatbox to Instruments Comparison"])
    
    with tab1:
        render_intra_participant_analysis()
    
    with tab2:
        render_inter_participant_analysis()
    
    with tab3:
        render_beatbox_to_instruments_analysis()

def render_intra_participant_analysis() -> None:
    """Render intra-participant sound consistency analysis"""
    st.header("Intra-Participant Sound Consistency")
    
    st.markdown("""
    This analysis examines how consistent each participant is when producing the same sound multiple times.
    We compare all attempts of the same sound by the same participant using MFCC-DTW similarity.
    """)
    
    # UI Controls
    display_heatmaps = st.checkbox("Display Similarity Heatmaps", value=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_participants = st.multiselect(
            "Select Participants:",
            options=participants,
            default=participants,  # Default to all participants
            key="intra_participants"
        )
    
    with col2:
        selected_sounds = st.multiselect(
            "Select Sounds:",
            options=list(beatbox_sounds.keys()),
            default=list(beatbox_sounds.keys()),  # Default to all sounds
            key="intra_sounds"
        )
    
    if st.button("Analyze Intra-Participant Consistency", type="primary"):
        if not selected_participants or not selected_sounds:
            st.warning("Please select at least one participant and one sound.")
            return
            
        analyze_intra_participant_consistency(selected_participants, selected_sounds, display_heatmaps)

def analyze_intra_participant_consistency(selected_participants, selected_sounds, display_heatmaps):
    """Perform intra-participant consistency analysis"""
    
    # Create temp_graphs directory if heatmaps are requested
    if display_heatmaps:
        os.makedirs("temp_graphs", exist_ok=True)
    
    # Initialize summary data structure
    summary_data = {}
    detailed_results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_combinations = len(selected_participants) * len(selected_sounds)
    current_progress = 0
    
    # Build multi-level column structure for summary DataFrame
    columns = []
    for sound in selected_sounds:
        columns.append((sound, "Avg. Similarity"))
        if display_heatmaps:
            columns.append((sound, "Heatmap"))
    
    # Initialize summary DataFrame with multi-level columns
    multi_columns = pd.MultiIndex.from_tuples(columns)
    summary_df = pd.DataFrame(index=selected_participants, columns=multi_columns)
    
    # Process each participant-sound combination
    for participant in selected_participants:
        summary_data[participant] = {}
        detailed_results[participant] = {}
        
        for sound in selected_sounds:
            status_text.text(f"Processing Participant {participant} - {sound}...")
            
            try:
                # Get all audio files for this participant and sound
                audio_paths = get_all_audios_from_participant_number_and_sound_name(
                    participant, sound, beatbox_sounds
                )
                
                if len(audio_paths) < 2:
                    # Not enough files to compare
                    summary_df.loc[participant, (sound, "Avg. Similarity")] = "N/A"
                    if display_heatmaps:
                        summary_df.loc[participant, (sound, "Heatmap")] = "N/A"
                    detailed_results[participant][sound] = "Not enough audio files to compare."
                    
                else:
                    # Perform similarity analysis
                    if display_heatmaps:
                        similarity_matrix, average_similarity, audio_filenames, fig, sd_similarity = compare_audios_with_each_other(
                            audio_paths, graph=True
                        )
                        
                        # Save heatmap
                        heatmap_filename = f"heatmap_p{participant}_{sound.replace(' ', '_')}.png"
                        heatmap_path = os.path.join("temp_graphs", heatmap_filename)
                        fig.savefig(heatmap_path, dpi=100, bbox_inches='tight')
                        
                        # Store results with standard deviation
                        summary_df.loc[participant, (sound, "Avg. Similarity")] = f"{average_similarity:.4f} (±{sd_similarity:.4f})"
                        summary_df.loc[participant, (sound, "Heatmap")] = heatmap_path
                        
                        # Close figure to save memory
                        import matplotlib.pyplot as plt
                        plt.close(fig)
                        
                    else:
                        similarity_matrix, average_similarity, audio_filenames, _, sd_similarity = compare_audios_with_each_other(
                            audio_paths, graph=False
                        )
                        summary_df.loc[participant, (sound, "Avg. Similarity")] = f"{average_similarity:.4f} (±{sd_similarity:.4f})"
                    
                    # Store detailed results
                    detailed_results[participant][sound] = {
                        'similarity_matrix': similarity_matrix,
                        'audio_filenames': audio_filenames,
                        'average_similarity': average_similarity,
                        'sd_similarity': sd_similarity
                    }
                
            except ValueError as e:
                # Handle the case where there are fewer than 2 audio files
                summary_df.loc[participant, (sound, "Avg. Similarity")] = "N/A"
                if display_heatmaps:
                    summary_df.loc[participant, (sound, "Heatmap")] = "N/A"
                detailed_results[participant][sound] = f"Error: {str(e)}"
                
            except Exception as e:
                st.error(f"Error processing Participant {participant} - {sound}: {str(e)}")
                summary_df.loc[participant, (sound, "Avg. Similarity")] = "Error"
                if display_heatmaps:
                    summary_df.loc[participant, (sound, "Heatmap")] = "Error"
                detailed_results[participant][sound] = f"Error: {str(e)}"
            
            current_progress += 1
            progress_bar.progress(current_progress / total_combinations)
    
    status_text.text("✅ Analysis completed!")
    
    # Display Summary DataFrame
    st.markdown("## 📊 Summary Results")
    st.dataframe(summary_df, use_container_width=True)
    
    # Download option for summary
    csv = summary_df.to_csv()
    st.download_button(
        label="📥 Download Summary as CSV",
        data=csv,
        file_name="intra_participant_similarity_summary.csv",
        mime="text/csv"
    )
    
    # Display Detailed Similarity Matrices
    st.markdown("## 🔍 Detailed Similarity Matrices")
    
    for participant in selected_participants:
        st.markdown(f"### Participant {participant}")
        
        for sound in selected_sounds:
            st.markdown(f"#### Sound: {sound}")
            
            result = detailed_results[participant][sound]
            
            if isinstance(result, str):
                # Error or N/A case
                st.info(result)
            else:
                # Valid result case
                similarity_matrix = result['similarity_matrix']
                audio_filenames = result['audio_filenames']
                average_similarity = result['average_similarity']
                sd_similarity = result.get('sd_similarity', 0.0)  # Use get() for backward compatibility
                
                # Create DataFrame from similarity matrix
                similarity_df = pd.DataFrame(
                    similarity_matrix,
                    index=audio_filenames,
                    columns=audio_filenames
                )
                
                st.markdown(f"**Average Similarity: {average_similarity:.4f} (±{sd_similarity:.4f})**")
                st.dataframe(similarity_df, use_container_width=True)
                
                # Show heatmap if available
                if display_heatmaps:
                    heatmap_path = summary_df.loc[participant, (sound, "Heatmap")]
                    if heatmap_path != "N/A" and heatmap_path != "Error" and os.path.exists(heatmap_path):
                        st.image(heatmap_path, caption=f"Similarity Heatmap - Participant {participant} - {sound}")
            
            st.markdown("---")

def render_inter_participant_analysis() -> None:
    """Render inter-participant sound comparison analysis"""
    st.header("Inter-Participant Sound Comparison")
    
    st.markdown("""
    This analysis compares how different participants produce the same sound. 
    We gather all audio files for a specific sound across selected participants and analyze their similarities using MFCC-DTW.
    """)
    
    # UI Controls
    display_heatmaps = st.checkbox("Display Group Similarity Heatmaps", value=False, key="inter_heatmaps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_sounds = st.multiselect(
            "Select Sounds to Analyze:",
            options=list(beatbox_sounds.keys()),
            default=list(beatbox_sounds.keys()),  # Default to all sounds
            key="inter_sounds"
        )
    
    with col2:
        selected_participants = st.multiselect(
            "Select Participants:",
            options=participants,
            default=participants,  # Default to all participants
            key="inter_participants"
        )
    
    if st.button("Analyze Inter-Participant Sound Comparison", type="primary"):
        if not selected_sounds or not selected_participants:
            st.warning("Please select at least one sound and one participant.")
            return
            
        analyze_inter_participant_comparison(selected_sounds, selected_participants, display_heatmaps)

def analyze_inter_participant_comparison(selected_sounds, selected_participants, display_heatmaps):
    """Perform inter-participant sound comparison analysis"""
    
    # Create temp_graphs directory if heatmaps are requested
    if display_heatmaps:
        os.makedirs("temp_graphs", exist_ok=True)
    
    # Initialize summary data structure
    detailed_results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_sounds = len(selected_sounds)
    current_progress = 0
    
    # Build column structure for summary DataFrame
    columns = ["Avg. Group Similarity"]
    if display_heatmaps:
        columns.append("Heatmap")
    
    # Initialize summary DataFrame
    summary_df = pd.DataFrame(index=selected_sounds, columns=columns)
    
    # Process each sound
    for sound in selected_sounds:
        status_text.text(f"Processing sound: {sound}...")
        
        try:
            # Gather all audio files for this sound across selected participants
            all_audio_paths = []
            
            for participant in selected_participants:
                try:
                    participant_audio_paths = get_all_audios_from_participant_number_and_sound_name(
                        participant, sound, beatbox_sounds
                    )
                    all_audio_paths.extend(participant_audio_paths)
                except FileNotFoundError:
                    # Skip participants who don't have files for this sound
                    continue
                except Exception as e:
                    st.warning(f"Error getting files for Participant {participant} - {sound}: {str(e)}")
                    continue
            
            if len(all_audio_paths) < 2:
                # Not enough files to compare across participants
                summary_df.loc[sound, "Avg. Group Similarity"] = "N/A"
                if display_heatmaps:
                    summary_df.loc[sound, "Heatmap"] = "N/A"
                detailed_results[sound] = f"Not enough audio files to compare. Found {len(all_audio_paths)} file(s) across all selected participants."
                
            else:
                # Perform similarity analysis
                if display_heatmaps:
                    similarity_matrix, average_similarity, audio_filenames, fig, sd_similarity = compare_audios_with_each_other(
                        all_audio_paths, graph=True
                    )
                    
                    # Save heatmap
                    heatmap_filename = f"inter_heatmap_{sound.replace(' ', '_')}.png"
                    heatmap_path = os.path.join("temp_graphs", heatmap_filename)
                    fig.savefig(heatmap_path, dpi=100, bbox_inches='tight')
                    
                    # Store results with standard deviation
                    summary_df.loc[sound, "Avg. Group Similarity"] = f"{average_similarity:.4f} (±{sd_similarity:.4f})"
                    summary_df.loc[sound, "Heatmap"] = heatmap_path
                    
                    # Close figure to save memory
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                    
                else:
                    similarity_matrix, average_similarity, audio_filenames, _, sd_similarity = compare_audios_with_each_other(
                        all_audio_paths, graph=False
                    )
                    summary_df.loc[sound, "Avg. Group Similarity"] = f"{average_similarity:.4f} (±{sd_similarity:.4f})"
                
                # Store detailed results
                detailed_results[sound] = {
                    'similarity_matrix': similarity_matrix,
                    'audio_filenames': audio_filenames,
                    'average_similarity': average_similarity,
                    'sd_similarity': sd_similarity,
                    'total_files': len(all_audio_paths)
                }
            
        except ValueError as e:
            # Handle the case where there are fewer than 2 audio files
            summary_df.loc[sound, "Avg. Group Similarity"] = "N/A"
            if display_heatmaps:
                summary_df.loc[sound, "Heatmap"] = "N/A"
            detailed_results[sound] = f"Error: {str(e)}"
            
        except Exception as e:
            st.error(f"Error processing sound {sound}: {str(e)}")
            summary_df.loc[sound, "Avg. Group Similarity"] = "Error"
            if display_heatmaps:
                summary_df.loc[sound, "Heatmap"] = "Error"
            detailed_results[sound] = f"Error: {str(e)}"
        
        current_progress += 1
        progress_bar.progress(current_progress / total_sounds)
    
    status_text.text("✅ Analysis completed!")
    
    # Display Summary DataFrame
    st.markdown("## 📊 Summary Results")
    st.markdown("Shows overall similarity for each sound across all selected participants.")
    st.dataframe(summary_df, use_container_width=True)
    
    # Download option for summary
    csv = summary_df.to_csv()
    st.download_button(
        label="📥 Download Summary as CSV",
        data=csv,
        file_name="inter_participant_similarity_summary.csv",
        mime="text/csv"
    )
    
    # Display Detailed Similarity Matrices
    st.markdown("## 🔍 Detailed Group Similarity Matrices")
    st.markdown("Each matrix shows similarities between all audio files for that sound across participants.")
    
    for sound in selected_sounds:
        st.markdown(f"### Detailed Matrix for: {sound}")
        
        result = detailed_results[sound]
        
        if isinstance(result, str):
            # Error or N/A case
            st.info(result)
        else:
            # Valid result case
            similarity_matrix = result['similarity_matrix']
            audio_filenames = result['audio_filenames']
            average_similarity = result['average_similarity']
            sd_similarity = result.get('sd_similarity', 0.0)  # Use get() for backward compatibility
            total_files = result['total_files']
            
            # Create DataFrame from similarity matrix
            similarity_df = pd.DataFrame(
                similarity_matrix,
                index=audio_filenames,
                columns=audio_filenames
            )
            
            st.markdown(f"**Average Group Similarity: {average_similarity:.4f} (±{sd_similarity:.4f})** (across {total_files} audio files)")
            st.dataframe(similarity_df, use_container_width=True)
            
            # Show heatmap if available
            if display_heatmaps:
                heatmap_path = summary_df.loc[sound, "Heatmap"]
                if heatmap_path != "N/A" and heatmap_path != "Error" and os.path.exists(heatmap_path):
                    st.image(heatmap_path, caption=f"Group Similarity Heatmap - {sound}")
        
        st.markdown("---")

# Example of detailed results structure:
# {
#     'attempt_details': [
#         {
#             'attempt_number': 1,
#             'similarity_score': 0.7523,
#             'dtw_graph_path': 'temp_graphs/dtw_p1_Kick_Imitate_attempt1.png',
#             'attempt_path': '/path/to/participant/audio.wav'
#         }
#     ],
#     'avg_similarity': 0.7523,
#     'std_similarity': 0.0,
#     'reference_path': '/path/to/reference/audio.wav'
# }

def render_beatbox_to_instruments_analysis() -> None:
    """Render imitation Similarity analysis"""
    st.header("Imitation Similarity Analysis")
    
    st.markdown("""
    This analysis compares participant imitation attempts with the original reference audio samples.
    We use MFCC-DTW to measure how accurately participants can imitate specific sounds.
    """)
    
    # Import required constants
    from utils.constants import imitatation_sounds_sample, folder_path_to_audio_samples
    
    # UI Controls
    display_dtw_graphs = st.checkbox("Display DTW Warping Path Graphs", value=False, key="dtw_graphs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_participants = st.multiselect(
            "Select Participants:",
            options=participants,
            default=participants,
            key="imitation_participants"
        )
    
    with col2:
        # Filter to only show imitation sounds
        imitation_sounds = [sound for sound in beatbox_sounds.keys() if 'Imitate' in sound]
        selected_sounds = st.multiselect(
            "Select Imitation Sounds:",
            options=imitation_sounds,
            default=imitation_sounds, 
            key="imitation_sounds"
        )
    
    if st.button("Analyze Imitation Accuracy", type="primary"):
        if not selected_participants or not selected_sounds:
            st.warning("Please select at least one participant and one imitation sound.")
            return
            
        analyze_imitation_accuracy(selected_participants, selected_sounds, display_dtw_graphs)

def analyze_imitation_accuracy(selected_participants, selected_sounds, display_dtw_graphs):
    """Perform imitation accuracy analysis"""
    
    # Import required constants and functions
    from utils.constants import imitatation_sounds_sample, folder_path_to_audio_samples
    import matplotlib.pyplot as plt
    
    # Create temp_graphs directory if DTW graphs are requested
    if display_dtw_graphs:
        os.makedirs("temp_graphs", exist_ok=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_combinations = len(selected_participants) * len(selected_sounds)
    current_progress = 0
    
    # Build multi-level column structure for summary DataFrame
    columns = []
    for sound in selected_sounds:
        columns.append((sound, "Average Similarity"))
        columns.append((sound, "Std. Deviation"))
    
    # Initialize summary DataFrame with multi-level columns
    multi_columns = pd.MultiIndex.from_tuples(columns)
    summary_df = pd.DataFrame(index=selected_participants, columns=multi_columns)
    
    # Initialize detailed results storage
    detailed_results = {}
    
    # Process each participant-sound combination
    for participant in selected_participants:
        detailed_results[participant] = {}
        
        for sound in selected_sounds:
            status_text.text(f"Processing Participant {participant} - {sound}...")
            
            try:
                # Get all audio files for this participant and sound
                audio_paths = get_all_audios_from_participant_number_and_sound_name(
                    participant, sound, beatbox_sounds
                )
                
                # Get reference audio path
                reference_filename = imitatation_sounds_sample[sound]
                reference_audio_path = os.path.join(folder_path_to_audio_samples, reference_filename)
                
                if not os.path.exists(reference_audio_path):
                    st.error(f"Reference audio not found: {reference_audio_path}")
                    summary_df.loc[participant, (sound, "Average Similarity")] = "Error"
                    summary_df.loc[participant, (sound, "Std. Deviation")] = "Error"
                    detailed_results[participant][sound] = f"Reference audio not found: {reference_filename}"
                    continue
                
                # Calculate similarity for each attempt
                individual_similarities = []
                attempt_details = []
                
                for i, attempt_path in enumerate(audio_paths, 1):
                    try:
                        if display_dtw_graphs:
                            similarity_score, fig = compare_audio_with_reference(
                                attempt_path, reference_audio_path, generate_graph=True
                            )
                            
                            # Save DTW graph
                            dtw_filename = f"dtw_p{participant}_{sound.replace(' ', '_')}_attempt{i}.png"
                            dtw_path = os.path.join("temp_graphs", dtw_filename)
                            fig.savefig(dtw_path, dpi=100, bbox_inches='tight')
                            plt.close(fig)
                            
                            attempt_details.append({
                                'attempt_number': i,
                                'similarity_score': similarity_score,
                                'dtw_graph_path': dtw_path,
                                'attempt_path': attempt_path
                            })
                        else:
                            similarity_score, _ = compare_audio_with_reference(
                                attempt_path, reference_audio_path, generate_graph=False
                            )
                            
                            attempt_details.append({
                                'attempt_number': i,
                                'similarity_score': similarity_score,
                                'dtw_graph_path': None,
                                'attempt_path': attempt_path
                            })
                        
                        individual_similarities.append(similarity_score)
                        
                    except Exception as e:
                        st.warning(f"Error processing attempt {i} for Participant {participant} - {sound}: {str(e)}")
                        continue
                
                if individual_similarities:
                    # Calculate statistics
                    avg_similarity = np.mean(individual_similarities)
                    std_similarity = np.std(individual_similarities) if len(individual_similarities) > 1 else 0.0
                    
                    # Store in summary DataFrame
                    summary_df.loc[participant, (sound, "Average Similarity")] = f"{avg_similarity:.4f}"
                    if len(individual_similarities) > 1:
                        summary_df.loc[participant, (sound, "Std. Deviation")] = f"{std_similarity:.4f}"
                    else:
                        summary_df.loc[participant, (sound, "Std. Deviation")] = "N/A"
                    
                    # Store detailed results
                    detailed_results[participant][sound] = {
                        'attempt_details': attempt_details,
                        'avg_similarity': avg_similarity,
                        'std_similarity': std_similarity,
                        'reference_path': reference_audio_path
                    }
                else:
                    summary_df.loc[participant, (sound, "Average Similarity")] = "No valid attempts"
                    summary_df.loc[participant, (sound, "Std. Deviation")] = "N/A"
                    detailed_results[participant][sound] = "No valid attempts found"
                
            except FileNotFoundError:
                summary_df.loc[participant, (sound, "Average Similarity")] = "No files"
                summary_df.loc[participant, (sound, "Std. Deviation")] = "N/A"
                detailed_results[participant][sound] = f"No audio files found for participant {participant} and sound {sound}"
                
            except Exception as e:
                st.error(f"Error processing Participant {participant} - {sound}: {str(e)}")
                summary_df.loc[participant, (sound, "Average Similarity")] = "Error"
                summary_df.loc[participant, (sound, "Std. Deviation")] = "Error"
                detailed_results[participant][sound] = f"Error: {str(e)}"
            
            current_progress += 1
            progress_bar.progress(current_progress / total_combinations)
    
    status_text.text("✅ Analysis completed!")
    
    # Display Summary Table
    st.markdown("## 📊 Summary Results")
    st.markdown("Shows average imitation accuracy and consistency for each participant-sound combination.")
    st.dataframe(summary_df, use_container_width=True)
    
    # Download option for summary
    csv = summary_df.to_csv()
    st.download_button(
        label="📥 Download Summary as CSV",
        data=csv,
        file_name="imitation_accuracy_summary.csv",
        mime="text/csv"
    )
    
    # Display Detailed Breakdown
    st.markdown("## 🔍 Detailed Comparison Results")
    
    for participant in selected_participants:
        with st.expander(f"Participant {participant}"):
            for sound in selected_sounds:
                st.subheader(f"Sound: {sound}")
                
                result = detailed_results[participant][sound]
                
                if isinstance(result, str):
                    # Error or no files case
                    st.info(result)
                else:
                    # Valid result case
                    attempt_details = result['attempt_details']
                    avg_similarity = result['avg_similarity']
                    std_similarity = result['std_similarity']
                    reference_path = result['reference_path']
                    
                    st.markdown(f"**Reference Audio:** `{os.path.basename(reference_path)}`")
                    st.markdown(f"**Average Similarity:** {avg_similarity:.4f}")
                    if len(attempt_details) > 1:
                        st.markdown(f"**Standard Deviation:** {std_similarity:.4f}")
                    else:
                        st.markdown(f"**Standard Deviation:** N/A (only 1 attempt)")
                    st.markdown("---")

                    # Display individual attempts
                    for attempt in attempt_details:
                        attempt_num = attempt['attempt_number']
                        similarity_score = attempt['similarity_score']
                        dtw_graph_path = attempt['dtw_graph_path']
                                            
                        # Display similarity score and audio player
                        st.markdown(f"Attempt {attempt_num} Similarity: {similarity_score:.4f}")
                        # Display DTW graph if available, placed below the audio player
                        if display_dtw_graphs and dtw_graph_path and os.path.exists(dtw_graph_path):
                            st.image(dtw_graph_path, caption=f"DTW Warping Path - Attempt {attempt_num}")
                        
                        st.markdown("---")
    
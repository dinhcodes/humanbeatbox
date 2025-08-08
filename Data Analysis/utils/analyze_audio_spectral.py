import librosa
import numpy as np
from scipy.spatial.distance import euclidean
import os
from dtaidistance import dtw, dtw_ndim, similarity
from dtaidistance import dtw_visualisation as dtwvis    
import matplotlib.pyplot as plt

from utils.publish_graphs import (visualize_chroma_cqt, visualize_spectrogram_from_features, 
                            visualize_melspectrogram_from_features, visualize_mfcc_from_features,
                            visualize_rms_from_features, visualize_spectral_centroid_from_features,
                            visualize_spectral_bandwidth_from_features, visualize_spectral_rolloff_from_features,
                            visualize_spectral_flatness_from_features, visualize_spectral_contrast_from_features,
                            visualize_zero_crossing_rate_from_features, visualize_dtw_warping_paths, create_similarity_heatmap)

list_of_features_to_be_extracted = [
    'chroma_cqt', 
    'spectrogram',
    'melspectrogram', 
    'mfcc', 
    'rms', 
    'spectral_centroid',
    'spectral_bandwidth', 
    'spectral_contrast', 
    'spectral_flatness',
    'spectral_rolloff', 
    'zero_crossing_rate'
]

common_n_fft = 512
common_hop_length = 128

r_common = 3200 # Exponential similarity decay factor

def get_all_spectral_features(path, sr=None, generate_graphs=False, figsize=(12, 8)):
    """
    Extract all spectral features from audio file and optionally generate visualizations.
    
    Parameters:
    -----------
    path : str
        Path to the audio file
    sr : int, optional
        Sample rate for loading audio
    generate_graphs : bool, default False
        Whether to generate visualization graphs
    figsize : tuple, default (12, 8)
        Figure size for plots
        
    Returns:
    --------
    features : dict
        Dictionary of extracted features
    features_summary_statistics : dict
        Summary statistics for each feature
    graphs : dict or None
        Dictionary of matplotlib figures if generate_graphs=True, else None
    """    
    y, sr = librosa.load(path, sr=sr)
    filename = os.path.basename(path)

    features = {
        'chroma_cqt': librosa.feature.chroma_cqt(y=y, sr=sr),
        'spectrogram': np.abs(librosa.stft(y, n_fft=common_n_fft, hop_length=common_hop_length)),
        'melspectrogram': librosa.feature.melspectrogram(y=y, sr=sr, n_fft=common_n_fft, hop_length=common_hop_length, n_mels=128),
        'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_fft=common_n_fft, hop_length=common_hop_length, n_mfcc=16),
        'rms': librosa.feature.rms(y=y, frame_length=common_n_fft, hop_length=common_hop_length),
        'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=common_n_fft, hop_length=common_hop_length),
        'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=common_n_fft, hop_length=common_hop_length),
        'spectral_contrast': librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=common_n_fft, hop_length=common_hop_length),
        'spectral_flatness': librosa.feature.spectral_flatness(y=y, n_fft=common_n_fft, hop_length=common_hop_length),
        'spectral_rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=common_n_fft, hop_length=common_hop_length, roll_percent=0.85),
        'zero_crossing_rate': librosa.feature.zero_crossing_rate(y=y)
    }
    
    feature_keys = [
        'chroma_cqt',
        'spectrogram',
        'melspectrogram',
        'mfcc',
        'rms',
        'spectral_centroid',
        'spectral_bandwidth',
        'spectral_contrast',
        'spectral_flatness',
        'spectral_rolloff',
        'zero_crossing_rate'
    ]

    # 2. Define the statistics you want to compute
    # We use lambda functions for percentiles to handle their unique syntax
    stats_to_compute = {
        'mean': np.mean,
        'std': np.std,
        'min': np.min,
        'p25': lambda arr: np.percentile(arr, 25, axis=1),
        'median': np.median,
        'p75': lambda arr: np.percentile(arr, 75, axis=1),
        'max': np.max,
    }

    # 3. Use a loop to programmatically build the dictionary
    features_summary_statistics = {}
    for key in feature_keys:
        if key in features:
            # Create a nested dictionary for the current feature
            features_summary_statistics[key] = {}
            # Loop through the desired stats and compute them
            for stat_name, stat_func in stats_to_compute.items():
                # Apply the statistical function along axis=1
                features_summary_statistics[key][stat_name] = stat_func(features[key])
        else:
            print(f"Warning: Feature '{key}' not found in features dictionary.")

    if not generate_graphs:
        return features, features_summary_statistics, None
    
    # Generate graphs
    graphs = {
        'chroma_cqt': visualize_chroma_cqt(features, sr, filename, figsize),
        'spectrogram': visualize_spectrogram_from_features(features, sr, common_hop_length, filename, figsize),
        'melspectrogram': visualize_melspectrogram_from_features(features, sr, common_hop_length, filename, figsize),
        'mfcc': visualize_mfcc_from_features(features, sr, common_hop_length, filename, figsize),
        'rms': visualize_rms_from_features(features, sr, common_hop_length, filename, figsize),
        'spectral_centroid': visualize_spectral_centroid_from_features(features, sr, common_hop_length, filename, figsize),
        'spectral_bandwidth': visualize_spectral_bandwidth_from_features(features, sr, common_hop_length, filename, figsize),
        'spectral_rolloff': visualize_spectral_rolloff_from_features(features, sr, common_hop_length, filename, figsize),
        'spectral_flatness': visualize_spectral_flatness_from_features(features, sr, common_hop_length, filename, figsize),
        'spectral_contrast': visualize_spectral_contrast_from_features(features, sr, common_hop_length, filename, figsize),
        'zero_crossing_rate': visualize_zero_crossing_rate_from_features(features, sr, filename, figsize)
    }

    return features, features_summary_statistics, graphs

def compare_audio_with_reference(participant_audio_path, reference_audio_path, sr=None, n_mfcc=16, n_fft=common_n_fft, hop_length=common_hop_length, generate_graph=False, figsize=(12, 8)):
    """
    Compare participant audio with reference audio using MFCC and DTW.
    
    Parameters:
    -----------
    participant_audio_path : str
        Path to participant's audio file
    reference_audio_path : str
        Path to reference audio sample
    sr : int
        Sample rate for audio loading
    n_mfcc : int
        Number of MFCC coefficients
    n_fft : int
        FFT window size
    hop_length : int
        Hop length for MFCC extraction
    generate_graph : bool, default False
        Whether to generate DTW warping path visualization
        
    Returns:
    --------
    If generate_graph=False:
        float: DTW distance (lower values indicate better similarity)
    If generate_graph=True:
        tuple: (DTW distance, fig, axes) where fig and axes are matplotlib objects
    """
    
    # Load both audio files
    y1, sr1 = librosa.load(participant_audio_path, sr=sr)
    y2, sr2 = librosa.load(reference_audio_path, sr=sr)
        
    # Extract MFCCs with consistent parameters
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, )
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    # Remove the first row (mean) from MFCCs    
    mfcc1_refined = mfcc1[1:, :]
    mfcc2_refined = mfcc2[1:, :]

    # Transpose to get time frames as rows and MFCC coefficients as columns    
    mfcc1_T = mfcc1_refined.T.astype(np.double)
    mfcc2_T = mfcc2_refined.T.astype(np.double)
    
    # Calculate DTW distance using the dtw module
    distance, paths = dtw_ndim.warping_paths(mfcc1_T, mfcc2_T, psi=3)
    
    if generate_graph:        
        # Import visualization function when needed        
        # Generate visualization
        fig, axes = visualize_dtw_warping_paths(mfcc1_T, 
                                                mfcc2_T, 
                                                paths=paths,
                                              title=f"Similarity: {similarity.distance_to_similarity(distance, method='exponential', r=r_common):.2f}",
                                              figsize=figsize)
        return similarity.distance_to_similarity(distance, method='exponential', r=r_common), fig
    else:
        return similarity.distance_to_similarity(distance, method='exponential', r=r_common), None

def compare_audios_with_each_other(audio_paths, sr=None, n_mfcc=16, n_fft=common_n_fft, hop_length=common_hop_length, parallel=False, graph=False):
    """
    Fast version using dtw_ndim.distance_matrix for batch computation.
    
    Parameters:
    -----------
    audio_paths : list
        List of paths to audio files
    sr : int, optional
        Sample rate for audio loading
    n_mfcc : int, default 16
        Number of MFCC coefficients
    n_fft : int
        FFT window size
    hop_length : int
        Hop length for MFCC extraction
    parallel : bool, default False
        Use parallel computation (disabled due to library issues)
        
    Returns:
    --------
    similarity_matrix : np.ndarray
        Matrix of similarity scores between all audio pairs
    average_similarity : float
        Average similarity score across all unique combinations
    audio_filenames : list
        List of filenames for reference
    fig: matplotlib.figure.Figure
        Heatmap visualization of the similarity matrix if graph=True, else None
    sd_similarity: float


    """
    
    if len(audio_paths) < 2:
        raise ValueError("Need at least 2 audio files to compute similarity")
    
    audio_filenames = [os.path.basename(path) for path in audio_paths]
    
    # Extract MFCC features for all audio files
    print("Extracting MFCC features...")
    mfcc_sequences = []
    
    for i, audio_path in enumerate(audio_paths):
        print(f"Processing {i+1}/{len(audio_paths)}: {audio_filenames[i]}")
        
        # Load audio
        y, sr_loaded = librosa.load(audio_path, sr=sr)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr_loaded, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        
        # Remove first row (mean) and transpose
        mfcc_refined = mfcc[1:, :]
        mfcc_T = mfcc_refined.T.astype(np.double)
        
        mfcc_sequences.append(mfcc_T)
    
    # Use manual pairwise computation for reliability
    print("Computing pairwise similarities...")
    n_files = len(audio_paths)
    similarity_matrix = np.zeros((n_files, n_files))
    
    # Set diagonal to 1.0 (perfect similarity with self)
    np.fill_diagonal(similarity_matrix, 1.0)
    
    similarity_scores = []
    
    for i in range(n_files):
        for j in range(i + 1, n_files):  # Only compute upper triangle
            print(f"  Computing similarity: {audio_filenames[i]} vs {audio_filenames[j]}")
            
            try:
                # Compute DTW distance
                distance, _ = dtw_ndim.warping_paths(mfcc_sequences[i], mfcc_sequences[j], psi=3)
                print(f"distance between {i} and {j} is {distance:.4f}")

                # Convert to similarity
                similarity_score = similarity.distance_to_similarity(distance, method='exponential', r=r_common)
                
                # Store in both positions (symmetric matrix)
                similarity_matrix[i, j] = similarity_score
                similarity_matrix[j, i] = similarity_score
                
                # Store for average calculation
                similarity_scores.append(similarity_score)
                
                print(f"    Similarity: {similarity_score:.4f}")
                
            except Exception as e:
                print(f"    Error computing similarity: {e}")
                # Set to 0 if computation fails
                similarity_matrix[i, j] = 0.0
                similarity_matrix[j, i] = 0.0
                similarity_scores.append(0.0)
    
    # Calculate average similarity
    average_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
    sd_similarity = np.std(similarity_scores) if similarity_scores else 0.0

    # print(f"\nAverage similarity across all unique pairs: {average_similarity:.4f}")
    
    if graph:
        # Generate heatmap visualization
        fig = create_similarity_heatmap(similarity_matrix, audio_filenames, average_similarity, sd_similarity=sd_similarity)
        return similarity_matrix, average_similarity, audio_filenames, fig, sd_similarity

    return similarity_matrix, average_similarity, audio_filenames, None, sd_similarity

def display_similarity_results(similarity_matrix, average_similarity, audio_filenames):
    """
    Display similarity results in a formatted way and return DataFrame + heatmap.
    
    Parameters:
    -----------
    similarity_matrix : np.ndarray
        Matrix of similarity scores
    average_similarity : float
        Average similarity score
    audio_filenames : list
        List of audio filenames
        
    Returns:
    --------
    similarity_df : pd.DataFrame
        DataFrame containing the similarity matrix
    fig : matplotlib.figure.Figure
        Heatmap visualization of the similarity matrix
    """
    
    import pandas as pd
    
    # Create a DataFrame for better visualization
    similarity_df = pd.DataFrame(
        similarity_matrix, 
        index=audio_filenames, 
        columns=audio_filenames
    )
    
    print("\n" + "="*80)
    print("AUDIO SIMILARITY ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nNumber of audio files: {len(audio_filenames)}")
    print(f"Average similarity: {average_similarity:.4f}")
    
    print(f"\nSimilarity Matrix:")
    print(similarity_df.round(4))
    
    # Find most and least similar pairs
    upper_triangle = np.triu(similarity_matrix, k=1)
    max_idx = np.unravel_index(np.argmax(upper_triangle), upper_triangle.shape)
    min_idx = np.unravel_index(np.argmin(upper_triangle + np.eye(len(audio_filenames))), upper_triangle.shape)
    
    print(f"\nMost similar pair:")
    print(f"  {audio_filenames[max_idx[0]]} ↔ {audio_filenames[max_idx[1]]}: {similarity_matrix[max_idx]:.4f}")
    
    print(f"\nLeast similar pair:")
    print(f"  {audio_filenames[min_idx[0]]} ↔ {audio_filenames[min_idx[1]]}: {similarity_matrix[min_idx]:.4f}")
    
    # Create heatmap visualization
    fig = create_similarity_heatmap(similarity_matrix, audio_filenames, average_similarity)
    
    return similarity_df, fig


def get_reference_audio_path(beatbox_sound, audio_samples_folder=r"C:\Users\Admin\Documents\SRIP Project 4_ Science of Human Voice Music Making - General\Audio Samples Preprocessed"):
    """
    Map beatbox sound to reference audio file path.
    
    Parameters:
    -----------
    beatbox_sound : str
        Name of the beatbox sound (e.g., 'Kick Imitate')
    audio_samples_folder : str
        Path to Audio Samples folder
        
    Returns:
    --------
    str: Path to reference audio file or None if not found
    """
    
    # Mapping of imitate sounds to reference audio files
    reference_mapping = {
        'Kick Imitate': '1AudioSample_Kick-preprocessed.wav',
        'Closed Hi-Hat Imitate': '2AudioSample_ClosedHiHat-preprocessed.wav',
        'Open Hi-Hat Imitate': '3AudioSample_OpenHiHat-preprocessed.wav',
        'Inward K Snare Imitate': '4AudioSample_KSnare-preprocessed.wav',
        'Rimshot Imitate': '5AudioSample_Rimshot-preprocessed.wav',
        'Spit Snare Imitate': '6AudioSample_SpitSnare-preprocessed.wav',
        'Pf Snare Imitate': '7AudioSample_PfSnare-preprocessed.wav',
        'Cymbal Crash Imitate': '8AudioSample_CymbalCrash-preprocessed.wav',
        'Descending Tom Sequence Imitate': '9AudioSample_Toms-preprocessed.wav',
        'Siren Imitate': '10AudioSample_Siren-preprocessed.wav',
        'Trumpet Imitate': '11AudioSample_Trumpet-preprocessed.wav',
        'Zipper Imitate': '12AudioSample_Zipper-preprocessed.wav',
        'Crab Scratch Bassy Imitate': '13aAudioSample_CrabLow-preprocessed.wav',
        'Crab Scratch High Pitch Imitate': '13bAudioSample_CrabHigh-preprocessed.wav',
        'Crab Scratch Meme Imitate': '13cAudioSample_CrabMeme-preprocessed.wav',
    }
    
    if beatbox_sound in reference_mapping:
        reference_file = reference_mapping[beatbox_sound]
        reference_path = os.path.join(audio_samples_folder, reference_file)
        
        if os.path.exists(reference_path):
            return reference_path
        else:
            print(f"Reference file not found: {reference_path}")
            return None
    else:
        # Not an imitate sound, return None
        return None

if __name__ == "__main__":
    # Example usage
    audio_path = "C:\\Users\\Admin\\Documents\\SRIP Project 4_ Science of Human Voice Music Making - General\\Audio Samples Preprocessed\\1AudioSample_Kick-preprocessed.wav"
    audio_path_2 = "C:\\Users\\Admin\\Documents\\SRIP Project 4_ Science of Human Voice Music Making - General\\Audio Samples Preprocessed\\2AudioSample_ClosedHiHat-preprocessed.wav"
    audio_path_3 = "C:\\Users\\Admin\\Documents\\SRIP Project 4_ Science of Human Voice Music Making - General\\Audio Samples Preprocessed\\3AudioSample_OpenHiHat-preprocessed.wav"
    audio_path_4 = "C:\\Users\\Admin\\Documents\\SRIP Project 4_ Science of Human Voice Music Making - General\\Audio Samples Preprocessed\\4AudioSample_KSnare-preprocessed.wav"

    # # Extract all features
    # features = get_all_spectral_features(audio_path)
    # features_2 = get_all_spectral_features(audio_path_2)

    # # Print feature shapes
    # for name, feature in features.items():
    #     print(f"{name}: {feature}")
    
    # for name, feature in features_2.items():
    #     print(f"{name}: {feature.shape}")
    # distance, fig, axes = compare_audio_with_reference(participant_audio_path=audio_path, reference_audio_path=audio_path_2, generate_graph=True, figsize=(12, 8))
    # plt.show()
    
    
    audio_paths = [audio_path, audio_path_2, audio_path_3, audio_path_4]

    sim_matrix, avg_sim, names, graph, sd_similarity = compare_audios_with_each_other(
        audio_paths, 
        parallel=False,  # Disabled parallel processing
        graph=True,
    )

    plt.show()
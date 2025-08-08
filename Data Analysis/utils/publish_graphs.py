import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

def create_similarity_heatmap(similarity_matrix, audio_filenames, average_similarity, sd_similarity=0, figsize=(10, 8)):
    """Create a simple heatmap visualization of the similarity matrix."""
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Clean labels
        clean_labels = [name.replace('.wav', '').replace('-preprocessed', '') for name in audio_filenames]
        
        # Create heatmap
        im = ax.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Similarity Score')
        
        # Add text annotations
        # Get the size of the matrix
        n = len(similarity_matrix)

        # --- Smart Annotation Logic ---
        # Only add text annotations if the matrix is not too dense to be readable.
        # A threshold of 20x20 is a good starting point.
        if n <= 22:
            # Dynamically calculate a font size. This simple formula decreases
            # the font size as the matrix gets larger.
            # We set a minimum size of 3 to prevent it from becoming invisible.
            font_size = max(3, 12 - 0.5 * n)

            # Add text annotations
            for i in range(n):
                for j in range(n):
                    text_color = "white" if similarity_matrix[i, j] < 0.5 else "black"
                    ax.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                            ha="center", va="center",
                            color=text_color,
                            fontsize=font_size)
                    ax.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                            ha="center", va="center",
                            color=text_color,
                            fontsize=font_size)
        
        # Set labels and title
        ax.set_xticks(range(len(clean_labels)))
        ax.set_yticks(range(len(clean_labels)))
        ax.set_xticklabels(clean_labels, rotation=45, ha='right')
        ax.set_yticklabels(clean_labels)
        ax.set_title(f'Audio Similarity Heatmap (Avg: {average_similarity:.3f}, SD: {sd_similarity:.3f})')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        return None


def visualize_chroma_cqt(features, sr, filename, figsize=(12, 6)):
    """Visualize Chroma CQT features."""
    try:
        fig, ax = plt.subplots(figsize=figsize)
        img = librosa.display.specshow(features['chroma_cqt'], y_axis='chroma', x_axis='time', 
                                     sr=sr, hop_length=512, ax=ax)
        ax.set_title(f'Chroma CQT: {filename}')
        plt.colorbar(img, ax=ax)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error visualizing chroma CQT: {e}")
        return None

def visualize_spectrogram_from_features(features, sr, hop_length, filename, figsize=(12, 6)):
    """Visualize spectrogram from extracted features."""
    try:
        fig, ax = plt.subplots(figsize=figsize)
        S_db = librosa.amplitude_to_db(features['spectrogram'], ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, 
                                     x_axis='time', y_axis='hz', ax=ax)
        ax.set_title(f'Spectrogram: {filename}')
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error visualizing spectrogram: {e}")
        return None

def visualize_melspectrogram_from_features(features, sr, hop_length, filename, figsize=(12, 6)):
    """Visualize mel-spectrogram from extracted features."""
    try:
        fig, ax = plt.subplots(figsize=figsize)
        M_db = librosa.power_to_db(features['melspectrogram'], ref=np.max)
        img = librosa.display.specshow(M_db, sr=sr, hop_length=hop_length,
                                     x_axis='time', y_axis='mel', ax=ax)
        ax.set_title(f'Mel-Spectrogram: {filename}')
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error visualizing mel-spectrogram: {e}")
        return None

def visualize_mfcc_from_features(features, sr, hop_length, filename, figsize=(12, 6)):
    """Visualize MFCC from extracted features."""
    try:
        fig, ax = plt.subplots(figsize=figsize)
        img = librosa.display.specshow(features['mfcc'], sr=sr, hop_length=hop_length,
                                     x_axis='time', ax=ax)
        ax.set_title(f'MFCC: {filename}')
        ax.set_ylabel('MFCC Coefficients')
        plt.colorbar(img, ax=ax)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error visualizing MFCC: {e}")
        return None

def visualize_rms_from_features(features, sr, hop_length, filename, figsize=(12, 6)):
    """Visualize RMS energy from extracted features."""
    try:
        fig, ax = plt.subplots(figsize=figsize)
        times = librosa.frames_to_time(range(features['rms'].shape[1]), sr=sr, hop_length=hop_length)
        ax.plot(times, features['rms'][0], label='RMS Energy')
        ax.set_title(f'RMS Energy: {filename}')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('RMS Energy')
        ax.legend()
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error visualizing RMS: {e}")
        return None

def visualize_spectral_centroid_from_features(features, sr, hop_length, filename, figsize=(12, 6)):
    """Visualize spectral centroid from extracted features."""
    try:
        fig, ax = plt.subplots(figsize=figsize)
        times = librosa.frames_to_time(range(features['spectral_centroid'].shape[1]), sr=sr, hop_length=hop_length)
        ax.plot(times, features['spectral_centroid'][0])
        ax.set_title(f'Spectral Centroid: {filename}')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Frequency (Hz)')
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error visualizing spectral centroid: {e}")
        return None

def visualize_spectral_bandwidth_from_features(features, sr, hop_length, filename, figsize=(12, 6)):
    """Visualize spectral bandwidth from extracted features."""
    try:
        fig, ax = plt.subplots(figsize=figsize)
        times = librosa.frames_to_time(range(features['spectral_bandwidth'].shape[1]), sr=sr, hop_length=hop_length)
        ax.plot(times, features['spectral_bandwidth'][0])
        ax.set_title(f'Spectral Bandwidth: {filename}')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Frequency (Hz)')
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error visualizing spectral bandwidth: {e}")
        return None

def visualize_spectral_rolloff_from_features(features, sr, hop_length, filename, figsize=(12, 6)):
    """Visualize spectral rolloff from extracted features."""
    try:
        fig, ax = plt.subplots(figsize=figsize)
        times = librosa.frames_to_time(range(features['spectral_rolloff'].shape[1]), sr=sr, hop_length=hop_length)
        ax.plot(times, features['spectral_rolloff'][0])
        ax.set_title(f'Spectral Rolloff: {filename}')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Frequency (Hz)')
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error visualizing spectral rolloff: {e}")
        return None

def visualize_spectral_flatness_from_features(features, sr, hop_length, filename, figsize=(12, 6)):
    """Visualize spectral flatness from extracted features."""
    try:
        fig, ax = plt.subplots(figsize=figsize)
        times = librosa.frames_to_time(range(features['spectral_flatness'].shape[1]), sr=sr, hop_length=hop_length)
        ax.plot(times, features['spectral_flatness'][0])
        ax.set_title(f'Spectral Flatness: {filename}')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Flatness')
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error visualizing spectral flatness: {e}")
        return None

def visualize_spectral_contrast_from_features(features, sr, hop_length, filename, figsize=(12, 6)):
    """Visualize spectral contrast from extracted features."""
    try:
        fig, ax = plt.subplots(figsize=figsize)
        img = librosa.display.specshow(features['spectral_contrast'], sr=sr, hop_length=hop_length,
                                     x_axis='time', ax=ax)
        ax.set_title(f'Spectral Contrast: {filename}')
        ax.set_ylabel('Frequency Bands')
        plt.colorbar(img, ax=ax)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error visualizing spectral contrast: {e}")
        return None

def visualize_zero_crossing_rate_from_features(features, sr, filename, figsize=(12, 6)):
    """Visualize zero crossing rate from extracted features."""
    try:
        fig, ax = plt.subplots(figsize=figsize)
        times = librosa.frames_to_time(range(features['zero_crossing_rate'].shape[1]), sr=sr, hop_length=512)
        ax.plot(times, features['zero_crossing_rate'][0])
        ax.set_title(f'Zero Crossing Rate: {filename}')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Zero Crossing Rate')
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error visualizing zero crossing rate: {e}")
        return None
    
def visualize_dtw_warping_paths(mfcc1_T, mfcc2_T, psi=3, paths=None, title="DTW Warping Paths", figsize=(12, 8)):
    """
    Visualize DTW warping paths between two MFCC sequences.
    
    Parameters:
    -----------
    mfcc1_T : np.ndarray
        First MFCC sequence (transposed)
    mfcc2_T : np.ndarray  
        Second MFCC sequence (transposed)
    paths: DTW paths, passed by `dtw_ndim.warping_paths`
    title : str
        Title for the plot
    figsize : tuple, default (12, 8)
        Figure size as (width, height)
        
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    try:
        from dtaidistance import dtw, dtw_ndim
        from dtaidistance import dtw_visualisation as dtwvis
        
        # Calculate DTW distance and paths
        best_path = dtw.best_path(paths)
        
        # Create visualization
        fig, axes = dtwvis.plot_warpingpaths(mfcc1_T, 
                                           mfcc2_T, 
                                           paths, 
                                           best_path,
                                           showlegend=True)
        fig.suptitle(title, fontsize=16)
        return fig, axes
        
    except Exception as e:
        print(f"Error visualizing DTW warping paths: {e}")
        return None, None
    
def visualize_waveform(path: str, max_points=11025, title=None, figsize=(12, 6)):
    """
    Loads an entire audio file and visualizes its waveform.

    Args:
        path (str): The file path to the audio file.
        max_points (int): Maximum points for waveshow adaptive rendering. Default: 11025
        title (str): Custom title for the plot. If None, uses filename. Default: None
        figsize (tuple): Size of the figure as (width, height). Default: (12, 6)

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    try:
        # 1. Load the entire audio file.
        # y is the audio data, sr is the sampling rate.
        y, sr = librosa.load(path, sr=None)

        # 2. Create a figure to draw on.
        fig, ax = plt.subplots(figsize=figsize)

        # 3. Plot the waveform.
        librosa.display.waveshow(y, sr=sr, max_points=max_points, alpha=1, ax=ax)

        # 4. Add a title and clean up the layout.
        if title is None:
            title = f'Waveform: {os.path.basename(path)}'
        ax.set_title(title)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        plt.tight_layout()

        # 5. Return the figure instead of showing it
        return fig

    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def visualize_multiple_waveforms(audio_paths, labels=None, colors=None,
                                max_points=11025, alpha=0.7, title="Multiple Waveforms Overlay", figsize=(12, 6)):
    """
    Visualize multiple waveforms overlaid on top of each other using librosa.display.waveshow.
    
    Parameters:
    -----------
    audio_paths : list of str
        List of file paths to audio files
    labels : list of str, optional
        Labels for each waveform. If None, uses filenames
    colors : list of str, optional  
        Colors for each waveform. If None, uses default matplotlib colors
    max_points : int, default 11025
        Maximum points for waveshow adaptive rendering
    alpha : float, default 0.7
        Transparency level (0.0 to 1.0)
    title : str
        Title for the plot
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    try:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Default colors if not provided
        if colors is None:
            colors = plt.cm.tab10(range(len(audio_paths)))
        elif len(colors) < len(audio_paths):
            # Repeat colors if not enough provided
            colors = (colors * ((len(audio_paths) // len(colors)) + 1))[:len(audio_paths)]
            
        # Default labels if not provided
        if labels is None:
            labels = [os.path.basename(path) for path in audio_paths]
        
        print(f"Loading and overlaying {len(audio_paths)} waveforms...")
        
        # Load and plot each waveform
        for i, (path, label, color) in enumerate(zip(audio_paths, labels, colors)):
            if not os.path.exists(path):
                print(f"Warning: File not found: {path}")
                continue
                
            try:
                # Load audio file (librosa automatically handles sampling rate)
                y, sr = librosa.load(path, sr=None)
                print(f"  {i+1}. {label}: {len(y):,} samples at {sr}Hz, duration: {len(y)/sr:.2f}s")
                
                # Plot waveform using librosa's adaptive waveshow
                librosa.display.waveshow(y, sr=sr, 
                                       max_points=max_points,
                                       alpha=alpha,
                                       color=color,
                                       label=label,
                                       ax=ax)
                                       
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        
        # Customize the plot
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_ylabel("Amplitude", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        return fig, ax
        
    except Exception as e:
        print(f"Error in visualize_multiple_waveforms: {e}")
        return None, None


def visualize_spectrogram(path: str, title=None, figsize=(12, 6), n_fft=512, hop_length=128):
    """
    Visualize a regular spectrogram from an audio file.
    
    Parameters:
    -----------
    path : str
        Path to the audio file
    title : str, optional
        Custom title for the plot
    figsize : tuple, default (12, 6)
        Figure size as (width, height)
    n_fft : int, default 512
        FFT window size
    hop_length : int, default 128
        Number of samples between successive frames
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    try:
        # Load audio file
        y, sr = librosa.load(path, sr=None)
        
        # Generate STFT (Short-Time Fourier Transform)
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        
        # Convert to dB scale
        stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        img = librosa.display.specshow(
            stft_db, 
            sr=sr, 
            hop_length=hop_length, 
            x_axis='time', 
            y_axis='hz',
            cmap='viridis',
            ax=ax
        )
        
        # Add colorbar and labels
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Frequency (Hz)')
        
        # Set title
        if title is None:
            title = f'Spectrogram: {os.path.basename(path)}'
        ax.set_title(title)
        
        plt.tight_layout()
        
        return fig, ax
        
    except Exception as e:
        print(f"Error visualizing spectrogram for {path}: {e}")
        return None, None


def visualize_melspectrogram(path: str, title=None, figsize=(12, 6), n_fft=512, hop_length=128, n_mels=128):
    """
    Visualize a mel-spectrogram from an audio file.
    
    Parameters:
    -----------
    path : str
        Path to the audio file
    title : str, optional
        Custom title for the plot
    figsize : tuple, default (12, 6)
        Figure size as (width, height)
    n_fft : int, default 512
        FFT window size
    hop_length : int, default 128
        Number of samples between successive frames
    n_mels : int, default 128
        Number of mel frequency bins
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    try:
        # Load audio file
        y, sr = librosa.load(path, sr=None)
        
        # Generate mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr,
            n_fft=n_fft, 
            hop_length=hop_length, 
            n_mels=n_mels
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        img = librosa.display.specshow(
            mel_spec_db, 
            sr=sr, 
            hop_length=hop_length, 
            x_axis='time', 
            y_axis='mel',
            cmap='viridis',
            ax=ax
        )
        
        # Add colorbar and labels
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Mel Frequency')
        
        # Set title
        if title is None:
            title = f'Mel-Spectrogram: {os.path.basename(path)}'
        ax.set_title(title)
        
        plt.tight_layout()
        
        return fig, ax
        
    except Exception as e:
        print(f"Error visualizing mel-spectrogram for {path}: {e}")
        return None, None

if __name__ == "__main__":
    # Example usage
    audio_files = [
        "C:\\Users\\Admin\\Documents\\SRIP Project 4_ Science of Human Voice Music Making - General\\Audio Samples\\1AudioSample_Kick.wav",
        "C:\\Users\\Admin\\Documents\\SRIP Project 4_ Science of Human Voice Music Making - General\\Audio Samples\\2AudioSample_ClosedHiHat.wav",
        "C:\\Users\\Admin\\Documents\\SRIP Project 4_ Science of Human Voice Music Making - General\\Audio Samples\\3AudioSample_OpenHiHat.wav"
    ]
    
    # Test overlay visualization
    print("Testing overlay visualization...")
    visualize_multiple_waveforms(audio_files, 
                                colors=['blue', 'red', 'green'],
                                title="Drum Samples Overlay")
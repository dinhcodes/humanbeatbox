import librosa
import numpy as np
import os
import soundfile as sf

def trim_audio(file_path):
    """
    Trim leading and trailing silence from short beatbox audio files.
    Extended by 64 samples before and after the detected boundaries.
    
    Parameters:
    -----------
    file_path : str
        Path to the audio file
        
    Returns:
    --------
    y_trimmed : np.ndarray
        The trimmed audio signal (extended by 64 samples on each side)
    original_length : int
        Length of original audio
    trimmed_length : int
        Length of trimmed audio
    trim_indices : tuple
        (start_index, end_index) of the trimmed portion
    """
    top_db = 20  # dB differentiation threshold for trimming silence
    frame_length = 128
    hop_length = 16
    padding_leading = 128  # Extension in samples before and after
    padding_trailing = 256
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)
        original_length = len(y)
        
        # Trim silence with parameters optimized for short beatbox sounds
        y_trimmed_raw, index = librosa.effects.trim(
            y, 
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        # Extend the boundaries by 64 samples before and after
        start_index = max(0, index[0] - padding_leading)
        end_index = min(len(y), index[1] + padding_trailing)
        
        # Extract the extended trimmed audio
        y_trimmed = y[start_index:end_index]
        trimmed_length = len(y_trimmed)
        
        return y_trimmed, original_length, trimmed_length, (start_index, end_index)
        
    except Exception as e:
        print(f"Error trimming audio {file_path}: {e}")
        # Return original audio if trimming fails
        y, sr = librosa.load(file_path, sr=None)
        return y, len(y), len(y), (0, len(y))
    
def rms_normalize(input_audio, target_dbfs=-20.0, max_peak=0.99):
    """
    Performs RMS-based normalization, but falls back to peak normalization
    if the signal would clip.

    Parameters:
    - input_audio: Either a file path (str) or audio array (np.ndarray)
    - target_dbfs (float): The target RMS level in dBFS.
    - max_peak (float): The maximum allowable peak amplitude (e.g., 0.99).

    Returns:
    - np.ndarray: The normalized audio signal.
    - str: The method used for normalization ('RMS', 'Peak', or 'Silent').
    """
    # Check if input is file path or audio array
    if isinstance(input_audio, str):
        # Load from file path
        y, sr = librosa.load(input_audio, sr=None)
    elif isinstance(input_audio, np.ndarray):
        # Use provided audio array
        y = input_audio
    else:
        raise ValueError("input_audio must be either a file path (str) or audio array (np.ndarray)")

    # Calculate the required scaling factor for RMS normalization
    initial_rms = np.sqrt(np.mean(y**2))

    # Handle silent audio
    if initial_rms == 0:
        return y, 'Silent'

    target_amp = 10**(target_dbfs / 20)
    rms_scaling_factor = target_amp / initial_rms

    # "Look ahead" to see if clipping would occur with RMS normalization
    predicted_peak = np.max(np.abs(y)) * rms_scaling_factor

    # If the predicted peak exceeds the maximum allowed, fall back to peak normalization
    if predicted_peak > max_peak:
        # Calculate the scaling factor for peak normalization
        peak_scaling_factor = max_peak / np.max(np.abs(y))
        y_normalized = y * peak_scaling_factor
        normalization_method = 'Peak'
    else:
        # Otherwise, proceed with the safe RMS normalization
        y_normalized = y * rms_scaling_factor
        normalization_method = 'RMS'

    return y_normalized, normalization_method

def batch_preprocess_all_audio(input_dir: str, output_dir: str) -> None:
    """
    Simple function to preprocess all audio files in Participant Audio Data folder.
    Creates new folder with same structure and "-preprocessed" filenames.
    """    
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    
    for root, dirs, files in os.walk(input_dir):
        # Create output directory structure
        rel_path = os.path.relpath(root, input_dir)
        out_dir = os.path.join(output_dir, rel_path)
        os.makedirs(out_dir, exist_ok=True)
        # files = files[:5] # Limit to first 5 files for testing

        # Process each audio file
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                input_file = os.path.join(root, file)
                
                # Create output filename with "-preprocessed"
                name, ext = os.path.splitext(file)
                output_file = os.path.join(out_dir, f"{name}-preprocessed{ext}")
                
                try:
                    print(f"Processing: {input_file}")
                    
                    # Step 1: Trim audio
                    y_trimmed, _, trimmed_length, _ = trim_audio(input_file)
                    
                    # Step 2: Normalize using the updated function with audio array
                    y_final, norm_method = rms_normalize(y_trimmed)
                    
                    # Step 3: Save (get sample rate from original file)
                    _, sr = librosa.load(input_file, sr=None, duration=0.1)  # Just get metadata
                    sf.write(output_file, y_final, sr)
                    print(f"Saved: {output_file} (method: {norm_method})")
                    
                except Exception as e:
                    print(f"Error processing {input_file}: {e}")


if __name__ == "__main__":
    # For Participant Audio Data Preprocessing
    # input_dir = r"c:\Users\Admin\Documents\SRIP Project 4_ Science of Human Voice Music Making - General\Participant Audio Data"
    # output_dir = r"c:\Users\Admin\Documents\SRIP Project 4_ Science of Human Voice Music Making - General\Participant Audio Data Preprocessed"

    input_dir = r"C:\Users\Admin\Documents\SRIP Project 4_ Science of Human Voice Music Making - General\Participant Audio Data"
    output_dir = r"C:\Users\Admin\Documents\SRIP Project 4_ Science of Human Voice Music Making - General\Participant Audio Data Preprocessed"

    batch_preprocess_all_audio(input_dir=input_dir, output_dir=output_dir)
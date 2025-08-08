import os

def get_sound_code(sound_name, beatbox_sounds):
    """
    Get the sound code from the beatbox_sounds dictionary.
    
    Args:
        sound_name (str): The key from beatbox_sounds dict (e.g., 'Kick Separate')
        beatbox_sounds (dict): The dictionary mapping sound names to codes.
    
    Returns:
        str: The sound code
        
    Raises:
        ValueError: If sound_name not found in beatbox_sounds
    """
    sound_code = beatbox_sounds.get(sound_name)
    if sound_code is None:
        raise ValueError(f"Sound name '{sound_name}' not found in beatbox_sounds.")
    return sound_code

def get_participant_folder_path(participant_number, base_dir='Participant Audio Data'):
    """
    Construct path to participant folder relative to project root.
    
    Args:
        participant_number (str or int): The participant's folder name
        base_dir (str): The base directory containing participant folders
        
    Returns:
        str: Full path to participant folder
        
    Raises:
        FileNotFoundError: If participant folder doesn't exist
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    participant_folder = os.path.join(project_root, base_dir, str(participant_number))
    
    if not os.path.isdir(participant_folder):
        raise FileNotFoundError(f"Participant folder not found at '{participant_folder}'")
        
    return participant_folder

def find_audio_files_in_phases(participant_folder, filenames):
    """
    Search for audio files across all phase folders.
    
    Args:
        participant_folder (str): Path to participant folder
        filenames (list): List of filenames to search for
        
    Returns:
        list: List of found file paths
    """
    found_files = []
    
    for phase_folder in os.listdir(participant_folder):
        phase_path = os.path.join(participant_folder, phase_folder)
        if os.path.isdir(phase_path):
            for filename in filenames:
                file_path = os.path.join(phase_path, filename)
                if os.path.isfile(file_path):
                    found_files.append(file_path)
    
    return found_files


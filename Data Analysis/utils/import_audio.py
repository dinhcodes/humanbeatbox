import os
import sys

# Add the utils directory to the path to enable imports
utils_dir = os.path.dirname(os.path.abspath(__file__))
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

# Now use absolute imports instead of relative imports
from audio_helpers import get_sound_code, get_participant_folder_path, find_audio_files_in_phases
from constants import beatbox_sounds, participants

# Constants
MAX_ATTEMPTS = 10
BASE_DIR = 'Participant Audio Data Preprocessed'
SUFFIX = '-preprocessed'

def get_participant_audio(participant_number, sound_name, attempt_number, beatbox_sounds):
    """
    Returns the file path for a given participant number, sound name, and attempt number.
    
    Args:
        participant_number (str or int): The participant's folder name (e.g., '1', '2', ...)
        sound_name (str): The key from beatbox_sounds dict (e.g., 'Kick Separate')
        attempt_number (int): The attempt number
        beatbox_sounds (dict): The dictionary mapping sound names to codes.

    Returns:
        str: The file path to the audio file
    """
    
    # Get the sound code using helper function
    sound_code = get_sound_code(sound_name, beatbox_sounds)
    
    # Get participant folder path using helper function
    participant_folder = get_participant_folder_path(participant_number, BASE_DIR)

    # Create filename (always with attempt number)
    filename = f"{participant_number}-{sound_code}-{attempt_number}{SUFFIX}.wav"

    # Search for the specific file using helper function
    found_files = find_audio_files_in_phases(participant_folder, [filename])
    
    if not found_files:
        raise FileNotFoundError(f"Audio file '{filename}' not found for participant {participant_number} in any phase folder.")
    
    return found_files[0]  # Return the first (and only) found file


def get_all_audios_from_participant_number_and_sound_name(participant_number, sound_name, beatbox_sounds):
    """
    Returns a list of all audio file paths for a given participant and sound name.
    
    Args:
        participant_number (str or int): The participant's folder name (e.g., '1', '2', ...)
        sound_name (str): The key from beatbox_sounds dict (e.g., 'Kick Separate')
        beatbox_sounds (dict): The dictionary mapping sound names to codes.

    Returns:
        list: List of file paths to all found audio files
    """
    
    # Get the sound code using helper function
    sound_code = get_sound_code(sound_name, beatbox_sounds)
    
    # Get participant folder path using helper function
    participant_folder = get_participant_folder_path(participant_number, BASE_DIR)
    
    # Generate all possible filenames for attempts 1 to max_attempts
    possible_filenames = []
    for attempt_number in range(1, MAX_ATTEMPTS + 1):
        filename = f"{participant_number}-{sound_code}-{attempt_number}{SUFFIX}.wav"
        possible_filenames.append(filename)
    
    # Search for files using helper function
    found_files = find_audio_files_in_phases(participant_folder, possible_filenames)
    
    if not found_files:
        raise FileNotFoundError(f"No audio files found for participant {participant_number} and sound '{sound_name}' (attempts 1-{MAX_ATTEMPTS})")
    
    return found_files


def get_all_audios_from_sound_name(sound_name, beatbox_sounds, participants):
    """
    Returns a list of all audio file paths for a given sound name across all participants.
    
    Args:
        sound_name (str): The key from beatbox_sounds dict (e.g., 'Kick Separate')
        beatbox_sounds (dict): The dictionary mapping sound names to codes.
        participants (list): List of participant numbers to search through.

    Returns:
        list: List of file paths to all found audio files across all participants
    """
    
    all_found_files = []
    
    # Iterate through each participant and collect their audio files
    for participant_number in participants:
        try:
            # Use existing function to get all files for this participant and sound
            participant_files = get_all_audios_from_participant_number_and_sound_name(
                participant_number, sound_name, beatbox_sounds
            )
            all_found_files.extend(participant_files)
            
        except FileNotFoundError:
            # Skip participants who don't have files for this sound
            continue
    
    if not all_found_files:
        raise FileNotFoundError(f"No audio files found for sound '{sound_name}' across any participants (attempts 1-{MAX_ATTEMPTS})")
    
    return all_found_files


if __name__ == "__main__":
    # Example usage    
    # Test single audio file
    particular_file = get_participant_audio(1, 'Kick Separate', 1, beatbox_sounds)
    print(f"Found audio file: {particular_file}")
    
    # Test all audio files for a participant/sound combination
    all_files = get_all_audios_from_participant_number_and_sound_name(1, 'Kick Separate', beatbox_sounds)
    print(f"Found {len(all_files)} audio files for participant 1:")
    for file_path in all_files:
        print(f"  {file_path}")
    
    # Test all audio files for a sound across all participants
    all_sound_files = get_all_audios_from_sound_name('Kick Separate', beatbox_sounds, participants)
    print(f"Found {len(all_sound_files)} 'Kick Separate' files across all participants:")
    for file_path in all_sound_files:
        print(f"  {file_path}")
    
    
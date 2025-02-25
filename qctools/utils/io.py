import os
import uuid


# Function to ensure 'tmp' directory exists
def ensure_directory(tmp: str='tmp') -> str:
    tmp_dir = os.path.join(os.getcwd(), tmp)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    return tmp_dir

# Function to generate a unique file name
def generate_unique_file_path(directory, extension: str=".txt") -> str:
    while True:
        file_name = f"{uuid.uuid4()}{extension}"
        file_path = os.path.join(directory, file_name)
        if not os.path.exists(file_path):
            return file_path
        
def setup_run_directory(tmp: str="tmp"):
    """Create a temporary run-specific directory."""
    # Get the current directory path
    # current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    current_dir = os.getcwd()

    # Create 'tmp' directory if it doesn't exist
    tmp_dir = os.path.join(current_dir, tmp)
    os.makedirs(tmp_dir, exist_ok=True)

    # Create a unique subdirectory for the current run
    run_dir = os.path.join(tmp_dir, f"run_{uuid.uuid4().hex}")
    os.makedirs(run_dir)

    return run_dir
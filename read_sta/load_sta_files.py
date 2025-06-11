import os
import glob
from tqdm import tqdm
# ------------------------------------------------------------------------------------- 
# STA Loader.
#
# Loads all STA files from a specified directory and its subdirectories.
#
# Usage:
#   python load_sta_files.py
# -------------------------------------------------------------------------------------
def find_sta_files(directory):
    """Recursively find all STA files in the given directory."""
    pattern = os.path.join(directory, '**', '*.sta')
    sta_files = glob.glob(pattern, recursive=True)
    return sta_files

def load_sta_files(sta_files):
    """Load all STA files into a list of strings with a progress bar."""
    data = []
    for file in tqdm(sta_files, desc="Loading STA files"):
        with open(file, 'r') as f:
            data.append(f.read())
    return data

def print_loaded_files(sta_files):
    """Print the list of loaded STA files."""
    n = len(sta_files)
    for file in sta_files:
        print(f"Loaded: {file}")
    print(f"Total files loaded: {n}")

def main():
    """Main driver function."""
    sta_directory = 'sta_files'
    sta_files = find_sta_files(sta_directory)
    data = load_sta_files(sta_files)
    print_loaded_files(sta_files)

if __name__ == "__main__":
    main()
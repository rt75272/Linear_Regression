import os
import glob
from tqdm import tqdm
import concurrent.futures
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

def read_sta_file(file):
    """Helper function to read a single STA file."""
    with open(file, 'r') as f:
        return f.read()

def load_sta_files(sta_files, n):
    """Load all STA files into a list with a progress bar and multithreading."""
    data = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = executor.map(read_sta_file, sta_files)
        results = list(tqdm(futures, total=n, desc="Loading STA files"))
        data.extend(results)
    return data

def print_loaded_files(sta_files, n):
    """Print the list of loaded STA files."""
    for file in sta_files:
        print(f"Loaded: {file}")
    print(f"Total files loaded: {n}")

def main():
    """Main driver function."""
    sta_directory = 'sta_files'
    sta_files = find_sta_files(sta_directory)
    n = len(sta_files)
    data = load_sta_files(sta_files)
    print_loaded_files(sta_files, n)

if __name__ == "__main__":
    main()
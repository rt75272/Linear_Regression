import os
import glob
import pandas as pd
from tqdm import tqdm
import concurrent.futures
# ------------------------------------------------------------------------------------- 
# CSV Loader.
#
# Loads all CSV files from a specified directory and its subdirectories.
#
# Usage:
#   python load_csv_files.py
# -------------------------------------------------------------------------------------
def find_csv_files(directory):
    """Recursively find all CSV files in the given directory."""
    pattern = os.path.join(directory, '**', '*.csv')
    csv_files = glob.glob(pattern, recursive=True)
    return csv_files

def load_csv_files(csv_files, n):
    """Load all CSV files into DataFrames with a progress bar and multithreading."""
    dataframes = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = executor.map(pd.read_csv, csv_files)
        results = list(tqdm(futures, total=n, desc="Loading CSV files"))
        dataframes.extend(results)
    return dataframes

def print_loaded_files(csv_files, n):
    """Print the list of loaded CSV files."""
    for file in csv_files:
        print(f"Loaded: {file}")
    print(f"Total files loaded: {n}")

def main():
    """Main driver function."""
    csv_directory = 'csv_files'
    csv_files = find_csv_files(csv_directory)
    n = len(csv_files)
    dataframes = load_csv_files(csv_files, n)
    print_loaded_files(csv_files, n)

# Big red activation button.
if __name__ == "__main__":
    main()

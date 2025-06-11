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

def print_data_info(dataframes):
    """Print important information about the loaded CSV data."""
    print("\n--- CSV Data Information ---")
    print(f"Number of CSV files loaded: {len(dataframes)}")
    if dataframes:
        print(f"First DataFrame shape: {dataframes[0].shape}")
        print(f"First DataFrame columns: {list(dataframes[0].columns)}")
        print(f"Preview of first DataFrame:\n{dataframes[0].head()}")
    print("----------------------------")

def main():
    """Main driver function."""
    csv_directory = 'csv_files'
    csv_files = find_csv_files(csv_directory)
    n = len(csv_files)
    dataframes = load_csv_files(csv_files, n)
    print_loaded_files(csv_files, n)
    print_data_info(dataframes)

# Big red activation button.
if __name__ == "__main__":
    main()

import os
import glob
import math
import psutil
import pandas as pd
import concurrent.futures
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
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

def get_max_workers():
    """Determine max_workers based on available RAM (1 worker per 1GB, at least 1)."""
    total_gb = math.ceil(psutil.virtual_memory().total / (1024**3))
    print(f"Total RAM: {total_gb} GB Available")
    max_workers = max(1, total_gb)
    return max_workers

def load_csv_files(csv_files, n):
    """Load all CSV files into DataFrames with a progress bar and multithreading."""
    max_workers = get_max_workers()
    dataframes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(pd.read_csv, csv_files)
        results = list(tqdm(futures,
                        total=n, 
                        desc=f"Loading CSV files ({max_workers} workers)"))
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

def linear_regression_first_dataframe(dataframes):
    """Performs linear regression using all columns except the last as features,
    and the last as target. Splits data into training and testing sets. 
    Plots results and outputs model accuracy."""
    if not dataframes:
        print("No DataFrames available for regression.")
        return
    df = dataframes[0]
    if df.shape[1] < 2:
        print("Not enough columns for regression.")
        return
    X = df.iloc[:, :-1].values # All columns except the last.
    y = df.iloc[:, -1].values # Last column as target.
    # Split into training and testing sets (70% train, 30% test).
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.3, 
                                                        random_state=42)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"Linear Regression Coefficients: {model.coef_}")
    print(f"Linear Regression Intercept: {model.intercept_}")
    y_pred = model.predict(X_test)
    # Output model accuracy (R^2 score).
    accuracy = model.score(X_test, y_test)
    print(f"Linear Regression Model R^2 Accuracy (Test Set): {accuracy:.4f}")
    if X.shape[1] == 1:
        # One feature: plot regression line on test data.
        plt.scatter(X_test, y_test, color='blue', label='Actual (Test)')
        plt.plot(X_test, y_pred, color='red', label='Regression Line (Test)')
        plt.xlabel(df.columns[0])
        plt.ylabel(df.columns[-1])
        plt.title("Linear Regression Fit (Single Feature, Test Data)")
        plt.legend()
        plt.show()
    else:
        # Multiple features: plot predicted vs actual for test data.
        plt.scatter(y_test, y_pred, color='purple', alpha=0.75)
        plt.plot([y_test.min(), 
                  y_test.max()], 
                  [y_test.min(), 
                   y_test.max()], 
                   'r--', label='Ideal Fit')
        plt.xlabel("Actual Values (Test)")
        plt.ylabel("Predicted Values (Test)")
        plt.title("Linear Regression: Predicted vs Actual (Test Data)")
        plt.legend()
        plt.show()

def main():
    """Main driver function."""
    csv_directory = 'csv_files'
    csv_files = find_csv_files(csv_directory)
    n = len(csv_files)
    dataframes = load_csv_files(csv_files, n)
    # print_loaded_files(csv_files, n)
    # print_data_info(dataframes)
    linear_regression_first_dataframe(dataframes)

# Big red activation button.
if __name__ == "__main__":
    main()

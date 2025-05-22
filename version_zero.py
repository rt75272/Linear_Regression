import argparse
import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore
# --------------------------------------------------------------------------------------
# Basic Linear Regression Implementation.
# 
# This script loads a generic dataset (CSV), splits it into training and testing sets,
# implements simple linear regression, evaluates its performance, and visualizes the 
# results.
# 
# Usage:
#     python version_zero.py --data <dataset.csv> --target <target_column_name>
# --------------------------------------------------------------------------------------
def load_data(filepath="example_data.csv"):
    """Loads a CSV dataset file."""
    dataset = pd.read_csv(filepath)
    print(f"Loaded dataset with shape: {dataset.shape}")
    return dataset

class LinearRegression:
    """Linear Regression supporting multiple features."""
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """Fits the linear regression model to data using the normal equation."""
        X = np.asarray(X)
        y = np.asarray(y)
        # Input validation.
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Input contains NaN values.")
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            raise ValueError("Input contains infinite values.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y do not match.")
        # Add intercept if needed.
        if self.fit_intercept:
            X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_b = X
        # Check for singularity.
        cond_number = np.linalg.cond(X_b.T @ X_b)
        if cond_number > 1e12:
            print("Warning: Design matrix is close to singular. Results may not be reliable.")
        # Normal equation.
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta

    def predict(self, X):
        """Predicts using the linear regression model."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.fit_intercept:
            return X @ self.coef_ + self.intercept_
        else:
            return X @ self.coef_

def main():
    """Main driver function to run the linear regression model."""
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Linear Regression")
    parser.add_argument('--data', type=str, default="example_data.csv", help='data')
    parser.add_argument('--target', type=str, default="target", help='Target column')
    args = parser.parse_args()
    df = load_data(args.data) # Load data.
    # Separate features and target.
    X = df.drop(columns=[args.target]).values
    y = df[args.target].values
    # Split into train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    model = LinearRegression(fit_intercept=True) # Init the linear regression model.
    model.fit(X_train, y_train) # Train the linear regression model.
    y_pred = model.predict(X_test) # Predict on test set.
    # Evaluate the model and display the results.
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    # Visualize predictions vs actual (for single target).
    plt.scatter(y_test, y_pred, alpha=0.7, label="Test Data Point")
    y_min = y_test.min()
    y_max = y_test.max()
    plt.plot([y_min, y_max], [y_min, y_max], 'r--', label="Ideal Fit")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Linear Regression: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.show()

# The big red activation button.
if __name__=="__main__":
    main() # Running the main driver function.
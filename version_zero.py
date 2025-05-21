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
    """Linear Regression class/object to work with multiple features."""
    def __init__(self):
        """Constructor for the LinearRegression class."""
        self.coef_ = None  # Coefficients (weights).
        self.intercept_ = None  # Intercept (bias).

    def fit(self, X, y):
        """Fits the linear regression model to data using the normal equation."""
        # Add a column of ones to X for the intercept term
        X_b = np.hstack([np.ones((X.shape[0], 1)), X]) # X_b is the design matrix.
        # Normal equation: theta = (X^T * X)^(-1) * X^T * y.
        X_b_T = X_b.T # X_b_T is the transpose of X_b.
        X_b_T_X_b = X_b_T @ X_b # X_b_T_X_b is the dot product of X_b_T and X_b.
        X_b_T_X_b_inv = np.linalg.pinv(X_b_T_X_b)# X_b_T_X_b_inv is the pseudo-inverse.
        X_b_T_y = X_b_T @ y # X_b_T_y is the dot product of X_b_T and y.
        theta = X_b_T_X_b_inv @ X_b_T_y # The theta is the vector of coefficients.
        self.intercept_ = theta[0] # The intercept is the first element of theta. 
        self.coef_ = theta[1:] # And the rest are the coefficients.

    def predict(self, X):
        """Generates and returnsd a prediction using the linear regression model."""
        y_pred = X @ self.coef_ # y_pred is the dot product of X and the coefficients.
        y_pred = y_pred + self.intercept_ # Add the intercept to the predictions.
        return y_pred 

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
    model = LinearRegression() # Initialize the linear regression model.
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
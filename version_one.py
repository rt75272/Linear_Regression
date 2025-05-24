import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# --------------------------------------------------------------------------------------
# Polynomial Linear Regression Implementation.
# 
# Linear regression with polynomial features using the normal equation. Which is 
# capable to multiple features.
# 
# Usage:
#     python version_one.py --data <dataset.csv> --target <target_column_name>
# --------------------------------------------------------------------------------------
class PolynomialLinearRegression:
    """Polynomial Linear Regression using the normal equation.
    Supports fitting polynomial features up to a specified degree.
    """
    def __init__(self, degree=2, fit_intercept=True):
        self.degree = degree
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def _poly_features(self, X):
        """Generate polynomial features up to the given degree.
        Args:
            X (np.ndarray): Input features (n_samples, n_features)
        Returns:
            np.ndarray: Expanded feature matrix with polynomial terms
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        features = [np.ones(n_samples)] if self.fit_intercept else []
        # Add polynomial features for each degree and feature
        for d in range(1, self.degree + 1):
            for i in range(n_features):
                features.append(X[:, i] ** d)
        return np.vstack(features).T

    def fit(self, X, y):
        """Fit the polynomial regression model using the normal equation.
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
        """
        X_poly = self._poly_features(X)
        # Solve for theta using the pseudo-inverse (robust to singular matrices)
        theta = np.linalg.pinv(X_poly.T @ X_poly) @ X_poly.T @ y
        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta

    def predict(self, X):
        """Predict target values using the polynomial regression model.
        Args:
            X (np.ndarray): Input features
        Returns:
            np.ndarray: Predicted values
        """
        X_poly = self._poly_features(X)
        if self.fit_intercept:
            return X_poly @ np.concatenate(([self.intercept_], self.coef_))
        else:
            return X_poly @ self.coef_

def load_data(filepath="example_data.csv"):
    """Loads a CSV dataset file.
    Args:
        filepath (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    dataset = pd.read_csv(filepath)
    print(f"Loaded dataset with shape: {dataset.shape}")
    return dataset

def main():
    """Main driver function to run the polynomial regression model.
    Handles argument parsing, data loading, training, evaluation, and visualization.
    """
    parser = argparse.ArgumentParser(description="Polynomial Linear Regression")
    parser.add_argument('--data', type=str, default="example_data.csv", help='Dataset')
    parser.add_argument('--target', type=str, default="target", help='Target column')
    parser.add_argument('--degree', type=int, default=2, help='Polynomial degree')
    args = parser.parse_args()
    df = load_data(args.data) # Load data from CSV file.
    # Separate features and target.
    X = df.drop(columns=[args.target]).values
    y = df[args.target].values
    # Split into train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    # Initialize and train the polynomial regression model.
    model = PolynomialLinearRegression(degree=args.degree)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Evaluate the model and display the results.
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    # Visualization for single feature.
    if X.shape[1] == 1:
        x_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_plot = model.predict(x_plot)
        plt.scatter(X_test, y_test, color='blue', label='Test Data')
        plt.plot(x_plot, y_plot, color='red', label='Polynomial Fit')
        plt.xlabel("Feature")
        plt.ylabel("Target")
        plt.title(f"Polynomial Regression (degree={args.degree})")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        # For multi-feature, plot predicted vs actual.
        plt.scatter(y_test, y_pred, alpha=0.7, label="Test Data Point")
        y_min = y_test.min()
        y_max = y_test.max()
        plt.plot([y_min, y_max], [y_min, y_max], 'r--', label="Ideal Fit")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Polynomial Regression: Actual vs Predicted")
        plt.legend()
        plt.tight_layout()
        plt.show()

# The Big red activation button.
if __name__ == "__main__":
    main() # Running the main driver function.
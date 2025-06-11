import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
# ------------------------------------------------------------------------------------- 
# Linear Regression Comparison. 
#
# This script generates synthetic multivariate linear data, trains four different 
# regression models (Linear, Lasso, Ridge, and Polynomial), and compares their 
# performance.
#
# Usage:
#   python linear_compare.py
# -------------------------------------------------------------------------------------
def generate_data(seed=0, n_samples=442, n_features=3):
    """Generates synthetic multivariate linear data with noise."""
    np.random.seed(seed)
    X = 2 * np.random.rand(n_samples, n_features)
    true_coefs = np.array([3, -2, 1])  # True coefficients for the linear model.
    y = 4 + X @ true_coefs + np.random.randn(n_samples)  # Linear relation with noise.
    return X, y

def train_models(X_train, y_train, degree=2):
    """Trains Linear, Lasso, Ridge, and Polynomial regression models on the training 
    data. Returns the fitted models and the polynomial transformer."""
    lin_reg = LinearRegression()
    lasso_reg = Lasso(alpha=0.1)
    ridge_reg = Ridge(alpha=1.0)
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    poly_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    lasso_reg.fit(X_train, y_train)
    ridge_reg.fit(X_train, y_train)
    poly_reg.fit(X_train_poly, y_train)
    return lin_reg, lasso_reg, ridge_reg, (poly_reg, poly)

def evaluate_models(models, X_test, y_test):
    """Evaluates the given models on the test data. Returns predictions and mean 
    squared errors for each model."""
    lin_reg, lasso_reg, ridge_reg, (poly_reg, poly) = models
    y_pred_lin = lin_reg.predict(X_test)
    y_pred_lasso = lasso_reg.predict(X_test)
    y_pred_ridge = ridge_reg.predict(X_test)
    X_test_poly = poly.transform(X_test)
    y_pred_poly = poly_reg.predict(X_test_poly)
    y_preds = [y_pred_lin, y_pred_lasso, y_pred_ridge, y_pred_poly]
    mses = [mean_squared_error(y_test, y_pred) for y_pred in y_preds]
    return y_preds, mses

def plot_mse_comparison(mses):
    """Plots a bar chart comparing the mean squared errors of the models."""
    models = ['Linear', 'Lasso', 'Ridge', 'Poly']
    plt.figure(figsize=(8, 5))
    plt.bar(models, mses, color=['blue', 'green', 'red', 'purple'])
    plt.title("MSE Comparison of Regression Models on Multivariate Data")
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Model")
    plt.show()

def plot_predictions(X_test, y_test, y_preds):
    """Plots the actual test data and the prediction lines for each model,
    using the first feature for the x-axis."""
    # Sort by the first feature for smooth lines.
    sorted_idx = np.argsort(X_test[:, 0])
    X_sorted = X_test[sorted_idx, 0]
    y_actual_sorted = y_test[sorted_idx]
    y_pred_lin_sorted = y_preds[0][sorted_idx]
    y_pred_lasso_sorted = y_preds[1][sorted_idx]
    y_pred_ridge_sorted = y_preds[2][sorted_idx]
    y_pred_poly_sorted = y_preds[3][sorted_idx]
    # Plotting.
    plt.figure(figsize=(10, 6))
    plt.scatter(
        X_sorted, y_actual_sorted,
        color='black', label='Actual', alpha=0.7
    )
    plt.plot(
        X_sorted, y_pred_lin_sorted,
        color='blue', label='Linear Prediction',
        linewidth=2
    )
    plt.plot(
        X_sorted, y_pred_lasso_sorted,
        color='green', label='Lasso Prediction',
        linewidth=2
    )
    plt.plot(
        X_sorted, y_pred_ridge_sorted,
        color='red', label='Ridge Prediction',
        linewidth=2
    )
    plt.plot(
        X_sorted, y_pred_poly_sorted,
        color='purple', label='Poly Prediction',
        linewidth=2, linestyle='dashed'
    )
    plt.title("Model Predictions vs Actual (Feature 1)")
    plt.xlabel("Feature 1 Value")
    plt.ylabel("Target Value")
    plt.legend()
    plt.show()

def main():
    """Main driver function for data generation, training, evaluation, and plotting."""
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42)
    models = train_models(X_train, y_train)
    y_preds, mses = evaluate_models(models, X_test, y_test)
    plot_mse_comparison(mses)
    plot_predictions(X_test, y_test, y_preds)
    # Print out the top performing model.
    model_names = ['Linear', 'Lasso', 'Ridge', 'Poly']
    best_idx = np.argmin(mses)
    print(
        f"\nTop performing model: {model_names[best_idx]} Regression "
        f"(MSE = {mses[best_idx]:.4f})")
    # Print out the MSE score for all models.
    print("\nModel Mean Squared Error (MSE) on test set:")
    for name, mse in zip(model_names, mses):
        print(f"{name} Regression: MSE = {mse:.4f}")

# The big red activation button.
if __name__ == "__main__":
    main()

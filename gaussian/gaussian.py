import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
# ------------------------------------------------------------------------------------- 
# Gaussian Function.
#
# Demonstrates how to use Gaussian features for regression tasks.
#
# Usage:
#   python gaussian.py
# -------------------------------------------------------------------------------------
class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input."""
    def __init__(self, N, width_factor=2.0):
        self.N = N  # Number of Gaussian features.
        self.width_factor = width_factor  # Factor to control the width of Gaussians.

    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        """Compute the Gaussian basis function."""
        arg = (x - y) / width
        exponent = -0.5 * np.sum(arg**2, axis)
        result = np.exp(exponent)
        return result

    def fit(self, X, y=None):
        """Fit the Gaussian features to the input data X."""
        self.centers_ = np.linspace(X.min(), X.max(), self.N) # Create N centers.
        # Set the width of the Gaussians based on the spacing of centers.
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self

    def transform(self, X):
        """Transform input data X into Gaussian features."""
        X_expanded = X[:, :, np.newaxis] # Transform input data X into Gaussian.
        gauss_features = self._gauss_basis(
            X_expanded, 
            self.centers_, 
            self.width_, axis=1)
        return gauss_features

def generate_data(seed=0, n_points=100):
    """Generate noisy sine wave data."""
    np.random.seed(seed)
    x = np.linspace(0, 10, n_points)
    y = np.sin(x) + 0.1 * np.random.randn(n_points) # Sine wave with noise.
    return x, y

def fit_gaussian_model(x, y, n_features=20):
    """Fit a linear regression model using Gaussian features."""
    gauss_model = make_pipeline(
        GaussianFeatures(n_features), # Transform input with Gaussian features.
        LinearRegression()) # Fit linear regression on transformed features.
    gauss_model.fit(x[:, np.newaxis], y)
    return gauss_model

def plot_results(x, y, xfit, yfit):
    """Plot the original data and the Gaussian fit."""
    plt.scatter(x, y, color='blue', label='Data')
    plt.plot(xfit, yfit, color='red', label='Gaussian fit')
    plt.xlim(0, 10)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gaussian Feature Fit Example')
    plt.show()

def main():
    """Main driver function to generate data, fit the model, and plot results."""
    x, y = generate_data() # Generate synthetic data.
    gauss_model = fit_gaussian_model(x, y) # Fit the Gaussian feature model.
    xfit = np.linspace(0, 10, 1000) # Create a dense grid for predictions.
    yfit = gauss_model.predict(xfit[:, np.newaxis])
    plot_results(x, y, xfit, yfit)

# The big red activation button.
if __name__=="__main__":
    main()
# core/differential_geom/auto_diff.py

"""
A simplified module for manifold-aware automatic differentiation.
This module uses a finite-difference approach as a placeholder for integrating with
more advanced AD frameworks (e.g., JAX, PyTorch) that support manifold constraints.
"""

import numpy as np

def manifold_gradient(func, point, manifold_metric_fn, epsilon=1e-6):
    """
    Compute a finite-difference gradient of 'func' at 'point', taking into account the manifold's metric.
    
    Parameters:
      - func: callable, function from ℝ^n to ℝ.
      - point: numpy array, the point on the manifold.
      - manifold_metric_fn: function that returns the metric tensor at a given point.
      - epsilon: step size for finite differences.
    
    Returns:
      - adjusted_grad: gradient vector adjusted by the inverse metric tensor.
    """
    grad = np.zeros_like(point)
    for i in range(len(point)):
        delta = np.zeros_like(point)
        delta[i] = epsilon
        f_plus = func(point + delta)
        f_minus = func(point - delta)
        grad[i] = (f_plus - f_minus) / (2 * epsilon)
    
    metric = manifold_metric_fn(point)
    inv_metric = np.linalg.inv(metric)
    adjusted_grad = inv_metric @ grad
    return adjusted_grad

# Example usage
if __name__ == "__main__":
    # A simple function on ℝ²: f(x, y) = x² + 3y².
    def f(x):
        return x[0]**2 + 3*x[1]**2

    def metric_fn(x):
        # For Euclidean space, the metric is the identity.
        return np.eye(len(x))
    
    point = np.array([1.0, 2.0])
    grad = manifold_gradient(f, point, metric_fn)
    print("Gradient at", point, "is", grad)

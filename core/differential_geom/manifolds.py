# core/differential_geom/manifolds.py

import numpy as np
from abc import ABC, abstractmethod

class Manifold(ABC):
    """
    Abstract base class for a manifold.
    A manifold is a space that locally resembles Euclidean space.
    """

    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the manifold."""
        pass

    @abstractmethod
    def metric(self, point: np.ndarray) -> np.ndarray:
        """
        Return the metric tensor at a given point on the manifold.
        """
        pass

class EuclideanManifold(Manifold):
    """
    A simple Euclidean manifold.
    """

    def __init__(self, dim: int):
        self._dim = dim

    def dimension(self) -> int:
        return self._dim

    def metric(self, point: np.ndarray = None) -> np.ndarray:
        # In Euclidean space, the metric is the identity matrix.
        return np.eye(self._dim)

# Example usage
if __name__ == "__main__":
    manifold = EuclideanManifold(3)
    print("Dimension:", manifold.dimension())
    print("Metric:\n", manifold.metric())

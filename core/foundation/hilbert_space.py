# core/foundation/hilbert_space.py

import numpy as np
from abc import ABC, abstractmethod

class HilbertSpace(ABC):
    """
    Abstract base class for Hilbert Spaces.
    Represents the state space for a quantum system.
    """

    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the Hilbert space."""
        pass

    @abstractmethod
    def inner_product(self, state1: np.ndarray, state2: np.ndarray) -> complex:
        """Compute the inner product between two states."""
        pass

class FiniteHilbertSpace(HilbertSpace):
    """
    A concrete implementation of HilbertSpace for finite-dimensional systems.
    """

    def __init__(self, dim: int):
        self.dim = dim

    def dimension(self) -> int:
        return self.dim

    def inner_product(self, state1: np.ndarray, state2: np.ndarray) -> complex:
        if state1.shape[0] != self.dim or state2.shape[0] != self.dim:
            raise ValueError("State vector dimension mismatch.")
        return np.vdot(state1, state2)  # vdot computes the conjugate dot product

# Example usage
if __name__ == "__main__":
    hs = FiniteHilbertSpace(2)
    psi = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    phi = np.array([1/np.sqrt(2), -1/np.sqrt(2)])
    print("Inner Product:", hs.inner_product(psi, phi))

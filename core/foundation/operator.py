# core/foundation/operator.py

import numpy as np
from abc import ABC, abstractmethod

class Operator(ABC):
    """
    Abstract base class for quantum operators.
    """

    @abstractmethod
    def matrix(self) -> np.ndarray:
        """Return the matrix representation of the operator."""
        pass

    def __mul__(self, other):
        """Overload multiplication for operator composition and scaling."""
        if isinstance(other, Operator):
            return CompositeOperator(self, other)
        elif isinstance(other, (int, float, complex)):
            return ScaledOperator(self, other)
        else:
            raise NotImplementedError("Multiplication not supported for type: " + str(type(other)))

class MatrixOperator(Operator):
    """
    Represents an operator with an explicit matrix representation.
    """

    def __init__(self, mat: np.ndarray):
        self._mat = mat

    def matrix(self) -> np.ndarray:
        return self._mat

    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        Apply the operator to a quantum state vector.
        """
        return self._mat @ state

class CompositeOperator(Operator):
    """
    Represents the composition (product) of two operators (A * B).
    """

    def __init__(self, A: Operator, B: Operator):
        self.A = A
        self.B = B

    def matrix(self) -> np.ndarray:
        return self.A.matrix() @ self.B.matrix()

class ScaledOperator(Operator):
    """
    Represents an operator scaled by a constant factor.
    """

    def __init__(self, op: Operator, scalar: complex):
        self.op = op
        self.scalar = scalar

    def matrix(self) -> np.ndarray:
        return self.scalar * self.op.matrix()

# Example usage
if __name__ == "__main__":
    # Pauli-X and Pauli-Z matrices as examples
    X = MatrixOperator(np.array([[0, 1], [1, 0]]))
    Z = MatrixOperator(np.array([[1, 0], [0, -1]]))
    composite = X * Z
    print("Composite operator matrix:\n", composite.matrix())

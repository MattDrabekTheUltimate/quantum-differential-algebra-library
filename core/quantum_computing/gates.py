# core/quantum_computing/gates.py

import numpy as np
from abc import ABC, abstractmethod

class Gate(ABC):
    """
    Abstract base class for a quantum gate.
    """
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """Return the matrix representation of the gate."""
        pass

    def apply(self, state: np.ndarray, num_qubits: int, targets) -> np.ndarray:
        """
        Apply this gate to the given quantum state.
        For simplicity, this implementation currently supports single-qubit gates.
        """
        mat = self.matrix()
        if isinstance(targets, int):
            return apply_single_qubit_gate(state, num_qubits, targets, mat)
        else:
            raise NotImplementedError("Multi-qubit gate application not implemented.")

def apply_single_qubit_gate(state: np.ndarray, num_qubits: int, target: int, gate_matrix: np.ndarray) -> np.ndarray:
    """
    Apply a single-qubit gate to a specific qubit in the state vector.
    
    This routine constructs the full unitary by applying the gate on the target qubit
    and identity on the rest, then applies it to the state vector.
    """
    dim = 2 ** num_qubits
    new_state = np.zeros_like(state, dtype=complex)
    # Loop over all basis states.
    for i in range(dim):
        # Determine the value of the target qubit in state |i>.
        bit = (i >> target) & 1
        # For each possible output bit b, update the amplitude.
        for b in [0, 1]:
            # Flip the target bit to b in index j.
            j = (i & ~(1 << target)) | (b << target)
            new_state[i] += gate_matrix[bit, b] * state[j]
    return new_state

# Concrete gate implementations

class XGate(Gate):
    """
    Pauli-X gate.
    """
    def matrix(self) -> np.ndarray:
        return np.array([[0, 1],
                         [1, 0]], dtype=complex)

    def __str__(self):
        return "XGate"

class HGate(Gate):
    """
    Hadamard gate.
    """
    def matrix(self) -> np.ndarray:
        return (1/np.sqrt(2)) * np.array([[1, 1],
                                          [1, -1]], dtype=complex)

    def __str__(self):
        return "HGate"

# Additional gates can be added in a similar fashion.

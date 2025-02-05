# core/quantum_computing/circuit.py

from typing import List, Union
from .gates import Gate
import numpy as np

class QuantumCircuit:
    """
    A simple quantum circuit class that stores a sequence of gates and their target qubits.
    """
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.gates: List[(Gate, Union[int, List[int]])] = []

    def add_gate(self, gate: Gate, targets: Union[int, List[int]]):
        """
        Add a gate to the circuit.
        
        Parameters:
          - gate: an instance of Gate.
          - targets: qubit index (or indices) where the gate is applied.
        """
        self.gates.append((gate, targets))

    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        Apply all gates in the circuit sequentially to the input state.
        """
        new_state = state.copy()
        for gate, targets in self.gates:
            new_state = gate.apply(new_state, self.num_qubits, targets)
        return new_state

    def __str__(self):
        desc = f"QuantumCircuit on {self.num_qubits} qubits with {len(self.gates)} gates:\n"
        for i, (gate, targets) in enumerate(self.gates):
            desc += f"  Gate {i+1}: {gate} on qubits {targets}\n"
        return desc

# Example usage
if __name__ == "__main__":
    from .gates import XGate, HGate
    qc = QuantumCircuit(1)
    qc.add_gate(HGate(), targets=0)
    qc.add_gate(XGate(), targets=0)
    state = np.array([1, 0], dtype=complex)  # |0> state
    final_state = qc.apply(state)
    print("Final state:", final_state)
    print(qc)

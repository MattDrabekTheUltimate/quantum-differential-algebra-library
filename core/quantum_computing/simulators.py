# core/quantum_computing/simulators.py

import numpy as np
from .circuit import QuantumCircuit

def simulate_state_vector(circuit: QuantumCircuit, initial_state: np.ndarray) -> np.ndarray:
    """
    Simulate the quantum circuit using a state vector representation.
    """
    return circuit.apply(initial_state)

def simulate_density_matrix(circuit: QuantumCircuit, initial_state: np.ndarray) -> np.ndarray:
    """
    Simulate the quantum circuit using a density matrix representation.
    """
    state = circuit.apply(initial_state)
    return np.outer(state, state.conj())

# Example usage
if __name__ == "__main__":
    from .gates import HGate, XGate
    qc = QuantumCircuit(1)
    qc.add_gate(HGate(), targets=0)
    qc.add_gate(XGate(), targets=0)
    initial_state = np.array([1, 0], dtype=complex)
    final_state = simulate_state_vector(qc, initial_state)
    print("State vector simulation:", final_state)
    final_density = simulate_density_matrix(qc, initial_state)
    print("Density matrix simulation:\n", final_density)

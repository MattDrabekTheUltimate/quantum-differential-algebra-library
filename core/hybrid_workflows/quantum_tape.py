# core/hybrid_workflows/quantum_tape.py

"""
QuantumTape module for recording operations in a quantum circuit.
This is similar in spirit to frameworks like PennyLane's QuantumTape,
and can later be used to enable automatic differentiation through the circuit.
"""

class QuantumTape:
    def __init__(self):
        self.operations = []

    def record(self, operation):
        """
        Record an operation applied in the quantum circuit.
        """
        self.operations.append(operation)

    def clear(self):
        """
        Clear all recorded operations.
        """
        self.operations = []

    def get_operations(self):
        """
        Return the list of recorded operations.
        """
        return self.operations

# Example usage
if __name__ == "__main__":
    tape = QuantumTape()
    tape.record("HGate on qubit 0")
    tape.record("XGate on qubit 0")
    print("Recorded operations:", tape.get_operations())

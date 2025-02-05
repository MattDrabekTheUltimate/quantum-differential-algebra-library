# core/foundation/tensor_algebra.py

import numpy as np

def tensor_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute the tensor (Kronecker) product of two matrices.
    """
    return np.kron(A, B)

def partial_trace(rho: np.ndarray, system_dims: list, traced_system: int) -> np.ndarray:
    """
    Compute the partial trace over one subsystem for a bipartite system.
    
    Parameters:
    - rho: density matrix of the composite system.
    - system_dims: list of dimensions [d_A, d_B] for the two subsystems.
    - traced_system: index of the subsystem to trace out (0 or 1).
    
    Returns:
    - The reduced density matrix of the remaining subsystem.
    """
    total_dim = np.prod(system_dims)
    if rho.shape != (total_dim, total_dim):
        raise ValueError("Density matrix dimensions do not match product of system dimensions.")

    if len(system_dims) != 2:
        raise NotImplementedError("Partial trace is only implemented for bipartite systems.")

    dA, dB = system_dims
    rho_reshaped = rho.reshape(dA, dB, dA, dB)
    if traced_system == 0:
        # Trace out the first subsystem: sum over indices 0 and 2.
        return np.trace(rho_reshaped, axis1=0, axis2=2)
    elif traced_system == 1:
        # Trace out the second subsystem: sum over indices 1 and 3.
        return np.trace(rho_reshaped, axis1=1, axis2=3)
    else:
        raise ValueError("Invalid subsystem index for partial trace.")

# Example usage
if __name__ == "__main__":
    # Create a Bell state density matrix for two qubits.
    psi = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    rho = np.outer(psi, psi.conj())
    reduced_rho = partial_trace(rho, [2, 2], traced_system=1)
    print("Reduced density matrix:\n", reduced_rho)

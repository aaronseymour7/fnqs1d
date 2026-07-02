"""
1D J1-J2 Heisenberg chain Hamiltonian, Hilbert space and graph builders.

Mirrors the pattern NetKet/the FNQS authors use for the 2D J1-J2 model
(nqs-models/j1j2_square_10x10 on HuggingFace), just with a 1D Chain graph
in place of the 2D Hypercube.
"""
import netket as nk


def build_j1j2_chain(N: int, J2: float, J1: float = 1.0, total_sz: float = 0.0,
                      pbc: bool = True, inverted_ordering: bool = True):
    """Build the graph, Hilbert space and Hamiltonian for a 1D J1-J2
    Heisenberg chain of N sites with periodic boundary conditions.

    H = J1 * sum_{<i,j>} S_i . S_j + J2 * sum_{<<i,j>>} S_i . S_j

    where <i,j> are nearest neighbours and <<i,j>> next-nearest neighbours.

    Returns
    -------
    graph, hilbert, hamiltonian (as a jax operator)
    """
    graph = nk.graph.Chain(length=N, pbc=pbc, max_neighbor_order=2)
    hilbert = nk.hilbert.Spin(s=0.5, N=N, total_sz=total_sz,
                               inverted_ordering=inverted_ordering)
    # J=[J1, J2] pairs with the two neighbor orders defined by max_neighbor_order=2
    # sign_rule=[False, False] -> no Marshall sign rule applied, since the FNQS
    # ansatz is trained directly on the physical (un-rotated) basis.
    hamiltonian = nk.operator.Heisenberg(
        hilbert=hilbert, graph=graph, J=[J1, J2], sign_rule=[False, False]
    ).to_jax_operator()
    return graph, hilbert, hamiltonian

"""
Multi-Hamiltonian ("family") Stochastic Reconfiguration.

Implements the training scheme of Sec. II / II.1 of the FNQS paper
(arXiv:2502.09488): a single set of variational parameters theta is
optimized jointly on a family of Hamiltonians {H_gamma_1, ..., H_gamma_R}.

Because the coupling states |gamma> are mutually orthogonal (delta(gamma-gamma')),
the extended S-matrix and force vector of the combined system (Eq. 4) reduce to
a simple average (weighted by P(gamma)) of the per-system quantities computed
independently for each Hamiltonian at the *same* shared parameters theta:

    S_avg = (1/R) sum_k S_k(theta)
    F_avg = (1/R) sum_k F_k(theta)

and a single SR linear solve S_avg @ dtheta = F_avg gives the joint update.
This is exactly equivalent to pooling the M/R Monte Carlo samples from every
system into one combined batch and running standard single-system SR on it.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Sequence

import jax
import jax.numpy as jnp
import netket as nk
import jax.scipy.sparse.linalg as jssl


def make_apply_fn(model, coupling: float):
    """Wrap a ViTFNQS flax module into a NetKet-compatible apply_fun for a
    *fixed* coupling value, broadcasting `coupling` to every sample in the
    batch. Shape note: `coups` must be rank-3, (batch, 1, 1), so that the
    broadcast_to inside ViTFNQS.__call__ (target shape (batch, L_eff, 1))
    lines up correctly for batch sizes > 1.
    """
    def apply_fn(variables, x, **kwargs):
        batch = x.shape[0]
        coups = jnp.full((batch, 1, 1), coupling, dtype=jnp.float64)
        return model.apply(variables, x, coups, **kwargs)
    return apply_fn


@dataclass
class FamilyMember:
    """One Hamiltonian in the family, with its own graph/Hilbert
    space/sampler/MCState, all sharing the same variational parameters."""
    coupling: float
    hamiltonian: object
    vstate: "nk.vqs.MCState"


def build_family(model, variables, couplings: Sequence[float], N: int,
                  n_chains: int, n_samples: int, n_discard_per_chain: int,
                  hamiltonian_builder: Callable, seed: int = 0) -> List[FamilyMember]:
    """Instantiate one MCState per coupling value in `couplings`, all
    initialized with the same shared `variables` (the *full* flax variable
    dict, i.e. `{'params': {...}}` as returned by `model.init`).
    """
    members = []
    for i, J2 in enumerate(couplings):
        graph, hilbert, H = hamiltonian_builder(N, J2)
        sampler = nk.sampler.MetropolisExchange(
            hilbert=hilbert, graph=graph, d_max=2, n_chains=n_chains, sweep_size=N
        )
        vstate = nk.vqs.MCState(
            sampler=sampler,
            apply_fun=make_apply_fn(model, J2),
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
            variables=variables,
            sampler_seed=seed + i,
            seed=seed + 1000 + i,
        )
        members.append(FamilyMember(coupling=J2, hamiltonian=H, vstate=vstate))
    return members


def sync_params(members: List[FamilyMember], params):
    """Push the shared *trainable-weights* pytree (i.e. `variables['params']`,
    with no outer 'params' wrapper) into every family member's MCState."""
    for m in members:
        m.vstate.variables = {"params": params}


def family_sr_step(members: List[FamilyMember], params, diag_shift: float,
                    cg_tol: float = 1e-6, cg_maxiter: int = 200):
    """One joint SR step across the whole family, at fixed shared `params`.

    `params` is the trainable-weights pytree, i.e. `variables['params']`
    (no outer 'params' wrapper) -- this is the same convention used by
    `nk.vqs.MCState.expect_and_grad`, which is what makes `dtheta` directly
    tree-compatible with `params` for the update `params <- params - lr*dtheta`.

    Returns (dtheta, energies) where `energies` is a list of the per-system
    nk.stats.Stats objects (so you can log/monitor each J2 separately).
    """
    sync_params(members, params)

    R = len(members)
    grads = []
    energies = []
    qgts = []

    for m in members:
        m.vstate.sample()
        E, grad = m.vstate.expect_and_grad(m.hamiltonian)
        energies.append(E)
        grads.append(grad)
        qgts.append(nk.optimizer.qgt.QGTJacobianPyTree(m.vstate, diag_shift=diag_shift))

    # F_avg = (1/R) sum_k grad_k   (Eq. 1, non-disordered / uniform P(gamma) over the R systems)
    F_avg = jax.tree_util.tree_map(lambda *gs: sum(gs) / R, *grads)

    def S_avg_matvec(x):
        outs = [S @ x for S in qgts]
        return jax.tree_util.tree_map(lambda *o: sum(o) / R, *outs)

    dtheta, _ = jssl.cg(S_avg_matvec, F_avg, tol=cg_tol, maxiter=cg_maxiter)

    return dtheta, energies

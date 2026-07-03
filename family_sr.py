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
import gc
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import jax.scipy.sparse.linalg as jssl


class NonFiniteSRError(RuntimeError):
    """Raised when an SR step produces a non-finite update or energy."""


def tree_has_nonfinite(tree) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return False
    return bool(
        jnp.logical_not(
            jnp.all(jnp.array([jnp.all(jnp.isfinite(x)) for x in leaves]))
        )
    )


def tree_global_norm(tree) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return 0.0
    sq = sum(jnp.sum(jnp.abs(x) ** 2) for x in leaves)
    return float(jnp.sqrt(sq))


def clip_tree_by_global_norm(tree, max_norm: Optional[float]):
    """Rescale `tree` so its global L2 norm is at most `max_norm`.
    Returns (clipped_tree, original_norm, was_clipped).
    """
    norm = tree_global_norm(tree)
    if max_norm is None or norm <= max_norm or norm == 0.0:
        return tree, norm, False
    scale = max_norm / (norm + 1e-12)
    clipped = jax.tree_util.tree_map(lambda x: x * scale, tree)
    return clipped, norm, True


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
                  hamiltonian_builder: Callable, seed: int = 0,
                  chunk_size: Optional[int] = None) -> List[FamilyMember]:
    """Instantiate one MCState per coupling value in `couplings`, all
    initialized with the same shared `variables` (the *full* flax variable
    dict, i.e. `{'params': {...}}` as returned by `model.init`).

    `chunk_size`, if given, is forwarded to each MCState so that the
    forward/backward passes (and the QGT Jacobian, see `family_sr_step`)
    are evaluated in chunks instead of materializing a (n_samples,
    n_params) array all at once for every member. This is the main lever
    for host-memory blowups: with R members alive simultaneously the full
    per-member Jacobians (n_samples x n_params, complex128) add up fast,
    and chunking trades a bit of speed for a hard cap on peak memory.
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
            chunk_size=chunk_size,
        )
        members.append(FamilyMember(coupling=J2, hamiltonian=H, vstate=vstate))
    return members


def sync_params(members: List[FamilyMember], params):
    """Push the shared *trainable-weights* pytree (i.e. `variables['params']`,
    with no outer 'params' wrapper) into every family member's MCState."""
    for m in members:
        m.vstate.variables = {"params": params}


def family_sr_step(members: List[FamilyMember], params, diag_shift: float,
                    cg_tol: float = 1e-6, cg_maxiter: int = 200,
                    grad_clip_norm: Optional[float] = None,
                    update_clip_norm: Optional[float] = None,
                    qgt_type: str = "jacobian"):
    """One joint SR step across the whole family, at fixed shared `params`.

    `params` is the trainable-weights pytree, i.e. `variables['params']`
    (no outer 'params' wrapper) -- this is the same convention used by
    `nk.vqs.MCState.expect_and_grad`, which is what makes `dtheta` directly
    tree-compatible with `params` for the update `params <- params - lr*dtheta`.

    Stability / memory notes vs. the original version:

    - `diag_shift` is wrapped in `jnp.asarray` before being handed to
      `QGTJacobianPyTree`. Passing a bare Python float there means JAX
      treats it as a *static* trace constant, so a linearly-decayed
      schedule (a different float every iteration) triggers a fresh XLA
      compilation every single step. Over thousands of iterations those
      compiled executables accumulate in host memory and are a very
      plausible cause of an OOM that isn't otherwise explained by the
      array sizes involved. Making it a traced array fixes this.
    - `qgt_type="onthefly"` swaps QGTJacobianPyTree (which materializes a
      full (n_samples, n_params) Jacobian per family member, held for all
      R members simultaneously across the whole CG solve) for
      QGTOnTheFly (recomputes vjp/jvp products on demand, O(n_params)
      memory instead of O(n_samples * n_params) per member). Slower per
      step, but removes the dominant memory cost when R is not small.
    - Per-member gradients can be clipped by global norm (`grad_clip_norm`)
      before averaging, and the final SR update can be clipped
      (`update_clip_norm`) before being returned. Both default to off.
    - Non-finite grads/energies/dtheta raise `NonFiniteSRError` instead of
      being silently applied to `params` -- this is what was producing the
      periodic energy spikes in the log (a bad CG solve on an
      ill-conditioned S-matrix was being applied unconditionally). The
      caller is expected to catch this and back off (see train_1d_j1j2.py).

    Returns (dtheta, energies, diagnostics) where `energies` is a list of
    the per-system nk.stats.Stats objects and `diagnostics` is a dict with
    per-member grad norms and whether clipping fired, for logging.
    """
    sync_params(members, params)
    diag_shift = jnp.asarray(diag_shift, dtype=jnp.float64)

    R = len(members)
    grads = []
    energies = []
    qgts = []
    grad_norms = []
    grad_clipped_flags = []

    for m in members:
        m.vstate.sample()
        E, grad = m.vstate.expect_and_grad(m.hamiltonian)

        if not np.isfinite(float(E.mean.real)):
            raise NonFiniteSRError(
                f"Non-finite energy for coupling J2={m.coupling}: {E.mean}"
            )
        if tree_has_nonfinite(grad):
            raise NonFiniteSRError(
                f"Non-finite gradient for coupling J2={m.coupling}"
            )

        grad, gnorm, clipped = clip_tree_by_global_norm(grad, grad_clip_norm)
        grad_norms.append(gnorm)
        grad_clipped_flags.append(clipped)

        energies.append(E)
        grads.append(grad)

        if qgt_type == "onthefly":
            qgt = nk.optimizer.qgt.QGTOnTheFly(m.vstate, diag_shift=diag_shift)
        else:
            qgt = nk.optimizer.qgt.QGTJacobianPyTree(m.vstate, diag_shift=diag_shift)
        qgts.append(qgt)

    # F_avg = (1/R) sum_k grad_k   (Eq. 1, non-disordered / uniform P(gamma) over the R systems)
    F_avg = jax.tree_util.tree_map(lambda *gs: sum(gs) / R, *grads)

    def S_avg_matvec(x):
        outs = [S @ x for S in qgts]
        return jax.tree_util.tree_map(lambda *o: sum(o) / R, *outs)

    dtheta, cg_info = jssl.cg(S_avg_matvec, F_avg, tol=cg_tol, maxiter=cg_maxiter)

    # Explicitly drop references to the per-member Jacobians/QGTs now that
    # the CG solve is done -- they're the largest live objects in this
    # function and there's no reason to keep them past this point.
    del qgts, grads
    gc.collect()

    if tree_has_nonfinite(dtheta):
        raise NonFiniteSRError(
            "CG solve produced a non-finite update (S_avg is likely "
            "ill-conditioned at this diag_shift); discarding this step."
        )

    dtheta, update_norm, update_clipped = clip_tree_by_global_norm(dtheta, update_clip_norm)

    diagnostics = {
        "grad_norms": grad_norms,
        "grad_clipped": any(grad_clipped_flags),
        "update_norm": update_norm,
        "update_clipped": update_clipped,
        "cg_info": cg_info,
    }

    return dtheta, energies, diagnostics

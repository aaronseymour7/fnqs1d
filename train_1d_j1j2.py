"""
Train a Foundation Neural-Network Quantum State (FNQS, arXiv:2502.09488) on
the 1D J1-J2 Heisenberg chain, following the paper's structure but scaled
down to 1D / N=20 sites, with J2/J1 as the "Hamiltonian family" coupling
(Sec. II of the paper): a single network is trained simultaneously on R
values of J2/J1, so it generalizes across the whole frustration axis
instead of being re-trained per point.

Usage
-----
    python train_1d_j1j2.py                      # defaults: N=20, J2 in [0, 1], R=9
    python train_1d_j1j2.py --n_iter 4000 --couplings 0.0 0.2 0.4 0.5 0.6 0.8 1.0 --chunk_size 2048

Outputs
-------
    out_dir/checkpoint.pkl   -- latest {'params': ..., 'iter': ...}
    out_dir/log.csv          -- per-iteration energy per J2 value
"""
import argparse
import csv
import gc
import os
import pickle
import shutil
import time
from collections import deque

import jax
jax.config.update("jax_enable_x64", True)  # the whole ViTFNQS architecture is float64
import jax.numpy as jnp
import numpy as np
import netket as nk

from .transformer_fnqs import ViTFNQS
from .hamiltonians import build_j1j2_chain
from .family_sr import build_family, family_sr_step, NonFiniteSRError


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # System
    p.add_argument("--N", type=int, default=20, help="chain length")
    p.add_argument("--couplings", type=float, nargs="+", default=None,
                    help="explicit list of J2/J1 values forming the training family "
                         "(default: 9 points linspace(J2_min, J2_max, R))")
    p.add_argument("--J2_min", type=float, default=0.0)
    p.add_argument("--J2_max", type=float, default=1.0)
    p.add_argument("--R", type=int, default=9, help="number of systems in the family "
                                                      "(used only if --couplings not given)")
    # Architecture (ViTFNQS / FMHA hyperparameters)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--b", type=int, default=1, help="patch size; b=1 means one site per token, "
                                                      "the natural choice for a 20-site chain")
    p.add_argument("--transl_invariant", action="store_true", default=True)
    # Monte Carlo / sampler
    p.add_argument("--n_chains", type=int, default=256, help="MCMC chains PER system")
    p.add_argument("--n_samples", type=int, default=1024, help="MC samples PER system PER iteration")
    p.add_argument("--n_discard_per_chain", type=int, default=16)
    # Optimization
    p.add_argument("--n_iter", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--lr_final", type=float, default=0.005, help="linearly decayed to this by n_iter")
    p.add_argument("--diag_shift_init", type=float, default=1e-2)
    p.add_argument("--diag_shift_final", type=float, default=1e-4)
    p.add_argument("--cg_tol", type=float, default=1e-6)
    p.add_argument("--cg_maxiter", type=int, default=250)
    # Memory
    p.add_argument("--chunk_size", type=int, default=None,
                    help="forwarded to each MCState; chunks forward/backward passes "
                         "(and the QGT Jacobian) instead of materializing them for the "
                         "full n_samples at once. Main lever for host-memory OOMs.")
    p.add_argument("--qgt", type=str, default="jacobian", choices=["jacobian", "onthefly"],
                    help="'jacobian' (QGTJacobianPyTree) is faster but holds a "
                         "(n_samples x n_params) array per family member for the whole "
                         "CG solve, x R members simultaneously. 'onthefly' (QGTOnTheFly) "
                         "is slower but O(n_params) memory per member -- use this if "
                         "R * n_samples * n_params is large relative to available RAM.")
    # SR stability safeguards
    p.add_argument("--grad_clip_norm", type=float, default=None,
                    help="clip each member's raw gradient to this global L2 norm before "
                         "averaging into F_avg. Off by default.")
    p.add_argument("--update_clip_norm", type=float, default=None,
                    help="clip the final SR update dtheta to this global L2 norm. Off by default.")
    p.add_argument("--max_bad_steps", type=int, default=5,
                    help="consecutive non-finite/divergent SR steps allowed before rolling "
                         "back params to the last good checkpoint and continuing.")
    p.add_argument("--diag_shift_bump", type=float, default=5.0,
                    help="multiplicative factor applied to diag_shift, temporarily, after a "
                         "bad step (decays back to schedule value once steps are clean again).")
    p.add_argument("--spike_window", type=int, default=50,
                    help="number of recent per-member energies used to detect an energy spike "
                         "(median absolute deviation based) even when the step didn't NaN out.")
    p.add_argument("--spike_mad_factor", type=float, default=8.0,
                    help="a step is treated as a divergence spike if any member's new energy "
                         "deviates from the rolling median by more than this many MADs.")
    p.add_argument("--gc_every", type=int, default=20,
                    help="run gc.collect() + jax.clear_caches() every N iterations, as a "
                         "backstop against any residual host-memory growth.")
    # Housekeeping
    p.add_argument("--out_dir", type=str, default="./fnqs_1d_j1j2_run")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--ckpt_every", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def linear_schedule(start, end, n_iter):
    def sched(it):
        t = min(it / max(n_iter - 1, 1), 1.0)
        return start + t * (end - start)
    return sched


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    couplings = args.couplings if args.couplings is not None else \
        list(np.linspace(args.J2_min, args.J2_max, args.R))
    R = len(couplings)
    print(f"Training FNQS on 1D J1-J2 chain, N={args.N}, family of R={R} couplings:")
    print("  J2/J1 =", [f"{c:.3f}" for c in couplings])

    L_eff = args.N // args.b
    assert L_eff * args.b == args.N, "N must be divisible by patch size b"

    model = ViTFNQS(
        num_layers=args.num_layers,
        d_model=args.d_model,
        heads=args.heads,
        L_eff=L_eff,
        b=args.b,
        complex=True,          # amplitude + phase (needed: J1-J2 has sign-problem-relevant frustration)
        disorder=False,        # coupling is a scalar "family" index, not a per-site disorder field
        transl_invariant=args.transl_invariant,
        two_dimensional=False, # <-- the only architectural switch needed to go from the paper's 2D setup to 1D
    )

    key = jax.random.PRNGKey(args.seed)
    dummy_spins = jnp.ones((1, args.N), dtype=jnp.float64)
    dummy_coups = jnp.zeros((1, 1, 1), dtype=jnp.float64)
    variables = model.init(key, dummy_spins, dummy_coups)
    params = variables["params"]
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Number of variational parameters: {n_params}")

    start_it = 0
    latest_path = os.path.join(args.out_dir, "checkpoint_latest.pkl")

    if args.resume and os.path.exists(latest_path):
        with open(latest_path, "rb") as f:
            state = pickle.load(f)
    
        params = state["params"]
        start_it = state["iter"] + 1
        print(f"Resumed from iteration {start_it}")
    

    members = build_family(
        model, {"params": params}, couplings, args.N,
        n_chains=args.n_chains, n_samples=args.n_samples,
        n_discard_per_chain=args.n_discard_per_chain,
        hamiltonian_builder=build_j1j2_chain, seed=args.seed,
        chunk_size=args.chunk_size,
    )

    lr_sched = linear_schedule(args.lr, args.lr_final, args.n_iter)
    ds_sched = linear_schedule(args.diag_shift_init, args.diag_shift_final, args.n_iter)

    log_path = os.path.join(args.out_dir, "log.csv")
    write_header = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(["iter", "wall_time"] + [f"E_J2={c:.4f}" for c in couplings]
                             + [f"Eerr_J2={c:.4f}" for c in couplings])

    # Rolling per-member energy history for MAD-based spike detection, and
    # the last set of params that produced a clean (non-diverging) step --
    # this is what --max_bad_steps / --diag_shift_bump / --spike_* actually
    # drive; previously these CLI args were parsed but never used.
    energy_history = [deque(maxlen=args.spike_window) for _ in couplings]
    last_good_params = params
    consecutive_bad_steps = 0
    diag_shift_bumped = False

    def prune_checkpoints(keep_last: int = 3):
        """Delete old checkpoint_it*.pkl files, keeping only the most recent
        `keep_last` (checkpoint_latest.pkl is a separate copy and untouched)."""
        ckpts = sorted(
            f for f in os.listdir(args.out_dir)
            if f.startswith("checkpoint_it") and f.endswith(".pkl")
        )
        for f in (ckpts[:-keep_last] if keep_last > 0 else ckpts):
            try:
                os.remove(os.path.join(args.out_dir, f))
            except OSError:
                pass

    t0 = time.time()
    it = start_it
    while it < args.n_iter:
        lr = lr_sched(it)
        base_diag_shift = ds_sched(it)
        diag_shift = base_diag_shift * (args.diag_shift_bump if diag_shift_bumped else 1.0)

        try:
            dtheta, energies, diagnostics = family_sr_step(
                members, params, diag_shift=diag_shift,
                cg_tol=args.cg_tol, cg_maxiter=args.cg_maxiter,
                grad_clip_norm=args.grad_clip_norm,
                update_clip_norm=args.update_clip_norm,
                qgt_type=args.qgt,
            )
        except NonFiniteSRError as e:
            consecutive_bad_steps += 1
            print(f"[it {it:5d}] SR step diverged ({e}); rolling back to last good "
                  f"params, bumping diag_shift (bad step {consecutive_bad_steps}/{args.max_bad_steps})")
            params = last_good_params
            diag_shift_bumped = True
            if consecutive_bad_steps >= args.max_bad_steps:
                raise RuntimeError(
                    f"Exceeded --max_bad_steps ({args.max_bad_steps}) consecutive "
                    f"divergent SR steps at iteration {it}; aborting."
                )
            continue  # retry the same iteration with the bumped diag_shift

        means = [float(e.mean.real) / args.N for e in energies]  # energy per site

        # Spike check: does any member's new energy fall far outside the
        # recent rolling median (in units of MAD), even though the SR step
        # itself didn't produce a non-finite result? This is what was
        # producing the periodic energy spikes visible in log.csv (sharp
        # excursions that later relax back) -- the step wasn't NaN, just a
        # bad CG solve on an ill-conditioned S-matrix.
        spiked = False
        min_hist = max(8, args.spike_window // 4)
        for hist, m in zip(energy_history, means):
            if len(hist) >= min_hist:
                arr = np.array(hist)
                med = np.median(arr)
                mad = np.median(np.abs(arr - med)) + 1e-12
                if abs(m - med) > args.spike_mad_factor * mad:
                    spiked = True
                    break

        if spiked:
            consecutive_bad_steps += 1
            print(f"[it {it:5d}] energy spike detected (>{args.spike_mad_factor} MAD); "
                  f"rolling back to last good params, bumping diag_shift "
                  f"(bad step {consecutive_bad_steps}/{args.max_bad_steps})")
            params = last_good_params
            diag_shift_bumped = True
            if consecutive_bad_steps >= args.max_bad_steps:
                raise RuntimeError(
                    f"Exceeded --max_bad_steps ({args.max_bad_steps}) consecutive "
                    f"divergent/spiking SR steps at iteration {it}; aborting."
                )
            continue  # retry the same iteration with the bumped diag_shift

        # Clean step: accept it, reset the bad-step/bump state, and record
        # the energies used for spike detection on future iterations.
        consecutive_bad_steps = 0
        diag_shift_bumped = False
        params = jax.tree_util.tree_map(lambda p, d: p - lr * d, params, dtheta)
        last_good_params = params
        for hist, m in zip(energy_history, means):
            hist.append(m)

        if it % args.log_every == 0 or it == args.n_iter - 1:
            errs = [float(e.error_of_mean) / args.N for e in energies]
            log_writer.writerow([it, time.time() - t0] + means + errs)
            log_file.flush()
            msg = " | ".join(f"J2={c:.2f}: e={m:.5f}+/-{er:.5f}"
                              for c, m, er in zip(couplings, means, errs))
            print(f"[it {it:5d}] lr={lr:.4f} diag_shift={diag_shift:.1e} :: {msg}")

        if it % args.ckpt_every == 0 or it == args.n_iter - 1:
            ckpt_path = os.path.join(
                args.out_dir,
                f"checkpoint_it{it:06d}.pkl"
            )

            with open(ckpt_path, "wb") as f:
                pickle.dump({
                    "params": params,
                    "iter": it,
                    "couplings": couplings,
                    "args": vars(args),
                }, f)

            shutil.copyfile(ckpt_path, latest_path)
            prune_checkpoints(keep_last=3)

        # Backstop against residual host-memory growth: gc.collect() alone
        # (already done per-step inside family_sr_step) won't release JAX's
        # compiled-executable cache; jax.clear_caches() does. This is what
        # --gc_every was added for but never wired in.
        if args.gc_every and it % args.gc_every == 0:
            gc.collect()
            jax.clear_caches()

        it += 1

    log_file.close()
    print(f"Done. Latest checkpoint: {latest_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()

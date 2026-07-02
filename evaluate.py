"""
Evaluate a trained FNQS checkpoint on the 1D J1-J2 chain at one or more
J2/J1 values -- including values *not* seen during training, to check the
generalization behaviour that is the whole point of the FNQS approach
(Sec. II / Fig. 1 of arXiv:2502.09488).

Usage
-----
    python -m fnqs1d.evaluate --ckpt ./fnqs_1d_j1j2_run/checkpoint.pkl \
        --J2 0.0 0.25 0.5 0.75 1.0 --compare_ed
"""
import argparse
import pickle

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import netket as nk

from .transformer_fnqs import ViTFNQS
from .hamiltonians import build_j1j2_chain
from .family_sr import make_apply_fn


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--J2", type=float, nargs="+", required=True,
                    help="J2/J1 values to evaluate at (can include values never seen in training)")
    p.add_argument("--n_chains", type=int, default=512)
    p.add_argument("--n_samples", type=int, default=8192)
    p.add_argument("--n_discard_per_chain", type=int, default=32)
    p.add_argument("--compare_ed", action="store_true",
                    help="also run exact Lanczos diagonalization (fine up to N~24 in the Sz=0 sector)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.ckpt, "rb") as f:
        state = pickle.load(f)
    params = state["params"]
    train_args = state["args"]
    N = train_args["N"]
    L_eff = N // train_args["b"]

    model = ViTFNQS(
        num_layers=train_args["num_layers"], d_model=train_args["d_model"],
        heads=train_args["heads"], L_eff=L_eff, b=train_args["b"],
        complex=True, disorder=False, transl_invariant=train_args["transl_invariant"],
        two_dimensional=False,
    )

    print(f"Loaded checkpoint from iter {state['iter']}, trained on J2/J1 in {state['couplings']}")

    for J2 in args.J2:
        graph, hilbert, H = build_j1j2_chain(N, J2)
        sampler = nk.sampler.MetropolisExchange(
            hilbert=hilbert, graph=graph, d_max=2, n_chains=args.n_chains, sweep_size=N
        )
        vstate = nk.vqs.MCState(
            sampler=sampler, apply_fun=make_apply_fn(model, J2),
            n_samples=args.n_samples, n_discard_per_chain=args.n_discard_per_chain,
            variables={"params": params}, sampler_seed=args.seed, seed=args.seed + 1,
        )
        vstate.sample()
        E = vstate.expect(H)
        e_per_site = E.mean.real / N
        tag = "(in training family)" if J2 in state["couplings"] else "(out of distribution)"
        line = f"J2/J1={J2:.3f} {tag}: E/N = {e_per_site:.6f} +/- {E.error_of_mean/N:.6f}"

        if args.compare_ed:
            E0 = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=False)[0]
            rel_err = abs(E.mean.real - E0) / abs(E0)
            line += f"   | ED: E/N={E0/N:.6f}   rel. err={rel_err:.2e}"

        print(line)


if __name__ == "__main__":
    main()

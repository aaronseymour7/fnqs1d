# FNQS on the 1D J1-J2 Heisenberg chain (N=20)

This adapts the architecture from Rende, Viteritti, Becca, Scardicchio, Laio &
Carleo, *"Foundation Neural-Network Quantum States"* (arXiv:2502.09488) to a
**1D, 20-site J1-J2 Heisenberg chain**, trained the same way the paper trains
its 2D J1-J2-J3 model: **one network, simultaneously optimized on a family of
J2/J1 values**, rather than one network per coupling.

## What changed vs. the paper's 2D scripts, and what didn't

| File | Status |
|---|---|
| `attentions.py` | **Unchanged.** `FMHA` already implements the 1D translationally-invariant attention kernel (the `two_dimensional=False` branch using `roll`), it's just unused in the 2D scripts. |
| `transformer_fnqs.py` | **Unchanged.** `ViTFNQS`/`Embed` already branch on `two_dimensional` for patch extraction (`extract_patches1d` vs `extract_patches2d`); for a chain you just leave the flag `False`. |
| `config.py`, `modeling.py` | Unchanged logic, only the config defaults flipped to 1D (`two_dim=False`, `L_eff=20`, `b=1`). Optional — only needed if you want to package the checkpoint the way the paper's HuggingFace models are packaged. |
| `hamiltonians.py` | **New.** 1D J1-J2 chain builder (`nk.graph.Chain` + `nk.operator.Heisenberg` with `max_neighbor_order=2`), mirroring the pattern used in the paper's own `nqs-models/j1j2_square_10x10` NetKet snippet, just swapping the 2D `Hypercube` graph for a 1D `Chain`. |
| `family_sr.py` | **New.** Implements the paper's multi-Hamiltonian Stochastic Reconfiguration (Sec. II.1, Eq. 1-4): a single set of parameters is updated using the *R-averaged* energy gradient and quantum geometric tensor across the family of Hamiltonians. |
| `train_1d_j1j2.py` | **New.** Training loop (checkpointing, logging, LR/diag-shift annealing). |
| `evaluate.py` | **New.** Loads a checkpoint and evaluates the energy at any J2/J1 (including out-of-distribution points, to check generalization — the whole point of FNQS). |

The key point: **the architecture itself required zero changes**. The paper's
`ViTFNQS` module was already written generically over `two_dimensional`; a
1D chain is just `extract_patches1d` + the 1D branch of `FMHA`'s
translation-invariant kernel, both of which already existed in the code you
pasted. All the new code is the *training loop*, which the 2D snippets you
had don't include.

## The "J2 is the Hamiltonian family" training scheme

Per the paper (Eq. 1), rather than training a separate network per J2/J1,
FNQS defines a family of Hamiltonians `H_gamma` for `gamma = J2/J1` and
minimizes

```
<Phi_theta|H|Phi_theta> = sum_k (1/R) * <psi_theta(gamma_k)|H_{gamma_k}|psi_theta(gamma_k)> / <psi_theta(gamma_k)|psi_theta(gamma_k)>
```

i.e. one shared `theta` minimizing the *average* energy across R chosen
values of J2/J1 (this is the non-disordered case, `P(gamma) = (1/R) sum_k
delta(gamma-gamma_k)`, Sec. II). Because the coupling states are mutually
orthogonal, the paper shows (Sec. II.1) that the extended SR linear system
for the shared update reduces to solving

```
S_avg @ dtheta = F_avg,    S_avg = (1/R) sum_k S_k(theta),   F_avg = (1/R) sum_k F_k(theta)
```

where `S_k`/`F_k` are the ordinary single-system quantum geometric
tensor/force computed at the shared `theta` for `H_{gamma_k}`. That's
exactly what `family_sr.py::family_sr_step` does: it keeps one `MCState`
per J2 value (so each gets its own MCMC chains, same as any NQS), computes
each system's gradient and `QGTJacobianPyTree`, averages them, and does one
matrix-free CG solve shared across the whole family — this is what "trains
one network across the frustration axis" instead of training R separate
networks.

The scalar `coups` fed into `ViTFNQS.__call__` is exactly the J2/J1 value
for whichever system is currently being sampled (broadcast to every patch,
since `disorder=False` — a single global coupling, not a per-site field).

## Install

```bash
pip install -r requirements.txt
```

These exact versions are pinned deliberately and tested end-to-end (model
forward pass, a full family-SR training run, and `evaluate.py --compare_ed`
matching exact Lanczos diagonalization). In particular: **do not** `pip
install netket` unpinned — recent NetKet releases (3.16+) require
`jax>=0.4.35`, which is incompatible with this repo's float64 ViTFNQS
architecture as tested. If you need a newer JAX/NetKet, you'll want to
re-verify the whole pipeline (forward pass + a short training run +
`--compare_ed`) before trusting results, not just check that imports
succeed.

## Train

```bash
# from the directory *containing* fnqs1d/
python -m fnqs1d.train_1d_j1j2 \
    --N 20 --J2_min 0.0 --J2_max 1.0 --R 9 \
    --num_layers 4 --d_model 32 --heads 4 --b 1 \
    --n_chains 256 --n_samples 1024 \
    --n_iter 3000 --lr 0.02 --lr_final 0.005 \
    --diag_shift_init 1e-2 --diag_shift_final 1e-4 \
    --out_dir ./fnqs_1d_j1j2_run \
    --ckpt_every 50 \
    --resume
```

This trains one network simultaneously at J2/J1 = 0.0, 0.125, ..., 1.0 (9
points spanning the unfrustrated Néel phase through the frustrated/dimerized
regime around J2/J1≈0.5 and beyond). Increase `--R` for denser coverage of
the phase diagram (the paper notes cost stays roughly flat in `R` since
total samples `M` are just split across more systems — the per-step cost
does grow, but not by re-training from scratch per point).

Resume an interrupted run with `--resume` (reads `checkpoint.pkl` in
`--out_dir`).

## Evaluate (including out-of-distribution J2 points)

```bash
python -m fnqs1d.evaluate --ckpt ./fnqs_1d_j1j2_run/checkpoint.pkl \
    --J2 0.0 0.2 0.4 0.5 0.6 0.8 1.0 --compare_ed
```

`--compare_ed` runs exact Lanczos diagonalization for comparison — for
N=20 in the `total_sz=0` sector the Hilbert space dimension is
`C(20,10) = 184756`, which Lanczos handles comfortably (seconds to low
minutes), so this is a genuinely useful sanity check at this system size.

## Notes / things you'll likely want to tune

- **`b` (patch size):** `b=1` (one site per token) is the natural choice
  for a 20-site chain; the paper only patches (`b=2` in 2D) for much larger
  lattices where `L_eff` (the attention sequence length) would otherwise be
  too big. For N=20 you could try `b=2` (`L_eff=10`) if you want a smaller
  attention footprint, but `b=1` is the direct analogue of the 1D TFI
  experiments in the paper.
- **Family design:** the default is a uniform grid over `[J2_min, J2_max]`.
  If you mainly care about accuracy right at the frustration crossover, note
  there are two distinct points of interest in the 1D chain: the gapless
  spin-fluid → dimerized (BKT) transition sits at the Okamoto-Nomura point,
  J2/J1 ≈ 0.2411, while J2/J1 = 0.5 is the Majumdar-Ghosh point, an exactly
  solvable point *inside* the dimerized phase (product of nearest-neighbor
  singlets), not the transition itself. Bias `--couplings` to be denser
  around whichever you care about, e.g.
  `--couplings 0.0 0.15 0.2 0.24 0.28 0.35 0.5 0.7 1.0` to resolve the BKT
  transition, or add points near 0.5 if you want to check against the exact
  Majumdar-Ghosh energy.
- **`total_sz=0`** is hard-coded in `hamiltonians.py` (appropriate ground
  state sector for the antiferromagnetic J1-J2 chain); the
  `MetropolisExchange` sampler preserves it automatically via spin-exchange
  moves.
- **No Marshall sign rule** (`sign_rule=[False, False]`) is used, matching
  the paper's own 2D snippet — the network learns the sign structure
  directly rather than via a basis rotation. Marshall's sign rule is only
  exact at J2=0 (the unfrustrated, bipartite case); it degrades
  progressively as J2 grows rather than breaking at any single threshold,
  which is why it's simplest to just not apply it and let the network learn
  signs directly across the whole family.
- **Energy convention:** NetKet's `nk.operator.Heisenberg` is built from
  Pauli matrices (eigenvalues ±1), i.e. `H = J * sum sigma_i . sigma_j`,
  which is **4x** the textbook spin-1/2 convention `H = J * sum S_i . S_j`
  (`S = sigma/2`). This only rescales the absolute energy scale uniformly —
  it doesn't affect coupling ratios, phase boundaries, or anything you'd
  compare across J2/J1 — but keep it in mind when comparing `evaluate.py`
  output to literature values. E.g. at the Majumdar-Ghosh point (J2/J1=0.5)
  the textbook exact energy is E0/N = -3/8 = -0.375; in NetKet's convention
  that's E0/N = -1.5, which is what `--compare_ed` will report.

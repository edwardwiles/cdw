# Envelope-scalar formula (reused, not re-derived, from `../ad_benchmark/`)

This directory's ForwardDiff benchmark (`benchmark_forwarddiff.jl`) differentiates the exact
same function `../ad_benchmark/derivative_core.jl::envelope_scalar_div_ctx(θ, ctx)` that
`../ad_benchmark/README.md` already validated against production to relative error ~1e-15
(`../ad_benchmark/correctness_results.csv`). It is not re-derived or re-validated here —
only its **performance**, as a function of ForwardDiff configuration, is new work in this
directory.

## What is differentiated

```julia
function envelope_scalar_div_ctx(θ, ctx)   # ctx = (U, γobj, λ, arg1, d, outer_constr_index)
    H = moment_map(θ, ctx.U, ctx.γobj, ctx.d)     # H = [K | 1 | G], EK_moments_gammanorm_directgp!
    s = Σ_draw arg1[draw] * Σ_{j=1}^{outer_constr_index-1} λ[j] * G[draw,j]
    return (1e10/N) * s
end
```

reproducing exactly the divergence-budget-constraint contraction production computes post-hoc
from the dense Jacobian at `cc_algo/PsiObjectiveBundle.jl:216-222`.

## What is held fixed (the envelope theorem step)

`λ` (inner-problem multipliers) and `arg1` (per-draw dΨ weight) are **frozen at their values
from the real KNITRO inner solve at the benchmark θ** — passed into `ctx`, never
differentiated. This is exactly the envelope-theorem step §9 of the originating prompt asked
for: solve the inner CC problem once, hold its optimal dual objects fixed, differentiate only
the resulting scalar. `gradient_θ envelope_scalar_div_ctx = gradient_θ minimum_divergence(θ)`
holds because the inner problem's own first-order conditions make the direct partial derivative
w.r.t. `(λ, arg1)` vanish at the optimum — the standard envelope argument, not re-derived here
(see `../ad_benchmark/derivative_formulas.md` for the full derivation and the caveat about the
SEPARATE gravity constraint, which is NOT a pure envelope scalar and needs a genuine
implicit-function-theorem term — out of scope for both this directory and `ad_benchmark/`, and
explicitly flagged rather than silently approximated).

## What this directory adds on top

Only the ForwardDiff *configuration* around calling this function — chunk size, `GradientConfig`
caching, closure vs. functor, preallocated output — see `forwarddiff_benchmark.csv` and the
final report for results. The mathematical content is unchanged from `../ad_benchmark/`.

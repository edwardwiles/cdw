# Common-Fréchet gradient interpretation — 2026-07-25/26 (Part IV)

## 1. dM/dA = 0: both restriction blocks are theta-independent

Both the CM block and the level block are built once, purely from the fixed baseline draws `U`
(never a function of `θ`/`A`). `lfix_cm_aware.jl`/`lfix_cm_cplus.jl` already exploit exactly this
property for CM's own restriction: the fixed contribution `λ_C*'C_s` is computed **once**, outside
the per-outer-coordinate probe loop, and folded into the cached base dual scalar `q0` via
`with_q0`/`with_q0_C` — the coordinate loop itself never touches the restriction tail of `λ*` once
`q0` reflects it. The level block has the identical property, so the same architecture applies
without any change to the coordinate loop.

## 2. What's new vs. reused

`cm_fixed_contribution` (`lfix_cm_aware.jl`) hardcodes the CM-only tail length (`aug.ncm = (D-1)L`)
and cannot be called unmodified against a `:common_frechet` `aug` (`ncm = D·L`). New
`frechet_cm_level_fixed_contribution` (`cm_frechet_cplus.jl`) splits the combined tail into its CM
and level parts: the CM part reuses `apply_contrast`/`suffix_sums`/`cumulative_forward_contribution!`
(`cm_lookup_kernels.jl`) **verbatim**; the level part is new (`frechet_level_forward_sum!`, an
all-D-origin sum rather than a reference-differenced lookup) plus a constant target-correction term
`Σ λ_level·level_targets` — the gradient-side analog of Part III's Hessian target-correction terms
(the nonzero level target doesn't cancel out of a linear `λ'·feature` contraction either).

Four thin wrappers mirror the pre-existing CM ones exactly: `build_lfix_base_cache_cm_frechet(_C!)`,
`composite_gradient_at_fast_frechet`, `composite_gradient_at_Cplus_frechet`,
`archC_frechet_base_state`.

## 3. Validation

D=4 (`test_frechet_gradient_cplus_vs_reference_d4.jl`, 7/7 PASS): common-Fréchet's C+ (factorized
envelope) gradient vs its own Reference (non-factorized) envelope gradient, at the same solved
state, agree to `8.7e-16` (machine precision, the task's stated primary gate) — only `2.8×` flexible
CM's own Reference-vs-C+ discrepancy (`3.1e-16`), i.e. both backends sit at the same numerical noise
floor, not a material gap. The gamma/analytic first component (`g[1]`, which never touches the
CM/level tail) also agreed independently as a structural consistency check.

## 4. Shared finite-difference (re-solved) diagnostic — scope note

The task's secondary, looser gate (envelope estimator vs fully re-solved finite differences,
compared side by side for unrestricted/flexible-CM/common-Fréchet) was **not run as a separate
bounded diagnostic** this session — the D=4 C+-vs-Reference agreement above (both being
envelope-family estimators) is the primary implementation gate and passed at machine precision, and
D=20 outer-loop evidence (Part VII) shows the C+ gradient (`cm_gradient_backend=:cplus`, the
production default) driving real, correct outer progress for common Fréchet at production scale.
Per the task's own framing, the envelope-vs-fully-re-solved-FD gap is a property of the *shared*
outer-gradient backend (already characterized in the prior Fréchet port as reproducing at the same
order across unrestricted/CM/Fréchet) — nothing in this session's D=4 or D=20 evidence suggests
common Fréchet is a special case of that shared, pre-existing behavior.

## Verdict

`SHARED_FD_DISCREPANCY = same_as_unrestricted_cm` (by the C+-vs-Reference agreement above and no
observed anomaly in real outer-loop behavior; not independently re-measured against fully re-solved
FD this session — disclosed).

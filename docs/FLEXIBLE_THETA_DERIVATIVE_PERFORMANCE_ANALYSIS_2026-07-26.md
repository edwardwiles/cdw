# Flexible-theta derivative: why it's expensive, with real numbers — 2026-07-26

Follow-up to the matched-comparison finding in `FLEXIBLE_THETA_MATCHED_COMPARISON_2026-07-25.md`
(flexible theta loses to fixed transformed-A at both tested budgets). This document answers *why*,
with a real instrumented measurement, not just a structural argument — a live-data correction of
an initial hypothesis is included below because the numbers didn't match it.

Script: `full_aod_diag/d4_exact/profile_theta_derivative_cost_2026-07-26.jl` (diagnostic-only, not
part of the gate suite). Real D=20/W=80,000/seed=20260719, `OPENBLAS_NUM_THREADS=1`.

## The mechanism

The 380 gp/A-block coordinates get their gradient from **one shared analytic pass**:
`composite_gradient_at_Cplus` — a factorized, threaded kernel that computes all 380 partials
together (closer to an adjoint/reverse-mode computation than 380 finite differences). Its cost is
essentially independent of which of the 380 coordinates you're asking about.

Theta's own partial derivative does **not** go through that kernel. In flexible mode, every
`cb_G!` callback (`c10_d20_production_driver_unified.jl`) instead does:

```julia
D_plus  = theta_fixed_dual_delta_pivot_A(w_plus,  inner_x_fixed, ctx, xy)   # theta+h, full reconstruction
D_minus = theta_fixed_dual_delta_pivot_A(w_minus, inner_x_fixed, ctx, xy)   # theta-h, full reconstruction
grad_eta_theta = (D_plus - D_minus) / (2 * h_theta)
...
ctx.obj.moments!(@view(ctx.obj.H[:, 1]), ..., θ_full_base, ctx.obj.U, ctx.obj)   # theta_base, AGAIN
```

That's **three separate full calls** to `obj.moments!`/`CS.reconstruct_full` per gradient
callback — a central-difference secant (2 calls) plus one more at the current point that appears
to re-derive something the base dual state (`base.θ_full0`/`base.m_star`, already computed
upstream in the same callback) may already carry.

## Measured cost (one representative gradient callback, warmed/JIT'd)

| Block | wall time | allocated |
|---|---|---|
| **A** — shared analytic C+ gradient, all 380 coordinates, ONE call | 1,730 ms | 144 MB |
| **B** — ONE `theta_fixed_dual_delta_pivot_A` call (secant needs 2 of these) | 1,452 ms | **743 MB** |
| **C** — `build_pivot_elimination_cheap` alone (rebuilt redundantly x2 inside B) | 35 ms | 83 KB |
| **D** — the base-point `moments!`/`reconstruct_full` call (the 3rd reconstruction) | 1,358 ms | **743 MB** |

**Correction to an initial hypothesis, stated plainly:** I first guessed the redundant
`build_pivot_elimination_cheap` rebuild inside `decode_and_expand_flexible_A` (called on every
`theta_fixed_dual_delta_pivot_A` invocation, i.e. twice per gradient, when a `pgc` cache is
already available and threaded through elsewhere in the driver) was a meaningful contributor. It
is real and redundant, but **it is not the cost** — 35ms / 83KB per call, ~70ms / 166KB total
across both secant evaluations, against a ~4.3-second theta block. It's free to fix but won't
move the needle on its own.

**The actual dominant cost is the full `moments!` reconstruction itself**, done 3 times per
gradient callback instead of the 1 time fixed mode needs (fixed mode never calls this path at
all — theta doesn't move, so nothing theta-dependent needs re-deriving). Each call allocates
**~743 MB** and costs **~1.35-1.45 seconds**. This directly matches what you flagged — "copy to
make a new big array every single time" — just located inside `obj.moments!`/
`CS.reconstruct_full`'s own internals rather than in the pivot-cache path. This report did not
drill into *which specific line* inside `moments!`/`reconstruct_full` produces the 743 MB (that
would need per-line `@allocated`/`Profile.jl` inside those functions, not attempted here) — a
natural next step for whoever picks this up.

**Net effect per gradient callback**: the theta block (2×B + 1×D) costs **~4,260 ms — 2.5x the
cost of the entire 380-coordinate analytic gradient (1,730 ms)** — and accounts for **~71% of
total `cb_G!` wall time**, while allocating roughly **2.2 GB** of garbage per callback (vs 144 MB
for the 380-coordinate block). Over a real campaign's 20-80 gradient calls (per the matched-
comparison logs), that's tens of GB of allocation from this one code path alone, plus whatever GC
pressure that induces — a second-order cost this measurement doesn't isolate but which plausibly
compounds the wall-clock gap.

## Why this happened despite theta having an "exact chain rule" elsewhere in this port

The A-block coordinates get an *exact, O(1), zero-extra-reconstruction* chain rule
(`d(Delta)/d(a) = d(Delta)/d(z)·(-theta)`, `gradient_transform_unified`) precisely because that
relationship is a closed-form scalar rescale of an already-computed quantity. Theta's own
derivative has no equivalent closed form implemented here — the "fixed-dual" framing (hold ζ,λ
fixed, only let the moment-generating objects move with theta) is the right *idea* for keeping
the derivative cheap and consistent with the no-re-solve discipline the rest of this port follows,
but the actual implementation still gets that derivative via **brute-force central-difference
finite differencing across two full forward model reconstructions**, rather than via an analytic
implicit-function-theorem-style derivative of the same fixed-dual objective with respect to theta
directly. That gap — numerical secant over full reconstructions, instead of an analytic partial
derivative of a quantity that's (by the fixed-dual construction) already a fairly explicit
function of theta — is the structural inefficiency underneath the raw numbers above.

## Not fixed in this port

This analysis is diagnostic only; no change was made to `theta_fixed_dual_delta_pivot_A`,
`decode_and_expand_flexible_A`, or `cb_G!`'s flexible-mode block as part of this task. Candidate
directions (for you to weigh, not a recommendation ranking):
- Thread the already-available `pgc`/`pivot_elim_from_cache` through
  `theta_fixed_dual_delta_pivot_A`/`decode_and_expand_flexible_A` instead of rebuilding it — free,
  ~70ms/call, worth doing regardless of anything else.
- Drop the third (base-point) `moments!` call by reusing `base.θ_full0`/`base.m_star` (already
  computed upstream in the same callback) instead of recomputing `θ_full_base` from scratch — would
  cut the theta block from 3 reconstructions to 2, a ~30% reduction on its own if D and B cost the
  same per-call.
- Investigate whether `obj.moments!`'s ~743 MB/call allocation can be reduced (in-place
  construction into a preallocated workspace, mirroring the `CompressedFactualWorkspace`/
  canonical-price-precompute pattern the allocation/Hessian production release already applied
  elsewhere in this codebase) — likely the largest lever, but requires understanding what inside
  `moments!`/`reconstruct_full` is allocating that much, which this report did not pin down.
- Whether an analytic (rather than finite-difference) theta derivative of the fixed-dual objective
  is tractable at all is an open modeling question, not an engineering one — outside this report's
  scope.

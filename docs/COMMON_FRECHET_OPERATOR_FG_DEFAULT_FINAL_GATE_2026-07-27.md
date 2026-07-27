```text
COMMON_FRECHET_INNER_FG_BACKEND_DEFAULT = dense_reference (UNCHANGED — broader gate confirms the
    prior narrow-gate decision, not merely leaves it unresolved)
```

# Common Fréchet Operator-FG Default — Broader Gate — 2026-07-27 (Section 3.3)

## Starting point

`docs/COMMON_FRECHET_FG_D20_FINAL_GATE_2026-07-27.md` (same day, earlier session) found and fixed a
real crash bug (`skip_cm_fill_ref` silently starving the Hessian callback of dense CM/level columns
under `:cm_frechet_lookup`), then measured exactly **one** real D=20 point (the previously-crashing
calibration point, `L=50`, `contrasts=:anchored` only) and explicitly declined to flip
`CM_FRECHET_INNER_FG_BACKEND_DEFAULT`, citing insufficient breadth: *"this session's own gates cover
exactly two D=20 points (calibration, gp0*1.01) at one L — not the breadth of configurations a
genuine default-flip decision should rest on."* This document is the broader gate that decision
asked for.

## Gate design

`bench_frechet_operator_fg_default_gate_2026-07-27.jl`, real D=20/W=80,000/L=50
(`destination_sample=:exclude_row`):

- **Both contrasts** (`:anchored`, `:orthonormal`).
- **Three points**: calibration, `near_delta1_perturbed` (small random nudge of every free
  coordinate), and `hard_point_x1.01` (`x_free0 .* 1.01`, the exact perturbation
  `COMMON_FRECHET_FG_D20_FINAL_GATE_2026-07-27.md` established as this family's own stress point).
- **Measured**: complete-solve time and allocation (median of 5 warm repetitions via `@timed`),
  correctness (KNITRO status + `ζ*` agreement), `n_fg`/`n_hess` call counts (iteration-count proxy),
  and isolated per-FG-callback allocation for the warmed-up `CMFrechetLookupState` (same methodology
  `docs/COMMON_FRECHET_OPERATOR_FG_FINAL_GATE_2026-07-26.md` used to find the historical ~14MB/call
  regression, reused here to check whether it is still present post-fix).
- Both backends built with `cm_cross_hessian_backend = CM_FRECHET_CROSS_HESSIAN_BACKEND_DEFAULT[]`
  (this session's own Section 3 Hessian-backend work, now `:winner_bin` — see
  `docs/COMMON_FRECHET_WINNER_AWARE_HER_RELEASE_2026-07-27.md`), so this gate measures the FG-backend
  question in isolation against the real current production Hessian configuration, not an
  artificially-fixed one.

## Results (real, this session — full log in the branch's own scratch history, summarized here)

Correctness: **ALL PASS, 24/24 checks** — every point/contrast combination feasible on both
backends, KNITRO status identical (`0` throughout), `ζ*` agreeing to `2.6e-16`–`3.0e-13`.

| contrasts | point | speedup (dense/lookup) | alloc_ratio (lookup/dense) | n_fg (d/l) | n_hess (d/l) |
|---|---|---|---|---|---|
| anchored | calib | 1.171x | 1.0394 | 1/1 | 0/0 |
| anchored | near_delta1_perturbed | 1.172x | 1.0394 | 1/1 | 0/0 |
| anchored | hard_point_x1.01 | 1.171x | 1.0394 | 1/1 | 0/0 |
| orthonormal | calib | 1.117x | 1.0394 | 1/1 | 0/0 |
| orthonormal | near_delta1_perturbed | 1.213x | 1.0394 | 1/1 | 0/0 |
| orthonormal | hard_point_x1.01 | 1.115x | 1.0394 | 1/1 | 0/0 |

Isolated per-FG-callback allocation, warmed-up `CMFrechetLookupState`: **8,256 bytes** — NOT the
~14MB/call regression `docs/COMMON_FRECHET_OPERATOR_FG_FINAL_GATE_2026-07-26.md` found and left
unresolved. That regression does not appear to be present anymore (measured identically at both
contrasts' own warm state), though this session did not re-run that document's own bisection
methodology to positively root-cause *why* it went away — the most likely explanation is the same
`skip_cm_fill_ref` removal that fixed the D20 crash also incidentally fixed whatever interacted
with the `obj::Any` devirtualization gap that document diagnosed, but that is an inference, not a
verified mechanism.

## What this rules in / rules out, vs the narrower gate

- **Iteration count is not a factor**: `n_fg=1`/`n_hess=0` identical between backends at every one
  of the 6 configurations, not just the single point measured before. Confirms
  `COMMON_FRECHET_OPERATOR_FG_FINAL_GATE_2026-07-26.md`'s own finding that this codebase's own
  addendum question ("if allocation-free operator + different iteration count explains the gap,
  report that") does not apply here — it is a genuine per-callback allocation difference, not an
  iteration-count artifact, confirmed now across contrasts and points rather than one point.
- **Speedup is real and consistent**: 1.115x–1.213x across all 6 configurations, no outliers, no
  regressions at any point tried (including the hard point).
- **Allocation ratio is real, consistent, and in the WRONG direction for a flip**: `1.0394` at every
  single point, essentially identical to the single-point `1.0389` measured before (this is not new
  information changing the number, it is confirmation the number is stable, not a fluke of the one
  point previously tried).

## Decision: default NOT flipped

`CM_FRECHET_INNER_FG_BACKEND_DEFAULT` **stays `:dense_reference`**. This codebase's own established
flip criterion (see `CM_INNER_FG_BACKEND_DEFAULT`'s own docstring, `core_exact_hessian.jl`, and
`ORIGINZC_FG_BACKEND_DEFAULT`'s docstring which explicitly restates it) requires allocation parity
*or* reduction alongside a real speed win — precedent flips in this codebase were justified by
allocation at-parity-or-better plus speed, never speed alone. A consistent ~4% allocation *increase*
across 6 configurations does not meet that bar.

This is a **stronger, not weaker**, version of the prior session's own non-flip decision: that
decision was explicitly provisional ("not the breadth... a genuine default-flip decision should rest
on"); this one is not — the breadth asked for has now been run, and it confirms the same conclusion
rather than overturning it. `:cm_frechet_lookup` remains available, provably correct (the D20 crash
this family previously hit is fixed and re-verified here at 3 points x 2 contrasts, all feasible,
all agreeing with dense to nanometer-scale `ζ*` precision), and a real, consistent, welcome ~1.1-1.2x
speedup for any caller who wants to opt in today (`inner_fg_backend=:cm_frechet_lookup`).

## What would change this decision

If a future session either (a) closes the `obj::Any`/devirtualization allocation gap
`COMMON_FRECHET_OPERATOR_FG_FINAL_GATE_2026-07-26.md` diagnosed but did not fix (bringing allocation
to genuine parity or below), or (b) this project's own flip criterion is deliberately revised to
weight speed over allocation for this family, the flip would be straightforward — the correctness
and iteration-count evidence already fully support it; only the allocation number is against it.

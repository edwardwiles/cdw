# Primitive formulas vs. task-prompt spec — code-level check (2026-08-01)

Per the task's §2 instruction: "map its current keep statistic and prove whether the constant is
already embedded. Do not add the France constant twice" — done by direct read of
`core_exact_hessian.jl`'s `WinnerPairHessCtx`/`build_winner_pair_ctx` (the single construction site
all four fixed functions share) and the four functions' own use sites in `winner_pair_cross_hessian.jl`
/ `threaded_cross_hessian.jl`. No new code was written for this check — it is a direct line-by-line
correspondence between the task's LaTeX spec and the already-fixed (per `f1f969b`) production code.

## Bilateral row, `E_j(w) = kappa0[j]*M_d(w)*[1{w_d(w)=o} - lambda_od]`, `Lambda_j = kappa0[j]*lambda_od`

`core_exact_hessian.jl:435-440`:

```julia
k0 = cf.nrm[j] * cf.gdiv[j]
kappa0[j] = k0
Lam_homog[j] = k0 * cf.Pmat[o, slot]
```

`Lam_homog[j] == kappa0[j] * Pmat[o, target_slot[j]]` exactly — matches the spec's `Lambda_j =
kappa0[j]*lambda_od` with `lambda_od == Pmat[o, target_slot[j]]` (production's own name for the
destination-o win-share coefficient). Confirmed used as the sole multiplier on the destination-specific
correction term (`corr`/`MSumX[target_slot[j],l]`/`T0_slot[target_slot[j]]`/`TZ[d,x]`, depending on
function) in all four fixed call sites (`winner_pair_cross_hessian.jl:297`, `:668`, `:763`;
`:560`/threaded `threaded_cross_hessian.jl` twin) — never `pi_vec[j]` (the old structured-formulation
constant, confirmed still used, unchanged, only on the `use_profiled_correction=false` branch).

## France/cf row, `E_cf(w) = kappa0_cf*[cf_raw(w) + denom_cf - rho*M_f(w)]`, `rho=gp^sigma`, `Lambda_cf = kappa0_cf*rho`

`core_exact_hessian.jl:456-461`:

```julia
k0cf = cf.nrm[jcf] * cf.gdiv[jcf]
kappa0[jcf] = k0cf
Lam_homog[jcf] = k0cf * gpσ          # == kappa0_cf * rho
denom_cf_scaled = k0cf * denom_cf    # == kappa0_cf * denom_cf
cf_raw_scaled = k0cf .* cf.cf_raw    # the per-draw "keep" term, kappa0_cf * cf_raw(w)
```

`Lam_homog[jcf] == kappa0_cf*gpσ == kappa0_cf*rho` — matches `Lambda_cf` exactly. `denom_cf_scaled ==
kappa0_cf*denom_cf` — matches the spec's `denom_cf_scaled` definition verbatim (task prompt literally
names both quantities the same way; this is not a coincidence, the source-branch derivation and the
task prompt describe the same fix).

**No double-count check**: `cf_raw_scaled` (the per-draw "keep" statistic, contributing
`kappa0_cf*cf_raw(w)` to every accumulation touching draw `w`) never itself contains `denom_cf` or a
`gpσ*M_f(w)` term — confirmed by its construction (`k0cf .* cf.cf_raw`, `cf.cf_raw` is raw per-draw data
from `CompressedFactual`, populated upstream of `build_winner_pair_ctx` and independent of `gpσ`/
`denom_cf`, both of which are keyword-only optional arguments to `build_winner_pair_ctx` unused anywhere
in `cf`'s own construction). The additive `denom_cf_scaled*<restriction-column-sum>` term is added
exactly once, at each of the four call sites' own accumulation line (`winner_pair_cross_hessian.jl:322`
[H_EC], `:689` [H_EF colsum!], `:774` [H_EF esum!], `:584`/threaded twin [H_EZ]) — one call site per
function, one addition per call site. No second addition site exists (grepped
`denom_cf_scaled` across the whole `full_aod_diag/d4_exact/` tree outside test files — exactly the 5
production use sites above, matching the 4 fixed functions + 1 struct field).

## Verdict

`PRIMITIVE_FORMULAS`:
- `H_EC`: matches spec exactly (fixed prior session, `6f6a878`) — pass.
- `H_EF`: matches spec exactly (fixed this lineage, `f1f969b`) — pass.
- `H_EZ_serial`: matches spec exactly — pass.
- `H_EZ_threaded`: identical edit to serial, bit-identical output confirmed by the source branch's own
  pre-existing threaded-vs-serial test — pass.
- `France_rows`: `Lambda_cf`/`denom_cf_scaled` construction matches spec exactly, confirmed single
  accumulation site per function, no double-count — pass.

This is a **code-reading confirmation**, not independent numerical truth — it establishes that the
already-fixed code matches the task's own restated formula, which is expected since both describe the
same `6f6a878`/`f1f969b` bugfix. It does **not** replace Phase 4/5's ForwardDiff/reduced-G oracles,
which are the actual independent-truth gates the task requires before trusting these primitives in a
reduced-layout KNITRO solve.

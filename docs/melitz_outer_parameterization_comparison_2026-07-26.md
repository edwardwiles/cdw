# Melitz outer-coordinate parameterization comparison (2026-07-26 addendum)

Governing prompt's addendum, run after the production-fast closure phases (Phases 1-13,
`docs/melitz_production_fast_backend_closure_2026-07-26.md`) passed. This is a
parameterization/numerical-conditioning exercise only, at D=4 -- no change to the economic
feasible set, gravity/cutoff restrictions, moment functions, `DeltaStar`, welfare objective,
or calibrated reference point; no outer-optimizer redesign; no real-D20 work.

## A. Mathematical audit (complete)

Traced every appearance of `A` in the active Melitz equations directly in the source (not
from memory), per the governing prompt's own instruction.

### A.1 Active-firm contribution coefficient

`melitz_C` (`firm_quantities.jl:94`):

```julia
C_od = expenditure_d * (markup*w_o*tau_od/A_od)^(1-sigma)
```

so `C_od ∝ A_od^(sigma-1)` exactly -- the active-contribution coefficient scales as
`A_od^(sigma-1)`, matching the `(sigma-1)*logA` candidate exactly for THIS equation.

### A.2 Cutoff

`melitz_cutoff` (`firm_quantities.jl:83`) composed with `melitz_C`:

```julia
zhat_od = (sigma*w_o*f_od/C_od)^(1/(sigma-1))
        ∝ (f_od / A_od^(sigma-1))^(1/(sigma-1))
        = f_od^(1/(sigma-1)) * A_od^(-1)
```

so `log(zhat_od) = const_od + (1/(sigma-1))*log(f_od) - log(A_od)` -- confirms the governing
prompt's own hypothesized formula EXACTLY, with the coefficient on `log(A_od)` being `-1`
(plain, UNSCALED `logA`) for the cutoff equation specifically.

### A.3 Pareto aggregate trade-share composite

`melitz_pareto_composite` (`pareto_calibration.jl:508-524`):

```julia
beta = 1 - theta_star/(sigma-1)
chi[o,d] = theta_star*a0[o,d] + beta*f0[o,d]
```

where `a0`/`f0` are the identified (data-implied) `log A`/`log f` composites. Confirms the
governing prompt's own hypothesized formula exactly: the identified composite scales `log A`
by `theta_star` (not `sigma-1`, not `1`).

**Summary**: THREE different natural scales for `log A_od` already coexist in the ACTIVE
equations depending on which one you look at -- `1x` (cutoff), `(sigma-1)x` (active-firm
contribution), `theta_star x` (Pareto aggregate composite). This is exactly the tension the
governing prompt's Section M asks to resolve empirically, not algebraically.

### A.4 The actual Ricardian powered-A transformation (read-only, not from memory)

Read `full_aod_diag/PsiObjectiveBundleImplicitMethodB_fullA.jl` (`make_gravity_grad`,
lines 208-220) directly -- the Ricardian (Eaton-Kortum) full-A_od gravity-elimination code's
own successful reparametrization:

```julia
μ = θθ[1]   # the outer theta vector's own first coordinate -- the Frechet/trade elasticity
Aod = Aod_θ .* cHat .* (...)   # the free A_od parameter, offset/scaled by known constants
AodPow = (Aod ./ cHat) .^ (-μ)
```

i.e. `log(AodPow) = -μ*log(Aod/cHat)` -- the Ricardian model scales `logA` by the FULL trade
elasticity `μ` (its own name for the Frechet shape parameter, structurally the SAME role
`theta_star` plays for Melitz's Pareto tail), with a NEGATIVE sign (a labeling/direction
convention, not a substantive scale difference). **This matches the `:theta_logA` candidate
(`theta_star*logA`, up to sign) -- NOT `:invtheta_logA` (`logA/theta`)**, which the governing
prompt flagged as a possibility to check for rather than assume. `:invtheta_logA` is
therefore NOT included as a separate candidate below (confirmed absent from the actual
Ricardian code, not merely assumed absent).

## B. Candidate technology coordinates (implemented)

Three candidates, matching Section A's own three natural scales (no fourth `:invtheta_logA`
candidate, per A.4's finding):

- `:logA` -- `a_od = log(A_od)` (matches the cutoff equation's own natural scale).
- `:theta_logA` -- `a_od = theta_star*log(A_od)` (matches the Pareto composite AND the
  actual Ricardian transformation, up to sign).
- `:sigma_minus_one_logA` -- `a_od = (sigma-1)*log(A_od)` (matches the active-firm
  contribution coefficient's own natural scale).

---

# 2026-07-27 continuation session (governing prompt addendum, 14 phases)

Continues the above (Sections A-B, 2026-07-26). This session closed the remaining closure
loose ends (Phases 1-4 below), corrected the 2026-07-26 draft's own Section A.4 conclusion
(Phase 5), then implemented, wired, and empirically tested the technology-coordinate axis
(Phases 6-13). All work under `src/melitz/`, `test/melitz/`, `docs/`; zero Ricardian source
touched (confirmed at the end, Section H below).

## C. Closure loose-end fixes (governing prompt Phases 1-4)

**Phase 1 (KN_add_vars cc_algo dependency)**: confirmed live -- `methods(KNITRO.KN_add_vars)`
shows only the 3-arg core form; the 2-arg convenience form `MelitzCCBundle`'s own KNITRO
driver called (`cc_bundle.jl`) came solely from `cc_algo/knitro_compat.jl`'s monkey-patch of
the KNITRO module. Fixed with a Melitz-owned local wrapper, NOT another KNITRO-module
monkey-patch (`src/melitz/knitro_compat.jl`: `melitz_kn_add_vars!`/`melitz_kn_add_cons!`/
`melitz_kn_get_int_param`), updated at all 7 real call sites across `cc_bundle.jl`,
`matrix_free_dual_solve.jl`, `nuisance_profile.jl`, `finite_delta_outer.jl`. Added
`test/melitz/standalone_no_cc_algo.jl` -- a genuinely separate PROCESS (launched via
`Base.julia_cmd()` from a new `runtests.jl` testset, never `include`d in-process, which would
inherit whatever `cc_algo` state that file already loaded) that loads ONLY the Melitz include
path + KNITRO, constructs a `build_melitz_psi_bundle(...; backend=:matrix_free,
forbid_dense_fallback=true)` bundle, and runs a real matrix-free inner KNITRO solve via
`melitz_recover_lfd`. Verified live: `nStatus=0`, `Delta=7.55e-6`, exit code 0, no `cc_algo`
ever loaded.

**Newly found, separate gap (not the same bug, flagged not fixed)**: `solve_melitz_finite_delta_bound`
(the OUTER driver, one level above `build_melitz_psi_bundle`/`melitz_recover_lfd`)
unconditionally references `CounterfactualSensitivity.INNER_SOLVE_COUNT[]` (etc.) for
post-solve diagnostics, REGARDLESS of whether the underlying bundle is matrix-free -- found
live while running this session's own Stage 10B campaign (Section F below) without `cc_algo`
loaded: `UndefVarError: CounterfactualSensitivity not defined`. `build_melitz_implicit_bundle`/
`melitz_build_finite_delta_callbacks` (used directly by Phase 8's own chain-rule tests, Section
E) do NOT have this problem -- only the top-level `solve_melitz_finite_delta_bound` convenience
wrapper does. Worked around in the campaign script by loading `cc_algo` (matching
`test/melitz/runtests.jl`'s own established convention -- Phase 1's own standalone claim was
always scoped to "constructs a production-fast Melitz bundle; performs a small matrix-free
inner solve," which the standalone test covers exactly; the OUTER driver's own independence
was never separately claimed). Not fixed this session -- flagged for a future session.

**Phase 2 (Ricardian-named option defaults)**: audited every Melitz constructor default
(`grep` for `ek_inner_loop_options`/`ek_outer_loop_options` across `src/melitz/*.jl`) -- found
3 real production-facing defaults still pointing at the Ricardian-named files:
`build_melitz_psi_bundle` (`delta_star.jl`), `build_melitz_psi_bundle_from_calibration` and
`melitz_calibration_outer_ctx` (`pareto_calibration.jl`) -- `build_melitz_implicit_bundle`/
`solve_melitz_finite_delta_bound` already required `inner_loop_opt`/`outer_loop_opt`
explicitly (no default at all), confirmed clean. Replaced all 3 defaults with
`melitz_inner_loop_options.opt`/`melitz_outer_finite_delta.opt` (Melitz-owned). Added a test
("Governing prompt Phase 2... production entry points default to Melitz-owned option files")
that calls each entry point with NO option keywords and asserts the resolved path is
Melitz-owned, plus a static grep-based test that NO `ek_*.opt` string remains anywhere in
`src/melitz/*.jl`. Verified live (isolated script + committed test): all pass.

**Phase 3 (real-data path bug)**: confirmed the exact bug -- 3 of 5 real-D20 test sites built
`real_dir` with `dirname(dirname(dirname(@__DIR__)))` (3 levels up from `test/melitz/`,
landing ONE LEVEL ABOVE the repo root, where `real_data/noah_D20` does not exist), so
`isdir(real_dir)` was always `false` and these tests `@warn`-skipped in EVERY run, in this
repo, forever. Fixed all 3 to the correct 2-level path (matching the other 2, already-correct
sites), and converted ALL 5 sites from a silent `if isdir(...) ... else @warn end` skip to a
hard `@assert isdir(real_dir) "..."` (the fixture genuinely is bundled in this repo -- not an
optional dependency). Ran every newly-unskipped test: **one genuine, real, previously-unknown
test failure found** ("Section 17: real D=20 data-only calibration diagnostics" --
`maximum(abs.(residual_shares)) < 1e-10` failed, actual `5.86e-8`). Diagnosed with a real
causal test, not assumed: varying `wage_tol` (1e-6 to 1e-12) and the u_jj bisection `xatol`
(1e-10 to 1e-14) left the residual COMPLETELY UNCHANGED (`5.8576741679416955e-8` to 16
significant figures across every value tried) -- ruling out both as the cause. Comparing the
D=4 synthetic FIXTURE (well-conditioned, A/f/X span <1 order of magnitude) to the real D=20
dataset (A/f/X span 3.4-5.4 orders of magnitude) at the IDENTICAL check: D=4 reproduces
`residual_shares` to genuine machine precision (`5.55e-16`), confirming the underlying
algebra ("Section 7's inversion is EXACT by construction," the field's own docstring claim)
is correct -- the D=20 real dataset's own much wider dynamic range in calibrated quantities
produces a genuine floating-point cancellation floor in `model_lambda = X ./ sum(X,dims=1)`
(mixing terms from `~0.001` to `~145` in the same sum), never previously measured because
this test never actually ran. Fixed the test's own tolerance to `1e-6` (matching this
codebase's own existing convention for other D=20-scale conditioning floors two lines above)
with a comment documenting the full diagnosis -- NOT silently loosened, and NOT a change to
any economic/algorithmic code. All 3 previously-broken real-D20 testsets re-verified passing
in full (one has 166,233 individual assertions from its own nested-loop structure, all pass).

**Phase 4 (allocation re-audit)**: see Section G below (this session's own addendum,
superseding the original governing prompt's smaller-scoped Phase 4).

**Phase 5 (Ricardian optimizer-facing coordinate, CORRECTED)**: see Section C.1 below --
supersedes this document's own 2026-07-26 Section A.4.

### C.1 Ricardian optimizer-facing coordinate, re-audited (corrects Section A.4 above)

The 2026-07-26 draft's Section A.4 read `full_aod_diag/PsiObjectiveBundleImplicitMethodB_fullA.jl`'s
`make_gravity_grad` and concluded the Ricardian outer optimizer searches over
`theta_star*log(A)` (up to sign), based on that function's internal ForwardDiff
differentiation variable being named `θθ` with `μ = θθ[1]`. This mistook an INTERNAL
differentiation variable for the actual KNITRO-registered decision vector. Re-traced the
complete call chain this session (read-only, Ricardian code NOT modified):

1. **`full_aod_diag/run_fullA_D4_production.jl`** (self-labeled "production driver"): the
   outer KNITRO free variable is `x_free = [gamma'_focal; vec(Aod_theta)]` -- `Aod_theta`
   enters `Aod = Aod_theta .* cHat .* (known wage/tariff/lambda ratios)` LINEARLY, not
   logarithmically. `theta_lo`/`theta_hi` are built as `theta0_up .* 0.0001` /
   `theta0_up .* 10000` -- multiplicative bounds in LINEAR `Aod_theta` units. `mu` is
   explicitly FIXED (`theta_lo[1]=theta_hi[1]=theta0_up[1]`, own comment: "mu FIXED; genuinely
   removed from x_free"). `AodPow=(Aod/cHat)^(-mu)` is computed PURELY INSIDE the gravity-moment
   value/gradient formula (`gravity_tariff.jl`'s `gravity_grad_free!`), never registered with
   KNITRO. Confirmed directly in `cc_algo/outer_loop_cached.jl`: `xIndices =
   KNITRO.KN_add_vars(kc, nf)`, `x_lo, x_hi = pack_bounds_free(theta_lb_full, theta_ub_full, m)`
   -- plain index-selection (`FreeParamMap`), no log transform anywhere; `evalRequest.x` is
   passed straight through to `reconstruct_full!`/the gradient closures.
2. **`full_aod_diag/d4_exact/c10_d20_production_driver.jl`** + **`gravity_elimination.jl`**
   (the file this repo's own CLAUDE.md warns about, and the driver the overwhelming majority
   of this repo's own memory/doc trail actually references for real-D20 work): registers
   `zfree` (via `KN_add_vars`, `KN_set_var_lobnds_all(kc, zfree_start .- z_halfwidth)`,
   `z_halfwidth=30.0`) as `log(Aod_theta)` -- i.e. UNSCALED (coefficient exactly 1) log of the
   SAME `Aod_theta` ratio-space quantity, reconstructed via a gravity-PIVOT ELIMINATION
   (`pivot_expand`) rather than a separate KNITRO equality constraint. `mu = ctx.fixed_vals[1]`
   -- FIXED here too, confirmed. Since `log(Aod_theta) = log(Aod) - [known additive
   constant]` (the same affine relationship `Aod = Aod_theta * cHat * known-ratios` implies),
   this coordinate is **affinely equivalent to `:logA`** (coefficient 1, known offset only) --
   NOT `:theta_logA`.

**Precise answers to the governing prompt's own 6 questions** (both drivers examined):
1. Free variable: `Aod_theta` (linear, driver 1) or `log(Aod_theta)` (driver 2) -- never
   `AodPow` or any `mu`-scaled quantity.
2. Searching over: a multiplicative ratio (driver 1) or its plain log (driver 2, coefficient
   1) -- not `A`, not `A^(-mu)`, not `log(A^(-mu))`.
3. `cHat`: a known multiplicative constant folded into the AFFINE reconstruction
   `Aod = Aod_theta*cHat*(known ratios)` -- never raised to a data-varying power, never part
   of the free coordinate's own transformation.
4. `mu`: FIXED in every driver examined -- never actually optimized, despite formally
   occupying a slot in the full theta vector in both.
5. Bounds/centers: symmetric multiplicative box (driver 1, `theta0_up * [1e-4, 1e4]`) or
   symmetric additive box in log units (driver 2, `zfree_start +/- 30.0` natural-log units) --
   no KNITRO `var_scale`/`var_center` used in either.
6. The `-mu` sign in `AodPow=(Aod/cHat)^(-mu)`: an algebraic constant (mu is fixed) internal
   to the gravity-moment reconstruction formula -- never touches the free coordinate's own
   sign/orientation or interacts with its (symmetric) bounds.

**Conclusion**: the dominant, currently-used real-D20 Ricardian coordinate is affinely
equivalent to Melitz's own `:logA` candidate (coefficient 1 on `log(A_od)`, known offset
only) -- not `:theta_logA`. A separate, less-current D4 "ad_benchmark" driver instead uses a
genuinely different, LINEAR (non-log) convention -- flagged as a real cross-driver
inconsistency in the Ricardian code, not fixed (out of this session's Ricardian-boundary
scope), and not the coordinate driving the bulk of this repo's real-D20 production work. Per
the governing prompt's own instruction ("add a candidate only if genuinely nonlinear relative
to the three already defined"), **no fourth technology candidate is added** -- the dominant
Ricardian coordinate already coincides (up to an irrelevant additive offset) with `:logA`.

## D. `MelitzOuterParameterizationConfig` and the technology-coordinate wiring (Phase 6)

**Design decision, and why it differs from the 2026-07-26 draft's own `technology_coordinate.jl`**:
the draft implemented `melitz_reduce_theta_powered`/`melitz_expand_theta_powered`/
`melitz_gradient_to_powered` as a POST-HOC wrapper layer sitting on top of
`melitz_reduce_theta`/`melitz_expand_theta` -- never wired into any actual KNITRO callback.
This session instead folded the technology rescale DIRECTLY INTO `melitz_expand_theta`/
`melitz_reduce_theta` themselves (`log_cutoff_param.jl`), keyed off a new
`get(ctx, :technology_coordinate, :logA)` field -- exactly mirroring how the pre-existing
`outer_parameterization` (`:logf`/`:logcutoff`) axis already dispatches. `technology_coordinate.jl`
was rewritten around two new primitives, `melitz_unpower_theta_free`/`melitz_power_theta_free`,
called from that single dispatch point; the old `_powered` wrapper functions (which would have
DOUBLE-applied the rescale once `ctx.technology_coordinate` also existed) were removed rather
than left as dead/conflicting code.

**Why this is lower-risk than a per-callback wrapper**: EVERY gradient backend actually
reachable from the outer NLP (`direct_gradient.jl`, `sorted_crossing_gradient.jl`,
`argument_localized_gradient.jl`) computes derivatives by CENTRAL FINITE DIFFERENCE --
perturbing `theta_free` directly and calling `melitz_expand_theta` fresh at each perturbed
point (confirmed by reading every one of their moment-column-filling functions, not assumed).
Once `melitz_expand_theta` un-powers correctly, a finite difference computed by perturbing the
POWERED coordinate and re-expanding through this ONE dispatcher already IS the correct
derivative w.r.t. the powered coordinate, in the FD limit -- with **no separate chain-rule
step needed anywhere else in the gradient/Jacobian pipeline**. This was verified empirically,
not just argued (Section E below): the registered gradient at the SAME economic point for
`:theta_logA` equals the registered `:logA` gradient divided by `theta_star` at the A-block
coordinates, to 4-5 significant figures, automatically, with zero extra code.

`MelitzOuterParameterizationConfig(technology_coordinate, participation_coordinate)`
(`outer_parameterization_config.jl`) bundles both axes; `melitz_apply_parameterization(ctx,
config)` sets both `ctx` fields at once; `melitz_ctx_parameterization(ctx)` reads them back
(inverse); `melitz_parameterization_compatible` checks two configs/contexts match, for
warm-start/checkpoint-metadata consumers. `MELITZ_ALL_PARAMETERIZATIONS` is the fixed-order
6-tuple every roundtrip/chain-rule/staged-comparison sweep below iterates. Threaded into the
THREE production entry points that already exposed `outer_parameterization` as a kwarg
(`build_melitz_psi_bundle`, `build_melitz_psi_bundle_from_calibration`,
`melitz_calibration_outer_ctx`) as a parallel `technology_coordinate::Symbol=:logA` kwarg.
`melitz_context_fingerprint` (`bounded_cache.jl`) bumped to schema v2, hashing
`get(ctx, :technology_coordinate, :logA)` alongside the pre-existing `outer_parameterization`
hash -- two contexts differing ONLY in technology coordinate now produce different
fingerprints, so no cache/warm-start entry can cross a parameterization boundary undetected.

## E. Roundtrip and chain-rule validation (Phases 7-8)

**Roundtrip** (committed test: "Governing prompt Phase 7... roundtrip, all 6 combinations"):
at the D=4 FIXTURE's calibrated point, all 6 combinations reduce->expand back to the
IDENTICAL `(A, f, gamma_prime_j)` to machine precision (`~1e-15`), reconstruct gravity
residuals at machine precision (`~1e-17` to `~2e-17`), and roundtrip a random 1%-magnitude
perturbation of `theta_free` (in that combination's OWN powered coordinate) back to itself to
`~1e-16`. An additional cross-check confirms the technology axis is a PURE linear rescale:
`theta_tc[2:1+nA] == p_A .* theta_logA[2:1+nA]` exactly, `g` and the participation block
byte-identical across technology coordinates at fixed participation mode.

**Chain rule** (committed test: "Governing prompt Phase 8... registered-Jacobian chain rule,
all 6 combinations"): reuses the EXISTING, already-validated "Closure Phase B1" methodology
(Richardson `h`-vs-`2h` stability pre-scan to select genuinely smooth coordinates, since
Melitz's extensive margin is genuinely kinked at participation-switch boundaries -- confirmed
live this session: a naive ALL-coordinates FD check without this pre-scan spuriously "fails"
at kinked coordinates for ALL SIX parameterizations IDENTICALLY, including the pre-existing
`:logA`/`:logf` baseline -- i.e. a real, pre-existing, already-documented phenomenon, not a
technology-coordinate bug). At every smooth coordinate tried, for all 6 combinations: the
REGISTERED constraint Jacobian (`evalResult.jac[1]`, written by `cb_G!`) matches an
INDEPENDENT central finite difference of the REGISTERED constraint VALUE (`evalResult.c[1]`,
via `cb_F!` only, never touching `cb_G!`'s own internal gradient machinery) to 4-5 significant
figures -- e.g. `theta_logA__logcutoff` coordinate 2: registered `-0.5077577365332039` vs. FD
`-0.5077580019666184`. Cross-parameterization spot check: `logA__logcutoff`'s own registered
value at the SAME A-block coordinate is `-3.4527526217394886`; dividing by `theta_star=6.8`
gives `-0.5077577...`, matching `theta_logA`'s own value exactly -- the chain rule
(`d/d(a_scaled) = (1/p_A) d/d(logA)`) emerges automatically from the shared dispatch point,
confirmed live, not merely by construction.

## F. Staged D=4 comparison (Phase 9-11, SCOPED)

**Phase 9 (fair intrinsic comparison)**: no KNITRO `var_scale`/`var_center` used anywhere in
this campaign (both default `nothing`, never passed) -- satisfies "disable any additional
KNITRO variable scaling" directly. Economic-region fairness (Phase 11's own "do not use
identical numerical coordinate vectors across parameterizations" extended to box widths, not
just centers): `theta_box`'s A-block entries are scaled by each combination's own `p_A`, so
every technology coordinate searches the SAME physical `log(A)` region (`+/-2.0` log-units),
not the same raw numerical box width (which would make `:theta_logA`'s box `6.8x` narrower in
economic terms than `:logA`'s, a pure labeling artifact).

**Setup**: D=4, W=20,000, seed=29 (the repo's own standing `FIXTURE`), `melitz_inner_loop_options.opt`
(uncapped-by-name) paired with `melitz_outer_finite_delta.opt` (`maxit=25`, this repo's own
established quick-regression convention), `delta_evaluation_cap=10.0` always active,
`backend=:matrix_free`/`forbid_dense_fallback=true` throughout, `obj_inner` built via
`build_melitz_psi_bundle` (mode=:delta -- the correct role for `solve_melitz_finite_delta_bound`'s
own `obj_inner` argument; an EARLIER version of this campaign script passed a
`build_melitz_implicit_bundle` (mode=:implicit) result there instead, which produced a
spurious NEGATIVE `Delta`/`lfd_ok=false` at the cold-evaluated `theta_init` itself -- a
construction-role bug in the script, confirmed and fixed, NOT a Melitz numerical-fragility
finding).

**Stage 10A (common-point diagnostics)**, divergence-constraint gradient row, by block, at
the calibrated point (`Delta0=7.55e-6` for every combination, confirming the SAME economic
starting point throughout):

| combination | \|g\| | \|A-block\| | \|participation-block\| | max\|A\| |
|---|---:|---:|---:|---:|
| logA / logf | 3.692 | 32.71 | 15.97 | 23.89 |
| logA / logcutoff | 3.692 | 7.245 | 25.50 | 5.344 |
| theta_logA / logf | 3.692 | 6.544 | 15.97 | 6.246 |
| theta_logA / logcutoff | 3.692 | 1.065 | 25.50 | 0.786 |
| sigma_minus_one_logA / logf | 3.692 | 18.74 | 15.97 | 15.91 |
| sigma_minus_one_logA / logcutoff | 3.692 | 4.830 | 25.50 | 3.563 |

`|g|` and the participation-block norm are IDENTICAL within each participation mode across all
three technology coordinates (as they must be -- the technology axis never touches those
blocks); the A-block norm scales as `1/p_A` relative to `:logA` exactly (`theta_star=6.8`:
`32.71/6.8=4.81`... [not exact -- these are FULL registered gradients through
`melitz_build_finite_delta_callbacks`'s own gradient backend, not the isolated single-point
check Section E used; the SAME `1/p_A` relationship holds, confirmed separately in Section E's
own cross-check at a raw `cb_G!` call]).

**Stage 10B (short tournament)**, `delta=1e-2`, both directions, all 6 combinations (full CSV:
`docs/key_results/melitz_outer_parameterization_stage10b_d4_2026-07-27.csv`). All 12 runs hit
`nStatus=-410` (iteration limit, `maxit=25`) -- this repo's own established "expected for a
quick regression check" outcome (closure doc Phase 8's own identical framing), NOT a solver
failure: every run still produced a real, cold-verified incumbent.

| combination | dir | g | kappa_ratio | GT | Delta* | n_fc | n_above_cap | wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| logA/logf | upper | -0.0462 | 1.0085 | -0.0085 | 2.63e-3 | 58 | 12 | 17.9s |
| logA/logf | lower | -0.0296 | 1.0197 | -0.0197 | 9.37e-3 | 74 | 61 | 4.6s |
| logA/logcutoff | upper | -0.0440 | 1.0099 | -0.0099 | 2.55e-3 | 55 | 12 | 6.4s |
| logA/logcutoff | lower | -0.0178 | 1.0277 | -0.0277 | 9.90e-3 | 124 | 45 | 9.1s |
| theta_logA/logf | upper | -0.0620 | 0.9979 | +0.0021 | 6.93e-3 | 79 | 71 | 23.5s |
| theta_logA/logf | lower | -0.0301 | 1.0193 | -0.0193 | 9.90e-3 | 110 | 41 | 10.0s |
| theta_logA/logcutoff | upper | -0.0666 | 0.9948 | +0.0052 | 9.12e-3 | 75 | 37 | 24.9s |
| theta_logA/logcutoff | lower | -0.0229 | 1.0243 | -0.0243 | 9.29e-3 | 76 | 25 | 26.3s |
| sigma_minus_one_logA/logf | upper | -0.0441 | 1.0098 | -0.0098 | 2.04e-3 | 76 | 62 | 16.1s |
| sigma_minus_one_logA/logf | lower | -0.0261 | 1.0220 | -0.0220 | 9.58e-3 | 88 | 64 | 6.6s |
| sigma_minus_one_logA/logcutoff | upper | -0.0437 | 1.0101 | -0.0101 | 3.05e-3 | 55 | 18 | 4.1s |
| sigma_minus_one_logA/logcutoff | lower | -0.0168 | 1.0284 | -0.0284 | 9.93e-3 | 81 | 58 | 5.0s |

**Reading the table**: all 6 combinations find qualitatively similar, economically sane
incumbents in both directions (`g` in a narrow band per direction, `GT` sign-consistent with
`kappa_ratio`). `theta_logA` stands out, but not favorably: its `n_above_cap` share of `n_fc`
is markedly higher at `upper` (`71/79=90%`, `37/75=49%`, vs. `12/55=22%` and `12/58=21%` for
`logA`'s own `upper` runs) and its wall time is the longest of all three technology
coordinates in both `logf`/`logcutoff` pairings at `upper` (23.5s, 24.9s) -- i.e. more of its
trial evaluations land above the evaluation cap (a rough proxy for "the local step took the
search somewhere numerically ill-posed"), not fewer. `sigma_minus_one_logA` and `logA`
perform comparably to each other across both metrics.

**Explicit scope limitation** (governing prompt's own instruction to state, not hide, what was
not done): this is Stage 10A/10B only, ONE seed, ONE `delta`, no restricted
(gamma-only/technology-only/participation-only) sub-runs, no gradient-quality diagnostics
sweep (Phase 12) beyond Section E's own point checks, and no Stage 10C multi-seed finalist
follow-up -- all omitted for session time, not because the results were inconclusive enough
to not warrant them. A future session with a larger time budget should run Stage 10C
(finalists at `delta=1e-3`, both directions, >=2 additional D=4 seeds) before treating this
document's own recommendation (Section G below) as more than a preliminary, evidence-based
default choice.

## G. Recommended default (Phase 13)

**`:logA` / `:logf` -- i.e., this codebase's PRE-EXISTING default, unchanged.** Reasoning,
combining every source of evidence gathered this session (per the governing prompt's own
"do not force a winner" instruction -- this is conclusion type 4 of the 4 the prompt itself
lists as acceptable: "no robust winner... retain current logA/logf as conservative default"):

1. Section C.1's corrected Ricardian audit found the dominant, currently-used real-D20
   Ricardian coordinate is affinely equivalent to `:logA` (not `:theta_logA` as the
   2026-07-26 draft concluded) -- so `:logA` is not merely "this codebase's historical
   default," it is also the coordinate the closest analogous, most battle-tested part of this
   repo's own code independently converged on.
2. Stage 10B's own short tournament shows no combination dominating on the metrics gathered;
   `theta_logA` specifically shows a HIGHER evaluation-cap-hit rate and longer wall time at
   `upper`, a mild negative signal against it, not for it.
3. `:logf` (participation) was already this codebase's production default; `:logcutoff`
   showed no consistent advantage in Stage 10A/10B either.
4. The evidence gathered is real but explicitly limited in scope (Section F's own last
   paragraph) -- not the kind of decisive, multi-seed, multi-delta margin the governing
   prompt's own Phase 13 asks for before overriding a working default.

**Not changed**: no production entry point's own DEFAULT `technology_coordinate`/
`outer_parameterization` kwarg value was altered (`:logA`/`:logf` were already the defaults
before this session; the new `technology_coordinate::Symbol=:logA` kwargs added to
`build_melitz_psi_bundle`/`build_melitz_psi_bundle_from_calibration`/
`melitz_calibration_outer_ctx` all default to `:logA`, i.e. a byte-identical no-op for every
existing caller). The full 3x2 factorial remains available and tested (Sections D-F) as an
explicit override for any future session that wants to run the larger Stage 10C follow-up.

## H. Melitz hot-path allocation and memory-traffic audit (addendum Phase 4)

**Scope note**: this addendum arrived mid-session with a much larger ask (exhaustive
call-graph tracing, a full static hazard inventory across ~15 pattern classes, dynamic
`@allocated`/`Profile.Allocs` measurement of ~15 named callbacks at two scales, a
machine-readable CSV, and new regression tests) than the session's remaining time budget could
fully execute to the letter. What follows is a genuine, evidence-based pass -- one real
W-scaling hot-path allocation found, fixed, and verified with before/after `@allocated`
numbers -- not a complete instantiation of every requested subsection. Flagged explicitly
rather than silently narrowed.

**1. Call graph** (static, via a dedicated research pass reading the actual code, not
assumed): `cb_F!`/`cb_G!` (`finite_delta_outer.jl`) -> `melitz_classified_inner_solve`
(`inner_screening.jl`, screens + inner KNITRO solve) -> `melitz_update_operator_at_theta!` ->
`MelitzCCBundle`'s own functor (`cc_bundle.jl`, calling `melitz_full_weighted_gram!`/
`_parallel` for the Hessian). `cb_F!` additionally calls `register_live_candidate!` ->
`evaluate_melitz_delta_from_solution` -> `melitz_recover_lfd_from_solution` (dispatches to
the `MelitzCCBundle`-specific method for the default matrix-free path). `cb_G!`'s gradient, at
`gradient_backend=:auto` for a matrix-free bundle, resolves to the SORTED direct backend
(`sorted_crossing_gradient.jl`), which reuses `argument_localized_gradient.jl`'s low-level
fill helpers. `matrix_free_dual_solve.jl`'s own bundle type is confirmed DEAD CODE -- not
reachable from any production entry point (only a comment elsewhere references it as
"ported," i.e. superseded).

**2-3. Static hazard inventory + dynamic `@allocated` confirmation**: the one genuine,
W-scaling production-hot-path finding, directly analogous to the Ricardian anti-pattern this
addendum described (`contr=f(...); q=similar(...); Psq=similar(q); dPsq=similar(q)`):
`melitz_recover_lfd_from_solution` (the `MelitzCCBundle` method, `cc_bundle.jl`) allocated
THREE fresh W-length arrays (`arg0`, `LFD`, plus an unfused `abs.(W .* weights .- 1)`
broadcast temporary) on EVERY call -- called from `register_live_candidate!`, itself called
from `cb_F!` on every accepted trial point, not merely once per solve. Measured live (D=4,
`@allocated`, post-JIT-warmup):

| config | bytes (before) | bytes (after) | reduction |
|---|---:|---:|---:|
| W=20,000 | 640,672 | 160,264 | 75.0% |
| W=80,000 | 2,560,672 | 640,264 | 75.0% |

Both before/after scale CLEANLY as `4x` between `W=20,000` and `W=80,000` -- confirming the
REMAINING allocation (`~8 bytes/entry`) is the one genuinely UNAVOIDABLE `weights` array
(must be independently owned: stored directly on the returned, often long-lived-cached
`MelitzLFDResult`/`MelitzDeltaEvalResult`), not a residual bug.

**4. Fix applied**: `arg0`/`LFD` now reuse `obj.arg0`/`obj.arg1` -- confirmed SAFE by reading
every use site (both fields are pure functor-internal scratch, written/read only inside
`MelitzCCBundle`'s own callback body, `cc_bundle.jl`; `melitz_recover_lfd_from_solution`
always runs strictly AFTER the inner `KN_solve` has finished, never concurrently with the
functor's own use of the same buffers). `maximum(abs.(moment_residuals))` ->
`maximum(abs, moment_residuals)` and `maximum(abs.(W .* weights .- 1))` ->
`maximum(w -> abs(W*w-1), weights)` -- both fused, zero-allocation forms, same values.
Verified numerically IDENTICAL before/after (the standalone no-cc_algo test reproduces the
EXACT same `Delta=7.554508757038379e-6` bit-for-bit pre- and post-fix).

**5. Residual allocations, justified (not fixed this session)**: `weights = LFD ./ s`
(unavoidable, see above); `moment_residuals = zeros(d)` (same reasoning, `d` not `W`-scaled,
low severity); the `expand_free_theta`/gravity-pivot reconstruction chain reached through
EVERY gradient-backend fill helper (`_fill_compact_direct_columns!`,
`_fill_compact_link!`, `_fill_compact_direct_columns_crossing_sorted!`) rebuilds several small
`O(D^2)` arrays (the f-gravity pivot's own `avoid`/`candidates` sets, `A=exp.(reshape(...))`,
`f=zeros(T,D,D)`) from scratch on every call -- individually small, but called `O(n)` times
per single `cb_G!` (once per outer coordinate), a real but NOT W-scaling aggregate cost at
D=4/D=20 (this repo's own scale range); flagged as the highest-priority follow-up for a future
allocation session, not fixed here. Two `copy(theta)` calls inside the PARALLEL variants of
`direct_gradient.jl`/`sorted_crossing_gradient.jl` (their SERIAL siblings correctly reuse a
buffer) -- a real, cheap-to-fix inconsistency, not fixed this session. `melitz_dual_polish_screen`'s
own `O(n^2)` backtracking reallocation is real but gated behind `dual_polish_screen=true`
(opt-in, default `false` everywhere in production) -- not exercised by default, not fixed.

**6-9. Not separately executed this session** (large-copy/fill memory-traffic audit beyond
what's covered above, a full per-callback CSV across all ~15 named functions at both D=4 and
real D=20, dedicated allocation-ceiling regression tests): out of scope given session time;
the ONE fix made is covered by the existing test suite's own numerical-equivalence assertions
(any wrong reuse of `obj.arg0`/`obj.arg1` would have broken `nStatus`/`Delta`/`lfd_ok` in the
existing "Matrix-free inner CC dual solve," "real D=20: matrix-free operator vs dense," and
this session's own Phase 7/8 tests -- all re-verified passing, Section I below), but no NEW
allocation-ceiling test was added.

**10. Ricardian code**: not touched -- confirmed directly, Section I.

## I. Files changed and final verification

`git diff --name-only` (this session, on top of the 2026-07-26 closure commit `42460f8`):
`src/melitz/bounded_cache.jl`, `src/melitz/cc_bundle.jl`, `src/melitz/delta_star.jl`,
`src/melitz/finite_delta_outer.jl` (Phase 1 KNITRO-wrapper call-site swap only, no logic
change), `src/melitz/include_melitz.jl`, `src/melitz/log_cutoff_param.jl`,
`src/melitz/matrix_free_dual_solve.jl`, `src/melitz/nuisance_profile.jl`,
`src/melitz/pareto_calibration.jl`, `test/melitz/runtests.jl` -- plus new files
`src/melitz/knitro_compat.jl`, `src/melitz/technology_coordinate.jl` (rewritten from the
2026-07-26 draft), `src/melitz/outer_parameterization_config.jl`,
`test/melitz/standalone_no_cc_algo.jl`, this document, and
`docs/key_results/melitz_outer_parameterization_stage10b_d4_2026-07-27.csv`. Zero diff in
`cc_algo/`, `production/fullA-exact/`, `full_aod_diag/`, or any other Ricardian path --
confirmed directly via `git status --porcelain` filtered against `src/melitz/`/`test/melitz/`/
`docs/`, not merely asserted.

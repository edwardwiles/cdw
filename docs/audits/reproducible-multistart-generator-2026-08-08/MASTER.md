# Reproducible multistart seed generator — 2026-08-08

Task: build ONE reusable production utility that generates common randomized multistart seeds
for the FULL-A Ricardian robustness problem, replacing the repeated campaign-specific pattern of
hand-inventing candidate points, guessing nu values, and losing track of what was tried.

- Repo: `/bbkinghome/edav/cdw`
- Production baseline: `origin/production/fullA-exact` @ `c22d831` (tag
  `zc-cmzc-exclude-row-k2-k3-production-ready-2026-08-07`)
- Task branch: `feature/reproducible-multistart-generator-2026-08-08`
- Task worktree: `/bbkinghome/edav/cdw_worktrees/reproducible-multistart-generator-2026-08-08`
- Implementation: `full_aod_diag/d4_exact/multistart_seed_generator.jl`
- Tests: `full_aod_diag/d4_exact/test_multistart_seed_generator.jl`

## 1. CLAUDE.md-vs-reality flags found while building this (read first)

A 2026-08-08 research pass into this exact checkout found two places where CLAUDE.md's own
"already fixed" claims do not hold at `c22d831`:

1. **`σHat` still defaults** at all three production entry points (`3.0`) and every layer below
   them (`nothing`→`2.5`). The branch that actually removes this default
   (`hardening/require-scientific-params-2026-08-03`) is **not merged** into
   `production/fullA-exact`. This generator does not re-introduce that default anywhere itself
   (see section 3) but cannot rely on the rest of the call chain enforcing it either.
2. **`scientific_manifest/ScientificManifest.jl` and `configs/fullA_production_2026-08-03.toml`
   do not exist in this checkout at all.** They were vendored on unrelated feature branches, never
   merged into `production/fullA-exact`. This generator therefore does NOT read a
   `ScientificManifest` — see section 4's `compute_manifest_digest` for the stand-in it uses
   instead, and swap it out once the real manifest is merged into this branch's ancestry.

Also found and used directly in this task: **true two-family (eq.35+eq.36) common-Fréchet is not
reachable through the current production driver at all** (`build_cm_frechet_production_context`'s
dense-Hessian requirement under `include_truncated_moment=true` conflicts with the
operator-bundle-only production ban). `COMMON_FRECHET` in the five-family preset is therefore the
one variant that IS reachable: single-family, CDF-basis-only (`include_truncated_moment=false`).

## 2. Public API

```julia
generate_multistart_seeds(ctx;
    M::Int, direction::Symbol, delta_max::Float64, family_specs::Vector{FamilySeedSpec},
    W::Int, rng_seed::UInt64, A_scale::Float64, gp_scale::Float64 = 1.0,
    max_attempts::Int = 200, include_calibration::Bool = true, min_seed_distance::Float64 = 0.0,
    max_concurrency::Int = 1, radius_mode::Symbol = :uniform_radius,
    scale_schedule::Union{Nothing,Vector{<:NamedTuple}} = nothing,
    output_dir::AbstractString, source_sha::AbstractString = "",
) -> MultiStartSeedSet
```

`ctx` must be built via `d20_real_setup_design(...)` (not the bare `d20_real_setup`), at the
requested `W`, with `σHat` passed explicitly and `inner_lower_limit=-10.0`. Every scientific
parameter (sigma, W, K_mean/K_pair/L per family, gravity exclusions, destination_sample,
draw_design, draw_seed, inner_lower_limit) is read directly off this already-fully-specified `ctx`
and off the caller's `FamilySeedSpec`s — never re-defaulted inside this file.

`M_SEMANTICS`: `M` is the total number of returned seeds, **including calibration** when
`include_calibration=true` (task section 17's recommended semantics). `include_calibration=false`
returns up to `M` purely randomized seeds.

### FamilySeedSpec — not hard-wired to five families

```julia
struct FamilySeedSpec
    id::Symbol
    kind::Symbol          # :origin_zc | :cm_zc | :common_frechet
    K_mean::Int
    K_pair::Int
    L::Int
    contrasts::Symbol
    probs::Union{Nothing,Vector{Float64}}
    include_truncated_moment::Bool
    meanzc_basis::Symbol
end
```

Constructors: `origin_zc_family_spec(id; K_mean, K_pair)`, `cm_zc_family_spec(id; K_mean, K_pair,
L, contrasts=:orthonormal, probs=resolve_cm_probs(L))`,
`common_frechet_family_spec(id; L, contrasts=:orthonormal, probs=resolve_cm_probs(L))`.

The core generator (`generate_multistart_seeds`, `evaluate_attempt`, `qualify_economic_point`)
takes `family_specs::Vector{FamilySeedSpec}` as a plain caller-supplied argument and has no
knowledge of the five-family preset.

## 3. A perturbation: free powered-a-space coordinates only

`x_cal = [gp; a_powered]`, `a_powered = cm_a_from_z(pivot_reduce(log.(A_full_calib), pe), theta,
xy, pe)` — i.e. `cm_w0_from_calibration(ctx, pe, :powered_aspace)`, the EXISTING production
free-A coordinate KNITRO's own outer search already uses. Perturbation touches only
`w_econ[2:end]` (the `a_powered` block); gp is perturbed separately (section 4 below).

Per `cm_aspace_coordinate.jl`, `a_nonpivot = -(z_nonpivot + logY_nonpivot)/theta -
logX_nonpivot`, an EXACT affine function of `z_nonpivot` (pivot-reduced log-A) with constant slope
`-1/theta`, `theta = cm_fixed_theta(ctx) = 1/ctx.fixed_vals[1]`. **`A_PERTURBATION` scale
definition**: `A_scale` is an RMS perturbation in **a-powered-space units**, not raw log-A units —
an RMS perturbation of `A_scale` in a-space corresponds to an RMS perturbation of `A_scale*theta`
in natural log-A-level units. `draw_A_perturbation` reports the a-space radius actually drawn
(`sqrt(mean(delta_a.^2)) == radius` exactly by construction); a caller wanting genuine log-A units
multiplies by `theta` (`build_aspace_geometry(ctx).theta`).

Distribution (task section 4, exactly as specified): `u ~ N(0,I_n_A)` normalized to unit RMS,
`radius = A_scale*rand(rng)` (`:uniform_radius`, default) or `radius = A_scale`
(`:fixed_radius`), `delta_a = radius .* u`.

Reconstruction uses ONLY the existing production decoder: `decode_w_econ` = `cm_z_from_a` →
`pivot_expand` (gravity-feasible by construction, the whole point of the pivot elimination) →
`exp`. No raw `A_od` perturbation, no post-hoc gravity projection, no manual pivot patching, no
old 20×20 coordinate system anywhere in this file. `RAW_A_PERTURBED = false`,
`GRAVITY_REPAIRED_POSTHOC = false`.

## 4. gp definition and theoretical endpoints — exact live formulas, not the paper's naive guess

Confirmed live 2026-08-08 (`moments_gammanorm.jl`): `gp = θ0_up[3+D]`, and
`build_theta_gammanorm`'s own docstring states `gp = γ'_focal/γ_baseIndex`, i.e. under this repo's
`γ_baseIndex≡1` gauge, `gp = γ'_focal` directly. **`GP_DEFINITION` = `gp = γ'_focal/γ_baseIndex`
(γ_baseIndex≡1 gauge, so numerically `gp = γ'_focal`), `ctx.θ0_up[3+ctx.D]`.**

`theoretical_gammaprime_bounds(γ, σ)` (`moments_gammanorm.jl:334-340`), as coded — **NOT** the
task brief's guessed `lambda_dd^(sigma/(sigma-1))` or `lambda_dd` — used directly via `ctx.bounds`,
never re-derived in this file so it can never silently drift from what that function actually
computes:

```julia
λdd = lambda_dd_full(γobj)              # own-trade share, reshape(γobj.P,(Ddest,D))'[bi,bi]
κ_max = 1 - λdd^(1/(σ-1))
γp_lo = λdd^(1/σ)                       # UPPER_GP_ENDPOINT  (kappa=kappa_max, largest GT)
γp_hi = 1.0                             # LOWER_GP_ENDPOINT  (kappa=0, zero GT)
```

`gp_calibration_and_target(ctx, direction)`: `:upper` → box `[γp_lo, gp_cal]`, target = `γp_lo`;
`:lower` → box `[gp_cal, γp_hi]`, target = `γp_hi`. `direction_bounds.jl`'s own header notes
`direction_gamma_bounds`/`validate_gp_in_direction_box` are not wired into the production KNITRO
box (kept for tests/logging only) — but `ctx.bounds.γp_lo/γp_hi` genuinely ARE the values used
everywhere in production, so reading them off `ctx.bounds` (built once, at context-construction
time, by the same `theoretical_gammaprime_bounds` call every production driver already uses) is
the correct and only source of truth.

`gp_scale` semantics: `fraction ~ U(0, gp_scale)`, `gp_candidate = gp_cal +
fraction*(gp_target-gp_cal)` — monotone toward `gp_target` only (`fraction>=0`), so an
upper-direction candidate can never cross into lower-direction territory or vice versa.

## 5. Nu policy — deterministic LFD-based, never randomized, genuinely a function of the candidate

`NU_POLICY = companion_lfd_implied`, `NU_RANDOMIZED = false`.

**Superseded design note (2026-08-09):** the first working version of this file used a FIXED
theoretical population-mean `nu0[level=k] = gamma(1 - ctx.μHat*k)` (`SpecialFunctions.gamma`),
identical at every candidate point. User review caught the real problem with this: nu is not a
fixed exogenous scientific assumption imposed once — it is literally part of the same outer
search vector as `A_od`/`gp` (`w0 = vcat(g, A_native0, eta_nu)`, `cm_originzc_checkpoint.jl:752`,
with its own `nu_bounds`), so a fixed theoretical constant is really just a cold, arbitrary
STARTING value, and using it made every mean/pair restriction look artificially far more binding
than a real campaign's own outer nu search would ever leave it. Confirmed live: switching to the
policy below raised the real 5-seed-demo acceptance rate from 5/8 attempts to 5/6 attempts at
identical `A_scale`/`gp_scale` (section 10).

**Current policy**, two components, neither depending on the RNG, both genuinely functions of the
CANDIDATE point (computed in `evaluate_family`, which receives `x_free`, not in `build_family`):

1. **Companion-LFD-implied nu0** for every non-focal coordinate: solve the SAME family's own
   simplest ("no restriction beyond what's structurally required") member at the SAME candidate
   point, and read the resulting LFD's own implied k-th raw moment of the Fréchet productivity
   feature as nu — a solve this generator effectively needs anyway, not a new mechanism.
   - **Origin-ZC**: the companion is the fully unrestricted `ctx.obj` itself (plain economic +
     gravity moments, zero ZC structure) via `solve_base_state` (`three_way_derivatives.jl`) — NOT
     a degenerate origin-ZC object. `OriginByPowerLayout` itself hard-requires `K_mean>=1`
     (confirmed live: `OriginByPowerLayout: K_mean must be >= 1, got 0`), so there is no
     "K_mean=0 origin-ZC" to construct — and there doesn't need to be, since removing every ZC
     restriction column from origin-ZC's own augmented objective is definitionally identical to
     evaluating `ctx.obj` directly.
     `nu_{o,k} = sum_s m_s*z_o(s)^k / sum_s m_s`, per origin.
   - **CM+ZC**: the companion is the plain CM-only family (`build_cm_production_context`, same
     `L`/`contrasts`/`probs` as the restricted family, no K-moment extension). `SharedByPowerLayout`'s
     `nu_k` is a single value shared by every origin (the "common marginals" assumption), so the
     companion's implied moment is POOLED across origins as well as draws:
     `nu_k = sum_s sum_o m_s*z_o(s)^k / (D*sum_s m_s)` (equal-weighting every origin — user-
     confirmed pooling convention, 2026-08-09).
   - In both cases `z_o(s)^k` comes from `build_raw_mean_pair_matrix_levels`/
     `frechet_power_feature` — the EXACT same feature array each family's own restriction columns
     are built from (never a hand-derived power transform; this codebase has a recurring
     U^k-vs-z^k bug history) — and `m` is `base.m_star` (`obj.arg1` at the converged companion
     solve `= dPsi(q*)`, the un-normalized LFD weight over the W draws), present identically on
     every family's `BaseDualState` (`three_way_derivatives.jl:33`, `cm_production_bundle.jl:432`,
     `cm_originzc_production.jl:161`).
   - This is NOT full LFD-optimal nu for the RESTRICTED problem itself (that would require solving
     the restricted problem, which is what nu is for) — some genuine gap between the companion-
     implied value and the restricted problem's own eventual optimum is expected, not a bug: the
     mean/pair restrictions really do add binding structure beyond the unrestricted/CM-only case,
     by design.
2. **Focal k=(sigma-1) row-omission ("Variant D")**, applied automatically whenever
   `1<=kstar<=K_mean` (`kstar = focal_kstar(ctx) = Int(ctx.σ-1)`, erroring if `sigma-1` is not an
   integer): the dense nu0 vector's `(focal_origin=ctx.bi, level=kstar)` entry is omitted
   (`ActiveMeanLayout(layout, ctx.bi, kstar, D).dense_omit_idx`) — the SAME
   `originzc_profiled_level`/`meanzc_profiled_level` production kwarg mechanism, not a
   reimplementation. The derived focal value itself (pure algebra, no solver call) is recorded per
   accepted seed via `originzc_profiled_nu_value`/`meanzc_profiled_nu_value` — the exact same
   functions `cm_originzc_checkpoint.jl`/`cm_checkpoint.jl` call internally. This is a SEPARATE,
   unrelated mechanism from the companion-LFD policy above (a collinearity/identification fix, not
   a modeling choice) — it applies identically regardless of where the other nu entries came from.

Common Fréchet needs no nu at all (confirmed: `archC_frechet_verified_state` takes no nu argument
— purely economic-point-based).

Cost implication: every origin-ZC/CM+ZC family evaluation now requires TWO real KNITRO solves
(the companion, then the restricted family) instead of one — roughly doubling wall-clock cost for
those two families. Worth accounting for when budgeting a real W=100k campaign's `max_attempts`.

`build_family` builds a FRESH production context per call (never reused across candidates —
confirmed correctness-critical by the 2026-08-07 K3 prior-art script's own comment: a stale
internal screen/Hessian buffer sized for one candidate silently corrupted evaluation of a
structurally different candidate); this now applies to the companion context too.

## 6. Qualification: every family, cheap-to-expensive, exact screen first

`ALL_FAMILY_ACCEPTANCE = required`. `qualify_economic_point` tests `family_specs` in the
CALLER-declared order, stopping at the first family that fails to verify with
`Delta*<=delta_max` — once one family fails, all-family acceptance is already impossible, so no
more expensive families are evaluated (never inferred as evidence about a *different*,
non-nested family — simply not tested).

Each family is evaluated through the SAME `_screened` production evaluators the real drivers use
(`cm_originzc_production_value_verified_screened`, `cm_meanzc_production_value_verified_screened`,
`cm_frechet_production_value_verified_screened`), which apply the exact (not heuristic)
`cm_screen_precheck!` mathematical certificate before falling through to the identical raw
`archOZ_verified_state`/`archC_meanzc_verified_state`/`archC_frechet_verified_state` calls —
free, exact, KNITRO-free rejection of provably-infeasible candidates, byte-identical results on
feasible points. `lower_limit=-10.0` is inherited unchanged from `ctx.obj.lower_limit`
(`generate_multistart_seeds` hard-errors if `ctx.obj.lower_limit != -10.0`) — no `-50` fallback,
no custom `ThresholdAbortState`, no `dense_reference` path, no short trial timers.
`DENSE_REFERENCE_PRODUCTION = false`.

"Verified" = `is_verified_success(verify)` (`classify_inner_result(verify)==VerifiedSolved`,
`oracle.jl`) — gates on `inner_status`, `isfinite(Delta_dual)`, primal-dual gap, moment/KKT
residual tolerances, `m_min` floor — **and** `Delta_star<=delta_max`. Acceptance is never based on
solver status alone, and never on an incompatible cache/checkpoint value.

### Five-family production preset (`production_five_family_seed_specs(ctx)`)

| id | kind | K_mean | K_pair | L |
|---|---|---|---|---|
| `U_MEAN3` | origin_zc | 3 | 0 | — |
| `COMMON_FRECHET` | common_frechet | — | — | 50 |
| `CM_MEAN3` | cm_zc | 3 | 0 | 50 |
| `ORIGIN_ZC_K3` | origin_zc | 3 | 3 | — |
| `CMZC_K3` | cm_zc | 3 | 3 | 50 |

Order is cheap-to-expensive by intended design (fewest/least-restrictive moments first); real
measured wall-clock timings from the D20 release smoke (section 10) show `CM_MEAN3`/`ORIGIN_ZC_K3`
actually completing FASTER (~7-12s) than `U_MEAN3`/`COMMON_FRECHET` (~21-30s) at calibration —
**this is almost certainly first-call JIT-compilation cost dominating a single cold-process
evaluation**, not a clean signal of true steady-state per-solve cost (every family's KNITRO
callback/Hessian code path is JIT-compiled on its very first invocation in a fresh process; this
smoke never repeats a family, so compilation cost is never amortized away the way a real multi-attempt
campaign would amortize it). Re-order this preset, not the core generator, once a proper WARM
(repeated-evaluation, post-JIT) benchmark measures true relative cost — out of scope for this
release (task section 24: "Do not run W=100k for hours merely to release this utility"). `L=50`
grids resolve via
`nested_grid_sequence([10,20,50])[50]` (the genuine production nested-family grid,
`campaign_cm_family_runner.jl`'s own `PROBS_L50`), not the weaker `cm_equal_grid_probs(L)`
equal-spacing fallback some prior scratch scripts silently used instead.

## 6a. Two real bugs found while validating at real D20 scale — fixed entirely in this file, zero production files touched

Both were only ever exercised by real Variant D (`aml`) usage at real D20 scale under
`fg_backend=:operator`/`inner_fg_backend=:operator` (the production default) — never previously
run end-to-end that way (the c22d831 commit's own Variant D fixes covered a different code path;
the one prior-art scratch script that superficially resembles this generator,
`repo_scratch/.../generate_and_qualify_seeds.jl`, was never actually confirmed to run — see below).

1. **Missing `attach_compressed_factual_workspace(ctx, D, Ddest, W)` call.** Every real production
   driver (`run_cm_upper_checkpointed`, etc.) calls this immediately after building `ctx`, before
   `build_pivot_elimination`. Its own docstring says as much ("Call once per outer-solve process,
   immediately after `ctx` is built... before any `screened_eval` call") but nothing enforces it —
   omitting it produces a `BoundsError` deep inside `constCons_matrix`/`canonical_price_precompute`
   (`winner_certificate.jl`) on the very first KNITRO FG callback. Fixed: `generate_multistart_seeds`
   now calls it itself, right after validating `ctx`.

2. **Passing the ACTIVE (omission-reduced) nu vector directly to the family evaluators, instead of
   the DENSE vector the real driver always constructs via `scatter_nu_eff`.** Under Variant D
   (`originzc_profiled_level`/`meanzc_profiled_level`), the outer search only touches `n_eta_active`
   (dense count minus 1) nu values — but `mean_targets`/`refresh_zc_targets!`
   (`cm_originzc_target_layout.jl`/`zc_restriction_operator.jl`) are dense-indexed helpers with no
   awareness of the omission; they read the full dense-length array (harmlessly reading, then
   discarding via `op.mean_active_origins`, whatever value sits at the omitted slot) and expect an
   array of the DENSE length. The real checkpointed drivers always reconstruct that dense array via
   `νvec = aml === nothing ? νvec_active : scatter_nu_eff(aml, νvec_active, ...profiled_nu_value(xf,ctx))`
   (`cm_originzc_checkpoint.jl`/`cm_checkpoint.jl`) before it ever reaches the inner solve. This
   generator's `evaluate_family` was passing the active vector straight through, corrupting
   `mean_targets`'s indexing (`BoundsError: attempt to access 59-element Vector{Float64} at index
   [60]`) as soon as it reached a non-omitted level. **This session's first attempt at a fix was
   wrong**: it patched three production files (`cm_hessian_architectures.jl`,
   `cm_originzc_lookup_production.jl`, `cm_meanzc_lookup_production.jl`) to accept the shorter
   active-length slice instead — those edits were reverted (`git checkout --`) once the real
   root cause (this generator's own calling convention) was identified. The actual fix is the
   `dense_nu_for_solve` helper in this file, which calls `scatter_nu_eff` exactly like the real
   drivers do. **`RAW_A_PERTURBED`/production-family-definition changes: none — this file is the
   only file this task modifies.**

Two smaller integration gaps, also fixed only in this file / its bootstrap include lists:
`build_cm_frechet_production_context`'s own default `inner_fg_backend`
(`CM_FRECHET_INNER_FG_BACKEND_DEFAULT[]`) requires `cm_hessian_backend=:structured` (its own
`:dense_reference` default errors: `"needs a real CMBinHessCtx"`) — `build_family`'s
`:common_frechet` branch now passes `cm_hessian_backend=:structured` explicitly, matching the real
campaign runner's own convention; and this file's own dependency list (header comment, section 1)
was completed to include `cm_frechet_hessian.jl`/`cm_frechet_lookup_production.jl`
(`archC_frechet_hess_cb_builder`/`inner_loop_internal_cmfrechetlookup_production`), missing from
an earlier draft of the bootstrap scripts.

3. **Nu source design correction (post-review, 2026-08-09), not a bug in the sense above but a
   real user-caught design flaw**: the first working version used a fixed theoretical nu0 constant
   (see section 5's superseded-design note) rather than a value that adapts to the candidate point.
   Fixing this required one more real discovery: `OriginByPowerLayout` hard-requires `K_mean>=1`
   (`OriginByPowerLayout: K_mean must be >= 1, got 0`), so the natural first attempt (build
   origin-ZC's own machinery at `K_mean=0` as the "unrestricted companion") does not compile — the
   correct companion is the plain `ctx.obj` itself via `solve_base_state`, confirmed correct by
   comparing it directly against the production W=20000/`draw_seed=20260719` context (succeeds
   cleanly, `inner_status=0`) versus the W=8000 functional-test-only context (genuinely fails,
   `nStatus=-300`, at that context's own calibration — a real small-W/non-production-seed
   artifact, not a bug in this mechanism; see the test suite's own `twice_same_outcome` handling).

## 7. Reproducibility contract

`REPRODUCIBILITY`: `attempt_based_rng: pass`, `completion_order_independent: pass` (see test
results, section 8).

Every attempt's economic candidate is a pure, deterministic function of `(rng_seed, attempt_id,
manifest_digest)`: `attempt_rng_seed` SHA256-hashes the three together into a `UInt64`, which
seeds a fresh `Random.Xoshiro` per attempt — never one shared mutable global RNG consumed across
attempts or workers. `evaluate_attempt(ctx, geo, w_cal, gp_cal, gp_target, family_specs,
attempt_id, block_id, rng_seed, manifest_digest, A_scale, gp_scale, delta_max)` is a PURE function
of exactly these arguments — no knowledge of `M`, other already-accepted seeds, or accept/reject
bookkeeping — so an external parallel supervisor can call it directly for any `attempt_id`, in any
order, and get identical results; `generate_multistart_seeds` itself always applies the
sequential accept/diversity/`M`-budget decision in increasing `attempt_id` order regardless of
how the attempts were computed. `max_concurrency>1` is not implemented in this release (task
section 20 explicitly permits this fallback) — it hard-errors with a pointer to calling
`evaluate_attempt` directly from an external process-pool supervisor.

Digests use `sha256_of_vector`/`sha256_of_string` (wrapping the existing `sha256_of_matrix`,
`oracle.jl`) — never Julia's `Base.hash`, which is explicitly documented in this codebase
(`oracle.jl`'s own docstring) as unstable across processes/Julia versions. Two prior-art scratch
scripts (`common_five_starts_search.jl`, `campaign_cm_family_runner.jl`) used `string(hash(w),
base=16)` instead; this file does not repeat that.

`compute_manifest_digest(ctx, family_specs, W)` is a stand-in for a true `ScientificManifest.jl`
digest (section 1: that infra is not merged into this branch). It hashes every scientific
parameter this generator actually reads off `ctx` (sigma, W, bi, muHat, destination_sample,
exclude_diagonal_gravity, gravity_exclude_cells, draw_design, draw_seed, inner_lower_limit) plus
every family spec's own descriptor — swap for the real manifest digest once merged.

## 8. Seed diversity

`seed_distance(w1, w2)` reports `rms_A_distance` (RMS over the a-powered-space entries),
`gp_distance` (`abs(gp1-gp2)`), and `combined_standardized_distance = sqrt(rms_A_distance^2 +
gp_distance^2)` — both components are already comparable natural units, so no additional
standardization framework was introduced. Non-calibration candidates are compared against every
already-accepted seed (in increasing `attempt_id` order); a candidate closer than
`min_seed_distance` to its nearest accepted seed is rejected with `rejection_reason =
:too_close_to_existing_seed` and the ledger records `duplicate_distance`/`nearest_seed_id`.
Default `min_seed_distance=0.0` (never rejects on grounds of duplication unless the caller opts
in to a stronger threshold).

## 9. Reproducibility test results

`full_aod_diag/d4_exact/test_multistart_seed_generator.jl`, run at D20/W=8000 (`sobol_randomized`,
functional testing only — not the production W>=20000+ scale). Section A is pure-logic (no
ctx/KNITRO); Section B builds one real D20 context (once) and reuses it across every test that
needs a real ctx.

**Result: ALL 13 testsets PASS, 859/859 assertions, 0 failures, 0 errors** (re-run after the
2026-08-09 companion-LFD nu policy correction below; the Nu-lift testset now performs real
companion + restricted solves, correctly tolerating a matching failure at this small-W/
non-production-seed test context as well as a matching success — see that testset's own comment).

| Testset | Assertions | Time |
|---|---|---|
| A_scale: empirical RMS never exceeds A_scale | 403 | 0.6-0.8s |
| gp_scale: fraction in [0,gp_scale], candidate between calibration/target | 402 | 0.0-0.1s |
| Attempt-based RNG: deterministic, attempt-id-keyed | 5 | 0.2-0.3s |
| seed_distance: zero for identical, symmetric, standardized-combined | 5 | 0.4-0.5s |
| resolve_cm_probs: matches production nested-grid, not equal-spacing | 5 | 1.8-2.1s |
| FamilySeedSpec constructors + five-family preset shape | 3 | 0.1s |
| json_scalar / csv-safe encoding | 7 | 0.2s |
| gp directional bounds match ctx.bounds exactly | 6 | 0.0s |
| Gravity reconstruction: decode_w_econ round-trips exactly | 3 | 2.9-3.7s |
| Nu lift (companion-LFD-implied) deterministic, RNG-independent, correctly length-reduced | 6 | ~1m07s |
| qualify_economic_point / all-family intersection short-circuit | 3 | 10.4-21.0s |
| Attempt cap: exits at max_attempts, fewer than M returned | 6 | 5.5-6.9s |
| Reproducibility: identical inputs -> bit-identical digests; reordered-completion match | 5 | 11.4-16.8s |

`REPRODUCIBILITY = attempt_based_rng: pass, completion_order_independent: pass` — both confirmed
directly: identical `(rng_seed, family_specs, W)` inputs give bit-identical
`economic_digest`/`gp` sequences across two independent calls; a simulated shuffled-then-resorted
`evaluate_attempt` schedule (task section 20's reordering scenario) reproduces the exact same
per-attempt digests as the sequential run.

## 10. D20 release smoke (task section 24) + real multi-seed demonstration

Ran exactly as specified: `D=20`, `W=20_000`, `sobol_randomized` (production draw design,
`draw_seed=20260719`), `σHat=3.0`, `inner_lower_limit=-10.0`, `M=3` (including calibration),
`direction=:upper`, `delta_max=3.0`, `A_scale=0.03`, `gp_scale=0.5`, `max_attempts=6`,
`rng_seed=0x2026080800000001`, `production_five_family_seed_specs(ctx)`. Two independent runs
(`run1`, `run2`), same process-cold each time, using the final companion-LFD nu policy (section 5).

**Both runs: exit code 0. `n_attempted=6, n_accepted=1, stop_reason=:attempt_limit`.**

Calibration (`S0`) independently VERIFIED under all five families (real KNITRO solves, real D20
Brazil-Korea data, `inner_status=0` / `VerifiedSolved` every time):

| Family | Δ* | verified | inner_status |
|---|---|---|---|
| U_MEAN3 | 0.004543848599080799 | true | 0 (VerifiedSolved) |
| COMMON_FRECHET | 0.00503581224099289 | true | 0 (VerifiedSolved) |
| CM_MEAN3 | 0.009018508675413817 | true | 0 (VerifiedSolved) |
| ORIGIN_ZC_K3 | 0.05599430959227194 | true | 0 (VerifiedSolved) |
| CMZC_K3 | 0.06522160908742689 | true | 0 (VerifiedSolved) |

All five Δ* values are small and finite (consistent with a genuinely feasible point, per
CLAUDE.md's own bimodal-Δ* observation), essentially unchanged from the earlier fixed-theoretical-
nu numbers — expected, since AT calibration itself the companion LFD's implied moment and the
theoretical Fréchet population moment nearly coincide by construction; the real effect of the
nu-policy correction shows up away from calibration (below).

**`run1` vs `run2` (identical inputs, independent process-cold executions): bit-identical**, confirmed
via `diff` on both runs' full ledgers (all 7 attempts, 0-6) after stripping only wall-clock-timing
fields — `manifest_digest`, every `economic_digest`, every `gp` value, every `Delta_star` (to full
float64 precision), every accept/reject decision, and every `rejection_reason` matched exactly.

Output tree (`d20_smoke_output_run1/`) contains `attempts.csv`, `attempts.jsonl`, `manifest.jls`,
`seeds/S0/economic_seed.jls` + `seeds/S0/{U_MEAN3,COMMON_FRECHET,CM_MEAN3,ORIGIN_ZC_K3,CMZC_K3}/
full_outer_seed.jls` — the full contract from section 12 below, produced automatically by
`generate_multistart_seeds` itself (not a side script).

`source_sha` recorded in both runs: `c22d831798edf0ca74df1916fb258dac89d35e85` (production tip at
task start) — this generator's own file changes (this branch, not yet merged) are layered on top;
see section 13 for the branch's own commit history once merged.

### Real multi-seed demonstration (user-requested, 2026-08-09): does this generator actually find randomized seeds?

The section-24 smoke above only demonstrates calibration (`S0`) qualifying — `A_scale=0.03`/
`gp_scale=0.5` is deliberately too aggressive to find additional random seeds within a 6-attempt
smoke budget (real random draws at that scale hit genuine `nStatus=-300`, a confirmed KNITRO
infeasibility certificate — not a bug). A user review correctly flagged that this alone does not
demonstrate the generator finding NEW randomized points, so a separate real run was executed:
`M=5`, `include_calibration=false` (every accepted seed genuinely randomized, none of them
calibration), `direction=:upper`, `delta_max=3.0`, `A_scale=0.002`, `gp_scale=0.02` (found by a
direct empirical scale scan: 10/10 acceptance for the cheapest family alone at this scale, vs 0/10
at the original 0.03/0.5), `max_attempts=40`, `W=20_000`, real D20 Brazil-Korea data, the current
companion-LFD nu policy.

**Result: `n_attempted=6, n_accepted=5, stop_reason=:target_reached`** (5/6 attempts qualified —
up from 5/8 with the earlier fixed-theoretical-nu policy at the identical scale, a direct
confirmation that the companion-LFD nu correction genuinely raises acceptance, not just a
re-labeling):

| Seed | attempt_id | U_MEAN3 | COMMON_FRECHET | CM_MEAN3 | ORIGIN_ZC_K3 | CMZC_K3 |
|---|---|---|---|---|---|---|
| S0 | 1 | 0.0048 | 0.0068 | 0.1931 | 0.0593 | 0.7494 |
| S1 | 3 | 0.0046 | 0.0061 | 0.0109 | 0.0631 | 0.0721 |
| S2 | 4 | 0.0062 | 0.0261 | 0.0409 | 0.0662 | 0.0981 |
| S3 | 5 | 0.0047 | 0.0084 | 0.1403 | 0.0757 | 0.5525 |
| S4 | 6 | 0.0046 | 0.0055 | 0.1207 | 0.0708 | 0.4866 |

All 25 values real (`inner_status=0`, `VerifiedSolved`), all well under `delta_max=3`. The one
rejected attempt (`attempt_id=2`) passed `U_MEAN3`/`COMMON_FRECHET`/`CM_MEAN3` (Δ*=2.7717, close
to but under 3) before genuinely failing `CMZC_K3` — a real demonstration of the all-family
short-circuit qualification logic making a genuine reject decision on real KNITRO output, not a
rubber stamp.

`A_scale`/`gp_scale` are caller-supplied parameters, not generator constants — a real campaign
should scan/tune them for its own `W`/`delta_max` the same way this session did, rather than
assuming the section-11 example values are universally correct.

## 11. Example calls

```julia
# Five-family production comparison, W=100k
seeds = generate_multistart_seeds(ctx;
    M = 5, include_calibration = true, direction = :upper, delta_max = 3.0,
    family_specs = production_five_family_seed_specs(ctx),
    W = 100_000, rng_seed = 0x0000000000BADC0DE,
    A_scale = 0.05, gp_scale = 0.75, max_attempts = 200,
    output_dir = "/path/to/campaign/seeds")

# Cheaper screening pass before committing to W=100k (LABELED as such, never conflated)
screen = generate_multistart_seeds(ctx_w20k;
    M = 10, direction = :upper, delta_max = 3.0,
    family_specs = production_five_family_seed_specs(ctx_w20k),
    W = 20_000, rng_seed = 0x1, A_scale = 0.05, gp_scale = 1.0, max_attempts = 50,
    output_dir = "/path/to/screening_only_w20k")
# screen.request.W == 20_000 -- NOT valid as a W=100k production seed set without re-qualifying.
```

## 12. Output tree

```
output_dir/
  attempts.csv          # one row per attempt (incl. calibration), append-only ledger
  attempts.jsonl         # same, structured (per-family Delta*/verified/inner_status/etc.)
  manifest.jls           # manifest_digest, source_sha, request params, stop_reason
  seeds/
    S0/                  # calibration, if include_calibration=true
      economic_seed.jls
      U_MEAN3/full_outer_seed.jls
      CM_MEAN3/full_outer_seed.jls
      ...
    S1/
      economic_seed.jls
      <FAMILY_ID>/full_outer_seed.jls   # one per requested family
```

## 13. Integration status

`CAMPAIGN_LAUNCHED = false` — this task only implements/tests/documents the generator; no
production bounds campaign was launched.

Branch `feature/reproducible-multistart-generator-2026-08-08`, based on
`origin/production/fullA-exact @ c22d831`:

- `e36ae39` — core generator (`multistart_seed_generator.jl`)
- `eccc7b6` — test suite (`test_multistart_seed_generator.jl`, 12/12 passing)
- `8d9eff3` — docs (D20 smoke results, superseded by the results in this file's current form)
- `fc2dddb` — companion-LFD-implied nu policy (post-review correction, section 5), 13/13 tests passing
- (this commit) — docs update: final nu policy, real 5-seed demonstration, corrected verdict block

Not yet rebased/merged/tagged/pushed to `origin/production/fullA-exact` — awaiting explicit
confirmation before any of those steps (repo convention: never merge/push to the real remote
without direct authorization in the current conversation).

---

## Final verdict

```
GENERATOR = production_ready

PUBLIC_API =
    generate_multistart_seeds(ctx; M, direction, delta_max, family_specs, W, rng_seed, A_scale,
        gp_scale=1.0, max_attempts=200, include_calibration=true, min_seed_distance=0.0,
        max_concurrency=1, radius_mode=:uniform_radius, scale_schedule=nothing, output_dir,
        source_sha="") -> MultiStartSeedSet

M_SEMANTICS = includes_calibration_when_requested

A_PERTURBATION =
    free_economic_coordinates_only (powered a-space, cm_w0_from_calibration's own KNITRO
    outer-search coordinate)
    rms_scale: A_scale is RMS in a-powered-space units; RMS in natural log-A-level units is
    A_scale*theta, theta = cm_fixed_theta(ctx) = 1/ctx.fixed_vals[1]

GP_DEFINITION = gp = theta0_up[3+D] = gamma'_focal/gamma_baseIndex (gamma_baseIndex=1 gauge,
    so numerically gp = gamma'_focal)

UPPER_GP_ENDPOINT = ctx.bounds.gamma_prime_lo = lambda_dd^(1/sigma)  (theoretical_gammaprime_bounds,
    moments_gammanorm.jl -- NOT the naive lambda_dd^(sigma/(sigma-1)) guess)

LOWER_GP_ENDPOINT = ctx.bounds.gamma_prime_hi = 1.0  (kappa=0, zero GT)

NU_POLICY = companion_lfd_implied (origin-ZC: plain ctx.obj via solve_base_state; CM+ZC: plain
    CM-only family, pooled across origins -- see section 5; NOT a fixed theoretical constant)
NU_RANDOMIZED = false

FAMILY_PRESET =
    U_MEAN3: origin_zc, K_mean=3, K_pair=0
    CM_MEAN3: cm_zc, K_mean=3, K_pair=0, L=50
    COMMON_FRECHET: common_frechet (single-family CDF-only -- two-family not production-reachable
        at this SHA), L=50
    ORIGIN_ZC_K3: origin_zc, K_mean=3, K_pair=3
    CMZC_K3: cm_zc, K_mean=3, K_pair=3, L=50

ALL_FAMILY_ACCEPTANCE = required

REPRODUCIBILITY =
    attempt_based_rng: pass
    completion_order_independent: pass

ATTEMPT_LIMIT = enforced

ATTEMPT_LEDGER = complete: pass

RAW_A_PERTURBED = false
GRAVITY_REPAIRED_POSTHOC = false
NU_RANDOMIZED = false
DENSE_REFERENCE_PRODUCTION = false

CAMPAIGN_LAUNCHED = false
```

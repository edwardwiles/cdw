# Reusable production multistart seed generator for the FULL-A Ricardian robustness problem
# (task: reproducible-multistart-generator-2026-08-08).
#
# ============================================================================================
# READ THIS BEFORE WRITING A NEW CAMPAIGN-SPECIFIC SEED SCRIPT. If you need multiple randomized-
# but-verified starting points for an outer campaign (any combination of origin-ZC/CM+ZC/common-
# Fréchet), this file already does it -- do not hand-invent candidate A_od/gp points, guess nu
# values, or write a one-off script; that exact pattern is what this file replaces.
#
# QUICK START:
#   ctx = d20_real_setup_design(W = 100_000, δ = 1.0, find_smallest = true,
#       draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
#       exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
#       σHat = 3.0, inner_lower_limit = -10.0)   # every scientific param explicit, nothing defaulted
#   seeds = generate_multistart_seeds(ctx;
#       M = 5, include_calibration = true, direction = :upper, delta_max = 3.0,
#       family_specs = production_five_family_seed_specs(ctx),   # or your own Vector{FamilySeedSpec}
#       W = 100_000, rng_seed = 0x0000000000000001,
#       A_scale = <SCAN THIS, see below>, gp_scale = <SCAN THIS>,
#       max_attempts = 200, output_dir = "/path/to/campaign/seeds")
#   # seeds.accepted :: Vector{AcceptedSeed}; output_dir/attempts.csv|.jsonl + seeds/S<k>/... are
#   # written automatically -- do not re-implement ledger/output serialization elsewhere.
#
# A_scale/gp_scale are NOT tuned defaults -- an aggressive scale can make every random draw
# genuinely infeasible (confirmed live: 0/10 vs 10/10 acceptance between two scale choices at the
# same W). Scan a small (A_scale,gp_scale) grid at your own W/delta_max first (evaluate_attempt is
# exposed exactly for this -- see the empirical-scan pattern in
# docs/audits/reproducible-multistart-generator-2026-08-08/MASTER.md) before trusting any specific
# value, including the one in this comment.
#
# Full API reference, the exact live gp/nu-policy formulas, reproducibility test results, and a
# real D20/W=20000 release smoke (including a genuine 5-random-seed acceptance run) are in
# docs/audits/reproducible-multistart-generator-2026-08-08/MASTER.md -- read that before extending
# or debugging this file, not just this header.
# ============================================================================================
#
# Replaces the one-off campaign-specific pattern (manually inventing candidate A/gp points,
# guessing nu values, losing track of what was tried between campaigns) with a single reusable
# utility: randomly perturb the CALIBRATION ECONOMIC POINT (free powered-a-space A coordinates +
# gp) and, for a caller-specified list of production family specs, deterministically lift each
# candidate into every family's nuisance (nu) layout via that family's existing LFD machinery,
# then verify Delta*<=delta_max for EVERY requested family before accepting the point as a shared
# multistart seed.
#
# Dependencies (NOT included here -- matches this codebase's existing convention, e.g.
# cm_aspace_coordinate.jl's own guard, of assuming the wider d4_exact family machinery is already
# loaded into Main by whatever driver/test/smoke script includes this file last):
#   gravity_elimination.jl        (PivotGravityElim, build_pivot_elimination, pivot_reduce, pivot_expand)
#   cm_aspace_coordinate.jl       (cm_z_from_a, cm_a_from_z, cm_fixed_theta, precompute_cm_aspace_xy,
#                                   cm_w0_from_calibration)
#   cm_originzc_target_layout.jl  (OriginByPowerLayout, SharedByPowerLayout, ActiveMeanLayout, n_eta,
#                                   target_index, scatter_nu_eff)
#   cm_originzc_production.jl     (build_originzc_production_context, archOZ_verified_state)
#   cm_meanzc_production.jl       (build_cm_meanzc_production_context, archC_meanzc_verified_state)
#   cm_production_bundle.jl       (build_cm_production_context, archC_verified_state, CMExpectedSolveFailure --
#                                   the plain CM-only companion used to derive CM+ZC's nu, section 8)
#   cm_meanzc_moments.jl          (build_raw_mean_pair_matrix_levels, frechet_power_feature -- the
#                                   SAME raw z^k feature construction production's own restriction
#                                   columns use, reused here for the companion-LFD nu policy)
#   three_way_derivatives.jl      (solve_base_state, BaseDualState -- the plain unrestricted
#                                   ctx.obj companion solve for origin-ZC's own nu policy)
#   cm_frechet_level.jl           (build_cm_frechet_production_context)
#   cm_frechet_cplus.jl           (cm_frechet_production_value_verified_screened)
#   cm_screen_bridge.jl           (cm_originzc_production_value_verified_screened,
#                                   cm_meanzc_production_value_verified_screened,
#                                   cm_production_value_verified_screened)
#   cm_originzc_checkpoint.jl     (originzc_profiled_nu_value)
#   cm_checkpoint.jl              (meanzc_profiled_nu_value)
#   oracle.jl                     (classify_inner_result, is_verified_success, sha256_of_matrix)
#   country_resolve.jl            (default_gravity_exclude_cells_brazil_korea)
#   nested_quantile_grids.jl      (nested_grid_sequence)
#   compressed_factual_buffer_reuse.jl (attach_compressed_factual_workspace -- REQUIRED once per
#                                   ctx, before any real family evaluation; see generate_multistart_seeds)
#   fast_range_screen.jl          (build_ranged_screen_context, evaluate_fullA_screened_ranged --
#                                   paper_upper_v1 extension, :unrestricted family kind only; pulls
#                                   in its own chain -- compressed_live.jl, compressed_moments.jl,
#                                   structured_moment_build.jl, compressed_cc_inner.jl, dual_bank.jl,
#                                   etc. -- see c10_d20_production_driver.jl's own include list for
#                                   the full canonical order, or test_multistart_seed_generator.jl's
#                                   include list for a validated concrete example)
#   `CS` (the CounterfactualSensitivity module) already bound in Main -- same convention as every
#   other cm_*_production.jl file (see cc_algo/include_cc_algo.jl).
#
# ctx must be built via d20_real_setup_design(...) (not the bare d20_real_setup) so that
# ctx.draw_design/ctx.draw_seed are populated -- these feed the manifest digest and are required,
# not defaulted, per this repo's "never let a function default a scientific parameter" rule
# (CLAUDE.md). sigma/W/K_mean/K_pair/gravity exclusions/destination_sample/draw_design/draw_seed/
# inner_lower_limit are therefore never re-defaulted inside this file -- they are read off the
# caller's own already-fully-specified `ctx`, or (for K_mean/K_pair/L/family kind) taken as
# required fields on the caller-supplied FamilySeedSpec.

for _dep in (:build_pivot_elimination, :cm_w0_from_calibration, :OriginByPowerLayout, :scatter_nu_eff,
             :build_originzc_production_context, :build_cm_meanzc_production_context,
             :build_cm_frechet_production_context, :cm_originzc_production_value_verified_screened,
             :cm_meanzc_production_value_verified_screened, :cm_frechet_production_value_verified_screened,
             :build_cm_production_context, :cm_production_value_verified_screened, :CMExpectedSolveFailure,
             :build_raw_mean_pair_matrix_levels, :solve_base_state,
             :originzc_profiled_nu_value, :meanzc_profiled_nu_value, :classify_inner_result,
             :is_verified_success, :sha256_of_matrix, :default_gravity_exclude_cells_brazil_korea,
             :nested_grid_sequence, :attach_compressed_factual_workspace, :CS,
             # paper_upper_v1 extension (2026-08-08): :unrestricted / :cm_only family kinds, added
             # so this SAME reusable generator can qualify seeds against the plain (no-ZC)
             # Unrestricted and Common-Marginals families too, not just the three ZC-flavor kinds
             # the original release covered. Reuses `evaluate_fullA_screened_ranged`'s own
             # `build_ranged_screen_context(ctx)` companion -- the exact per-point value-only
             # verified evaluator `run_polish_checkpointed_unified` itself calls (fast_range_screen.jl).
             :build_ranged_screen_context, :evaluate_fullA_screened_ranged)
    isdefined(Main, _dep) ||
        error("multistart_seed_generator.jl requires `$(_dep)` to already be defined -- include the " *
              "full d4_exact family machinery (see this file's header comment) before this file.")
end

using Random, LinearAlgebra, SHA, Serialization, Printf

# ============================================================================
# 1. Family spec: the caller-supplied, non-hard-wired family list.
# ============================================================================

"""
    FamilySeedSpec

One requested production family for seed qualification. `kind` selects the machinery:
`:origin_zc` (origin-specific ZC, `run_originzc_upper_checkpointed`'s value-only path),
`:cm_zc` (CM(L-grid, common-flexible) + K-moment ZC combined, `run_cm_upper_checkpointed`'s
`cm_extension=:cm_plus_moments` value-only path), or `:common_frechet` (single-family CDF-only
common Fréchet, `marginal_restriction=:common_frechet`, `cm_extension=:cm_only` -- confirmed
2026-08-08 that the two-family/eq.35+eq.36 variant is not reachable through the current production
driver at all, see docs/audits/reproducible-multistart-generator-2026-08-08/MASTER.md).

`L`/`contrasts`/`probs`/`include_truncated_moment`/`meanzc_basis` are consulted only by `:cm_zc`
and `:common_frechet` kinds; ignored (but still required, for a uniform struct) by `:origin_zc`.
"""
struct FamilySeedSpec
    id::Symbol
    kind::Symbol                 # :origin_zc | :cm_zc | :common_frechet
    K_mean::Int
    K_pair::Int
    L::Int
    contrasts::Symbol
    probs::Union{Nothing,Vector{Float64}}
    include_truncated_moment::Bool
    meanzc_basis::Symbol
end

"""
    resolve_cm_probs(L) -> Vector{Float64}

Resolves the L-grid cutpoints the SAME way the real production campaign runner does
(`campaign_cm_family_runner.jl`: `nested_grid_sequence([10,20,50])[CM_L]`), not the weaker
`cm_equal_grid_probs(L)` equal-spacing default some prior scratch scripts silently fell back to.
Falls back to the plain equal-spacing grid only for `L` values outside the validated nested
family `{10,20,50}` (documented, not silent).
"""
function resolve_cm_probs(L::Int)
    if L in (10, 20, 50)
        return nested_grid_sequence([10, 20, 50])[L]
    end
    return collect(range(1 / L, (L - 1) / L, length = L))
end

function origin_zc_family_spec(id::Symbol; K_mean::Int, K_pair::Int)
    FamilySeedSpec(id, :origin_zc, K_mean, K_pair, 0, :orthonormal, nothing, false, :direct)
end

function cm_zc_family_spec(id::Symbol; K_mean::Int, K_pair::Int, L::Int,
                            contrasts::Symbol = :orthonormal,
                            probs::Union{Nothing,Vector{Float64}} = resolve_cm_probs(L))
    FamilySeedSpec(id, :cm_zc, K_mean, K_pair, L, contrasts, probs, false, :direct)
end

function common_frechet_family_spec(id::Symbol; L::Int, contrasts::Symbol = :orthonormal,
                                     probs::Union{Nothing,Vector{Float64}} = resolve_cm_probs(L))
    FamilySeedSpec(id, :common_frechet, 0, 0, L, contrasts, probs, false, :direct)
end

"""
    unrestricted_family_spec(id) -> FamilySeedSpec

paper_upper_v1 extension (2026-08-08): the plain, wholly unrestricted family -- no CM grid, no ZC
moments. `K_mean`/`K_pair`/`L`/`contrasts`/`probs` are unused (uniform struct only); qualification
calls `evaluate_fullA_screened_ranged` directly (see `build_family`/`evaluate_family` below), the
same value-only verified evaluator `run_polish_checkpointed_unified` itself uses at a candidate
point -- no nu lift, no companion solve (there is nothing to lift; this IS the companion the other
families' own nu policies solve internally).
"""
function unrestricted_family_spec(id::Symbol)
    FamilySeedSpec(id, :unrestricted, 0, 0, 0, :orthonormal, nothing, false, :direct)
end

"""
    cm_only_family_spec(id; L, contrasts=:orthonormal, probs=resolve_cm_probs(L)) -> FamilySeedSpec

paper_upper_v1 extension (2026-08-08): plain flexible Common-Marginals, `cm_extension=:cm_only`
(no ZC moments) -- `run_cm_upper_checkpointed`'s own default extension. `K_mean`/`K_pair` are
unused (uniform struct only, always 0). Qualification reuses the EXACT SAME plain-CM builder/
evaluator pair (`build_cm_production_context` + `cm_production_value_verified_screened`) the
`:cm_zc` kind's own `companion_implied_nu_cmzc` already calls as its companion solve -- this spec
just evaluates that companion directly as ITS OWN family, rather than as an internal nu-lift step.
"""
function cm_only_family_spec(id::Symbol; L::Int, contrasts::Symbol = :orthonormal,
                              probs::Union{Nothing,Vector{Float64}} = resolve_cm_probs(L))
    FamilySeedSpec(id, :cm_only, 0, 0, L, contrasts, probs, false, :direct)
end

# ============================================================================
# 2. Five-family production preset (task section 10). Not hard-wired into the core generator.
# ============================================================================

"""
    production_five_family_seed_specs(ctx) -> Vector{FamilySeedSpec}

The CURRENT five-family comparison, in cheap-to-expensive qualification order (task section 12).
Order was set from real measured wall-clock timings recorded in the D20 release smoke ledger
(see docs/audits/reproducible-multistart-generator-2026-08-08/MASTER.md) -- re-order this preset,
not the core generator, if relative family cost changes in a future production revision.

Per 2026-08-08 research audit: true two-family (eq.35+eq.36) common-Fréchet is NOT reachable
through the current production driver (`build_cm_frechet_production_context`'s dense-Hessian
requirement when `include_truncated_moment=true` conflicts with the operator-bundle-only
production ban) -- COMMON_FRECHET below is therefore single-family (CDF-basis only,
`include_truncated_moment=false`), matching the one variant that actually is production-reachable.
"""
function production_five_family_seed_specs(ctx)
    return [
        origin_zc_family_spec(:U_MEAN3; K_mean = 3, K_pair = 0),
        common_frechet_family_spec(:COMMON_FRECHET; L = 50),
        cm_zc_family_spec(:CM_MEAN3; K_mean = 3, K_pair = 0, L = 50),
        origin_zc_family_spec(:ORIGIN_ZC_K3; K_mean = 3, K_pair = 3),
        cm_zc_family_spec(:CMZC_K3; K_mean = 3, K_pair = 3, L = 50),
    ]
end

"""
    paper_five_family_seed_specs(ctx) -> Vector{FamilySeedSpec}

The `paper_upper_v1` protocol's actual five scientific families (protocols/paper_upper_v1.toml),
in cheap-to-expensive qualification order: Unrestricted (one companion-only solve) -> Common
Marginals (`cm_only`, one plain-CM solve) -> Common-Fréchet (single-family CDF-only, the one
variant production actually reaches -- see `production_five_family_seed_specs`'s own docstring)
-> Origin-ZC K_mean=K_pair=3 (companion + restricted, two solves) -> CM+ZC K_mean=K_pair=3
(companion + restricted, two solves). Distinct from `production_five_family_seed_specs` (which
qualifies three ZC-flavor K_mean=3/K_pair=0-or-3 variants for a different, non-paper comparison) --
this is the exact five-family set the paper protocol freezes: no redundant mean moments added to
Unrestricted or Common Marginals, family definitions unchanged from their standing production
meaning.
"""
function paper_five_family_seed_specs(ctx)
    return [
        unrestricted_family_spec(:UNRESTRICTED),
        cm_only_family_spec(:COMMON_MARGINALS; L = 50),
        common_frechet_family_spec(:COMMON_FRECHET; L = 50),
        origin_zc_family_spec(:ORIGIN_ZC; K_mean = 3, K_pair = 3),
        cm_zc_family_spec(:CM_PLUS_ZC; K_mean = 3, K_pair = 3, L = 50),
    ]
end

# ============================================================================
# 3. gp definition, theoretical endpoints (task section 5-6).
# ============================================================================
#
# Confirmed live 2026-08-08 (moments_gammanorm.jl): gp = theta0_up[3+D] = gamma'_focal / gamma_focal
# under this repo's gamma_baseIndex==1 gauge. `ctx.bounds` (built by
# `theoretical_gammaprime_bounds(ctx.γ, ctx.σ)`) already carries the EXACT formulas as coded:
#   gamma_prime_hi = 1.0                          (kappa=0, zero-GT endpoint)
#   gamma_prime_lo = lambda_dd^(1/sigma)           (kappa=kappa_max, theoretical upper-GT endpoint)
#   kappa_max      = 1 - lambda_dd^(1/(sigma-1))
# NOTE this is NOT the naively-guessed lambda_dd^(sigma/(sigma-1)) or lambda_dd -- use ctx.bounds
# directly rather than re-derive, so this file can never silently drift from whatever
# `theoretical_gammaprime_bounds` actually computes.

"""
    gp_calibration_and_target(ctx, direction::Symbol) -> (gp_cal, gp_target)

`direction=:upper` -> box `[gamma_prime_lo, gp_cal]` (larger kappa/GT), target = `ctx.bounds.γp_lo`.
`direction=:lower` -> box `[gp_cal, gamma_prime_hi]` (smaller kappa/GT, toward zero-GT), target =
`ctx.bounds.γp_hi`. Matches `direction_bounds.jl`'s own `find_smallest=true<->upper` convention.
"""
function gp_calibration_and_target(ctx, direction::Symbol)
    direction in (:upper, :lower) || error("gp_calibration_and_target: direction must be :upper or :lower, got :$(direction)")
    gp_cal = ctx.θ0_up[3 + ctx.D]
    gp_target = direction == :upper ? ctx.bounds.γp_lo : ctx.bounds.γp_hi
    return gp_cal, gp_target
end

# ============================================================================
# 4. Digest helpers (task section 9/18/19). sha256, not Base.hash (process/version-unstable).
# ============================================================================

sha256_of_vector(v::AbstractVector{Float64}) = sha256_of_matrix(reshape(Vector{Float64}(v), :, 1))

function sha256_of_string(s::AbstractString)
    return bytes2hex(SHA.sha256(codeunits(s)))
end

function family_spec_descriptor(spec::FamilySeedSpec)
    probs_desc = spec.probs === nothing ? "nothing" : join(round.(spec.probs, digits = 10), ",")
    return "id=$(spec.id)|kind=$(spec.kind)|K_mean=$(spec.K_mean)|K_pair=$(spec.K_pair)|L=$(spec.L)|" *
           "contrasts=$(spec.contrasts)|include_truncated_moment=$(spec.include_truncated_moment)|" *
           "meanzc_basis=$(spec.meanzc_basis)|probs=$(probs_desc)"
end

"""
    compute_manifest_digest(ctx, family_specs, W) -> String

Stand-in for a true `ScientificManifest.jl` digest -- confirmed 2026-08-08 that
`scientific_manifest/ScientificManifest.jl` is NOT merged into `production/fullA-exact` at this
SHA (only vendored on unrelated feature branches). Built directly from every scientific parameter
this generator actually reads off `ctx` plus the caller's family specs and `W`, so two calls with
an identical `ctx`+`family_specs`+`W` always produce an identical digest, and any change to a
scientific parameter changes the digest. Swap this for the real manifest digest once
`ScientificManifest.jl` is merged into this branch's ancestry.
"""
function compute_manifest_digest(ctx, family_specs::Vector{FamilySeedSpec}, W::Int)
    grav = join(sort(collect(ctx.gravity_exclude_cells)), ";")
    desc = "sigma=$(ctx.σ)|W=$(W)|bi=$(ctx.bi)|muHat=$(ctx.μHat)|" *
           "destination_sample=$(ctx.destination_sample)|exclude_diagonal_gravity=$(ctx.exclude_diagonal_gravity)|" *
           "gravity_exclude_cells=$(grav)|draw_design=$(ctx.draw_design)|draw_seed=$(ctx.draw_seed)|" *
           "inner_lower_limit=$(ctx.obj.lower_limit)|" *
           "families=[" * join((family_spec_descriptor(s) for s in family_specs), "||") * "]"
    return sha256_of_string(desc)
end

# ============================================================================
# 5. Attempt-based deterministic RNG (task section 7). Fresh RNG per attempt, seeded from a
#    stable hash of (master seed, attempt_id, manifest digest) -- never one mutable global RNG.
# ============================================================================

function attempt_rng_seed(rng_seed::UInt64, attempt_id::Int, manifest_digest::String)
    buf = IOBuffer()
    write(buf, rng_seed)
    write(buf, Int64(attempt_id))
    write(buf, codeunits(manifest_digest))
    h = SHA.sha256(take!(buf))
    return only(reinterpret(UInt64, h[1:8]))
end

attempt_rng(rng_seed::UInt64, attempt_id::Int, manifest_digest::String) =
    Random.Xoshiro(attempt_rng_seed(rng_seed, attempt_id, manifest_digest))

# ============================================================================
# 6. Economic (free-A) perturbation with an interpretable RMS scale (task section 4).
# ============================================================================
#
# The free-A coordinate perturbed here is `a_powered` (KNITRO's actual outer-search coordinate,
# `A_coordinate_mode=:powered_aspace`), NOT raw log-A. Per cm_aspace_coordinate.jl:
#   a_nonpivot = -(z_nonpivot + logY_nonpivot)/theta - logX_nonpivot,   theta = cm_fixed_theta(ctx)
# i.e. a is an EXACT affine function of z_nonpivot (pivot-reduced log-A) with constant slope
# -1/theta. So an RMS perturbation of `A_scale` in a-space corresponds to an RMS perturbation of
# `A_scale * theta` in log-A (natural log-level) units -- NOT `A_scale` log-A units directly.
# This file always reports both the a-space RMS actually drawn (== the r_A,j radius by
# construction) and theta, so a caller can convert if they need genuine log-A units.

"""
    draw_A_perturbation(rng, n_A, A_scale; radius_mode=:uniform_radius) -> (delta_a, radius)

`u ~ N(0,I_n_A)` normalized to unit RMS, `radius = A_scale*rand(rng)` (`:uniform_radius`, default)
or `radius = A_scale` (`:fixed_radius`), `delta_a = radius .* u`. `radius` IS the resulting RMS of
`delta_a` by construction (`sqrt(mean(delta_a.^2)) == radius` exactly).
"""
function draw_A_perturbation(rng::AbstractRNG, n_A::Int, A_scale::Float64; radius_mode::Symbol = :uniform_radius)
    A_scale > 0 || error("draw_A_perturbation: A_scale must be > 0, got $(A_scale)")
    radius_mode in (:uniform_radius, :fixed_radius) ||
        error("draw_A_perturbation: radius_mode must be :uniform_radius or :fixed_radius, got :$(radius_mode)")
    u = randn(rng, n_A)
    u ./= sqrt(sum(abs2, u) / n_A)
    radius = radius_mode == :uniform_radius ? A_scale * rand(rng) : A_scale
    return radius .* u, radius
end

"""
    draw_gp_perturbation(rng, gp_cal, gp_target, gp_scale) -> (gp_candidate, fraction)

`fraction ~ U(0, gp_scale)`, `gp_candidate = gp_cal + fraction*(gp_target - gp_cal)`. Monotone
toward `gp_target` only (`fraction>=0`), so an upper-direction candidate can never be perturbed
toward the lower-direction endpoint or vice versa (task section 6).
"""
function draw_gp_perturbation(rng::AbstractRNG, gp_cal::Float64, gp_target::Float64, gp_scale::Float64)
    0.0 < gp_scale <= 1.0 || error("draw_gp_perturbation: gp_scale must be in (0,1], got $(gp_scale)")
    fraction = gp_scale * rand(rng)
    return gp_cal + fraction * (gp_target - gp_cal), fraction
end

# ============================================================================
# 7. Coordinate decode (task section 3): free-A perturbation -> full active A / gravity pivot /
#    economic free vector, using ONLY the existing production decoder. Never touches raw A_od,
#    never manually patches the pivot.
# ============================================================================

struct AspaceGeometry
    pe::Any                # PivotGravityElim
    theta::Float64
    xy::Any                # CMAPivotXY
    n_A::Int                # length of the free a-space (== D*D_dest-1)
end

function build_aspace_geometry(ctx)
    pe = build_pivot_elimination(ctx)
    theta = cm_fixed_theta(ctx)
    xy = precompute_cm_aspace_xy(ctx)
    w0 = cm_w0_from_calibration(ctx, pe, :powered_aspace)
    return AspaceGeometry(pe, theta, xy, length(w0) - 1)
end

"w_econ = [gp; a_powered] -> x_free = [gp; vec(A_full levels)], via the EXISTING production decoder (cm_z_from_a + pivot_expand + exp)."
function decode_w_econ(geo::AspaceGeometry, w_econ::AbstractVector{Float64})
    gp = w_econ[1]
    z_nonpivot = cm_z_from_a(w_econ[2:end], geo.theta, geo.xy, geo.pe)
    logA_full = pivot_expand(z_nonpivot, geo.pe)
    return vcat(gp, vec(exp.(logA_full)))
end

"Cheap domain/gravity-reconstruction precheck (task section 21) -- no heuristics, only exact finiteness/positivity, which is all that can fail in an otherwise-deterministic algebraic reconstruction."
function precheck_candidate(x_free::AbstractVector{Float64})
    gp = x_free[1]
    A = @view x_free[2:end]
    return isfinite(gp) && 0.0 < gp <= 1.0 && all(isfinite, A) && all(>(0.0), A)
end

# ============================================================================
# 8. Deterministic nu lift (task section 9): NEVER randomized, but genuinely a function of the
#    CURRENT candidate economic point -- not a fixed theoretical constant. For each of the
#    non-focal nu coordinates, this solves the SAME family's own unrestricted/CM-only companion
#    member (a solve the caller needs no new machinery for -- origin-ZC's own K_mean=0/K_pair=0
#    case, or CM+ZC's own CM-only case) at the SAME (A_od, gp) point, and reads the resulting
#    LFD's own implied k-th raw moment of the Frechet productivity feature z_o=U_o^{-mu} as nu.
#    Confirmed 2026-08-09 (user directive, replacing an earlier fixed-theoretical-constant policy):
#    using a FIXED nu (independent of the candidate point) makes every mean/pair restriction
#    artificially far more binding than the eventual campaign's own outer nu search would ever
#    leave it -- nu is itself part of the outer search vector `w0` (`w0 = vcat(g, A_native0,
#    eta_nu)`, cm_originzc_checkpoint.jl:752, with its own `nu_bounds`), so a THEORETICAL constant
#    is really just an arbitrarily cold starting value, not a scientific target. The companion-LFD
#    value is a much warmer, still-fully-deterministic starting value -- NOT full LFD-optimal nu
#    for the RESTRICTED problem itself (that would require solving the restricted problem, which
#    is what nu is FOR), so some genuine gap between it and the restricted problem's own eventual
#    optimum is expected, not a bug (the mean/pair restrictions really do add binding structure
#    beyond the unrestricted/CM-only case, by design -- see section 9 below).
#
#    The ONE separate, unrelated mechanism layered on top is the focal k=(sigma-1) row-omission
#    (Variant D): whenever K_mean>=kstar>=1, that ONE coordinate is not independently identified
#    (a collinearity issue, not a modeling choice) and must instead be derived analytically via
#    the existing production KKT/envelope formula, then scattered back into the dense vector via
#    scatter_nu_eff -- exactly mirroring the live originzc_profiled_level/meanzc_profiled_level
#    production kwarg. This applies identically regardless of where the OTHER nu entries came from.
# ============================================================================

"kstar = sigma-1, the CURRENT focal profiling level. Requires sigma-1 to be a positive integer (as production's own ActiveMeanLayout(...,kstar,...) requires)."
function focal_kstar(ctx)
    raw = ctx.σ - 1
    isinteger(raw) || error("focal_kstar: sigma-1 must be an integer for the focal row-omission rule, got sigma=$(ctx.σ)")
    return Int(raw)
end

"Whether K_mean is large enough for the focal k=sigma-1 profiling rule to apply to this family."
profiled_level_for(ctx, K_mean::Int) = begin
    k = focal_kstar(ctx)
    (1 <= k <= K_mean) ? k : nothing
end

struct FamilyBuild
    spec::FamilySeedSpec
    pcx::Any
    layout::Any
    aml::Any                       # ActiveMeanLayout or nothing
end

"""
    build_family(ctx, pe, spec::FamilySeedSpec) -> FamilyBuild

Builds a FRESH production context per call (never reused across candidates -- confirmed
correctness-critical by the 2026-08-07 K3 prior-art script: a stale internal screen/Hessian
buffer sized for one candidate silently corrupted a structurally different candidate's solve).
Does NOT compute nu here -- nu is a genuine function of the candidate economic point (section 8
above), so it is computed in `evaluate_family`, which receives `x_free`.
"""
function build_family(ctx, spec::FamilySeedSpec)
    if spec.kind == :origin_zc
        layout = OriginByPowerLayout(ctx.D, spec.K_mean, spec.K_pair)
        kstar = profiled_level_for(ctx, spec.K_mean)
        aml = kstar === nothing ? nothing : ActiveMeanLayout(layout, ctx.bi, kstar, ctx.D)
        pcx = build_originzc_production_context(ctx, CS, layout; aml = aml)
        return FamilyBuild(spec, pcx, layout, aml)
    elseif spec.kind == :cm_zc
        layout = SharedByPowerLayout(spec.K_mean, spec.K_pair)
        kstar = profiled_level_for(ctx, spec.K_mean)
        aml = kstar === nothing ? nothing : ActiveMeanLayout(layout, ctx.bi, kstar, ctx.D)
        pcx = build_cm_meanzc_production_context(ctx, CS; L = spec.L, K_mean = spec.K_mean, K_pair = spec.K_pair,
            include_truncated_moment = spec.include_truncated_moment, contrasts = spec.contrasts,
            meanzc_basis = spec.meanzc_basis, probs = spec.probs, moment_representation = :operator, aml = aml)
        return FamilyBuild(spec, pcx, layout, aml)
    elseif spec.kind == :common_frechet
        # cm_hessian_backend=:structured required: build_cm_frechet_production_context's own
        # default inner_fg_backend (CM_FRECHET_INNER_FG_BACKEND_DEFAULT[] = :cm_frechet_lookup)
        # hard-errors otherwise ("needs a real CMBinHessCtx") -- confirmed live 2026-08-08 at real
        # D20/W=20000. Matches the real campaign runner's own cm_hessian_backend=:structured
        # convention for every CM-family builder call.
        pcx = build_cm_frechet_production_context(ctx, CS; L = spec.L, include_truncated_moment = spec.include_truncated_moment,
            contrasts = spec.contrasts, probs = spec.probs, cm_hessian_backend = :structured, moment_representation = :operator)
        return FamilyBuild(spec, pcx, nothing, nothing)
    elseif spec.kind == :unrestricted
        # paper_upper_v1 extension: `pcx` slot holds the RangedScreenContext (fast_range_screen.jl),
        # built fresh per candidate for the same stale-buffer-safety reason build_family never
        # reuses a context across candidates for the ZC/CM kinds (see this function's own docstring).
        rsc = build_ranged_screen_context(ctx)
        return FamilyBuild(spec, rsc, nothing, nothing)
    elseif spec.kind == :cm_only
        # Exactly companion_implied_nu_cmzc's own companion-builder call (section 8 above), just
        # evaluated here as ITS OWN family rather than as an internal nu-lift step.
        pcx0 = build_cm_production_context(ctx, CS; L = spec.L, include_truncated_moment = spec.include_truncated_moment,
            contrasts = spec.contrasts, probs = spec.probs, moment_representation = :operator)
        return FamilyBuild(spec, pcx0, nothing, nothing)
    else
        error("build_family: unknown family kind :$(spec.kind) for spec $(spec.id)")
    end
end

"""
    companion_implied_nu_originzc(ctx, x_free, layout_target; eval_id=0) -> Vector{Float64}

Solves the truly unrestricted companion at this candidate's economic point -- the PLAIN, un-
augmented `ctx.obj` (base economic+gravity moments only, zero ZC structure), via `solve_base_state`
-- and reads the resulting LFD's own implied k-th raw moment of the Frechet productivity feature
per origin: `nu_{o,k} = sum_s m_s*z_o(s)^k / sum_s m_s`, `z_o(s)^k` from
`build_raw_mean_pair_matrix_levels`/`frechet_power_feature` -- the EXACT same feature array
origin-ZC's own restriction columns are built from (never a hand-derived power transform; this
codebase has a recurring U^k-vs-z^k bug history). `m` is `base.m_star` (`obj.arg1` at the
converged companion solve = `dPsi(q*)`, the un-normalized LFD weight over the W draws) -- present
identically on every family's `BaseDualState` (confirmed at `three_way_derivatives.jl:33`/
`cm_production_bundle.jl:432`).

NOTE: `OriginByPowerLayout` itself requires `K_mean>=1` (confirmed live 2026-08-09 -- there is no
"K_mean=0 origin-ZC" object to construct), so the companion is NOT origin-ZC's own machinery at
K_mean=0 -- it is the strictly simpler base `ctx.obj`, which is exactly what "zero ZC restriction"
means anyway (origin-ZC's restriction columns are pure ADDITIONS on top of `ctx.obj`; removing all
of them is identical to evaluating `ctx.obj` directly, not a degenerate case of the ZC machinery).
"""
function companion_implied_nu_originzc(ctx, x_free::AbstractVector{Float64}, layout_target::OriginByPowerLayout; eval_id::Int = 0)
    K_mean = layout_target.K_mean
    Zraw_all, _ = build_raw_mean_pair_matrix_levels(ctx.U, K_mean, 0; μ = ctx.μHat)
    base = solve_base_state(collect(x_free), ctx)
    m = base.m_star
    s = sum(m)
    nu = Vector{Float64}(undef, n_eta(layout_target))
    for k in 1:K_mean, o in 1:ctx.D
        nu[target_index(layout_target, o, k)] = dot(m, view(Zraw_all[k], :, o)) / s
    end
    return nu
end

"""
    companion_implied_nu_cmzc(ctx, x_free, spec; eval_id=0) -> Vector{Float64}

CM+ZC analog: solves the plain CM-only companion (`build_cm_production_context`, the un-widened
CM family, SAME `L`/`contrasts`/`probs` as the restricted family so the CM-grid structure
matches) at this candidate's economic point, and pools the companion LFD's implied moment ACROSS
ORIGINS as well as draws -- `SharedByPowerLayout`'s `nu_k` is a single value shared by every
origin (the "common marginals" assumption), so `nu_k = sum_s sum_o m_s*z_o(s)^k / (D*sum_s m_s)`,
equal-weighting every origin (user-confirmed pooling, 2026-08-09).
"""
function companion_implied_nu_cmzc(ctx, x_free::AbstractVector{Float64}, spec::FamilySeedSpec; eval_id::Int = 0)
    K_mean = spec.K_mean
    D = ctx.D
    Zraw_all, _ = build_raw_mean_pair_matrix_levels(ctx.U, K_mean, 0; μ = ctx.μHat)
    pcx0 = build_cm_production_context(ctx, CS; L = spec.L, include_truncated_moment = spec.include_truncated_moment,
        contrasts = spec.contrasts, probs = spec.probs, moment_representation = :operator)
    _, base, verify = cm_production_value_verified_screened(collect(x_free), pcx0; eval_id = eval_id)
    is_verified_success(verify) || throw(CMExpectedSolveFailure(
        "companion_implied_nu_cmzc: CM-only companion failed to verify at this candidate (inner_status=$(verify.inner_status))"))
    m = base.m_star
    s = sum(m)
    nu = Vector{Float64}(undef, K_mean)
    for k in 1:K_mean
        nu[k] = sum(dot(m, view(Zraw_all[k], :, o)) for o in 1:D) / (D * s)
    end
    return nu
end

"Derived focal nu value at this economic point (nothing for common_frechet / non-profiled families) -- pure algebra, no solver call. Unrelated to the companion-LFD nu policy above: this ONE coordinate is not independently identified (collinearity, not a modeling choice), see section 8's header."
function derived_focal_nu(ctx, fb::FamilyBuild, x_free::AbstractVector{Float64})
    fb.aml === nothing && return nothing
    if fb.spec.kind == :origin_zc
        return originzc_profiled_nu_value(x_free, ctx)
    elseif fb.spec.kind == :cm_zc
        return meanzc_profiled_nu_value(x_free, ctx)
    end
    return nothing
end

# ============================================================================
# 9. Per-family verified evaluation (task section 11/22). Uses the SAME screened production
#    evaluators as the real production drivers -- cheap exact-certificate prescreen (section 21)
#    then the identical KNITRO solve, same lower_limit=-10.0 inherited unchanged from ctx.
# ============================================================================

struct FamilyLiftResult
    family_id::Symbol
    kind::Symbol
    full_outer_vector_digest::String
    nu_values::Vector{Float64}
    nu_policy::Symbol
    derived_focal_nu::Union{Nothing,Float64}
    Delta_star::Float64
    verified::Bool
    inner_status::Int
    verification_class::Symbol
    layout_digest::String
    wall_seconds::Float64
end

"""
    dense_nu_for_solve(fb, nu0_active, focal_nu) -> Vector{Float64}

Reconstructs the DENSE nu vector the real production checkpointed drivers always feed the inner
solve, via `scatter_nu_eff` -- matching `run_originzc_upper_checkpointed`/`run_cm_upper_checkpointed`'s
own `νvec = aml === nothing ? νvec_active : scatter_nu_eff(aml, νvec_active, ...profiled_nu_value(xf,ctx))`
pattern (`cm_originzc_checkpoint.jl`/`cm_checkpoint.jl`) exactly. Confirmed live 2026-08-08: passing
the ACTIVE (omission-reduced) nu vector directly to `inner_loop_internal_originzc_operator`/
`inner_loop_internal_meanzc_operator` corrupts `mean_targets`'s `target_index` lookups --
`refresh_zc_targets!`/`mean_targets` are written to consume the FULL dense nu (any placeholder value
survives at the omitted slot, since `op.mean_active_origins`/`mean_offset`, already aml-aware from
`ZCRestrictionOperator`'s own construction, discard it downstream) and internally handle the actual
active-origin reduction on the OUTPUT restriction targets, not on the INPUT nu vector. `nu0_active`
is the (already active-length) companion-LFD-implied nu (section 8) -- this function only handles
the SEPARATE focal-omission reconstruction, agnostic to where nu0_active came from.
"""
function dense_nu_for_solve(fb::FamilyBuild, nu0_active::Vector{Float64}, focal_nu::Union{Nothing,Float64})
    fb.aml === nothing && return nu0_active
    return scatter_nu_eff(fb.aml, nu0_active, focal_nu::Float64)
end

function evaluate_family(ctx, fb::FamilyBuild, x_free::AbstractVector{Float64}; eval_id::Int = 0)
    t0 = time()
    if fb.spec.kind == :origin_zc
        nu0_dense = companion_implied_nu_originzc(ctx, x_free, fb.layout::OriginByPowerLayout; eval_id = eval_id)
        nu0_active = fb.aml === nothing ? nu0_dense : nu0_dense[setdiff(1:length(nu0_dense), fb.aml.dense_omit_idx)]
        focal_nu = derived_focal_nu(ctx, fb, x_free)
        nu_dense = dense_nu_for_solve(fb, nu0_active, focal_nu)
        K, base, verify = cm_originzc_production_value_verified_screened(x_free, nu_dense, fb.pcx; eval_id = eval_id)
        full_vec = vcat(x_free, nu_dense)
    elseif fb.spec.kind == :cm_zc
        nu0_dense = companion_implied_nu_cmzc(ctx, x_free, fb.spec; eval_id = eval_id)
        nu0_active = fb.aml === nothing ? nu0_dense : nu0_dense[setdiff(1:length(nu0_dense), fb.aml.dense_omit_idx)]
        focal_nu = derived_focal_nu(ctx, fb, x_free)
        nu_dense = dense_nu_for_solve(fb, nu0_active, focal_nu)
        K, base, verify = cm_meanzc_production_value_verified_screened(x_free, nu_dense, fb.pcx; eval_id = eval_id)
        full_vec = vcat(x_free, nu_dense)
    elseif fb.spec.kind == :common_frechet
        nu0_active = Float64[]
        focal_nu = nothing
        K, base, verify = cm_frechet_production_value_verified_screened(x_free, fb.pcx; eval_id = eval_id)
        full_vec = x_free
    elseif fb.spec.kind == :unrestricted
        nu0_active = Float64[]
        focal_nu = nothing
        verify, _prof_meta = evaluate_fullA_screened_ranged(collect(x_free), ctx, fb.pcx;
            moment_representation = :compressed, use_cache = false, use_witness = false)
        full_vec = x_free
    else
        @assert fb.spec.kind == :cm_only "evaluate_family: unknown family kind :$(fb.spec.kind) for spec $(fb.spec.id)"
        nu0_active = Float64[]
        focal_nu = nothing
        K, base, verify = cm_production_value_verified_screened(x_free, fb.pcx; eval_id = eval_id)
        full_vec = x_free
    end
    wall = time() - t0
    cls = classify_inner_result(verify)
    layout_desc = fb.aml === nothing ? "dense" : "active_omit=$(fb.aml.dense_omit_idx)_kstar_focal=$(fb.aml.kstar)"
    return FamilyLiftResult(fb.spec.id, fb.spec.kind, sha256_of_vector(full_vec), nu0_active,
        :companion_lfd_implied, focal_nu,
        verify.Delta_dual, is_verified_success(verify), verify.inner_status, Symbol(string(cls)),
        sha256_of_string(layout_desc), wall)
end

"""
    qualify_economic_point(ctx, family_specs, x_free, delta_max; eval_id=0) -> (families_attempted, family_results, rejection_reason)

Tests `family_specs` in the CALLER-DECLARED order (task section 12: cheap-to-expensive), stopping
at the first family that fails to verify with `Delta*<=delta_max` -- ALL-family acceptance is
already impossible once one family fails, so no more expensive families are evaluated (task
section 12: failure under one non-nested family does not imply failure under another; simply stop
because all-family acceptance already failed). `rejection_reason===nothing` means every requested
family verified with `Delta*<=delta_max`.
"""
function qualify_economic_point(ctx, family_specs::Vector{FamilySeedSpec}, x_free::Vector{Float64},
                                 delta_max::Float64; eval_id::Int = 0)
    families_attempted = Symbol[]
    family_results = FamilyLiftResult[]
    rejection_reason = nothing
    for spec in family_specs
        push!(families_attempted, spec.id)
        local res
        try
            fb = build_family(ctx, spec)
            res = evaluate_family(ctx, fb, x_free; eval_id = eval_id)
        catch e
            rejection_reason = Symbol("inner_failure_$(spec.id)")
            break
        end
        push!(family_results, res)
        if !(res.verified && isfinite(res.Delta_star) && res.Delta_star <= delta_max)
            rejection_reason = res.verified ? Symbol("Delta_above_max_$(spec.id)") : Symbol("verification_failure_$(spec.id)")
            break
        end
    end
    return families_attempted, family_results, rejection_reason
end

# ============================================================================
# 10. Seed diversity (task section 8).
# ============================================================================

"""
    seed_distance(w1, w2) -> (rms_A_distance, gp_distance, combined_standardized_distance)

`w1`/`w2` are `[gp; a_powered]` economic vectors (same coordinate system perturbations are drawn
in). `combined_standardized_distance = sqrt(rms_A_distance^2 + gp_distance^2)` -- both components
are already in comparable natural units (a-powered-space RMS units and raw gp units), so no
additional standardization framework is introduced (task section 8: "do not invent an aggressive
... framework").
"""
function seed_distance(w1::AbstractVector{Float64}, w2::AbstractVector{Float64})
    length(w1) == length(w2) || error("seed_distance: vectors must be the same length")
    gp1, gp2 = w1[1], w2[1]
    a1, a2 = @view(w1[2:end]), @view(w2[2:end])
    n_A = length(a1)
    rms_A = n_A == 0 ? 0.0 : sqrt(sum(abs2, a1 .- a2) / n_A)
    gp_dist = abs(gp1 - gp2)
    return (rms_A_distance = rms_A, gp_distance = gp_dist, combined_standardized_distance = sqrt(rms_A^2 + gp_dist^2))
end

# ============================================================================
# 11. Per-attempt evaluation (task section 7/20): a PURE function of (ctx, attempt_id, block_id,
#     rng_seed, manifest_digest, scales) -- no knowledge of M, other seeds, or accept/reject
#     bookkeeping. This is the function an external parallel supervisor should call directly
#     (task section 20) if it wants to run many attempts concurrently; scheduling attempt IDs
#     ahead of time and reordering by attempt_id afterward reproduces the sequential result
#     exactly, since this function's output depends on nothing but its own arguments.
# ============================================================================

function evaluate_attempt(ctx, geo::AspaceGeometry, w_cal::Vector{Float64}, gp_cal::Float64, gp_target::Float64,
                           family_specs::Vector{FamilySeedSpec}, attempt_id::Int, block_id::Int,
                           rng_seed::UInt64, manifest_digest::String, A_scale::Float64, gp_scale::Float64,
                           delta_max::Float64; radius_mode::Symbol = :uniform_radius)
    t0 = time()
    seed_key = attempt_rng_seed(rng_seed, attempt_id, manifest_digest)
    rng = Random.Xoshiro(seed_key)
    gp_c, gp_fraction = draw_gp_perturbation(rng, gp_cal, gp_target, gp_scale)
    delta_a, A_radius = draw_A_perturbation(rng, geo.n_A, A_scale; radius_mode = radius_mode)
    w_econ = vcat(gp_c, w_cal[2:end] .+ delta_a)
    x_free = decode_w_econ(geo, w_econ)
    econ_digest = sha256_of_vector(w_econ)
    grav_ok = precheck_candidate(x_free)
    if grav_ok
        families_attempted, family_results, rejection_reason = qualify_economic_point(ctx, family_specs, x_free, delta_max; eval_id = attempt_id)
    else
        families_attempted, family_results, rejection_reason = Symbol[], FamilyLiftResult[], :economic_domain_failure
    end
    wall = time() - t0
    return (attempt_id = attempt_id, block_id = block_id, rng_seed_hex = string(seed_key, base = 16),
            A_scale = A_scale, A_radius = A_radius, gp_scale = gp_scale, gp_fraction = gp_fraction, gp = gp_c,
            w_econ = w_econ, x_free = x_free, economic_digest = econ_digest, gravity_reconstruction_pass = grav_ok,
            families_attempted = families_attempted, family_results = family_results,
            rejection_reason = rejection_reason, wall_seconds = wall)
end

# ============================================================================
# 12. Public result types (task section 2).
# ============================================================================

struct AcceptedSeed
    seed_id::String
    attempt_id::Int
    economic_vector::Vector{Float64}       # w_econ = [gp; a_powered]
    economic_digest::String
    gp::Float64
    A_perturbation_radius::Float64
    A_perturbation_rms::Float64            # == A_perturbation_radius by construction (draw_A_perturbation)
    gp_perturbation_fraction::Float64
    families::Vector{FamilyLiftResult}
end

struct MultiStartSeedSet
    manifest_digest::String
    source_sha::String
    request::NamedTuple
    accepted::Vector{AcceptedSeed}
    ledger::Vector{<:NamedTuple}            # one row per attempt (incl. calibration if requested), in attempt_id order
    n_attempted::Int
    n_accepted::Int
    stop_reason::Symbol                     # :target_reached | :attempt_limit
end

git_head_sha() = try
    strip(read(`git -C $(normpath(joinpath(@__DIR__, "..", ".."))) rev-parse HEAD`, String))
catch
    "unknown"
end

# ============================================================================
# 13. Public entry point.
# ============================================================================

"""
    generate_multistart_seeds(ctx; M, direction, delta_max, family_specs, W, rng_seed, A_scale,
        gp_scale=1.0, max_attempts=200, include_calibration=true, min_seed_distance=0.0,
        max_concurrency=1, radius_mode=:uniform_radius, scale_schedule=nothing, output_dir,
        source_sha="") -> MultiStartSeedSet

Generates up to `M` common economic multistart seeds (including calibration itself as `S0` when
`include_calibration=true`, task section 17) by randomly perturbing ONLY the free powered-a-space
A coordinates + gp around calibration (task section 3-6), requiring EVERY requested family in
`family_specs` to independently verify `Delta*<=delta_max` (task section 11) before accepting a
point, and writing a complete reproducible attempt ledger + per-seed/per-family outer vectors to
`output_dir` (task section 18-19). See
docs/audits/reproducible-multistart-generator-2026-08-08/MASTER.md for the full contract.

`ctx` must be built via `d20_real_setup_design(...)` (not the bare `d20_real_setup`) at the SAME
`W` this call requests -- sigma/K_mean/K_pair/L/gravity exclusions/destination_sample/draw_design/
draw_seed/inner_lower_limit are never re-defaulted here; they are read directly off `ctx` (already
fully specified by the caller) and off `family_specs` (required fields, no defaults).
"""
function generate_multistart_seeds(ctx;
        M::Int, direction::Symbol, delta_max::Float64, family_specs::Vector{FamilySeedSpec},
        W::Int, rng_seed::UInt64, A_scale::Float64, gp_scale::Float64 = 1.0,
        max_attempts::Int = 200, include_calibration::Bool = true, min_seed_distance::Float64 = 0.0,
        max_concurrency::Int = 1, radius_mode::Symbol = :uniform_radius,
        scale_schedule::Union{Nothing,Vector{<:NamedTuple}} = nothing,
        output_dir::AbstractString, source_sha::AbstractString = "")
    M >= 1 || error("generate_multistart_seeds: M must be >= 1, got $(M)")
    direction in (:upper, :lower) || error("generate_multistart_seeds: direction must be :upper or :lower, got :$(direction)")
    delta_max > 0 || error("generate_multistart_seeds: delta_max must be > 0, got $(delta_max)")
    isempty(family_specs) && error("generate_multistart_seeds: family_specs must be non-empty")
    length(unique(s.id for s in family_specs)) == length(family_specs) ||
        error("generate_multistart_seeds: family_specs ids must be unique, got $([s.id for s in family_specs])")
    A_scale > 0 || error("generate_multistart_seeds: A_scale must be > 0, got $(A_scale)")
    0.0 < gp_scale <= 1.0 || error("generate_multistart_seeds: gp_scale must be in (0,1], got $(gp_scale)")
    max_attempts >= 0 || error("generate_multistart_seeds: max_attempts must be >= 0, got $(max_attempts)")
    min_seed_distance >= 0 || error("generate_multistart_seeds: min_seed_distance must be >= 0, got $(min_seed_distance)")
    radius_mode in (:uniform_radius, :fixed_radius) || error("generate_multistart_seeds: radius_mode must be :uniform_radius or :fixed_radius, got :$(radius_mode)")
    max_concurrency == 1 || error("generate_multistart_seeds: max_concurrency>1 is not implemented in this release " *
        "(task section 20 permits this fallback) -- call evaluate_attempt(...) directly per attempt_id from an " *
        "external process-pool supervisor instead, then feed accepted attempt_ids back in increasing order.")
    (hasproperty(ctx, :draw_design) && hasproperty(ctx, :draw_seed)) ||
        error("generate_multistart_seeds: ctx must be built via d20_real_setup_design(...), not the bare " *
              "d20_real_setup(...) -- ctx.draw_design/ctx.draw_seed are required for the manifest digest/" *
              "reproducibility contract.")
    ctx.W == W || error("generate_multistart_seeds: requested W=$(W) does not match ctx.W=$(ctx.W) -- " *
        "qualification must use the SAME W the campaign will use (task section 14); build ctx at the " *
        "requested W, do not qualify at one W and label the result valid at another.")
    ctx.obj.lower_limit == -10.0 || error("generate_multistart_seeds: ctx.obj.lower_limit must be the current " *
        "production value -10.0, got $(ctx.obj.lower_limit).")

    isdir(output_dir) || mkpath(output_dir)
    # Required once per outer-solve process, before any real family evaluation -- exactly what
    # every real production driver (e.g. run_cm_upper_checkpointed) does immediately after
    # building `ctx`, and before `build_pivot_elimination`. Idempotent/no-op if `ctx` already
    # carries a correctly-shaped workspace (attach_compressed_factual_workspace's own contract).
    ctx = attach_compressed_factual_workspace(ctx, ctx.D, ctx.D_dest, W)
    manifest_digest = compute_manifest_digest(ctx, family_specs, W)
    geo = build_aspace_geometry(ctx)
    w_cal = cm_w0_from_calibration(ctx, geo.pe, :powered_aspace)
    gp_cal, gp_target = gp_calibration_and_target(ctx, direction)
    src_sha = isempty(source_sha) ? git_head_sha() : String(source_sha)

    ledger = NamedTuple[]
    accepted = AcceptedSeed[]

    if include_calibration
        x_free_cal = decode_w_econ(geo, w_cal)
        fams0, fam_results0, rej0 = qualify_economic_point(ctx, family_specs, x_free_cal, delta_max; eval_id = 0)
        row0 = (attempt_id = 0, block_id = 0, rng_seed_hex = "calibration", A_scale = 0.0, A_radius = 0.0,
            gp_scale = 0.0, gp_fraction = 0.0, gp = gp_cal, economic_digest = sha256_of_vector(w_cal),
            gravity_reconstruction_pass = true, duplicate_distance = nothing, nearest_seed_id = nothing,
            families_attempted = fams0, family_results = fam_results0, accepted = (rej0 === nothing),
            rejection_reason = rej0, wall_seconds = sum((fr.wall_seconds for fr in fam_results0); init = 0.0))
        push!(ledger, row0)
        if rej0 !== nothing
            error("generate_multistart_seeds: CALIBRATION FAILED to verify Delta*<=delta_max=$(delta_max) " *
                  "under family/reason :$(rej0) -- this is a real scientific problem with the requested family " *
                  "set / delta_max / ctx, not something to silently skip (task section 17). Family results: " *
                  "$(fam_results0)")
        end
        push!(accepted, AcceptedSeed("S0", 0, w_cal, sha256_of_vector(w_cal), gp_cal, 0.0, 0.0, 0.0, fam_results0))
    end

    M_random = include_calibration ? M - 1 : M
    blocks = scale_schedule === nothing ? [(A_scale = A_scale, gp_scale = gp_scale, attempts = max_attempts)] : scale_schedule
    stop_reason = :target_reached
    global_attempt_id = 0

    if M_random > 0
        stop_reason = :attempt_limit
        block_loop_done = false
        for (block_id, blk) in enumerate(blocks)
            block_loop_done && break
            for _ in 1:blk.attempts
                global_attempt_id += 1
                res = evaluate_attempt(ctx, geo, w_cal, gp_cal, gp_target, family_specs, global_attempt_id, block_id,
                    rng_seed, manifest_digest, blk.A_scale, blk.gp_scale, delta_max; radius_mode = radius_mode)
                accepted_flag = false
                dup_dist = nothing
                nearest_id = nothing
                reason = res.rejection_reason
                if reason === nothing
                    best_d = nothing
                    best_id = nothing
                    for s in accepted
                        d = seed_distance(res.w_econ, s.economic_vector).combined_standardized_distance
                        if best_d === nothing || d < best_d
                            best_d = d
                            best_id = s.seed_id
                        end
                    end
                    dup_dist = best_d
                    nearest_id = best_id
                    if best_d !== nothing && best_d < min_seed_distance
                        reason = :too_close_to_existing_seed
                    else
                        accepted_flag = true
                    end
                end
                row = (attempt_id = global_attempt_id, block_id = block_id, rng_seed_hex = res.rng_seed_hex,
                    A_scale = res.A_scale, A_radius = res.A_radius, gp_scale = res.gp_scale, gp_fraction = res.gp_fraction,
                    gp = res.gp, economic_digest = res.economic_digest, gravity_reconstruction_pass = res.gravity_reconstruction_pass,
                    duplicate_distance = dup_dist, nearest_seed_id = nearest_id, families_attempted = res.families_attempted,
                    family_results = res.family_results, accepted = accepted_flag, rejection_reason = reason,
                    wall_seconds = res.wall_seconds)
                push!(ledger, row)
                if accepted_flag
                    sid = "S$(length(accepted))"
                    push!(accepted, AcceptedSeed(sid, global_attempt_id, res.w_econ, res.economic_digest, res.gp,
                        res.A_radius, res.A_radius, res.gp_fraction, res.family_results))
                end
                # Live progress + incremental checkpoint (2026-08-08): without this, a long real
                # run gives ZERO visibility until it either finishes or is killed -- confirmed live
                # this is a genuine usability problem, not a cosmetic one, when running unattended
                # overnight. Prints one line per attempt and re-writes the FULL output set (cheap;
                # write_seed_set overwrites in place) after every attempt, not just on acceptance,
                # so `tail -f attempts.csv` / re-reading seeds/manifest.jls at any time reflects
                # real, current progress -- including whatever has been accepted SO FAR if this
                # process is killed mid-run.
                println("[multistart] attempt ", global_attempt_id, "/", max_attempts, " block=", block_id,
                    " accepted=", accepted_flag, reason === nothing ? "" : " reason=$(reason)",
                    " families_reached=", length(res.families_attempted), "/", length(family_specs),
                    " progress=", length(accepted), "/", M, " wall=", round(res.wall_seconds, digits = 1), "s")
                flush(stdout)
                let interim_request = (M = M, direction = direction, delta_max = delta_max,
                        family_ids = [s.id for s in family_specs], W = W, rng_seed = rng_seed, A_scale = A_scale,
                        gp_scale = gp_scale, max_attempts = max_attempts, include_calibration = include_calibration,
                        min_seed_distance = min_seed_distance, max_concurrency = max_concurrency,
                        radius_mode = radius_mode, output_dir = String(output_dir), scale_schedule = scale_schedule)
                    interim = MultiStartSeedSet(manifest_digest, src_sha, interim_request, accepted, ledger,
                        global_attempt_id, length(accepted), :in_progress)
                    write_seed_set(interim, output_dir)
                end
                if accepted_flag && length(accepted) >= M
                    stop_reason = :target_reached
                    block_loop_done = true
                    break
                end
            end
        end
    end

    request = (M = M, direction = direction, delta_max = delta_max, family_ids = [s.id for s in family_specs],
        W = W, rng_seed = rng_seed, A_scale = A_scale, gp_scale = gp_scale, max_attempts = max_attempts,
        include_calibration = include_calibration, min_seed_distance = min_seed_distance,
        max_concurrency = max_concurrency, radius_mode = radius_mode, output_dir = String(output_dir),
        scale_schedule = scale_schedule)

    result = MultiStartSeedSet(manifest_digest, src_sha, request, accepted, ledger,
        global_attempt_id, length(accepted), stop_reason)
    write_seed_set(result, output_dir)
    return result
end

# ============================================================================
# 14. Output serialization (task section 18-19): attempts.csv/attempts.jsonl (append-only
#     attempt ledger, every attempt including rejected ones), seeds/S<k>/economic_seed.jls,
#     seeds/S<k>/<FAMILY_ID>/full_outer_seed.jls. No hand-rolled JSON framework -- this codebase
#     has no JSON dependency (checked Project.toml), so a minimal escaping encoder is used, scoped
#     to exactly the flat scalar/vector/NamedTuple shapes this file ever writes.
# ============================================================================

json_scalar(x::Nothing) = "null"
json_scalar(x::Bool) = x ? "true" : "false"
json_scalar(x::Integer) = string(x)
json_scalar(x::AbstractFloat) = isfinite(x) ? string(x) : "null"
json_scalar(x::AbstractString) = "\"" * replace(String(x), "\\" => "\\\\", "\"" => "\\\"") * "\""
json_scalar(x::Symbol) = json_scalar(string(x))
json_scalar(x::AbstractVector) = "[" * join((json_scalar(v) for v in x), ",") * "]"

function family_result_json(r::FamilyLiftResult)
    fields = ["\"family_id\":" * json_scalar(r.family_id),
              "\"kind\":" * json_scalar(r.kind),
              "\"full_outer_vector_digest\":" * json_scalar(r.full_outer_vector_digest),
              "\"nu_policy\":" * json_scalar(r.nu_policy),
              "\"derived_focal_nu\":" * json_scalar(r.derived_focal_nu),
              "\"Delta_star\":" * json_scalar(r.Delta_star),
              "\"verified\":" * json_scalar(r.verified),
              "\"inner_status\":" * json_scalar(r.inner_status),
              "\"verification_class\":" * json_scalar(r.verification_class),
              "\"layout_digest\":" * json_scalar(r.layout_digest),
              "\"wall_seconds\":" * json_scalar(r.wall_seconds)]
    return "{" * join(fields, ",") * "}"
end

function ledger_row_json(row)
    fam_json = "[" * join((family_result_json(fr) for fr in row.family_results), ",") * "]"
    fields = ["\"attempt_id\":" * json_scalar(row.attempt_id),
              "\"block_id\":" * json_scalar(row.block_id),
              "\"rng_seed_hex\":" * json_scalar(row.rng_seed_hex),
              "\"A_scale\":" * json_scalar(row.A_scale),
              "\"A_radius\":" * json_scalar(row.A_radius),
              "\"gp_scale\":" * json_scalar(row.gp_scale),
              "\"gp_fraction\":" * json_scalar(row.gp_fraction),
              "\"gp\":" * json_scalar(row.gp),
              "\"economic_digest\":" * json_scalar(row.economic_digest),
              "\"gravity_reconstruction_pass\":" * json_scalar(row.gravity_reconstruction_pass),
              "\"duplicate_distance\":" * json_scalar(row.duplicate_distance),
              "\"nearest_seed_id\":" * json_scalar(row.nearest_seed_id),
              "\"families_attempted\":" * json_scalar(row.families_attempted),
              "\"family_results\":" * fam_json,
              "\"accepted\":" * json_scalar(row.accepted),
              "\"rejection_reason\":" * json_scalar(row.rejection_reason),
              "\"wall_seconds\":" * json_scalar(row.wall_seconds)]
    return "{" * join(fields, ",") * "}"
end

function write_seed_set(result::MultiStartSeedSet, output_dir::AbstractString)
    isdir(output_dir) || mkpath(output_dir)

    open(joinpath(output_dir, "attempts.csv"), "w") do io
        println(io, "attempt_id,block_id,rng_seed_hex,A_scale,A_radius,gp_scale,gp_fraction,gp,economic_digest," *
                     "gravity_reconstruction_pass,duplicate_distance,nearest_seed_id,families_attempted,accepted," *
                     "rejection_reason,wall_seconds,family_deltas")
        for row in result.ledger
            fam_str = join(("$(fr.family_id)=$(fr.Delta_star)($(fr.verified))" for fr in row.family_results), ";")
            println(io, join(Any[row.attempt_id, row.block_id, row.rng_seed_hex, row.A_scale, row.A_radius,
                row.gp_scale, row.gp_fraction, row.gp, row.economic_digest, row.gravity_reconstruction_pass,
                something(row.duplicate_distance, ""), something(row.nearest_seed_id, ""),
                join(row.families_attempted, ";"), row.accepted, something(row.rejection_reason, ""),
                row.wall_seconds, fam_str], ","))
        end
    end

    open(joinpath(output_dir, "attempts.jsonl"), "w") do io
        for row in result.ledger
            println(io, ledger_row_json(row))
        end
    end

    seeds_dir = joinpath(output_dir, "seeds")
    isdir(seeds_dir) || mkpath(seeds_dir)
    for seed in result.accepted
        sdir = joinpath(seeds_dir, seed.seed_id)
        isdir(sdir) || mkpath(sdir)
        serialize(joinpath(sdir, "economic_seed.jls"), (seed_id = seed.seed_id, attempt_id = seed.attempt_id,
            economic_vector = seed.economic_vector, economic_digest = seed.economic_digest, gp = seed.gp,
            A_perturbation_radius = seed.A_perturbation_radius, A_perturbation_rms = seed.A_perturbation_rms,
            gp_perturbation_fraction = seed.gp_perturbation_fraction, manifest_digest = result.manifest_digest,
            source_sha = result.source_sha, W = result.request.W))
        for fr in seed.families
            fdir = joinpath(sdir, String(fr.family_id))
            isdir(fdir) || mkpath(fdir)
            serialize(joinpath(fdir, "full_outer_seed.jls"), (economic_digest = seed.economic_digest,
                layout_digest = fr.layout_digest, full_outer_vector_digest = fr.full_outer_vector_digest,
                nu_policy = fr.nu_policy, nu_values = fr.nu_values, derived_focal_nu = fr.derived_focal_nu,
                Delta_star = fr.Delta_star, verified = fr.verified, inner_status = fr.inner_status,
                verification_class = fr.verification_class, source_sha = result.source_sha,
                manifest_digest = result.manifest_digest, W = result.request.W))
        end
    end

    serialize(joinpath(output_dir, "manifest.jls"), (manifest_digest = result.manifest_digest,
        source_sha = result.source_sha, request = result.request, n_attempted = result.n_attempted,
        n_accepted = result.n_accepted, stop_reason = result.stop_reason,
        accepted_seed_ids = [s.seed_id for s in result.accepted]))

    return output_dir
end

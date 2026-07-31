# ============================================================================
# Continuation 9, Phase 1 integration: real-data D=20 context builder for the
# full-A_od diagnostic machinery. Additive only -- does NOT modify
# context.jl's d4_exact_setup or context_scaled.jl's d_exact_setup_scaled
# (both remain synthetic-only, unchanged). Mirrors context_scaled.jl's
# build_ad_context_scaled / d_exact_setup_scaled pattern exactly, but forces
# fakeData=3 (real data, see setup/importData.jl) and DFake=20 instead of
# accepting an arbitrary synthetic D.
#
# Real-data support (setup/importData.jl's fakeData==3 branch,
# setup/defineCounter.jl's autarky tau^Inf fix) was ported into this worktree
# from trade_robustness_modular_perf @ 778c362 (feature/sequential-inversion-perf)
# -- see docs/fullA_D20_production_path_audit.md for the audit that found this
# worktree previously had NO real-data loading code at all. Data files:
# real_data/noah_D20/{countries,L,pi,tau}.csv, copied verbatim (md5-verified)
# from that worktree.
#
# Focal country: France (baseIndex=2), matches AD_PARAMS's existing default --
# no override needed. sigma=2.5 (assumed), theta estimated via gravity
# (thetaHat=0) -- both already AD_PARAMS defaults, unchanged.
# ============================================================================
isdefined(Main, :d4_exact_setup) || include(joinpath(@__DIR__, "context.jl"))   # -> AD_PARAMS, build_ad_context, master_setup etc., CS, d4_exact_setup -- guarded (2026-07-24) so combining this file with context_scaled.jl in one script doesn't redefine the CS module twice (causes exported-name ambiguity, e.g. active_destinations)
include(joinpath(@__DIR__, "infeasibility_screen.jl"))   # -> precompute_pairwise_M, build_extreme_draw_witness (Continuation 10, Section 5)

const D20_REAL = 20

# exclude-ROW-destination production release (2026-07-24). Recorded provenance for a checkpoint's
# resolved economic sample/estimation code path -- distinct from destination_sample itself
# (:exclude_row|:all_legacy is the RUNTIME choice; these version numbers are the CODE that
# implements whichever choice was made, bumped whenever the underlying sample-construction or
# theta-estimation logic changes).
const GRAVITY_SAMPLE_VERSION = 2       # 1 = pre-release legacy full-sample construction (square
                                        # D x D, single code path); 2 = this release -- gravity
                                        # sample recomputed from raw observations for the resolved
                                        # destination_sample (rectangular for :exclude_row, square
                                        # for :all_legacy), not derived by deleting a column from a
                                        # previously-residualized full-sample object.
const THETA_CALIBRATION_VERSION = 2    # 1 = theta_star estimated once on the full square sample,
                                        # reused for every destination_sample; 2 = this release --
                                        # theta_star (hence mu=1/theta_star and the productivity-
                                        # draw transform) is re-estimated on the resolved sample's
                                        # own gravity regression.

"""
Build (so, pp, params_used) for the REAL D=20 economy at a given W, independent of AD_PARAMS.

`U=nothing` (default): draw U internally (unchanged pseudorandom path). `U` given (`W x D`,
already Exp(1)-transformed): use it directly -- the single injection point for every non-default
draw design (randomized Sobol, scrambled Halton, precomputed), threaded straight to
`master_prepare_cc`. See `draw_design.jl::d20_real_setup_design` for the resolver that decides
which case applies. `exclude_diagonal_gravity`/`σHat` (2026-07-30, sigma=3 campaign prep, merged
from production/fullA-exact) are unrelated to draw design -- unchanged, opt-in passthroughs to
`master_setup`'s params, unified here in the one place both live regardless of draw design.
"""
function build_ad_context_real_d20(; W::Int, row_idx::Union{Nothing,Int} = nothing,
        U::Union{Nothing,AbstractMatrix{Float64}} = nothing,
        exclude_diagonal_gravity::Bool = false,
        # σHat override (2026-07-30, sigma=3 campaign prep): AD_PARAMS.σHat=2.5 is this repo's
        # single source of truth for sigma on the real D20 path (audited -- no CES/gravity formula
        # anywhere hardcodes 2.5 directly; every consumer reads ctx.σ/σHat as a variable). `nothing`
        # (default) reproduces AD_PARAMS's own σHat unchanged, bit-exact with every pre-existing
        # caller -- this is opt-in, not a change to the historical default.
        σHat::Union{Nothing,Float64} = nothing)
    σ_override = σHat === nothing ? NamedTuple() : (σHat = σHat,)
    params = merge(AD_PARAMS, (fakeData = 3, DFake = D20_REAL, W = W, Jac_W = W, row_idx = row_idx,
        exclude_diagonal_gravity = exclude_diagonal_gravity), σ_override)
    so = master_setup(params)
    @assert so.D == D20_REAL "master_setup returned D=$(so.D), expected $(D20_REAL) -- real_data/noah_D20 CSVs may be malformed"
    up = (; params..., D = so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
    checkParams(up)
    ps = master_prestep(so.data, so.counters, up)
    pp = master_prepare_cc(so.data, so.counters, ps, up; U = U)
    return so, pp, params
end

"""
    d20_real_setup(; W, δ=1.0, find_smallest=true) -> NamedTuple

Mirrors `context_scaled.jl::d_exact_setup_scaled` exactly (same free-parameter
layout, bounds construction, PsiObjectiveBundleImplicit wiring) but against the
REAL D=20 dataset (France focal) instead of a synthetic economy at an arbitrary
D. Returns the same field set so every existing D=4 diagnostic function
(`evaluate_fullA`, `compute_winners`, `build_pivot_elimination`, etc.) works
unchanged on the returned `ctx`.

`U=nothing` (default) draws U internally (unchanged pseudorandom path, `Random.seed!(seedU)` via
`master_prepare_cc`). `U` given (`W x D`, already Exp(1)-transformed) uses it directly instead --
this is the ONE injection point every non-pseudorandom draw design (randomized Sobol, scrambled
Halton, precomputed) goes through; screen construction, threshold-state construction, and every
other field below are built identically regardless of which branch supplied `U` (task "unify
random-draw production pipeline" 2026-07-30 §8 -- there is exactly one copy of this logic, not
one per draw design).
"""
function d20_real_setup(; W::Int, δ::Float64 = 1.0, find_smallest::Bool = true,
        outer_loop_opt::AbstractString = joinpath(D4X_ROOT, "full_aod_diag", "csw_outer_25.opt"),
        inner_loop_opt::AbstractString = joinpath(D4X_ROOT, "full_aod_diag", "ek_inner.opt"),
        needs_outer_moment_jacobian::Bool = false,
        build_screen::Bool = true,
        # Part A (2026-07-23 release): omit-ROW-destination as the new production default.
        # :exclude_row -- true dimension shrink (origins stay all 20, destinations become the 19
        # named countries, ROW=country 20 dropped as destination/kept as origin; theta
        # re-estimated on the rectangular sample). :all_legacy -- exact pre-Part-A square D x D
        # behavior, bit-for-bit (regression-safety opt-out). No third option, no silent fallback.
        destination_sample::Symbol = :exclude_row,
        # Draw-design injection point (unify-random-draw-production-pipeline, 2026-07-30): nothing
        # keeps today's internal pseudorandom draw; a caller-supplied W x D Exp(1) matrix routes
        # every other draw design through this exact same setup function.
        U::Union{Nothing,AbstractMatrix{Float64}} = nothing,
        # exclude_diagonal_gravity (2026-07-30, user-directed fix): ALSO drop own-trade (o==d)
        # cells from the theta-identification regression and the outer gravity constraint's
        # coefficient vector, matching the Stata side's `sum_{o!=d,d!=ROW}` restriction. `false`
        # is the default and reproduces every pre-existing caller's behavior bit-exactly -- this
        # is opt-in, not a change to d20_real_setup's historical default.
        exclude_diagonal_gravity::Bool = false,
        # σHat passthrough to build_ad_context_real_d20's own kwarg of the same name (2026-07-30).
        # `nothing` default reproduces AD_PARAMS.σHat=2.5 unchanged.
        σHat::Union{Nothing,Float64} = nothing)
    destination_sample in (:exclude_row, :all_legacy) ||
        error("d20_real_setup: destination_sample must be :exclude_row or :all_legacy, got :$destination_sample")
    row_idx = destination_sample == :exclude_row ? D20_REAL : nothing
    # false is the PRODUCTION default (matches run_fullA_D4_production.jl /
    # run_fullA_D10_production.jl -- neither needs an analytic outer moment
    # Jacobian, both use a ForwardDiff/Method-B gradient path instead), NOT
    # context_scaled.jl's diagnostic default of true. This matters far more
    # here than at D4/D6/8/10: `jac_h` is a dense N x (d+2) x l tensor
    # (draws x moments x full-theta-length) -- at D=20/W=80000 that's
    # 80000 x 404 x 423 x 8 bytes ~= 109GB (confirmed by direct measurement,
    # this session -- a real-data D=20 benchmark run at the old true default
    # stabilized at ~109GB RSS; a W=800000 probe at the same default was
    # killed after climbing to ~780GB and still rising, headed past 1TB).
    # See docs/fullA_D20_production_path_audit.md and the continuation-9
    # W80k/W800k microbenchmark docs for the full incident writeup.
    so, pp, params_used = build_ad_context_real_d20(W = W, row_idx = row_idx, U = U,
        exclude_diagonal_gravity = exclude_diagonal_gravity, σHat = σHat)
    Dact = so.D; bi = params_used.baseIndex; σ = params_used.σHat; μHat = pp.γ.μHat
    # exclude-ROW-destination production release (2026-07-24): reject focal_country==ROW. `bi`
    # (baseIndex, AD_PARAMS's own default is 2=France, see this file's header comment) is the
    # focal country whose gamma-prime the whole outer loop solves for and whose GT the reported
    # kappa is a monotonic function of (frechet_benchmark_gp etc.) -- it is meaningless to ask for
    # a focal country's own GT when that country isn't even a valid destination in the resolved
    # sample, so a focal country coinciding with the omitted ROW destination under :exclude_row is
    # a hard error, not a silently-nonsensical result.
    row_idx === nothing || bi != row_idx ||
        error("d20_real_setup: focal country (baseIndex=$bi) coincides with the omitted ROW " *
              "destination (row_idx=$row_idx) under destination_sample=:exclude_row -- GT is undefined " *
              "for a focal country that is not itself a valid destination in the resolved sample.")
    Ddest = row_idx === nothing ? Dact : Dact - 1
    @unpack θ_initial, θ_initial_up, U, γ, outer_constr_index, nTotalMoments, complement_index, inequality_index = pp
    Aod_offset = 3 + Dact

    θ0_up = build_theta_gammanorm(θ_initial_up, Dact, bi, μHat, σ)
    bounds = theoretical_gammaprime_bounds(γ, σ)
    θ0_up[3+Dact] = clamp(θ0_up[3+Dact], bounds.γp_lo, bounds.γp_hi)

    θ_lo = (θ0_up .* 0.0001)[:]; θ_hi = (θ0_up .* 10000)[:]
    θ_lo[2] = θ0_up[2]; θ_hi[2] = θ0_up[2]
    θ_lo[1] = θ0_up[1]; θ_hi[1] = θ0_up[1]
    for d in 1:Dact
        θ_lo[2+d] = θ0_up[2+d]; θ_hi[2+d] = θ0_up[2+d]
    end
    θ_lo[3+Dact] = bounds.γp_lo; θ_hi[3+Dact] = bounds.γp_hi

    l_full = length(θ0_up)
    free_idx = vcat(3 + Dact, collect(Aod_offset+1:Aod_offset+Dact*Ddest))
    fixed_idx = vcat(1, 2, collect(3:2+Dact))
    fixed_vals = θ0_up[fixed_idx]
    m = CS.FreeParamMap(l_full, free_idx, fixed_idx, fixed_vals)
    @assert CS.n_free(m) == 1 + Dact * Ddest

    Aod_free_pos = [1 + (d - 1) * Dact + o for o in 1:Dact, d in 1:Ddest]
    τ = γ.τ
    q_tilde, N_obs = precompute_q_tilde(τ; exclude_diagonal = exclude_diagonal_gravity)

    obj = CS.PsiObjectiveBundleImplicit(δ = δ, find_smallest = find_smallest, γ = γ,
        (moments!) = EK_moments_gammanorm_directgp!, moments_jacobian! = error, d = nTotalMoments,
        outer_constr_index = outer_constr_index, inequality_index = inequality_index,
        complement_index = complement_index, l = l_full, U = U, N = params_used.Jac_W,
        lower_limit = -50, use_cached_x = true,
        # Part C (2026-07-23 release): delta_auto_reject_threshold=10 production default, gated
        # by the compatibility rule (disables automatically once delta is no longer safely below
        # 10) -- separate from and does not change the pre-existing lower_limit=-50 backstop above.
        threshold_state = CS.ThresholdAbortState(CS.resolve_threshold_for_delta(δ)),
        outer_loop_opt = outer_loop_opt, inner_loop_opt = inner_loop_opt,
        needs_outer_moment_jacobian = needs_outer_moment_jacobian)
    @assert obj.outer_constr_index == obj.d

    # ---- Continuation 10, Section 5: build the exact infeasibility-screen structures
    # ONCE here, at context-construction time, per the standing brief's explicit
    # instruction ("build the exact extreme-draw witness index ONCE during context
    # setup ... not per-evaluation"). Both structures are draw-free at every
    # SUBSEQUENT outer point (they depend only on ctx.U/ctx.D, never on theta), so
    # building them once per ctx and reusing across the whole outer-loop run is
    # exactly the right amortization boundary -- see docs/fullA_D20_infeasibility_screening_report.md
    # sec 6 for the (already independently measured) per-scale cost: 2.47s/0.365GB at
    # W=80,000, 32.5s/3.65GB at W=800,000, both negligible next to a single outer-loop
    # run (hundreds-to-thousands of evaluations) and both already confirmed
    # memory-safe against this repo's own 40GB self-imposed kill threshold.
    # `build_screen=false` is available for callers that do not need the screen
    # (e.g. a microbenchmark that only cares about context build time) -- default
    # `true` per this task's "on by default" requirement.
    screen_pairwise = nothing; screen_witness = nothing
    t_pairwise = NaN; t_witness = NaN
    if build_screen
        # D = origin count, D_dest = destination count (Part A, 2026-07-23); precompute_pairwise_M
        # needs only D (it's an origin x origin object), build_extreme_draw_witness/downstream
        # screens need both -- see infeasibility_screen.jl.
        ctx_min = (U = U, D = Dact, D_dest = Ddest)
        t_pairwise = @elapsed screen_pairwise = precompute_pairwise_M(ctx_min)
        t_witness = @elapsed screen_witness = build_extreme_draw_witness(ctx_min)
    end

    return (so = so, pp = pp, D = Dact, D_dest = Ddest, row_idx = row_idx,
            destination_sample = destination_sample,
            # cc_algo/active_layout.jl accessors (active_origins/active_destinations/active_od_cells):
            # origins are NEVER restricted (ROW is retained as an origin in both modes); destinations
            # are truncated to 1:Ddest under :exclude_row (the omitted destination, row_idx, is by
            # construction the LAST index, Dact -- see Aod_free_pos's own `d in 1:Ddest` convention
            # above, which already treats destination indices 1:Ddest as the active/free set).
            active_origins = Base.OneTo(Dact), active_destinations = Base.OneTo(Ddest),
            gravity_sample_version = GRAVITY_SAMPLE_VERSION, theta_calibration_version = THETA_CALIBRATION_VERSION,
            W = W, bi = bi, σ = σ, μHat = μHat, γ = γ, U = U,
            θ0_up = θ0_up, θ_lo = θ_lo, θ_hi = θ_hi, l_full = l_full,
            free_idx = free_idx, fixed_idx = fixed_idx, fixed_vals = fixed_vals, m = m,
            Aod_offset = Aod_offset, Aod_free_pos = Aod_free_pos,
            τ = τ, q_tilde = q_tilde, N_obs = N_obs, exclude_diagonal_gravity = exclude_diagonal_gravity, obj = obj,
            nTotalMoments = nTotalMoments, outer_constr_index = outer_constr_index,
            bounds = bounds, δ = δ, find_smallest = find_smallest,
            pairwise = screen_pairwise, witness = screen_witness,
            screen_setup_wall = (pairwise = t_pairwise, witness = t_witness))
end

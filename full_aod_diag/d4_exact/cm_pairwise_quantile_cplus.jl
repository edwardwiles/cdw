# ================================================================================================
# CM + pairwise-quantile family (#7) onto Backend C+ (the FACTORIZED price representation),
# 2026-08-12.
#
# WHY. The dense economic path (`build_lfix_base_cache`, lfix_incremental.jl) materializes `price0`
# and `pTsigma0` as `W x D x Ddest` tensors -- 304 MB EACH at W=100,000/D=20/Ddest=19 -- and refills
# them on every gradient call. Backend C+ (`lfix_factorized.jl`) never materializes either: it stores
# the log-space factorization `S_{sod} = logCC_{od} + mulU_{so}` in O(W*D) + O(D^2) and reconstructs
# any cell in O(1) on demand. Every other production family has an adapter onto it
# (`cm_originzc_cplus.jl`, `cm_meanzc_cplus.jl`, `cm_frechet_cplus.jl`, `lfix_cm_cplus.jl`,
# `pairwise_quantile_cplus.jl`); this file is family #7's, and it is a near-verbatim port of the
# standalone pairwise-quantile one.
#
# THE DECOMPOSITION ARGUMENT, and why it carries for THIS family's extra block.
# `composite_gradient_at_Cplus_from_cache` requires only that the restriction term in
#     q_s(theta) = -zeta* - lambda_E*'E_s(theta) - lambda_R*'G^R_s - lambda_CM*'G^CM_s
# be CONSTANT across every outer `(gp, A_od)` probe. Both of this family's restriction blocks are:
#
#   * level + pair rows are indicators of `bin[w,o]`, the PQ bin assignment on the Frechet draws,
#     whose cutoffs are campaign constants (selected once from CM's grid). No theta dependence.
#   * CM-grid rows are `1{U_o <= z_l} - 1{U_ref <= z_l}` on CM's own fixed thresholds -- functions of
#     the draws and the grid only. The eq.36 (`n_families==2`) weights are
#     `Pow = frechet_power_feature(U, sigma-1, muHat)`, and BOTH `sigma` and `muHat` are campaign
#     constants, not free outer coordinates (`context_real_d20.jl`: `free_idx` is `gp` and the
#     `A_od` block only). So the truncated-power block is theta-independent too.
#
# Hence `composite_gradient_at_Cplus_from_cache` is reused here COMPLETELY UNMODIFIED, exactly as
# the other five adapters do.
#
# AND THE CONSTANCY IS ABOUT THE DERIVATIVE, NOT `q0`. `q0` is the per-draw LEVEL the economic block
# linearizes AROUND; dropping either restriction term linearizes about the wrong point. This family
# has TWO such terms, so the fold is factored into ONE shared function,
# `cmpq_restriction_q0_contribution!` (cm_pairwise_quantile_outer_production.jl), called by the dense
# builder and by this one -- there is deliberately no second copy of it here to drift. Its exact
# cross-check against the verifier's independently recomputed `r` is likewise shared
# (`cmpq_assert_q0_matches_r`), and `LFixBaseCacheC` carries its own `q0` field, so nothing about
# that check changes under the factorized representation.
#
# PURELY ADDITIVE: modifies no shared economic file, and the dense path stays available as the
# reference the gate compares against (test_cm_pairwise_quantile_cplus_gate.jl).
# ================================================================================================

isdefined(Main, :with_q0_C) || include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
isdefined(Main, :build_lfix_base_cache_C!) || include(joinpath(@__DIR__, "lfix_factorized_workspace.jl"))
isdefined(Main, :cmpq_restriction_q0_contribution!) ||
    include(joinpath(@__DIR__, "cm_pairwise_quantile_outer_production.jl"))

"""
    build_lfix_base_cache_cmpq_C!(ws, x_free0, ctx_cm, base; verify=nothing, q0_check_tol=1e-8,
                                  validate_dense=false) -> LFixBaseCacheC

Backend C+ twin of `build_lfix_base_cache_cmpq`. Calls `build_lfix_base_cache_C!` UNCHANGED, then
folds in BOTH of this family's restriction blocks via the SHARED
`cmpq_restriction_q0_contribution!` and `with_q0_C`.

SIGN: the fold helper's operators SUBTRACT into their accumulator, so starting from zeros it yields
exactly the term `q0` is missing and it is ADDED -- the same convention, and literally the same
function, the dense path uses.
"""
function build_lfix_base_cache_cmpq_C!(ws::LFixFactorizedWorkspace, x_free0::AbstractVector,
        ctx_cm, base::BaseDualState; verify = nothing, q0_check_tol::Float64 = 1e-8,
        validate_dense::Bool = false)
    cache0 = build_lfix_base_cache_C!(ws, x_free0, ctx_cm, base; validate_dense = validate_dense)
    contrib = zeros(ctx_cm.cmpq_ctx.op.W)
    cmpq_restriction_q0_contribution!(contrib, ctx_cm, base)
    q0_new = cache0.q0 .+ contrib
    cmpq_assert_q0_matches_r(q0_new, verify, q0_check_tol, "build_lfix_base_cache_cmpq_C!")
    return with_q0_C(cache0, q0_new)
end

"""
    cm_pairwise_quantile_production_gradient_cplus(x_free0, raw_masses, pcx, ctx, pe, pool, ws;
        base=nothing, verify=nothing, timers=nothing, kwargs...) -> (g_ext, meta)

`:cplus`-backend analog of `cm_pairwise_quantile_production_gradient`. Returns the SAME
`vcat(g_econ, g_mass)` vector, length `D*Ddest + (L-1)`, in the same coordinate order -- only the
economic half is computed through the factorized representation.

The restriction half is untouched and already free: `cmpq_mass_gradient_vec` is the exact closed-form
envelope derivative over `L-1` coordinates and measured 0.0000 s at D=4. (The standalone family's,
over `D*(L-1)`, measured 0.0001 s at W=100,000 -- this one has 20x fewer coordinates.)

`pool`/`ws` are the caller-owned persistent workspaces from `cm_pairwise_quantile_cplus_workspaces`
-- built ONCE per run, never per call, which is the entire point of the backend.
"""
function cm_pairwise_quantile_production_gradient_cplus(x_free0::AbstractVector,
        raw_masses::AbstractVector{Float64}, pcx, ctx, pe, pool::GradWorkspacePool,
        ws::LFixFactorizedWorkspace; base::Union{Nothing,BaseDualState} = nothing, verify = nothing,
        timers::Union{Nothing,CMPQGradTimers} = nothing, kwargs...)
    t_enter = time()
    if base === nothing || verify === nothing
        t0 = time()
        base, verify = archCMPQ_verified_state(x_free0, raw_masses, pcx.ctx_cm)
        timers === nothing || (timers.t_solve += time() - t0)
    end
    # Same mutable-ambient-state hazard as the dense path: cmpq_mass_state is per-outer-point and
    # some intervening solve may have moved it. See ensure_cmpq_masses!.
    ensure_cmpq_masses!(pcx.ctx_cm, raw_masses)
    t0 = time()
    cache = build_lfix_base_cache_cmpq_C!(ws, x_free0, pcx.ctx_cm, base; verify = verify)
    timers === nothing || (timers.t_cache += time() - t0)
    t0 = time()
    g_econ, meta = composite_gradient_at_Cplus_from_cache(x_free0, pcx.ctx_cm, pe, pool, cache;
        base = base, kwargs...)
    timers === nothing || (timers.t_econ += time() - t0)
    t0 = time()
    g_mass = cmpq_mass_gradient_vec(base, verify, pcx.ctx_cm, raw_masses)
    timers === nothing || (timers.t_mass += time() - t0)
    timers === nothing || (timers.t_total += time() - t_enter)
    return vcat(g_econ, g_mass), meta
end

"""
    cm_pairwise_quantile_cplus_workspaces(ctx) -> (pool, ws)

The two persistent Backend C+ workspaces this family needs, sized off `ctx`. Build ONCE per run and
hold them for its lifetime -- rebuilding per gradient call would reintroduce exactly the allocation
this backend exists to remove.
"""
function cm_pairwise_quantile_cplus_workspaces(ctx)
    D = ctx.D
    Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    W = size(ctx.obj.U, 1)
    return (build_grad_workspace_pool(W), build_lfix_factorized_workspace(D, Ddest, W))
end

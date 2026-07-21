# ============================================================================
# Shared D=4 exact full-A context for this diagnostic investigation
# (diag/fullA-d4-exact branch). Thin, additive wrapper around the EXISTING,
# already-validated setup in full_aod_diag/ad_benchmark/setup_context.jl and
# full_aod_diag/moments_gammanorm.jl -- does not redefine anything those files
# already provide. Every script under full_aod_diag/d4_exact/ should
# `include(joinpath(@__DIR__, "context.jl"))` first.
# ============================================================================
const D4X_ROOT = dirname(dirname(@__DIR__))   # repo root (two levels up from d4_exact/)
const ADB = joinpath(D4X_ROOT, "full_aod_diag", "ad_benchmark")
include(joinpath(ADB, "setup_context.jl"))     # -> AD_PARAMS, build_ad_context, CS, EK_moments_gammanorm_directgp! etc.
CS.include(joinpath(D4X_ROOT, "full_aod_diag", "PsiObjectiveBundleImplicitMethodB_fullA.jl"))
CS.include(joinpath(@__DIR__, "parallelism_guards.jl"))   # ported from diag/fullA-inner-blas-threading: guard_enter/exit_inner_solve!, guard_enter/exit_coord_pool! -- injected into the CS module namespace so inner_loop_KNITRO (cc_algo/inner_loop_functions.jl) can call them unqualified; Main-scope callers use CS.guard_*
include(joinpath(D4X_ROOT, "full_aod_diag", "gravity_tariff.jl"))
include(joinpath(ADB, "derivative_core.jl"))   # -> envelope_scalar_div_ctx, moment_map!

"""
    d4_exact_setup() -> NamedTuple

Builds the standard D=4 synthetic economy + FreeParamMap + PsiObjectiveBundleImplicit
bundle used throughout this investigation, identical in construction to
`full_aod_diag/run_fullA_D4_production.jl` (lines 33-77) -- reused verbatim so every
diagnostic in this directory targets EXACTLY the same free-parameter layout, bounds,
and moment/gravity wiring as the production driver, not a re-derived variant.
"""
function d4_exact_setup(; δ::Float64 = 1.0, find_smallest::Bool = true,
                          outer_loop_opt::AbstractString = joinpath(D4X_ROOT, "full_aod_diag", "csw_outer_25.opt"),
                          inner_loop_opt::AbstractString = joinpath(D4X_ROOT, "full_aod_diag", "ek_inner.opt"),
                          needs_outer_moment_jacobian::Bool = true)
    so, pp = build_ad_context()
    D = so.D; bi = AD_PARAMS.baseIndex; σ = AD_PARAMS.σHat; μHat = pp.γ.μHat
    @unpack θ_initial, θ_initial_up, U, γ, outer_constr_index, nTotalMoments, complement_index, inequality_index = pp
    Aod_offset = 3 + D

    θ0_up = build_theta_gammanorm(θ_initial_up, D, bi, μHat, σ)
    bounds = theoretical_gammaprime_bounds(γ, σ)
    θ0_up[3+D] = clamp(θ0_up[3+D], bounds.γp_lo, bounds.γp_hi)

    θ_lo = (θ0_up .* 0.0001)[:]; θ_hi = (θ0_up .* 10000)[:]
    θ_lo[2] = θ0_up[2]; θ_hi[2] = θ0_up[2]
    θ_lo[1] = θ0_up[1]; θ_hi[1] = θ0_up[1]
    for d in 1:D
        θ_lo[2+d] = θ0_up[2+d]; θ_hi[2+d] = θ0_up[2+d]
    end
    θ_lo[3+D] = bounds.γp_lo; θ_hi[3+D] = bounds.γp_hi

    l_full = length(θ0_up)
    free_idx = vcat(3 + D, collect(Aod_offset+1:Aod_offset+D^2))
    fixed_idx = vcat(1, 2, collect(3:2+D))
    fixed_vals = θ0_up[fixed_idx]
    m = CS.FreeParamMap(l_full, free_idx, fixed_idx, fixed_vals)
    @assert CS.n_free(m) == 1 + D^2

    Aod_free_pos = [1 + (d - 1) * D + o for o in 1:D, d in 1:D]
    τ = γ.τ
    q_tilde, N_obs = precompute_q_tilde(τ)

    obj = CS.PsiObjectiveBundleImplicit(δ = δ, find_smallest = find_smallest, γ = γ,
        (moments!) = EK_moments_gammanorm_directgp!, moments_jacobian! = error, d = nTotalMoments,
        outer_constr_index = outer_constr_index, inequality_index = inequality_index,
        complement_index = complement_index, l = l_full, U = U, N = AD_PARAMS.Jac_W,
        lower_limit = -50, use_cached_x = true,
        outer_loop_opt = outer_loop_opt, inner_loop_opt = inner_loop_opt,
        needs_outer_moment_jacobian = needs_outer_moment_jacobian)
    @assert obj.outer_constr_index == obj.d

    return (so = so, pp = pp, D = D, bi = bi, σ = σ, μHat = μHat, γ = γ, U = U,
            θ0_up = θ0_up, θ_lo = θ_lo, θ_hi = θ_hi, l_full = l_full,
            free_idx = free_idx, fixed_idx = fixed_idx, fixed_vals = fixed_vals, m = m,
            Aod_offset = Aod_offset, Aod_free_pos = Aod_free_pos,
            τ = τ, q_tilde = q_tilde, N_obs = N_obs, obj = obj,
            nTotalMoments = nTotalMoments, outer_constr_index = outer_constr_index,
            bounds = bounds, δ = δ, find_smallest = find_smallest)
end

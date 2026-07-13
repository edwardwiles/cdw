# ============================================================================
# Full-A production driver, D=4: exact-point cache + free-only ForwardDiff
# envelope gradient (mu removed from the differentiated/optimized vector) +
# tariff-residualized analytic gravity gradient. Wires together
# cc_algo/free_param_map.jl, cc_algo/outer_eval_cache.jl,
# cc_algo/outer_loop_cached.jl, full_aod_diag/gravity_tariff.jl.
#
# Validates against full_aod_diag/PsiObjectiveBundleImplicitMethodB_fullA.jl
# (today's best-prior validated full-A path: ForwardDiff-over-full-theta,
# trivial gravity gradient, no cache, mu still nominally in theta with equal
# bounds) at the SAME starting point/bounds/draws/delta. The gravity
# CONSTRAINT SET is unchanged under the new tariff formula (both are "=0" for
# a moment that's a positive rescaling of the same underlying quantity -- see
# gravity_tariff.jl's FWL derivation), so final kappa/gravity-residual should
# match to solver tolerance; only the internal gradient computation path,
# vector dimensionality, and caching differ.
#
# Run: julia --project=. full_aod_diag/run_fullA_D4_production.jl
# ============================================================================
using Parameters, Base.Threads, Random, Dates, DelimitedFiles, Printf
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, JLD2
const ADB = joinpath(dirname(@__DIR__), "full_aod_diag", "ad_benchmark")
include(joinpath(ADB, "setup_context.jl"))
CS.include(joinpath(dirname(@__DIR__), "full_aod_diag", "PsiObjectiveBundleImplicitMethodB_fullA.jl"))
include(joinpath(dirname(@__DIR__), "full_aod_diag", "gravity_tariff.jl"))
include(joinpath(ADB, "derivative_core.jl"))   # envelope_scalar_div_ctx

const OUT = @__DIR__
const OUTER_OPT = joinpath(dirname(@__DIR__), "full_aod_diag", "csw_outer_25.opt")
const INNER_OPT = joinpath(dirname(@__DIR__), "full_aod_diag", "ek_inner.opt")

so, pp = build_ad_context()
D = so.D; bi = AD_PARAMS.baseIndex; σ = AD_PARAMS.σHat; μHat = pp.γ.μHat
@unpack θ_initial, θ_initial_up, U, γ, outer_constr_index, nTotalMoments, complement_index, inequality_index = pp
Aod_offset = 3 + D

θ0_up = build_theta_gammanorm(θ_initial_up, D, bi, μHat, σ)
bounds = theoretical_gammaprime_bounds(γ, σ)
θ0_up[3+D] = clamp(θ0_up[3+D], bounds.γp_lo, bounds.γp_hi)

# ---- bounds, matching run_fullA_D10_methodB.jl's convention exactly (mu FIXED) ----
θ_lo = (θ0_up .* 0.0001)[:]; θ_hi = (θ0_up .* 10000)[:]
θ_lo[2] = θ0_up[2]; θ_hi[2] = θ0_up[2]                      # sigma pinned
θ_lo[1] = θ0_up[1]; θ_hi[1] = θ0_up[1]                      # mu FIXED (equal bounds; genuinely removed from x_free below)
for d in 1:D
    θ_lo[2+d] = θ0_up[2+d]; θ_hi[2+d] = θ0_up[2+d]          # gamma_theta slots pinned/inert
end
θ_lo[3+D] = bounds.γp_lo; θ_hi[3+D] = bounds.γp_hi

println(">>> D=$D  l_full=$(length(θ0_up))  mu=$(θ0_up[1])  sigma=$(θ0_up[2])")

# ---- FreeParamMap: free = gamma'_focal + all D^2 Aod entries; fixed = mu, sigma, gamma_theta ----
l_full = length(θ0_up)
free_idx = vcat(3 + D, collect(Aod_offset+1:Aod_offset+D^2))
fixed_idx = vcat(1, 2, collect(3:2+D))
fixed_vals = θ0_up[fixed_idx]
m = CS.FreeParamMap(l_full, free_idx, fixed_idx, fixed_vals)
println(">>> n_free = ", CS.n_free(m), " (expect ", 1 + D^2, ")")
@assert CS.n_free(m) == 1 + D^2
@assert CS.round_trip_check(θ0_up, m)

# Aod_theta[o,d] -> position in x_free (position 1 is gamma'_focal; Aod entries follow in the
# SAME column-major order as reshape(theta[Aod_offset+1:...],(D,D)))
Aod_free_pos = [1 + (d - 1) * D + o for o in 1:D, d in 1:D]

τ = γ.τ
q_tilde, N_obs = precompute_q_tilde(τ)

# ---- production bundle: moments! computes K/G from the FULL theta (unchanged contract) ----
obj = PsiObjectiveBundleImplicit(δ = 1.0, find_smallest = true, γ = γ,
    (moments!) = EK_moments_gammanorm_directgp!, moments_jacobian! = error, d = nTotalMoments,
    outer_constr_index = outer_constr_index, inequality_index = inequality_index,
    complement_index = complement_index, l = l_full, U = U, N = AD_PARAMS.Jac_W,
    lower_limit = -50, use_cached_x = true,
    outer_loop_opt = OUTER_OPT, inner_loop_opt = INNER_OPT)
@assert obj.outer_constr_index == obj.d "expected exactly one extra outer-loop moment (gravity)"

# ---- free-only divergence-envelope gradient (Method B, wrapped through FreeParamMap) ----
function make_div_grad_fn!(obj, m)
    ncon_inner = obj.d - obj.outer_constr_index + 2
    cfg_cache = Ref{Any}(nothing)
    return function (g_free, x_free, θ_full, inner_x)
        # populate obj.arg1 at the just-solved inner point (envelope theorem: hold lambda, arg1 fixed)
        obj(inner_x, Float64[], Float64[]; constr = zeros(ncon_inner))
        λ = @view inner_x[2:end]
        ctx = (U = obj.U, γobj = obj.γ, λ = λ, arg1 = obj.arg1, d = obj.d, outer_constr_index = obj.outer_constr_index)
        f = x -> envelope_scalar_div_ctx(reconstruct_full(x, m), ctx)
        if cfg_cache[] === nothing
            cfg_cache[] = ForwardDiff.GradientConfig(f, x_free)
        end
        ForwardDiff.gradient!(g_free, f, x_free, cfg_cache[])
        return g_free
    end
end
div_grad_fn! = make_div_grad_fn!(obj, m)

function obj_grad_fn!(g_free, x_free)
    fill!(g_free, 0.0)
    g_free[1] = (-1.0)^obj.find_smallest   # K = x_free[1] (gamma'_focal) directly; objective sign convention
end

function gravity_grad_fn!(g_free, x_free)
    μ = fixed_vals[1]   # mu is fixed_idx[1] by construction above
    gravity_grad_free!(g_free, x_free, D, Aod_free_pos, μ, q_tilde, N_obs)
end

println(">>> Solving (cached, free-only, tariff-gravity) ..."); flush(stdout)
t0 = time()
r_new = CS.outer_loop_cached(obj, m, θ_lo, θ_hi, θ0_up;
    obj_grad_fn! = obj_grad_fn!, div_grad_fn! = div_grad_fn!,
    gravity_grad_fn! = gravity_grad_fn!, has_gravity = true, gravity_value_scale = -1.0 / N_obs,
    use_cache = true, outer_loop_opt = OUTER_OPT)
t_new = time() - t0
CS.summarize(r_new.cache; label = "full-A production (cached, free-only, tariff-gravity)")
γp_new = r_new.θ_min_full[3+D]
κ_new = 1 - γp_new^(σ / (σ - 1))
println("NEW: status=", r_new.nStatus, " opt_err=", r_new.opt_err, " feas_err=", r_new.feas_err,
        " outer_iters=", r_new.outer_iters, " gamma'_focal=", γp_new, " kappa=", κ_new,
        " wall=", round(t_new, digits = 2), "s")

# independent gravity + divergence feasibility check at the reported solution (not reusing the
# search's own bookkeeping) -- both the NEW tariff formula's value and the OLD production formula's
# value, which should be proportional (gravity_value = -1/N_obs * old_sumGrav)
cHat = γ.cHat; wHat_g = γ.wHat; lambda_g = reshape(γ.P, (D, D))'
function AodPow_from_Aodtheta(Aod_theta, μ)
    Aod = Aod_theta .* cHat .* (((wHat_g .* τ) ./ (wHat_g[1, 1] .* τ[1, :]')) .^ (1 / μ)) .* (lambda_g ./ lambda_g[1, :]')
    return (Aod ./ cHat) .^ (-μ)
end
Aod_theta_sol = reshape(r_new.θ_min_full[Aod_offset+1:Aod_offset+D^2], D, D)
AodPow_sol = AodPow_from_Aodtheta(Aod_theta_sol, r_new.θ_min_full[1])
g_new_at_sol = gravity_value(τ, AodPow_sol, q_tilde, N_obs)
Wτ_sol = withinTransform(τ); WAodPow_sol = withinTransform(AodPow_sol)
sumGrav_old_at_sol = sum(Wτ_sol .* WAodPow_sol)
println("INDEPENDENT CHECK: gravity_value (new formula) at solution = ", g_new_at_sol,
        "   (should be ~0; old-formula sumGrav = ", sumGrav_old_at_sol, ", ratio check: ",
        g_new_at_sol / (-sumGrav_old_at_sol / N_obs), " should be ~1)")
flush(stdout)

# ---- reference: today's best-prior validated path (full-theta ForwardDiff, no cache, mu nominally free-but-bounds-pinned) ----
println(">>> Solving (reference: PsiObjectiveBundleImplicitMethodBFullA, no cache, full-theta AD) ..."); flush(stdout)
ggrav_ref = CS.make_gravity_grad(γ, D)
obj_ref = CS.PsiObjectiveBundleImplicitMethodBFullA(δ = 1.0, find_smallest = true, γ = γ,
    (moments!) = EK_moments_gammanorm_directgp!, moments_jacobian! = error, d = nTotalMoments,
    outer_constr_index = outer_constr_index, inequality_index = inequality_index,
    complement_index = complement_index, l = l_full, U = U, N = AD_PARAMS.Jac_W,
    lower_limit = -50, use_cached_x = true, gravity_grad = ggrav_ref,
    outer_loop_opt = OUTER_OPT, inner_loop_opt = INNER_OPT)
t0 = time()
γp_ref, θ_ref, nStatus_ref, _ = outer_loop(obj_ref, θ_lo, θ_hi, θ0_up)
t_ref = time() - t0
κ_ref = 1 - γp_ref^(σ / (σ - 1))
println("REF: status=", nStatus_ref, " gamma'_focal=", γp_ref, " kappa=", κ_ref, " wall=", round(t_ref, digits = 2), "s")

println("\n================ COMPARISON ================")
println("kappa:            new=", κ_new, "  ref=", κ_ref, "  absdiff=", abs(κ_new - κ_ref))
println("gamma'_focal:     new=", γp_new, "  ref=", γp_ref, "  absdiff=", abs(γp_new - γp_ref))
println("wall time:        new=", round(t_new, digits=2), "s  ref=", round(t_ref, digits=2), "s")
println("unique free-x pts (new): ", length(Set(r.x_hash for r in r_new.cache.trace)))
println("inner solves (new):      ", r_new.cache.n_inner_solve)

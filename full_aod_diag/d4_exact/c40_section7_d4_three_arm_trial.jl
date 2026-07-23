# Task brief Section 7: matched D=4 three-arm scientific trial (CM only / CM+mean / CM+mean+ZC)
# at delta in {0.1, 1.0}, matched economic starts, cold verification of every reported
# incumbent. NOT a claim of global convergence -- reports best VERIFIED FEASIBLE incumbents only.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "mean_zero_cov_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_lfix_aware.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production_bundle.jl"))
include(joinpath(@__DIR__, "cm_meanzc_config.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "c40_meanzc_outer_driver.jl"))
using Printf, LinearAlgebra, Random, Dates, DelimitedFiles

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
pe = build_pivot_elimination(ctx)
const L = 10

x_free_calib = ctx.θ0_up[ctx.free_idx]
D = ctx.D
z0 = log.(reshape(x_free_calib[2:end], D, D))
w0_econ = vcat(x_free_calib[1], pivot_reduce(z0, pe))   # matched economic start, every arm

nu_lo, nu_hi = nu_feasible_interval(ctx.U)
eta_nu_lo, eta_nu_hi = log(nu_lo), log(nu_hi)
println("nu feasible interval: [$nu_lo, $nu_hi]")

"""
Bounded local profile of Delta_dual over eta_nu at a FIXED economic point (used both to
initialize eta_nu0 and to report the pre-outer-loop floor). Default lo/hi is a MODERATE band
around eta_nu=0 (nu=1, the Exp(1) draws' theoretical mean), not the full hard necessary interval
(nu_feasible_interval) -- that wide interval is a NECESSARY, not sufficient, feasibility
condition (Section 3), and its extreme endpoints routinely make the inner CC dual solve
genuinely fail (confirmed directly: nu=0.5 already fails at this D=4 config, see Section 6.3
gate log), so profiling the full interval wastes every grid point outside a much narrower band
where the reweighting stays numerically sane. The full interval is still reported (printed
above) as the Section-3-required feasibility check; it is not the profiling search band.
"""
function local_profile_eta(pcx, x_free0; ngrid = 15, nrefine = 25, lo = -1.2, hi = 1.2)
    aug = pcx.aug
    function safe_eval(η)
        aug.nu_ref[] = exp(η)
        try
            base = archC_base_state(x_free0, pcx.ctx_cm, pcx.cctx)
            return delta_dual_from_base(pcx.ctx_cm.obj, base)
        catch e
            e isa CMExpectedSolveFailure || rethrow()
            return Inf
        end
    end
    ηs = range(lo, hi, length = ngrid)
    vals = [safe_eval(η) for η in ηs]
    all(isinf, vals) && error("local_profile_eta: every grid point infeasible")
    i = argmin(vals); ηb = ηs[i]; vb = vals[i]
    lo2 = max(lo, ηb - step(ηs)); hi2 = min(hi, ηb + step(ηs))
    for η in range(lo2, hi2, length = nrefine)
        v = safe_eval(η)
        if v < vb
            vb = v; ηb = η
        end
    end
    return ηb, vb
end

"""
Cold-verify a reported incumbent: rebuild pcx (fresh Zraw/Zpairraw/CM/obj, no cache reuse),
re-solve at the exact incumbent point, recompute every reported diagnostic from scratch.
"""
function cold_verify_arm(arm::Symbol, w_incumbent::Vector{Float64})
    if arm === :cm_only
        aug2 = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
        ctx_cm2 = merge(ctx, (obj = aug2.obj_cm,))
        xf = x_free_from_w(w_incumbent, pe)
        r = evaluate_fullA(xf, ctx_cm2; use_cache = false, warm = false)
        return (Delta_dual = r.Delta_dual, nu = NaN, max_mean_resid = NaN, max_pair_resid = NaN,
                primal_dual_gap = NaN, mean_m_resid = NaN, gravity_resid = NaN)
    else
        nu_ref2 = Ref(exp(w_incumbent[end]))
        pcx2 = build_cm_meanzc_production_context(ctx, CS; L = L, cm_extension = arm, nu_ref = nu_ref2)
        xf = x_free_from_w(w_incumbent[1:end-1], pe)
        _, base, verify = cm_meanzc_production_value_verified(xf, pcx2)
        ν = nu_ref2[]
        resid_mean = recovered_mean_residuals(base.m_star, pcx2.aug.Zraw, ν)
        resid_pair = pcx2.aug.n_pair > 0 ? recovered_pair_residuals(base.m_star, pcx2.aug.Zpairraw, ν) : Float64[]
        gravity_col = @view CS.select_G_from_H(pcx2.ctx_cm.obj, pcx2.ctx_cm.obj.H)[:, end]
        gravity_resid = dot(base.m_star, gravity_col) / length(base.m_star)
        return (Delta_dual = verify.Delta_dual, nu = ν, max_mean_resid = maximum(abs.(resid_mean)),
                max_pair_resid = isempty(resid_pair) ? NaN : maximum(abs.(resid_pair)),
                primal_dual_gap = verify.primal_dual_gap, mean_m_resid = verify.mean_m_resid,
                gravity_resid = gravity_resid)
    end
end

const ARMS = (:cm_only, :cm_plus_mean, :cm_plus_mean_zero_covariance)
const DELTAS = (0.1, 1.0)
results = NamedTuple[]

# Single-process-per-(arm,delta) mode (ARGS = [arm_string, delta_string]): originally added on a
# (WRONG) hypothesis that the extended arms' "every grid point infeasible" failure was cross-arm
# global-state leakage from the cm_only arm's long outer loop. Root cause was actually a plain
# bug in this file (local_profile_eta was called with the pivot-REDUCED w0_econ vector where it
# needed the NATURAL x_free vector, i.e. x_free_from_w(w0_econ, pe) -- archC_base_state has no
# idea about pivot elimination, so it silently reconstructed garbage theta and every solve failed
# regardless of process state). Confirmed by the fact that even fully isolated fresh processes
# reproduced the identical failure. The per-process split is harmless and kept anyway (cheap,
# and still a reasonable robustness practice), but is NOT load-bearing for correctness.
const ARG_ARM = length(ARGS) >= 1 ? Symbol(ARGS[1]) : nothing
const ARG_DELTA = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : nothing
const ARM_LOOP = ARG_ARM === nothing ? ARMS : (ARG_ARM,)
const DELTA_LOOP = ARG_DELTA === nothing ? DELTAS : (ARG_DELTA,)

for δ in DELTA_LOOP
    println("\n", "="^100)
    println("delta = $δ")
    println("="^100)
    for arm in ARM_LOOP
        println("\n--- arm=$arm  delta=$δ ---")
        if arm === :cm_only
            aug = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
            ctx_cm = merge(ctx, (obj = aug.obj_cm,))
            pcx = (ctx_cm = ctx_cm, aug = aug, bins = cm_bin_indices_for(ctx, aug), cctx = build_cm_bin_ctx(ctx, aug))
            out = run_cm_upper(pcx, ctx, pe, copy(w0_econ); delta = δ, maxtime_real = 90.0, verbose = true)
            if out.best === nothing
                println("  NO verified feasible incumbent found for arm=$arm delta=$δ")
                continue
            end
            cv = cold_verify_arm(arm, out.best.w)
            push!(results, (arm = string(arm), delta = δ, kappa = out.kappa, gp = out.best.gp,
                Delta_dual = cv.Delta_dual, nu = cv.nu, max_mean_resid = cv.max_mean_resid,
                max_pair_resid = cv.max_pair_resid, primal_dual_gap = cv.primal_dual_gap,
                mean_m_resid = cv.mean_m_resid, gravity_resid = cv.gravity_resid,
                n_eval = out.n_eval, wall = out.wall, knitro_status = out.knitro_status))
            @printf "  kappa=%.6f  gp=%.6f  Delta_dual(cold)=%.8f  n_eval=%d  wall=%.1fs\n" out.kappa out.best.gp cv.Delta_dual out.n_eval out.wall
        else
            nu_ref = Ref(1.0)
            pcx = build_cm_meanzc_production_context(ctx, CS; L = L, cm_extension = arm, nu_ref = nu_ref)
            η0, _ = local_profile_eta(pcx, x_free_from_w(w0_econ, pe))
            nu_ref[] = exp(η0)
            @printf "  local eta_nu profile at matched start: eta0=%.4f (nu0=%.4f)\n" η0 exp(η0)
            w0 = vcat(w0_econ, η0)
            out = run_meanzc_upper(pcx, ctx, pe, copy(w0); delta = δ, maxtime_real = 90.0,
                eta_nu_lo = eta_nu_lo, eta_nu_hi = eta_nu_hi, verbose = true)
            if out.best === nothing
                println("  NO verified feasible incumbent found for arm=$arm delta=$δ")
                continue
            end
            cv = cold_verify_arm(arm, out.best.w)
            push!(results, (arm = string(arm), delta = δ, kappa = out.kappa, gp = out.best.gp,
                Delta_dual = cv.Delta_dual, nu = cv.nu, max_mean_resid = cv.max_mean_resid,
                max_pair_resid = cv.max_pair_resid, primal_dual_gap = cv.primal_dual_gap,
                mean_m_resid = cv.mean_m_resid, gravity_resid = cv.gravity_resid,
                n_eval = out.n_eval, wall = out.wall, knitro_status = out.knitro_status))
            @printf "  kappa=%.6f  gp=%.6f  nu*=%.6f  Delta_dual(cold)=%.8f  n_eval=%d  wall=%.1fs\n" out.kappa out.best.gp cv.nu cv.Delta_dual out.n_eval out.wall
        end
    end
end

out_dir = joinpath(@__DIR__, "..", "..", "results", "experiment_cm_pairwise_zero_cov")
mkpath(out_dir)
csv_path = joinpath(out_dir, "section7_d4_three_arm_trial.csv")
cols = [:arm, :delta, :kappa, :gp, :Delta_dual, :nu, :max_mean_resid, :max_pair_resid,
        :primal_dual_gap, :mean_m_resid, :gravity_resid, :n_eval, :wall, :knitro_status]

if ARG_ARM === nothing
    # all-combos-in-one-process mode: full decomposition + fresh CSV
    println("\n", "="^100)
    println("SECTION 7 DECOMPOSITION")
    println("="^100)
    for δ in DELTAS
        sub = filter(r -> r.delta == δ, results)
        isempty(sub) && continue
        k = Dict(r.arm => r.kappa for r in sub)
        if haskey(k, "cm_only") && haskey(k, "cm_plus_mean") && haskey(k, "cm_plus_mean_zero_covariance")
            mean_effect = k["cm_plus_mean"] - k["cm_only"]
            dep_effect = k["cm_plus_mean_zero_covariance"] - k["cm_plus_mean"]
            total_effect = k["cm_plus_mean_zero_covariance"] - k["cm_only"]
            @printf "delta=%.1f: exact-mean effect (kappa) = %.6e   dependence effect (kappa) = %.6e   total effect (kappa) = %.6e\n" δ mean_effect dep_effect total_effect
        else
            println("delta=$δ: incomplete arm set (some arm found no verified incumbent) -- decomposition skipped")
        end
    end
    open(csv_path, "w") do io
        println(io, join(String.(cols), ","))
        for r in results
            println(io, join([string(getfield(r, c)) for c in cols], ","))
        end
    end
    println("\nWrote $csv_path")
else
    # single-(arm,delta)-per-process mode: append this run's rows only; the driver shell loop
    # writes the header once (via --init) before the first real invocation.
    open(csv_path, "a") do io
        for r in results
            println(io, join([string(getfield(r, c)) for c in cols], ","))
        end
    end
    println("\nAppended ", length(results), " row(s) to $csv_path")
end

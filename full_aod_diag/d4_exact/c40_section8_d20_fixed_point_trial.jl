# Task brief Section 8: real D=20 fixed-point and performance trial. France, W=80,000,
# KNITRO 13.0.1, this file expects to be launched with `julia -t 20` (20 Julia threads) and
# OPENBLAS_NUM_THREADS=1 (avoid BLAS/Julia-thread oversubscription, per this repo's own
# established convention -- OPENBLAS_NUM_THREADS is NOT bounded by JULIA_NUM_THREADS
# automatically). Tests L=10 and L=50 grids before any long trajectory (Section 9 is the
# long-trajectory step, gated on this file's results).
include(joinpath(@__DIR__, "context_real_d20.jl"))   # -> d20_real_setup (includes context.jl internally)
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
include(joinpath(@__DIR__, "cm_outer_driver.jl"))   # -> x_free_from_w
using Printf, LinearAlgebra, Random, Dates, Serialization

function vmrss_kb()
    for line in eachline("/proc/self/status")
        startswith(line, "VmRSS:") && return parse(Int, split(line)[2])
    end
    return -1
end
gb(kb) = round(kb / 1e6, digits = 3)

println("Section 8 start: ", now(), "  Threads.nthreads()=", Threads.nthreads()); flush(stdout)
println("VmRSS at start = ", gb(vmrss_kb()), " GB"); flush(stdout)

t0 = time()
ctx = d20_real_setup(W = 80000, δ = 1.0)
println("d20_real_setup(W=80000) wall = ", round(time() - t0, digits = 1), "s"); flush(stdout)
println("VmRSS after context = ", gb(vmrss_kb()), " GB"); flush(stdout)

x_free_calib = ctx.θ0_up[ctx.free_idx]
nu_lo, nu_hi = nu_feasible_interval(ctx.U)
println("nu feasible interval: [$nu_lo, $nu_hi]"); flush(stdout)

pe = build_pivot_elimination(ctx)
D = ctx.D
z0 = log.(reshape(x_free_calib[2:end], D, D))
w0_calib = vcat(x_free_calib[1], pivot_reduce(z0, pe))

# Second test point: the existing (schema-1-vintage, UNTRUSTED best_Delta) L=50 CM continuation
# incumbent's best_w -- cold-reverified here under the corrected canonical Delta_dual, not
# trusted as-is (its own stored best_Delta/kappa predate the ab1c74f fix). Falls back to just
# the calibration point if the artifact is absent/incompatible (w = [gp; zfree], length must
# match w0_calib's own pivot-reduced dimension, NOT the natural x_free length).
function load_existing_incumbent_point()
    path = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c13_d20_cm_continuation", "stage_L50_latest.jls")
    isfile(path) || return nothing
    d = deserialize(path)
    length(d.best_w) == length(w0_calib) || return nothing
    return d.best_w
end
existing_w = load_existing_incumbent_point()
POINTS = existing_w === nothing ? [("calibration", nothing)] :
                                   [("calibration", nothing), ("existing_L50_incumbent", existing_w)]
println("Test points: ", [p[1] for p in POINTS]); flush(stdout)

const ARMS = (:cm_only, :cm_plus_mean, :cm_plus_mean_zero_covariance)
const LS = (10, 50)
results = NamedTuple[]

"Local eta_nu profile (moderate band around eta=0, i.e. nu near the Exp(1) theoretical mean) -- same rationale as the D=4 trial's local_profile_eta."
function local_profile_eta_d20(pcx, x_free0; ngrid = 5, nrefine = 4, lo = -1.0, hi = 1.0)
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
    all(isinf, vals) && return (nothing, Inf)
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

for L in LS
    println("\n", "="^100, "\nL=$L\n", "="^100); flush(stdout)
    for (pname, w_override) in POINTS
        w0 = w_override === nothing ? w0_calib : w_override
        xf = x_free_from_w(w0, pe)
        println("\n--- point=$pname L=$L ---"); flush(stdout)

        # ---- CM only ----
        t0 = time()
        aug_cm = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
        ctx_cm_plain = merge(ctx, (obj = aug_cm.obj_cm,))
        t_build_cm = time() - t0
        t0 = time()
        r_cm = evaluate_fullA(xf, ctx_cm_plain; use_cache = false, warm = false)
        t_solve_cm = time() - t0
        Δ_cm = r_cm.Delta_dual   # canonical (NOT -r_cm.zeta, which is the F1-buggy proxy that
                                  # silently omits mean(Psi(q*)) and overstates Delta at tail-active
                                  # points -- exactly the bug ab1c74f fixed elsewhere in this repo;
                                  # evaluate_fullA already returns the canonical field, use it directly
        rss_after_cm = vmrss_kb()
        @printf "  CM only:  Delta=%.8f  inner_status=%d  build=%.2fs  solve=%.2fs  VmRSS=%.3fGB\n" Δ_cm r_cm.inner_status t_build_cm t_solve_cm gb(rss_after_cm)
        push!(results, (arm = "cm_only", L = L, point = pname, basis = "n/a", Delta_dual = Δ_cm,
            nu = NaN, mean_incr = NaN, zc_incr = NaN, max_mean_resid = NaN, max_pair_resid = NaN,
            max_abs_offdiag_cov = NaN, cond_H = NaN, t_build = t_build_cm, t_solve = t_solve_cm,
            vmrss_gb = gb(rss_after_cm), inner_status = r_cm.inner_status))

        Δ_prev = Δ_cm
        for arm in (:cm_plus_mean, :cm_plus_mean_zero_covariance)
            η_shared = nothing   # ν* found by the (cheaper, single) :direct profile, reused for :anchored's
                                  # conditioning-only comparison solve -- both bases share the SAME feasible
                                  # set (math note Section 3), so re-profiling anchored from scratch would
                                  # only re-find the same nu* at 2x the cost, not a different optimum.
            for basis in (:direct, :anchored)
                nu_ref = Ref(1.0)
                t0 = time()
                pcx = build_cm_meanzc_production_context(ctx, CS; L = L, cm_extension = arm, meanzc_basis = basis, nu_ref = nu_ref)
                t_build = time() - t0
                mean_only = arm === :cm_plus_mean
                @printf "  [%s/%s] pair matrix allocated: %s (expected %s)\n" arm basis (pcx.aug.Zpairraw !== nothing) (!mean_only)

                local η_best, t_profile
                if basis === :direct
                    t0 = time()
                    η_best, _ = local_profile_eta_d20(pcx, xf)
                    t_profile = time() - t0
                    if η_best === nothing
                        println("    [$arm/$basis] profile FAILED (every grid point infeasible) -- skipping")
                        continue
                    end
                    η_shared = η_best
                else
                    η_shared === nothing && (println("    [$arm/$basis] skipped: :direct profile did not find a usable nu*"); continue)
                    η_best = η_shared
                    t_profile = 0.0
                end
                nu_ref[] = exp(η_best)

                t0 = time()
                _, base, verify = cm_meanzc_production_value_verified(xf, pcx)
                t_solve = time() - t0
                Δ_floor = verify.Delta_dual
                rss_after = vmrss_kb()

                # Conditioning comparison (task Section 2/8: direct vs anchored mean basis) --
                # reuses the SAME structured-Hessian machinery the production inner solve already
                # uses (cheap; NOT the dense Architecture-A reference, which would be
                # (NCORE_ext+ncm)^2-sized and is exactly the "too slow at D=20" risk the brief
                # flags -- Architecture C avoids materializing that cost).
                n_hess = pcx.ctx_cm.obj.outer_constr_index
                inner_x = vcat(base.ζstar, base.λstar)
                hpacked = Vector{Float64}(undef, n_hess * (n_hess + 1) ÷ 2)
                _archC_prep_for_hessian!(pcx.ctx_cm.obj, inner_x)
                hessian_cm_structured!(hpacked, pcx.ctx_cm.obj, pcx.cctx)
                Hdense = Matrix{Float64}(undef, n_hess, n_hess)
                k = 1
                for i in 1:n_hess, j in i:n_hess
                    Hdense[i, j] = hpacked[k]; Hdense[j, i] = hpacked[k]
                    k += 1
                end
                cond_H = cond(Hdense)
                @printf "    [%s/%s] structured-Hessian cond number = %.4e  (n=%d)\n" arm basis cond_H n_hess

                resid_mean = recovered_mean_residuals(base.m_star, pcx.aug.Zraw, nu_ref[])
                max_mean_resid = maximum(abs.(resid_mean))
                max_pair_resid = NaN
                max_offdiag = NaN
                if pcx.aug.n_pair > 0
                    resid_pair = recovered_pair_residuals(base.m_star, pcx.aug.Zpairraw, nu_ref[])
                    max_pair_resid = maximum(abs.(resid_pair))
                    Σ = recovered_covariance_matrix(base.m_star, pcx.aug.Zraw, D)
                    offdiag = [abs(Σ[i, j]) for i in 1:D, j in 1:D if i != j]
                    max_offdiag = maximum(offdiag)
                    diagpos = all(Σ[i, i] > 0 for i in 1:D)
                    corrs = [Σ[i, j] / sqrt(Σ[i, i] * Σ[j, j]) for i in 1:D-1 for j in i+1:D]
                    @printf "    [%s/%s] largest |offdiag cov|=%.4e  positive-variance-diag=%s  max|corr|=%.4e\n" arm basis max_offdiag diagpos maximum(abs.(corrs))
                end
                incr = Δ_floor - Δ_prev
                @printf "  [%s/%s]  nu*=%.6f  Delta_floor=%.8f  incr_vs_prev=%.4e  build=%.2fs  profile=%.2fs  solve=%.2fs  VmRSS=%.3fGB  max_mean_resid=%.3e  max_pair_resid=%.3e\n" arm basis nu_ref[] Δ_floor incr t_build t_profile t_solve gb(rss_after) max_mean_resid max_pair_resid
                push!(results, (arm = string(arm), L = L, point = pname, basis = string(basis),
                    Delta_dual = Δ_floor, nu = nu_ref[], mean_incr = arm === :cm_plus_mean ? incr : NaN,
                    zc_incr = arm === :cm_plus_mean_zero_covariance ? incr : NaN,
                    max_mean_resid = max_mean_resid, max_pair_resid = max_pair_resid,
                    max_abs_offdiag_cov = max_offdiag, cond_H = cond_H, t_build = t_build, t_solve = t_solve,
                    vmrss_gb = gb(rss_after), inner_status = base.inner_status))
                basis === :direct && (Δ_prev = Δ_floor)
            end
        end
    end
end

out_dir = joinpath(@__DIR__, "..", "..", "results", "experiment_cm_pairwise_zero_cov")
mkpath(out_dir)
csv_path = joinpath(out_dir, "section8_d20_fixed_point_trial.csv")
cols = [:arm, :L, :point, :basis, :Delta_dual, :nu, :mean_incr, :zc_incr, :max_mean_resid,
        :max_pair_resid, :max_abs_offdiag_cov, :cond_H, :t_build, :t_solve, :vmrss_gb, :inner_status]
open(csv_path, "w") do io
    println(io, join(String.(cols), ","))
    for r in results
        println(io, join([string(getfield(r, c)) for c in cols], ","))
    end
end
println("\nWrote $csv_path")
println("Section 8 done: ", now())

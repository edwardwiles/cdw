# ============================================================================
# Continuation 9, Phase 3C: D=20/W=80,000 comparison of the three dense-
# Hessian-free inner-solve variants (compressed_inner_alt_solvers.jl) against
# the existing dense-Hessian compressed baseline (compressed_live.jl's
# inner_loop_KNITRO_compressed, hessopt=exact + lazy dense materialization).
#
# Isolates the INNER SOLVE itself (inner_loop_internal_compressed_variant),
# NOT the full evaluate_fullA_fast_compressed pipeline -- per this task's
# framing ("eliminate dense materialization from the inner CC dual solve").
# Post-solve reporting (Delta_dual, moment residuals) is done ENTIRELY from
# the compressed representation too (compressed_cc_value_grad/
# compressed_moment_resid), NOT via materialize_dense_factual! -- a genuine
# additional dense-elimination beyond what compressed_live.jl's own tail does
# today (that tail still materializes dense G for reporting; this benchmark's
# own reporting shows that isn't necessary, though wiring that into
# production evaluate_fullA_fast_compressed is out of this task's scope,
# flagged as a side-finding in the report, not done here).
#
# Metrics per the Phase 3C brief: cold/warm solve time, callback time
# (n_fg/n_hess call counts, standing in for callback-time given this
# benchmark's time budget -- exact callback-only wall time would need finer
# @prof scoping than this script adds), iteration count, peak memory
# (process-level VmHWM, not per-variant -- isolating per-variant peak would
# need separate processes), final duals, primal/dual divergence (Delta_dual),
# moment residuals, robustness across starts (2 extra randomly-perturbed
# cold starts per variant, checked for agreement with the natural-start
# solution).
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))   # -> includes context.jl exactly once
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "compressed_inner_alt_solvers.jl"))
using Statistics, Printf, Dates, Random, LinearAlgebra

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c9_phase3c_d20_bench")
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...)
    println(LOGIO, xs...)
    flush(stdout); flush(LOGIO)
end
logprint("c9_phase3c_d20_bench.jl starting at ", now(), "  commit=", COMMIT, "  nthreads=", Threads.nthreads())

function vmhwm_kb()
    try
        for line in eachline("/proc/self/status")
            startswith(line, "VmHWM:") && return parse(Int, split(line)[2])
        end
    catch
    end
    return -1
end

t0 = time()
ctx = d20_real_setup(W = 80000)
t_setup = time() - t0
D = ctx.D
obj = ctx.obj
logprint("d20_real_setup(W=80000) wall = ", round(t_setup, digits = 2), "s  VmHWM=", round(vmhwm_kb() / 1e6, digits = 2), " GB")

xf_nat = ctx.θ0_up[ctx.free_idx]
θ_full = CS.reconstruct_full(xf_nat, ctx.m)
ncol = obj.outer_constr_index - 1
n_inner = obj.outer_constr_index
logprint("D=", D, "  ncol (inner dual vars, excl zeta)=", ncol, "  n_inner (incl zeta)=", n_inner)

const DEFAULT_OPT = obj.inner_loop_opt
const OPTDIR = joinpath(D4X_ROOT, "full_aod_diag", "d4_exact")

# ---- trusted reference for correctness cross-check: existing dense-Hessian compressed baseline ----
logprint("\n---- reference: existing compressed baseline (hessopt=exact, lazy dense materialization) ----")
obj.inner_loop_opt = DEFAULT_OPT
obj.x .= NaN
t0 = time()
K_ref, x_ref, nStatus_ref, nfg_ref, nhess_ref, st_ref = inner_loop_internal_compressed(obj, θ_full, ctx)
t_ref = time() - t0
f_ref, gz_ref, gl_ref, q_ref, dPsq_ref = compressed_cc_value_grad(x_ref[1], x_ref[2:end], st_ref.cf; Psi! = obj.Psi!, dPsi! = obj.dPsi!)
Delta_dual_ref = -f_ref
logprint(@sprintf("  cold: status=%d n_fg=%d n_hess=%d wall=%.3fs Delta_dual=%.10f",
          nStatus_ref, nfg_ref, nhess_ref, t_ref, Delta_dual_ref))

# ---- variant runner ----
function run_variant(label, opt_path, variant_sym; n_extra_starts::Int = 2, seed::Int = 424242)
    obj.inner_loop_opt = opt_path
    logprint("\n---- ", label, " (opt=", basename(opt_path), ", variant=", variant_sym, ") ----")

    # cold
    obj.x .= NaN
    iters0 = CS.INNER_ITERS_TOTAL[]
    t0 = time()
    K_c, x_c, status_c, nfg_c, nhess_c, st_c = inner_loop_internal_compressed_variant(obj, θ_full, ctx; variant = variant_sym)
    t_cold = time() - t0
    iters_c = CS.INNER_ITERS_TOTAL[] - iters0   # delta of the cumulative counter = this call's own KNITRO iteration count
    f_c, _, _, _, dPsq_c = compressed_cc_value_grad(x_c[1], x_c[2:end], st_c.cf; Psi! = obj.Psi!, dPsi! = obj.dPsi!)
    Delta_dual_c = -f_c
    mresid_c = compressed_moment_resid(st_c.cf, ones(size(obj.U,1))) ./ size(obj.U,1)
    max_mresid_c = maximum(abs.(mresid_c))
    dzeta_c = abs(x_c[1] - x_ref[1]); dlambda_c = maximum(abs.(x_c[2:end] .- x_ref[2:end]))
    logprint(@sprintf("  cold: status=%d n_fg=%d n_hess=%d n_iters=%d wall=%.3fs Delta_dual=%.10f max|benchmark_unweighted_moment_mean|=%.3e |dζ vs ref|=%.3e max|dλ vs ref|=%.3e",
              status_c, nfg_c, nhess_c, iters_c, t_cold, Delta_dual_c, max_mresid_c, dzeta_c, dlambda_c))

    # warm (re-solve from the just-converged point -- obj.x already holds x_c via inner_loop_internal_compressed_variant's own obj.x .= x assignment)
    t0 = time()
    K_w, x_w, status_w, nfg_w, nhess_w, st_w = inner_loop_internal_compressed_variant(obj, θ_full, ctx; variant = variant_sym)
    t_warm = time() - t0
    logprint(@sprintf("  warm: status=%d n_fg=%d n_hess=%d wall=%.3fs", status_w, nfg_w, nhess_w, t_warm))

    # robustness across starts: n_extra_starts randomly-perturbed cold starts
    rng = MersenneTwister(seed)
    robust_rows = NamedTuple[]
    for i in 1:n_extra_starts
        obj.use_cached_x = true
        obj.x = vcat(x_ref[1], x_ref[2:end]) .+ 0.5 .* randn(rng, length(x_ref))
        t0 = time()
        K_r, x_r, status_r, nfg_r, nhess_r, st_r = inner_loop_internal_compressed_variant(obj, θ_full, ctx; variant = variant_sym)
        t_r = time() - t0
        obj.use_cached_x = false
        dz = abs(x_r[1] - x_ref[1]); dl = status_r in (0,-100,-101,-103) ? maximum(abs.(x_r[2:end] .- x_ref[2:end])) : NaN
        logprint(@sprintf("  start %d: status=%d wall=%.3fs |dζ vs ref|=%.3e max|dλ vs ref|=%.3e", i, status_r, t_r, dz, dl))
        push!(robust_rows, (start = i, status = status_r, wall_s = t_r, dzeta = dz, dlambda = dl))
    end

    obj.inner_loop_opt = DEFAULT_OPT
    return (label = label, status_cold = status_c, n_fg_cold = nfg_c, n_hess_cold = nhess_c, n_iters_cold = iters_c, wall_cold_s = t_cold,
            status_warm = status_w, n_fg_warm = nfg_w, n_hess_warm = nhess_w, wall_warm_s = t_warm,
            Delta_dual = Delta_dual_c, max_moment_resid = max_mresid_c,
            dzeta_vs_ref = dzeta_c, dlambda_vs_ref = dlambda_c,
            robust_rows = robust_rows)
end

results = NamedTuple[]
push!(results, run_variant("qn_bfgs", joinpath(OPTDIR, "ek_inner_bfgs.opt"), :qn))
push!(results, run_variant("qn_sr1", joinpath(OPTDIR, "ek_inner_sr1.opt"), :qn))
push!(results, run_variant("qn_lbfgs", joinpath(OPTDIR, "ek_inner_lbfgs.opt"), :qn))
push!(results, run_variant("denseaccum", DEFAULT_OPT, :denseaccum))
push!(results, run_variant("hvp", joinpath(OPTDIR, "ek_inner_hvp.opt"), :hvp))

logprint("\n", "="^100)
logprint("SUMMARY (reference: existing compressed baseline, hessopt=exact + lazy dense materialization)")
logprint(@sprintf("  %-14s %8s %8s %10s | %8s %8s %10s | %14s %12s", "variant", "cold_s", "n_fg_c", "n_hess_c", "warm_s", "n_fg_w", "n_hess_w", "Delta_dual", "max_mresid"))
logprint(@sprintf("  %-14s %8.3f %8d %10d | %8s %8s %10s | %14.10f %12.3e", "REFERENCE(dense)", t_ref, nfg_ref, nhess_ref, "-", "-", "-", Delta_dual_ref, NaN))
for r in results
    logprint(@sprintf("  %-14s %8.3f %8d %10d | %8.3f %8d %10d | %14.10f %12.3e",
              r.label, r.wall_cold_s, r.n_fg_cold, r.n_hess_cold, r.wall_warm_s, r.n_fg_warm, r.n_hess_warm,
              r.Delta_dual, r.max_moment_resid))
end

write_csv_rows(joinpath(OUTDIR, "phase3c_summary.csv"),
    vcat([(variant = "reference_dense", wall_cold_s = t_ref, n_fg_cold = nfg_ref, n_hess_cold = nhess_ref,
           wall_warm_s = NaN, n_fg_warm = 0, n_hess_warm = 0, Delta_dual = Delta_dual_ref, max_moment_resid = NaN,
           dzeta_vs_ref = 0.0, dlambda_vs_ref = 0.0)],
          [(variant = r.label, wall_cold_s = r.wall_cold_s, n_fg_cold = r.n_fg_cold, n_hess_cold = r.n_hess_cold,
            wall_warm_s = r.wall_warm_s, n_fg_warm = r.n_fg_warm, n_hess_warm = r.n_hess_warm,
            Delta_dual = r.Delta_dual, max_moment_resid = r.max_moment_resid,
            dzeta_vs_ref = r.dzeta_vs_ref, dlambda_vs_ref = r.dlambda_vs_ref) for r in results]))

write_csv_rows(joinpath(OUTDIR, "phase3c_robustness.csv"),
    [(variant = r.label, start = row.start, status = row.status, wall_s = row.wall_s,
      dzeta = row.dzeta, dlambda = row.dlambda) for r in results for row in r.robust_rows])

logprint("\nVmHWM at end of run = ", round(vmhwm_kb() / 1e6, digits = 2), " GB")
logprint("\nc9_phase3c_d20_bench.jl COMPLETE at ", now())
close(LOGIO)

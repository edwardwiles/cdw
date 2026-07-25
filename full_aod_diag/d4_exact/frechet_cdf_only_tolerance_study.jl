# Trial-tolerance study (task brief §8), CDF-only fixed Fréchet, D=20/W=80,000/L=50/:exclude_row.
# At P1 (cold) and P2 (warm-from-P1), compares Delta*, dual objective, max moment/KKT residual, and
# wall time across opttol/opttol_abs/ftol in {1e-6,1e-8,1e-10,1e-12} (feastol left at its production
# 1e-12 -- only optimality tolerance is loosened, feasibility is not).
#
# Usage: JULIA_NUM_THREADS=<N> julia --project=. frechet_cdf_only_tolerance_study.jl <base_nt>
# <base_nt> selects which frechet_bench_opts/tol_variants/ek_inner_nt<base_nt>_tol*.opt family to use.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "draw_design.jl"))
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
include(joinpath(@__DIR__, "cm_hessian_threaded.jl"))
include(joinpath(@__DIR__, "cm_screen_bridge.jl"))
include(joinpath(@__DIR__, "knitro_status.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "cm_frechet_config.jl"))
include(joinpath(@__DIR__, "frechet_reference_targets.jl"))
include(joinpath(@__DIR__, "cm_frechet_moments.jl"))
include(joinpath(@__DIR__, "cm_frechet_hessian.jl"))
include(joinpath(@__DIR__, "cm_frechet_hessian_threaded.jl"))
include(joinpath(@__DIR__, "cm_frechet_bases.jl"))
include(joinpath(@__DIR__, "cm_frechet_power_hessian_structured.jl"))
include(joinpath(@__DIR__, "cm_frechet_production_bundle.jl"))
include(joinpath(@__DIR__, "cm_frechet_production_bundle_threaded.jl"))
include(joinpath(@__DIR__, "cm_frechet_lfix_aware.jl"))
include(joinpath(@__DIR__, "cm_frechet_cdf_only_gradient.jl"))
using Printf, LinearAlgebra, Dates

base_nt = length(ARGS) >= 1 ? ARGS[1] : "1"
const OUTDIR = joinpath(@__DIR__, "..", "..", "results", "fixed_frechet_cdf_only_bench_2026-07-24")
mkpath(OUTDIR)
const LOGIO = open(joinpath(OUTDIR, "tolerance_study.txt"), "w")
function lp(xs...)
    println(xs...); println(LOGIO, xs...); flush(stdout); flush(LOGIO)
end
lp("frechet_cdf_only_tolerance_study.jl  base_nt=$base_nt  nthreads=$(Threads.nthreads())  started=$(now())")

t0 = time()
const W = 80_000; const L = 50; const DELTA = 1.0
ctx = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true, needs_outer_moment_jacobian = false,
    destination_sample = :exclude_row)
lp(@sprintf("[%.1fs] ctx built  D=%d D_dest=%d", time()-t0, ctx.D, ctx.D_dest))
pe = build_pivot_elimination(ctx)
cfg = CMFrechetConfig(cm = CMConfig(cm_grid_size = L, contrasts = :orthonormal, cm_hessian_backend = :structured),
    marginal_mode = :frechet_reference, frechet_feature_set = :cdf_only, frechet_basis = :cumulative)
fpcx = build_cm_frechet_production_context(ctx, CS, cfg; L = L)
tls = build_thread_local_scratch(fpcx.fctx.cctx)
obj = fpcx.ctx_cm.obj
lp(@sprintf("[%.1fs] fpcx built", time()-t0))

x_free_calib = ctx.θ0_up[ctx.free_idx]
z_star = log.(reshape(x_free_calib[2:end], ctx.D, ctx.D_dest))
zfree_star = pivot_reduce(z_star, pe)
σ = ctx.σ
κ_star = 1 - x_free_calib[1]^(σ / (σ - 1))
gp_P1 = (1 - (κ_star + 1e-4))^((σ - 1) / σ)
gp_P2 = (1 - (κ_star + 2e-4))^((σ - 1) / σ)
x_free_P1 = vcat(gp_P1, vec(exp.(pivot_expand(zfree_star, pe))))
x_free_P2 = vcat(gp_P2, vec(exp.(pivot_expand(zfree_star, pe))))
θ_P1 = CS.reconstruct_full(x_free_P1, fpcx.ctx_cm.m)
θ_P2 = CS.reconstruct_full(x_free_P2, fpcx.ctx_cm.m)

hcb_thread(_o) = archC_frechet_hess_cb_builder_v2(fpcx.fctx; threaded_bins = true, tls = tls, use_syrk = true)

function verify_at(θ, x_free, name, opt_file; force_cold)
    obj.inner_loop_opt = opt_file
    force_cold && (obj.x .= NaN)
    t = @elapsed begin
        K, x_sol, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ; hess_cb_builder = hcb_thread)
    end
    base = BaseDualState(collect(x_free), θ, x_sol[1], collect(x_sol[2:end]), copy(obj.arg1), nStatus)
    W_ = size(obj.U, 1)
    G = CS.select_G_from_H(obj, obj.H)
    ncon = obj.d - obj.outer_constr_index + 2
    cbuf = zeros(ncon)
    obj(x_sol, constr = @view(cbuf[1:ncon]))
    Delta_dual = cbuf[1] / 1e10
    m_weights = copy(obj.arg1)
    Delta_primal = primal_divergence(m_weights)
    nkkt = min(length(base.λstar), size(G, 2))
    max_kkt = kkt_residual_blas(G, m_weights, nkkt, W_)
    lp(@sprintf("%-10s wall=%7.2fs nStatus=%-4d Delta_dual=%.10f Delta_primal=%.10f |gap|=%.3e max_kkt=%.3e",
        name, t, nStatus, Delta_dual, Delta_primal, abs(Delta_dual-Delta_primal), max_kkt))
    return (name=name, wall=t, nStatus=nStatus, Delta_dual=Delta_dual, Delta_primal=Delta_primal, max_kkt=max_kkt)
end

for k in (6, 8, 10, 12)
    opt_file = joinpath(@__DIR__, "frechet_bench_opts", "tol_variants", "ek_inner_nt$(base_nt)_tol$(k).opt")
    isfile(opt_file) || (lp("MISSING $opt_file, skipping"); continue)
    lp(""); lp("-"^100); lp("tolerance=1e-$k  ($opt_file)"); lp("-"^100)
    verify_at(θ_P1, x_free_P1, "P1(cold)", opt_file; force_cold = true)
    verify_at(θ_P2, x_free_P2, "P2(warm)", opt_file; force_cold = false)
end

lp(""); lp("SUMMARY total_wall=$(round(time()-t0,digits=1))s  finished=$(now())")
close(LOGIO)

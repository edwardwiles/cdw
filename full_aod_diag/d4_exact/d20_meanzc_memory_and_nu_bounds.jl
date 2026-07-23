# Step 5 (memory design) + Step 6 (nu bounds / wide-grid profiling) at real D=20/W=80,000/L=50.
# Deliberately SCOPED, not a full campaign: ONE cold inner solve for memory measurement, plus a
# small (7-point) eta_nu grid at that same point to check the default box is not binding. Uses
# the SAME production draw convention (W=80000, draw_seed=20260719, :pseudorandom,
# contrasts=:orthonormal) as the actual 2026-07-22 CM campaign for a fair comparison.
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
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
include(joinpath(@__DIR__, "cm_meanzc_config.jl"))
using Printf, Dates

lp(xs...) = (println(xs...); flush(stdout))

function vmrss_kb()
    for line in eachline("/proc/self/status")
        startswith(line, "VmRSS:") && return parse(Int, split(line)[2])
    end
    return -1
end
function vmhwm_kb()
    for line in eachline("/proc/self/status")
        startswith(line, "VmHWM:") && return parse(Int, split(line)[2])
    end
    return -1
end

lp(">>> d20_meanzc_memory_and_nu_bounds starting at ", now(), "  Threads.nthreads()=", Threads.nthreads())
lp(">>> VmRSS at start = ", vmrss_kb()/1e6, " GB")

const W = 80_000
const L = 50
const K_mean, K_pair = 1, 1
const DRAW_SEED = 20260719
const CONTRASTS = :orthonormal   # matches the approved 2026-07-22 production contrast-basis decision

t_ctx0 = time()
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :pseudorandom, draw_seed = DRAW_SEED)
lp(">>> ctx built in ", round(time()-t_ctx0, digits=1), "s   VmRSS = ", vmrss_kb()/1e6, " GB")

snaps = nested_grid_sequence([10, 20, 50])
probs = snaps[L]

t_pcx0 = time()
pcx = build_cm_meanzc_production_context(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = CONTRASTS, meanzc_basis = :direct, probs = probs)
lp(">>> pcx built in ", round(time()-t_pcx0, digits=1), "s   VmRSS = ", vmrss_kb()/1e6, " GB")
lp(">>> NCORE_ext (widened economic block) = ", pcx.cctx.NCORE, "   ncm = ", pcx.cctx.ncm, "   n = NCORE_ext+ncm = ", pcx.cctx.NCORE + pcx.cctx.ncm)

# --- memory decomposition of the theta-independent precomputed objects ---
sz_Zraw = Base.summarysize(pcx.aug.Zraw_all[1]) / 1e6
sz_Zpair = Base.summarysize(pcx.aug.Zpairraw_all[1]) / 1e6
sz_CM = Base.summarysize(pcx.aug.CM) / 1e6
sz_Bidx = Base.summarysize(pcx.cctx.Bidx) / 1e6
sz_Hfull = Base.summarysize(pcx.cctx.Hfull) / 1e6
sz_H = Base.summarysize(pcx.ctx_cm.obj.H) / 1e6
lp(">>> memory decomposition (MB): Zraw=", round(sz_Zraw,digits=1), " Zpairraw=", round(sz_Zpair,digits=1),
   " CM(dense reference)=", round(sz_CM,digits=1), " Bidx=", round(sz_Bidx,digits=1),
   " Hfull(scratch)=", round(sz_Hfull,digits=1), " obj.H=", round(sz_H,digits=1))

# calibration start point
x_free_calib = ctx.θ0_up[ctx.free_idx]
ν0 = [1.0]

GC.gc()
lp(">>> VmRSS before cold solve = ", vmrss_kb()/1e6, " GB")
t0 = time()
base0, verify0 = archC_meanzc_verified_state(x_free_calib, ν0, pcx.ctx_cm, pcx.cctx)
t_solve = time() - t0
lp(">>> cold solve done in ", round(t_solve, digits=1), "s  Delta_dual=", verify0.Delta_dual,
   "  verified=", is_verified_success(verify0), "  inner_status=", verify0.inner_status)
lp(">>> VmRSS after cold solve = ", vmrss_kb()/1e6, " GB   VmHWM (peak so far) = ", vmhwm_kb()/1e6, " GB")

# gradient too (touches composite_gradient_at_fast's own allocations)
pe = build_pivot_elimination(ctx)
t1 = time()
g_ext, meta = cm_meanzc_production_gradient(x_free_calib, ν0, pcx, ctx, pe; base = base0, verify = verify0)
t_grad = time() - t1
lp(">>> gradient done in ", round(t_grad, digits=1), "s  length(g_ext)=", length(g_ext))
lp(">>> VmRSS after gradient = ", vmrss_kb()/1e6, " GB   VmHWM (peak so far) = ", vmhwm_kb()/1e6, " GB")

println()
lp("="^100)
lp("Step 6: eta_nu profiling at this D=20/L=50 point")
lp("="^100)
lp(">>> FIRST PASS (superseded, kept for record): a coarse eta_nu in [log(0.3),log(3.5)], 7 points,")
lp(">>> found only 1/7 feasible (the actual KNITRO-feasible neighborhood of nu at this joint (g,A) point")
lp(">>> is much narrower than the hard finite-support interval alone would suggest) -- re-profiling with")
lp(">>> a finer grid concentrated near the natural anchor (nu=1) instead.")
η_grid = range(log(0.6), log(1.8), length = 9)   # finer, still several-fold wider than the eventual optimum region
results = NamedTuple[]
for η in η_grid
    ν = exp(η)
    try
        t = time()
        base, verify = archC_meanzc_verified_state(x_free_calib, [ν], pcx.ctx_cm, pcx.cctx)
        push!(results, (η = η, ν = ν, Delta = verify.Delta_dual, verified = is_verified_success(verify), wall = time()-t))
        lp("  eta_nu=", round(η,digits=3), " (nu=", round(ν,digits=3), ")  Delta_dual=", verify.Delta_dual,
           "  verified=", is_verified_success(verify), "  wall=", round(time()-t,digits=1), "s")
    catch e
        e isa CMExpectedSolveFailure || rethrow()
        lp("  eta_nu=", round(η,digits=3), " (nu=", round(ν,digits=3), ")  INFEASIBLE/failed inner solve")
    end
end
verified_results = filter(r -> r.verified, results)
if !isempty(verified_results)
    best = verified_results[argmin([r.Delta for r in verified_results])]
    lp(">>> best over grid: eta_nu=", round(best.η,digits=3), " (nu=", round(best.ν,digits=3), ")  Delta_dual=", best.Delta)
    interior = best.η > minimum(η_grid) + 1e-6 && best.η < maximum(η_grid) - 1e-6
    lp(">>> optimum interior to the profiled grid (not at an edge): ", interior)
end
lp(">>> default production nu_bounds (eta_nu-space, +/-4x hard finite-support interval): ", meanzc_default_nu_bounds(ctx, 1))

lp(">>> VmRSS at end = ", vmrss_kb()/1e6, " GB   VmHWM (overall peak) = ", vmhwm_kb()/1e6, " GB")
lp(">>> d20_meanzc_memory_and_nu_bounds DONE at ", now())

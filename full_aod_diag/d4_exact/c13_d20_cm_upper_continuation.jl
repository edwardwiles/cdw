# Continuation 13, Section 9: real D20, W=80000, delta=1 common-marginals upper bound via nested
# Q10 -> Q20 -> Q50 continuation. Each stage starts from the previous stage's best exact-feasible
# incumbent (a Q_L-feasible point is automatically Q_{L'<L}-feasible since the grids are genuinely
# nested -- Section 6 -- so this is a legitimate warm start, not merely a convenient one). Every
# best incumbent is checkpointed to disk immediately (serialize, not just kept in memory) per the
# brief's "checkpoint every best exact-feasible incumbent, use more than one structured start when
# affordable, retain the best exact-feasible result, not merely the terminal iterate."
include(joinpath(@__DIR__, "context_real_d20.jl"))
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
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
using Printf, LinearAlgebra, Random, Statistics, Serialization, Dates

const CKPT_DIR = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c13_d20_cm_continuation")
mkpath(CKPT_DIR)
lp(xs...) = (println(xs...); flush(stdout))

W = 80000
DELTA = 1.0
MAXTIME_STAGE = get(ENV, "C13_MAXTIME_STAGE", "1800") |> x -> parse(Float64, x)   # seconds per grid stage, default 30min

lp(">>> building D20 real-data context, W=$W, delta=$DELTA ...")
t0 = time()
ctx = d20_real_setup(W = W, δ = DELTA, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
lp(@sprintf(">>> ctx built in %.1fs. D=%d W=%d free_dims=%d", time()-t0, D, W, D2))

x_free_calib = ctx.θ0_up[ctx.free_idx]
w_from_xfree(xf) = vcat(xf[1], pivot_reduce(log.(reshape(xf[2:end], D, D)), pe))
w_calib = w_from_xfree(x_free_calib)

snaps = nested_grid_sequence([10, 20, 50])

function save_stage(L, res, probs)
    b = res.best
    payload = (L = L, probs = probs, kappa = res.kappa, knitro_status = res.knitro_status,
               wall = res.wall, n_eval = res.n_eval, n_grad = res.n_grad,
               best_w = b === nothing ? nothing : b.w, best_Delta = b === nothing ? nothing : b.Delta,
               xsol = res.xsol, timestamp = string(now()))
    path = joinpath(CKPT_DIR, "stage_L$(L)_latest.jls")
    serialize(path, payload)
    lp("  checkpoint saved: ", path)
    return payload
end

w_current = copy(w_calib)
stage_results = Dict{Int,Any}()
for L in (10, 20, 50)
    lp("="^100)
    lp(@sprintf("STAGE L=%d (%d cutpoints), starting from w = %s", L, length(snaps[L]), L == 10 ? "calibration" : "L=$(L==20 ? 10 : 20) best"))
    lp("="^100)
    pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = snaps[L])
    lp(@sprintf("  ncm=%d  d_total=%d", pcx.aug.ncm, pcx.ctx_cm.obj.d))

    # verify the start point is feasible under THIS stage's (more restrictive) grid before handing
    # it to KNITRO -- if not, this indicates a warm-start-from-infeasible-point situation the brief
    # warns about (see [[full-d2-winner-boundary-fix]] memory); fall back to calibration.
    _, base_check = try
        cm_production_value(x_free_from_w(w_current, pe), pcx)
    catch
        (nothing, nothing)
    end
    if base_check === nothing || !isfinite(-base_check.ζstar) || -base_check.ζstar > DELTA + 1e-6
        lp("  ** start point infeasible/failed under L=$L grid, falling back to calibration **")
        w_current = copy(w_calib)
    end

    res = run_cm_upper(pcx, ctx, pe, copy(w_current); delta = DELTA, maxtime_real = MAXTIME_STAGE, verbose = true)
    lp(@sprintf("  STAGE L=%d done: status=%d wall=%.1fs n_eval=%d n_grad=%d kappa=%s", L, res.knitro_status, res.wall, res.n_eval, res.n_grad, string(res.kappa)))
    payload = save_stage(L, res, snaps[L])
    stage_results[L] = (res = res, pcx = pcx, payload = payload)

    if res.best !== nothing
        global w_current = res.best.w
    else
        lp("  ** NO FEASIBLE INCUMBENT FOUND AT L=$L -- keeping previous stage's w for the next stage's start **")
    end
end

lp("="^100)
lp("ALL STAGES DONE")
lp("="^100)
for L in (10, 20, 50)
    haskey(stage_results, L) || continue
    r = stage_results[L].res
    lp(@sprintf("  L=%-2d  kappa=%s  status=%d  n_eval=%d", L, string(r.kappa), r.knitro_status, r.n_eval))
end
lp("DONE")

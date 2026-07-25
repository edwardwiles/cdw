# ============================================================================
# Unrestricted BLAS-thread sweep across P0/P1/P2 (allocation/Hessian port task §5), filling the
# gap the source port branch's own blas_sweep_u_2026-07-25.jl disclosed (P0/calibration only).
#
# P0 = genuine calibrated start (ctx.θ0_up).
# P1/P2 = real points harvested from a single longer diagnostic trajectory FROM P0
#         (full_trace_ref), not separately hand-constructed -- P1 is an early/easy accepted
#         point (near-optimal, small Delta), P2 is the LAST accepted point of that trajectory
#         (deepest/hardest reached). Saved to a .jls fixture (blas_sweep_u_p1p2_points.jls) so
#         later phases (matched 300s runs, correctness gates) can reuse the exact same points.
#
# For each of P0/P1/P2 x BLAS in [1,4,8,10,20]: a fixed-budget run_polish_checkpointed call
# WARM-STARTED from that point (same mechanism/discipline as the source branch's P0-only sweep),
# same explicit outer algorithm (pin_outer_algorithm=true) for a controlled comparison.
#
# Usage: JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#          -t 20 full_aod_diag/d4_exact/blas_sweep_u_p0p1p2_2026-07-25.jl <outdir>
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Dates, Serialization, Printf, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))
const OUTDIR = abspath(ARGS[1])
mkpath(OUTDIR)
const HARVEST_BUDGET_S = parse(Float64, get(ENV, "HARVEST_BUDGET_S", "150.0"))
const SWEEP_BUDGET_S = parse(Float64, get(ENV, "SWEEP_BUDGET_S", "30.0"))
const BLAS_SETTINGS = [1, 4, 8, 10, 20]

lp(">>> blas_sweep_u_p0p1p2 starting ", now(), " Julia threads=", Threads.nthreads())

t0 = time()
ctx0 = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
g0 = x_free_calib[1]
zfree0 = pivot_reduce(reshape(log.(x_free_calib[2:end]), ctx0.D, ctx0.D_dest), pe0)
lp(">>> context build wall = ", round(time() - t0, digits = 3), "s")

# ---------------------------------------------------------------------------
# Harvest P1 (early/easy accepted point) and P2 (last/hardest accepted point) from one real
# trajectory starting at P0 -- both genuine production-driver-visited points, not synthetic.
# ---------------------------------------------------------------------------
trace_ref = Ref(NamedTuple[])
harvest = run_polish_checkpointed("harvest", true, g0, zfree0; maxtime_real = HARVEST_BUDGET_S,
    W_in = 80_000, delta_in = 1.0, draw_seed_in = 20260719, destination_sample = :exclude_row,
    ckpt_dir = joinpath(OUTDIR, "ckpt_harvest"), checkpoint_interval_s = 9999.0,
    pin_outer_algorithm = true, full_trace_ref = trace_ref)
accepted = filter(r -> r.accepted, trace_ref[])
lp(">>> harvest: n_eval=", harvest.n_eval, " n_accepted=", length(accepted), " wall=", round(time() - t0, digits = 3), "s")
length(accepted) >= 2 || error("blas_sweep_u_p0p1p2: harvest trajectory produced <2 accepted points -- cannot form P1/P2 distinctly, widen HARVEST_BUDGET_S")

# P1: first accepted point after P0 itself (idx==1 is P0's own seed eval in most cases; take the
# next distinct accepted point). P2: last accepted point (deepest reached).
p1_w = accepted[min(2, length(accepted))].w
p2_w = accepted[end].w
D = ctx0.D; Ddest = ctx0.D_dest
points = Dict(
    :P0 => (g = g0, zfree = zfree0),
    :P1 => (g = p1_w[1], zfree = p1_w[2:end]),
    :P2 => (g = p2_w[1], zfree = p2_w[2:end]),
)
serialize(joinpath(OUTDIR, "blas_sweep_u_p1p2_points.jls"), points)
for (k, p) in points
    lp(">>> ", k, ": g=", p.g, " ||zfree||=", norm(p.zfree))
end

# ---------------------------------------------------------------------------
# BLAS sweep at each point.
# ---------------------------------------------------------------------------
results = NamedTuple[]
for pname in (:P0, :P1, :P2), b in BLAS_SETTINGS
    p = points[pname]
    tt0 = time()
    r = run_polish_checkpointed("sweep_$(pname)_blas$(b)", true, p.g, p.zfree; maxtime_real = SWEEP_BUDGET_S,
        W_in = 80_000, delta_in = 1.0, draw_seed_in = 20260719, destination_sample = :exclude_row,
        ckpt_dir = joinpath(OUTDIR, "ckpt_$(pname)_blas$(b)"), checkpoint_interval_s = 9999.0,
        pin_outer_algorithm = true, blas_threads = b)
    wall = time() - tt0
    push!(results, (point = pname, blas_threads = b, wall = wall, n_eval = r.n_eval, n_grad = r.n_grad_calls,
                     best_gp = r.best_feasible === nothing ? NaN : r.best_feasible.gp,
                     best_Delta = r.best_feasible === nothing ? NaN : r.best_feasible.Delta,
                     knitro_status = r.knitro_status))
    lp(">>> ", pname, " BLAS=", b, ": wall=", round(wall, digits = 2), "s n_eval=", r.n_eval,
       " n_grad=", r.n_grad_calls, " best_Delta=", results[end].best_Delta)
end

open(joinpath(OUTDIR, "blas_sweep_u_p0p1p2_2026-07-25.csv"), "w") do io
    println(io, "point,blas_threads,wall_s,n_eval,n_grad,best_gp,best_Delta,knitro_status")
    for r in results
        println(io, r.point, ",", r.blas_threads, ",", round(r.wall, digits = 3), ",", r.n_eval, ",",
                r.n_grad, ",", r.best_gp, ",", r.best_Delta, ",", r.knitro_status)
    end
end
lp(">>> DONE wall_total=", round(time() - t0, digits = 2), "s")

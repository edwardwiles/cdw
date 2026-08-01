# ============================================================================
# Claude Code task 2026-08-01, follow-up (live user request): build shared
# starting points at gains-from-trade (GT) targets of 7% and 8%, for the
# matched full-vs-profiled A/B. GT = kappa = 1 - gp^(sigma/(sigma-1))
# (confirmed formula, c12_sign_convention_smoke_test.jl and others).
#
# A direct jump from calibration to the target gp (holding A at calibration)
# fails to converge well before reaching GT=7-8% (probed: frac=0.982 already
# gives inner_status=-400). This script instead CONTINUES gp down in small
# steps, re-optimizing A at each step via a SHORT run_profile_checkpointed
# call (maxit=10) warm-started from the previous step's converged zfree --
# reusing the trusted production driver at each step rather than writing new
# continuation machinery. Reaches GT=8% (the further target), checkpointing
# the GT=7% waypoint along the way (same path, since 7% < 8% in gp-distance).
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Printf, Serialization

const W = 80_000
const DELTA = 1.0
const STEP_MULT = 0.996   # gp *= STEP_MULT each continuation step
const STEP_MAXIT = 10
const STEP_BUDGET = 300.0

kappa_of(gp, σ) = 1 - gp^(σ / (σ - 1))

ctx_probe = d20_real_setup(W = W, find_smallest = true, δ = DELTA, destination_sample = :exclude_row)
pe_probe = build_pivot_elimination(ctx_probe)
D = ctx_probe.D
σ = ctx_probe.σ
Aod_theta_natural = ctx_probe.θ0_up[ctx_probe.Aod_offset+1:ctx_probe.Aod_offset+ctx_probe.D*ctx_probe.D_dest]
z0 = log.(Aod_theta_natural)
zfree_cur = pivot_reduce(reshape(z0, ctx_probe.D, ctx_probe.D_dest), pe_probe)
gp_cur = ctx_probe.θ0_up[3+D]
kappa_calib = kappa_of(gp_cur, σ)
println("gp0=$gp_cur  sigma=$σ  kappa_calib=$kappa_calib"); flush(stdout)

const TARGETS = [parse(Float64, get(ENV, "GT_TARGET", "0.05"))]
waypoints = Dict{Float64,NamedTuple}()
target_idx = 1

CKPT_DIR = joinpath(D4X_ROOT, "results", "profiled_ab_2026-08-01", "continuation_gt")
mkpath(CKPT_DIR)

step = 0
while target_idx <= length(TARGETS)
    global step, gp_cur, zfree_cur, target_idx
    target = TARGETS[target_idx]
    gp_target = (1 - target)^((σ - 1) / σ)
    if gp_cur <= gp_target
        # reached/passed this target -- record waypoint using CURRENT (already-converged) state, then advance to next target
        println("Reached target GT=$target (gp_cur=$gp_cur <= gp_target=$gp_target)"); flush(stdout)
        waypoints[target] = (gp = gp_cur, zfree = copy(zfree_cur), kappa = kappa_of(gp_cur, σ))
        serialize(joinpath(CKPT_DIR, "waypoint_GT$(round(Int,target*100)).jls"), waypoints[target])
        global target_idx += 1
        continue
    end
    step += 1
    gp_next = max(gp_cur * STEP_MULT, gp_target)
    label = "continuation_step$(step)"
    println("\n--- step $step: gp $gp_cur -> $gp_next (kappa $(kappa_of(gp_cur,σ)) -> $(kappa_of(gp_next,σ))) ---"); flush(stdout)
    t0 = time()
    res = run_profile_checkpointed(label, gp_next, true, zfree_cur;
        maxtime_real = STEP_BUDGET, W_in = W, delta_in = DELTA, draw_seed_in = 20260719,
        ckpt_dir = CKPT_DIR, checkpoint_interval_s = 300.0, maxit_override = STEP_MAXIT)
    println("  step $step done: wall=$(round(time()-t0,digits=1))s n_eval=$(res.n_eval) native_outer_iters=$(res.native_outer_diag.n_iters)")
    b = res.best
    if b === nothing
        error("continuation step $step: no verified feasible point found at gp=$gp_next -- continuation stalled, need smaller STEP_MULT or larger STEP_MAXIT")
    end
    println("  best: Delta=$(b.Delta_dual)"); flush(stdout)
    global gp_cur = gp_next
    global zfree_cur = b.zfree
end

println("\n" * "="^90); println("CONTINUATION COMPLETE"); println("="^90)
for target in TARGETS
    wp = waypoints[target]
    println("GT=$target: gp=$(wp.gp)  kappa=$(wp.kappa)")
end

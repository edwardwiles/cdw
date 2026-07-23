# Part III.2, 2026-07-23: shadow-mode measurement of complete-state cache hit opportunity on a
# real CM trajectory. Opt-in instrumentation only (shadow_stats kwarg, cm_checkpoint.jl) -- does
# NOT change numerical behavior, does NOT wire in the actual CompleteStateCache. Writes only to
# this worktree's own results/ directory.
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
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
using Printf, Random, Dates, LinearAlgebra

const BACKEND = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :reference
const MAXTIME = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 300.0
const DELTA = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.1
const OUTROOT = joinpath(@__DIR__, "..", "..", "results", "cm_cplus_followup", "cache_shadow_measurement")
mkpath(OUTROOT)

const W = 80_000; const L = 50
const DRAW_DESIGN = :pseudorandom; const DRAW_SEED = 20260719
const CM_CONTRASTS = :orthonormal
snaps = nested_grid_sequence([10, 20, 50])
probs = snaps[L]

ctx0 = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true, draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED)
pe0 = build_pivot_elimination(ctx0)
D = ctx0.D
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
w0 = vcat(x_free_calib[1], pivot_reduce(log.(reshape(x_free_calib[2:end], D, D)), pe0))

shadow_stats = Dict{Symbol,Any}(
    :f_keys => Dict{UInt64,Int}(),
    :g_keys => Dict{UInt64,Int}(),
    :g_same_as_last_F => Ref(0),
    :g_could_have_hit_cache => Ref(0),
)

println(">>> shadow-measurement run: backend=", BACKEND, " delta=", DELTA, " maxtime=", MAXTIME)
flush(stdout)
ckpt_dir = joinpath(OUTROOT, "$(BACKEND)_delta$(DELTA)")
t0 = time()
res = run_cm_upper_checkpointed(w0; W, delta = DELTA, draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED,
    L, contrasts = CM_CONTRASTS, probs, maxtime_real = MAXTIME, ckpt_dir = ckpt_dir,
    run_id = "shadow_$(BACKEND)", label = "shadow", checkpoint_interval_s = 60.0,
    cm_gradient_backend = BACKEND, verbose = true, shadow_stats = shadow_stats)
wall = time() - t0

n_F = res.n_eval; n_G = res.n_grad
f_keys = shadow_stats[:f_keys]
g_keys = shadow_stats[:g_keys]
n_F_repeat_calls = sum(v for v in values(f_keys) if v > 1; init = 0) - count(v -> v > 1, values(f_keys))
n_F_unique = length(f_keys)
n_G_same_as_last_F = shadow_stats[:g_same_as_last_F][]
n_G_could_have_hit_cache = shadow_stats[:g_could_have_hit_cache][]
n_G_genuine_miss = n_G - n_G_same_as_last_F - n_G_could_have_hit_cache

println()
println("="^100)
println("SHADOW MEASUREMENT SUMMARY (backend=$BACKEND, delta=$DELTA, maxtime=$MAXTIME, wall=$(round(wall,digits=1))s)")
println("="^100)
@printf "  cb_F! calls: %d total, %d unique points, %d REPEAT F-calls (exact point solved >1x in this process)\n" n_F n_F_unique n_F_repeat_calls
@printf "  cb_G! calls: %d total\n" n_G
@printf "    -- same-as-last-F (ALREADY handled by last_F_state, free): %d (%.1f%%)\n" n_G_same_as_last_F (n_G>0 ? 100*n_G_same_as_last_F/n_G : 0.0)
@printf "    -- COULD have hit a complete-state cache (point solved earlier by some cb_F!, not the immediately-preceding one): %d (%.1f%%)\n" n_G_could_have_hit_cache (n_G>0 ? 100*n_G_could_have_hit_cache/n_G : 0.0)
@printf "    -- genuine fresh solve needed (point never seen before in this process): %d (%.1f%%)\n" n_G_genuine_miss (n_G>0 ? 100*n_G_genuine_miss/n_G : 0.0)
println("  state size for one BaseDualState/verify entry: ~", D^2 * 8 * 3, " bytes (rough, D^2-scale vectors only, order-of-magnitude)")

open(joinpath(OUTROOT, "shadow_summary_$(BACKEND)_delta$(DELTA).csv"), "w") do io
    println(io, "backend,delta,maxtime,wall,n_F,n_F_unique,n_F_repeat_calls,n_G,n_G_same_as_last_F,n_G_could_have_hit_cache,n_G_genuine_miss")
    println(io, "$BACKEND,$DELTA,$MAXTIME,$wall,$n_F,$n_F_unique,$n_F_repeat_calls,$n_G,$n_G_same_as_last_F,$n_G_could_have_hit_cache,$n_G_genuine_miss")
end
println()
println("DONE")

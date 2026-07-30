# Gate 1 (2026-07-29 reduced-q validation session): replay the exact D4 reduced-q winning
# states (delta in {0.1,0.5}, direction in {:upper,:lower}, 4 cells total) at larger W.
#
# Governing prompt: "Hold the economic state fixed. Do not reoptimize or repair it." -- every
# cell's `theta_full` comes from `melitz_reducedq_phase12_incumbent_checkpoints_2026-07-29.jls`
# (this session's own CORRECTED Phase 12 rerun, Phase 0 -- the ORIGINAL Phase 12 run's own
# incumbent states were never persisted to disk, so there is nothing else to "recover"; using
# this session's own freshly-validated incumbents is the only available source, disclosed
# explicitly, not silently substituted). The same `theta_full` vector is re-evaluated, COLD
# (no warm start -- consistent with this repo's own `cold_verified` convention), against FOUR
# independently-built `MelitzSyntheticData` bundles that share the identical base economy
# (`primitives`/`equilibrium`/`counterfactual`/`L`, all seeded from base_seed=29) and differ
# ONLY in `z_draws` (`pareto_draws(W, D, theta_star; seed=base_seed, mode=:halton)`), exactly
# the nested-QMC-prefix construction `docs/melitz_aq_q_bandwidth_convergence_2026-07-29.md`
# Phase 4 verified (`rhalton`'s digit-permutation draws are independent of `n`, so W_small is a
# bit-for-bit PREFIX of W_large under a fixed seed) -- re-verified live below, not assumed.

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Printf, Serialization
println("Julia threads: ", Threads.nthreads()); flush(stdout)

const OUTDIR = joinpath(REPO, "docs", "key_results")
const CAP = 10.0
const POLICY = CappedEvaluation(CAP)
const BASE_SEED = 29
const W_GRID = [20_000, 80_000, 320_000, 1_280_000]

ckpt_path = joinpath(OUTDIR, "melitz_reducedq_phase12_incumbent_checkpoints_2026-07-29.jls")
checkpoints = open(deserialize, ckpt_path)
println("Loaded ", length(checkpoints), " Gate 1 incumbent checkpoints from ", ckpt_path)

# --- Nested-QMC-prefix re-verification (governing prompt: "test this property, don't assume it") ---
let D = 4, theta_star = 6.8
    z_small = pareto_draws(20_000, D, theta_star; seed=BASE_SEED, mode=:halton)
    z_large = pareto_draws(1_280_000, D, theta_star; seed=BASE_SEED, mode=:halton)
    maxdiff = maximum(abs.(z_small .- z_large[1:20_000, :]))
    println("Nested-QMC-prefix check: W=20,000 is exact prefix of W=1,280,000 under seed=$BASE_SEED -- max|diff|=", maxdiff)
    @assert maxdiff == 0.0 "nested-QMC-prefix property failed to hold live -- ABORTING Gate 1 (would invalidate the whole replay design)"
end
flush(stdout)

const D4_ECONOMY_DATA0 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=BASE_SEED, W=20_000)

"""
    build_bundle_at_W(W; scramble_seed=BASE_SEED)

Holds the ECONOMY (primitives/equilibrium/counterfactual/L, all seeded from `BASE_SEED=29`)
bit-identical across every call, per Gate 1's own requirement -- only `z_draws` (the QMC
scramble) varies with `scramble_seed`. NEVER regenerates the economy from a different seed
(that would change the economic point itself, not merely the draw sample) -- the mandatory
grid uses `scramble_seed=BASE_SEED` (the SAME scramble the states were originally found
under); only the optional held-out-scramble leg passes a different `scramble_seed`.
"""
function build_bundle_at_W(W::Int; scramble_seed::Int=BASE_SEED)
    data0 = D4_ECONOMY_DATA0
    z = pareto_draws(W, data0.primitives.D, data0.primitives.theta_star; seed=scramble_seed, mode=:halton)
    data = MelitzSyntheticData(data0.primitives, data0.equilibrium, data0.counterfactual, data0.L, z, data0.seed)
    obj, _ = build_melitz_psi_bundle(data; outer_parameterization=:logcutoff,
        policy=POLICY, backend=:matrix_free, forbid_dense_fallback=true)
    return obj, obj.γ
end

replay_rows = NamedTuple[]

function replay_one(cp, W::Int; scramble_label::String="tuning_seed29", scramble_seed::Int=BASE_SEED)
    obj, ctx = build_bundle_at_W(W; scramble_seed=scramble_seed)
    theta_full = collect(Float64.(cp.theta_full))
    session = MelitzInnerSession(obj, ctx, POLICY)
    t0 = time()
    res = solve_melitz_delta!(session, theta_full, POLICY)
    wall = time() - t0
    classification = nameof(typeof(res))

    obj.use_cached_x = false; obj.x .= NaN
    lfd = melitz_recover_lfd(obj, theta_full)

    Delta = res isa FiniteSolved ? res.Delta : NaN
    within_original_budget = !isnan(Delta) && Delta <= cp.delta
    gt_pct = isnan(Delta) ? NaN : 100 * (1 - kappa_ratio_of_g(cp.objective * (cp.direction==:upper ? 1.0 : -1.0), ctx))

    row = (label=cp.label, delta=cp.delta, direction=string(cp.direction), scramble=scramble_label,
           scramble_seed=scramble_seed, W=W, classification=string(classification),
           Delta=Delta, Delta_W20000_original=cp.Delta_W20000, within_original_budget=within_original_budget,
           gt_pct=gt_pct, gt_pct_W20000_original=cp.gt_pct_W20000,
           primal_feasible=lfd.lfd_ok && lfd.min_weight >= -1e-10,
           dual_feasible=isfinite(lfd.dual_divergence),   # dual is unconstrained -> always feasible if finite
           moment_residual=lfd.maximum_weighted_moment_residual,
           normalization_error=lfd.probability_normalization_residual,
           primal_dual_agreement=lfd.primal_dual_gap,
           kkt_opt_error=lfd.kkt_opt_error, kkt_feas_error=lfd.kkt_feas_error,
           lfd_ok=lfd.lfd_ok, wall_s=wall, warm_start_used=false,
           nStatus=(res isa FiniteSolved ? res.nStatus : -1))
    @printf("  [%s delta=%.1f %-5s] W=%9d scramble=%-14s -> %-24s Delta=%s within_budget=%s lfd_ok=%s moment_resid=%.2e wall=%.2fs\n",
            cp.label, cp.delta, string(cp.direction), W, scramble_label, string(classification),
            isnan(Delta) ? "NaN" : @sprintf("%.6e", Delta), within_original_budget, lfd.lfd_ok,
            lfd.maximum_weighted_moment_residual, wall)
    flush(stdout)
    push!(replay_rows, row)
    return row
end

println("\n" * "="^100)
println("Gate 1: mandatory W-grid replay, all 4 cells, held-out scramble = tuning scramble (seed=29, same as construction)")
println("="^100)
for (key, cp) in sort(collect(checkpoints); by=x -> (x[1][1], string(x[1][2])))
    for W in W_GRID
        replay_one(cp, W)
    end
end

println("\n" * "="^100)
println("Gate 1 optional: held-out scramble for the two delta=0.5 winners only, W=320,000, scramble_seed=49")
println("="^100)
# seed=49 (not 141): this D4 fixture is known-seed-fragile (memory
# feedback-melitz-d4-seed-fragility -- most seeds fail generate_fake_melitz_data's own
# export-selection consistency check; 141 fails it live, confirmed this run). Only
# [29,49,50,52,53,94,107,110] are known-good; 29 is the tuning seed already used above, so 49
# is the first distinct known-good held-out choice.
for (key, cp) in sort(collect(checkpoints); by=x -> (x[1][1], string(x[1][2])))
    if cp.delta == 0.5
        replay_one(cp, 320_000; scramble_label="held_out_seed49", scramble_seed=49)
    end
end

header = ["label", "delta", "direction", "scramble", "scramble_seed", "W", "classification", "Delta",
          "Delta_W20000_original", "within_original_budget", "gt_pct", "gt_pct_W20000_original",
          "primal_feasible", "dual_feasible", "moment_residual", "normalization_error",
          "primal_dual_agreement", "kkt_opt_error", "kkt_feas_error", "lfd_ok", "wall_s",
          "warm_start_used", "nStatus"]
csv_path = joinpath(OUTDIR, "melitz_gate1_d4_w_replay_2026-07-29.csv")
melitz_write_typed_counter_csv(csv_path, header, replay_rows)
println("\nWrote Gate 1 replay CSV (", length(replay_rows), " rows): ", csv_path)
flush(stdout)

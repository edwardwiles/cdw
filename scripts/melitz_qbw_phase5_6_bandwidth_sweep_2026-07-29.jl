# q-bandwidth convergence campaign (2026-07-29), Phases 5-6: predeclared bandwidth family
# sweep (cheap fixed-dual mode, full grid) + coordinatewise D4 screen (expensive reoptimized
# ground truth, a disclosed representative policy subset -- see header note below).
#
# DISCLOSED SCOPE REDUCTION (compute-time budget, stated up front, not discovered after the
# fact): the full predeclared design calls for reoptimized secants (2 KNITRO solves each) at
# EVERY (coordinate x policy x W x scramble x base-point) cell -- for D4's 14 free q
# coordinates x 16 policy configs x 4 W levels x 3 scrambles x 2 base points that is >10,000
# KNITRO solves, far outside a single session's realistic wall-clock budget. Reduction:
#   - FIXED-DUAL mode (cheap, O(D^2 log W), no KNITRO re-solve) is run at the FULL grid:
#     every one of the 14 free D4 q coordinates, every policy family, W in
#     {20000,80000,320000,1280000}, all 3 scrambles, both reachable D4 base points
#     (delta~0.1, delta~0.5). This is Phase 5's own primary comparison.
#   - REOPTIMIZED mode (expensive, ground truth) is run on a representative SUBSET: the
#     mandatory alpha=1/2 power-scaled policy and one representative fixed-crossing policy
#     (target=25), for EVERY coordinate (not a further coordinate subset -- Rule 11: do not
#     silently reduce the mandatory D4 scope), at W in {20000,80000,320000} (1,280,000
#     dropped from the coordinatewise reoptimized sweep specifically, kept for fixed-dual and
#     for the dense-direction Phase 7 script instead), tuning scramble only for all 14
#     coordinates, PLUS the 2 held-out scrambles for 5 representative coordinates (highest/
#     lowest q-pivot leverage, plus 3 evenly spaced others) to check scramble-stability
#     without recomputing all 14 three times.
#
# Base economic points: from Phase 3's saved theta_q vectors
# (docs/key_results/melitz_qbw_phase3_theta_q_2026-07-29.csv) -- D4 delta~0.1 and delta~0.5
# (delta~1/2 confirmed NOT reachable in the D4 fixed-A/f corridor, Phase 3).
#
# 3 QMC scrambles, SAME economy (Phase 3's seed=29 D4 primitives/equilibrium/counterfactual/L
# held fixed) -- only z_draws differ: seed=29 (tuning), seed=141/271 (held-out).

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Random

const OUTDIR = joinpath(REPO, "docs", "key_results")
isdir(OUTDIR) || mkpath(OUTDIR)
CAP = 10.0
policy_cap = CappedEvaluation(CAP)

println("Julia threads: ", Threads.nthreads())
flush(stdout)

# --- load Phase 3 base states (D4 only) ---
function load_theta_q_rows(path)
    rows = Dict{Tuple{String,Float64},Vector{Float64}}()
    for line in eachline(path)
        parts = split(line, ",")
        label = parts[1]
        target = parse(Float64, parts[2])
        # parts[3]=W, parts[4]=seed, rest = theta_q entries
        theta = parse.(Float64, parts[5:end])
        rows[(label, target)] = theta
    end
    return rows
end
theta_q_rows = load_theta_q_rows(joinpath(OUTDIR, "melitz_qbw_phase3_theta_q_2026-07-29.csv"))

const D4_BASE_TARGETS = [0.1, 0.5]
const SCRAMBLES = [(29, "tuning"), (141, "held_out_1"), (271, "held_out_2")]
const W_GRID = [20_000, 80_000, 320_000, 1_280_000]
const W_REF = 80_000

"Build the SAME D4 economy (primitives/eq/counterfactual/L from seed=29) at a given
scramble's z_draws and W -- never regenerates a different economy."
function build_scrambled_bundle(base_theta_q::Vector{Float64}, W::Int, scramble_seed::Int)
    data0 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=W_REF)
    z = pareto_draws(W, data0.primitives.D, data0.primitives.theta_star; seed=scramble_seed, mode=:halton)
    data = MelitzSyntheticData(data0.primitives, data0.equilibrium, data0.counterfactual, data0.L, z, scramble_seed)
    obj, theta0_native = build_melitz_psi_bundle(data; outer_parameterization=:logcutoff,
        policy=policy_cap, backend=:matrix_free, forbid_dense_fallback=true)
    ctx = obj.γ
    # theta0_native's g/A-block are re-derived fresh at this W/scramble's own pareto point;
    # OVERWRITE with the Phase-3 base point's (g, A_free, q_free) -- theta_q's economic
    # content (not the pareto default) is what we want to hold fixed across W/scramble.
    theta = copy(base_theta_q)
    return obj, ctx, theta
end

qpiv_leverage = nothing   # filled after first bundle build
h_ref_cache_global = Dict{Tuple{Float64,Int,String},Float64}()

results_fixed = NamedTuple[]
results_reopt = NamedTuple[]

for target in D4_BASE_TARGETS
    base_theta_q = theta_q_rows[("D4_seed29_W20000", target)]
    D = 4
    nA = D^2 - 1
    nq = length(base_theta_q) - 1 - nA
    println("\n" * "="^100)
    @printf("Base point target=%.1f  nq=%d\n", target, nq)
    println("="^100)
    flush(stdout)

    for (scramble_seed, scramble_label) in SCRAMBLES
        for W in W_GRID
            obj, ctx, theta0 = build_scrambled_bundle(base_theta_q, W, scramble_seed)
            sorted_ctx = ctx.sorted_tail_ctx

            obj.use_cached_x = false; obj.x .= NaN
            lfd0 = melitz_recover_lfd(obj, theta0)
            if !lfd0.lfd_ok
                @printf("  [SKIP target=%.1f scramble=%s W=%d] base point failed to verify (nStatus=%d)\n",
                        target, scramble_label, W, lfd0.nStatus)
                flush(stdout)
                continue
            end
            x0 = copy(lfd0.dual_x)
            @printf("  target=%.1f scramble=%s(seed=%d) W=%d: Delta0=%.6e nStatus=%d\n",
                    target, scramble_label, scramble_seed, W, lfd0.Delta, lfd0.nStatus)
            flush(stdout)

            if qpiv_leverage === nothing
                qpiv = build_q_gravity_pivot(ctx)
                global qpiv_leverage = abs.(qpiv.c[qpiv.other] ./ qpiv.c[qpiv.pivot])
            end

            # --- Phase 5: fixed-dual bandwidth family sweep, every coordinate ---
            for m in 1:nq
                # Family A: fixed raw h
                for h in (1e-5, 1e-4, 1e-3)
                    pol = FixedRawQBandwidth(h)
                    r = melitz_q_coordinate_probe(theta0, m, pol, obj, ctx; x0=x0, mode=:fixed_dual)
                    push!(results_fixed, (target=target, scramble=scramble_label, seed=scramble_seed, W=W, m=m,
                                           family="A_fixed_raw", param=h, h=r.h, cplus=r.crossings_plus_total,
                                           cminus=r.crossings_minus_total, cmin=r.crossings_min_side,
                                           secant=r.secant, leverage=qpiv_leverage[m]))
                end
                # Family C: fixed two-sided crossing targets
                for T in (10, 25, 50, 100)
                    pol = FixedCrossingQBandwidth(T)
                    r = melitz_q_coordinate_probe(theta0, m, pol, obj, ctx; x0=x0, mode=:fixed_dual)
                    push!(results_fixed, (target=target, scramble=scramble_label, seed=scramble_seed, W=W, m=m,
                                           family="C_fixed_crossing", param=T, h=r.h, cplus=r.crossings_plus_total,
                                           cminus=r.crossings_minus_total, cmin=r.crossings_min_side,
                                           secant=r.secant, leverage=qpiv_leverage[m]))
                end
                # Family D: growing crossing targets (T_ref at W_REF)
                for Tref in (10, 25, 50)
                    pol = GrowingCrossingQBandwidth(Tref, W_REF)
                    r = melitz_q_coordinate_probe(theta0, m, pol, obj, ctx; x0=x0, mode=:fixed_dual)
                    push!(results_fixed, (target=target, scramble=scramble_label, seed=scramble_seed, W=W, m=m,
                                           family="D_growing_crossing", param=Tref, h=r.h, cplus=r.crossings_plus_total,
                                           cminus=r.crossings_minus_total, cmin=r.crossings_min_side,
                                           secant=r.secant, leverage=qpiv_leverage[m]))
                end
            end

            # Family B: power-scaled h_W = h_ref*(W_REF/W)^alpha -- h_ref anchors calibrated
            # ONCE at W==W_REF via the fixed-crossing bisection (targets 25/100, min-side),
            # PER COORDINATE, tuning scramble only (then reused at every W/scramble for that
            # coordinate).
            if W == W_REF && scramble_seed == 29
                for m in 1:nq
                    for (anchor_label, Tanchor) in (("anchor25", 25), ("anchor100", 100))
                        h_ref, _, _ = _melitz_bisect_h_two_sided(Tanchor, theta0, m, ctx, sorted_ctx)
                        h_ref_cache_global[(target, m, anchor_label)] = h_ref
                    end
                end
            end
            for m in 1:nq
                for (anchor_label, _) in (("anchor25", 25), ("anchor100", 100))
                    h_ref = get(h_ref_cache_global, (target, m, anchor_label), nothing)
                    h_ref === nothing && continue
                    for alpha in (1/3, 1/2, 2/3)
                        pol = PowerScaledQBandwidth(h_ref, W_REF, alpha)
                        r = melitz_q_coordinate_probe(theta0, m, pol, obj, ctx; x0=x0, mode=:fixed_dual)
                        push!(results_fixed, (target=target, scramble=scramble_label, seed=scramble_seed, W=W, m=m,
                                               family="B_power_scaled_$(anchor_label)", param=alpha, h=r.h,
                                               cplus=r.crossings_plus_total, cminus=r.crossings_minus_total,
                                               cmin=r.crossings_min_side, secant=r.secant, leverage=qpiv_leverage[m]))
                    end
                end
            end

            # --- Phase 6: reoptimized ground truth, disclosed representative subset ---
            if W in (20_000, 80_000, 320_000)
                coords_this_pass = if scramble_seed == 29
                    1:nq   # full coordinate sweep, tuning scramble
                else
                    sorted_leverage_idx = sortperm(qpiv_leverage)
                    unique(vcat(sorted_leverage_idx[1], sorted_leverage_idx[end],
                                 sorted_leverage_idx[[1 + nq ÷ 4, 1 + nq ÷ 2, 1 + 3 * nq ÷ 4]]))
                end
                for m in coords_this_pass
                    for (polname, pol) in (("B_alpha_half_anchor25",
                                             PowerScaledQBandwidth(get(h_ref_cache_global, (target, m, "anchor25"), 1e-4), W_REF, 0.5)),
                                            ("C_target25", FixedCrossingQBandwidth(25)))
                        r = melitz_q_coordinate_probe(theta0, m, pol, obj, ctx; x0=x0, mode=:reoptimized)
                        push!(results_reopt, (target=target, scramble=scramble_label, seed=scramble_seed, W=W, m=m,
                                               policy=polname, h=r.h, cplus=r.crossings_plus_total,
                                               cminus=r.crossings_minus_total, cmin=r.crossings_min_side,
                                               secant_fixed_dual=NaN, secant_reopt=r.secant,
                                               boundary_p=r.boundary_hit_plus, boundary_m=r.boundary_hit_minus,
                                               leverage=qpiv_leverage[m]))
                        # matching fixed-dual secant at the SAME h (for the direct comparison Phase 6 wants)
                        rp = melitz_q_coordinate_probe(theta0, m, FixedRawQBandwidth(r.h), obj, ctx; x0=x0, mode=:fixed_dual)
                        results_reopt[end] = merge(results_reopt[end], (secant_fixed_dual=rp.secant,))
                    end
                end
                @printf("    reoptimized ground truth done for %d coordinates at W=%d\n", length(coords_this_pass), W)
                flush(stdout)
            end
        end
    end
end

open(joinpath(OUTDIR, "melitz_qbw_phase5_fixed_dual_sweep_2026-07-29.csv"), "w") do io
    println(io, "target,scramble,seed,W,m,family,param,h,cplus,cminus,cmin,secant,leverage")
    for r in results_fixed
        println(io, join([r.target, r.scramble, r.seed, r.W, r.m, r.family, r.param, r.h,
                           r.cplus, r.cminus, r.cmin, r.secant, r.leverage], ","))
    end
end
open(joinpath(OUTDIR, "melitz_qbw_phase6_coordinatewise_reopt_2026-07-29.csv"), "w") do io
    println(io, "target,scramble,seed,W,m,policy,h,cplus,cminus,cmin,secant_fixed_dual,secant_reopt,boundary_p,boundary_m,leverage")
    for r in results_reopt
        println(io, join([r.target, r.scramble, r.seed, r.W, r.m, r.policy, r.h, r.cplus, r.cminus, r.cmin,
                           r.secant_fixed_dual, r.secant_reopt, r.boundary_p, r.boundary_m, r.leverage], ","))
    end
end
println("\nPhase 5-6 sweep complete. Rows: fixed_dual=", length(results_fixed), "  reopt=", length(results_reopt))
flush(stdout)

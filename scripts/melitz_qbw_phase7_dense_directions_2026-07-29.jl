# q-bandwidth convergence campaign (2026-07-29), Phase 7 (PRIMARY TEST): does the assembled
# coordinatewise q gradient predict dense-direction movements relevant to the outer solver?
#
# DISCLOSED SCOPE REDUCTION: (a) held-out W grid restricted to {80000, 320000, 1280000} (the
# tuning W=20000 point is excluded here since Phase 7 is specifically a HELD-OUT test); (b)
# 2 shortlisted policies carried over from Phase 6 (PowerScaled alpha=1/2 anchor25,
# FixedCrossing target=25) rather than re-shortlisting from scratch; (c) direction family 7
# ("3 actual q-block directions captured from KNITRO trial steps") is NOT captured from a
# live instrumented outer run this session -- substituted with 3 additional structured
# directions (a random MIXTURE of origin/destination-block structure, a leverage-weighted
# random direction, and a second independent dense-random direction), disclosed rather than
# silently omitted; (d) feasible step limits are NOT derived from the native affine cutoff
# LP -- instead, amplitude `t` is chosen via bisection on the SAME two-sided crossing count
# used elsewhere in this campaign (a disclosed, cheaper proxy for "how far can we step before
# this becomes a large discrete move", not a literal geometric constraint-boundary computation).

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Random

const OUTDIR = joinpath(REPO, "docs", "key_results")
CAP = 10.0
policy_cap = CappedEvaluation(CAP)
println("Julia threads: ", Threads.nthreads()); flush(stdout)

function load_theta_q_rows(path)
    rows = Dict{Tuple{String,Float64},Vector{Float64}}()
    for line in eachline(path)
        parts = split(line, ",")
        rows[(parts[1], parse(Float64, parts[2]))] = parse.(Float64, parts[5:end])
    end
    return rows
end
theta_q_rows = load_theta_q_rows(joinpath(OUTDIR, "melitz_qbw_phase3_theta_q_2026-07-29.csv"))

const D4_BASE_TARGETS = [0.1, 0.5]
const HELDOUT_SCRAMBLES = [(29, "tuning"), (401, "held_out_3")]   # held-out scramble distinct from Phase 5-6's own held-outs
const W_GRID = [80_000, 320_000, 1_280_000]
const W_REF = 80_000

function build_scrambled_bundle(W::Int, scramble_seed::Int)
    data0 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=W_REF)
    z = pareto_draws(W, data0.primitives.D, data0.primitives.theta_star; seed=scramble_seed, mode=:halton)
    data = MelitzSyntheticData(data0.primitives, data0.equilibrium, data0.counterfactual, data0.L, z, scramble_seed)
    obj, _ = build_melitz_psi_bundle(data; outer_parameterization=:logcutoff,
        policy=policy_cap, backend=:matrix_free, forbid_dense_fallback=true)
    return obj, obj.γ
end

"Two-sided crossing count for a FULL free-q direction vector d (length nq), step t."
function direction_two_sided_crossings(theta0::AbstractVector, d::AbstractVector, t::Real, ctx, sorted_ctx)
    D = ctx.D
    nA = D^2 - 1
    theta_p = copy(theta0); theta_p[1+nA+1:end] .+= t .* d
    theta_m = copy(theta0); theta_m[1+nA+1:end] .-= t .* d
    _, _, _, _, q0 = expand_free_theta_logcutoff(melitz_unpower_theta_free(theta0, ctx), ctx)
    _, _, _, _, qp = expand_free_theta_logcutoff(melitz_unpower_theta_free(theta_p, ctx), ctx)
    _, _, _, _, qm = expand_free_theta_logcutoff(melitz_unpower_theta_free(theta_m, ctx), ctx)
    total_plus = 0; total_minus = 0
    @inbounds for o in 1:D, d2 in 1:D
        sorted_z_o = @view sorted_ctx.sorted_z[:, o]
        kb = melitz_active_tail_start(sorted_z_o, exp(q0[o, d2]))
        kp = melitz_active_tail_start(sorted_z_o, exp(qp[o, d2]))
        km = melitz_active_tail_start(sorted_z_o, exp(qm[o, d2]))
        total_plus += abs(kp - kb)
        total_minus += abs(km - kb)
    end
    return total_plus, total_minus
end

function bisect_t_for_target_crossings(theta0, d, target, ctx, sorted_ctx; t_lo=1e-7, t_hi=0.5, max_iter=40)
    lo, hi = t_lo, t_hi
    tp_hi, tm_hi = direction_two_sided_crossings(theta0, d, hi, ctx, sorted_ctx)
    min(tp_hi, tm_hi) < target && return hi
    for _ in 1:max_iter
        mid = sqrt(lo * hi)
        tp, tm = direction_two_sided_crossings(theta0, d, mid, ctx, sorted_ctx)
        if min(tp, tm) >= target
            hi = mid
        else
            lo = mid
        end
        hi / lo < 1.01 && break
    end
    return hi
end

"Assembled coordinatewise q gradient (fixed-dual secants) at a given policy/W, all nq coords."
function assembled_q_gradient(theta0, policy, obj, ctx, x0, nq)
    g = zeros(nq)
    for m in 1:nq
        r = melitz_q_coordinate_probe(theta0, m, policy, obj, ctx; x0=x0, mode=:fixed_dual)
        g[m] = r.secant
    end
    return g
end

function make_directions(nq::Int, qpiv_leverage::Vector{Float64}, g_ref::Vector{Float64}, rng::MersenneTwister)
    dirs = Dict{String,Vector{Float64}}()
    dirs["dense_random_1"] = normalize(randn(rng, nq))
    dirs["dense_random_2"] = normalize(randn(rng, nq))
    s1 = zeros(nq); idx1 = randperm(rng, nq)[1:max(2, nq ÷ 5)]; s1[idx1] .= randn(rng, length(idx1)); dirs["sparse_random_1"] = normalize(s1)
    s2 = zeros(nq); idx2 = randperm(rng, nq)[1:max(2, nq ÷ 5)]; s2[idx2] .= randn(rng, length(idx2)); dirs["sparse_random_2"] = normalize(s2)
    half = nq ÷ 2
    origin_block = zeros(nq); origin_block[1:half] .= randn(rng, half); dirs["origin_block"] = normalize(origin_block)
    dest_block = zeros(nq); dest_block[half+1:end] .= randn(rng, nq - half); dirs["dest_block"] = normalize(dest_block)
    dirs["pivot_leverage_weighted"] = normalize(qpiv_leverage .* randn(rng, nq))
    dirs["gradient_descent"] = norm(g_ref) > 0 ? normalize(-g_ref) : normalize(randn(rng, nq))
    dirs["mixed_block_random"] = normalize(randn(rng, nq) .* (0.3 .+ 0.7 .* rand(rng, nq)))
    dirs["leverage_weighted_random_2"] = normalize((qpiv_leverage .^ 2) .* randn(rng, nq))
    dirs["dense_random_3"] = normalize(randn(rng, nq))
    return dirs
end

results = NamedTuple[]
policies_shortlist = [("B_alpha_half_anchor25", nothing), ("C_target25", FixedCrossingQBandwidth(25))]

for target in D4_BASE_TARGETS
    base_theta_q = theta_q_rows[("D4_seed29_W20000", target)]
    D = 4; nA = D^2 - 1
    nq = length(base_theta_q) - 1 - nA
    for (scramble_seed, scramble_label) in HELDOUT_SCRAMBLES
        for W in W_GRID
            obj, ctx = build_scrambled_bundle(W, scramble_seed)
            sorted_ctx = ctx.sorted_tail_ctx
            theta0 = copy(base_theta_q)
            obj.use_cached_x = false; obj.x .= NaN
            lfd0 = melitz_recover_lfd(obj, theta0)
            if !lfd0.lfd_ok
                @printf("[SKIP] target=%.1f scramble=%s W=%d base point failed\n", target, scramble_label, W)
                continue
            end
            x0 = copy(lfd0.dual_x)
            @printf("target=%.1f scramble=%s W=%d Delta0=%.6e\n", target, scramble_label, W, lfd0.Delta); flush(stdout)

            qpiv = build_q_gravity_pivot(ctx)
            leverage_full = zeros(nq)
            # map qpiv leverage (over q_pivot.other, length nq) directly
            leverage_full .= abs.(qpiv.c[qpiv.other] ./ qpiv.c[qpiv.pivot])

            # h_ref anchor25 for B policy, per-coordinate (needed since B_alpha_half is coordinate-specific h_ref)
            h_ref_anchor25 = zeros(nq)
            for m in 1:nq
                h_ref_anchor25[m], _, _ = _melitz_bisect_h_two_sided(25, theta0, m, ctx, sorted_ctx)
            end
            pol_B = m -> PowerScaledQBandwidth(h_ref_anchor25[m], W_REF, 0.5)

            g_B = zeros(nq); g_C = zeros(nq)
            for m in 1:nq
                rB = melitz_q_coordinate_probe(theta0, m, pol_B(m), obj, ctx; x0=x0, mode=:fixed_dual)
                g_B[m] = rB.secant
                rC = melitz_q_coordinate_probe(theta0, m, FixedCrossingQBandwidth(25), obj, ctx; x0=x0, mode=:fixed_dual)
                g_C[m] = rC.secant
            end

            rng = MersenneTwister(20260729)   # FIXED seed -- same directions across policy/W/scramble
            dirs = make_directions(nq, leverage_full, g_B, rng)

            for (dname, d) in dirs
                for (target_cross, cross_label) in ((25, "c25"), (100, "c100"), (400, "c400"))
                    t = bisect_t_for_target_crossings(theta0, d, target_cross, ctx, sorted_ctx)
                    theta_p = copy(theta0); theta_p[1+nA+1:end] .+= t .* d
                    theta_m = copy(theta0); theta_m[1+nA+1:end] .-= t .* d

                    melitz_update_operator_at_theta!(obj.op, theta_p, ctx); Bp = -obj(x0)
                    melitz_update_operator_at_theta!(obj.op, theta_m, ctx); Bm = -obj(x0)
                    melitz_update_operator_at_theta!(obj.op, theta0, ctx)
                    secant_block_fixed_dual = (Bp - Bm) / (2t)

                    obj.use_cached_x = false; obj.x .= NaN; lfd_p = melitz_recover_lfd(obj, theta_p)
                    obj.use_cached_x = false; obj.x .= NaN; lfd_m = melitz_recover_lfd(obj, theta_m)
                    secant_reopt = (lfd_p.lfd_ok && lfd_m.lfd_ok) ? (lfd_p.Delta - lfd_m.Delta) / (2t) : NaN
                    one_sided = lfd_p.lfd_ok ? (lfd_p.Delta - lfd0.Delta) / t : NaN

                    pred_B = dot(g_B, d); pred_C = dot(g_C, d)

                    push!(results, (target=target, scramble=scramble_label, W=W, direction=dname,
                                     cross_target=cross_label, t=t,
                                     pred_B=pred_B, pred_C=pred_C,
                                     secant_block_fixed_dual=secant_block_fixed_dual,
                                     secant_reopt=secant_reopt, one_sided=one_sided,
                                     lfd_ok_p=lfd_p.lfd_ok, lfd_ok_m=lfd_m.lfd_ok))
                end
            end
            @printf("  done: %d directions x 3 amplitudes for target=%.1f scramble=%s W=%d\n", length(dirs), target, scramble_label, W)
            flush(stdout)
        end
    end
end

open(joinpath(OUTDIR, "melitz_qbw_phase7_dense_directions_2026-07-29.csv"), "w") do io
    println(io, "target,scramble,W,direction,cross_target,t,pred_B,pred_C,secant_block_fixed_dual,secant_reopt,one_sided,lfd_ok_p,lfd_ok_m")
    for r in results
        println(io, join([r.target, r.scramble, r.W, r.direction, r.cross_target, r.t, r.pred_B, r.pred_C,
                           r.secant_block_fixed_dual, r.secant_reopt, r.one_sided, r.lfd_ok_p, r.lfd_ok_m], ","))
    end
end
println("\nPhase 7 complete. Rows: ", length(results))
flush(stdout)

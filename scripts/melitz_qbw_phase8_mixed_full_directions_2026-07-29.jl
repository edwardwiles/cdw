# q-bandwidth convergence campaign (2026-07-29), Phase 8: mixed full-outer directions
# (d_gamma, d_A, d_q), combining the exact gamma secant + exact A gradient + the Phase 6
# shortlisted q-bandwidth estimator.
#
# DISCLOSED SCOPE REDUCTION: one representative W (320,000, mid-grid), tuning scramble only,
# 6 direction families (not the full menu) -- pure welfare / pure A / pure q / mixed A+q /
# mixed welfare+A+q / one leverage-weighted mixed direction -- at both reachable D4 base
# points (delta~0.1, delta~0.5). No live-captured KNITRO trial-step directions (same
# limitation as Phase 7).

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

const W = 320_000
results = NamedTuple[]

for target in (0.1, 0.5)
    base_theta_q = theta_q_rows[("D4_seed29_W20000", target)]
    D = 4; nA = D^2 - 1
    nq = length(base_theta_q) - 1 - nA
    n = length(base_theta_q)

    data = generate_fake_melitz_data(; D=D, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=W)
    obj, _ = build_melitz_psi_bundle(data; outer_parameterization=:logcutoff,
        policy=policy_cap, backend=:matrix_free, forbid_dense_fallback=true)
    ctx = obj.γ
    sorted_ctx = ctx.sorted_tail_ctx
    theta0 = copy(base_theta_q)

    obj.use_cached_x = false; obj.x .= NaN
    lfd0 = melitz_recover_lfd(obj, theta0)
    if !lfd0.lfd_ok
        @printf("[SKIP] target=%.1f base point failed to verify\n", target); continue
    end
    x0 = copy(lfd0.dual_x)
    @printf("target=%.1f W=%d Delta0=%.6e kappa0=%.4f%%\n", target, W, lfd0.Delta, 100*kappa_ratio_of_g(theta0[1], ctx))
    flush(stdout)

    # exact A gradient (fixed q)
    A0, f0, gpj0, fjj0 = melitz_expand_theta(theta0, ctx)
    state0 = MelitzExpandedState(D)
    state0.A .= A0; state0.f .= f0; state0.gamma_prime_j = gpj0; state0.f_jj = fjj0
    melitz_update_operator_at_theta!(obj.op, theta0, ctx)
    exact_A_free, _ = melitz_exact_a_gradient(obj, x0, state0, ctx)

    # exact gamma secant (fixed-dual, cheap)
    gamma_h = 1e-6
    theta_p = copy(theta0); theta_p[1] += gamma_h
    theta_m = copy(theta0); theta_m[1] -= gamma_h
    melitz_update_operator_at_theta!(obj.op, theta_p, ctx); Dp = -obj(x0)
    melitz_update_operator_at_theta!(obj.op, theta_m, ctx); Dm = -obj(x0)
    melitz_update_operator_at_theta!(obj.op, theta0, ctx)
    gamma_secant = (Dp - Dm) / (2 * gamma_h)

    # q-block secants via the Phase 6 shortlisted policy (PowerScaled alpha=1/2, h_ref
    # calibrated per coordinate at anchor25)
    g_q = zeros(nq)
    for m in 1:nq
        h_ref, _, _ = _melitz_bisect_h_two_sided(25, theta0, m, ctx, sorted_ctx)
        pol = PowerScaledQBandwidth(h_ref, W, 0.5)
        r = melitz_q_coordinate_probe(theta0, m, pol, obj, ctx; x0=x0, mode=:fixed_dual)
        g_q[m] = r.secant
    end

    g_full = vcat(gamma_secant, exact_A_free, g_q)   # full assembled gradient, length n

    rng = MersenneTwister(20260729)
    directions = Dict{String,Vector{Float64}}()
    directions["pure_welfare"] = vcat(1.0, zeros(nA), zeros(nq))
    da = zeros(nA); da[1] = 1.0; directions["pure_A"] = vcat(0.0, da, zeros(nq))
    dq = zeros(nq); dq[1] = 1.0; directions["pure_q"] = vcat(0.0, zeros(nA), dq)
    da2 = normalize(randn(rng, nA)); dq2 = normalize(randn(rng, nq))
    directions["mixed_Aq"] = normalize(vcat(0.0, da2, dq2))
    dg3 = 1.0; da3 = normalize(randn(rng, nA)); dq3 = normalize(randn(rng, nq))
    directions["mixed_welfare_A_q"] = normalize(vcat(dg3, da3, dq3))
    directions["gradient_descent_full"] = norm(g_full) > 0 ? normalize(-g_full) : normalize(randn(rng, n))

    for (dname, draw) in directions
        d = draw ./ norm(draw)   # ensure unit-norm direction
        for scale in (1e-4, 1e-3)
            theta_p = theta0 .+ scale .* d
            theta_m = theta0 .- scale .* d

            pred = dot(g_full, d) * scale

            melitz_update_operator_at_theta!(obj.op, theta_p, ctx); Bp = -obj(x0)
            melitz_update_operator_at_theta!(obj.op, theta_m, ctx); Bm = -obj(x0)
            melitz_update_operator_at_theta!(obj.op, theta0, ctx)
            secant_fd = (Bp - Bm) / 2

            obj.use_cached_x = false; obj.x .= NaN; lp = melitz_recover_lfd(obj, theta_p)
            obj.use_cached_x = false; obj.x .= NaN; lm = melitz_recover_lfd(obj, theta_m)
            secant_reopt = (lp.lfd_ok && lm.lfd_ok) ? (lp.Delta - lm.Delta) / 2 : NaN
            gt_p = lp.lfd_ok ? 100 * kappa_ratio_of_g(theta_p[1], ctx) : NaN
            gt_m = lm.lfd_ok ? 100 * kappa_ratio_of_g(theta_m[1], ctx) : NaN
            dGT_pct = (!isnan(gt_p) && !isnan(gt_m)) ? (gt_p - gt_m) : NaN

            @printf("  [%-20s scale=%.0e] pred=%.4e  secant_fd=%.4e  secant_reopt=%.4e  dGT=%.5f pp  classif_p=%s classif_m=%s\n",
                    dname, scale, pred, secant_fd, secant_reopt, dGT_pct,
                    lp.lfd_ok ? "FiniteSolved" : "other", lm.lfd_ok ? "FiniteSolved" : "other")
            flush(stdout)

            push!(results, (target=target, direction=dname, scale=scale, pred=pred, secant_fd=secant_fd,
                             secant_reopt=secant_reopt, dGT_pct=dGT_pct, lfd_ok_p=lp.lfd_ok, lfd_ok_m=lm.lfd_ok))
        end
    end
end

open(joinpath(OUTDIR, "melitz_qbw_phase8_mixed_full_directions_2026-07-29.csv"), "w") do io
    println(io, "target,direction,scale,pred,secant_fd,secant_reopt,dGT_pct,lfd_ok_p,lfd_ok_m")
    for r in results
        println(io, join([r.target, r.direction, r.scale, r.pred, r.secant_fd, r.secant_reopt,
                           r.dGT_pct, r.lfd_ok_p, r.lfd_ok_m], ","))
    end
end
println("\nPhase 8 complete. Rows: ", length(results))
flush(stdout)
